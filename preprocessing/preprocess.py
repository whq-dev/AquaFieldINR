
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, label

FOUR = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
INVALID_SCL = (0, 1, 3, 8, 9, 10, 11)
BAND_GROUPS = {10: ("B2", "B3", "B4", "B8"), 20: ("B5", "B6", "B7", "B8A")}
ORIGIN = datetime(2021, 1, 1, tzinfo=timezone.utc)


def time_days(value):
    t = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if t.tzinfo is None:
        raise ValueError('E_PREPROCESS_022')
    return (t.astimezone(timezone.utc) - ORIGIN).total_seconds() / 86400.0


def scene_quality(scl, cloud_probability):
    
    scl = np.asarray(scl)
    probability = np.asarray(cloud_probability, dtype=float)
    if scl.shape != probability.shape or scl.ndim != 2:
        raise ValueError('E_PREPROCESS_031')
    missing = ~np.isfinite(probability) | (probability < 0) | (probability > 100)
    cloud_shadow = np.isin(scl, (3, 8, 9, 10)) | (probability >= 40)
    y, x = np.ogrid[-6:7, -6:7]
    vicinity = binary_dilation(cloud_shadow, structure=(x*x + y*y <= 36))
    return ~np.isin(scl, INVALID_SCL) & ~missing & ~vicinity


def water_observation(bands, quality):
    green, red, nir, swir = (np.asarray(bands[b], float) for b in ("B3", "B4", "B8", "B11"))
    valid = quality.copy()
    for array in (green, red, nir, swir):
        if array.shape != valid.shape:
            raise ValueError('E_PREPROCESS_044')
        valid &= np.isfinite(array) & (array >= 0) & (array <= 1)
    dg, dr, ds = green+nir, nir+red, green+swir
    valid &= (dg > 0) & (dr > 0) & (ds > 0)
    ndwi = np.divide(green-nir, dg, out=np.zeros_like(dg), where=dg > 0)
    ndvi = np.divide(nir-red, dr, out=np.zeros_like(dr), where=dr > 0)
    mndwi = np.divide(green-swir, ds, out=np.zeros_like(ds), where=ds > 0)
    water = valid & (ndvi < .30) & ((ndwi > 0) | ((mndwi > .10) & (nir < .15)))
    return water, valid


def _load_scene(entry, base):
    path = (base / entry["file"]).resolve()
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key].copy() for key in z.files}


def daily_water(entries, base):
    
    best = None
    selected_water = selected_valid = None
    for entry in entries:
        arrays = _load_scene(entry, base)
        if "cloud_probability" not in arrays:

            continue
        quality = scene_quality(arrays["SCL"], arrays["cloud_probability"])
        water, valid = water_observation(arrays, quality)
        probability = np.asarray(arrays["cloud_probability"], float)
        if best is None:
            best = np.full(valid.shape, np.inf)
            selected_water = np.zeros(valid.shape, bool)
            selected_valid = np.zeros(valid.shape, bool)
        elif best.shape != valid.shape:
            raise ValueError('E_PREPROCESS_078')
        take = valid & (probability < best)
        best[take] = probability[take]
        selected_water[take] = water[take]
        selected_valid[take] = True
    if best is None:
        raise ValueError('E_PREPROCESS_084')
    return selected_water, selected_valid


def reference_domain(manifest_path, output):
    
    path = Path(manifest_path)
    cfg = json.loads(path.read_text(encoding="utf-8"))
    base = path.parent
    by_month = defaultdict(lambda: defaultdict(list))
    for entry in cfg["scenes"]:

        date = entry["local_date"]
        year, month = int(date[:4]), int(date[5:7])
        if year in (2021, 2022, 2023) and 4 <= month <= 10:
            by_month[(year, month)][date].append(entry)
    with np.load(base / cfg["domain_constraints"], allow_pickle=False) as z:
        candidate = z["candidate"].astype(bool)

        association = z["association"].astype(bool)
        review = z["review"].astype(np.int8) if "review" in z else np.full(candidate.shape, -1, np.int8)
    if candidate.shape != association.shape or review.shape != candidate.shape:
        raise ValueError('E_PREPROCESS_106')
    if not np.isin(review, (-1, 0, 1)).all():
        raise ValueError('E_PREPROCESS_108')
    score_sum = np.zeros(candidate.shape, np.float32)
    eligible = np.zeros(candidate.shape, np.uint8)
    years = np.zeros(candidate.shape, np.uint8)
    months = np.zeros(candidate.shape, np.uint8)
    for (year, month), days in sorted(by_month.items()):
        valid_count = np.zeros(candidate.shape, np.uint16)
        water_count = np.zeros(candidate.shape, np.uint16)
        for entries in days.values():
            if not any("cloud_probability" in _load_scene(e, base) for e in entries):
                continue
            water, valid = daily_water(entries, base)
            valid_count += valid
            water_count += water
        good = valid_count >= 2
        fraction = np.divide(water_count, valid_count, out=np.zeros(candidate.shape, np.float32), where=good)
        score_sum += fraction
        eligible += good
        years[good] |= 1 << (year-2021)
        months[good] |= 1 << (month-4)
    score = np.divide(score_sum, eligible, out=np.full(candidate.shape, np.nan, np.float32), where=eligible > 0)
    popcount = np.array([int(i).bit_count() for i in range(128)], np.uint8)
    supported = (eligible >= 12) & (years == 7) & (popcount[months] >= 4)
    preliminary = candidate & association & supported & (score >= .50)
    boundary = binary_dilation(preliminary, structure=FOUR) ^ binary_erosion(preliminary, structure=FOUR)
    yy, xx = np.ogrid[-3:4, -3:4]
    shoreline = binary_dilation(boundary, structure=xx*xx+yy*yy <= 9)
    near_threshold = np.isfinite(score) & (score >= .45) & (score <= .55)

    padded = np.pad(preliminary, 1)
    up, down = padded[:-2, 1:-1], padded[2:, 1:-1]
    left, right = padded[1:-1, :-2], padded[1:-1, 2:]
    narrow = preliminary & ((up & down & ~left & ~right) | (left & right & ~up & ~down))
    review_required = candidate & association & (shoreline | near_threshold | ~supported | narrow)
    result = preliminary.copy()
    result[review_required] = review[review_required] == 1

    result[(review == 0) & candidate] = False
    result[(review == 1) & candidate & association] = True
    components, _ = label(result, structure=FOUR)
    sizes = np.bincount(components.ravel())
    result &= sizes[components] >= 9
    components, number = label(result, structure=FOUR)
    transform = np.asarray(cfg["transform"], float)
    if transform.shape != (6,) or not np.allclose(transform[[1, 2, 4, 5]], [10, 0, 0, -10]):
        raise ValueError('E_PREPROCESS_153')
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, water=result, components=components,
                        score=score, eligible_months=eligible,
                        review_required=review_required, review=review,
                        transform=transform, epsg=np.int32(cfg["epsg"]))
    output.with_suffix(".json").write_text(json.dumps({
        "water_cells": int(result.sum()), "water_area_m2": int(result.sum())*100,
        "components": int(number), "cell_size_m": 10, "epsg": cfg["epsg"],
        "transform": transform.tolist(),
    }, indent=2), encoding="utf-8")


def full_footprints(valid10, size):
    
    if size == 10:
        return valid10
    if size != 20 or any(n % 2 for n in valid10.shape):
        raise ValueError('E_PREPROCESS_172')
    h, w = valid10.shape
    return valid10.reshape(h//2, 2, w//2, 2).all(axis=(1, 3))


def balanced_indices(valid, budget=128):
    
    candidates = np.column_stack(np.nonzero(valid))
    if len(candidates) <= budget:
        return candidates
    centre = candidates.mean(0)
    current = int(np.argmin(np.sum((candidates-centre)**2, axis=1)))
    chosen = []
    minimum = np.full(len(candidates), np.inf)
    for _ in range(budget):
        chosen.append(current)
        minimum = np.minimum(minimum, np.sum((candidates-candidates[current])**2, axis=1))
        minimum[chosen] = -1
        current = int(np.argmax(minimum))
    return candidates[chosen]


def optical_records(manifest_path, domain_path, output):
    
    path = Path(manifest_path)
    cfg = json.loads(path.read_text(encoding="utf-8"))
    with np.load(domain_path, allow_pickle=False) as z:
        water, transform = z["water"].astype(bool), z["transform"]
    records = []
    catalog = []
    seen = set()
    for entry in cfg["scenes"]:
        acquisition = entry["acquisition_id"]
        if acquisition in seen:
            raise ValueError('E_PREPROCESS_206')
        seen.add(acquisition)
        arrays = _load_scene(entry, path.parent)
        if "cloud_probability" not in arrays:
            continue
        quality = scene_quality(arrays["SCL"], arrays["cloud_probability"])
        time = time_days(entry["time"])
        retained = 0
        for size, bands in BAND_GROUPS.items():
            valid = full_footprints(quality & water, size).copy()
            for band in bands:
                a = np.asarray(arrays[band], float)
                if a.shape != valid.shape:
                    raise ValueError('E_PREPROCESS_219')
                valid &= np.isfinite(a) & (a >= 0) & (a <= 1)
            selected = balanced_indices(valid, budget=128)
            for row, col in selected:
                x = float(transform[0]+size*(col+.5))
                y = float(transform[3]-size*(row+.5))

                delta = size/(2*np.sqrt(3))
                support = [[x+sx*delta, y+sy*delta, time, .25] for sx in (-1, 1) for sy in (-1, 1)]
                for band in bands:
                    records.append({"kind": "optical", "variable": band, "value": float(arrays[band][row, col]),
                                    "scene": acquisition, "support": support,
                                    "processing_baseline": entry["processing_baseline"],
                                    "footprint_id": f"{acquisition}_{size}_{row}_{col}"})
            retained += len(selected)*len(bands)
        catalog.append({"acquisition_id": acquisition, "time": entry["time"], "retained_scalar_records": retained})
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(records, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    out.with_suffix(".catalog.json").write_text(json.dumps(catalog, indent=2), encoding="utf-8")


def assemble(config_path, output):
    
    path = Path(config_path)
    cfg = json.loads(path.read_text(encoding="utf-8"))
    def read(name):
        return json.loads((path.parent/cfg[name]).read_text(encoding="utf-8"))
    records = read("concentration_records")
    if "optical_records" in cfg:
        records.extend(read("optical_records"))
    dataset = {"meshes": read("meshes"), "weather": read("weather"), "records": records}
    if "position_aliases" in cfg:
        aliases = cfg["position_aliases"]
        if isinstance(aliases, str):
            aliases = json.loads((path.parent/aliases).read_text(encoding="utf-8"))
        if not isinstance(aliases, dict) or any(not isinstance(v, list) for v in aliases.values()):
            raise ValueError('E_PREPROCESS_256')
        dataset["position_aliases"] = aliases
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dataset, separators=(",", ":"), allow_nan=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    mask = sub.add_parser("mask")
    mask.add_argument("--manifest", required=True)
    mask.add_argument("--output", required=True)
    optical = sub.add_parser("optical")
    optical.add_argument("--manifest", required=True)
    optical.add_argument("--domain", required=True)
    optical.add_argument("--output", required=True)
    bundle = sub.add_parser("assemble")
    bundle.add_argument("--config", required=True)
    bundle.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "mask":
        reference_domain(args.manifest, args.output)
    elif args.command == "optical":
        optical_records(args.manifest, args.domain, args.output)
    else:
        assemble(args.config, args.output)


if __name__ == "__main__":
    main()
