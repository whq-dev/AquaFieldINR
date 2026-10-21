
import copy
import json
from pathlib import Path
import numpy as np
from .geometry import Mesh,ConstantBasis

BANDS = ("B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A")
INDICATORS = ("TN", "TP", "Chla")


class Dataset:
    def __init__(self, path, excluded_positions=()):
        self.document = json.loads(Path(path).read_text(encoding="utf-8"))
        self.meshes = [Mesh(**m) for m in self.document["meshes"]]
        if not self.meshes:raise ValueError('E_DATA_016')
        self.meshes.append(ConstantBasis(self.meshes[0]))
        excluded=set(excluded_positions)
        self.records = [copy.deepcopy(r) for r in self.document["records"] if r.get("position_id") not in excluded and r.get("source_position_id") not in excluded]
        if not self.records:
            raise ValueError('E_DATA_021')
        for r in self.records:
            if r["kind"] not in ("concentration", "optical"):
                raise ValueError('E_DATA_024')
            if r["kind"]=="concentration" and not r.get("position_id"):
                raise ValueError('E_DATA_026')
            if r["variable"] not in (INDICATORS if r["kind"] == "concentration" else BANDS):
                raise ValueError('E_DATA_028')
            a = np.asarray(r["support"], float)
            if a.ndim != 2 or a.shape[1] != 4 or not np.isfinite(a).all():
                raise ValueError('E_DATA_031')
            if np.any(a[:,3] < 0) or not np.isclose(a[:,3].sum(),1):
                raise ValueError('E_DATA_033')
            if r.get("mode", "average") not in ("average", "instantaneous_time_mixture"):
                raise ValueError('E_DATA_035')
            if r["kind"] == "optical" and ("scene" not in r or np.ptp(a[:,2]) > 1e-10):
                raise ValueError('E_DATA_037')
            if ("value" in r) == ("interval" in r):
                raise ValueError('E_DATA_039')
            if "value" in r and not np.isfinite(r["value"]):
                raise ValueError('E_DATA_041')
            if r["kind"]=="concentration" and "value" in r and r["value"]<=0:
                raise ValueError('E_DATA_043')
            if "interval" in r:
                lower,upper = r["interval"]
                if not lower < upper or r["kind"] == "optical":
                    raise ValueError('E_DATA_047')
        self._attach_error_blocks()
        self.times = np.unique(np.concatenate([np.asarray(r["support"])[:,2] for r in self.records]))
        self.weather = np.asarray(self.document["weather"], float)
        if self.weather.ndim != 2 or self.weather.shape[1] != 5 or not np.isfinite(self.weather).all():
            raise ValueError('E_DATA_052')
        if not np.all(np.diff(self.weather[:,0]) == 1) or np.any(self.weather[:,0] != np.floor(self.weather[:,0])):
            raise ValueError('E_DATA_054')
        if self.weather[0,0] > np.floor(self.times.min()) or self.weather[-1,0]+1 < self.times.max():
            raise ValueError('E_DATA_056')
        dates = np.unique(np.floor(self.times))
        train_weather = self.weather[np.isin(self.weather[:,0], dates),1:]
        self.weather_center = train_weather.mean(0)
        self.weather_scale = train_weather.std(0)
        self.weather_scale[self.weather_scale == 0] = 1
        self.weather[:,1:] = (self.weather[:,1:]-self.weather_center)/self.weather_scale
        self.scales = []
        for variable in INDICATORS:
            values = [r["value"] for r in self.records if r["kind"] == "concentration" and r["variable"] == variable and r.get("value",0)>0]
            if not values:
                raise ValueError('E_DATA_067')
            self.scales.append(float(np.median(values)/np.logaddexp(0,1)))
        self.optical_scales = []
        for variable in BANDS:
            values = [r["value"] for r in self.records if r["kind"] == "optical" and r["variable"] == variable]
            self.optical_scales.append(max(float(np.std(values)),1e-4) if values else 1e-4)

    def _attach_error_blocks(self):
        specifications = self.document.get("concentration_error_blocks", {})
        original = {r["record_id"]: r for r in self.document["records"] if "record_id" in r}
        if len(original) != sum("record_id" in r for r in self.document["records"]):
            raise ValueError('E_DATA_078')
        retained = {r["record_id"]: r for r in self.records if "record_id" in r}
        assigned = set()
        for identity, spec in specifications.items():
            ids = spec["record_ids"]
            if not ids or len(set(ids)) != len(ids) or assigned.intersection(ids):
                raise ValueError('E_DATA_084')
            if any(i not in original or original[i]["kind"] != "concentration" for i in ids):
                raise ValueError('E_DATA_086')
            assigned.update(ids)
            correlation = np.asarray(spec["correlation"], dtype=float)
            if correlation.shape != (len(ids), len(ids)) or not np.isfinite(correlation).all():
                raise ValueError('E_DATA_090')
            if not np.allclose(correlation, correlation.T, atol=1e-12) or not np.allclose(correlation.diagonal(), 1):
                raise ValueError('E_DATA_092')
            try:
                np.linalg.cholesky(correlation)
            except np.linalg.LinAlgError as error:
                raise ValueError('E_DATA_096') from error
            points = spec.get("rectangle_points", 4096)
            limit = spec.get("max_time_combinations", 65536)
            if type(points) is not int or points < 2 or points & (points - 1):
                raise ValueError('E_DATA_100')
            if type(limit) is not int or limit < 1:
                raise ValueError('E_DATA_102')
            kept = [k for k, rid in enumerate(ids) if rid in retained]
            reduced = correlation[np.ix_(kept, kept)].tolist()
            for order, k in enumerate(kept):
                row = retained[ids[k]]
                row["_error_block"] = identity
                row["_error_order"] = order
                row["_error_correlation"] = reduced
                row["_rectangle_points"] = points
                row["_time_mixture_limit"] = limit

    def normalized_query_weather(self, raw):
        raw = np.asarray(raw,float).copy()
        if raw.ndim!=2 or raw.shape[1]!=5 or not np.isfinite(raw).all() or not np.all(np.diff(raw[:,0])==1):
            raise ValueError('E_DATA_116')
        raw[:,1:] = (raw[:,1:]-self.weather_center)/self.weather_scale
        common,original_index,query_index=np.intersect1d(self.weather[:,0],raw[:,0],return_indices=True)
        if len(common) and not np.allclose(self.weather[original_index,1:],raw[query_index,1:],rtol=1e-10,atol=1e-10):
            raise ValueError('E_DATA_120')
        return raw
