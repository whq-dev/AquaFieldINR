
import hashlib
import json
from pathlib import Path
import tempfile
import numpy as np
import torch
from .data import Dataset, INDICATORS
from .runtime import isolated_rng, close_posterior
from .training import fit, save, load, select, physical_position_groups, expand_exclusions


def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def _concentrations(document):
    return sorted((json.dumps(r,sort_keys=True) for r in document["records"] if r["kind"]=="concentration"))


def _evaluate(checkpoint,held,draws,device="cpu"):
    with tempfile.TemporaryDirectory() as temp:
        path=Path(temp)/"fit.pt";save(checkpoint,path);model,posterior,_=load(path,device)
    try:
        numeric=[r for r in held if "value" in r]
        skipped=[{"position_id":r["position_id"],"variable":r["variable"],"status":"not_scored","reason":'E_WORKFLOW_026',"interval":r["interval"]} for r in held if "value" not in r]
        if not numeric:return skipped
        times=np.unique(np.concatenate([np.asarray(r["support"])[:,2] for r in numeric]))
        full_times,states=posterior.complete(times,draws,seed=int(checkpoint["settings"]["seed"]))
        rows=[]
        with torch.no_grad(),isolated_rng(device,int(checkpoint["settings"]["seed"])+137):
            for record in numeric:
                support=np.asarray(record["support"])
                field=model.field(states,full_times,support[:,:2],support[:,2],"concentration",record["variable"])
                weights=torch.as_tensor(support[:,3,None],device=field.device,dtype=field.dtype)
                if record.get("mode")=="instantaneous_time_mixture":
                    indices=torch.multinomial(weights[:,0],draws,replacement=True)
                    values=field[indices,torch.arange(draws,device=field.device)]
                else:values=(field*weights).sum(0)
                j=INDICATORS.index(record["variable"])
                error=model.scales[j]*(1e-4+torch.nn.functional.softplus(model.error_raw[j]))
                values=values+error*torch.randn_like(values)
                observed=record["value"]
                crps=(values-observed).abs().mean()-.5*(values[:,None]-values[None,:]).abs().mean()
                rows.append({"position_id":record["position_id"],"variable":record["variable"],"observed":observed,"predicted":float(values.mean()),"crps":float(crps),"scaled_crps":float(crps/model.scales[j])})
        return rows+skipped
    finally:
        close_posterior(posterior)


def nested_validate(data_path,concentration_candidates,optical_candidates,draws=128,output_dir=None,device="cpu"):
    
    base=json.loads(Path(data_path).read_text(encoding="utf-8"))
    if not concentration_candidates:raise ValueError('E_WORKFLOW_060')
    if not optical_candidates:raise ValueError('E_WORKFLOW_061')
    if not any(r["kind"]=="optical" for r in base["records"]):raise ValueError('E_WORKFLOW_062')
    groups=physical_position_groups(base)
    if len(groups)<3:raise ValueError('E_WORKFLOW_064')
    source_hashes={str(Path(data_path).resolve()):hashlib.sha256(Path(data_path).read_bytes()).hexdigest()}
    originals=[]
    for candidate in concentration_candidates:
        path=Path(candidate.get("data",data_path));document=json.loads(path.read_text(encoding="utf-8"))
        if _concentrations(document)!=_concentrations(base) or document["weather"]!=base["weather"] or [r for r in document["records"] if r["kind"]=="optical"]!=[r for r in base["records"] if r["kind"]=="optical"]:
            raise ValueError('E_WORKFLOW_070')
        if physical_position_groups(document)!=groups:
            raise ValueError('E_WORKFLOW_072')
        originals.append(document)
        source_hashes[str(path.resolve())]=hashlib.sha256(path.read_bytes()).hexdigest()
    output=Path(output_dir) if output_dir else None
    if output:output.mkdir(parents=True,exist_ok=True)
    folds=[]
    with tempfile.TemporaryDirectory() as temporary:
        root=Path(temporary);prepared=[]
        for index,(candidate,document) in enumerate(zip(concentration_candidates,originals)):
            concentration=dict(document);concentration["records"]=[r for r in document["records"] if r["kind"]=="concentration"]
            target=root/f"concentration_{index}.json";target.write_text(json.dumps(concentration),encoding="utf-8")
            prepared.append({"data":str(target),"config":candidate.get("config",{})})
        def run_selection(excluded,label):
            first=select(data_path,prepared,draws,"concentration",excluded,device)
            chosen=first["selected_candidate"];retained=originals[chosen]
            path=root/f"multisource_{label}.json";path.write_text(json.dumps(retained),encoding="utf-8")
            config=concentration_candidates[chosen].get("config",{})
            second_candidates=[]
            for candidate in optical_candidates:
                overrides=candidate.get("config",candidate)
                if set(overrides)-{"optical_rank","alpha_gamma"}:raise ValueError('E_WORKFLOW_092')
                second_candidates.append({"data":str(path),"config":config|overrides})
            second=select(path,second_candidates,draws,"optical",excluded,device)
            final=fit(path,second["selected"]["config"],excluded,device)

            first["selected"]={"candidate":chosen,"config":prepared[chosen]["config"]}
            second["selected"]={"candidate":second["selected_candidate"],"config":second["selected"]["config"]}
            return first,second,final
        for index,(identity,aliases) in enumerate(sorted(groups.items())):
            excluded=expand_exclusions(base,aliases)
            first,second,checkpoint=run_selection(excluded,f"outer_{index}")
            held=[r for r in base["records"] if r["kind"]=="concentration" and str(r["position_id"]) in {str(value) for value in excluded}]
            rows=_evaluate(checkpoint,held,draws,device)
            fold={"position_id":identity,"excluded_positions":excluded,"concentration_selection":first,"optical_selection":second,"scores":rows,"scoring_reference_count":sum("crps" in row for row in rows),"censored_reference_count":sum(row.get("status")=="not_scored" for row in rows)}
            folds.append(fold)
            if output:
                save(checkpoint,output/f"outer_{index:03d}.pt")
                (output/f"outer_{index:03d}.json").write_text(json.dumps(fold,indent=2),encoding="utf-8")
        first,second,final=run_selection([],"all")
        if output:save(final,output/"final.pt")
    report={"schema_version":1,"source_sha256":source_hashes,"folds":folds,"final_concentration_selection":first,"final_optical_selection":second,"final_checkpoint":str(output/"final.pt") if output else None}
    if output:(output/"nested_validation.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    return report,final
