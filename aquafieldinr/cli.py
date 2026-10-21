
import argparse
import csv
import json
import tempfile
import os
from contextlib import ExitStack
from pathlib import Path
import numpy as np
import torch
from .training import fit,save,load,validate,select
from .data import INDICATORS
from .runtime import close_posterior


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(path,rows):
    if not rows:raise ValueError('E_CLI_021')
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with Path(path).open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def summarize(values):
    values=np.asarray(values)
    return {"mean":float(values.mean()),"sd":float(values.std(ddof=1)),"q025":float(np.quantile(values,.025)),"q50":float(np.quantile(values,.5)),"q975":float(np.quantile(values,.975))}


def _query_source(path, chunk_size):
    
    path=Path(path)
    if path.suffix.lower()==".jsonl":
        metadata={}
        with path.open(encoding="utf-8") as handle:
            first=json.loads(next(handle))
            if "metadata" in first:metadata=first["metadata"]
        def factory():
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        item=json.loads(line)
                        if "metadata" not in item:yield item
        return metadata,factory
    doc=read_json(path)
    if "raster" not in doc:return doc,lambda:iter(doc["queries"])
    spec=doc["raster"]
    mask_path=Path(spec["mask_npy"])
    if not mask_path.is_absolute():mask_path=path.parent/mask_path
    mask=np.load(mask_path,mmap_mode="r")
    if mask.ndim!=2 or mask.dtype.kind!="b":raise ValueError('E_CLI_054')
    transform=np.asarray(spec["transform"],float)
    if transform.shape!=(6,) or not np.isfinite(transform).all():raise ValueError('E_CLI_056')
    def factory():
        for t in spec["times"]:
            for variable in spec.get("variables",INDICATORS):
                for start in range(0,mask.size,chunk_size):
                    local=np.flatnonzero(mask.reshape(-1)[start:start+chunk_size])+start
                    if not len(local):continue
                    row,col=np.divmod(local,mask.shape[1]);col=col+.5;row=row+.5
                    x=transform[0]*col+transform[1]*row+transform[2]
                    y=transform[3]*col+transform[4]*row+transform[5]
                    yield {"raster_points":np.column_stack((x,y)),"pixel_index":local,"time":float(t),"variable":variable,"kind":spec.get("kind","concentration")}
    return doc,factory


def predict(checkpoint,query_path,output,draws_path=None,draws=512,seed=2021,regional=False,chunk_size=4096,device="cpu"):
    if draws<2 or chunk_size<1:raise ValueError('E_CLI_071')
    model,posterior,_=load(checkpoint,device)
    try:return _predict(model,posterior,query_path,output,draws_path,draws,seed,regional,chunk_size)
    finally:close_posterior(posterior)


def _predict(model,posterior,query_path,output,draws_path,draws,seed,regional,chunk_size):
    doc,query_factory=_query_source(query_path,chunk_size)
    requested=set();row_count=0;raster_mode="raster" in doc
    if raster_mode and regional:raise ValueError('E_CLI_080')
    for q in query_factory():
        if "raster_points" in q:
            requested.add(q["time"]);row_count+=len(q["raster_points"])
        else:
            support=np.asarray(q["support"],float)
            if support.ndim!=2 or support.shape[1]!=4 or not len(support) or not np.isfinite(support).all() or np.any(support[:,3]<0) or not np.isclose(support[:,3].sum(),1):
                raise ValueError('E_CLI_087')
            requested.update(support[:,2]);row_count+=(7 if "thresholds" in q else 3) if regional else 1
    if not row_count:raise ValueError('E_CLI_089')
    weather=model.data.normalized_query_weather(doc["weather"]) if "weather" in doc else None
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="aquafield-output-",dir=output.parent) as temporary, ExitStack() as resources:
        temporary=Path(temporary)
        if hasattr(posterior,"complete_to_disk"):
            times,states=posterior.complete_to_disk(sorted(requested),draws,temporary/"states",weather,seed)
        else:times,states=posterior.complete(sorted(requested),draws,weather,seed)
        if hasattr(posterior,"close_completion"):resources.callback(posterior.close_completion)
        saved_draws=np.lib.format.open_memmap(temporary/"draws.npy",mode="w+",dtype=np.float64,shape=(row_count,draws)) if draws_path else None
        if saved_draws is not None:resources.callback(saved_draws._mmap.close)
        index=0
        with (temporary/"summary.csv").open("w",newline="",encoding="utf-8") as handle:
            writer=None
            def emit(row,values):
                nonlocal writer,index
                values=np.asarray(values)
                if writer is None:
                    writer=csv.DictWriter(handle,fieldnames=list(row)+list(summarize(values)))
                    writer.writeheader()
                writer.writerow({**row,**summarize(values)})
                if saved_draws is not None:saved_draws[index]=values
                index+=1
            with torch.no_grad():
                for q in query_factory():
                    kind=q.get("kind","concentration")
                    if "raster_points" in q:
                        field=model.field(states,times,q["raster_points"],np.full(len(q["raster_points"]),q["time"]),kind,q["variable"],chunk_size).cpu().numpy()
                        for pixel,values in zip(q["pixel_index"],field):
                            emit({"pixel_index":int(pixel),"time":q["time"],"variable":q["variable"]},values)
                        continue
                    support=np.asarray(q["support"],float)
                    if regional:
                        if kind!="concentration":raise ValueError('E_CLI_122')
                        means=np.zeros((3,draws));areas=np.zeros_like(means);joint=np.zeros(draws)
                        thresholds=np.asarray(q["thresholds"],float) if "thresholds" in q else None
                        if thresholds is not None and thresholds.shape!=(3,):raise ValueError('E_CLI_125')
                        for start in range(0,len(support),chunk_size):
                            block=support[start:start+chunk_size];w=block[:,3,None]
                            fields=np.stack([model.field(states,times,block[:,:2],block[:,2],kind,v,chunk_size).cpu().numpy() for v in INDICATORS])
                            means+=(fields*w[None]).sum(1)
                            if thresholds is not None:
                                high=fields>thresholds[:,None,None]
                                areas+=(high*w[None]).sum(1);joint+=(high.all(0)*w).sum(0)
                        for j,v in enumerate(INDICATORS):
                            emit({"query_id":q["id"],"statistic":"regional_mean","variable":v},means[j])
                            if thresholds is not None:emit({"query_id":q["id"],"statistic":"high_area_fraction","variable":v},areas[j])
                        if thresholds is not None:emit({"query_id":q["id"],"statistic":"joint_high_area_fraction","variable":"TN+TP+Chla"},joint)
                    else:
                        values=np.zeros(draws)
                        measurement=q.get("measurement",False)
                        generator=torch.Generator().manual_seed(seed+index+1)
                        mixture=measurement and q.get("mode")=="instantaneous_time_mixture"
                        if mixture:
                            if kind!="concentration":raise ValueError('E_CLI_143')
                            selected=torch.multinomial(torch.as_tensor(support[:,3]),draws,replacement=True,generator=generator).numpy()


                            used=np.unique(selected)
                            for start in range(0,len(used),chunk_size):
                                ids=used[start:start+chunk_size];block=support[ids]
                                field=model.field(states,times,block[:,:2],block[:,2],kind,q["variable"],chunk_size).cpu().numpy()
                                chosen=np.flatnonzero(np.isin(selected,ids))
                                values[chosen]=field[np.searchsorted(ids,selected[chosen]),chosen]
                        else:
                            for start in range(0,len(support),chunk_size):
                                block=support[start:start+chunk_size]
                                field=model.field(states,times,block[:,:2],block[:,2],kind,q["variable"],chunk_size).cpu().numpy()
                                values+=(field*block[:,3,None]).sum(0)
                        if measurement:
                            if kind!="concentration":raise ValueError('E_CLI_159')
                            j=INDICATORS.index(q["variable"])
                            sigma=model.scales[j]*(1e-4+torch.nn.functional.softplus(model.error_raw[j]))
                            values+=float(sigma.detach().cpu())*torch.randn(draws,generator=generator).numpy()
                        emit({"query_id":q["id"],"variable":q["variable"]},values)
        if index!=row_count:raise RuntimeError('E_CLI_164')
        if saved_draws is not None:
            saved_draws.flush()
            rows=np.lib.format.open_memmap(temporary/"row_index.npy",mode="w+",dtype=np.int64,shape=(row_count,))
            for start in range(0,row_count,chunk_size):rows[start:start+chunk_size]=np.arange(start,min(start+chunk_size,row_count))
            resources.callback(rows._mmap.close)
            rows.flush()
            destination=Path(draws_path);destination.parent.mkdir(parents=True,exist_ok=True)
            with (temporary/"draws.npz").open("wb") as handle:
                np.savez_compressed(handle,draws=saved_draws,row_index=rows,realization_id=np.arange(draws))
            del rows,saved_draws
            os.replace(temporary/"draws.npz",destination)
        del states
        os.replace(temporary/"summary.csv",output)


def main():
    parser=argparse.ArgumentParser(prog="aquafieldinr",allow_abbrev=False)
    sub=parser.add_subparsers(dest="command",required=True)
    train=sub.add_parser("train");train.add_argument("--data",required=True);train.add_argument("--config");train.add_argument("--output",required=True)
    train.add_argument("--exclude-position",action="append",default=[],help='')
    for command in ("predict","regional"):
        p=sub.add_parser(command);p.add_argument("--checkpoint",required=True);p.add_argument("--queries",required=True);p.add_argument("--output",required=True);p.add_argument("--draws-output");p.add_argument("--draws",type=int,default=512);p.add_argument("--seed",type=int,default=2021);p.add_argument("--chunk-size",type=int,default=4096)
    val=sub.add_parser("validate");val.add_argument("--data",required=True);val.add_argument("--config");val.add_argument("--output",required=True);val.add_argument("--draws",type=int,default=128)
    sel=sub.add_parser("select");sel.add_argument("--data",required=True);sel.add_argument("--candidates",required=True);sel.add_argument("--output",required=True);sel.add_argument("--draws",type=int,default=128);sel.add_argument("--stage",choices=("concentration","optical"),default="concentration")
    nested=sub.add_parser("nested-validate");nested.add_argument("--data",required=True);nested.add_argument("--concentration-candidates",required=True);nested.add_argument("--optical-candidates",required=True);nested.add_argument("--output-dir",required=True);nested.add_argument("--draws",type=int,default=128)
    dispatch_parser=sub.add_parser("dispatch");dispatch_parser.add_argument("--manifest",required=True);dispatch_parser.add_argument("--devices",required=True);dispatch_parser.add_argument("--output-dir",required=True)
    for command_parser in (train,val,sel,nested,*[sub.choices[name] for name in ("predict","regional")]):
        command_parser.add_argument("--device",default="cpu",help="cpu, cuda or cuda:N; CUDA requests fail if unavailable.")
    for command_parser in sub.choices.values():command_parser.allow_abbrev=False
    args=parser.parse_args()
    if args.command=="dispatch":
        from .dispatch import dispatch
        dispatch(args.manifest,args.devices.split(","),args.output_dir)
        return
    if args.command=="train":
        checkpoint=fit(args.data,read_json(args.config) if args.config else None,args.exclude_position,device=args.device)
        save(checkpoint,args.output)
    elif args.command=="nested-validate":
        from .workflow import nested_validate
        nested_validate(args.data,read_json(args.concentration_candidates),read_json(args.optical_candidates),args.draws,args.output_dir,args.device)
    elif args.command=="validate":
        write_csv(args.output,validate(args.data,read_json(args.config) if args.config else None,args.draws,device=args.device))
    elif args.command=="select":
        Path(args.output).write_text(json.dumps(select(args.data,read_json(args.candidates),args.draws,args.stage,device=args.device),indent=2),encoding="utf-8")
    else:
        predict(args.checkpoint,args.queries,args.output,args.draws_output,args.draws,args.seed,args.command=="regional",args.chunk_size,args.device)


if __name__=="__main__":main()
