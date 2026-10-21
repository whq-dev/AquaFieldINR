
import copy
import json
from pathlib import Path
import numpy as np
import torch
import time
import math
import tempfile
from .runtime import resolve_device, isolated_rng, seed_streams, cpu_state, close_posterior
from .data import Dataset
from .model import AquaFieldINR
from .inference import MarkovPosterior
from .exact import ExactActivePosterior
from .exact_dense import ExactActivePosterior as DenseLegacyPosterior
from .canonical import canonical_representation
from .likelihood import groups,log_likelihood,regularization


DEFAULTS={"backend":"exact_active","max_active_dimension":6000,"shared_rank":2,"optical_rank":1,"joint_steps":4000,"refine_steps":1000,"starts":3,"learning_rate":.001,"draws_per_step":2,"validation_draws":32,"check_every":100,"relative_tolerance":1e-4,"patience":5,"batch_groups":16,"alpha_lambda":.001,"alpha_gamma":.001,"alpha2":.0001,"seed":2021,"threads":4}


def fit(data_path,config=None,excluded_positions=(),device="cpu"):
    with isolated_rng(device) as execution_device:
        return _fit(data_path,config,excluded_positions,execution_device)


def _fit(data_path,config,excluded_positions,device):
    settings=DEFAULTS| (config or {})
    torch.set_num_threads(int(settings["threads"]))
    document=json.loads(Path(data_path).read_text(encoding="utf-8"))
    excluded_positions=expand_exclusions(document,excluded_positions)
    dataset=Dataset(data_path,excluded_positions)
    observations=groups(dataset.records)
    best=None;best_score=float("inf");histories=[]
    for start in range(int(settings["starts"])):
        candidate,score,history=_fit_start(dataset,observations,settings,start,device)
        if score<best_score:best=candidate;best_score=score
        histories.append(history)
    best.update({"schema_version":2,"settings":settings,"data":dataset.document,"excluded_positions":list(excluded_positions),"objective":best_score,"history":histories})
    return best


def _fit_start(dataset,observations,settings,start,device):
    posterior=None
    try:
        seed_streams(int(settings["seed"])+start,device)
        rng=np.random.default_rng(int(settings["seed"])+start)
        exact=settings["backend"]=="exact_active"
        if settings["backend"] not in ("exact_active","markov"):raise ValueError("Unknown inference backend.")
        model=AquaFieldINR(dataset,settings["shared_rank"],settings["optical_rank"],settings["max_active_dimension"],"sparse" if exact else "markov")
        posterior=ExactActivePosterior(model) if exact else MarkovPosterior(model)
        posterior.to(device)
        history=[]
        def objective(batch,count):
            states,kl=posterior.sample_and_kl(count)
            ll=log_likelihood(model,states,batch)*len(observations)/len(batch)
            return (kl-ll)/len(dataset.records)+regularization(model,settings["alpha_lambda"],settings["alpha_gamma"],settings["alpha2"])
        for stage,steps in (("joint",settings["joint_steps"]),("state",settings["refine_steps"])):
            for p in model.parameters():p.requires_grad_(stage=="joint")
            if exact:
                for parameter in posterior.variational_parameters():parameter.requires_grad_(True)
            # model is registered under posterior, so remove frozen parameters.
            parameters=[p for p in posterior.parameters() if p.requires_grad]
            optimizer=torch.optim.Adam(parameters,lr=settings["learning_rate"])
            stale=0;previous=float("inf")
            for step in range(int(steps)):
                chosen=rng.choice(len(observations),min(int(settings["batch_groups"]),len(observations)),replace=False)
                batch=[observations[int(i)] for i in chosen]
                optimizer.zero_grad()
                loss=objective(batch,int(settings["draws_per_step"]))
                if not torch.isfinite(loss):raise FloatingPointError("Nonfinite optimization objective.")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters,5)
                optimizer.step()
                if (step+1)%int(settings["check_every"])==0 or step+1==int(steps):
                    with torch.no_grad(), isolated_rng(device):
                        seed_streams(int(settings["seed"])+991,device)
                        value=float(objective(observations,int(settings["validation_draws"])))
                    history.append({"stage":stage,"step":step+1,"objective":value})
                    if np.isfinite(previous) and (previous-value)/max(abs(previous),1)<settings["relative_tolerance"]:stale+=1
                    else:stale=0
                    previous=value
                    if stale>=int(settings["patience"]):break
        with torch.no_grad():
            seed_streams(int(settings["seed"])+991,device)
            score=float(objective(observations,int(settings["validation_draws"])))
        best={"model_state":cpu_state(model.state_dict()),"posterior_state":cpu_state(posterior.state_dict()),"canonical":canonical_representation(model)}
        return best,score,history
    finally:
        if posterior is not None:close_posterior(posterior)


def save(checkpoint,path):
    _validate_checkpoint(checkpoint)
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    torch.save(checkpoint,path)


def _validate_checkpoint(checkpoint):
    
    required={"model_state","posterior_state","settings","data","excluded_positions","objective","history"}
    allowed=required|{"schema_version","canonical"}
    if not isinstance(checkpoint,dict) or not required.issubset(checkpoint) or not set(checkpoint).issubset(allowed):
        raise ValueError('E_TRAINING_105')
    def json_value(value,path,depth=0):
        if depth>64:raise ValueError('E_TRAINING_107')
        if value is None or type(value) in (str,int,float,bool):return
        if type(value) is list:
            for i,child in enumerate(value):json_value(child,f"{path}[{i}]",depth+1)
            return
        if type(value) is dict and all(type(key) is str for key in value):
            for key,child in value.items():json_value(child,f"{path}.{key}",depth+1)
            return
        raise ValueError('E_TRAINING_115')
    if checkpoint.get("schema_version",1) not in (1,2):raise ValueError('E_TRAINING_116')
    if checkpoint.get("schema_version",1)==2 and "canonical" not in checkpoint:raise ValueError('E_TRAINING_117')
    if "canonical" in checkpoint:json_value(checkpoint["canonical"],"canonical")
    for key in ("settings","data","excluded_positions","objective","history"):
        json_value(checkpoint[key],key)
    settings=checkpoint["settings"]
    if type(settings) is not dict or set(settings)!=set(DEFAULTS):
        raise ValueError('E_TRAINING_123')
    if settings["backend"] not in ("exact_active","markov"):
        raise ValueError('E_TRAINING_125')
    integer_keys=("max_active_dimension","shared_rank","optical_rank","joint_steps","refine_steps","starts","draws_per_step","validation_draws","check_every","patience","batch_groups","seed","threads")
    for key in integer_keys:
        lower=0 if key in ("shared_rank","optical_rank","joint_steps","refine_steps","seed") else 1
        if type(settings[key]) is not int or settings[key]<lower:
            raise ValueError('E_TRAINING_130')
    for key in ("learning_rate","relative_tolerance","alpha_lambda","alpha_gamma","alpha2"):
        value=settings[key]
        if type(value) not in (float,int) or not math.isfinite(value) or value<0 or (key=="learning_rate" and value==0):
            raise ValueError('E_TRAINING_134')
    if type(checkpoint["data"]) is not dict or type(checkpoint["excluded_positions"]) is not list or type(checkpoint["history"]) is not list:
        raise ValueError('E_TRAINING_136')
    if any(type(value) not in (str,int) for value in checkpoint["excluded_positions"]):
        raise ValueError('E_TRAINING_138')
    if type(checkpoint["objective"]) not in (int,float) or not math.isfinite(checkpoint["objective"]):
        raise ValueError('E_TRAINING_140')
    for key in ("model_state","posterior_state"):
        state=checkpoint[key]
        if not isinstance(state,dict) or not state:
            raise ValueError('E_TRAINING_144')
        for name,value in state.items():
            if type(name) is not str or type(value) is not torch.Tensor or value.layout!=torch.strided or not torch.isfinite(value).all():
                raise ValueError('E_TRAINING_147')
    for name,value in checkpoint["model_state"].items():
        duplicate=checkpoint["posterior_state"].get("model."+name)
        if duplicate is None or value.shape!=duplicate.shape or value.dtype!=duplicate.dtype or not torch.equal(value,duplicate):
            raise ValueError('E_TRAINING_151')


def load(path,device="cpu"):
    with isolated_rng(device) as execution_device:
        return _load(path,execution_device)


def _load(path,device):
    import tempfile
    cp=torch.load(path,map_location="cpu",weights_only=True)
    _validate_checkpoint(cp)
    with tempfile.TemporaryDirectory() as temp:
        data_path=Path(temp)/"data.json"
        data_path.write_text(json.dumps(cp["data"]),encoding="utf-8")
        data=Dataset(data_path,cp["excluded_positions"])
    exact=cp["settings"]["backend"]=="exact_active"
    model=AquaFieldINR(data,cp["settings"]["shared_rank"],cp["settings"]["optical_rank"],cp["settings"]["max_active_dimension"],("sparse" if cp.get("schema_version",1)>=2 else "dense") if exact else "markov")
    posterior=(ExactActivePosterior(model) if cp.get("schema_version",1)>=2 else DenseLegacyPosterior(model)) if exact else MarkovPosterior(model)
    try:
        reconstructed=model.state_dict()
        derived=("active","scales","optical_scales","factor_rows","factor_cols","factor_permutation","factor_mask","factor_inverse_permutation","inverse_permutation")
        for name in (key for key in derived if key in reconstructed):
            expected=reconstructed[name]
            supplied=cp["model_state"].get(name)
            if supplied is None or expected.shape!=supplied.shape or expected.dtype!=supplied.dtype or not torch.equal(expected,supplied):
                raise ValueError('E_TRAINING_177')
        model.load_state_dict(cp["model_state"])
        posterior.load_state_dict(cp["posterior_state"])
        if "canonical" in cp and cp["canonical"]!=canonical_representation(model):raise ValueError('E_TRAINING_180')
        posterior.to(device)
        return model,posterior,cp
    except BaseException:
        close_posterior(posterior)
        raise


def validate(data_path,config=None,draws=128,excluded_positions=(),device="cpu"):
    
    source_document=json.loads(Path(data_path).read_text(encoding="utf-8"))
    excluded_positions=expand_exclusions(source_document,excluded_positions)
    source=Dataset(data_path,excluded_positions)
    physical=physical_position_groups(source_document)
    physical={key:aliases for key,aliases in physical.items() if not set(aliases)&{str(value) for value in excluded_positions}}
    results=[]
    for identity,aliases in sorted(physical.items()):
        position=sorted(aliases)[0]
        cp=fit(data_path,config,list(excluded_positions)+aliases,device)
        from .workflow import _evaluate
        held=[r for r in source.records if r["kind"]=="concentration" and str(r["position_id"]) in aliases]
        rows=_evaluate(cp,held,draws,device)
        for row in rows:row["position_id"]=position
        results.extend(rows)
    return results


def select(data_path,candidates,draws=128,stage="concentration",excluded_positions=(),device="cpu"):
    
    if stage not in ("concentration","optical"):raise ValueError("Unknown candidate selection stage.")
    evaluations=[]
    for index,candidate in enumerate(candidates):
        started=time.perf_counter()
        candidate_path=candidate.get("data",str(data_path))
        config=candidate.get("config",{})
        with tempfile.TemporaryDirectory() as temporary:
            validation_path=candidate_path
            if stage=="concentration":
                concentration=json.loads(Path(candidate_path).read_text(encoding="utf-8"))
                concentration["records"]=[r for r in concentration["records"] if r["kind"]=="concentration"]
                validation_path=Path(temporary)/"concentration.json"
                validation_path.write_text(json.dumps(concentration),encoding="utf-8")
            rows=validate(validation_path,config,draws,excluded_positions,device)
        grouped={}
        for row in rows:
            if "scaled_crps" not in row:continue
            grouped.setdefault((row["position_id"],row["variable"]),[]).append(row["scaled_crps"])
        positions=sorted({p for p,v in grouped})
        position_scores=[float(np.mean([np.mean(values) for (p,v),values in grouped.items() if p==position])) for position in positions]
        if len(position_scores)<2:raise ValueError('E_TRAINING_229')
        data=Dataset(candidate_path)
        nodes=sum(len(mesh.nodes) for mesh in data.meshes)
        evaluations.append({"candidate":index,"mean":float(np.mean(position_scores)),"standard_error":float(np.std(position_scores,ddof=1)/np.sqrt(len(position_scores))),"geometry_nodes":nodes,"shared_rank":config.get("shared_rank",DEFAULTS["shared_rank"]),"optical_rank":config.get("optical_rank",DEFAULTS["optical_rank"]),"alpha_gamma":config.get("alpha_gamma",DEFAULTS["alpha_gamma"]),"elapsed_seconds":time.perf_counter()-started,"position_scores":dict(zip(positions,position_scores))})
    best=min(evaluations,key=lambda r:r["mean"])
    eligible=[r for r in evaluations if r["mean"]<=best["mean"]+best["standard_error"]]
    if stage=="optical":
        if len({(r["geometry_nodes"],r["shared_rank"]) for r in evaluations})!=1:
            raise ValueError('E_TRAINING_237')
        retained=min(eligible,key=lambda r:(r["optical_rank"],-r["alpha_gamma"]))
    else:
        retained=min(eligible,key=lambda r:(r["geometry_nodes"],r["shared_rank"],r["elapsed_seconds"]))
    return {"stage":stage,"selected_candidate":retained["candidate"],"selected":candidates[retained["candidate"]],"minimum_score_candidate":best["candidate"],"one_standard_error_limit":best["mean"]+best["standard_error"],"evaluations":evaluations}


def physical_position_groups(document):
    
    parent={}
    def find(value):
        value=str(value)
        parent.setdefault(value,value)
        if parent[value]!=value:parent[value]=find(parent[value])
        return parent[value]
    def union(a,b):
        aa,bb=find(a),find(b)
        parent[max(aa,bb)]=min(aa,bb)
    for position,aliases in document.get("position_aliases",{}).items():
        find(position)
        for alias in aliases:union(position,alias)
    coordinates={};observed=set()
    for record in document["records"]:
        if record["kind"]!="concentration":continue
        position=str(record["position_id"]);observed.add(position);find(position)
        if record.get("source_position_id") is not None:union(position,record["source_position_id"])
        xy=tuple(sorted({tuple(row[:2]) for row in record["support"]}))
        if xy in coordinates:union(position,coordinates[xy])
        coordinates[xy]=position
    result={}
    for value in parent:result.setdefault(find(value),[]).append(value)
    return {key:sorted(values) for key,values in result.items() if set(values)&observed}


def expand_exclusions(document,excluded_positions):
    excluded={str(value) for value in excluded_positions}
    for aliases in physical_position_groups(document).values():
        if excluded.intersection(aliases):excluded.update(aliases)
    result=sorted(excluded)
    for record in document["records"]:
        for key in ("position_id","source_position_id"):
            value=record.get(key)
            if type(value) is int and str(value) in excluded and value not in result:result.append(value)
    return result
