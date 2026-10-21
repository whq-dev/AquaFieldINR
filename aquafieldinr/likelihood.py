
import math
from collections import defaultdict
import numpy as np
import torch
from torch.nn import functional as F
from .data import INDICATORS
from .model import stabilize
from .correlated import concentration_block_log_likelihood


def groups(records):
    optical=defaultdict(list)
    concentration=defaultdict(list)
    result=[]
    for r in records:
        if r["kind"]=="optical":optical[r["scene"]].append(r)
        elif "_error_block" in r:concentration[r["_error_block"]].append(r)
        else:result.append([r])
    result.extend(sorted(rows, key=lambda r:r["_error_order"]) for rows in concentration.values())
    result.extend(optical.values())
    return result


def _native_predictions(model, states, records):
    
    grouped=defaultdict(list)
    for index,record in enumerate(records):
        grouped[(record["kind"],record["variable"])].append(index)
    result=[None]*len(records)
    for (kind,variable),indices in grouped.items():
        supports=[np.asarray(records[i]["support"]) for i in indices]
        joined=np.concatenate(supports)
        fields=model.field(states,model.data.times,joined[:,:2],joined[:,2],kind,variable)
        offset=0
        for index,support in zip(indices,supports):
            result[index]=(fields[offset:offset+len(support)],
                           torch.as_tensor(support[:,3],dtype=fields.dtype,device=fields.device))
            offset+=len(support)
    return result


def _prediction_groups(model,states,blocks):
    pending=[]
    def single_predictions():
        predictions=_native_predictions(model,states,[block[0] for block in pending])
        return list(zip(pending,[[p] for p in predictions]))
    for block in blocks:
        if len(block)==1 and block[0]["kind"]=="concentration" and "_error_block" not in block[0]:
            pending.append(block)
            if len(pending)>=256:
                yield from single_predictions()
                pending=[]
        else:
            if pending:
                yield from single_predictions()
                pending=[]
            yield block,_native_predictions(model,states,block)
    if pending:yield from single_predictions()


def log_likelihood(model,states,blocks):
    total=model.scales.new_zeros(())
    for block,predictions in _prediction_groups(model,states,blocks):
        if block[0]["kind"]=="optical":
            means=torch.stack([(field*w[:,None]).sum(0) for field,w in predictions])
            y=model.scales.new_tensor([r["value"] for r in block])
            chol=torch.linalg.cholesky(stabilize(model.optical_covariance(block)))
            residual=y[:,None]-means
            solved=torch.linalg.solve_triangular(chol,residual,upper=False)
            total+=(-.5*(len(block)*math.log(2*math.pi)+2*torch.log(chol.diagonal()).sum()+solved.square().sum(0))).mean()
        elif "_error_block" in block[0]:
            total += concentration_block_log_likelihood(model, block, predictions)
        else:
            r=block[0];field,w=predictions[0]
            j=INDICATORS.index(r["variable"])
            sigma=model.scales[j]*(1e-4+F.softplus(model.error_raw[j]))
            mixture=r.get("mode")=="instantaneous_time_mixture"
            mu=field if mixture else (field*w[:,None]).sum(0,keepdim=True)
            if "interval" in r:
                lower,upper=r["interval"]
                lo=(lower-mu)/sigma;hi=(upper-mu)/sigma
                # Evaluate in the nearer tail to avoid subtraction of two ones.
                log_big=torch.where(lo>0,torch.special.log_ndtr(-lo),torch.special.log_ndtr(hi))
                log_small=torch.where(lo>0,torch.special.log_ndtr(-hi),torch.special.log_ndtr(lo))
                lp=log_big+torch.log(-torch.expm1((log_small-log_big).clamp_max(-1e-15)))
            else:
                lp=-.5*math.log(2*math.pi)-torch.log(sigma)-.5*((r["value"]-mu)/sigma)**2
            total+=(torch.logsumexp(lp+torch.log(w.clamp_min(1e-300))[:,None],0) if mixture else lp.squeeze(0)).mean()
    return total


def regularization(model,alpha_lambda=1e-3,alpha_gamma=1e-3,alpha2=1e-4):
    penalty=alpha_lambda*torch.linalg.vector_norm(model.lambda_,dim=1).sum()
    penalty+=alpha2/2*sum(p.square().sum() for p in model.network.parameters())
    penalty+=alpha2/2*(model.background[:,-2:].square().sum()+model.forcing.square().sum())
    if any(r["kind"]=="optical" for r in model.data.records):
        penalty+=alpha_gamma*torch.linalg.vector_norm(model.gamma,dim=1).sum()
        band_factor=torch.tril(model.opt_band_raw,-1)+torch.diag(1e-4+F.softplus(model.opt_band_raw.diagonal()))
        penalty+=alpha2/2*(model.omega.square().sum()+band_factor.square().sum()+model.opt_scene_v.square().sum()+F.softplus(model.opt_scene_raw).square().sum())
    return penalty
