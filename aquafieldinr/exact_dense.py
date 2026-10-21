"""Exact active marginal, structured precision and conditional completion."""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .model import stabilize


class ExactActivePosterior(nn.Module):
    """Active-only Gaussian family with exact Gaussian-prior completion.

    Active covariances are evaluated directly from the OU × water covariance;
    this is algebraically the Schur-complement marginal, without constructing
    the complete time-by-node precision. Cholesky is bounded by an explicit
    capacity check. Cross-process within-node precision edges are retained.
    """
    def __init__(self,model):
        super().__init__()
        if model.backend!="dense":raise ValueError("Exact active inference requires the dense model backend.")
        if len(model.active)>model.max_active_dimension:
            raise MemoryError('E_EXACT_DENSE_022')
        model.factor_raw=nn.Parameter(torch.eye(len(model.active),dtype=model.scales.dtype,device=model.scales.device))
        model.factor_mask=torch.tril(torch.ones(len(model.active),len(model.active),dtype=model.scales.dtype,device=model.scales.device))
        self.model=model
        self.cache={}
        with torch.no_grad():
            mean,cov=self.active_prior()
            precision=torch.cholesky_inverse(torch.linalg.cholesky(stabilize(cov)))
            L=torch.linalg.cholesky(stabilize(precision))
            model.eta.copy_(mean)
            model.factor_raw.copy_(torch.tril(L,-1)+torch.diag(torch.log(torch.expm1((L.diagonal()-1e-5).clamp_min(1e-8)))))
            # Symbolic envelope: prior marginal fill-in plus all within-node
            # connections across shared/private/optical coordinates.
            pattern=(precision.abs()>1e-12).cpu().numpy()
            labels=[]
            for idx in model.active.cpu().numpy():
                z=np.searchsorted(model.offsets,idx,side="right")-1
                ell,kind,q,n=model.blocks[z]
                local=idx-model.offsets[z]
                labels.append((ell,int(local//n),int(local%n)))
            for i,a in enumerate(labels):
                for j,b in enumerate(labels):
                    if a==b:pattern[i,j]=True
            # Cholesky elimination fill-in.
            for k in range(len(pattern)):
                neighbours=np.flatnonzero(pattern[k+1:,k])+k+1
                pattern[np.ix_(neighbours,neighbours)]=True
            model.factor_mask.copy_(torch.tensor(np.tril(pattern),dtype=model.scales.dtype,device=model.scales.device))

    def active_prior(self):
        model=self.model;active=model.active
        means=[];covariances=[]
        for z,(ell,kind,q,n) in enumerate(model.blocks):
            selected=active[(active>=model.offsets[z])&(active<model.offsets[z+1])]-int(model.offsets[z])
            ti=selected//n;ni=selected%n
            times=torch.as_tensor(model.data.times,device=active.device)
            dt=(times[ti,None]-times[None,ti]).abs()
            space=model.data.meshes[ell].covariance(model.log_kappa[ell])
            amp=torch.exp(2*model.log_amp[z]) if kind=="private" else 1.
            cov=amp*torch.exp(-dt/torch.exp(model.log_tau[z]))*space[ni[:,None],ni[None,:]]
            means.append(model.temporal_mean(model.data.times,z)[ti])
            covariances.append(cov)
        return torch.cat(means),torch.block_diag(*covariances)

    def sample_and_kl(self,count,generator=None):
        model=self.model
        mean,cov=self.active_prior()
        prior_chol=torch.linalg.cholesky(stabilize(cov))
        prior_precision=torch.cholesky_inverse(prior_chol)
        qcov=model.q_covariance()
        difference=model.eta-mean
        logdet_prior=2*torch.log(prior_chol.diagonal()).sum()
        logdet_q=-2*torch.log(model.q_factor().diagonal()).sum()
        kl=.5*((prior_precision*qcov.T).sum()+difference@prior_precision@difference-len(mean)+logdet_prior-logdet_q)
        active=model.active_samples(count,generator)


        full=torch.zeros(model.total_dimension,count,device=active.device)
        full[model.active]=active
        return full,kl

    @torch.no_grad()
    def complete(self,query_times,count,weather=None,seed=0):
        model=self.model
        key=(int(seed),int(count))
        if key not in self.cache:
            generator=torch.Generator(device=model.scales.device).manual_seed(seed)
            self.cache[key]={"active":model.active_samples(count,generator),"queries":{},"weather":None if weather is None else np.asarray(weather).copy()}
        saved=self.cache[key]
        if (weather is None)!=(saved["weather"] is None) or (weather is not None and not np.array_equal(weather,saved["weather"])):
            raise ValueError('E_EXACT_DENSE_092')
        original=model.data.times
        all_times=np.unique(np.concatenate((original,np.asarray(query_times,float))))
        completed=[]
        for z,(ell,kind,q,n) in enumerate(model.blocks):
            selected_mask=(model.active>=model.offsets[z])&(model.active<model.offsets[z+1])
            selected=model.active[selected_mask]-int(model.offsets[z])
            selected=np.asarray(selected.cpu());ti=selected//n;ni=selected%n
            known_t=list(original[ti]);known_n=list(ni)
            known_draws=saved["active"][selected_mask]
            cache=saved["queries"].setdefault(z,{})
            if cache:
                known_t.extend(k[0] for k in cache);known_n.extend(k[1] for k in cache)
                known_draws=torch.cat((known_draws,torch.stack(list(cache.values()))))
            desired=[(float(t),node) for t in all_times for node in range(n)]
            known_keys=set(zip(known_t,known_n))
            unknown=[p for p in desired if p not in known_keys]
            if len(unknown)+len(known_t)>model.max_active_dimension:
                raise MemoryError('E_EXACT_DENSE_110')
            spatial=model.data.meshes[ell].covariance(model.log_kappa[ell])
            amp=torch.exp(2*model.log_amp[z]) if kind=="private" else 1.
            def kernel(a_t,a_n,b_t,b_n):
                a_t=model.scales.new_tensor(a_t);b_t=model.scales.new_tensor(b_t);a_n=torch.tensor(a_n,dtype=torch.long,device=model.scales.device);b_n=torch.tensor(b_n,dtype=torch.long,device=model.scales.device)
                return amp*torch.exp(-(a_t[:,None]-b_t[None,:]).abs()/torch.exp(model.log_tau[z]))*spatial[a_n[:,None],b_n[None,:]]
            mean_times=np.unique(np.concatenate((all_times,np.asarray(known_t))))
            mean_values=model.temporal_mean(mean_times,z,weather)
            mean={float(t):mean_values[i] for i,t in enumerate(mean_times)}
            if unknown:
                ut,un=zip(*unknown)
                Kuu=kernel(ut,un,ut,un)
                mu=torch.stack([mean[t] for t in ut])[:,None]
                if len(known_t):
                    Koo=kernel(known_t,known_n,known_t,known_n)
                    Kuo=kernel(ut,un,known_t,known_n)
                    L=torch.linalg.cholesky(stabilize(Koo))
                    A=torch.cholesky_solve(Kuo.T,L).T
                    mu=mu+A@(known_draws-torch.stack([mean[t] for t in known_t])[:,None])
                    Kuu=Kuu-A@Kuo.T
                generator=torch.Generator(device=model.scales.device).manual_seed(seed+1009*(z+1)+len(cache))
                noise=torch.randn(len(unknown),count,generator=generator,device=model.scales.device,dtype=model.scales.dtype)
                values=mu+torch.linalg.cholesky(stabilize(Kuu))@noise
                for p,value in zip(unknown,values):cache[p]=value
            mapping={p:value for p,value in zip(zip(known_t,known_n),known_draws)}
            mapping.update(cache)
            completed.append(torch.stack([mapping[p] for p in desired]))
        return all_times,torch.cat(completed)
