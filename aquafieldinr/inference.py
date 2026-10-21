"""Structured Gaussian Markov inference and coherent OU completion."""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .model import stabilize


class MarkovPosterior(nn.Module):
    """Block-bidiagonal precision with cross-process innovations at every node.

    Spatial innovations use the water-constrained covariance. Source Cholesky
    blocks mix shared, concentration-private and optical-private coordinates.
    Storage is O(TNK + TK² + N²).
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.means = nn.ParameterList()
        self.raw_cholesky = nn.ParameterList()
        self.raw_rho = nn.ParameterList()
        self.layers=[]
        self.cache={}
        T=len(model.data.times)
        for ell,mesh in enumerate(model.data.meshes):
            blocks=[z for z,b in enumerate(model.blocks) if b[0]==ell]
            K=len(blocks);N=len(mesh.nodes)
            self.layers.append(blocks)
            self.means.append(nn.Parameter(model.scales.new_zeros((T,N,K))))
            raw=model.scales.new_zeros((T,K,K))
            dt=model.scales.new_tensor(np.diff(model.data.times))
            tau=torch.exp(model.log_tau[blocks].detach())
            rho=torch.exp(-dt[:,None]/tau[None,:])
            variance=torch.cat((model.scales.new_ones((1,K)),1-rho**2),0)
            raw.diagonal(dim1=-2,dim2=-1).copy_(torch.log(torch.expm1(torch.sqrt(variance))))
            self.raw_cholesky.append(nn.Parameter(raw))
            self.raw_rho.append(nn.Parameter(torch.atanh(rho.clamp(max=.999999))))

    def source_cholesky(self,ell):
        raw=self.raw_cholesky[ell]
        return torch.tril(raw,-1)+torch.diag_embed(1e-6+F.softplus(raw.diagonal(dim1=-2,dim2=-1)))

    def sample_and_kl(self,count,generator=None):
        model=self.model
        block_states={}
        kl=torch.zeros((),device=model.scales.device)
        for ell,blocks in enumerate(self.layers):
            mu=self.means[ell];T,N,K=mu.shape
            spatial=model.data.meshes[ell].covariance(model.log_kappa[ell])
            chol=torch.linalg.cholesky(stabilize(spatial))
            precision=torch.cholesky_inverse(chol)
            source=self.source_cholesky(ell)
            rho=torch.tanh(self.raw_rho[ell])
            noise=torch.randn(T,count,N,K,device=mu.device,generator=generator)
            innov=torch.einsum("nm,tsmk,tlk->tsnl",chol,noise,source)
            zero=[]
            value=innov[0]
            zero.append(value)
            for t in range(1,T):
                value=value*rho[t-1]+innov[t]
                zero.append(value)
            draws=torch.stack(zero)+mu[:,None,:,:]
            for k,z in enumerate(blocks):
                block_states[z]=draws[:,:,:,k].permute(0,2,1).reshape(T*N,count)
            # Analytic KL: Markov prior expected quadratic and Gaussian entropy.
            innovation_cov=source@source.transpose(-1,-2)
            marginal=innovation_cov[0]
            expected_log_prior=torch.zeros((),device=mu.device)
            logdet_space=2*torch.log(chol.diagonal()).sum()
            entropy=.5*(T*N*K*(1+math.log(2*math.pi))+T*K*logdet_space+2*N*torch.log(source.diagonal(dim1=-2,dim2=-1)).sum())
            forcing_curves={z:model.temporal_mean(model.data.times,z) for z in blocks}
            for t in range(T):
                next_marginal=innovation_cov[t] if t==0 else rho[t-1,:,None]*marginal*rho[t-1,None,:]+innovation_cov[t]
                for k,z in enumerate(blocks):
                    amp=torch.exp(2*model.log_amp[z]) if model.blocks[z][1]=="private" else torch.ones((),device=mu.device)
                    forcing=forcing_curves[z]
                    if t==0:
                        variance=amp
                        mean_residual=mu[t,:,k]-forcing[t]
                        residual_variance=next_marginal[k,k]
                    else:
                        prior_rho=torch.exp(-(model.data.times[t]-model.data.times[t-1])/torch.exp(model.log_tau[z]))
                        variance=amp*(1-prior_rho**2)
                        mean_residual=mu[t,:,k]-prior_rho*mu[t-1,:,k]-(forcing[t]-prior_rho*forcing[t-1])
                        residual_variance=next_marginal[k,k]+prior_rho**2*marginal[k,k]-2*prior_rho*rho[t-1,k]*marginal[k,k]
                    quadratic=(mean_residual@precision@mean_residual+N*residual_variance)/variance
                    expected_log_prior=expected_log_prior-.5*(N*math.log(2*math.pi)+N*torch.log(variance)+logdet_space+quadratic)
                marginal=next_marginal
            kl=kl-entropy-expected_log_prior
        return torch.cat([block_states[z] for z in range(len(model.blocks))]),kl

    @torch.no_grad()
    def complete(self,query_times,count,weather=None,seed=0):
        
        model=self.model
        generator=torch.Generator(device=model.scales.device).manual_seed(seed)
        key=(int(seed),int(count))
        if key not in self.cache:
            active,_=self.sample_and_kl(count,generator)
            self.cache[key]={"active":active,"queries":{},"generators":{},"weather":None if weather is None else np.asarray(weather).copy()}
        saved=self.cache[key]
        if (weather is None)!=(saved["weather"] is None) or (weather is not None and not np.array_equal(weather,saved["weather"])):
            raise ValueError('E_INFERENCE_108')
        active=saved["active"]
        original=np.asarray(model.data.times)
        new=np.unique(np.asarray(query_times,float))
        all_times=np.unique(np.concatenate((original,new)))
        result=[];offset=0
        for z,(ell,kind,q,N) in enumerate(model.blocks):
            if z not in saved["generators"]:
                saved["generators"][z]=torch.Generator(device=model.scales.device).manual_seed(seed+1009*(z+1))
            bridge_generator=saved["generators"][z]
            known={float(t):active[offset+i*N:offset+(i+1)*N] for i,t in enumerate(original)}
            known.update(saved["queries"].setdefault(z,{}))
            offset+=len(original)*N
            covariance=model.data.meshes[ell].covariance(model.log_kappa[ell])
            amp=torch.exp(model.log_amp[z]) if kind=="private" else torch.ones((),device=active.device)
            chol=torch.linalg.cholesky(stabilize(covariance))*amp
            tau=torch.exp(model.log_tau[z])
            mean_times=np.unique(np.concatenate((all_times,np.asarray(list(known)))))
            means=model.temporal_mean(mean_times,z,weather)
            mean_lookup={float(t):means[i] for i,t in enumerate(mean_times)}
            for t in new:
                if float(t) in known:continue
                earlier=[x for x in known if x<t]
                later=[x for x in known if x>t]
                if not earlier:
                    right=min(later)
                    b=torch.exp(-(right-t)/tau)
                    mean=mean_lookup[float(t)]+b*(known[right]-mean_lookup[right])
                    variance=1-b*b
                elif later:
                    left=max(earlier)
                    a=torch.exp(-(t-left)/tau)
                    centered_left=known[left]-mean_lookup[left]
                    right=min(later);b=torch.exp(-(right-t)/tau)
                    denom=1-a*a*b*b
                    wleft=a*(1-b*b)/denom
                    wright=b*(1-a*a)/denom
                    mean=mean_lookup[float(t)]+wleft*centered_left+wright*(known[right]-mean_lookup[right])
                    variance=(1-a*a)*(1-b*b)/denom
                else:
                    left=max(earlier)
                    a=torch.exp(-(t-left)/tau)
                    centered_left=known[left]-mean_lookup[left]
                    mean=mean_lookup[float(t)]+a*centered_left
                    variance=1-a*a
                noise=torch.randn(N,count,generator=bridge_generator,device=active.device)
                known[float(t)]=mean+torch.sqrt(variance.clamp_min(0))*(chol@noise)
                saved["queries"][z][float(t)]=known[float(t)]
            result.append(torch.cat([known[float(t)] for t in all_times]))
        return all_times,torch.cat(result)
