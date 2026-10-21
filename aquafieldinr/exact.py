"""Exact active Gaussian marginal with sparse direct solves and completion."""
import tempfile
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components, reverse_cuthill_mckee
import torch
from torch import nn
from torch.nn import functional as F
from .sparse_prior import prior_precision
from .sparse_linalg import (matrix,factor,check_storage,SchurSystem,exact_kl,
                           triangular_draw,covariance_draw)


def _symbolic_factor(q,active,labels):
    """Eliminate inactive graph and retain all within-node process connections."""
    system=SchurSystem(q,active);a=len(active)
    adjacency=[set() for _ in range(a)]
    def clique(nodes):
        nodes=np.asarray(nodes,dtype=int)
        check_storage(sum(len(x) for x in adjacency)+len(nodes)**2,'Exact active marginal fill')
        group=set(nodes.tolist())
        for i in nodes:adjacency[i].update(group-{int(i)})
    coo=system.qa.tocoo()
    for i,j in zip(coo.row,coo.col):
        if i!=j:adjacency[i].add(int(j))
    if len(system.inactive):
        count,component=connected_components(q[system.inactive][:,system.inactive],directed=False)
        qua=system.qua.tocsr()
        for c in range(count):clique(np.unique(qua[component==c].indices))
    groups={}
    for i,key in enumerate(labels):groups.setdefault(key,[]).append(i)
    for nodes in groups.values():clique(nodes)
    graph_rows=np.concatenate([np.full(len(v),i,dtype=int) for i,v in enumerate(adjacency)])
    graph_cols=np.concatenate([np.asarray(sorted(v),dtype=int) for v in adjacency])
    graph=sp.csr_matrix((np.ones(len(graph_rows)),(graph_rows,graph_cols)),shape=(a,a))
    permutation=reverse_cuthill_mckee(graph,symmetric_mode=True).copy()
    inverse=np.empty(a,dtype=int);inverse[permutation]=np.arange(a)
    adjacency=[set(inverse[list(adjacency[i])].tolist()) for i in permutation]
    system=SchurSystem(q,np.asarray(active)[permutation])
    rows=[];cols=[];size=0
    for k in range(a):
        neighbours=sorted(i for i in adjacency[k] if i>k)
        rows.extend([k]+neighbours);cols.extend([k]*(1+len(neighbours)))
        size+=1+len(neighbours);check_storage(size,'Variational precision Cholesky fill',multiplier=160)
        higher=set(neighbours)
        for i in neighbours:
            adjacency[i].discard(k);adjacency[i].update(higher-{i})
        adjacency[k].clear()
    return np.asarray(rows),np.asarray(cols),system,permutation


def _initial_factor(system,rows,cols):
    """Assemble only structural Schur entries, in bounded right-hand-side batches."""
    a=len(system.active);values=np.zeros(len(rows))
    for k in range(0,a,16):
        ids=np.arange(k,min(k+16,a));rhs=np.zeros((a,len(ids)));rhs[ids,np.arange(len(ids))]=1
        y=system.apply(rhs);mask=(cols>=ids[0])&(cols<=ids[-1])
        values[mask]=y[rows[mask],cols[mask]-ids[0]]
    p=sp.csc_matrix((values,(rows,cols)),shape=(a,a));p=p+p.T-sp.diags(p.diagonal())
    lu=factor(p,natural=True)
    if not np.array_equal(lu.perm_r,np.arange(a)) or not np.array_equal(lu.perm_c,np.arange(a)):
        raise np.linalg.LinAlgError('Active SPD initialization unexpectedly pivoted.')
    lower=(lu.L@sp.diags(np.sqrt(lu.U.diagonal()))).tocsr()
    return np.asarray(lower[rows,cols]).ravel()


class ExactActivePosterior(nn.Module):
    """The SI active marginal family; no Markov variational restriction.

    Sparse direct factors and Schur applications replace global dense inverses.
    Deterministic trace solves retain the exact Gaussian KL.  Intrinsic fill-in
    is admitted only when the configurable sparse memory budget can hold it.
    """
    def __init__(self,model):
        super().__init__()
        if model.backend not in ('sparse','exact_active'):
            raise ValueError('Exact sparse posterior requires the sparse model backend.')
        self.model=model;self.cache={};self._temporary=tempfile.TemporaryDirectory(prefix='aquafield_states_',ignore_cleanup_errors=True)
        object.__setattr__(model,'_exact_posterior',self)
        with torch.no_grad():
            qr,qc,qv,n=prior_precision(model)
            active=model.active.cpu().numpy()
            if not len(active):raise ValueError('E_EXACT_084')
            labels=[]
            for idx in active:
                z=np.searchsorted(model.offsets,idx,side='right')-1
                ell,kind,k,nodes=model.blocks[z];local=idx-model.offsets[z]
                labels.append((ell,int(local//nodes),int(local%nodes)))
            rows,cols,system,permutation=_symbolic_factor(matrix(qv.cpu().numpy(),qr,qc,n),active,labels)
            initial=_initial_factor(system,rows,cols)
            raw=initial.copy();diagonal=rows==cols
            positive=np.maximum(initial[diagonal]-1e-5,1e-12)
            raw[diagonal]=positive+np.log(-np.expm1(-positive))
            model.factor_raw=nn.Parameter(torch.as_tensor(raw,dtype=model.scales.dtype,device=model.scales.device))
            model.register_buffer('factor_rows',torch.as_tensor(rows,dtype=torch.long,device=model.scales.device))
            model.register_buffer('factor_cols',torch.as_tensor(cols,dtype=torch.long,device=model.scales.device))
            model.register_buffer('factor_permutation',torch.as_tensor(permutation,dtype=torch.long,device=model.scales.device))
            self.rows=rows;self.cols=cols
            self.permutation=permutation;self.inverse_permutation=np.argsort(permutation)
            model.eta.copy_(self._active_mean())

    def variational_parameters(self):return (self.model.eta,self.model.factor_raw)

    def _active_mean(self):
        model=self.model;means=[]
        for z,(_,_,_,n) in enumerate(model.blocks):
            selected=model.active[(model.active>=model.offsets[z])&(model.active<model.offsets[z+1])]-int(model.offsets[z])
            means.append(model.temporal_mean(model.data.times,z)[selected//n])
        return torch.cat(means)

    def factor_values(self):
        raw=self.model.factor_raw;diag=self.model.factor_rows==self.model.factor_cols
        return torch.where(diag,1e-5+F.softplus(raw),raw)

    def q_factor(self):
        indices=torch.stack((self.model.factor_rows,self.model.factor_cols))
        return torch.sparse_coo_tensor(indices,self.factor_values(),(len(self.model.active),len(self.model.active)),check_invariants=True).coalesce()

    def active_samples(self,count,generator=None):
        m=self.model;noise=torch.randn(len(m.active),count,generator=generator,device=m.scales.device,dtype=m.scales.dtype)
        return m.eta[:,None]+triangular_draw(self.factor_values(),noise,self.rows,self.cols,len(m.active))[self.inverse_permutation]

    def active_prior(self):
        
        m=self.model;a=len(m.active);check_storage(a*a,'Requested explicit active covariance',multiplier=16)
        qr,qc,qv,n=prior_precision(m);lu=factor(matrix(qv.detach().cpu().numpy(),qr,qc,n));active=m.active.cpu().numpy()
        cov=np.empty((a,a))
        for k in range(0,a,16):
            ids=active[k:k+16];rhs=np.zeros((n,len(ids)));rhs[ids,np.arange(len(ids))]=1
            cov[:,k:k+len(ids)]=lu.solve(rhs)[active]
        return self._active_mean(),torch.as_tensor(cov,dtype=m.scales.dtype,device=m.scales.device)

    def sample_and_kl(self,count,generator=None):
        m=self.model;qr,qc,qv,n=prior_precision(m)
        kl=exact_kl(qv,self.factor_values(),(m.eta-self._active_mean())[self.permutation],qr,qc,self.rows,self.cols,m.active.cpu().numpy()[self.permutation],n)
        active=self.active_samples(count,generator)
        full=torch.zeros((m.total_dimension,count),device=active.device,dtype=active.dtype)
        return full.index_copy(0,m.active,active),kl

    def close_completion(self,reopen=True):
        
        for saved in self.cache.values():
            for mapped in saved.get('maps',[]):
                mapped.flush();mapped._mmap.close()
        self.cache.clear()
        if self._temporary is not None:self._temporary.cleanup()
        self._temporary=tempfile.TemporaryDirectory(prefix='aquafield_states_',ignore_cleanup_errors=True) if reopen else None

    def close(self):
        self.close_completion(reopen=False)

    @torch.no_grad()
    def complete(self,query_times,count,weather=None,seed=0):
        
        if self._temporary is None:
            self._temporary=tempfile.TemporaryDirectory(prefix='aquafield_states_',ignore_cleanup_errors=True)
        m=self.model;key=(int(seed),int(count));query_times=np.asarray(query_times,float)
        if key not in self.cache:
            gen=torch.Generator(device=m.scales.device).manual_seed(seed)
            self.cache[key]={'active':self.active_samples(count,gen).cpu().numpy(),'times':np.empty(0),'blocks':None,
                             'weather':None if weather is None else np.asarray(weather).copy(),'maps':[]}
        saved=self.cache[key]
        if (weather is None)!=(saved['weather'] is None) or (weather is not None and not np.array_equal(weather,saved['weather'])):
            raise ValueError('E_EXACT_170')
        times=np.unique(np.concatenate((m.data.times,query_times,saved['times'])))
        if saved['blocks'] is not None and np.array_equal(times,saved['times']):return times,saved['tensor']
        dimension=len(times)*sum(b[3] for b in m.blocks)
        path=Path(self._temporary.name)/f'{seed}_{count}_{len(times)}_{len(list(Path(self._temporary.name).iterdir()))}.dat'
        result=np.memmap(path,dtype=np.float64,mode='w+',shape=(dimension,count))
        qr,qc,qv,nfull=prior_precision(m,times)
        qfull=matrix(qv.cpu().numpy(),qr,qc,nfull)
        blocks=[];offset=0
        for z,(_,_,_,n) in enumerate(m.blocks):
            nblock=len(times)*n;precision=qfull[offset:offset+nblock,offset:offset+nblock]
            selected_mask=(m.active>=m.offsets[z])&(m.active<m.offsets[z+1])
            local=(m.active[selected_mask]-int(m.offsets[z])).cpu().numpy()
            known=np.searchsorted(times,m.data.times[local//n])*n+local%n
            source_draws=[saved['active'][selected_mask.cpu().numpy()]]
            if saved['blocks'] is not None:
                cached=(np.searchsorted(times,saved['times'])[:,None]*n+np.arange(n)[None,:]).ravel()
                cached_draws=saved['blocks'][z]

                keep=~np.isin(known,cached);known=np.concatenate((known[keep],cached));source_draws=[source_draws[0][keep],cached_draws]
            order=np.argsort(known);known=known[order]
            unknown=np.setdiff1d(np.arange(nblock),known,assume_unique=True)
            mean=m.temporal_mean(times,z,weather).cpu().numpy().repeat(n)
            target=result[offset:offset+nblock]
            if len(unknown):
                quu=precision[unknown][:,unknown].tocsc();qua=precision[unknown][:,known]
                lu=factor(quu)

            rng=np.random.default_rng(np.random.SeedSequence([int(seed),z,len(saved['times']),len(times)]))
            for start in range(0,count,16):
                end=min(start+16,count)
                observed=np.concatenate([source[:,start:end] for source in source_draws],axis=0)[order]
                target[known,start:end]=observed
                if len(unknown):
                    center=-lu.solve(qua@(observed-mean[known,None]))+mean[unknown,None]
                    target[unknown,start:end]=center+covariance_draw(lu,end-start,rng)
            blocks.append(target);offset+=nblock
        result.flush();tensor=torch.from_numpy(result)
        saved['maps'].append(result)
        saved.update({'times':times,'blocks':blocks,'tensor':tensor,'memmap':result})
        return times,tensor
