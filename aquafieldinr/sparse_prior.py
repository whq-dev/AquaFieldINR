"""Sparse OU × finite-element precision, without global covariance assembly."""
import numpy as np
import scipy.sparse as sp
import torch
from .sparse_linalg import inverse_mean


def spatial_precision(mesh, log_kappa):
    n=len(mesh.nodes)
    if not hasattr(mesh,'mass'):
        ids=np.arange(n);return ids,ids,torch.ones(n,dtype=log_kappa.dtype,device=log_kappa.device)
    lap=sp.coo_matrix(mesh.laplacian)
    # Include every diagonal even where the stiffness diagonal is zero.
    skeleton=(lap+sp.eye(n,format='coo')).tocoo()
    rows=skeleton.row;cols=skeleton.col
    base=np.asarray(sp.csr_matrix(lap)[rows,cols]).ravel()
    values=torch.as_tensor(base,dtype=log_kappa.dtype,device=log_kappa.device)
    diag=rows==cols
    addition=torch.zeros_like(values)
    addition[torch.as_tensor(diag,device=values.device)]=torch.as_tensor(np.asarray(mesh.mass)[rows[diag]],dtype=values.dtype,device=values.device)*torch.exp(2*log_kappa)
    values=values+addition
    scale=inverse_mean(values,rows,cols,n)
    return rows,cols,values*scale


def temporal_precision(times, log_tau):
    times=np.asarray(times,float);t=len(times)
    if t==1:return np.array([0]),np.array([0]),log_tau.new_ones(1)
    dt=torch.as_tensor(np.diff(times),dtype=log_tau.dtype,device=log_tau.device)
    rho=torch.exp(-dt/torch.exp(log_tau));inv=1/(-torch.expm1(-2*dt/torch.exp(log_tau)))
    diagonal=torch.cat((inv[:1],inv[:-1]+rho[1:]**2*inv[1:],inv[-1:]))
    off=-rho*inv;ids=np.arange(t);edge=np.arange(t-1)
    return np.concatenate((ids,edge,edge+1)),np.concatenate((ids,edge+1,edge)),torch.cat((diagonal,off,off))


def prior_precision(model,times=None):
    times=model.data.times if times is None else np.asarray(times,float)
    rows=[];cols=[];values=[];offset=0;spatial={}
    for z,(ell,kind,q,n) in enumerate(model.blocks):
        if ell not in spatial:spatial[ell]=spatial_precision(model.data.meshes[ell],model.log_kappa[ell])
        sr,sc,sv=spatial[ell];tr,tc,tv=temporal_precision(times,model.log_tau[z])
        rows.append((tr[:,None]*n+sr[None,:]+offset).ravel())
        cols.append((tc[:,None]*n+sc[None,:]+offset).ravel())
        amplitude=torch.exp(-2*model.log_amp[z]) if kind=='private' else 1.
        values.append((tv[:,None]*sv[None,:]*amplitude).ravel())
        offset+=len(times)*n
    return np.concatenate(rows),np.concatenate(cols),torch.cat(values),offset
