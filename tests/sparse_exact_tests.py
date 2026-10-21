
import json,sys,tempfile,time
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from aquafieldinr.sparse_linalg import exact_kl,factor,covariance_draw
from aquafieldinr.sparse_prior import prior_precision
from aquafieldinr.data import Dataset
from aquafieldinr.model import AquaFieldINR
from aquafieldinr.exact import ExactActivePosterior

torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
def run():
    torch.manual_seed(13);rng=np.random.default_rng(13);report={}
    n=9;a=4;b=rng.normal(size=(n,n));q=b@b.T+np.eye(n);active=np.array([0,2,5,8]);l=np.tril(rng.normal(size=(a,a)));l[np.diag_indices(a)]=2
    qr,qc=np.indices((n,n));qr=qr.ravel();qc=qc.ravel();lr,lc=np.tril_indices(a)
    qv=torch.tensor(q[qr,qc],requires_grad=True);lv=torch.tensor(l[lr,lc],requires_grad=True);d=torch.randn(a,requires_grad=True)
    k=exact_kl(qv,lv,d,qr,qc,lr,lc,active,n);k.backward()
    Q=qv.detach().reshape(n,n).requires_grad_();L=torch.zeros((a,a));L[lr,lc]=lv.detach();L.requires_grad_();D=d.detach().clone().requires_grad_()
    cov=torch.linalg.inv(Q)[active][:,active];P=torch.linalg.inv(cov);S=torch.linalg.inv(L@L.T)
    ref=.5*(torch.trace(P@S)+D@P@D-a+torch.linalg.slogdet(cov)[1]+2*torch.log(L.diagonal()).sum());ref.backward()
    errors=[abs(float(k.detach()-ref.detach())),np.max(abs(qv.grad.numpy()-Q.grad.numpy().ravel())),np.max(abs(lv.grad.numpy()-L.grad.numpy()[lr,lc])),np.max(abs(d.grad.numpy()-D.grad.numpy()))]
    assert max(errors)<1e-10;report['generic_kl_and_gradient_max_error']=float(max(errors))
    data=Dataset(ROOT/'validation_case/observations.json');model=AquaFieldINR(data,backend='sparse')
    with torch.no_grad():model.log_kappa.fill_(-.3)
    posterior=ExactActivePosterior(model)
    with torch.no_grad():model.eta.add_(torch.randn_like(model.eta)*.03);model.factor_raw.add_(torch.randn_like(model.factor_raw)*.01)
    _,sparsekl=posterior.sample_and_kl(2)
    names=['log_kappa','log_tau','log_amp','eta','factor_raw'];parameters=[getattr(model,x) for x in names]
    grads=torch.autograd.grad(sparsekl,parameters,retain_graph=True)
    mean,fullcov=model.prior();ac=model.active;cov=fullcov[ac][:,ac];P=torch.linalg.inv(cov)
    Lv=posterior.q_factor().to_dense();ip=posterior.inverse_permutation;S=torch.linalg.inv(Lv@Lv.T)[ip][:,ip];delta=model.eta-mean[ac]
    reference=.5*(torch.trace(P@S)+delta@P@delta-len(ac)+torch.linalg.slogdet(cov)[1]+2*torch.log(Lv.diagonal()).sum())
    densegrads=torch.autograd.grad(reference,parameters)
    errors={name:float((x-y).abs().max()) for name,x,y in zip(names,grads,densegrads)}
    assert abs(float((sparsekl-reference).detach()))<1e-7 and max(errors.values())<1e-6,(float((sparsekl-reference).detach()),errors)
    report['model_kl_error']=abs(float((sparsekl-reference).detach()));report['model_gradient_errors']=errors
    count=20000;times,states=posterior.complete([1.5],count,seed=42)

    original_n=sum(b[3] for b in model.blocks)*len(times)
    _,C=model.prior(times);C=C.detach().numpy();active_new=[]
    for z,(_,_,_,nodes) in enumerate(model.blocks):
        mask=(ac>=model.offsets[z])&(ac<model.offsets[z+1]);local=(ac[mask]-model.offsets[z]).numpy()
        offset=sum(b[3]*len(times) for b in model.blocks[:z])
        active_new.extend(offset+np.searchsorted(times,data.times[local//nodes])*nodes+local%nodes)
    active_new=np.asarray(active_new);unknown=np.setdiff1d(np.arange(len(C)),active_new);wanted=unknown[:3]
    cross=C[wanted][:,active_new];conditional=cross@np.linalg.inv(C[active_new][:,active_new]);expected=C[wanted][:,wanted]-conditional@cross.T
    fullmean=np.concatenate([model.temporal_mean(times,z).detach().numpy().repeat(b[3]) for z,b in enumerate(model.blocks)])
    avalues=states[active_new].numpy();residual=states[wanted].numpy()-fullmean[wanted,None]-conditional@(avalues-fullmean[active_new,None])
    empirical=np.cov(residual);err=float(np.max(abs(empirical-expected))/np.max(abs(expected)))
    assert err<.05,err;report['conditional_covariance_relative_error']=err
    old=states.clone();oldtimes=times.copy();times2,states2=posterior.complete([2.5],count,seed=42)
    offold=0;offnew=0;error=0.
    for b in model.blocks:
        nodes=b[3];idx=(np.searchsorted(times2,oldtimes)[:,None]*nodes+np.arange(nodes)[None,:]).ravel()+offnew
        error=max(error,float((old[offold:offold+len(oldtimes)*nodes]-states2[idx]).abs().max()));offold+=len(oldtimes)*nodes;offnew+=len(times2)*nodes
    assert error==0.;report['cached_realization_change']=error
    del states,states2;posterior.close()
    return report
if __name__=='__main__':
    report=run();(ROOT/'tests/sparse_exact_results.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
