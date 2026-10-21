
import json,sys,tempfile,time
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from aquafieldinr.data import Dataset
from aquafieldinr.model import AquaFieldINR
from aquafieldinr.exact import ExactActivePosterior
from aquafieldinr.sparse_prior import prior_precision

torch.set_num_threads(1)
doc=json.loads((ROOT/'validation_case/observations.json').read_text())
times=600;doc['weather']=[[t,15,2,120,3] for t in range(times)]
doc['records']=[{'kind':'concentration','variable':v,'value':x*(1+.1*np.sin(t/30)),'position_id':'p','support':[[.2,.2,t,1.]]} for t in range(times) for v,x in zip(('TN','TP','Chla'),(1.,.1,10.))]
with tempfile.TemporaryDirectory() as tmp:
    path=Path(tmp)/'input.json';path.write_text(json.dumps(doc));data=Dataset(path)
    start=time.perf_counter();model=AquaFieldINR(data,shared_rank=0,backend='sparse')
    with torch.no_grad():model.log_kappa.fill_(-.3)
    posterior=ExactActivePosterior(model);init=time.perf_counter()-start
    n=len(model.active);assert n==7200 and model.factor_raw.ndim==1
    qr,qc,qv,full=prior_precision(model)
    assert all(p.numel()<n*n for p in model.parameters())
    start=time.perf_counter();states,kl=posterior.sample_and_kl(2);forward=time.perf_counter()-start
    start=time.perf_counter();kl.backward();backward=time.perf_counter()-start
    assert abs(float(kl.detach()))<1e-6
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    report={'active_dimension':n,'full_dimension':full,'q_factor_nonzeros':model.factor_raw.numel(),'prior_nonzeros':qv.numel(),'initialization_seconds':init,'forward_seconds':forward,'backward_seconds':backward,'kl':float(kl.detach()),'largest_parameter_elements':max(p.numel() for p in model.parameters()),'no_global_dense_factor':True}
    posterior.close();(ROOT/'tests/sparse_large_results.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
