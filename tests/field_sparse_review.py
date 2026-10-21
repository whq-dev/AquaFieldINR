import sys,unittest
from pathlib import Path
import numpy as np
import torch
from scipy import sparse
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from aquafieldinr.data import Dataset
from aquafieldinr.model import AquaFieldINR
from aquafieldinr.exact import ExactActivePosterior
from aquafieldinr.sparse_prior import prior_precision
from aquafieldinr.sparse_linalg import matrix, factor, covariance_draw, exact_kl,triangular_draw
ROOT=Path(__file__).resolve().parents[1]
torch.set_num_threads(1)
class IndependentSparse(unittest.TestCase):
    def test_dense_prior_and_kl_gradients(self):
        torch.manual_seed(82);m=AquaFieldINR(Dataset(ROOT/'validation_case/observations.json'),backend='sparse');p=ExactActivePosterior(m)
        with torch.no_grad():m.factor_raw.add_(torch.randn_like(m.factor_raw)*.03);m.eta.add_(torch.randn_like(m.eta)*.05);m.forcing.normal_(0,.02)
        qr,qc,qv,n=prior_precision(m);Q=torch.sparse_coo_tensor(torch.tensor(np.stack((qr,qc))),qv,(n,n)).to_dense()
        C=Q.inverse();active=m.active
        oldmean,oldC=m.prior()
        torch.testing.assert_close(C,oldC,rtol=5e-10,atol=5e-10)
        Ca=C[active][:,active];P=Ca.inverse();L=p.q_factor().to_dense();cov=(L@L.T).inverse()
        delta=(m.eta-p._active_mean())[p.permutation]
        Pa=P[p.permutation][:,p.permutation]
        dense=.5*((Pa*cov.T).sum()+delta@Pa@delta-len(active)+torch.logdet(Ca)+2*L.diag().log().sum())
        _,actual=p.sample_and_kl(3)
        torch.testing.assert_close(actual,dense,rtol=5e-9,atol=5e-9)
        pars=[m.log_kappa,m.log_tau,m.log_amp,m.eta,m.factor_raw,m.forcing]
        ga=torch.autograd.grad(actual,pars,retain_graph=True);gd=torch.autograd.grad(dense,pars)
        for a,b in zip(ga,gd):torch.testing.assert_close(a,b,rtol=2e-7,atol=2e-8)
        p.close()
    def test_covariance_permutation(self):
        rng=np.random.default_rng(44);B=rng.normal(size=(9,9));Q=B@B.T+np.eye(9)
        lu=factor(sparse.csc_matrix(Q));draw=covariance_draw(lu,80000,rng)
        C=np.linalg.inv(Q);np.testing.assert_allclose(draw@draw.T/draw.shape[1],C,rtol=.12,atol=.012)
    def test_sparse_triangular_gradient(self):
        torch.manual_seed(9);L=torch.tril(torch.randn(6,6));L.diagonal().add_(4);rows,cols=np.tril_indices(6)
        values=L[rows,cols].clone().requires_grad_();noise=torch.randn(6,4,requires_grad=True)
        sparseout=triangular_draw(values,noise,rows,cols,6)
        densemat=torch.zeros(6,6);densemat[rows,cols]=values
        denseout=torch.linalg.solve_triangular(densemat.T,noise,upper=True)
        torch.testing.assert_close(sparseout,denseout)
        ga=torch.autograd.grad(sparseout.square().sum(),[values,noise],retain_graph=True);gd=torch.autograd.grad(denseout.square().sum(),[values,noise])
        for a,b in zip(ga,gd):torch.testing.assert_close(a,b,rtol=1e-10,atol=1e-10)
if __name__=='__main__':unittest.main(verbosity=2)
