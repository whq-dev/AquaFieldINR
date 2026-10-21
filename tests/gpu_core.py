
import sys
import unittest
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from aquafieldinr.data import Dataset
from aquafieldinr.model import AquaFieldINR
from aquafieldinr.exact import ExactActivePosterior
from aquafieldinr.likelihood import groups,log_likelihood
from aquafieldinr.sparse_linalg import triangular_draw
from aquafieldinr.sparse_prior import prior_precision
from aquafieldinr.canonical import canonical_representation
ROOT=Path(__file__).resolve().parents[1]
torch.set_num_threads(1)

class CoreDevice(unittest.TestCase):
    def make(self):
        torch.manual_seed(51)
        model=AquaFieldINR(Dataset(ROOT/'validation_case/observations.json'),backend='sparse')
        posterior=ExactActivePosterior(model)
        with torch.no_grad():
            model.factor_raw.add_(torch.randn_like(model.factor_raw)*.01)
            model.lambda_.add_(.1);model.gamma.add_(.1)
            model.forcing.normal_(0,.005)
        return model,posterior

    def compare(self,device):
        cpu,cp=self.make();other,op=self.make();op.to(device)
        rng=torch.Generator().manual_seed(72)
        source=torch.randn(cpu.total_dimension,3,generator=rng)*.15
        a=source.clone().requires_grad_();b=source.clone().requires_grad_()

        la=log_likelihood(cpu,a,groups(cpu.data.records))
        lb=log_likelihood(other,b,groups(other.data.records))
        self.assertEqual(lb.device.type,torch.device(device).type)
        pa=[a,*cpu.parameters()];pb=[b,*other.parameters()]
        ga=torch.autograd.grad(la,pa,allow_unused=True)
        gb=torch.autograd.grad(lb,pb,allow_unused=True)
        torch.testing.assert_close(la,lb.cpu(),rtol=1e-8,atol=1e-7)
        for x,y in zip(ga,gb):
            self.assertEqual(x is None,y is None)
            if x is not None:torch.testing.assert_close(x,y.cpu(),rtol=2e-7,atol=1e-7)
        _,ka=cp.sample_and_kl(2);_,kb=op.sample_and_kl(2)
        paramsa=[cpu.log_kappa,cpu.log_tau,cpu.log_amp,cpu.factor_raw,cpu.eta,cpu.forcing]
        paramsb=[other.log_kappa,other.log_tau,other.log_amp,other.factor_raw,other.eta,other.forcing]
        ga=torch.autograd.grad(ka,paramsa);gb=torch.autograd.grad(kb,paramsb)
        torch.testing.assert_close(ka,kb.cpu(),rtol=1e-8,atol=1e-8)
        for x,y in zip(ga,gb):torch.testing.assert_close(x,y.cpu(),rtol=2e-7,atol=1e-8)
        ca=canonical_representation(cpu);cb=canonical_representation(other)
        self.assertEqual(ca['version'],cb['version'])
        for x,y in zip(ca['blocks'],cb['blocks']):
            self.assertAlmostEqual(x.pop('time_scale'),y.pop('time_scale'),places=11)
            self.assertEqual(x,y)
        times,archive=op.complete([.5,1.5],3,seed=23)
        self.assertEqual(archive.device.type,'cpu')
        values=other.field(archive,times,np.array([[.2,.2],[.3,.3]]),np.array([.5,1.5]),'concentration','TN',chunk_size=1)
        self.assertEqual(values.device.type,torch.device(device).type)
        self.assertTrue(torch.isfinite(values).all())

        del values,archive
        cp.close();op.close()

    def test_cpu(self):self.compare('cpu')

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA device is unavailable')
    def test_correlated_censored_cuda(self):
        from aquafieldinr.correlated import gaussian_observation_log_probability


        record_sets=[
            [{'value':.2},{'interval':[-.4,1.2]},{'interval':[-float('inf'),.8]}],
            [{'interval':[-.5,.8]},{'interval':[-.4,1.2]},{'interval':[-float('inf'),.8]}],
        ]
        for records in record_sets:
            mean=torch.tensor([[.1,.2,-.1],[.3,-.2,.4],[.1,.5,.2]],requires_grad=True)
            covariance=torch.tensor([[1.,.25,-.1],[.25,.8,.15],[-.1,.15,1.2]],requires_grad=True)
            gm=mean.detach().cuda().requires_grad_();gc=covariance.detach().cuda().requires_grad_()
            a=gaussian_observation_log_probability(mean,covariance,records,points=256)
            b=gaussian_observation_log_probability(gm,gc,records,points=256)
            torch.testing.assert_close(a,b.cpu(),rtol=1e-9,atol=1e-9)
            ga=torch.autograd.grad(a.sum(),[mean,covariance]);gb=torch.autograd.grad(b.sum(),[gm,gc])
            for x,y in zip(ga,gb):
                self.assertTrue(torch.isfinite(y).all())
                torch.testing.assert_close(x,y.cpu(),rtol=2e-7,atol=1e-8)


    @unittest.skipUnless(torch.cuda.is_available(),'CUDA device is unavailable')
    def test_cuda(self):self.compare('cuda:0')

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA device is unavailable')
    def test_other_backends_cuda(self):
        from aquafieldinr.inference import MarkovPosterior
        from aquafieldinr.exact_dense import ExactActivePosterior as DensePosterior
        for backend,cls in [('markov',MarkovPosterior),('dense',DensePosterior)]:
            torch.manual_seed(16)
            model=AquaFieldINR(Dataset(ROOT/'validation_case/observations.json'),backend=backend)
            posterior=cls(model).to('cuda:0')
            states,kl=posterior.sample_and_kl(2)
            objective=log_likelihood(model,states,groups(model.data.records))-kl
            objective.backward()
            self.assertEqual(objective.device.type,'cuda')
            self.assertTrue(torch.isfinite(objective))
            for parameter in posterior.parameters():
                if parameter.grad is not None:self.assertTrue(torch.isfinite(parameter.grad).all())
            times,completed=posterior.complete([.25,1.5],2,seed=54)
            self.assertTrue(torch.isfinite(completed).all())

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA device is unavailable')
    def test_sparse_custom_autograd_cuda(self):
        rows,cols=np.tril_indices(5)
        torch.manual_seed(8)
        dense=torch.tril(torch.randn(5,5));dense.diagonal().add_(5)
        va=dense[rows,cols].clone().requires_grad_();vb=va.detach().cuda().requires_grad_()
        na=torch.randn(5,3,requires_grad=True);nb=na.detach().cuda().requires_grad_()
        a=triangular_draw(va,na,rows,cols,5);b=triangular_draw(vb,nb,rows,cols,5)
        torch.testing.assert_close(a,b.cpu(),rtol=1e-11,atol=1e-11)
        ga=torch.autograd.grad(a.square().sum(),[va,na]);gb=torch.autograd.grad(b.square().sum(),[vb,nb])
        for x,y in zip(ga,gb):torch.testing.assert_close(x,y.cpu(),rtol=1e-11,atol=1e-11)

if __name__=='__main__':unittest.main(verbosity=2)
