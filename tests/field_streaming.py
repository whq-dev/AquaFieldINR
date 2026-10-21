import sys, json, tempfile, unittest
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from aquafieldinr.data import Dataset,INDICATORS,BANDS
from aquafieldinr.model import AquaFieldINR
from aquafieldinr.geometry import Mesh
from aquafieldinr import cli
ROOT=Path(__file__).resolve().parents[1]
torch.set_num_threads(1)


def dense_field(model,states,times,points,qt,kind,variable):
    ti=np.searchsorted(times,qt)
    if kind=='concentration':
        j=INDICATORS.index(variable);result=(model.design(qt)@model.background[j])[:,None].expand(-1,states.shape[1]).clone()
    else:
        j=BANDS.index(variable);result=model.beta_opt[j].expand(len(points),states.shape[1]).clone()
    offset=0
    for z,(ell,bkind,q,n) in enumerate(model.blocks):
        a=states[offset:offset+len(times)*n].reshape(len(times),n,-1)[ti];offset+=len(times)*n
        if kind=='concentration':weight=model.lambda_[ell,j,q] if bkind=='shared' else (1. if bkind=='private' and q==j else 0.)
        else:weight=(model.gamma[ell,j,q] if bkind=='shared' else (model.omega[ell,j,q] if bkind=='optical' else 0.))*model.optical_scales[j]
        result=result+weight*torch.einsum('pn,pns->ps',model.basis(ell,points),a)
    return model.scales[j]*torch.nn.functional.softplus(result) if kind=='concentration' else result


def optical_oracle(model,records):
    band=torch.tensor([BANDS.index(r['variable']) for r in records]);F=torch.nn.functional
    B=torch.tril(model.opt_band_raw,-1)+torch.diag(1e-4+F.softplus(model.opt_band_raw.diagonal()))
    kb=B@B.T;ks=torch.outer(model.opt_scene_v,model.opt_scene_v)+torch.diag(F.softplus(model.opt_scene_raw)**2)
    rows=[]
    for a in records:
        aa=np.asarray(a['support']);row=[]
        for b in records:
            bb=np.asarray(b['support']);dist=torch.tensor(np.linalg.norm(aa[:,None,:2]-bb[None,:,:2],axis=2));w=torch.tensor(aa[:,3,None]*bb[None,:,3])
            row.append((torch.exp(-dist/torch.exp(model.log_error_range))*w).sum())
        rows.append(torch.stack(row))
    scales=model.optical_scales[band]
    return torch.outer(scales,scales)*(kb[band[:,None],band[None,:]]*torch.stack(rows)+ks[band[:,None],band[None,:]]+torch.diag((1e-3+F.softplus(model.opt_error[band]))**2))


class FieldTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(77)
        self.model=AquaFieldINR(Dataset(ROOT/'validation_case/observations.json'),backend='markov')
        with torch.no_grad():self.model.network[-2].weight.normal_(0,.1)
        self.times=self.model.data.times
        self.states=torch.randn(self.model.total_dimension,7)
    def test_field_values_gradients(self):
        model=self.model;points=np.array([[.1,.1],[.2,.2],[.4,.2],[.3,.3],[.05,.8]])
        qt=np.resize(self.times,len(points))
        for kind,variables in [('concentration',INDICATORS),('optical',BANDS)]:
            for variable in variables:
                new=model.field(self.states,self.times,points,qt,kind,variable,chunk_size=2)
                old=dense_field(model,self.states,self.times,points,qt,kind,variable)
                torch.testing.assert_close(new,old,rtol=1e-12,atol=1e-12)
                pars=[model.network[0].weight,model.background if kind=='concentration' else model.beta_opt, model.lambda_ if kind=='concentration' else model.gamma]
                g1=torch.autograd.grad(new.sum(),pars,retain_graph=True,allow_unused=True);g2=torch.autograd.grad(old.sum(),pars,allow_unused=True)
                for a,b in zip(g1,g2):torch.testing.assert_close(a,b,rtol=1e-10,atol=1e-12)
    def test_optical_values_gradients(self):
        rng=np.random.default_rng(3);records=[]
        for i in range(70):
            q=1+i%5;support=np.zeros((q,4));support[:,:2]=rng.uniform(0,500,(q,2));support[:,3]=rng.uniform(.1,1,q);support[:,3]/=support[:,3].sum()
            records.append({'variable':BANDS[i%8],'support':support.tolist()})
        new=self.model.optical_covariance(records);old=optical_oracle(self.model,records)
        torch.testing.assert_close(new,old,rtol=1e-12,atol=1e-14)
        pars=[self.model.log_error_range,self.model.opt_band_raw,self.model.opt_scene_raw,self.model.opt_scene_v,self.model.opt_error]
        a=torch.autograd.grad(new.square().sum(),pars);b=torch.autograd.grad(old.square().sum(),pars)
        for x,y in zip(a,b):torch.testing.assert_close(x,y,rtol=1e-11,atol=1e-13)
    def test_constrained_index(self):
        nodes=np.array([[0,0],[1,0],[0,1],[1,1],[2,0],[2,1]],float)
        mesh=Mesh(nodes,[[0,1,2],[1,3,2],[1,4,3],[4,5,3]],np.zeros(6,int))
        rng=np.random.default_rng(6);points=rng.uniform([0,0],[2,1],(500,2))
        ids,w=mesh.locate(points)
        np.testing.assert_allclose((nodes[ids]*w[:,:,None]).sum(1),points,atol=1e-13)

        hole=Mesh([[0,0],[1,0],[0,1],[2,2],[3,2],[2,3]],[[0,1,2],[3,4,5]],[0,0,0,1,1,1])
        with self.assertRaises(ValueError):hole.locate([[1.5,1.5]])
    def test_instantaneous_time_measurement_mixture(self):
        model=self.model;times=self.times;states=self.states
        class Posterior:
            def complete(self,*args):return times,states
        original=cli.load;cli.load=lambda _,device="cpu": (model,Posterior(),{})
        try:
            with tempfile.TemporaryDirectory() as root:
                root=Path(root)
                support=np.array([[.2,.2,times[0],.35],[.2,.2,times[-1],.65]])
                q={"queries":[{"id":"time","support":support.tolist(),"variable":"TN","measurement":True,"mode":"instantaneous_time_mixture"}]}
                (root/'q.json').write_text(json.dumps(q))
                cli.predict('unused',root/'q.json',root/'out.csv',root/'draws.npz',7,seed=71,chunk_size=1)
                field=model.field(states,times,support[:,:2],support[:,2],'concentration','TN').detach()
                gen=torch.Generator().manual_seed(72)
                selected=torch.multinomial(torch.tensor(support[:,3]),7,replacement=True,generator=gen)
                sigma=model.scales[0]*(1e-4+torch.nn.functional.softplus(model.error_raw[0]))
                noise=(sigma*torch.randn(7,generator=gen)).detach().numpy()
                expected=field[selected,torch.arange(7)].numpy()+noise
                with np.load(root/'draws.npz') as archive:actual=archive['draws'][0]
                np.testing.assert_allclose(actual,expected,atol=1e-13)
                averaged=(field*torch.tensor(support[:,3,None])).sum(0).numpy()+noise
                self.assertGreater(np.max(np.abs(actual-averaged)),1e-5)
        finally:cli.load=original

    def test_streaming_and_joint_draws(self):
        model=self.model;times=self.times;states=self.states
        class Posterior:
            def complete(self,*args):return times,states
        original=cli.load;cli.load=lambda _,device="cpu": (model,Posterior(),{})
        try:
            with tempfile.TemporaryDirectory() as root:
                root=Path(root);q=json.loads((ROOT/'validation_case/regional_queries.json').read_text());q['queries']*=3
                for item in q['queries']:
                    for support in item['support']:support[2]=float(times[0])
                (root/'queries.json').write_text(json.dumps(q));(root/'queries.jsonl').write_text('\n'.join(json.dumps(v) for v in q['queries']))
                for chunk,extension in [(1,'json'),(9,'jsonl')]:
                    cli.predict('unused',root/f'queries.{extension}',root/f'{chunk}.csv',root/f'{chunk}.npz',7,regional=True,chunk_size=chunk)
                x=np.load(root/'1.npz');y=np.load(root/'9.npz');np.testing.assert_allclose(x['draws'],y['draws'],atol=1e-13)
                np.testing.assert_array_equal(x['realization_id'],np.arange(7));self.assertEqual(x['draws'].shape,(21,7))
                mask=np.ones((2,2),bool);np.save(root/'mask.npy',mask)
                raster={'raster':{'mask_npy':'mask.npy','transform':[.1,0,.05,0,.1,.05],'times':[float(times[0])],'variables':['TN','TP']}}
                (root/'raster.json').write_text(json.dumps(raster))
                cli.predict('unused',root/'raster.json',root/'raster.csv',root/'raster.npz',7,chunk_size=2)
                x.close();y.close()
                with np.load(root/'raster.npz') as archive:z=archive['draws']
                self.assertEqual(z.shape,(8,7))
                pts=np.array([[.1,.1],[.2,.1],[.1,.2],[.2,.2]])
                expected=model.field(states,times,pts,np.full(4,times[0]),'concentration','TN').detach().numpy()
                np.testing.assert_allclose(z[:4],expected,atol=1e-13)
        finally:cli.load=original

if __name__=='__main__':unittest.main(verbosity=2)

