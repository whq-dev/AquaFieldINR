"""Joint neural field and continuous-time Gaussian state prior."""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .data import INDICATORS, BANDS

torch.set_default_dtype(torch.float64)


def stabilize(cov):
    ridge=cov.diagonal().abs().mean().detach().clamp_min(1e-12)*1e-10
    return (cov+cov.T)/2 + torch.eye(len(cov),device=cov.device,dtype=cov.dtype)*ridge


class AquaFieldINR(nn.Module):
    def __init__(self, data, shared_rank=2, optical_rank=1, max_active_dimension=6000, backend="markov"):
        super().__init__()
        self.data = data
        self.shared_rank = int(shared_rank)
        self.optical_rank = int(optical_rank) if any(r["kind"]=="optical" for r in data.records) else 0
        self.max_active_dimension = max_active_dimension
        self.backend = backend
        self.network = nn.Sequential(nn.Linear(5,32),nn.SiLU(),nn.Linear(32,32),nn.SiLU(),nn.Linear(32,1),nn.Tanh())
        nn.init.zeros_(self.network[-2].weight)
        nn.init.zeros_(self.network[-2].bias)
        # Tanh output is multiplied by two in the basis adapter.
        self.lambda_ = nn.Parameter(torch.randn(len(data.meshes),3,self.shared_rank)*.1)
        self.gamma = nn.Parameter(torch.randn(len(data.meshes),8,self.shared_rank)*.1)
        self.omega = nn.Parameter(torch.randn(len(data.meshes),8,self.optical_rank)*.1)
        self.beta_opt = nn.Parameter(torch.zeros(8))
        self.years = sorted(set(int(str(np.datetime64("2021-01-01")+np.timedelta64(int(t),"D"))[:4]) for t in data.times))
        self.background = nn.Parameter(torch.zeros(3,len(self.years)+2))
        self.log_kappa = nn.Parameter(torch.full((len(data.meshes),),-5.))
        self.blocks = []
        for ell,mesh in enumerate(data.meshes):
            for kind,count in (("shared",self.shared_rank),("private",3),("optical",self.optical_rank)):
                for q in range(count):
                    self.blocks.append((ell,kind,q,len(mesh.nodes)))
        k = len(self.blocks)
        self.log_tau = nn.Parameter(torch.linspace(math.log(3),math.log(30),k))
        self.log_amp = nn.Parameter(torch.zeros(k))
        self.forcing = nn.Parameter(torch.zeros(k,4))
        self.error_raw = nn.Parameter(torch.full((3,),-2.))
        self.opt_error = nn.Parameter(torch.full((8,),-2.))
        self.opt_band_raw = nn.Parameter(torch.eye(8)*-2)
        self.opt_scene_v = nn.Parameter(torch.zeros(8))
        self.opt_scene_raw = nn.Parameter(torch.full((8,),-3.))
        self.log_error_range = nn.Parameter(torch.tensor(math.log(500.)))
        self.register_buffer("scales",torch.tensor(data.scales))
        self.register_buffer("optical_scales",torch.tensor(data.optical_scales))
        offsets = np.cumsum([0]+[len(data.times)*b[3] for b in self.blocks])
        self.offsets = offsets
        active = set()
        for r in data.records:
            support = np.asarray(r["support"])
            for ell,mesh in enumerate(data.meshes):
                nodes,geometric_weights = mesh.locate(support[:,:2])
                for z,(layer,kind,q,n) in enumerate(self.blocks):
                    use = layer == ell and ((r["kind"]=="concentration" and (kind=="shared" or (kind=="private" and q==INDICATORS.index(r["variable"])))) or (r["kind"]=="optical" and kind in ("shared","optical")))
                    if use:
                        for row,t in enumerate(support[:,2]):
                            ti=np.searchsorted(data.times,t)
                            touched=nodes[row][geometric_weights[row]>0]
                            active.update((offsets[z]+ti*n+touched).tolist())
        active=np.array(sorted(active),int)
        self.register_buffer("active",torch.tensor(active,dtype=torch.long))
        self.total_dimension = int(offsets[-1])
        self.eta = nn.Parameter(torch.zeros(len(active))) if backend in ("dense","sparse","exact_active") else None
        # Posterior constructs its factor after symbolic sparse elimination.
        self.factor_raw = None
        self.register_buffer("factor_mask",torch.empty(0))
        self._initialize_background()

    def _initialize_background(self):
        for j,name in enumerate(INDICATORS):
            rows=[r for r in self.data.records if r["kind"]=="concentration" and r["variable"]==name and "value" in r]
            t=np.array([np.asarray(r["support"])[:,2].mean() for r in rows])
            X=self.design(t).detach().numpy()
            y=np.array([r["value"] for r in rows])/self.data.scales[j]
            inverse=y+np.log(-np.expm1(-y))
            self.background.data[j]=torch.tensor(np.linalg.lstsq(X,inverse,rcond=None)[0])
        for b,name in enumerate(BANDS):
            values=[r["value"] for r in self.data.records if r["kind"]=="optical" and r["variable"]==name]
            if values:self.beta_opt.data[b]=float(np.mean(values))

    def design(self,times):
        t=torch.as_tensor(times,dtype=self.scales.dtype,device=self.scales.device)
        years=np.array([int(str(np.datetime64("2021-01-01")+np.timedelta64(int(v),"D"))[:4]) for v in np.asarray(times)])

        nearest=np.argmin(abs(years[:,None]-np.asarray(self.years)[None,:]),axis=1)
        intercept=F.one_hot(torch.as_tensor(nearest,device=t.device),len(self.years)).to(t.dtype)
        return torch.cat((intercept,torch.sin(2*math.pi*t[:,None]/365.25),torch.cos(2*math.pi*t[:,None]/365.25)),1)

    def temporal_mean(self,times,block,weather=None):
        weather=self.data.weather if weather is None else weather
        times=np.asarray(times)
        tau=torch.exp(self.log_tau[block])
        out=[]
        start=float(self.data.weather[0,0])
        if times.min()<start-1e-10:
            raise ValueError('E_MODEL_103')
        value=torch.zeros((),device=tau.device)
        cursor=start
        for end in times:
            if end<cursor-1e-10:raise ValueError('E_MODEL_107')
            while cursor<float(end)-1e-12:
                day=math.floor(cursor)
                hit=np.flatnonzero(weather[:,0]==day)
                if len(hit)!=1:raise ValueError('E_MODEL_111')
                stop=min(day+1,float(end))
                dt=stop-cursor
                rho=torch.exp(-dt/tau)
                m=torch.as_tensor(weather[hit[0],1:],device=tau.device,dtype=tau.dtype)
                value=rho*value+tau*(1-rho)*(self.forcing[block]@m)
                cursor=stop
            out.append(value)
        return torch.stack(out)

    def prior(self,times=None,weather=None):
        times=self.data.times if times is None else np.asarray(times)
        dt=torch.as_tensor(abs(times[:,None]-times[None,:]),device=self.scales.device)
        covs=[]; means=[]
        for z,(ell,kind,q,n) in enumerate(self.blocks):
            spatial=self.data.meshes[ell].covariance(self.log_kappa[ell])
            amp=torch.exp(2*self.log_amp[z]) if kind=="private" else 1.
            temporal=torch.exp(-dt/torch.exp(self.log_tau[z]))
            covs.append(amp*torch.kron(temporal.contiguous(),spatial.contiguous()))
            means.append(self.temporal_mean(times,z,weather).repeat_interleave(n))
        return torch.cat(means),torch.block_diag(*covs)

    def q_factor(self):
        if hasattr(self, "_exact_posterior"):
            return self._exact_posterior.q_factor()
        lower=torch.tril(self.factor_raw,-1)*self.factor_mask
        return lower+torch.diag(1e-5+F.softplus(self.factor_raw.diagonal()))

    def q_covariance(self):
        L=self.q_factor()
        if L.is_sparse:
            L=L.to_dense()
        inv=torch.linalg.solve_triangular(L,torch.eye(len(L),device=L.device,dtype=L.dtype),upper=False)
        covariance=inv.T@inv
        if hasattr(self,"factor_permutation"):
            inverse=torch.argsort(self.factor_permutation)
            covariance=covariance[inverse][:,inverse]
        return covariance

    def active_samples(self,count,generator=None):
        if hasattr(self, "_exact_posterior"):
            return self._exact_posterior.active_samples(count,generator)
        noise=torch.randn(len(self.active),count,generator=generator,device=self.scales.device)
        return self.eta[:,None]+torch.linalg.solve_triangular(self.q_factor().T,noise,upper=True)

    def basis(self,ell,points):
        class Adapter(nn.Module):
            def __init__(self,net):super().__init__();self.net=net
            def forward(self,x):return 2*self.net(x)
        return self.data.meshes[ell].basis(points,Adapter(self.network))

    def local_basis(self,ell,points):
        class Adapter(nn.Module):
            def __init__(self,net):super().__init__();self.net=net
            def forward(self,x):return 2*self.net(x)
        return self.data.meshes[ell].local_basis(points,Adapter(self.network))

    def field(self,states,times,points,query_times,kind,variable,chunk_size=4096):
        
        points=np.asarray(points);query_times=np.asarray(query_times)
        if len(points)!=len(query_times):raise ValueError('E_MODEL_171')
        if chunk_size<1:raise ValueError('E_MODEL_172')
        if len(points)>chunk_size:
            return torch.cat([self.field(states,times,points[i:i+chunk_size],query_times[i:i+chunk_size],kind,variable,chunk_size)
                for i in range(0,len(points),chunk_size)],0)
        ti=np.searchsorted(times,query_times)
        if np.any(ti>=len(times)) or not np.allclose(np.asarray(times)[ti],query_times):
            raise ValueError('E_MODEL_178')
        if kind=="concentration":
            j=INDICATORS.index(variable)
            result=(self.design(query_times)@self.background[j])[:,None].expand(-1,states.shape[1]).clone()
        elif kind=="optical":
            j=BANDS.index(variable)
            result=self.beta_opt[j].expand(len(points),states.shape[1]).clone()
        else:raise ValueError('E_MODEL_185')
        offset=0;bases={}
        time_index=torch.as_tensor(ti,device=self.scales.device,dtype=torch.long)
        for z,(ell,bkind,q,n) in enumerate(self.blocks):
            start=offset;offset+=len(times)*n
            if kind=="concentration":
                if bkind=="optical" or (bkind=="private" and q!=j):continue
                weight=self.lambda_[ell,j,q] if bkind=="shared" else 1.
            else:
                if bkind=="private":continue
                weight=(self.gamma[ell,j,q] if bkind=="shared" else self.omega[ell,j,q])*self.optical_scales[j]
            if ell not in bases:bases[ell]=self.local_basis(ell,points)
            ids,weights=bases[ell]
            flat=start+time_index[:,None]*n+ids


            state_index=flat.to(device=states.device)
            values=states[state_index].to(device=self.scales.device,dtype=self.scales.dtype)
            result=result+weight*(weights[:,:,None]*values).sum(1)
        return self.scales[j]*F.softplus(result) if kind=="concentration" else result

    def optical_covariance(self,records):
        band=torch.tensor([BANDS.index(r["variable"]) for r in records],device=self.scales.device)
        B=torch.tril(self.opt_band_raw,-1)+torch.diag(1e-4+F.softplus(self.opt_band_raw.diagonal()))
        kb=B@B.T
        ks=torch.outer(self.opt_scene_v,self.opt_scene_v)+torch.diag(F.softplus(self.opt_scene_raw)**2)


        supports=[np.asarray(r["support"],float) for r in records]
        max_q=max(len(a) for a in supports)
        xy=np.zeros((len(records),max_q,2));weights=np.zeros((len(records),max_q))
        for i,a in enumerate(supports):
            xy[i,:len(a)]=a[:,:2];weights[i,:len(a)]=a[:,3]
        xy=torch.as_tensor(xy,device=band.device,dtype=self.scales.dtype)
        weights=torch.as_tensor(weights,device=band.device,dtype=self.scales.dtype)
        block_size=64;rows=[]
        for i in range(0,len(records),block_size):
            row=[]
            for j in range(0,len(records),block_size):
                delta=xy[i:i+block_size,None,:,None,:]-xy[None,j:j+block_size,None,:,:]
                distances=torch.linalg.vector_norm(delta,dim=-1)
                products=weights[i:i+block_size,None,:,None]*weights[None,j:j+block_size,None,:]
                row.append((torch.exp(-distances/torch.exp(self.log_error_range))*products).sum((-1,-2)))
            rows.append(torch.cat(row,1))
        spatial=torch.cat(rows,0)
        scales=self.optical_scales[band]
        return torch.outer(scales,scales)*(kb[band[:,None],band[None,:]]*spatial+ks[band[:,None],band[None,:]]+torch.diag((1e-3+F.softplus(self.opt_error[band]))**2))
