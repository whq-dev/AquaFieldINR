"""Exact sparse SPD operations with deterministic analytic reverse derivatives.

All inverse traces are evaluated by streamed solves, never Monte Carlo.  SuperLU
is used as a sparse direct solver; Gaussian draws additionally verify the SPD
LDL factor identity before using its triangular factors.
"""
import os
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import torch

CHUNK = 16

def budget_bytes():
    return int(os.environ.get('AQUAFIELD_SPARSE_MEMORY_MB', '4096')) * 1024**2

def check_storage(nnz, label, multiplier=96):
    if int(nnz)*multiplier > budget_bytes():
        raise MemoryError(f'{label} needs approximately {int(nnz)*multiplier/1024**3:.2f} GiB; increase AQUAFIELD_SPARSE_MEMORY_MB only when that memory is available.')

def matrix(values, rows, cols, n):
    return sp.csc_matrix((np.asarray(values), (rows, cols)), shape=(n,n))

def factor(a, natural=False):
    a=sp.csc_matrix(a)
    check_storage(a.nnz, 'Sparse input and direct factor workspace')
    lu=sla.splu(a, permc_spec='NATURAL' if natural else 'MMD_AT_PLUS_A', diag_pivot_thresh=0., options={'SymmetricMode': True})
    check_storage(lu.L.nnz+lu.U.nnz, 'Sparse direct factors')
    if np.any(lu.U.diagonal() <= 0):
        raise np.linalg.LinAlgError('Precision factor has a nonpositive pivot.')
    return lu

def logdet(lu):
    return np.log(lu.U.diagonal()).sum()

def selected_inverse(lu, rows, cols, n):
    result=np.empty(len(rows)); order=np.argsort(cols,kind='stable')
    unique=np.unique(cols)
    for k in range(0,len(unique),CHUNK):
        selected=unique[k:k+CHUNK]
        rhs=np.zeros((n,len(selected)));rhs[selected,np.arange(len(selected))]=1
        solved=lu.solve(rhs)
        lo=np.searchsorted(cols[order],selected[0]); hi=np.searchsorted(cols[order],selected[-1],side='right')
        ids=order[lo:hi]
        result[ids]=solved[rows[ids],np.searchsorted(selected,cols[ids])]
    return result

def covariance_draw(lu, count, rng):
    """Sample A^-1 from verified symmetric sparse LDL, with correct permutation."""
    if not np.array_equal(lu.perm_r,lu.perm_c):
        raise np.linalg.LinAlgError('SPD draw requires identical row/column permutation.')
    d=lu.U.diagonal()
    residual=lu.U-sp.diags(d)@lu.L.T
    scale=max(abs(lu.U.data).max(initial=0),1.)
    if residual.nnz and abs(residual.data).max(initial=0)>1e-9*scale:
        raise np.linalg.LinAlgError('Sparse factor is not a symmetric LDL decomposition.')
    noise=rng.standard_normal((len(d),count))/np.sqrt(d)[:,None]
    y=sla.spsolve_triangular(lu.L.T.tocsr(),noise,lower=False,unit_diagonal=True)
    return y[lu.perm_c]

class _Solve(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, rhs, rows, cols, n, transpose):
        a=matrix(values.detach().cpu().numpy(),rows,cols,n)
        lu=factor(a)
        x=lu.solve(rhs.detach().cpu().numpy(),trans='T' if transpose else 'N')
        ctx.save_for_backward(values);ctx.lu=lu;ctx.x=x;ctx.rows=rows;ctx.cols=cols;ctx.transpose=transpose
        return torch.as_tensor(x,dtype=rhs.dtype,device=rhs.device)
    @staticmethod
    def backward(ctx,g):
        z=ctx.lu.solve(g.detach().cpu().numpy(),trans='N' if ctx.transpose else 'T')
        x=ctx.x
        if x.ndim==1:x=x[:,None];z=z[:,None]
        if ctx.transpose: grad=-(x[ctx.rows]*z[ctx.cols]).sum(1)
        else:grad=-(z[ctx.rows]*x[ctx.cols]).sum(1)
        return torch.as_tensor(grad,dtype=g.dtype,device=g.device),torch.as_tensor(z.squeeze(1) if g.ndim==1 else z,dtype=g.dtype,device=g.device),None,None,None,None

def solve(values,rhs,rows,cols,n,transpose=False):
    # Keep the custom SciPy node on the CPU autograd engine. Explicit copy
    # nodes carry gradients across devices without running SuperLU from a
    # CUDA autograd worker (which is unsafe in some Windows SciPy builds).
    return _Solve.apply(values.cpu(),rhs.cpu(),rows,cols,n,transpose).to(rhs.device)

class _TriSolve(torch.autograd.Function):
    @staticmethod
    def forward(ctx,values,rhs,rows,cols,n):
        a=matrix(values.detach().cpu().numpy(),rows,cols,n)
        x=sla.spsolve_triangular(a.T.tocsr(),rhs.detach().cpu().numpy(),lower=False)
        ctx.a=a;ctx.x=x;ctx.rows=rows;ctx.cols=cols
        return torch.as_tensor(x,dtype=rhs.dtype,device=rhs.device)
    @staticmethod
    def backward(ctx,g):
        z=sla.spsolve_triangular(ctx.a.tocsr(),g.detach().cpu().numpy(),lower=True)
        grad=-(ctx.x[ctx.rows]*z[ctx.cols]).sum(1)
        return torch.as_tensor(grad,dtype=g.dtype,device=g.device),torch.as_tensor(z,dtype=g.dtype,device=g.device),None,None,None

def triangular_draw(values,noise,rows,cols,n):
    return _TriSolve.apply(values.cpu(),noise.cpu(),rows,cols,n).to(noise.device)

class _InverseMean(torch.autograd.Function):
    @staticmethod
    def forward(ctx,values,rows,cols,n):
        lu=factor(matrix(values.detach().cpu().numpy(),rows,cols,n))
        value=0.;grad=np.zeros(len(rows))
        for start in range(0,n,CHUNK):
            ids=np.arange(start,min(start+CHUNK,n));rhs=np.zeros((n,len(ids)));rhs[ids,np.arange(len(ids))]=1
            x=lu.solve(rhs);value+=x[ids,np.arange(len(ids))].sum()
            for e in range(0,len(rows),100000):
                sl=slice(e,e+100000);grad[sl]-=(x[rows[sl]]*x[cols[sl]]).sum(1)/n
        ctx.grad=grad
        return values.new_tensor(value/n)
    @staticmethod
    def backward(ctx,g):return g*torch.as_tensor(ctx.grad,dtype=g.dtype,device=g.device),None,None,None

def inverse_mean(values,rows,cols,n):return _InverseMean.apply(values.cpu(),rows,cols,n).to(values.device)

class SchurSystem:
    def __init__(self,q,active):
        self.q=sp.csc_matrix(q);self.active=np.asarray(active,dtype=int)
        self.inactive=np.setdiff1d(np.arange(q.shape[0]),self.active,assume_unique=True)
        self.qa=q[self.active][:,self.active].tocsc()
        self.qua=q[self.inactive][:,self.active].tocsc()
        self.lu=factor(q[self.inactive][:,self.inactive]) if len(self.inactive) else None
    def extend(self,x):
        out=np.zeros((self.q.shape[0],x.shape[1]));out[self.active]=x
        if self.lu is not None:out[self.inactive]=-self.lu.solve(self.qua@x)
        return out
    def apply(self,x):
        y=self.qa@x
        if self.lu is not None:y-=self.qua.T@self.lu.solve(self.qua@x)
        return y

class _KL(torch.autograd.Function):
    @staticmethod
    def forward(ctx,qvalues,lvalues,delta,qr,qc,lr,lc,active,n):
        q=matrix(qvalues.detach().cpu().numpy(),qr,qc,n);a=len(active)
        l=matrix(lvalues.detach().cpu().numpy(),lr,lc,a)
        system=SchurSystem(q,active);qlu=factor(q)
        d=delta.detach().cpu().numpy();pd=system.apply(d[:,None])[:,0]
        trace=0.;gq=np.zeros(len(qr));gl=np.zeros(len(lr))
        needq=ctx.needs_input_grad[0];needl=ctx.needs_input_grad[1]
        for k in range(0,a,CHUNK):
            ids=np.arange(k,min(k+CHUNK,a));rhs=np.zeros((a,len(ids)));rhs[ids,np.arange(len(ids))]=1
            x=sla.spsolve_triangular(l.T.tocsr(),rhs,lower=False)
            px=system.apply(x);trace+=(x*px).sum()
            if needq:
                y=system.extend(x)
                for e in range(0,len(qr),100000):
                    sl=slice(e,e+100000);gq[sl]+=(y[qr[sl]]*y[qc[sl]]).sum(1)
            if needl:
                w=sla.spsolve_triangular(l.tocsr(),px,lower=True)
                w=sla.spsolve_triangular(l.T.tocsr(),w,lower=False)
                mask=(lc>=ids[0])&(lc<=ids[-1]);gl[mask]=-w[lr[mask],lc[mask]-ids[0]]
        if needq:
            md=system.extend(d[:,None])[:,0]
            gq+=md[qr]*md[qc]-selected_inverse(qlu,qr,qc,n)
            if len(system.inactive):
                lookup=np.full(n,-1);lookup[system.inactive]=np.arange(len(system.inactive))
                mask=(lookup[qr]>=0)&(lookup[qc]>=0)
                gq[mask]+=selected_inverse(system.lu,lookup[qr[mask]],lookup[qc[mask]],len(system.inactive))
            gq*=.5
        diagonal=lr==lc
        if needl:gl[diagonal]+=1/l.diagonal()
        logdet_p=logdet(qlu)-(logdet(system.lu) if system.lu else 0.)
        result=.5*(trace+d@pd-a-logdet_p+2*np.log(l.diagonal()).sum())
        ctx.grads=(gq,gl,pd)
        return qvalues.new_tensor(result)
    @staticmethod
    def backward(ctx,g):
        grads=[g*torch.as_tensor(x,dtype=g.dtype,device=g.device) for x in ctx.grads]
        return *grads,None,None,None,None,None,None

def exact_kl(qvalues,lvalues,delta,qr,qc,lr,lc,active,n):
    return _KL.apply(qvalues.cpu(),lvalues.cpu(),delta.cpu(),qr,qc,lr,lc,np.asarray(active),n).to(qvalues.device)
