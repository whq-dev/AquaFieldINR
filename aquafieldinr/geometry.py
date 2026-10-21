
import numpy as np
import torch
from scipy import sparse


class Mesh:
    def __init__(self, nodes, triangles, components, scale=1.0):
        self.nodes = np.asarray(nodes, float)
        self.triangles = np.asarray(triangles, int)
        self.components = np.asarray(components, int)
        self.scale = float(scale)
        if self.nodes.ndim != 2 or self.nodes.shape[1] != 2:
            raise ValueError('E_GEOMETRY_014')
        if self.triangles.ndim != 2 or self.triangles.shape[1] != 3:
            raise ValueError('E_GEOMETRY_016')
        if self.components.shape != (len(self.nodes),):
            raise ValueError('E_GEOMETRY_018')
        if np.any(self.triangles < 0) or np.any(self.triangles >= len(self.nodes)):
            raise ValueError('E_GEOMETRY_020')
        if not np.all(np.isfinite(self.nodes)) or self.scale <= 0:
            raise ValueError('E_GEOMETRY_022')
        self.mass = np.zeros(len(self.nodes))
        stiffness_rows, stiffness_cols, stiffness_values = [], [], []
        self._inverse_design = []
        self._triangle_min = []
        self._triangle_max = []
        for tri in self.triangles:
            xy = self.nodes[tri]
            if len(np.unique(self.components[tri])) != 1:
                raise ValueError('E_GEOMETRY_031')
            area = abs(np.linalg.det(np.column_stack((xy[1]-xy[0], xy[2]-xy[0]))))/2
            if area <= 0:
                raise ValueError('E_GEOMETRY_034')
            self.mass[tri] += area/3

            design = np.column_stack((np.ones(3), xy))
            grad = np.linalg.inv(design)[1:, :]
            local = area*(grad.T @ grad)
            stiffness_rows.extend(np.repeat(tri, 3))
            stiffness_cols.extend(np.tile(tri, 3))
            stiffness_values.extend(local.ravel())
            self._inverse_design.append(np.linalg.inv(design))
            self._triangle_min.append(xy.min(0))
            self._triangle_max.append(xy.max(0))
        if np.any(self.mass <= 0):
            raise ValueError('E_GEOMETRY_047')
        self.laplacian = sparse.coo_matrix((stiffness_values, (stiffness_rows, stiffness_cols)), shape=(len(self.nodes), len(self.nodes))).tocsr()
        self._inverse_design = np.asarray(self._inverse_design)
        self._triangle_min = np.asarray(self._triangle_min)
        self._triangle_max = np.asarray(self._triangle_max)
        self._index_origin = self.nodes.min(0)
        self._index_size = max(1, min(256, int(np.sqrt(len(self.triangles)))))
        self._index_step = np.maximum(np.ptp(self.nodes, axis=0), 1e-12)/self._index_size
        self._bins = {}
        for i, (lo, hi) in enumerate(zip(self._triangle_min, self._triangle_max)):
            first = np.clip(np.floor((lo-self._index_origin-1e-8)/self._index_step).astype(int), 0, self._index_size-1)
            last = np.clip(np.floor((hi-self._index_origin+1e-8)/self._index_step).astype(int), 0, self._index_size-1)
            for x in range(first[0], last[0]+1):
                for y in range(first[1], last[1]+1):
                    self._bins.setdefault((x,y), []).append(i)
        self.origin = self.nodes.mean(0)
        self.extent = max(np.ptp(self.nodes, axis=0).max(), 1.0)

    def locate(self, points):
        
        points = np.asarray(points, float)
        ids = np.full((len(points), 3), -1, int)
        weights = np.zeros((len(points), 3))
        if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
            raise ValueError('E_GEOMETRY_071')
        bins = np.clip(np.floor((points-self._index_origin)/self._index_step).astype(int), 0, self._index_size-1)
        keys, groups = np.unique(bins, axis=0, return_inverse=True)
        for group, key in enumerate(keys):
            members = np.flatnonzero(groups == group)
            for index in self._bins.get(tuple(key), ()):
                candidate = members[(ids[members, 0] < 0) &
                    (points[members] >= self._triangle_min[index]-1e-8).all(1) &
                    (points[members] <= self._triangle_max[index]+1e-8).all(1)]
                if len(candidate) == 0:
                    continue
                bary = np.column_stack((np.ones(len(candidate)), points[candidate])) @ self._inverse_design[index]
                inside = (bary >= -1e-8).all(1)
                selected = candidate[inside]
                ids[selected] = self.triangles[index]
                weights[selected] = np.clip(bary[inside], 0, 1)
        if np.any(ids < 0):
            raise ValueError('E_GEOMETRY_088')
        weights /= weights.sum(1, keepdims=True)
        return ids, weights

    def covariance(self, log_kappa):
        mass = torch.as_tensor(self.mass, dtype=log_kappa.dtype, device=log_kappa.device)
        lap = torch.as_tensor(self.laplacian.toarray(), dtype=log_kappa.dtype, device=log_kappa.device)
        q = torch.diag(mass)*torch.exp(2*log_kappa) + lap
        cov = torch.cholesky_inverse(torch.linalg.cholesky(q))
        return cov / cov.diagonal().mean()

    def local_basis(self, points, network):
        
        ids, weights = self.locate(points)
        dtype = next(network.parameters()).dtype
        device = next(network.parameters()).device
        ids_t = torch.as_tensor(ids, device=device)
        p = torch.as_tensor(points, dtype=dtype, device=device)
        v = torch.as_tensor(self.nodes[ids], dtype=dtype, device=device)
        origin = torch.as_tensor(self.origin, dtype=dtype, device=device)
        x = ((p-origin)/self.extent)[:,None,:].expand(-1,3,-1)
        features = torch.cat((x, (p[:,None,:]-v)/self.extent,
            torch.full((*ids.shape,1), self.scale/self.extent, dtype=dtype,device=device)),2)
        h = network(features).squeeze(-1)
        w = torch.as_tensor(weights,dtype=dtype,device=device)*torch.exp(h-h.max(1,keepdim=True).values)
        w = w/w.sum(1,keepdim=True)
        return ids_t, w

    def basis(self, points, network):
        
        ids, weights = self.local_basis(points, network)
        out = weights.new_zeros((len(points),len(self.nodes)))
        return out.scatter_add(1,ids,weights)


class ConstantBasis:
    
    def __init__(self,reference):
        self.reference=reference
        self.identities=np.unique(reference.components)
        self.nodes=np.stack([reference.nodes[reference.components==c].mean(0) for c in self.identities])
        self.components=self.identities
        self.scale=0.
    def locate(self,points):
        ids,_=self.reference.locate(points)
        comp=self.reference.components[ids[:,0]]
        index=np.searchsorted(self.identities,comp)
        return np.repeat(index[:,None],3,axis=1),np.tile([1.,0.,0.],(len(points),1))
    def covariance(self,log_kappa):
        return torch.eye(len(self.nodes),dtype=log_kappa.dtype,device=log_kappa.device)
    def local_basis(self,points,network):
        ids, weights=self.locate(points)
        p=next(network.parameters())
        return torch.as_tensor(ids,device=p.device),torch.as_tensor(weights,dtype=p.dtype,device=p.device)
    def basis(self,points,network):
        ids,weights=self.local_basis(points,network)
        return weights.new_zeros((len(points),len(self.nodes))).scatter_add(1,ids,weights)
