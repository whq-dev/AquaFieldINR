"""Canonical factor labels without altering fitted Gaussian parameters."""
import torch


def canonical_representation(model):
    """Return joint state/loading/forcing permutations and signs, layer by layer.

    This readout view fixes labels without refactoring the fitted precision.
    Each canonical state is sign times its original state; its loadings and
    forcing receive the same sign, leaving every observable exactly unchanged.
    """
    entries=[]
    for layer in range(len(model.data.meshes)):
        for kind in ("shared","private","optical"):
            indices=[i for i,b in enumerate(model.blocks) if b[0]==layer and b[1]==kind]
            if kind!="private":indices.sort(key=lambda i:(float(model.log_tau[i].detach()),i))
            for canonical_index,source in enumerate(indices):
                _,_,q,_=model.blocks[source]
                sign=1
                if kind=="shared":values=torch.cat((model.lambda_[layer,:,q],model.gamma[layer,:,q]))
                elif kind=="optical":values=model.omega[layer,:,q]
                else:values=model.scales.new_ones(1)
                nonzero=values.detach()[values.detach()!=0]
                if len(nonzero) and float(nonzero[0])<0:sign=-1
                entries.append({"layer":layer,"kind":kind,"index":canonical_index,"source_block":source,"source_index":q,"sign":sign,"time_scale":float(model.log_tau[source].detach().cpu().exp())})
    return {"version":1,"time_scale_order":"ascending","blocks":entries}


def canonical_blocks(model,states,times,representation=None):
    """Iterate canonical state blocks, corresponding readouts and forcing."""
    representation=representation or canonical_representation(model)
    offsets=[0]
    for _,_,_,nodes in model.blocks:offsets.append(offsets[-1]+len(times)*nodes)
    for entry in representation["blocks"]:
        index=entry["source_block"];layer,kind,q,nodes=model.blocks[index];sign=entry["sign"]
        state=states[offsets[index]:offsets[index+1]]*sign
        if kind=="shared":
            concentration=model.lambda_[layer,:,q]*sign
            optical=model.gamma[layer,:,q]*sign
        elif kind=="private":
            concentration=torch.nn.functional.one_hot(torch.tensor(q,device=model.scales.device),3).to(model.scales)
            optical=model.gamma.new_zeros(8)
        else:
            concentration=model.lambda_.new_zeros(3)
            optical=model.omega[layer,:,q]*sign
        yield entry,state,concentration,optical,model.forcing[index]*sign
