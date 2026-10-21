
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aquafieldinr.data import Dataset
from aquafieldinr.model import AquaFieldINR
from aquafieldinr import likelihood


def original_predictions(model, states, blocks):
    for block in blocks:
        predictions=[]
        for record in block:
            support=np.asarray(record["support"])
            field=model.field(states,model.data.times,support[:,:2],support[:,2],record["kind"],record["variable"])
            predictions.append((field,torch.tensor(support[:,3],dtype=field.dtype)))
        yield block,predictions


def run():
    torch.set_num_threads(1);torch.manual_seed(512)
    data=Dataset(Path(__file__).resolve().parents[1]/"validation_case/observations.json")
    model=AquaFieldINR(data,backend="markov")
    states=(.2*torch.randn(model.total_dimension,4)).requires_grad_()
    parameters=[states,*model.parameters()]
    blocks=likelihood.groups(data.records)
    actual=likelihood.log_likelihood(model,states,blocks)
    grad=torch.autograd.grad(actual,parameters,allow_unused=True)
    with patch.object(likelihood,"_prediction_groups",original_predictions):
        expected=likelihood.log_likelihood(model,states,blocks)
        reference=torch.autograd.grad(expected,parameters,allow_unused=True)
    torch.testing.assert_close(actual,expected,rtol=1e-11,atol=1e-9)
    for got,wanted in zip(grad,reference):
        assert (got is None)==(wanted is None)
        if got is not None:torch.testing.assert_close(got,wanted,rtol=1e-10,atol=1e-7)

    model.data.records=[r for r in model.data.records if r["kind"]=="concentration"]
    first=likelihood.regularization(model)
    with torch.no_grad():
        model.gamma.fill_(100.)
        model.opt_scene_v.fill_(100.)
    second=likelihood.regularization(model)
    torch.testing.assert_close(first,second,rtol=0,atol=0)
    print('Tests passed.')


if __name__=="__main__":run()
