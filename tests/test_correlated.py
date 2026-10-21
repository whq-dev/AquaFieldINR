import copy
import json
import math
import tempfile
from pathlib import Path
import sys

import numpy as np
import torch
from scipy.stats import multivariate_normal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aquafieldinr.correlated import (
    rectangle_log_probability, gaussian_observation_log_probability,
    concentration_block_log_likelihood,
)
from aquafieldinr.data import Dataset
from aquafieldinr.likelihood import groups


def test_rectangle_and_gradient():
    torch.set_default_dtype(torch.float64)
    mean = torch.tensor([[0.], [0.]], requires_grad=True)
    covariance = torch.tensor([[1., .6], [.6, 1.]], requires_grad=True)
    p = rectangle_log_probability(mean, covariance, torch.zeros(2),
                                  torch.full((2,), float("inf")), 16384).exp()
    expected = .25 + math.asin(.6) / (2 * math.pi)
    assert abs(p.item() - expected) < 2e-5, (p, expected)
    p.sum().backward()
    assert torch.isfinite(mean.grad).all()
    assert torch.isfinite(covariance.grad).all()
    eps = 1e-5
    a = mean.detach().clone(); a[0] += eps
    b = mean.detach().clone(); b[0] -= eps
    plus = rectangle_log_probability(a, covariance.detach(), torch.zeros(2),
                                     torch.full((2,), float("inf")), 16384).exp()
    minus = rectangle_log_probability(b, covariance.detach(), torch.zeros(2),
                                      torch.full((2,), float("inf")), 16384).exp()
    assert abs(mean.grad[0].item() - ((plus-minus)/(2*eps)).item()) < 2e-5


def test_mixed_exact_censor():
    mean = torch.tensor([[.2], [-.1]], dtype=torch.float64)
    cov = torch.tensor([[2., .4], [.4, 1.]], dtype=torch.float64)
    records = [{"value": .3}, {"interval": [-.5, .7]}]
    result = gaussian_observation_log_probability(mean, cov, records).exp().item()
    from scipy.stats import norm
    conditional_mu = -.1 + .4 / 2 * (.3-.2)
    conditional_sd = math.sqrt(1-.4**2/2)
    expected = norm.pdf(.3, .2, math.sqrt(2)) * (
        norm.cdf(.7, conditional_mu, conditional_sd)-norm.cdf(-.5, conditional_mu, conditional_sd)
    )
    assert abs(result-expected) < 1e-12


def test_joint_time_mixture():
    class Model:
        scales = torch.ones(3, dtype=torch.float64)
        error_raw = torch.zeros(3, dtype=torch.float64)
    model = Model()
    corr = [[1., .5], [.5, 1.]]
    records = [
        {"variable":"TN","mode":"instantaneous_time_mixture","value":.7,"_error_correlation":corr},
        {"variable":"TP","mode":"instantaneous_time_mixture","value":.4},
    ]
    fields = [torch.tensor([[.1],[1.1]],dtype=torch.float64),
              torch.tensor([[.2],[.8]],dtype=torch.float64)]
    weights = [torch.tensor([.3,.7],dtype=torch.float64), torch.tensor([.6,.4],dtype=torch.float64)]
    result = concentration_block_log_likelihood(model, records, list(zip(fields,weights))).exp().item()
    covariance = np.asarray(corr) * (1e-4 + math.log(2))**2
    expected = sum(weights[0][i].item()*weights[1][j].item() *
                   multivariate_normal.pdf([.7,.4], [fields[0][i].item(),fields[1][j].item()],covariance)
                   for i in range(2) for j in range(2))
    assert abs(result-expected) < 1e-12
    density_at_average = multivariate_normal.pdf([.7,.4], [.8,.44], covariance)
    assert abs(result-density_at_average) > .01


def test_error_block_exclusion():
    source = Path(__file__).resolve().parents[1]/"validation_case/observations.json"
    document = json.loads(source.read_text())
    for i, row in enumerate(document["records"]):
        row["record_id"] = str(i)
    document["concentration_error_blocks"] = {
        "cross-position": {"record_ids":["0","9","18"], "correlation":[[1,.2,.3],[.2,1,.4],[.3,.4,1]]}
    }
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)/"input.json"
        path.write_text(json.dumps(document))
        data = Dataset(path, excluded_positions=["0"])
    blocks = [b for b in groups(data.records) if "_error_block" in b[0]]
    assert len(blocks) == 1 and [r["record_id"] for r in blocks[0]] == ["9","18"]
    assert blocks[0][0]["_error_correlation"] == [[1,.4],[.4,1]]
    assert not any(any(k.startswith("_error") for k in r) for r in data.document["records"])


if __name__ == "__main__":
    test_rectangle_and_gradient()
    test_mixed_exact_censor()
    test_joint_time_mixture()
    test_error_block_exclusion()
    print('Tests passed.')
