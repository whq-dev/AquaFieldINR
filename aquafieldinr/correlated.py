
import itertools
import math

import torch


def log_interval_mass(lower, upper):
    """Log[Phi(upper)-Phi(lower)], retaining accuracy in either normal tail."""
    right = lower > 0
    large = torch.where(right, torch.special.log_ndtr(-lower), torch.special.log_ndtr(upper))
    small = torch.where(right, torch.special.log_ndtr(-upper), torch.special.log_ndtr(lower))
    return large + torch.log(-torch.expm1((small - large).clamp_max(-1e-15)))


def _standardize_bound(bound, centre, scale):
    finite = torch.isfinite(bound)
    value = (torch.where(finite, bound, torch.zeros_like(bound)) - centre) / scale
    return torch.where(finite, value, bound.expand_as(value))


def rectangle_log_probability(mean, covariance, lower, upper, points=4096, seed=1789):
    """Gaussian rectangle probability by deterministic scrambled Sobol integration.

    The separation-of-variables transform integrates the conditional truncated
    normals. The one-dimensional case is analytic. ``points`` controls numerical
    quadrature, independently of posterior draws; gradients pass through the
    conditional Gaussian transform. Bounds may be infinite.
    """
    dimension, draws = mean.shape
    if dimension == 0:
        return mean.new_zeros(draws)
    chol = torch.linalg.cholesky(covariance)
    if dimension == 1:
        lo = _standardize_bound(lower[0], mean[0], chol[0, 0])
        hi = _standardize_bound(upper[0], mean[0], chol[0, 0])
        return log_interval_mass(lo, hi)
    if points < 2 or points & (points - 1):
        raise ValueError('E_CORRELATED_039')
    engine = torch.quasirandom.SobolEngine(dimension - 1, scramble=True, seed=seed)
    uniforms = engine.draw(points, dtype=mean.dtype).to(mean.device)
    latent = []
    log_weight = mean.new_zeros((points, draws))
    tiny = torch.finfo(mean.dtype).tiny
    eps = torch.finfo(mean.dtype).eps
    for j in range(dimension):
        shift = mean[j][None, :]
        if j:
            shift = shift + sum(chol[j, k] * latent[k] for k in range(j))
        lo = _standardize_bound(lower[j], shift, chol[j, j])
        hi = _standardize_bound(upper[j], shift, chol[j, j])
        mass_log = log_interval_mass(lo, hi)
        log_weight = log_weight + mass_log
        if j + 1 < dimension:
            u = uniforms[:, j, None]
            mass = torch.exp(mass_log)
            cdf = torch.special.ndtr(lo) + u * mass
            survival = torch.special.ndtr(-lo) - u * mass
            sample = torch.where(
                lo > 0,
                -torch.special.ndtri(survival.clamp(tiny, 1 - eps)),
                torch.special.ndtri(cdf.clamp(tiny, 1 - eps)),
            )
            latent.append(sample)
    return torch.logsumexp(log_weight, dim=0) - math.log(points)


def gaussian_observation_log_probability(mean, covariance, records, points=4096):
    
    exact = [i for i, r in enumerate(records) if "value" in r]
    censored = [i for i, r in enumerate(records) if "interval" in r]
    answer = mean.new_zeros(mean.shape[1])
    conditional_mean = mean[censored]
    conditional_covariance = covariance[censored][:, censored]
    if exact:
        values = mean.new_tensor([records[i]["value"] for i in exact])
        residual = values[:, None] - mean[exact]
        cee = covariance[exact][:, exact]
        chol = torch.linalg.cholesky(cee)
        standardized = torch.linalg.solve_triangular(chol, residual, upper=False)
        answer = -0.5 * (
            len(exact) * math.log(2 * math.pi)
            + 2 * torch.log(chol.diagonal()).sum()
            + standardized.square().sum(0)
        )
        if censored:
            cce = covariance[censored][:, exact]
            solution = torch.cholesky_solve(cce.T, chol)
            conditional_mean = conditional_mean + solution.T @ residual
            conditional_covariance = conditional_covariance - cce @ solution
            conditional_covariance = (conditional_covariance + conditional_covariance.T) / 2
    if censored:
        lower = mean.new_tensor([records[i]["interval"][0] for i in censored])
        upper = mean.new_tensor([records[i]["interval"][1] for i in censored])
        answer = answer + rectangle_log_probability(
            conditional_mean, conditional_covariance, lower, upper, points
        )
    return answer


def concentration_block_log_likelihood(model, records, predictions):
    indices = [("TN", "TP", "Chla").index(r["variable"]) for r in records]
    sigma = model.scales[indices] * (
        1e-4 + torch.nn.functional.softplus(model.error_raw[indices])
    )
    correlation = sigma.new_tensor(records[0]["_error_correlation"])
    covariance = correlation * sigma[:, None] * sigma[None, :]
    alternatives = []
    for record, (field, weights) in zip(records, predictions):
        if record.get("mode") == "instantaneous_time_mixture":
            alternatives.append([(field[k], weights[k]) for k in range(len(weights)) if weights[k] > 0])
        else:
            alternatives.append([((field * weights[:, None]).sum(0), field.new_tensor(1.0))])
    combinations = math.prod(len(a) for a in alternatives)
    limit = records[0].get("_time_mixture_limit", 65536)
    if combinations > limit:
        raise ValueError('E_CORRELATED_117')
    answer = None
    points = records[0].get("_rectangle_points", 4096)
    for choice in itertools.product(*alternatives):
        mean = torch.stack([part[0] for part in choice])
        log_weight = sum(torch.log(part[1]) for part in choice)
        log_probability = gaussian_observation_log_probability(mean, covariance, records, points)
        value = log_probability + log_weight
        answer = value if answer is None else torch.logaddexp(answer, value)
    return answer.mean()
