"""General posterior predictive simulation, separate from noise-free shortcuts."""
import numpy as np
from .beliefs import ParameterPosterior, ParticlePosterior
from .programs import as_designs


def weighted_deterministic_prediction(posterior: ParticlePosterior, designs, *, predict=None, ledger=None):
    """Noise-free mean and marginal variance under an existing weighted particle set.

    predict(design, theta) must return a deterministic output array. By default
    use hypothesis.simulate; stochastic-only programs must supply a separate,
    reviewed deterministic function or use sample_posterior_predictive instead.
    Outputs have shape (n_designs, *output_shape). No resampling, extra noise or
    Gaussian approximation is involved; no covariance across outputs is returned.
    """
    if not isinstance(posterior, ParticlePosterior):
        raise TypeError("Weighted prediction requires a ParticlePosterior.")
    designs = as_designs(designs)
    predict = posterior.hypothesis.simulate if predict is None else predict
    positive = posterior.weights > 0
    predictions = []
    for theta in posterior.particles[positive]:
        row = []
        for design in designs:
            if ledger is not None:
                ledger.add_sim()
            value = np.asarray(predict(design, theta), float)
            if value.ndim < 1 or not value.size or not np.all(np.isfinite(value)):
                raise ValueError("Deterministic predictions must be finite nonempty arrays.")
            row.append(value)
        predictions.append(np.stack(row))
    values = np.stack(predictions)
    weights = posterior.weights[positive]
    mean = np.tensordot(weights, values, axes=(0, 0))
    variance = np.tensordot(weights, (values-mean)**2, axes=(0, 0))
    return mean, np.maximum(variance, 0.)


def sample_posterior_predictive(posterior: ParameterPosterior, designs, rng, *, n=1000):
    """Return (n, designs, *output_shape) draws including simulator/measurement noise.

    One parameter vector is shared across each whole simulated dataset. Bounded
    prior violations from an approximate posterior are errors, not silently
    clipped or rejection-sampled draws from a different posterior.
    """
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError("Predictive sample count must be a positive integer.")
    designs = as_designs(designs)
    theta = np.asarray(posterior.sample(rng, n), float)
    if theta.shape != (n, posterior.hypothesis.n_params) or not np.all(np.isfinite(theta)):
        raise ValueError("Posterior returned malformed parameter samples.")
    draws = [np.asarray(posterior.hypothesis.sample_observations(designs, t, rng), float)
             for t in theta]
    shape = draws[0].shape
    if (len(shape) < 2 or shape[0] != len(designs)
            or any(d.shape != shape or not np.all(np.isfinite(d)) for d in draws)):
        raise ValueError("Predictive simulations must have matching finite observation layouts.")
    return np.stack(draws)


def prior_support_violation_rate(posterior: ParameterPosterior, rng, *, n=1000) -> float:
    """Fraction of posterior draws outside prior support; not a coverage statistic."""
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError("Diagnostic sample count must be a positive integer.")
    theta = np.asarray(posterior.sample(rng, n), float)
    if theta.shape != (n, posterior.hypothesis.n_params):
        raise ValueError("Posterior returned malformed parameter samples.")
    return float(np.mean([not np.isfinite(posterior.hypothesis.log_prior(t)) for t in theta]))
