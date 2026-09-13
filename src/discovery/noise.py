"""Observation noise: density of observations conditional on predictions."""
from dataclasses import dataclass
from typing import Protocol, runtime_checkable
import numpy as np
from .records import Array, RNG


@runtime_checkable
class NoiseModel(Protocol):
    """Observation model linking noise-free hypothesis simulation output to data.

    Kept separate from Hypothesis so the same structure can be scored under
    different observation models (deterministic Gaussian now; Poisson spike
    counts or learned-summary-statistic likelihoods later, like NeuronBench setup).
    """

    def log_likelihood(self, y_obs: Array, y_pred: Array) -> float:
        """Log p(y_obs | y_pred)."""
        ...

    def sample(self, y_pred: Array, rng: RNG) -> Array:
        """Draw a noisy observation around the noise-free prediction."""
        ...


@dataclass(frozen=True)
class GaussianNoise(NoiseModel):
    """Isotropic Gaussian observation noise.

    Can compute the density of the observation given the prediction, 
    or sample an observation conditional on a noise-free predicted mean.
    """

    sigma: float

    def log_likelihood(self, y_obs: Array, y_pred: Array) -> float:
        r = np.asarray(y_obs, float) - np.asarray(y_pred, float)
        n = r.size
        return float(
            -0.5 * np.sum(r**2) / self.sigma**2
            - n * np.log(self.sigma)
            - 0.5 * n * np.log(2.0 * np.pi)
        )

    def sample(self, y_pred: Array, rng: RNG) -> Array:
        return np.asarray(y_pred, float) + self.sigma * rng.standard_normal(
            np.shape(y_pred)
        )

