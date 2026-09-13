"""Exact reference inference.

Parameters and evidence are computed separately. Each update refits from the
prior on the complete observed history; it does not increment an old posterior.
"""
from dataclasses import dataclass
from typing import Sequence
import numpy as np
from scipy.special import logsumexp
from ..records import Array, Dataset, observed_arrays
from ..noise import NoiseModel
from ..hypotheses import Hypothesis, PolynomialHypothesis, design_matrix
from ..beliefs import GaussianPosterior, Beliefs, Updater
from ..programs import observation_hypothesis


def dataset_log_likelihood(
    hypothesis: Hypothesis, theta: Array, data: Dataset, noise: NoiseModel
) -> float:
    """Score fixed observations under the means predicted by one coefficient vector."""
    return float(sum(
        noise.log_likelihood(obs.y, hypothesis.simulate(obs.design, theta))
        for obs in data
    ))


def fit_parameters(
    hypothesis: PolynomialHypothesis, data: Dataset, sigma: float
) -> GaussianPosterior:
    """Refit from this model's prior using the full current observed dataset."""
    if not isinstance(hypothesis, PolynomialHypothesis):
        raise ValueError("Exact Gaussian inference requires PolynomialHypothesis.")
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("Noise SD must be positive and finite.")
    tau = hypothesis.prior_scale
    if not np.isfinite(tau) or tau <= 0:
        raise ValueError("Coefficient-prior SD must be positive and finite.")
    inputs, outputs = observed_arrays(data)
    X = design_matrix(hypothesis, inputs)
    identity = np.eye(hypothesis.n_params)
    A = identity/tau**2 + X.T @ X/sigma**2
    q = X.T @ outputs/sigma**2
    covariance = np.linalg.solve(A, identity)
    mean = np.linalg.solve(A, q)
    return GaussianPosterior(observation_hypothesis(hypothesis, sigma), mean, covariance)


def log_model_evidence(
    hypothesis: PolynomialHypothesis, data: Dataset, sigma: float
) -> float:
    """Integrate likelihood over the coefficient prior, not the fitted posterior."""
    if not isinstance(hypothesis, PolynomialHypothesis):
        raise ValueError("Exact Gaussian evidence requires PolynomialHypothesis.")
    tau = hypothesis.prior_scale
    if not np.all(np.isfinite([sigma, tau])) or sigma <= 0 or tau <= 0:
        raise ValueError("Noise and coefficient-prior SDs must be positive and finite.")
    inputs, outputs = observed_arrays(data)
    n = len(outputs)
    if n == 0:
        return 0.0  # With no observations, evidence is 1 and log evidence is 0.
    X = design_matrix(hypothesis, inputs)
    C = sigma**2*np.eye(n) + tau**2*(X @ X.T)
    sign, log_determinant = np.linalg.slogdet(C)
    if sign <= 0:
        raise ValueError("The prior-predictive covariance must be positive definite.")
    quadratic_term = outputs @ np.linalg.solve(C, outputs)
    return float(-0.5*(n*np.log(2*np.pi) + log_determinant + quadratic_term))


@dataclass
class ExactGaussianUpdater(Updater):
    """Exact polynomial inference with known Gaussian noise and equal model priors."""
    sigma: float

    def update(self, data: Dataset, hypotheses: Sequence[PolynomialHypothesis]) -> Beliefs:
        """Fit each model and normalize its evidence within this candidate pool."""
        names = [h.name for h in hypotheses]
        if not names or len(set(names)) != len(names):
            raise ValueError("Supply a non-empty pool with unique candidate names.")
        # This performs the exact inference to get parameter posteriors
        fits = {h.name: fit_parameters(h, data, self.sigma) for h in hypotheses}
        # This computes the log evidence of each hypothesis in the set
        scores = {h.name: log_model_evidence(h, data, self.sigma) for h in hypotheses}
        if not np.all(np.isfinite(list(scores.values()))):
            raise ValueError("Log evidences must be finite.")
        normalizer = logsumexp(list(scores.values()))
        probabilities = {name: float(np.exp(score-normalizer))
                         for name, score in scores.items()}
        return Beliefs(
            method="exact_gaussian",
            assessment="Relative support in this pool, not proof that any candidate is adequate.",
            parameters=fits,
            model_probabilities=probabilities,
            log_evidences=scores,
            support_kind="bayesian_model_probability",
        )
