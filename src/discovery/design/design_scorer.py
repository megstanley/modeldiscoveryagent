"""Scalar Gaussian design objectives corresponding to Murphy Appendix A.9.

Source: https://arxiv.org/html/2608.09696v4#A9
All functions score a *hypothetical* measurement from current numerical beliefs.
They never query the world, append observations, or refit on imaginary data.
The finite-pool polynomial setting is intentional, not a general SBI
implementation. Moment scores accept Gaussian, particle and neural coefficient
posteriors. Model/joint EIG still require GaussianPosterior explicitly.
Model/joint EIG are measured in nats; variance surrogates are
dimensionless; task variance reduction has squared target-output units.
"""
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.integrate import quad
from scipy.special import logsumexp
from scipy.stats import norm

from ..beliefs import Beliefs, ParameterPosterior, GaussianPosterior
from ..comparison import probability_weights, model_entropy
from ..hypotheses import PolynomialHypothesis, design_matrix
from ..noise import GaussianNoise
from .designs import Design


@dataclass(frozen=True)
class DesignScore:
    """An objective value plus named intermediate scalar quantities for inspection."""
    name: str # the string identifying the type of scoring used
    value: float
    components: dict[str, float]


def _input(design: Design) -> float:
    """Reject actions outside this implementation's finite scalar-input domain."""
    if set(design.as_dict()) != {"x"} or not np.isfinite(design["x"]):
        raise ValueError("These design scores require exactly one finite scalar input x.")
    return float(design["x"])


def _noise_variance(noise: GaussianNoise) -> float:
    """Require explicit positive Gaussian measurement noise, not predictive variance."""
    if not isinstance(noise, GaussianNoise) or not np.isfinite(noise.sigma) or noise.sigma <= 0:
        raise ValueError("These objectives require positive finite Gaussian noise SD.")
    return float(noise.sigma**2)


def _fits(beliefs: Beliefs) -> tuple[list[ParameterPosterior], np.ndarray]:
    """Require normalized model support and valid coefficient posterior moments."""
    if (beliefs.support_kind != "bayesian_model_probability"
            or beliefs.model_probabilities is None or beliefs.parameters is None):
        raise ValueError("Design scoring requires Bayesian model probabilities and parameter posteriors.")
    names, weights = probability_weights(beliefs.model_probabilities)
    if set(names) != set(beliefs.parameters):
        raise ValueError("Every probability must have exactly one matching parameter posterior.")
    fits = [beliefs.parameters[name] for name in names]
    for fit in fits:
        if not isinstance(fit, ParameterPosterior) or not isinstance(fit.hypothesis, PolynomialHypothesis):
            raise ValueError("This implementation requires polynomial parameter posteriors.")
        p = fit.hypothesis.n_params
        if (fit.mean.shape != (p,) or fit.covariance.shape != (p, p)
                or not np.all(np.isfinite(fit.mean)) or not np.all(np.isfinite(fit.covariance))
                or not np.allclose(fit.covariance, fit.covariance.T)
                or np.linalg.eigvalsh(fit.covariance).min() < -1e-10):
            raise ValueError("Parameter posterior has invalid mean or covariance.")
    return fits, weights


def predictive_components(design: Design, beliefs: Beliefs,
                          noise: GaussianNoise) -> dict[str, np.ndarray | float]:
    """Return model weights, means, coefficient variances and observed-output variances.

    The mean and variance under each model marginalize its coefficient posterior.
    Observed-output variances add measurement noise once. This function exposes
    the ingredients used by both the cheap surrogate and the EIG reference.
    """
    x = _input(design)
    noise_variance = _noise_variance(noise)
    fits, weights = _fits(beliefs)
    moments = [fit.predict(np.array([x])) for fit in fits]
    means = np.array([mu[0] for mu, _ in moments])
    variances = np.array([var[0] for _, var in moments])
    if not np.all(np.isfinite(means)) or not np.all(np.isfinite(variances)):
        raise ValueError("Candidate predictions must be finite.")
    return {"weights": weights, "means": means, "parameter_variances": variances,
            "observation_variances": variances + noise_variance,
            "noise_variance": noise_variance}


def model_disagreement(design: Design, beliefs: Beliefs, *, noise: GaussianNoise) -> DesignScore:
    """Murphy Eq. 34 scalar surrogate: between-model mean variance divided by noise variance.

    Each model's coefficient uncertainty is averaged out before comparing means.
    This is not exact model EIG: models with equal means but different predictive
    variances receive zero disagreement even when their outcomes distinguish them.
    """
    parts = predictive_components(design, beliefs, noise)
    weights, means = parts["weights"], parts["means"]
    mean = float(weights @ means)
    between = float(weights @ (means - mean)**2)
    variance = parts["noise_variance"]
    return DesignScore("model_disagreement", between / variance,
                       {"between_variance": between, "noise_variance": variance})


def joint_variance_score(design: Design, beliefs: Beliefs, *, noise: GaussianNoise) -> DesignScore:
    """Eq. 38 scalar surrogate: within-plus-between noise-free variance / noise variance.

    This targets joint model/parameter uncertainty but is not joint EIG in nats.
    Irreducible measurement noise is not itself rewarded as information.
    """
    parts = predictive_components(design, beliefs, noise)
    weights, means = parts["weights"], parts["means"]
    within = float(weights @ parts["parameter_variances"])
    between = float(weights @ (means - weights @ means)**2)
    variance = parts["noise_variance"]
    return DesignScore("joint_variance_score", (within + between) / variance,
                       {"within_variance": within, "between_variance": between,
                        "noise_variance": variance})


def model_eig(design: Design, beliefs: Beliefs, *, noise: GaussianNoise,
              tolerance: float = 1e-8) -> DesignScore:
    """Numerically integrate I(H;Y_a|D) in nats under the scalar Gaussian mixture.

    For each h integrate p(y|h,D,a) log[p(y|h,D,a)/p(y|D,a)], then average with
    current model weights. Unlike Eq. 34, this retains coefficient uncertainty
    in each model's predictive density. Integration is in standardized outcome
    coordinates z on [-10, 10]; omitted Gaussian tail mass is < 2e-23 per model.
    The reported quadrature error excludes truncation and inference error.
    This is a numerical reference, not a symbolic exact answer or a general
    likelihood-free estimator. No hypothetical observations enter the history.
    """
    if any(not isinstance(fit, GaussianPosterior) for fit in _fits(beliefs)[0]):
        raise ValueError("Gaussian model EIG cannot silently moment-match a particle posterior.")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Quadrature tolerance must be positive and finite.")
    parts = predictive_components(design, beliefs, noise)
    positive = parts["weights"] > 0
    weights = parts["weights"][positive]
    means = parts["means"][positive]
    scales = np.sqrt(parts["observation_variances"][positive])
    log_weights = np.log(weights)
    terms, errors = [], []
    for i, (mean, scale) in enumerate(zip(means, scales)):
        def integrand(z):
            y = mean + scale * z
            log_densities = norm.logpdf(y, loc=means, scale=scales)
            log_mixture = logsumexp(log_weights + log_densities)
            return float(norm.pdf(z) * (log_densities[i] - log_mixture))

        value, error = quad(integrand, -10, 10, epsabs=tolerance,
                            epsrel=tolerance, limit=200)
        terms.append(value)
        errors.append(error)
    value = float(weights @ terms)
    error = float(weights @ errors)
    entropy = model_entropy(beliefs.model_probabilities)
    if value < -max(1e-9, 5*error) or value > entropy + max(1e-9, 5*error):
        raise ArithmeticError("Model EIG violates its entropy bounds; inspect quadrature accuracy.")
    return DesignScore("model_eig", max(0.0, value), {"quadrature_error": error, "model_entropy": entropy})


def joint_eig(design: Design, beliefs: Beliefs, *, noise: GaussianNoise,
              tolerance: float = 1e-8) -> DesignScore:
    """Joint I(H,theta;Y_a|D), in nats, for the scalar linear-Gaussian mixture.

    Chain rule: model EIG + sum_h w_h I(theta;Y_a|h,D). The second term is
    analytic: 0.5 log(1 + coefficient-induced predictive variance / noise variance).
    The model term uses model_eig's numerical integration and its limitations.
    """
    parts = predictive_components(design, beliefs, noise)
    model = model_eig(design, beliefs, noise=noise, tolerance=tolerance)
    parameter = float(parts["weights"] @
                      (0.5 * np.log1p(parts["parameter_variances"] / parts["noise_variance"])))
    return DesignScore("joint_eig", model.value + parameter,
                       {"model_eig": model.value, "conditional_parameter_eig": parameter,
                        "quadrature_error": model.components["quadrature_error"]})


def task_variance_reduction(design: Design, beliefs: Beliefs, *, noise: GaussianNoise,
                             queries: Sequence[Design], query_weights=None) -> DesignScore:
    """Eq. 43 for forecasting the scalar noise-free output at specified query inputs.

    Average Cov(f_q,f_a|D)^2 / [Var(f_a|D) + sigma^2] over the supplied query
    distribution. Covariance includes paired within-model parameter covariance
    and between-model mean covariance. Query settings/weights are public task
    specifications, never held-out observed answers.

    Exact expected squared-error risk reduction for a single Gaussian model;
    a linear-Gaussian moment surrogate for a model mixture, not its exact VoI.
    Return units are squared output units. General target functionals, vector
    observations and action-dependent costs are outside this first implementation.
    """
    parts = predictive_components(design, beliefs, noise)
    fits, weights = _fits(beliefs)
    if not queries:
        raise ValueError("Task-aware scoring requires at least one query design.")
    qxs = np.array([_input(query) for query in queries])
    qweights = (np.ones(len(queries)) / len(queries) if query_weights is None
                else np.asarray(query_weights, float))
    if (qweights.shape != (len(queries),) or not np.all(np.isfinite(qweights))
            or np.any(qweights < 0) or not np.isclose(qweights.sum(), 1.0)):
        raise ValueError("Query weights must be non-negative, finite, normalized, and match queries.")
    qmeans, within_covariances = [], []
    for fit in fits:
        Xq = design_matrix(fit.hypothesis, qxs)
        va = design_matrix(fit.hypothesis, [_input(design)])[0]
        qmeans.append(Xq @ fit.mean)
        within_covariances.append(Xq @ fit.covariance @ va)
    qmeans = np.asarray(qmeans)
    means = parts["means"]
    between_covariance = weights @ ((qmeans - weights @ qmeans) * (means - weights @ means)[:, None])
    covariance = weights @ np.asarray(within_covariances) + between_covariance
    candidate_variance = float(weights @ (parts["parameter_variances"] + (means - weights @ means)**2))
    denominator = candidate_variance + parts["noise_variance"]
    reduction = float(qweights @ covariance**2 / denominator)
    return DesignScore("task_variance_reduction", reduction,
                       {"weighted_squared_covariance": float(qweights @ covariance**2),
                        "candidate_observation_variance": denominator,
                        "single_model_gaussian_exact": float(
                            np.count_nonzero(weights) == 1
                            and isinstance(fits[int(np.argmax(weights))], GaussianPosterior))})
