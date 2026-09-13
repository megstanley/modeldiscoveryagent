"""Compare models using existing observations; no proposed experiments here.

Inference supplies log evidences. Bayes factors compare those evidences, while
posterior probabilities also incorporate a model prior. Entropy and effective
sample size describe pool concentration, not absolute model adequacy.
"""
from collections.abc import Mapping

import numpy as np
from scipy.special import logsumexp


def probability_weights(probabilities: Mapping[str, float]) -> tuple[list[str], np.ndarray]:
    """Validate a non-empty probability dictionary."""
    names = list(probabilities)
    weights = np.array([probabilities[name] for name in names], dtype=float)
    if (not names or not np.all(np.isfinite(weights)) or np.any(weights < 0)
            or not np.isclose(weights.sum(), 1.0, rtol=1e-10, atol=1e-12)):
        raise ValueError("Model probabilities must be finite, non-negative, and sum to one.")
    return names, weights


def log_bayes_factor(log_evidences: Mapping[str, float], numerator: str, denominator: str) -> float:
    """Return log p(D|numerator) - log p(D|denominator), i.e. subtract the evidences, 
    excluding model prior odds."""
    values = np.array([log_evidences[numerator], log_evidences[denominator]], float)
    if not np.all(np.isfinite(values)):
        raise ValueError("This comparison requires finite log evidences.")
    return float(values[0] - values[1])


def model_probabilities(log_evidences: Mapping[str, float],
                        model_priors: Mapping[str, float] | None = None) -> dict[str, float]:
    """Computing actual prior * evidence, for the model with a default of uniform prior over models.

    Model priors are distinct from coefficient priors already integrated into
    the evidence for each model (given the dataset used).
      Zero model prior weight is allowed and remains zero.
    """
    names = list(log_evidences)
    scores = np.array([log_evidences[name] for name in names], float)
    if not names or not np.all(np.isfinite(scores)):
        raise ValueError("Supply a non-empty dictionary of finite log evidences.")
    if model_priors is None:
        prior = np.ones(len(names)) / len(names)
    else:
        probability_weights(model_priors)
        if set(model_priors) != set(names):
            raise ValueError("The model prior and evidence must name the same candidates.")
        prior = np.array([model_priors[name] for name in names])
    log_prior = np.full(len(names), -np.inf)
    np.log(prior, out=log_prior, where=prior > 0)
    log_weights = scores + log_prior
    weights = np.exp(log_weights - logsumexp(log_weights))
    return dict(zip(names, map(float, weights)))


def model_entropy(probabilities: Mapping[str, float]) -> float:
    """Return -sum w log(w), in nats; zero for a singleton regardless of fit."""
    _, weights = probability_weights(probabilities)
    positive = weights[weights > 0]
    return float(-np.sum(positive * np.log(positive)))


def effective_model_count(probabilities: Mapping[str, float]) -> float:
    """Return 1/sum(w**2), the weight ESS; not the number of correct mechanisms."""
    _, weights = probability_weights(probabilities)
    return float(1 / np.sum(weights**2))
