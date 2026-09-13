"""Here we have all the classes that record Beliefs (model selection evidence, posteriors etc) over the Hypotheses,
and keep also the posteriors over the parameters within those hypotheses. This file keeps the base classes, but specific 
inference choices for instance are in separate files."""
from dataclasses import dataclass
from typing import Optional, Protocol, Sequence
from abc import ABC, abstractmethod
import numpy as np
from .records import Array, Dataset
from .hypotheses import Hypothesis, PolynomialHypothesis, design_matrix


class ParameterPosterior(ABC):
    """Class to hold the posterior over parameters for a single hypothesis.

    This assumes such a posterior can be exactly, or approximately, inferred.

    The optional predict shortcut gives outputs of a model as if there were just
    noise-free polynomial moments rather 
    than sampling observations from the observation simulator of the Hypothesis, 
    which may also add observation noise.
    
    In general if a closed form posterior does not exist, models sample from 
    the simulator of a Hypothesis.
    Samples have shape (n, hypothesis.n_params). This deliberately does not
    require density evaluation: a weighted particle measure has no smooth PDF.
    """
    hypothesis: Hypothesis
    mean: Array
    covariance: Array

    @abstractmethod
    def sample(self, rng: np.random.Generator, n: int) -> Array: ...

    def predict(self, inputs: Array) -> tuple[Array, Array]:
        """Optional noise-free shortcut; general programs use predictive sampling."""
        raise NotImplementedError("Use sample_posterior_predictive for general programs.")


@dataclass
class GaussianPosterior(ParameterPosterior):
    """Gaussian coefficient posterior, where one fixed hypothesis is assumed."""
    hypothesis: PolynomialHypothesis
    mean: Array
    covariance: Array

    def sample(self, rng: np.random.Generator, n: int) -> Array:
        """Draw coefficient vectors from this Gaussian posterior."""
        return rng.multivariate_normal(self.mean, self.covariance, size=n)

    def predict(self, inputs: Array) -> tuple[Array, Array]:
        """Return noise-free predicted output at each requested input.
        Currently only works for the PolynomialHypothesis for pedagogical purposes
        """
        if not isinstance(self.hypothesis, PolynomialHypothesis):
            raise ValueError("Currently GaussianPosterior direct prediction of noise-free observations only supports a PolynomialHypothesis.")
        X = design_matrix(self.hypothesis, inputs)
        mean = X @ self.mean
        variance = np.sum((X @ self.covariance) * X, axis=1)
        return mean, np.maximum(variance, 0.0)


@dataclass
class ParticlePosterior(ParameterPosterior):
    """A posterior found using Particles (eg. output of SMC or Importance Sampling).

    This represents the uncertainty over parameters of a fixed `Hypothesis` that importantly, 
    actually has parameters.

    Covariance uses normalized probability weights, not the unbiased sample
    covariance correction. It is the covariance of this discrete distribution.
    """
    hypothesis: Hypothesis
    particles: Array
    weights: Array

    def __post_init__(self) -> None:
        self.particles = np.asarray(self.particles, float).copy()
        self.weights = np.asarray(self.weights, float).copy()
        if (self.particles.ndim != 2 or len(self.particles) == 0
                or self.particles.shape[1] != self.hypothesis.n_params
                or self.weights.shape != (len(self.particles),)
                or not np.all(np.isfinite(self.particles))
                or not np.all(np.isfinite(self.weights)) or np.any(self.weights < 0)
                or not np.isclose(self.weights.sum(), 1.0)):
            raise ValueError("Particles must be finite and weights nonnegative and normalized.")

    @property
    def mean(self) -> Array:
        return self.weights @ self.particles

    @property
    def covariance(self) -> Array:
        centered = self.particles - self.mean
        return (centered.T * self.weights) @ centered

    def sample(self, rng: np.random.Generator, n: int) -> Array:
        """Draw with replacement according to the stored posterior weights."""
        return self.particles[rng.choice(len(self.weights), n, p=self.weights)].copy()

    def predict(self, inputs: Array) -> tuple[Array, Array]:
        """Exact moments of polynomial predictions under the particle measure,
        Due to the use of `design_matrix` here this will fail for non PolynomialHypothesis.

        But that is ok, in those cases the predictions *must* be simulated.
        """
        X = design_matrix(self.hypothesis, inputs)
        return X @ self.mean, np.maximum(np.sum((X @ self.covariance) * X, axis=1), 0.0)


@dataclass
class GaussianNeuralPosterior(ParameterPosterior):
    """A neural density evaluated at one observed dataset. 

    This is Gaussian and conditioned on the observational data. 
    It is trained on simulated outputs resulting from a number of
    parameter draws from the prior.

    The trained estimator lives in inference/neural.py; this record holds its
    observation-conditioned output to be held by Beliefs. This first neural family
    is Gaussian, not a flow or a claim that general neural posteriors are Gaussian.
    
    """
    hypothesis: Hypothesis
    mean: Array
    covariance: Array

    def sample(self, rng: np.random.Generator, n: int) -> Array:
        """Sample the learned conditional Gaussian using the caller's RNG."""
        return rng.multivariate_normal(self.mean, self.covariance, size=n)

    def predict(self, inputs: Array) -> tuple[Array, Array]:
        """Noise-free polynomial predictions given the learned density. 
        This will fail if Hypothesis is not Polynomial"""
        X = design_matrix(self.hypothesis, inputs)
        return X @ self.mean, np.maximum(np.sum((X @ self.covariance) * X, axis=1), 0.0)


@dataclass
class Beliefs:
    """An assessment of a pool of Hypotheses, with optional numerical support, 
    deliberately left open so that an LLM-agent can write assessments
    in a way that it finds useful."""
    method: str
    assessment: str
    parameters: Optional[dict[str, ParameterPosterior]] = None
    model_probabilities: Optional[dict[str, float]] = None
    log_evidences: Optional[dict[str, float]] = None
    support_kind: Optional[str] = None


class Updater(Protocol):
    """Assess a supplied candidate pool using the complete observed history, 
    updating posteriors over both parameters and the hypothesis set if possible."""
    def update(self, data: Dataset, hypotheses: Sequence[Hypothesis]) -> Beliefs:
        ...


def predictive_moments(state: Beliefs, inputs: Array) -> dict[str, Array]:
    """Copmuting model posterior moments to allow for model selection. 
    
    Only works if the model pool has been computed in a fully Bayesian fashion
    """
    if (state.support_kind != "bayesian_model_probability"
            or state.parameters is None or state.model_probabilities is None):
        raise ValueError("Numerical predictions require model probabilities and coefficient posteriors.")
    names = list(state.model_probabilities)
    weights = np.array([state.model_probabilities[name] for name in names])
    moments = [state.parameters[name].predict(inputs) for name in names]
    means = np.array([item[0] for item in moments])
    variances = np.array([item[1] for item in moments])
    mean = weights @ means
    within = weights @ variances
    between = weights @ (means-mean)**2
    return {"mean": mean, "within": within, "between": between, "variance": within+between}
