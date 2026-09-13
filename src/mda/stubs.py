"""Working stub implementations of every protocol in interfaces.py.

These exist so that (a) the contracts are demonstrably sufficient to run a
full design->observe->infer round today, and (b) each Stage B component
agent can test its real implementation against stubs of everything else.
They are intentionally the simplest correct thing, not the paper's method:

    LineWorld               <- stands in for a benchmark world
    PolynomialHypothesis    <- stands in for LLM-proposed structures
    PriorISEngine           <- stands in for adaptive-tempered SMC
    FixedPoolProposer       <- stands in for the LLM proposer
    RandomDesigner          <- stands in for VoI
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np
import sympy
from scipy.special import logsumexp

from .interfaces import (
    RNG,
    Array,
    Dataset,
    Design,
    Designer,
    GaussianNoise,
    Hypothesis,
    HypothesisState,
    InferenceEngine,
    NoiseModel,
    Observation,
    Posterior,
    Proposer,
)


# ---------------------------------------------------------------------------
# World stub: y = a*x + b + eps on a discrete grid of x
# ---------------------------------------------------------------------------

@dataclass
class LineWorld:
    """Hidden mechanism y = a*x + b, Gaussian noise, x chosen from a grid."""

    a: float = 2.0
    b: float = -1.0
    sigma: float = 0.1
    grid: tuple[float, ...] = (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0)
    name: str = "line-world"

    def __post_init__(self) -> None:
        self.noise: NoiseModel = GaussianNoise(self.sigma)

    def description(self) -> str:
        return (
            "A scalar quantity y is measured after setting a single control "
            "variable x. The mechanism relating x to y is unknown. "
            "Measurements carry independent Gaussian noise."
        )

    def design_space(self) -> Sequence[Design]:
        return [Design.from_dict({"x": v}, name=f"x={v:g}") for v in self.grid]

    def _truth(self, design: Design) -> Array:
        return np.array([self.a * design["x"] + self.b])

    def run(self, design: Design, rng: RNG) -> Observation:
        return Observation(design=design, y=self.noise.sample(self._truth(design), rng))

    def test_designs(self) -> Sequence[Design]:
        # Held-out interventions OUTSIDE the training grid on purpose:
        # interventional evaluation should include extrapolation.
        return [Design.from_dict({"x": v}, name=f"x={v:g}") for v in (-4.0, 5.0)]


# ---------------------------------------------------------------------------
# Hypothesis stub: polynomial of fixed degree, N(0, scale^2) priors
# ---------------------------------------------------------------------------

@dataclass
class PolynomialHypothesis(Hypothesis):
    """Implements Hypothesis for y(x) = sum_k theta_k * x^k.

    The coefficients have independent N(0, prior_scale^2) priors.
    Explicit inheritance makes the implemented interface visible here.
    """

    degree: int
    prior_scale: float = 3.0
    name: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            self.name = f"poly-deg{self.degree}"

    @property
    def n_params(self) -> int:
        return self.degree + 1

    @property
    def param_names(self) -> Sequence[str]:
        return [f"c{k}" for k in range(self.n_params)]

    def sample_prior(self, rng: RNG, n: int) -> Array:
        return self.prior_scale * rng.standard_normal((n, self.n_params))

    def log_prior(self, theta: Array) -> float:
        theta = np.asarray(theta, float)
        return float(
            -0.5 * np.sum((theta / self.prior_scale) ** 2)
            - self.n_params * np.log(self.prior_scale)
            - 0.5 * self.n_params * np.log(2.0 * np.pi)
        )

    def simulate(self, design: Design, theta: Array) -> Array:
        x = design["x"]
        powers = np.array([x**k for k in range(self.n_params)])
        return np.array([float(np.dot(np.asarray(theta, float), powers))])

    def sympy_form(self) -> Optional[Any]:
        x = sympy.Symbol("x")
        cs = sympy.symbols(f"c0:{self.n_params}")
        return sum(c * x**k for k, c in enumerate(cs))

    def human_description(self) -> str:
        return (
            f"A degree-{self.degree} polynomial in x: y = {self.sympy_form()}. "
            "The coefficients have independent Gaussian priors centered on zero "
            f"with standard deviation {self.prior_scale:g}. "
            "Predictions exclude observation noise."
        )


# ---------------------------------------------------------------------------
# Inference stub: importance sampling from the prior
# ---------------------------------------------------------------------------
#
# For each hypothesis: draw N thetas from the prior, weight by likelihood.
#   log_evidence = log mean_i exp(loglik_i)   (unbiased-ish for large N)
# This is exactly what tempered SMC does better; same contract, worse
# estimator. Good enough for stub worlds with few observations.

@dataclass
class ISPosterior:
    hypothesis: Hypothesis
    thetas: Array          # (N, n_params)
    log_w: Array           # (N,) normalized: logsumexp == 0
    log_evidence: float

    def sample(self, rng: RNG, n: int) -> Array:
        idx = rng.choice(len(self.thetas), size=n, p=np.exp(self.log_w))
        return self.thetas[idx]

    def predict(self, design: Design) -> tuple[Array, Array]:
        preds = np.stack([self.hypothesis.simulate(design, th) for th in self.thetas])
        w = np.exp(self.log_w)[:, None]
        mean = np.sum(w * preds, axis=0)
        var = np.sum(w * (preds - mean) ** 2, axis=0)
        return mean, var


@dataclass
class PriorISEngine:
    n_samples: int = 4000

    def fit(
        self,
        hypothesis: Hypothesis,
        data: Dataset,
        noise: NoiseModel,
        rng: RNG,
    ) -> Posterior:
        thetas = hypothesis.sample_prior(rng, self.n_samples)
        loglik = np.zeros(self.n_samples)
        for obs in data:
            loglik += np.array(
                [noise.log_likelihood(obs.y, hypothesis.simulate(obs.design, th)) for th in thetas]
            )
        log_total = logsumexp(loglik)
        return ISPosterior(
            hypothesis=hypothesis,
            thetas=thetas,
            log_w=loglik - log_total,
            log_evidence=float(log_total - np.log(self.n_samples)),
        )


# ---------------------------------------------------------------------------
# Proposer stub: hand the agent a fixed menu, once
# ---------------------------------------------------------------------------

@dataclass
class FixedPoolProposer:
    """Returns a preset list of hypotheses on the first call, then nothing.

    Mimics the LLM proposer's *interface* with zero intelligence; lets the
    loop and tests run without an API key.
    """

    hypotheses: list[Hypothesis] = field(
        default_factory=lambda: [PolynomialHypothesis(0), PolynomialHypothesis(1)]
    )
    _spent: bool = False

    def propose(
        self,
        context: str,
        data: Dataset,
        pool: Sequence[HypothesisState],
        n_new: int,
        rng: RNG,
    ) -> list[Hypothesis]:
        if self._spent:
            return []
        self._spent = True
        return list(self.hypotheses[:n_new] if n_new < len(self.hypotheses) else self.hypotheses)


# ---------------------------------------------------------------------------
# Designer stub: uniform random choice
# ---------------------------------------------------------------------------

@dataclass
class RandomDesigner:
    """The baseline VoI must beat (claim C2)."""

    def choose(
        self,
        pool: Sequence[HypothesisState],
        candidates: Sequence[Design],
        rng: RNG,
    ) -> Design:
        return candidates[int(rng.integers(len(candidates)))]
