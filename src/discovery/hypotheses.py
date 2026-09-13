"""Candidate definitions; fitted coefficients are stored separately in beliefs."""
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, Sequence, runtime_checkable
import numpy as np
import sympy
from .records import Array, RNG
from .design.designs import Design
from .noise import GaussianNoise


@runtime_checkable
class Hypothesis(Protocol):
    """One candidate structure with free parameters theta.

    theta is always a flat float vector of length `n_params`; structure
    lives in code/symbolic form, parameters in theta, for now.

    This is meant to be a completely general way to represent any model of the world.

    In practice, this might be nice to extend to the concept of "meta-hypothesis",
    which is a language explanation with consequences for underlying model structures,
    without specifying precise the model. But that's a TODO for me :)
    """

    name: str

    @property
    def n_params(self) -> int: ...

    @property
    def param_names(self) -> Sequence[str]: ...

    def sample_prior(self, rng: RNG, n: int) -> Array:
        """(n, n_params) draws from the parameter prior."""
        ...

    def sample_observations(self, designs: Sequence[Design], theta: Array, rng: RNG) -> Array:
        """One synthetic dataset, including observation noise; shape (designs, *output_shape)."""
        ...

    def specification(self) -> dict:
        """Serializable scientific definition, excluding fitted values and display name."""
        ...

    def log_prior(self, theta: Array) -> float:
        """Log prior density of one theta vector."""
        ...

    def simulate(self, design: Design, theta: Array) -> Array:
        """Optional legacy noise-free prediction under parameters theta.

        General stochastic programs may raise NotImplementedError here. Their
        shared generative interface is sample_observations, including noise.

        Must return the same shape as Observations for the target world.
        This might be expensive and we should track the cost of these calls.

        The expense of course depends on the world being tested.
        """
        ...

    def sympy_form(self) -> Optional[Any]:
        """Symbolic form (sympy.Expr) if the model is a closed-form law,
        else None. Used by eval for symbolic-equivalence scoring
        (eg. ChemBench's sympy check); simulators (ODE worlds) return None.
        """
        ...

    def human_description(self) -> Optional[str]:
        """A human-readable description, or None when one is unavailable.

        Implementations must provide this method even when sympy_form exists.
        It can explain the model's variables and assumptions beyond its formula.
        """
        ...


@dataclass
class PolynomialHypothesis(Hypothesis):
    """Implements Hypothesis for y(x) = sum_k theta_k * x^k.

    The coefficients have independent N(0, prior_scale^2) priors.
    Explicit inheritance makes the implemented interface visible here.
    """

    degree: int
    prior_scale: float = 3.0
    name: str = ""
    noise: GaussianNoise = field(default_factory=lambda: GaussianNoise(0.2))

    def __post_init__(self) -> None:
        if isinstance(self.degree, bool) or not isinstance(self.degree, int) or self.degree < 0:
            raise ValueError("Polynomial degree must be a nonnegative integer.")
        if not np.isfinite(self.prior_scale) or self.prior_scale <= 0:
            raise ValueError("Coefficient prior scale must be positive and finite.")
        if not isinstance(self.noise, GaussianNoise) or not np.isfinite(self.noise.sigma) or self.noise.sigma <= 0:
            raise ValueError("Polynomial observations require positive Gaussian noise SD.")
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

    def sample_observations(self, designs: Sequence[Design], theta: Array, rng: RNG) -> Array:
        """Sample scalar observations; the existing simulate method remains noise-free."""
        theta = np.asarray(theta, float)
        if theta.shape != (self.n_params,) or not np.all(np.isfinite(theta)):
            raise ValueError("Expected one finite coefficient vector.")
        if any(set(d.as_dict()) != {"x"} or not np.isfinite(d["x"]) for d in designs):
            raise ValueError("Polynomial designs require one finite scalar x.")
        means = design_matrix(self, [d["x"] for d in designs]) @ theta
        return self.noise.sample(means, rng)[:, None]

    def specification(self) -> dict:
        """Include the coefficient prior and observation model in candidate identity."""
        return {"kind": "polynomial", "degree": self.degree,
                "prior_scale": self.prior_scale, "noise_sigma": self.noise.sigma}

    def sympy_form(self) -> Optional[Any]:
        x = sympy.Symbol("x")
        cs = sympy.symbols(f"c0:{self.n_params}")
        return sum(c * x**k for k, c in enumerate(cs))

    def human_description(self) -> str:
        return (
            f"A degree-{self.degree} polynomial in x: y = {self.sympy_form()}. "
            "The coefficients have independent Gaussian priors centered on zero "
            f"with standard deviation {self.prior_scale:g}. "
            f"simulate gives noise-free predictions; sample_observations adds Gaussian noise with SD {self.noise.sigma:g}."
        )


def design_matrix(hypothesis: PolynomialHypothesis, inputs: Array) -> Array:
    """Return one row per input and one column per polynomial coefficient."""
    if not isinstance(hypothesis, PolynomialHypothesis):
        raise ValueError("This analytic shortcut requires PolynomialHypothesis; use simulated predictions for programs.")
    return np.vander(np.asarray(inputs, float),
                     N=hypothesis.n_params, increasing=True)
