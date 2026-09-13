"""Adapters for reviewed simulator functions, never execution of generated source.

A program simulates an entire ordered dataset in one call. It owns all internal
randomness and observation noise. The adapter does not assume independent
observations and never adds noise a second time. Current neural consumers need
fixed output shapes and a fixed ordered design list.
"""
from dataclasses import dataclass, replace
from typing import Callable, Sequence
import hashlib
import json
import numpy as np

from .hypotheses import Hypothesis, PolynomialHypothesis
from .noise import GaussianNoise
from .records import Array, RNG
from .design.designs import Design


@dataclass(frozen=True)
class ParameterSpec:
    """One independent scalar prior: normal(mean, SD) or uniform(lower, upper)."""
    name: str
    distribution: str
    first: float
    second: float
    unit: str = "dimensionless"

    def __post_init__(self):
        if (not isinstance(self.name, str) or not self.name.strip() or not isinstance(self.unit, str)
                or not self.unit.strip() or not np.all(np.isfinite([self.first, self.second]))):
            raise ValueError("Parameter name, unit and finite prior arguments are required.")
        if self.distribution == "normal":
            valid = self.second > 0
        elif self.distribution == "uniform":
            valid = self.second > self.first
        else:
            raise ValueError("Supported priors are normal and uniform.")
        if not valid:
            raise ValueError("Invalid prior SD or interval.")

    def sample(self, rng: RNG, n: int) -> Array:
        """Sample the declared scalar prior, not a fitted parameter distribution."""
        if self.distribution == "normal":
            return rng.normal(self.first, self.second, n)
        return rng.uniform(self.first, self.second, n)

    def log_density(self, value: float) -> float:
        """Normalized prior density, returning -inf outside its support."""
        if not np.isfinite(value):
            return -np.inf
        if self.distribution == "normal":
            return float(-.5*((value-self.first)/self.second)**2
                         - np.log(self.second*np.sqrt(2*np.pi)))
        return -float(np.log(self.second-self.first)) if self.first <= value <= self.second else -np.inf

    def as_dict(self) -> dict:
        """Explicit prior metadata suitable for a proposal request or response."""
        return dict(name=self.name, distribution=self.distribution, first=self.first,
                    second=self.second, unit=self.unit)


@dataclass
class ProgramHypothesis(Hypothesis):
    """A reviewed callable plus an explicit scientific specification.

    simulator(designs, named_parameters, rng) returns (n_designs, *output_shape).
    program_id is a versioned registry ID: change it when implementation changes.
    Metadata cannot verify that a trusted callable honours its declared semantics.
    A likelihood is optional; simulation alone never implies one is available.
    """
    name: str
    program_id: str
    parameters: tuple[ParameterSpec, ...]
    simulator: Callable
    description: str
    input_names: tuple[str, ...] = ("x",)
    output_shape: tuple[int, ...] = (1,)
    likelihood: Callable | None = None
    noise_sigma: float | None = None

    def __post_init__(self):
        self.parameters = tuple(self.parameters)
        self.input_names, self.output_shape = tuple(self.input_names), tuple(self.output_shape)
        if (not all(isinstance(s, str) and s.strip() for s in (self.name, self.program_id, self.description))
                or not callable(self.simulator)):
            raise ValueError("A named, described, versioned trusted simulator is required.")
        if not self.parameters or any(not isinstance(p, ParameterSpec) for p in self.parameters):
            raise ValueError("Supply explicit ParameterSpec priors.")
        if len(set(self.param_names)) != self.n_params:
            raise ValueError("Parameter names must be unique.")
        if (not self.input_names or any(not isinstance(s, str) or not s for s in self.input_names)
                or len(set(self.input_names)) != len(self.input_names)):
            raise ValueError("Input names must be nonempty and unique.")
        if not self.output_shape or any(isinstance(s, bool) or not isinstance(s, int) or s < 1 for s in self.output_shape):
            raise ValueError("Output shape must contain positive integer dimensions.")
        if self.likelihood is not None and not callable(self.likelihood):
            raise ValueError("Likelihood must be a trusted callable or None.")
        if self.noise_sigma is not None and (not np.isfinite(self.noise_sigma) or self.noise_sigma <= 0):
            raise ValueError("Declared noise SD must be positive and finite.")

    @property
    def n_params(self) -> int:
        return len(self.parameters)

    @property
    def param_names(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.parameters)

    def sample_prior(self, rng: RNG, n: int) -> Array:
        """Return (n, n_params) independent prior draws."""
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            raise ValueError("Prior sample count must be a positive integer.")
        return np.column_stack([p.sample(rng, n) for p in self.parameters])

    def log_prior(self, theta: Array) -> float:
        """Evaluate the declared independent prior without likelihood information."""
        theta = np.asarray(theta, float)
        if theta.shape != (self.n_params,):
            raise ValueError("Parameter vector has the wrong shape.")
        return float(sum(p.log_density(v) for p, v in zip(self.parameters, theta)))

    def sample_observations(self, designs: Sequence[Design], theta: Array, rng: RNG) -> Array:
        """Generate one dataset, rejecting invalid outputs or out-of-prior parameters."""
        designs = tuple(designs)
        if any(set(d.as_dict()) != set(self.input_names)
               or not np.all(np.isfinite(list(d.as_dict().values()))) for d in designs):
            raise ValueError(f"Program requires finite settings {self.input_names}.")
        if not np.isfinite(self.log_prior(theta)):
            raise ValueError("Parameter draw is outside the declared prior support; it is not clipped or resampled.")
        values = self.simulator(designs, dict(zip(self.param_names, map(float, theta))), rng)
        output = np.asarray(values, float)
        if output.shape != (len(designs), *self.output_shape) or not np.all(np.isfinite(output)):
            raise ValueError("Program returned nonfinite observations or the wrong output shape.")
        return output.copy()

    def log_likelihood(self, data, theta: Array) -> float:
        """Use the declared observation likelihood, or fail rather than invent one."""
        if self.likelihood is None:
            raise NotImplementedError("This program supplies simulation, not a likelihood.")
        if not np.isfinite(self.log_prior(theta)):
            return -np.inf
        value = float(self.likelihood(data, dict(zip(self.param_names, map(float, theta)))))
        if np.isnan(value) or value == np.inf:
            raise ValueError("Likelihood must return a finite value or -inf.")
        return value

    def simulate(self, design, theta):
        """The polynomial noise-free interface is deliberately unsupported here."""
        raise NotImplementedError("A stochastic program has no generic noise-free prediction; use sample_observations.")

    def sympy_form(self):
        return None

    def human_description(self):
        return self.description

    def specification(self) -> dict:
        """Registry version, priors and measurement metadata define candidate identity."""
        return dict(kind="program", program_id=self.program_id,
                    parameters=[p.as_dict() for p in self.parameters],
                    input_names=list(self.input_names), output_shape=list(self.output_shape),
                    noise_sigma=self.noise_sigma)


def hypothesis_fingerprint(hypothesis: Hypothesis) -> str:
    """Deduplicate declared scientific definitions, not human-readable names."""
    encoded = json.dumps(hypothesis.specification(), sort_keys=True, allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def as_designs(inputs) -> tuple[Design, ...]:
    """Accept explicit designs; preserve old notebook calls using scalar x arrays."""
    values = tuple(inputs)
    if not values:
        raise ValueError("At least one experimental setting is required.")
    if all(isinstance(d, Design) for d in values):
        return values
    numbers = np.asarray(values, float)
    if numbers.ndim != 1 or not np.all(np.isfinite(numbers)):
        raise ValueError("Supply Design records or a finite one-dimensional x list.")
    return tuple(Design.from_dict({"x": float(x)}) for x in numbers)


def observation_hypothesis(hypothesis: Hypothesis, sigma: float | None) -> Hypothesis:
    """Bind legacy polynomial noise arguments; never overwrite a program's noise."""
    if sigma is None:
        return hypothesis
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("Noise SD must be positive and finite.")
    if isinstance(hypothesis, PolynomialHypothesis):
        return replace(hypothesis, noise=GaussianNoise(float(sigma)))
    if getattr(hypothesis, "noise_sigma", None) != sigma:
        raise ValueError("Program owns its observation noise; pass sigma=None or its declared noise SD.")
    return hypothesis


def exponential_decay(name="decay", *, sigma=0.05, lower=0.1, upper=2.0) -> ProgramHypothesis:
    """Reviewed y=exp(-k*x)+Gaussian noise; x is time, k has inverse-time units."""
    noise = GaussianNoise(sigma)

    def simulator(designs, parameters, rng):
        times = np.array([d["x"] for d in designs])
        if np.any(times < 0):
            raise ValueError("Decay measurement times must be nonnegative.")
        return noise.sample(np.exp(-parameters["k"]*times), rng)[:, None]

    def likelihood(data, parameters):
        if any(set(o.design.as_dict()) != {"x"} or not np.isfinite(o.design["x"]) or o.design["x"] < 0
               or np.shape(o.y) != (1,) or not np.all(np.isfinite(o.y)) for o in data):
            raise ValueError("Decay likelihood requires scalar observations at nonnegative times.")
        return sum(noise.log_likelihood(o.y, np.array([np.exp(-parameters["k"]*o.design["x"])]))
                   for o in data)

    return ProgramHypothesis(name, "exponential-decay-v1",
                             (ParameterSpec("k", "uniform", lower, upper, "1/time"),),
                             simulator, "Unit-amplitude exponential decay with known Gaussian observation noise; x is time.",
                             likelihood=likelihood, noise_sigma=sigma)
