"""The two-control pendulum scenario preserved from the original walkthrough.

The world's period uses the leading amplitude correction, not the exact
large-angle pendulum solution. Candidates are hand-written teaching fixtures,
not LLM discoveries. The original notebook remains a local historical example.
"""
from dataclasses import dataclass
import numpy as np

from .worlds import World
from ..design.designs import Design
from ..records import Observation
from ..noise import GaussianNoise
from ..programs import ParameterSpec, ProgramHypothesis


def _settings(design):
    """Read length in metres and release angle in radians."""
    if set(design.as_dict()) != {"L", "theta0"}:
        raise ValueError("Pendulum designs require L and theta0.")
    length, angle = design["L"], design["theta0"]
    if not np.all(np.isfinite([length, angle])) or length <= 0 or not 0 <= angle < np.pi:
        raise ValueError("Use positive length and a release angle in [0, pi).")
    return length, angle


@dataclass
class PendulumWorld(World):
    """Measure period after choosing length and amplitude; gravity is fixed at 9.81."""
    sigma: float = .05
    lengths: tuple = (.25, .5, 1., 1.5, 2.)
    amplitudes: tuple = (.15, .7, 1.3)
    name: str = "pendulum"

    def __post_init__(self):
        if not np.isfinite(self.sigma) or self.sigma <= 0:
            raise ValueError("Observation noise SD must be positive and finite.")
        if not self.lengths or not self.amplitudes:
            raise ValueError("The experiment menu must be nonempty.")
        self.noise = GaussianNoise(self.sigma)
        for design in self.design_space():
            _settings(design)

    def description(self):
        """Context without supplying the hidden period formula."""
        return ("Set rod length L (metres) and release angle theta0 (radians), "
                "then measure oscillation period T (seconds). Measurement noise is Gaussian.")

    def design_space(self):
        """The fifteen allowed settings in the original example by default."""
        return [Design.from_dict({"L": length, "theta0": angle}, name=f"L={length:g} th0={angle:g}")
                for length in self.lengths for angle in self.amplitudes]

    def _truth(self, design):
        """Hidden mechanism for generating data and held-out evaluation only."""
        length, angle = _settings(design)
        return np.array([2*np.pi*np.sqrt(length/9.81)*(1+angle**2/16)])

    def run(self, design, rng):
        """One noisy world observation; the caller owns experiment accounting."""
        return Observation(design, self.noise.sample(self._truth(design), rng))

    def test_designs(self):
        """Held-out settings, including extrapolation in length; not for the designer."""
        return [Design.from_dict({"L": length, "theta0": angle})
                for length, angle in [(3., .15), (3., 1.3), (.75, 1.)]]


def pendulum_mean(kind, design, theta):
    """Deterministic output of a candidate, not the world's hidden truth."""
    length, angle = _settings(design)
    theta = np.asarray(theta, float)
    dimensions = {"linear": 2, "power-law": 2, "power-law-amp": 3}
    if kind not in dimensions or theta.shape != (dimensions[kind],) or not np.all(np.isfinite(theta)):
        raise ValueError("Supply a known pendulum candidate and its parameter vector.")
    if kind == "linear":
        value = theta[0] + theta[1]*length
    else:
        value = theta[0]*length**theta[1]
        if kind == "power-law-amp":
            value *= 1+theta[2]*angle**2
    return np.array([value])


def _candidate(kind, means, scales, sigma):
    """Wrap one reviewed formula with the original independent Gaussian priors."""
    noise = GaussianNoise(sigma)
    # Power laws use the numerical ratio L / (1 metre), so c0 is measured in seconds.
    units = {"linear": ("s", "s/m"), "power-law": ("s", "dimensionless"),
             "power-law-amp": ("s", "dimensionless", "rad^-2")}[kind]
    parameters = tuple(ParameterSpec(f"c{i}", "normal", mean, scale, units[i])
                       for i, (mean, scale) in enumerate(zip(means, scales)))

    def theta(parameters):
        return np.array([parameters[f"c{i}"] for i in range(len(means))])

    def simulator(designs, parameters, rng):
        predictions = np.stack([pendulum_mean(kind, d, theta(parameters)) for d in designs])
        return noise.sample(predictions, rng)

    def likelihood(data, parameters):
        if any(np.shape(o.y) != (1,) or not np.all(np.isfinite(o.y)) for o in data):
            raise ValueError("Pendulum observations must be finite scalar-output arrays.")
        return sum(noise.log_likelihood(o.y, pendulum_mean(kind, o.design, theta(parameters))) for o in data)

    formulas = {"linear": "c0 + c1*L", "power-law": "c0*L**c1",
                "power-law-amp": "c0*L**c1*(1+c2*theta0**2)"}
    return ProgramHypothesis(kind, f"pendulum-{kind}-v1", parameters, simulator,
                             f"T = {formulas[kind]}; L is numerical length in metres, theta0 in radians; independent Gaussian coefficient priors.",
                             input_names=("L", "theta0"), likelihood=likelihood, noise_sigma=sigma)


def pendulum_hypotheses(*, sigma=.05):
    """Return linear, power-law and amplitude-corrected candidates in that order.

    Start with the first two; the third supplies the missing amplitude mechanism
    in a scripted expansion. These broad teaching priors can predict unphysical
    periods. They reproduce the old scenario, not a production physical prior.
    """
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("Observation noise SD must be positive and finite.")
    return (_candidate("linear", [0., 1.], [2., 1.], sigma),
            _candidate("power-law", [2., .5], [1., .4], sigma),
            _candidate("power-law-amp", [2., .5, 0.], [1., .4, .3], sigma))
