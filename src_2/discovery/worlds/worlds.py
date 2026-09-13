"""World interfaces and the simple scalar line world"""
from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable
import numpy as np
from .records import Array, RNG, Design, Observation
from .noise import NoiseModel, GaussianNoise


@runtime_checkable
class World(Protocol):
    """A ground-truth environment the agent experiments on.
    """

    name: str
    noise: NoiseModel

    def description(self) -> str:
        """Natural-language context for the Proposer, the description of the domain. We can contaminate by
        rewriting exactly this string (obfuscated names/units)."""
        ...

    def design_space(self) -> Sequence[Design]:
        """The discrete menu of interventions the agent may choose from."""
        ...

    def run(self, design: Design, rng: RNG) -> Observation:
        """Execute one experiment: ground-truth mechanism + noise."""
        ...

    def test_designs(self) -> Sequence[Design]:
        """Held-out interventions for evaluation. i.e. agent can't see these,
        they are to produce clean observations at test time.
        """
        ...


@dataclass
class LineWorld(World):
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
        # You ideally don't want to call this -- it's cheating!
        return np.array([self.a * design["x"] + self.b])

    def run(self, design: Design, rng: RNG) -> Observation:
        return Observation(design=design, y=self.noise.sample(self._truth(design), rng))

    def test_designs(self) -> Sequence[Design]:
        # Held-out interventions OUTSIDE the training grid on purpose:
        # interventional evaluation should include extrapolation.
        return [Design.from_dict({"x": v}, name=f"x={v:g}") for v in (-4.0, 5.0)]

