"""Experiment settings"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
Array = np.ndarray
RNG = np.random.Generator


@dataclass(frozen=True)
class Design:
    """One executable experimental configuration (an intervention).

    `params` is a tuple of (name, value) pairs rather than a dict so Designs
    are hashable — useful for dedup and for logging which design was chosen
    each round.
    """

    params: tuple[tuple[str, float], ...]
    name: str = ""

    @staticmethod
    def from_dict(d: dict[str, float], name: str = "") -> "Design":
        return Design(params=tuple(sorted(d.items())), name=name)

    def as_dict(self) -> dict[str, float]:
        return dict(self.params)

    def __getitem__(self, key: str) -> float:
        return self.as_dict()[key]