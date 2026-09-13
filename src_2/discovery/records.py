"""Records of measurements"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import numpy as np
from .design.designs import Design

Array = np.ndarray
RNG = np.random.Generator

@dataclass(frozen=True)
class Observation:
    """The outcome of running one design in the real world (noisy)."""

    design: Design
    y: Array  # observed outcome; shape is world-specific but fixed per world

Dataset = Sequence[Observation]


def observed_arrays(data: Dataset) -> tuple[Array, Array]:
    """Read known scalar inputs and measured outputs, retaining record order."""
    inputs, outputs = [], []
    for obs in data:
        if set(obs.design.as_dict()) != {"x"} or np.size(obs.y) != 1:
            raise ValueError("This example requires a known x and one measured y per record.")
        inputs.append(obs.design["x"])
        outputs.append(float(np.asarray(obs.y).reshape(-1)[0]))
    x_values, y_values = np.asarray(inputs, float), np.asarray(outputs, float)
    if not np.all(np.isfinite(x_values)) or not np.all(np.isfinite(y_values)):
        raise ValueError("Inputs and measurements must be finite.")
    return x_values, y_values

