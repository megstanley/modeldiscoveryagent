"""Choose among an explicit finite menu; keep the score table for inspection."""
from dataclasses import dataclass
from typing import Callable, Protocol, Sequence

import numpy as np

from ..beliefs import Beliefs
from ..records import RNG
from .designs import Design
from .design_scorer import DesignScore


@dataclass(frozen=True)
class Decision:
    """Selected action, selection method, and every computed candidate score.

    Random selection has an empty score table rather than invented utilities.
    Scores are stored in menu order. They refer to beliefs *before* execution.
    """
    design: Design
    method: str
    scores: tuple[tuple[Design, DesignScore], ...]


class Designer(Protocol):
    """Select without access to the world or evaluator's answers."""
    def choose(self, candidates: Sequence[Design], beliefs: Beliefs, rng: RNG) -> Decision:
        ...


def _menu(candidates: Sequence[Design]) -> tuple[Design, ...]:
    """Check the menu without removing distinct future repeats from the history."""
    menu = tuple(candidates)
    if not menu:
        raise ValueError("The experiment menu must not be empty.")
    if len({d.params for d in menu}) != len(menu):
        raise ValueError("Menu settings must be unique; repeating a chosen experiment is still allowed.")
    return menu


@dataclass
class EnumeratingDesigner(Designer):
    """Evaluate a configured score function; choose the first maximum in menu order.

    Bind noise/task settings with functools.partial. The score function receives
    only one Design and current Beliefs. Explicit name labels the recorded
    objective/method; it is not inferred from a callable or confused with units.
    """
    scorer: Callable[[Design, Beliefs], DesignScore]
    name: str

    def choose(self, candidates: Sequence[Design], beliefs: Beliefs, rng: RNG) -> Decision:
        """Score all candidates without spending or consuming the random-selection RNG."""
        menu = _menu(candidates)
        scores = tuple((design, self.scorer(design, beliefs)) for design in menu)
        if any(not np.isfinite(score.value) for _, score in scores):
            raise ValueError("Candidate scores must be finite before selection.")
        winner = max(range(len(scores)), key=lambda i: scores[i][1].value)
        return Decision(menu[winner], self.name, scores)


@dataclass
class RandomDesigner(Designer):
    """Uniform baseline over the same menu, using an explicit independent RNG."""
    def choose(self, candidates: Sequence[Design], beliefs: Beliefs, rng: RNG) -> Decision:
        """Return an action without pretending that random choice evaluated a utility."""
        menu = _menu(candidates)
        return Decision(menu[int(rng.integers(len(menu)))], "random", ())
