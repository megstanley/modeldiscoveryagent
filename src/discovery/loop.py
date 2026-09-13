"""Small controllers; only the experimental controller may execute the world."""
from dataclasses import dataclass
from copy import deepcopy
from time import perf_counter
from typing import TYPE_CHECKING
import numpy as np
from .beliefs import Beliefs, Updater
from .hypotheses import Hypothesis
from .records import Dataset, Observation, RNG
from .worlds import World
from .design.designers import Decision, Designer
from .design.design_scorer import predictive_components

if TYPE_CHECKING:
    from .proposal import ProposalRequest, ProposalReview


def run_episode(data, initial_hypotheses, proposer, updater: Updater) -> tuple[Beliefs, Beliefs]:
    """Assess a pool, add proposals, and reassess on exactly the same data."""
    pool = list(initial_hypotheses)
    before: Beliefs = updater.update(data, pool)
    additions: list[Hypothesis] = proposer(data, tuple(pool), before)
    pool.extend(additions)
    after: Beliefs = updater.update(data, pool)
    return before, after


@dataclass(frozen=True)
class ProposalRound:
    """A fixed-data search round, not an experiment or a claim of Bayesian program sampling."""
    request: "ProposalRequest"
    reviews: tuple["ProposalReview", ...]
    before: Beliefs
    after: Beliefs
    pool: tuple[Hypothesis, ...]
    seconds: float
    prompt: str | None = None
    response: str | None = None

    def as_dict(self) -> dict:
        """Serialize provenance and numerical results without serializing executable callables."""
        def state(beliefs):
            return {"method": beliefs.method, "assessment": beliefs.assessment,
                    "model_probabilities": beliefs.model_probabilities,
                    "log_evidences": beliefs.log_evidences,
                    "parameters": None if beliefs.parameters is None else {
                        name: {"mean": fit.mean.tolist(), "covariance": fit.covariance.tolist()}
                        for name, fit in beliefs.parameters.items()}}
        return {
            "request": self.request.as_dict(), "prompt": self.prompt, "response": self.response,
            "attempts": len(self.reviews), "seconds": self.seconds,
            "reviews": [{"accepted": r.accepted, "reason": r.reason,
                         "model_id": r.proposal.hypothesis.name if isinstance(r.proposal.hypothesis, Hypothesis) else None,
                         "specification": r.proposal.hypothesis.specification() if isinstance(r.proposal.hypothesis, Hypothesis) else None,
                         "parent_ids": list(r.proposal.parent_ids), "rationale": r.proposal.rationale,
                         "raw_specification": r.proposal.raw_specification} for r in self.reviews],
            "before": state(self.before), "after": state(self.after),
            "pool_ids": [h.name for h in self.pool],
        }


def run_proposal_round(data, initial_hypotheses, proposer, updater: Updater, *,
                       description: str, n_new: int, rng: RNG, parent_ids=()) -> ProposalRound:
    """Assess, propose, validate and reassess on the same observed data.

    Caller owns when to invoke this and which parents to request. No world calls,
    implicit pruning, prior updates or automatic retries occur. Legacy polynomial
    updaters receive explicitly matched noise specifications. Every response item
    counts as an attempt, including invalid, duplicate and over-budget items.
    """
    from .proposal import (Proposal, ProposalRequest, ProposalReview,
                           make_feedback, review_proposals)
    from .programs import observation_hypothesis
    from .hypotheses import PolynomialHypothesis

    started = perf_counter()
    if not data:
        raise ValueError("Proposal assessment needs observed data and its output layout.")
    observed = tuple(deepcopy(data))
    sigma = getattr(updater, "sigma", None)
    pool = tuple(observation_hypothesis(deepcopy(h), sigma) for h in initial_hypotheses)
    # Validate counts and parent references before performing any inference/callback.
    ProposalRequest(description, observed, pool, (), n_new, tuple(parent_ids))
    before = deepcopy(updater.update(deepcopy(observed), deepcopy(pool)))
    feedback = make_feedback(observed, pool, before, rng)
    request = ProposalRequest(description, deepcopy(observed), deepcopy(pool), feedback,
                              n_new, tuple(parent_ids))
    try:
        proposals = proposer.propose(deepcopy(request), rng)
        if not isinstance(proposals, (list, tuple)):
            raise ValueError("Proposer must return a list of Proposal records.")
        # Scoring may not silently override a different proposed observation model.
        checked = []
        for proposal in proposals:
            if (isinstance(proposal, Proposal) and isinstance(proposal.hypothesis, PolynomialHypothesis)
                    and sigma is not None and proposal.hypothesis.noise.sigma != sigma):
                proposal = Proposal(proposal.hypothesis, proposal.rationale, proposal.parent_ids,
                                    "Proposed noise differs from the configured updater noise.", proposal.raw_specification)
            checked.append(proposal)
        reviews = review_proposals(checked, request, rng)
    except Exception as error:
        # Fail the proposal attempt visibly, without swallowing inference failures.
        reviews = (ProposalReview(Proposal(None, "Proposal call failed", error=str(error)),
                                  False, f"{type(error).__name__}: {error}"),)
    additions = tuple(deepcopy(r.proposal.hypothesis) for r in reviews if r.accepted)
    expanded = pool+additions
    after = deepcopy(updater.update(deepcopy(observed), deepcopy(expanded))) if additions else deepcopy(before)
    return ProposalRound(deepcopy(request), deepcopy(reviews), before, after, deepcopy(expanded),
                         perf_counter()-started, getattr(proposer, "last_prompt", None),
                         getattr(proposer, "last_response", None))


@dataclass(frozen=True)
class ExperimentalRound:
    """One real measurement and its pre-observation decision/prediction snapshot."""
    number: int
    records_before: int
    decision: Decision
    before: Beliefs
    forecast_mean: float
    forecast_variance: float
    observation: Observation
    after: Beliefs


def run_experiments(world: World, initial_data: Dataset, hypotheses, updater: Updater,
                    designer: Designer, *, budget: int, measurement_rng: RNG,
                    selection_rng: RNG) -> tuple[tuple[Observation, ...], list[ExperimentalRound]]:
    """Run a fixed-pool loop with exactly one world call per selected experiment.

    Repeated settings are allowed; each measurement is a separate record. The
    menu comes from world.design_space(), but only designs and beliefs reach the
    designer. This is ordinary information separation, not a Python sandbox.
    Full-history inference begins from priors each time. No model proposal or
    evaluation occurs here. The initial dataset is copied, never extended in place.

    This controller requires scalar polynomial posterior moments and known
    Gaussian measurement noise to record forecasts. Text-only agents need a forecast
    capability later; they must not silently receive fabricated numerical beliefs.
    """
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 0:
        raise ValueError("Experiment budget must be a non-negative integer.")
    data = list(deepcopy(initial_data))
    pool = tuple(hypotheses)
    beliefs = updater.update(tuple(data), pool)
    trace = []
    for number in range(1, budget + 1):
        menu = tuple(world.design_space())
        decision = designer.choose(menu, deepcopy(beliefs), selection_rng)
        if decision.design not in menu:
            raise ValueError("Designer selected an experiment outside the allowed menu.")
        parts = predictive_components(decision.design, beliefs, world.noise)
        weights, means = parts["weights"], parts["means"]
        mean = float(weights @ means)
        variance = float(weights @ (parts["observation_variances"] + (means - mean)**2))
        before = deepcopy(beliefs)
        observation = world.run(decision.design, measurement_rng)
        if observation.design != decision.design:
            raise ValueError("World returned an observation for a different design.")
        data.append(observation)
        beliefs = updater.update(tuple(data), pool)
        trace.append(ExperimentalRound(number, len(data)-1, decision, before, mean,
                                        variance, deepcopy(observation), deepcopy(beliefs)))
    return tuple(data), trace
