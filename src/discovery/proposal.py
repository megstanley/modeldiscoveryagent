"""Small proposal boundary shared by scripted, replayed and provider-backed proposals.

The proposer suggests candidates, never model probabilities or observations.
Names serve as model IDs in the existing Beliefs API: a revision needs a new
unique name. Scientific fingerprints separately detect renamed duplicates.
Only registered, reviewed callables can become program hypotheses. Source text
is data, never passed to eval, exec, import or a shell.
"""
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import Callable
import json
import numpy as np

from .beliefs import Beliefs
from .comparison import probability_weights
from .hypotheses import Hypothesis, PolynomialHypothesis
from .noise import GaussianNoise
from .prediction import sample_posterior_predictive
from .programs import ProgramHypothesis, hypothesis_fingerprint
from .records import Dataset


@dataclass(frozen=True)
class ModelFeedback:
    """Observed-data predictions and residuals, not held-out truth or a validity verdict."""
    model_id: str
    weight: float | None
    prediction: list | None
    residuals: list | None
    prediction_method: str
    warning: str | None = None


@dataclass(frozen=True)
class ProposalRequest:
    """Everything a proposer may see; no World or hidden evaluator is supplied."""
    description: str
    data: Dataset
    pool: tuple[Hypothesis, ...]
    feedback: tuple[ModelFeedback, ...]
    n_new: int
    parent_ids: tuple[str, ...] = ()

    def __post_init__(self):
        if isinstance(self.n_new, bool) or not isinstance(self.n_new, int) or self.n_new < 1:
            raise ValueError("Requested proposal count must be a positive integer.")
        names = [h.name for h in self.pool]
        if not names or len(set(names)) != len(names):
            raise ValueError("The current pool needs unique model IDs.")
        if any(parent not in names for parent in self.parent_ids):
            raise ValueError("Selected parents must belong to the current pool.")

    def as_dict(self) -> dict:
        """A JSON-ready request that can be printed, saved and replayed verbatim."""
        return {
            "schema_version": 1, "description": self.description,
            "observations": [{"design": o.design.as_dict(), "output": np.asarray(o.y).tolist()}
                             for o in self.data],
            "pool": [{"id": h.name, "specification": h.specification(),
                      "description": h.human_description()} for h in self.pool],
            "feedback": [dict(vars(f)) for f in self.feedback],
            "n_new": self.n_new, "parent_ids": list(self.parent_ids),
        }


@dataclass(frozen=True)
class Proposal:
    """A new candidate, its lineage and stated rationale; parsing failures are retained."""
    hypothesis: Hypothesis | None
    rationale: str
    parent_ids: tuple[str, ...] = ()
    error: str | None = None
    raw_specification: dict | None = None


@dataclass(frozen=True)
class ProposalReview:
    """One charged proposal attempt and the reason it was accepted or rejected."""
    proposal: Proposal
    accepted: bool
    reason: str


class Proposer(ABC):
    """Generate ideas; the caller owns validation, inference and the search schedule."""

    @abstractmethod
    def propose(self, request: ProposalRequest, rng: np.random.Generator) -> list[Proposal]:
        """Return proposed candidates without mutating the supplied history or pool."""
        ...


class ScriptedProposer(Proposer):
    """Replay a fixed list of trusted candidates for inspectable tests, not discovery."""
    def __init__(self, proposals):
        self.proposals = tuple(deepcopy(proposals))

    def propose(self, request, rng):
        """Return fresh copies; exceeding the requested count is audited by the loop."""
        return list(deepcopy(self.proposals))


class LLMProposer(Proposer):
    """Provider-neutral JSON adapter with no client, credentials or live calls by default.

    complete(prompt) -> response_text is injected explicitly; recorded responses
    can use the same callback for replay. program_registry maps versioned IDs to
    reviewed ProgramHypothesis templates. The response cannot introduce code or
    change a registered program's priors/noise. Such revisions need a reviewed
    new template/version. This is deliberately not arbitrary program execution.
    """
    def __init__(self, complete: Callable[[str], str] | None = None, *, program_registry=None):
        self.complete = complete
        self.program_registry = dict(program_registry or {})
        if any(not isinstance(v, ProgramHypothesis) or k != v.program_id
               for k, v in self.program_registry.items()):
            raise ValueError("Registry keys must match reviewed ProgramHypothesis program IDs.")
        self.last_prompt = None
        self.last_response = None

    def propose(self, request, rng):
        """Ask for structured specifications and retain individual parse failures."""
        self.last_prompt = self.last_response = None
        if self.complete is None:
            raise RuntimeError("No LLM provider configured; inject a callback or a recorded response.")
        self.last_prompt = (
            "Propose scientific hypotheses using only the supplied context. Return a JSON array. "
            "Each object needs name, kind, rationale and parent_ids. "
            "For kind=polynomial also supply degree, prior_scale and noise_sigma. "
            "For kind=program supply program_id from the registered templates below. "
            "Do not return Python source or posterior probabilities. Priors are scientific "
            "assumptions, not fitted values; explain changes. Return at most n_new candidates.\n"
            + json.dumps({"request": request.as_dict(),
                          "registered_programs": {k: v.specification() for k, v in self.program_registry.items()}},
                         sort_keys=True, allow_nan=False))
        response = self.complete(self.last_prompt)
        if not isinstance(response, str):
            raise ValueError("Provider must return JSON text.")
        self.last_response = response
        items = json.loads(self.last_response)
        if not isinstance(items, list):
            raise ValueError("Proposal response must be a JSON array.")
        proposals = []
        for item in items:
            try:
                proposals.append(self._parse(item))
            except (ValueError, TypeError, KeyError) as error:
                proposals.append(Proposal(None, "Invalid response item", error=str(error),
                                          raw_specification=item if isinstance(item, dict) else None))
        return proposals

    def _parse(self, item) -> Proposal:
        """Accept a small allowlist of fields; in particular, never execute code."""
        if not isinstance(item, dict):
            raise ValueError("Each proposal must be an object.")
        common = {"name", "kind", "rationale", "parent_ids"}
        if not common <= item.keys():
            raise ValueError("Missing name, kind, rationale or parent_ids.")
        if not all(isinstance(item[k], str) and item[k].strip() for k in ("name", "kind", "rationale")):
            raise ValueError("Name, kind and rationale must be nonempty text.")
        if not isinstance(item["parent_ids"], list) or not all(isinstance(p, str) for p in item["parent_ids"]):
            raise ValueError("parent_ids must be a list of model IDs.")
        if item["kind"] == "polynomial":
            if set(item) != common | {"degree", "prior_scale", "noise_sigma"}:
                raise ValueError("Polynomial fields must specify degree, prior_scale and noise_sigma only.")
            hypothesis = PolynomialHypothesis(item["degree"], item["prior_scale"], item["name"],
                                              GaussianNoise(item["noise_sigma"]))
        elif item["kind"] == "program":
            if set(item) != common | {"program_id"}:
                raise ValueError("Programs may reference a reviewed program_id, not source code or overridden priors.")
            if item["program_id"] not in self.program_registry:
                raise ValueError("Unknown or unreviewed program_id.")
            hypothesis = deepcopy(self.program_registry[item["program_id"]])
            hypothesis.name = item["name"]
        else:
            raise ValueError("Unknown hypothesis kind.")
        return Proposal(hypothesis, item["rationale"], tuple(item["parent_ids"]),
                        raw_specification=deepcopy(item))


def make_feedback(data, pool, beliefs: Beliefs, rng, *, predictive_samples=256) -> tuple[ModelFeedback, ...]:
    """Describe fits on supplied observations; generic programs use explicit Monte Carlo."""
    designs = tuple(o.design for o in data)
    observed = np.stack([np.asarray(o.y, float) for o in data])
    reports = []
    for h in pool:
        weight = None if beliefs.model_probabilities is None else beliefs.model_probabilities[h.name]
        try:
            posterior = beliefs.parameters[h.name]
            if isinstance(h, PolynomialHypothesis):
                prediction = posterior.predict([d["x"] for d in designs])[0][:, None]
                method = "analytic noise-free posterior mean; in-sample"
            else:
                prediction = sample_posterior_predictive(posterior, designs, rng, n=predictive_samples).mean(0)
                method = f"mean of {predictive_samples} observation simulations; in-sample"
            if prediction.shape != observed.shape or not np.all(np.isfinite(prediction)):
                raise ValueError("Prediction layout differs from observations.")
            reports.append(ModelFeedback(h.name, weight, prediction.tolist(),
                                         (observed-prediction).tolist(), method))
        except (ValueError, NotImplementedError, TypeError, KeyError) as error:
            reports.append(ModelFeedback(h.name, weight, None, None, "unavailable", str(error)))
    return tuple(reports)


def select_parents(beliefs: Beliefs, n: int, rng) -> tuple[str, ...]:
    """Resample model IDs by their weights, with replacement; duplicates are intentional."""
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError("Parent count must be a positive integer.")
    if beliefs.model_probabilities is None:
        raise ValueError("Weighted parent selection requires model probabilities.")
    names, weights = probability_weights(beliefs.model_probabilities)
    return tuple(rng.choice(names, n, p=weights).tolist())


def needs_proposals(feedback, *, residual_threshold: float) -> bool:
    """Toy adequacy trigger: every available candidate exceeds observed-output RMSE.

    Threshold is in output units, not model probability. This in-sample check is
    not Murphy's pre-observation predictive check. Unavailable feedback raises;
    it cannot justify declaring a pool adequate or inadequate.
    """
    if not np.isfinite(residual_threshold) or residual_threshold < 0 or not feedback:
        raise ValueError("Supply feedback and a finite nonnegative RMSE threshold.")
    if any(f.residuals is None for f in feedback):
        raise ValueError("Adequacy cannot be assessed from missing predictions.")
    return min(np.sqrt(np.mean(np.asarray(f.residuals)**2)) for f in feedback) > residual_threshold


def review_proposals(proposals, request: ProposalRequest, rng, *, prior_checks=4):
    """Validate all attempts, including duplicates and over-budget outputs, without inference."""
    if isinstance(prior_checks, bool) or not isinstance(prior_checks, int) or prior_checks < 1:
        raise ValueError("Validation requires a positive number of prior checks.")
    names = {h.name for h in request.pool}
    known_parents = set(names)
    fingerprints = {hypothesis_fingerprint(h) for h in request.pool}
    designs = tuple(o.design for o in request.data)
    expected = np.stack([np.asarray(o.y) for o in request.data]).shape
    reviews = []
    for index, proposal in enumerate(proposals):
        try:
            if not isinstance(proposal, Proposal):
                raise ValueError("Proposer returned something other than Proposal.")
            if index >= request.n_new:
                raise ValueError("Proposal exceeds the requested attempt budget.")
            if proposal.error:
                raise ValueError(proposal.error)
            h = proposal.hypothesis
            if not isinstance(h, Hypothesis):
                raise ValueError("Candidate must implement Hypothesis.")
            if not proposal.rationale.strip() or not h.name or h.name in names:
                raise ValueError("A rationale and a new unique model ID are required.")
            if any(parent not in known_parents for parent in proposal.parent_ids):
                raise ValueError("Proposal refers to an unknown parent.")
            if request.parent_ids and (not proposal.parent_ids
                                      or not set(proposal.parent_ids) <= set(request.parent_ids)):
                raise ValueError("A parent-based request requires lineage to the selected parents.")
            fingerprint = hypothesis_fingerprint(h)
            if fingerprint in fingerprints:
                raise ValueError("Duplicate scientific definition, even if renamed.")
            seed = int(rng.integers(0, 2**31-1))
            theta = np.asarray(h.sample_prior(np.random.default_rng(seed), prior_checks), float)
            if theta.shape != (prior_checks, h.n_params) or not np.all(np.isfinite(theta)):
                raise ValueError("Malformed prior samples.")
            for t in theta:
                a = np.asarray(h.sample_observations(designs, t, np.random.default_rng(seed)), float)
                b = np.asarray(h.sample_observations(designs, t, np.random.default_rng(seed)), float)
                if a.shape != expected or not np.all(np.isfinite(a)):
                    raise ValueError("Simulation does not match the measured observation layout.")
                if not np.array_equal(a, b):
                    raise ValueError("Simulation is not reproducible for the same explicit seed.")
            names.add(h.name)
            fingerprints.add(fingerprint)
            reviews.append(ProposalReview(proposal, True, "Compatible prior and seeded observation simulations."))
        except Exception as error:
            if not isinstance(proposal, Proposal):
                proposal = Proposal(None, "Malformed proposal")
            reviews.append(ProposalReview(proposal, False, str(error)))
    return tuple(reviews)
