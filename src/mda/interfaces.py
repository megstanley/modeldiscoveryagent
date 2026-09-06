"""Core contracts for MDA-mini.

Everything in this file is a *contract* between components, so that the four
component modules (worlds/, inference/, hypotheses/, design/ + eval/) can be
built independently and still snap together in the agent loop (agent.py).

Mapping to the paper (arXiv 2608.09696, "Model Discovery Agent"):

    World            <- a benchmark environment (ForceBench / ChemBench /
                        NeuronBench world): hidden ground-truth mechanism,
                        a discrete menu of interventions, noisy observations.
    Hypothesis       <- one candidate mechanistic model: a *structure*
                        (proposed by the LLM or an enumerator) with free
                        parameters and a prior over them.
    InferenceEngine  <- fits one Hypothesis to data, returning a Posterior
                        over its parameters *and* the log model evidence.
                        The paper uses adaptive-tempered SMC; the stub uses
                        prior importance sampling. Evidence is what makes
                        structure comparison (and the Occam penalty) work.
    Posterior        <- parameter posterior for one structure, exposing the
                        posterior-predictive mean/variance that VoI needs.
    Designer         <- experiment selection: VoI / random / LLM-chosen.
    Proposer         <- hypothesis generation: LLM (or grammar enumerator,
                        needed for probe P3 "is it discovery or recall?").
    CostLedger       <- token + simulator-call accounting. In from day one
                        because probes P2 (budget matching) and E5 (cost
                        Pareto frontier) depend on it.

Deliberate simplifications, and where the seams are if we widen them later:

  * Deterministic simulate + explicit NoiseModel. `Hypothesis.simulate` is
    noise-free; observation noise lives in `World.noise` / the likelihood.
    This is the regime where the paper's *analytic* VoI applies (Gaussian
    noise -> pick the design with max posterior-predictive variance).
    Stochastic dynamics (NeuronBenchStoch) and summary-statistic likelihoods
    enter later by adding NoiseModel implementations and letting `simulate`
    return summary statistics instead of raw trajectories — the contracts
    below don't change shape.
  * Discrete design space. The paper discretizes (13 options in ForceBench,
    36 in NeuronBench); extension E2 (continuous designs) would generalize
    `World.design_space` to a sampler/optimizer, touching only Designer.
  * Structure-level posterior = normalized evidences. The pool of
    HypothesisState objects plus `pool_log_weights` IS the posterior over
    structures; there is no separate cross-structure sampler.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, Sequence, runtime_checkable

import numpy as np
from scipy.special import logsumexp

Array = np.ndarray
RNG = np.random.Generator


# ---------------------------------------------------------------------------
# Experiments and data
# ---------------------------------------------------------------------------

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


@dataclass(frozen=True)
class Observation:
    """The outcome of running one design in the real world (noisy)."""

    design: Design
    y: Array  # observed outcome; shape is world-specific but fixed per world


Dataset = Sequence[Observation]


# ---------------------------------------------------------------------------
# Noise / likelihood
# ---------------------------------------------------------------------------

@runtime_checkable
class NoiseModel(Protocol):
    """Observation model linking noise-free simulate() output to data.

    Kept separate from Hypothesis so the same structure can be scored under
    different observation models (deterministic Gaussian now; Poisson spike
    counts or learned-summary-statistic likelihoods later, per the paper's
    NeuronBench setup).
    """

    def log_lik(self, y_obs: Array, y_pred: Array) -> float:
        """Log p(y_obs | y_pred)."""
        ...

    def sample(self, y_pred: Array, rng: RNG) -> Array:
        """Draw a noisy observation around the noise-free prediction."""
        ...


@dataclass(frozen=True)
class GaussianNoise:
    """Isotropic Gaussian observation noise — the base case.

    This is the regime where the paper derives the analytic VoI expression
    (max posterior-predictive variance), so it is the canonical NoiseModel
    for Phase 0/1 worlds.
    """

    sigma: float

    def log_lik(self, y_obs: Array, y_pred: Array) -> float:
        r = np.asarray(y_obs, float) - np.asarray(y_pred, float)
        n = r.size
        return float(
            -0.5 * np.sum(r**2) / self.sigma**2
            - n * np.log(self.sigma)
            - 0.5 * n * np.log(2.0 * np.pi)
        )

    def sample(self, y_pred: Array, rng: RNG) -> Array:
        return np.asarray(y_pred, float) + self.sigma * rng.standard_normal(
            np.shape(y_pred)
        )


# ---------------------------------------------------------------------------
# World (benchmark environment)
# ---------------------------------------------------------------------------

@runtime_checkable
class World(Protocol):
    """A ground-truth environment the agent experiments on.

    The agent may call `run` at most `budget` times (AgentConfig.budget);
    `test_designs` are held-out interventions used only by the eval harness
    to measure interventional prediction — never shown to the agent.
    """

    name: str
    noise: NoiseModel

    def description(self) -> str:
        """Natural-language context handed to the Proposer (the paper's
        'description of the domain'). Contamination probe P1 works by
        rewriting exactly this string (obfuscated names/units)."""
        ...

    def design_space(self) -> Sequence[Design]:
        """The discrete menu of interventions the agent may choose from."""
        ...

    def run(self, design: Design, rng: RNG) -> Observation:
        """Execute one experiment: ground-truth mechanism + noise."""
        ...

    def test_designs(self) -> Sequence[Design]:
        """Held-out interventions for evaluation (eval harness only)."""
        ...


# ---------------------------------------------------------------------------
# Hypothesis (candidate mechanistic model)
# ---------------------------------------------------------------------------

@runtime_checkable
class Hypothesis(Protocol):
    """One candidate structure with free parameters theta.

    theta is always a flat float vector of length `n_params`; structure
    lives in code/symbolic form, parameters in theta. That split is what
    lets one InferenceEngine serve every proposed structure.
    """

    name: str

    @property
    def n_params(self) -> int: ...

    @property
    def param_names(self) -> Sequence[str]: ...

    def sample_prior(self, rng: RNG, n: int) -> Array:
        """(n, n_params) draws from the parameter prior."""
        ...

    def log_prior(self, theta: Array) -> float:
        """Log prior density of one theta vector."""
        ...

    def simulate(self, design: Design, theta: Array) -> Array:
        """Noise-free predicted outcome of `design` under parameters theta.

        Must return the same shape as Observation.y for the target world.
        This is the expensive call — implementations should route counts
        through CostLedger via the caller.
        """
        ...

    def sympy_form(self) -> Optional[Any]:
        """Symbolic form (sympy.Expr) if the model is a closed-form law,
        else None. Used by eval for symbolic-equivalence scoring
        (ChemBench's sympy check); simulators (ODE worlds) return None.
        """
        ...


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@runtime_checkable
class Posterior(Protocol):
    """Parameter posterior for ONE hypothesis, given a dataset.

    `log_evidence` is log p(data | structure) — the quantity that both
    ranks structures in the pool (with its built-in Occam penalty; claim C5)
    and, normalized across the pool, forms the posterior over structures.
    """

    hypothesis: Hypothesis
    log_evidence: float

    def sample(self, rng: RNG, n: int) -> Array:
        """(n, n_params) posterior draws."""
        ...

    def predict(self, design: Design) -> tuple[Array, Array]:
        """Posterior-predictive mean and variance of the NOISE-FREE outcome
        at `design` (epistemic/parameter uncertainty only).

        Observation noise is deliberately excluded: for Gaussian noise it
        adds a design-independent constant, so VoI rankings are unchanged,
        and excluding it keeps `predict` reusable for evaluation.
        """
        ...


@runtime_checkable
class InferenceEngine(Protocol):
    """Fits one hypothesis to data. The paper's engine is adaptive-tempered
    SMC (Stage B, inference/); the stub is prior importance sampling."""

    def fit(
        self,
        hypothesis: Hypothesis,
        data: Dataset,
        noise: NoiseModel,
        rng: RNG,
    ) -> Posterior:
        ...


# ---------------------------------------------------------------------------
# The hypothesis pool
# ---------------------------------------------------------------------------

@dataclass
class HypothesisState:
    """One pool entry: a structure, its current posterior, and its
    normalized weight in the posterior over structures."""

    hypothesis: Hypothesis
    posterior: Posterior
    log_weight: float = 0.0  # normalized: logsumexp over pool == 0


def normalize_pool(states: Sequence[HypothesisState]) -> list[HypothesisState]:
    """Set each state's log_weight from the pool's log evidences.

    Assumes a uniform prior over structures in the pool, so the posterior
    over structures is just softmax(log_evidence). If we later want
    LLM-provided structure priors, they enter here as an additive term.
    """
    if not states:
        return []
    log_z = np.array([s.posterior.log_evidence for s in states], float)
    log_w = log_z - logsumexp(log_z)
    for s, lw in zip(states, log_w):
        s.log_weight = float(lw)
    return list(states)


# ---------------------------------------------------------------------------
# Design and proposal
# ---------------------------------------------------------------------------

@runtime_checkable
class Designer(Protocol):
    """Chooses the next experiment given the current pool.

    Implementations planned (design/): VoIDesigner (analytic Gaussian case:
    argmax over candidates of pool-mixture predictive variance),
    RandomDesigner, LLMDesigner (baseline for claim C2).
    """

    def choose(
        self,
        pool: Sequence[HypothesisState],
        candidates: Sequence[Design],
        rng: RNG,
    ) -> Design:
        ...


@runtime_checkable
class Proposer(Protocol):
    """Generates new candidate structures.

    `context` is World.description(); `pool` carries the current hypotheses
    (the paper conditions proposals on previous hypotheses and their
    residual errors — pass those through pool + data rather than widening
    this signature). Implementations planned (hypotheses/): LLMProposer,
    GrammarProposer (the no-LLM enumerator needed for probe P3), and the
    FixedPoolProposer stub.
    """

    def propose(
        self,
        context: str,
        data: Dataset,
        pool: Sequence[HypothesisState],
        n_new: int,
        rng: RNG,
    ) -> list[Hypothesis]:
        ...


# ---------------------------------------------------------------------------
# Cost accounting
# ---------------------------------------------------------------------------

@dataclass
class CostLedger:
    """Running totals for everything a fair comparison needs (probes P2/E5).

    Callers increment this — components stay pure. The agent loop charges
    sim_calls when it invokes simulate/run; LLM clients charge tokens.
    """

    sim_calls: int = 0
    experiments_run: int = 0
    llm_prompt_tokens: int = 0
    llm_completion_tokens: int = 0
    started_at: float = field(default_factory=time.time)

    def add_sim(self, n: int = 1) -> None:
        self.sim_calls += n

    def add_experiment(self, n: int = 1) -> None:
        self.experiments_run += n

    def add_llm(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.llm_prompt_tokens += prompt_tokens
        self.llm_completion_tokens += completion_tokens

    def wall_seconds(self) -> float:
        return time.time() - self.started_at


# ---------------------------------------------------------------------------
# Agent configuration and results
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AgentConfig:
    """Knobs of Algorithm 1. Defaults follow the paper's main setting."""

    budget: int = 8            # B: max experiments
    tau_r: float = 0.1         # residual threshold that triggers M-open expansion
    n_particles: int = 1000    # inference engine fidelity
    n_propose: int = 5         # structures requested per expansion
    shrink_log_weight: float = np.log(1e-3)  # drop structures below this pool weight
    seed: int = 0


@dataclass
class RoundRecord:
    """What happened in one design->observe->infer round (for analysis)."""

    round_index: int
    chosen: Design
    observation: Observation
    log_evidences: dict[str, float]     # hypothesis name -> log evidence
    residual: float                     # predictive-check statistic vs tau_r
    expanded: bool                      # did the proposer get called?
    dropped: list[str] = field(default_factory=list)  # shrunk-away hypotheses


@dataclass
class AgentResult:
    """Everything the eval harness needs, no more."""

    pool: list[HypothesisState]
    data: list[Observation]
    rounds: list[RoundRecord]
    ledger: CostLedger
    config: AgentConfig
