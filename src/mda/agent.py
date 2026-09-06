"""The MDA loop (Algorithm 1 of the paper). STAGE C — not yet implemented.

This file pins down the loop's contract and pseudocode so every Stage B
component knows exactly how it will be called. Implementing it (plus the
four sanity checks in tests/) is the integration step, owned by one agent.

Pseudocode (paper's Algorithm 1, adapted to our interfaces):

    pool  <- []                          # list[HypothesisState]
    data  <- initial observations (may be empty)
    props <- proposer.propose(world.description(), data, pool, n_propose)
    pool  <- fit each proposal with engine; normalize_pool(pool)

    for t in 1..config.budget:
        d    <- designer.choose(pool, world.design_space(), rng)
        obs  <- world.run(d, rng);  ledger.add_experiment()
        data <- data + [obs]
        refit every pool entry on data; normalize_pool(pool)

        # --- M-open machinery (claim C4) ---
        residual <- predictive_check(pool, obs)     # e.g. standardized error
                                                    # of the pool-mixture
                                                    # prediction at d
        if residual > config.tau_r:                 # EXPAND
            props <- proposer.propose(..., n_propose)
            fit and add to pool; normalize_pool(pool)
        drop entries with log_weight < config.shrink_log_weight   # SHRINK

        record RoundRecord(t, d, obs, evidences, residual, expanded, dropped)

    return AgentResult(pool, data, rounds, ledger, config)

Design notes for the implementer:
  * predictive_check: the paper flags inadequacy when residual error stays
    too large. Concretely here: standardized residual of the pool-mixture
    posterior-predictive at the just-run design. Its exact form is a probe
    target (P5, tau_r sensitivity), so keep it a small injectable function
    rather than burying it in the loop.
  * Refitting from scratch each round is O(budget * pool * fit). Fine for
    Phase 0/1 scale; SMC engines may later expose incremental updates
    behind the same InferenceEngine contract.
  * The loop charges the ledger for experiments; engines/hypotheses charge
    sim calls; LLM proposers charge tokens. One ledger, passed everywhere.
"""
from __future__ import annotations

from .interfaces import (
    RNG,
    AgentConfig,
    AgentResult,
    CostLedger,
    Designer,
    InferenceEngine,
    Proposer,
    World,
)


def run_mda(
    world: World,
    proposer: Proposer,
    engine: InferenceEngine,
    designer: Designer,
    config: AgentConfig,
    rng: RNG,
    ledger: CostLedger | None = None,
) -> AgentResult:
    """Run the full design->observe->infer loop. Stage C implements this."""
    raise NotImplementedError(
        "Stage C: implement Algorithm 1 here, then un-skip the four sanity "
        "checks in tests/test_check_*.py — they are the Phase 0 gate."
    )
