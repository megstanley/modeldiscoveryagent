"""Stage A smoke test: the contracts are sufficient to run a full
design -> observe -> infer round using only stubs.

This deliberately hand-rolls the loop (agent.run_mda is Stage C) — it is
also a readable, minimal demo of how the pieces connect.
"""
import numpy as np

from mda import CostLedger, HypothesisState, normalize_pool
from mda.stubs import FixedPoolProposer, LineWorld, PriorISEngine, RandomDesigner


def test_one_full_round_end_to_end():
    rng = np.random.default_rng(0)
    world = LineWorld()
    proposer = FixedPoolProposer()
    engine = PriorISEngine(n_samples=4000)
    designer = RandomDesigner()
    ledger = CostLedger()

    # Propose structures (stub: constant vs linear polynomial).
    hyps = proposer.propose(world.description(), data=[], pool=[], n_new=5, rng=rng)
    assert len(hyps) == 2

    # Run a few designed (here: random) experiments.
    data = []
    for _ in range(5):
        pool = normalize_pool(
            [HypothesisState(h, engine.fit(h, data, world.noise, rng)) for h in hyps]
        )
        d = designer.choose(pool, world.design_space(), rng)
        data.append(world.run(d, rng))
        ledger.add_experiment()

    pool = normalize_pool(
        [HypothesisState(h, engine.fit(h, data, world.noise, rng)) for h in hyps]
    )

    # Contract sanity: weights normalize, predictions are finite, var >= 0.
    assert np.isclose(np.exp([s.log_weight for s in pool]).sum(), 1.0)
    for s in pool:
        mean, var = s.posterior.predict(world.design_space()[0])
        assert np.all(np.isfinite(mean)) and np.all(var >= 0)
    assert ledger.experiments_run == 5

    # The world is linear: with 5 observations at sigma=0.1, the linear
    # structure's evidence should dominate the constant's decisively.
    by_name = {s.hypothesis.name: s for s in pool}
    gap = (
        by_name["poly-deg1"].posterior.log_evidence
        - by_name["poly-deg0"].posterior.log_evidence
    )
    assert gap > 5.0, f"expected linear >> constant, got log-evidence gap {gap:.2f}"


def test_design_is_hashable_and_roundtrips():
    from mda import Design

    d = Design.from_dict({"x": 1.5}, name="x=1.5")
    assert d["x"] == 1.5
    assert d.as_dict() == {"x": 1.5}
    assert hash(d) == hash(Design.from_dict({"x": 1.5}, name="x=1.5"))
