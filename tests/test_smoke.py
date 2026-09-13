"""A readable design -> observe -> infer smoke test using discovery components."""
import numpy as np

from discovery.costs import CostLedger
from discovery.design.designs import Design
from discovery.design.designers import RandomDesigner
from discovery.hypotheses import PolynomialHypothesis
from discovery.inference.particles import PolynomialParticleUpdater
from discovery.worlds import LineWorld


def test_one_full_round_end_to_end():
    rng = np.random.default_rng(0)
    world = LineWorld()
    updater = PolynomialParticleUpdater(world.sigma, method="importance", n_particles=4000)
    designer = RandomDesigner()
    ledger = CostLedger()

    # A fixed candidate pool keeps proposal separate from this experiment loop.
    hyps = [PolynomialHypothesis(0), PolynomialHypothesis(1)]

    # Run a few designed (here: random) experiments.
    data = []
    for _ in range(5):
        beliefs = updater.update(data, hyps)
        decision = designer.choose(world.design_space(), beliefs, rng)
        data.append(world.run(decision.design, rng))
        ledger.add_experiment()

    beliefs = updater.update(data, hyps)

    # Contract sanity: weights normalize, predictions are finite, var >= 0.
    assert np.isclose(sum(beliefs.model_probabilities.values()), 1.0)
    for posterior in beliefs.parameters.values():
        mean, var = posterior.predict(np.array([world.design_space()[0]["x"]]))
        assert np.all(np.isfinite(mean)) and np.all(var >= 0)
    assert ledger.experiments_run == 5

    # The world is linear: with 5 observations at sigma=0.1, the linear
    # structure's evidence should dominate the constant's decisively.
    gap = beliefs.log_evidences["poly-deg1"] - beliefs.log_evidences["poly-deg0"]
    assert gap > 5.0, f"expected linear >> constant, got log-evidence gap {gap:.2f}"


def test_design_is_hashable_and_roundtrips():
    d = Design.from_dict({"x": 1.5}, name="x=1.5")
    assert d["x"] == 1.5
    assert d.as_dict() == {"x": 1.5}
    assert hash(d) == hash(Design.from_dict({"x": 1.5}, name="x=1.5"))
