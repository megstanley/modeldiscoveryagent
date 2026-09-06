"""Phase 0 sanity check (d): M-OPEN EXPANSION TRIGGERS CORRECTLY.

Spec
----
Uses the full run_mda loop (agent.py) with a scripted proposer whose
propose() calls we can count.

Trigger case:
  World truth: quadratic. Initial pool: {constant, linear} — truth is
  OUTSIDE the hypothesis class (M-open). Within the budget of 8, the
  predictive check must exceed tau_r at least once and the proposer must
  be called to expand. If the scripted proposer then supplies the
  quadratic, final pool weight must concentrate on it.

Control case (no false alarms):
  World truth: linear, initial pool: {constant, linear}, same seeds and
  tau_r. The proposer must NOT be called after round 1's initial
  proposal — residuals from a well-specified pool should stay below
  tau_r (checks the threshold isn't trivially always firing).

Assert both directions; a check that only fires (or never fires) is
useless. tau_r sensitivity beyond this single value is probe P5, not
this gate.

Why this gates Phase 1: expansion + shrinking is the paper's central
M-open claim (C4); the ChemBench ablation replication (plan item 1.2)
assumes this machinery demonstrably works.
"""
import pytest

pytestmark = pytest.mark.skip(
    reason="Stage C gate — needs run_mda (agent.py) implemented first"
)


def test_expansion_fires_when_truth_outside_pool():
    raise NotImplementedError


def test_expansion_stays_quiet_when_pool_is_adequate():
    raise NotImplementedError
