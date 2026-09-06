"""Phase 0 sanity check (b): OCCAM VIA EVIDENCE — claim C5 in miniature.

Spec
----
World: LineWorld (truth is linear).
Pool: PolynomialHypothesis(1) vs PolynomialHypothesis(3). The cubic NESTS
the truth — it fits at least as well in likelihood, so only the evidence's
complexity penalty can prefer the linear structure.
Engine: the real adaptive-tempered SMC.

After ~6 observations (fixed seed), assert:
  1. log_evidence(linear) > log_evidence(cubic).
  2. The gap GROWS from n=3 to n=6 (the penalty compounds with data).

Also run the reverse control: make the world cubic; evidence must then
favor the cubic. (Guards against an engine that just always prefers
fewer parameters.)

Why this gates Phase 1: structure selection IS the paper's mechanism-
discovery claim; without a working Occam penalty the pool posterior is
meaningless.
"""
import pytest

pytestmark = pytest.mark.skip(
    reason="Stage C gate — implement against the real SMC engine (Stage B inference/)"
)


def test_evidence_prefers_true_structure_over_nested_overparameterized():
    raise NotImplementedError


def test_reverse_control_cubic_world_prefers_cubic():
    raise NotImplementedError
