"""Phase 0 sanity check (c): VoI MATCHES HAND COMPUTATION.

Spec
----
Construct the analytically solvable case:
  * Two hypotheses with FIXED parameters (delta posteriors; no parameter
    uncertainty): h1: y = x, h2: y = x^2, pool weights (0.5, 0.5).
  * Gaussian noise, known sigma.
  * Candidate designs: x in {0, 1, 2}.

Under the paper's analytic VoI (max posterior-predictive variance of the
pool mixture), the predictive variance at design x is the between-
hypothesis disagreement:  w1*w2 * (h1(x) - h2(x))^2  (+ const).
Hand computation: disagreement at x=0 is 0, at x=1 is 0, at x=2 is 4.
So VoIDesigner MUST pick x=2 — the only design that discriminates.

Assert:
  1. VoIDesigner.choose returns x=2.
  2. The designer's internal scores match w1*w2*(h1-h2)^2 to numerical
     tolerance for all three candidates.
  3. Degenerate pool (one hypothesis, delta posterior): all scores equal;
     designer falls back to a deterministic-under-seed tie-break rather
     than crashing.

Why this gates Phase 1: claim C2 (VoI > random > ...) can only be tested
if the VoI implementation provably computes the quantity the paper
defines.
"""
import pytest

pytestmark = pytest.mark.skip(
    reason="Stage C gate — implement against the real VoIDesigner (Stage B design/)"
)


def test_voi_picks_the_discriminating_design():
    raise NotImplementedError


def test_voi_scores_match_hand_computation():
    raise NotImplementedError


def test_voi_degenerate_pool_does_not_crash():
    raise NotImplementedError
