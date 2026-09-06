"""Phase 0 sanity check (a): POSTERIOR CONTRACTION — M-closed.

Spec
----
World: LineWorld (truth y = a*x + b, known Gaussian noise).
Pool: the true structure ONLY (PolynomialHypothesis(degree=1)).
Engine: the real adaptive-tempered SMC (Stage B inference/).

Fit on n = 2, 4, 8, 16 observations (fixed seed). Assert:
  1. Coverage: true (b, a) lies within 3 posterior standard deviations of
     the posterior mean at every n.
  2. Contraction: posterior sd of each parameter at n=16 is smaller than
     at n=2 (allow non-monotonicity in between; require the endpoints).

Why this gates Phase 1: if the engine can't recover parameters when the
structure is KNOWN, nothing downstream (evidence, VoI, expansion) means
anything.
"""
import pytest

pytestmark = pytest.mark.skip(
    reason="Stage C gate — implement against the real SMC engine (Stage B inference/)"
)


def test_posterior_contracts_and_covers_truth():
    raise NotImplementedError
