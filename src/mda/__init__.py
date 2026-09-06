"""MDA-mini: minimal reimplementation of the Model Discovery Agent
(arXiv 2608.09696) for replication, probing, and extension.

Start reading at interfaces.py (the contracts), then stubs.py (proof the
contracts run end to end), then agent.py (the Stage C loop skeleton).
"""

from .interfaces import (  # noqa: F401
    AgentConfig,
    AgentResult,
    CostLedger,
    Dataset,
    Design,
    Designer,
    GaussianNoise,
    Hypothesis,
    HypothesisState,
    InferenceEngine,
    NoiseModel,
    Observation,
    Posterior,
    Proposer,
    RoundRecord,
    World,
    normalize_pool,
)
