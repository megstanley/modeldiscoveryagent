# mda-mini

Minimal reimplementation of the **Model Discovery Agent**
([arXiv 2608.09696](https://arxiv.org/abs/2608.09696), Murphy 2026) for
replication, probing, and extension. The paper releases the
[NeuronBench benchmark](https://github.com/murphyk/neuronbench) but **not**
the agent — this repo rebuilds the agent from the paper's description.

The research plan this serves (claim ledger C1–C7, phases, probes P1–P6,
extensions E1–E6) lives in the companion plan document; item numbers below
refer to it.

## Reading order

1. `src/mda/interfaces.py` — the contracts. Every design decision is
   documented inline, including what was deliberately simplified and where
   the seams are (stochastic likelihoods, continuous designs).
2. `src/mda/stubs.py` — simplest-correct implementations of every contract,
   proving they compose end to end without an API key.
3. `notebooks/mda_walkthrough.ipynb` — **the guided tour**: every contract
   exercised on a pendulum world with a hidden large-amplitude correction, so
   the full M-open arc plays out (evidence selects power-law → predictive
   check fires at large amplitude → expansion discovers the corrected law →
   VoI's blind spot closes). Includes dataflow and loop-sequence diagrams;
   ships pre-executed with all outputs.
4. `tests/test_smoke.py` — a hand-rolled design→observe→infer loop on the
   stubs; doubles as the minimal usage demo. **Runs green today.**
5. `src/mda/agent.py` — Algorithm 1 pseudocode and the `run_mda` signature
   (Stage C; not yet implemented).
6. `tests/test_check_*.py` — the four Phase-0 sanity checks, currently
   skipped, each with a precise spec in its docstring. Un-skipping these
   (against the real components) is the Phase-0 gate.

## Component map

| Directory | Paper component | Stage B owner builds | Plan items served |
|---|---|---|---|
| `worlds/` | benchmark environments | toy worlds (drag projectile, Michaelis–Menten), NeuronBench adapter | 0.1, 1.1, 1.2 |
| `inference/` | adaptive-tempered SMC + evidence | real `InferenceEngine`, validated on conjugate cases | checks (a)(b), C5 |
| `hypotheses/` | LLM proposer / structures | sympy→JAX models, grammar enumerator, LLM client (stubbed in tests) | P1, P3 |
| `design/` | VoI experiment selection | `VoIDesigner` (analytic Gaussian), LLM designer baseline | check (c), C2 |
| `eval/` | metrics + reporting | nMSE, RMSLE, sympy equivalence, results tables | 1.1–1.3, P6 |
| `agent.py` | Algorithm 1 loop | Stage C integration (one owner) | check (d), C4 |

Cross-cutting from day one: `CostLedger` (sim calls, tokens) — probes P2
(budget-matched comparison) and E5 (cost Pareto) need it, and it can't be
retrofitted honestly.

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev,notebooks]"
.venv/bin/python -m ipykernel install --user --name mda-mini \
    --display-name "Python (mda-mini)"   # register kernel for any Jupyter/VS Code
.venv/bin/pytest            # expect: smoke tests pass, 4 checks skipped
.venv/bin/jupyter lab notebooks/mda_walkthrough.ipynb   # the guided tour
```

Optional extras: `.[full]` (jax/diffrax/blackjax, for Stage B inference and
ODE worlds), `.[bench]` (neuronbench).

## Status

- [x] Stage A — contracts, stubs, test skeletons (this commit)
- [ ] Stage B — four component agents in parallel (worlds, inference,
      hypotheses, design+eval), each owning one directory + its tests
- [ ] Stage C — `run_mda` loop + un-skip the four sanity checks (the gate)
- [ ] Stage D — Phase 1 replications, one agent per experiment
