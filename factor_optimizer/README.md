# factor_optimizer

Evidence-guided factor mutation and search. Defines a mutation grammar, search
orchestration (train/validation/test sealed splits), and acceptance via
desirability + Pareto winners — coordinating factor_engine (compute/validate)
and quant_evaluator (evidence) through adapter protocols only.

**Version:** 0.1.0 ｜ **Repo:** https://github.com/HKUST-QUANT-SOCIETY/factor_optimizer (private)

## What it does / does NOT

- **Does**: search orchestration, trial ledger, mutation grammar, budget control,
  winner selection, treatment-integrity checks, admission policies.
- **Does NOT**: execute factors, compute metrics, or make final admission decisions —
  it coordinates FE/QE through protocols.

## Install

```bash
git clone https://github.com/HKUST-QUANT-SOCIETY/factor_optimizer.git
cd factor_optimizer
pip install -e .                     # core
pip install -e ".[factor_engine,quant_evaluator]"   # optional adapters
```

## Core API

```python
from factor_optimizer.search.runner import SearchRunner, SearchConfig, SearchSession
from factor_optimizer.search.winner_selector import select_winner
from factor_optimizer.search.uncertainty_winner import UncertaintyAwareWinnerSelector

session = SearchSession(SearchConfig(search_budget=...))
runner = SearchRunner(session, fe_adapter=..., qe_adapter=...)
# runner.run(...) -> trial ledger, selection, acceptance
```

- `grammar/` — `mutation_spec`, `registry`, `validation` (mutation grammar).
- `search/` — `SearchRunner`, `SearchSession`, `desirability` (`Desirability`,
  catastrophic floor), `pareto` (`ParetoPoint`/`ParetoFrontier`/`ParetoArchive`),
  `uncertainty_winner` (uncertainty-aware winner selector), `tiered_evaluation`,
  `multifidelity` (fidelity tiers), `plateau` (plateau detection),
  `treatment_decision`, `lineage`.
- `policy/` — admission criteria/verdict, diagnosis + repair (LLM repair mapper),
  governance.
- `contracts/` — `Trial`, `TrialLedger`, `SearchBudget`, `TreatmentIntegrity`
  (evidence schema version), `CandidateMutation`, `Objective`, `Splits`, `Validator`.
- `adapters/` — `factor_engine.py` (canonical hash, validate mutation, estimate
  complexity, operator metadata), `quant_evaluator.py` (evidence).
- `llm/` — prompts + proposal for LLM-assisted mutation.
- `seen/` — candidate identity; `complexity/` — complexity profile/budget.
- `data_capabilities.py` / `data_providers.py` — train/validation/test data scoping.

## Sealed-test discipline

`SealedTestExecutor` enforces train/validation/test separation with purge/embargo
isolation; search budget caps total evaluations; `TreatmentIntegrityCheck` binds
evidence schema version to treatment results.

## Related repos

- **factor_engine** — validate & hash mutations, estimate complexity
- **quant_evaluator** — evidence for acceptance
- **factor_assets** — admission library for accepted treatments
- **quant_platform** — candidate pipeline orchestration (platform worker may drive FO)
