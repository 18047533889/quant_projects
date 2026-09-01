# modeling

Model layer: training, walk-forward splitting, label-interval purge, embargo,
leakage guard-rails, artifact lifecycle, and prediction evaluation — the single
source of truth for temporal-leakage-safe model training on the quant platform.

**Version:** 0.1.0 ｜ **Repo:** https://github.com/HKUST-QUANT-SOCIETY/modeling (private)
**Authority:** `AUTHORITY.md` (MODEL2-P0-006) — enforcement lives here, not in callers.

## Install

```bash
git clone https://github.com/HKUST-QUANT-SOCIETY/modeling.git
cd modeling
pip install -e .
```

## Key concepts

- **Date-authoritative splits** — every row of a date belongs to exactly one
  split. `date_bounded_split`, `split_by_date_cutoff`, `assert_date_authoritative`.
- **Walk-forward** — `make_walk_forward_splits`, `WalkForwardSpec` (incl. `gap_days`),
  `purge_overlap` (label-interval purge), `apply_embargo`, `purge_before_boundary`,
  `check_fold_order`, `nested_splits`, `stitch_oos_windows`.
- **Leakage guard-rails** — `TrainOnlyFitGuard` (proves fit is never called during
  scoring), `assert_frozen_preprocessing`, plus a battery of negative controls
  (`future_poison`, `label_poison`, `scaler_poison`, `hyperparam_poison`,
  `universe_poison`, `revision_poison`, `execution_clock_poison`,
  `run_all_negative_controls`).
- **Timing contracts** — `DecisionClock` (decision at `t_close`, execution at
  `t+1 VWAP`, label matured at `t+H close`), `LabelContract`
  (`return_basis="vwap_to_vwap"`, `purge_by_interval`, `embargo_bars`),
  `EmbargoSpec`, `ApplicationWindow`, `AFTER_CLOSE_TO_NEXT_VWAP` /
  `BEFORE_SAME_DAY_VWAP`.
- **Training** — `train_model(train_ds, spec, learner, ...)` → `ModelArtifact`
  (frozen preprocessing + model + manifest); `TrainResult`; governance
  (`GovernanceContract`, `CandidateExposureLedger`, `select_best_validation`,
  `neighborhood_stability`); `hyperparams.validate_search_grid`.
- **Prediction** — `Predictor`, `predict_oos` (OOS-safety enforced),
  `PredictionOutputContract` (finite / aligned / status-gated),
  `PredictionBatch`.
- **Evaluation** — `per_date_rank_ic`, `ic_series`, `cross_sectional_ic`,
  `block_aware_ic`, `evaluate_predictions`, `EvaluationReport`,
  `run_walk_forward_evidence`.
- **Learners** — `learners/`: base + elastic_net, pls, pcr, regime, mixture_of_experts
  (model families: local-rolling / recursive-state / same-time cross-sectional /
  predictive-supervised / research-structural per `ModelExecutionClass`).
- **Artifact lifecycle** — `artifact.py` (ModelArtifact, FrozenPreprocessing,
  ModelArtifactManifest), `model_catalog.py`, `registry.py`,
  `monitoring.py` (drift: PSI, production drift metrics), `diagnostics.py`
  (parameter stability, feature ablation, label-shuffle control, complexity),
  `evidence.py`, `ledger.py`, `timing.py`, `dsl_bridge.py`, `sample_policy.py`.

## Example

```python
from modeling.dataset import PanelDataset, FeatureSchema
from modeling.trainer import train_model
from modeling.walk_forward import make_walk_forward_splits, purge_and_embargo
from modeling.contracts import LabelContract, AFTER_CLOSE_TO_NEXT_VWAP

label = LabelContract(label_name="fwd_vwap_5d", horizon_bars=5,
                      return_basis="vwap_to_vwap")
folds = make_walk_forward_splits(ds, spec=...)
train_ds = purge_and_embargo(folds[0].train, label)
artifact = train_model(train_ds, spec=..., learner=...)
pred = artifact.predict_oos(folds[0].valid)   # leakage-safe
```

## Hard rules

- **vwap-to-vwap** labels only (`LabelContract.return_basis`).
- **Never** use shuffled / non-temporal splits — always date-authoritative.
- Preprocessing must be frozen to train-only state before scoring.
- Run tests from the **repo root** of the monorepo:
  `pytest tests/modeling -q` (33 test files).

## Related repos

- **factor_preprocess** — produces `FeatureBundle` inputs (FP is the upstream
  feature pipeline; modeling enforces frozen-preprocessing)
- **quant_evaluator** — independent metric computation for model OOS checks
- **data_access** — panel data reads
- **quant_platform** — model DTOs (`ModelTrainingRequest`/`ModelArtifactRef`)
  map onto modeling artifacts
