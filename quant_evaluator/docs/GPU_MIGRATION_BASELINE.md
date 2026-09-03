# GPU Migration Baseline

> Auto-generated during the GPU batch evaluation refactor (spec §60 Phase 0).
> Target repo: `/home/sunhaiwei/quant_projects` — HEAD `dbbc3fe75f38d75c88f62ad60e3e0eedc666f026`.

## Environment

| item | value |
|---|---|
| GPU | NVIDIA L20 (CUDA 13.0, driver 580.126.20) |
| VRAM free/total | ~39.6 / 46.1 GB |
| CuPy | `cupy-cuda13x==14.2.0` (installed for this work) |
| numba | 0.67.0 |
| CPU cores / RAM | 32 / 92 GB |

## Registry / audit facts

- `MetricRegistry` registered metrics: **55** (registry state `building`).
- Existing GPU artifacts: `backends/cupy_backend.py` (GPUBackend with
  `fast_ic_batch_gpu`, `_spearman_rank_corr_gpu` still Python T×F loop,
  `_fast_rank_gpu` per-unique-value loop — these are the spec §1.3 bottlenecks).
- Existing `runtime/evaluator.py` `evaluate()` accepts no backend param yet.
- `adapters/data_access.py` is still a stub (`da_frame_to_factor_batch` et al.
  raise `NotImplementedError`).
- `pyproject.toml` has no GPU optional-dependency group.
- Stale docs present: `docs/METRIC_REGISTRY_COVERAGE.csv`,
  `docs/METRIC_COVERAGE_COMPILER.csv` (drift vs live registry).

## New runtime contracts (created during this work)

- `contracts/backend_policy.py` — `BackendPolicy` / `PrecisionPolicy` /
  `GPUExecutionPolicy` / `DeviceFactorBatch` / `DeviceLabelPanel` /
  `DeviceEvaluationContext` (spec §4).
- `backends/capability_registry.py` — `BackendImplementation` /
  `BackendCapabilityRegistry` (spec §11).

## GPU kernels (created and parity-verified)

All in `quant_evaluator/kernels/gpu/`:

- `rank.py` — `batched_rank` (batched segmented sort + average-tie, **exact**
  pandas `rank(method='average')` parity incl. NaN excluded from ranking),
  `batched_distinct_level_count`, `batched_quantile_assignment` (**exact** CPU
  `assign_quantiles` QE-Q-P0-001/002 percentile-boundary parity),
  `batched_rank_weights`.
- `correlation.py` — `batched_pearson_ic` (**exact** CPU parity),
  `batched_spearman_ic` (**exact** CPU parity: pairwise-finite ranking,
  per-(t,f) label rank, average-tie, distinct-level floor `min_levels =
  max(min_obs//2,2)`, min_obs).
- `runtime/device_session.py` — `DeviceEvaluationSession` (stage-once,
  pinned/streams, factor tiling, OOM retile, VRAM budget) — spec §5/§7/§8/§41.

## Parity status (verified on host, L20)

| metric | parity | max abs diff |
|---|---|---|
| rank (avg-tie incl. NaN) | exact | 0.0 |
| quantile assignment | exact | match |
| pearson IC | exact | ~1e-16 |
| spearman RankIC | exact | ~4e-17 |

## Not yet done (definition of done gaps)

GPU turnover / rank stability, probe cohort portfolio GPU batch, robustess GPU
(HAC/autocorr/half-life/bootstrap/subsample), DataAccess production adapter,
metric expansion (quantile shape / stability / exposure / tradability /
novelty / integrity / data-quality), capability matrix generator, benchmark,
500-factor acceptance, tests matrix.
