# GPU Migration Progress

> Updated during the GPU batch evaluation refactor (spec §60, §67 Definition of
> Done).  HEAD `dbbc3fe75f38d75c88f62ad60e3e0eedc666f026`.  GPU: NVIDIA L20,
> cuPy `cupy-cuda13x==14.2.0`.

## Done (parity-verified on L20)

**Runtime contracts (spec §4)** — `contracts/backend_policy.py`:
`BackendPolicy` (AUTO/CPU_REFERENCE/CPU_FAST/CUDA/CUDA_STRICT), `PrecisionPolicy`,
`GPUExecutionPolicy`, `DeviceFactorBatch`, `DeviceLabelPanel`,
`DeviceEvaluationContext`.

**Capability registry (spec §11)** — `backends/capability_registry.py`:
`BackendImplementation`, `BackendCapabilityRegistry`.

**Device session (spec §5/§7/§8/§41)** — `runtime/device_session.py`:
`DeviceEvaluationSession` (stage-once, pinned/streams scaffold, factor tiling,
OOM retile, VRAM budget, memory pool).

**GPU kernels (spec §12)** — `kernels/gpu/`, all EXACT CPU parity:
- `rank.py`: batched average-tie rank (pandas-parity incl. NaN excluded),
  distinct-level count, quantile assignment (QE-Q-P0-001/002 percentile-boundary
  parity), rank weights.
- `correlation.py`: batched Pearson IC (exact), batched Spearman RankIC
  (exact: pairwise-finite ranking, per-(t,f) label rank, average-tie,
  distinct-level floor `min_levels=max(min_obs//2,2)`).
- `quantile.py`: batched quantile returns.
- `turnover.py`: batched turnover (0.5·Σ|Δw|).
- `stability.py`: batched rank stability (Spearman adjacent-day).

**GPU Executor (spec §3/§43/§44)** — `runtime/gpu_executor.py` `GPUExecutor`
+ `api/batch_bundle.py` `BatchEvaluationBundle` (columnar).  `evaluate()`
acquired `backend=`, `gpu_policy=` parameters and returns the columnar bundle
when `backend="cuda"`.  Verified GPU==CPU for rank_ic/ic_ir/pearson_ic/coverage.

**DataAccess** (spec §21-25, §58) — `adapters/data_access.py` replaced stub with
`DataAccessEvaluationLoader`:
- reads `ashare_stock_daily_adj` via `store.read_factors` (Arrow-first),
- `TargetSpec` / `TARGET_CONTRACT` for H01/H05/H10/H20 (producer-bound timing,
  **QE never shifts labels**),
- verified live: `TargetVwapReturnH10 == AdjVwap[t+11]/AdjVwap[t+1]-1`
  (local cos_data mirror root), 0 diff.
- semantic `volume = Volume/Factor`; `return_bp` decimal handling.

**Capability matrix (spec §49)** — `scripts/generate_capability_matrix.py`
generates `docs/METRIC_CAPABILITY_MATRIX.csv` (55 metric rows); stale
`METRIC_REGISTRY_COVERAGE.csv` / `METRIC_COVERAGE_COMPILER.csv` superseded.

**Packaging (spec §48)** — added `gpu-cuda12x`/`gpu-cuda13x` optional deps
(not core); `backends/__init__` exposes the new contracts.

**Tests (spec §62)** — `tests/test_gpu_parity.py` (11 tests) GREEN on L20:
rank (incl. heavy-ties / NaN-excluded / distinct-level floor / constant
reject), quantile, quantile-returns, turnover, rank-stability, pearson-ic,
spearman-ic.  Full CPU suite: **452 passed, 2 skipped** (skips are pre-existing
documented-contract tests).

## In flight (subagents)

- Agent C: GPU turnover/rank-stability parity tests (kernels already verified here).
- Agent D: GPU batch cohort portfolio + portfolio risk metrics (`kernels/gpu/portfolio.py`,
  `drawdown.py`) — spec §16/§17/§19/§31.
- Agent E: GPU temporal/robustness (`kernels/gpu/temporal.py`, `robustness.py`)
  — spec §20/§32.

## Not yet done

- 500-factor acceptance benchmark JSON (running), tile invariance / OOM retile
  tests, memory-leak test, clean wheel build, `evaluate_many` public facade,
  metric-expansion families (quantile-shape / style exposure / tradability /
  novelty / integrity / data-quality) GPU.
