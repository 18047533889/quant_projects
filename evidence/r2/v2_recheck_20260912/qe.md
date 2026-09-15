# QE V2 delta recheck receipt — 2026-09-12

- Tree: `/home/sunhaiwei/quant_projects`; baseline HEAD `e94ac507d670fd1c16b1d6a63fc5d6286daa5970`.
- Scope: QE-01..QE-35 and V-13; no branch, commit, deployment, production data, or FE/FA/FP edits.
- Hardware: NVIDIA L20, 46068 MiB total, 33459 MiB free at audit start.
- Stable QE main check: 1250 passed, 2 skipped in 141.30s; pytest exit 0. Log `qe_stable.log`, SHA-256 `963354ae698f275829792f82a9e71b3b4868c90ca80f5986a45f60984b9abdf5`. The wrapper recorded `SOURCE_CHANGED_DURING_RUN` only because concurrent owners changed three `quant_platform` files; the QE test scope passed.
- Adapter integration check: 2 passed in 0.49s, wrapper status PASS, with no source changes during run. Log `qe_adapters_final.log`, SHA-256 `20619a60c8efda4d324a5a70a0e930bfb50d333788e787c3ab098dfc9a220956`.
- Delta-focused run: 86 passed in 1.70s.
- Real CUDA-focused run: 52 passed, 172 deselected in 2.05s (no skips).

| ID | Recheck status | Code reference | Public-path / counterexample reference |
|---|---|---|---|
| QE-01 | mapped, existing fix retained | `runtime/evaluator.py`; `adapters/execution_trajectory.py` | `tests/test_v5_metric_instances.py`; `tests/test_v7_public_axes.py` |
| QE-02 | mapped, existing fix retained | `contracts/metric_artifacts.py`; `runtime/evaluator.py` | `tests/test_v3_public_artifacts.py`; `tests/test_daily_quantile_artifact_v3.py` |
| QE-03 | mapped, existing fix retained | `runtime/evaluator.py`; `metrics/coverage_compiler.py` | `tests/metrics/test_qe_p0_coverage_binding.py`; `tests/test_v3_public_artifacts.py` |
| QE-04 | mapped, existing fix retained | `runtime/evaluator.py`; `runtime/device_session.py` | `tests/test_v3_gpu_semantics.py`; `tests/test_v3_input_contracts.py` |
| QE-05 | mapped, existing fix retained | `api/batch_bundle.py`; `runtime/gpu_executor.py` | `tests/test_v5_gpu_capability_matrix.py`; `tests/test_public_gpu_tiling.py` |
| QE-06 | mapped, existing fix retained | `runtime/gpu_executor.py`; `metrics/ic_summary.py` | `tests/test_icir_exact_variance.py`; `tests/test_v3_gpu_semantics.py` |
| QE-07 | mapped, existing fix retained | `metrics/turnover.py`; `kernels/gpu/turnover.py` | `tests/test_turnover_validity.py`; `tests/test_v8_numeric_goldens.py` |
| QE-08 | mapped, existing fix retained | `metrics/quantile_shape.py`; `kernels/gpu/quantile_shape.py` | `tests/test_public_gpu_tiling.py` |
| QE-09 | mapped, existing fix retained | `contracts/portfolio_inputs.py`; `runtime/gpu_executor.py` | `tests/test_v7_gpu_portfolio_contracts.py`; `tests/test_gpu_probe_parity.py` |
| QE-10 | mapped, existing fix retained | `runtime/gpu_executor.py` | `tests/test_v7_gpu_portfolio_contracts.py` |
| QE-11 | mapped, existing fix retained | `metrics/portfolio_stats.py`; `metrics/underwater.py` | `tests/test_v3_drawdown_contracts.py` |
| QE-12 | mapped, existing fix retained | `metrics/portfolio_stats.py`; `kernels/gpu/drawdown.py` | `tests/test_v3_drawdown_contracts.py`; `tests/test_v7_gpu_portfolio_contracts.py` |
| QE-13 | mapped, existing fix retained | `metrics/portfolio_stats.py` | `tests/test_v3_drawdown_contracts.py` |
| QE-14 | mapped, existing fix retained | `metrics/underwater.py` | `tests/test_v8_numeric_goldens.py` |
| QE-15 | mapped, existing fix retained | `metrics/underwater.py` | `tests/test_v8_numeric_goldens.py` |
| QE-16 | mapped, existing fix retained | `metrics/underwater.py` | `tests/test_v3_risk_variants.py` |
| QE-17 | mapped, existing fix retained | `metrics/portfolio_stats.py`; `metrics/underwater.py` | `tests/test_v3_risk_variants.py` |
| QE-18 | mapped, existing fix retained | `metrics/calendar_returns.py` | `tests/test_calendar_returns.py`; `tests/test_v3_calendar_public.py` |
| QE-19 | mapped, existing fix retained | `metrics/portfolio_stats.py` | `tests/metrics/test_v8_probe_stress_contracts.py` |
| QE-20 | mapped, existing fix retained | `contracts/quantile_policy.py`; `metrics/portfolio_stats.py` | `tests/test_v5_discrete_rank_semantics.py` |
| QE-21 | mapped, existing fix retained | `contracts/portfolio_inputs.py`; `metrics/probe_portfolio/_core.py` | `tests/test_v3_trade_eligibility.py` |
| QE-22 | mapped, existing fix retained | `metrics/portfolio_stats.py`; `metrics/turnover_cost.py` | `tests/test_turnover_cost.py` |
| QE-23 | mapped, existing fix retained | `metrics/probe_portfolio/_core.py`; `kernels/gpu/portfolio.py` | `tests/test_v3_cohort_intervals.py`; `tests/test_gpu_probe_parity.py` |
| QE-24 | mapped, existing fix retained | `contracts/portfolio_schedule.py`; `metrics/probe_portfolio/_core.py` | `tests/test_v3_cohort_intervals.py`; `tests/test_v7_gpu_portfolio_contracts.py` |
| QE-25 | mapped, existing probe-only fix retained | `contracts/portfolio_inputs.py`; `adapters/execution_trajectory.py` | `tests/test_v3_trade_eligibility.py`; `tests/test_v5_execution_trajectory_adapter.py` |
| QE-26 | mapped, existing fix retained | `contracts/portfolio_inputs.py`; `metrics/underwater.py` | `tests/test_v8_numeric_goldens.py` |
| QE-27 | mapped, existing fix retained | `contracts/metric_artifacts.py`; `runtime/evaluator.py` | `tests/test_v3_public_artifacts.py`; `tests/test_v5_metric_instances.py` |
| QE-28 | mapped, existing fix retained | `runtime/evaluator.py`; `runtime/gpu_executor.py` | `tests/test_v7_bounded_stream.py`; `tests/test_v7_chunk_cache.py` |
| QE-29 | mapped, existing fix retained | `contracts/axis_refs.py`; `contracts/label_bundle.py` | `tests/test_v3_input_contracts.py`; `tests/test_v7_public_axes.py` |
| QE-30 | mapped, existing fix retained | `contracts/label_bundle.py` | `tests/test_v3_input_contracts.py`; `tests/test_v8_clock_sink_acceptance.py` |
| QE-31 | mapped, existing fix retained | `contracts/label_bundle.py`; `runtime/cache_v2.py` | `tests/test_v3_immutable_inputs.py`; `tests/test_v7_chunk_cache.py` |
| QE-32 | mapped, existing fix retained | `api/horizons.py`; `metrics/temporal.py` | `tests/test_horizons.py` |
| QE-33 | mapped, existing fix retained | `metrics/shape_evidence.py` | `tests/test_v3_shape_contracts.py` |
| QE-34 | mapped, existing fix retained | `metrics/shape_evidence.py` | `tests/test_v3_shape_contracts.py` |
| QE-35 | mapped, existing fix retained | `contracts/adaptive_bins_policy.py` | `tests/test_v5_discrete_rank_semantics.py`; `tests/test_v8_numeric_goldens.py` |

## V-13 delta fixed

`NET_EXECUTABLE` trajectories previously emerged from the QE public execution adapter as the same `ProbePortfolioArtifact` runtime type used by assumed-cost research probes. Added `ExecutablePortfolioArtifact`, which requires `execution_certified=true`, `cost_scope=NET_EXECUTABLE`, and a non-empty execution-ledger ref. Assumed-cost trajectories remain exactly `ProbePortfolioArtifact` with `execution_certified=false`. Public evaluation accepts both typed identities without allowing labels or untyped panels.

The QE→FA synthetic integration fixture now binds production assembly to the actual winner recipe identity and treatment-selection timestamp, and carries the same universe/snapshot/split context in the admission artifact. A wrong-recipe replay is explicitly rejected by FA's unchanged production gate.

## Remaining blockers / non-claims

- QE does not claim that a research probe is executable or capacity evidence. Missing execution/fill/borrow/cost evidence still fails closed in the execution-domain trajectory contract.
- FE `research_compute/UNVERIFIED` durable-manifest assurance cannot be safely inferred from the current recipe's bare string source ref. A typed projection must be supplied by FE/coordinator; QE must preserve it as research-only metadata and must never translate it into `execution_certified`.
- The two pre-existing capability skips (no zscore/standardize kernel; no drop-row kernel) remain explicit and were not converted into passes.
- The prior cross-package fixture failure is closed by binding its real synthetic provenance; no FA production gate was weakened.
