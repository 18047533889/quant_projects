# AGENT_DIRECT_OPERATORS — Agent 可直接调用的生产算子

> 生成：2026-09-04 · HEAD `a07ea23020f86f4a63f50f67ec5b0ae1023b5b22`（工作树 dirty）
> 语义：**production_admitted ∧ terminal** 的算子（DirectUse R64 ladder，fail-closed；intermediate/condition/state 车道排除出 terminal surface，100k GO §30）。
> 机器快照：`artifacts/AGENT_DIRECT_OPERATORS.json`（schema `fe.agent_direct_operators.v1`）。

## 总览

| 指标 | 值 |
|------|----|
| production_admitted（P0#1 重生成后实测） | **86** |
| admitted ∧ terminal（Agent 可直接调用，不猜） | **62** |
| admitted ∧ intermediate（可作 expression child，非 terminal factor） | 24 |
| 数据源 | `artifacts/operator_audit/current_head/agent_direct_allowlist.json`（62）⊕ `operator_matrix.csv`（1673 行 admission 列） |

**Agent 用这 62 个不用猜。** 每一个都满足：

- `production_admitted=True`（registry valid / causal / field contract / backend parity / lookback contract 五门全过，见 GO_NO_GO §2 §99 对 admitted 集判定）；
- `terminal=True`（可作最终因子输出，非中间量）；
- `authoring_tier=daily` / `scope` 明确 / `output.grain` 明确 / `preferred_backend` 已定；
- 缺失 gate 如实标注在 `production_gates_nonpass`（如 BACKEND_PARITY_PASS=NOT_RUN —— 诚实未跑，不是冒充通过）。

## 62 个 terminal 算子明细

| canonical | mining_lane | authoring_tier | output.grain | execution_model | preferred_backend | stateful | causality |
|---|---|---|---|---|---|---|---|
| abs | alpha_direct | daily | daily | independent_with_warmup | polars | False | causal |
| add | alpha_direct | daily | daily | independent_with_warmup | polars | False | causal |
| clip | alpha_direct | daily | daily | independent_with_warmup | polars | False | causal |
| cs_demean | alpha_direct | daily | daily | independent_with_warmup | polars | False | causal |
| cs_mad_zscore | alpha_direct | daily | daily | independent_with_warmup | polars | False | causal |
| cs_pct_rank | alpha_direct | daily | daily | independent_with_warmup | polars | False | causal |
| divide | alpha_direct | daily | daily | independent_with_warmup | polars | False | causal |
| exp | alpha_direct | daily | daily | independent_with_warmup | polars | False | causal |
| group_neutralize | alpha_direct | daily | daily | independent_with_warmup | polars | False | causal |
| group_normalize | alpha_direct | daily | daily | independent_with_warmup | polars | False | causal |
| group_rank | alpha_direct | daily | daily | independent_with_warmup | polars | False | causal |
| group_winsorize | alpha_direct | daily | daily | independent_with_warmup | polars | False | causal |
| group_zscore | alpha_direct | daily | daily | independent_with_warmup | polars | False | causal |
| intra_amihud | alpha_direct | daily | daily | independent_with_warmup | polars | False | causal |
| intra_bipower_variation | alpha_direct | daily | daily | independent_with_warmup | polars | False | causal |
| intra_concentration | alpha_direct | daily | daily | independent_with_warmup | polars | False | causal |
| intra_entropy | alpha_direct | daily | daily | independent_with_warmup | polars | False | causal |
| intra_extreme_bar_return | alpha_direct | daily | daily | independent_with_warmup | polars | False | causal |
| intra_jump_ratio | alpha_direct | daily | daily | independent_with_warmup | polars | False | causal |
| intra_kyle_lambda_proxy | alpha_direct | daily | daily | independent_with_warmup | polars | False | causal |
| intra_path_efficiency | alpha_direct | daily | daily | independent_with_warmup | polars | False | causal |
| intra_realized_semivariance | alpha_direct | daily | daily | independent_with_warmup | polars | False | causal |
| intra_realized_variance | alpha_direct | daily | daily | independent_with_warmup | polars | False | causal |
| intra_segment_return | alpha_direct | daily | daily | independent_with_warmup | polars | False | causal |
| intra_segment_volume_share | alpha_direct | daily | daily | independent_with_warmup | polars | False | causal |
| intra_signed_imbalance_proxy | alpha_direct | daily | daily | independent_with_warmup | polars | False | causal |
| intra_vwap_above_ratio | alpha_direct | daily | daily | independent_with_warmup | polars | False | causal |
| inverse | alpha_direct | daily | daily | independent_with_warmup | polars | False | causal |
| log | alpha_direct | daily | daily | independent_with_warmup | polars | False | causal |
| log_abs | alpha_direct | daily | daily | independent_with_warmup | polars | False | causal |
| maximum | alpha_direct | daily | daily | independent_with_warmup | polars | False | causal |
| minimum | alpha_direct | daily | daily | independent_with_warmup | polars | False | causal |
| multiply | alpha_direct | daily | daily | independent_with_warmup | polars | False | causal |
| neg | alpha_direct | daily | daily | independent_with_warmup | polars | False | causal |
| normalize | alpha_direct | daily | daily | independent_with_warmup | polars | False | causal |
| rank | alpha_direct | daily | daily | independent_with_warmup | polars | False | causal |
| sign | alpha_direct | daily | daily | independent_with_warmup | polars | False | causal |
| signed_sqrt | alpha_direct | daily | daily | independent_with_warmup | polars | False | causal |
| sqrt | alpha_direct | daily | daily | independent_with_warmup | polars | False | causal |
| subtract | alpha_direct | daily | daily | independent_with_warmup | polars | False | causal |
| tanh | alpha_direct | daily | daily | independent_with_warmup | polars | False | causal |
| ts_autocorr | alpha_direct | daily | daily | independent_with_warmup | polars | False | causal |
| ts_beta | alpha_direct | daily | daily | independent_with_warmup | polars | False | causal |
| ts_corr | alpha_direct | daily | daily | independent_with_warmup | polars | False | causal |
| ts_cov | alpha_direct | daily | daily | independent_with_warmup | polars | False | causal |
| ts_delay | alpha_direct | daily | daily | independent_with_warmup | polars | False | causal |
| ts_delta | alpha_direct | daily | daily | independent_with_warmup | polars | False | causal |
| ts_log_return | alpha_direct | daily | daily | independent_with_warmup | polars | False | causal |
| ts_max | alpha_direct | daily | daily | independent_with_warmup | polars | False | causal |
| ts_mean | alpha_direct | daily | daily | independent_with_warmup | polars | False | causal |
| ts_median | alpha_direct | daily | daily | independent_with_warmup | polars | False | causal |
| ts_min | alpha_direct | daily | daily | independent_with_warmup | polars | False | causal |
| ts_pct | alpha_direct | daily | daily | independent_with_warmup | polars | False | causal |
| ts_rank | alpha_direct | daily | daily | independent_with_warmup | polars | False | causal |
| ts_sharpe | alpha_direct | daily | daily | independent_with_warmup | polars | False | causal |
| ts_std | alpha_direct | daily | daily | independent_with_warmup | polars | False | causal |
| ts_sum | alpha_direct | daily | daily | independent_with_warmup | polars | False | causal |
| ts_var | alpha_direct | daily | daily | independent_with_warmup | polars | False | causal |
| ts_zscore | alpha_direct | daily | daily | independent_with_warmup | polars | False | causal |
| where | alpha_direct | daily | daily | independent_with_warmup | polars | False | causal |
| winsorize | alpha_direct | daily | daily | independent_with_warmup | polars | False | causal |
| zscore | alpha_direct | daily | daily | independent_with_warmup | polars | False | causal |

> 完整字段（input_slots / scalar_parameters / searchable_params / aliases / cost / production_gates_nonpass）见机器快照 `AGENT_DIRECT_OPERATORS.json` 的 `operators[]`。

## 与 admission 全集的差异（86 − 62 = 24 被排除的 intermediate）

production_admitted 的 86 中，24 个是 `terminal=False`（intermediate/condition/state 车道）。它们**不能**作为 Agent 的 terminal factor 输出（100k GO §30：Intermediate 只能作 expression child），因此不出现在本 Agent 直连清单；但仍在 `OPERATOR_CORRECTNESS_MATRIX.parquet`（全 1673 行）中可查。

## 一致性校验

- `agent_direct_allowlist.json`（62 terminal）⊂ admission（86）⊂ 全 1673 canonical；三集合均由 `operator_audit/current_head` 同一次 inventory 重建导出。
- fail-closed 保证：任一 terminal 算子关键列（production_admitted / terminal_usable / backend_passed）为假即被剔除，不会以「看起来能用」混入。
