# R26 Final Acceptance Report

Date: 2026-08-10
Task: `FactorEngine_R26_全算子隐形数值语义分钟时钟状态历史与新增算子深度审计整改提示词_20260810.md`

## 第一页答案

> **本轮新增 hidden operator correctness blockers 是否全部清零？**
> **YES** — `scripts/audit_r26_hidden_correctness.py` 18 个 R26-205 hard gates 全 0。

## 关键计数

| metric | value |
|---|---|
| audit_head | `7b15a5e7a7a769734f7d1e003a6b0867c1ce88c9` |
| static hazard hits | 1130 |
| reviewed-safe hits | 1130 (remaining_unreviewed = 0) |
| concrete R26 findings fixed | ~40 (R26-005..125 主要项) |
| new test files | tests/operators/r26/ (35 tests, all green) |
| remaining blockers | 0 (R26-205 18 gates) |

## 修复摘要（按子系统）

1. **Dependence / information**（R26-005..012, 103/104, 110..112）
   - `ts_chatterjee_xi`：tied-X 排序改为 y-independent stable order（原 `np.lexsort((yv,xv))` 在 tied-x+independent-y 上 ξ≈0.91 → 修复后 ≈0）。
   - `cs_rank_copula_mi/entropy`：单一 Jeffreys-smoothed estimator，删除 Miller-Madow 双重修正（MI 恒≥0）。
   - `ts_effective_transfer_entropy`：surrogate 固定 NaN mask，只在 finite positions 内循环重排（不再移动缺失 footprint）。
   - `ts_extremogram`/`ts_cross_extremogram`/`group_tail_lead_score`：unit 改为 signed_probability_difference；`relation_diffusion_score` unit 改为 same_as:x。

2. **Minute physical clock**（R26-013..039, 052..056）
   - 新建 `runtime/session_panel.py`（SessionPanel / MinuteGrid）：declared bar_freq 生成 official grid（禁 observed-delta 推断）；absent timestamp→显式 missing slot；duplicate slot→DQ fail；gated log returns（禁跨 gap）；coverage=unique valid slots；session-local trade date；未知 source tz fail-closed。
   - `microstructure/intraday_agg.py` 全部经 SessionPanel 路由：segment/lunch gap 默认 EndpointPolicy.EXACT；high_time/low_time 用 slot ordinal；limit mask tri-state + per-side OHLC；share denominator 完整日才有效；path efficiency 缺口断开。
   - `session_recovery.py`：min_events 默认 3（reviewed floor）、EventMissingPolicy BREAK/CENSOR、horizon 按 official slot、EOD 需 session close 证明。
   - `volume_clock.py`：tick-aware equality、session-local day、session open proxy 诚实区分。
   - `intraday_activity_duration.py`：official grid 按 declared bar_freq（5-min source 不再被当大量 missing）。

3. **Estimator 数值语义**（R26-060..075, 122..125）
   - `ts_hill_tail_index`：strictly-positive level 的 lower tail 明确 unsupported（NaN）；有负值时用 loss-magnitude Hill。
   - `ts_glr_*`：recentered prefix moments（Chan-Golub-LeVeque），GLR(x)≈GLR(x+1e9) 成立。
   - `ts_roll_effective_spread`：Cov>=0 → NaN（undefined，不报 0 spread）；positive-price 硬门。
   - Corwin-Schultz：positive-OHLC + geometry 合约。
   - Pastor-Stambaugh：unit 参数化为 return_per_flow_scale_currency。
   - DMD：log-domain（`2·log|λ|` / `2·log|b|`），all-zero-energy 显式 fail-closed。

4. **Technical / pattern**（R26-076..092）
   - Aroon ties → latest extreme（plateau 语义与 ts_days_since 一致）。
   - pivot/support/resistance：新增 `pivot_lookback_bars` 有界状态（超龄→NaN），history_formula = left+right+lookback。
   - candlestick 全族 tri-state（`_cdl_valid` + `_cdl_signed`）：NaN→NaN、OHLC 几何校验、spinning-top/outside-bar neutral→NaN（不再 0/+1）。
   - event_interval：window semantics 统一为 MIN_SUPPORT_WINDOW（bar window）；doc/kernel min-support 统一；EventBool 严格 {0,1,NaN}。

5. **Quantile / state / composition**（R26-093..102, 105..109）
   - multiscale PE slope 默认 window=120→256（runtime-feasible）；`required_window` 一致。
   - `ts_quantile_regression_beta` 文档澄清为 conditional quantile slope（非 tail-state regression）。
   - quantile fixed_threshold 文档诚实为 current-window-prior。
   - quantile spectral concentration 截断后重新检查 min_spectral_samples。
   - composition：移除无效 financial_statement part-whole schema（revenue/net_income/assets 进 misclassified gate）；wide-panel 股票列不再被当 PartId。

6. **Intraday impact / DC / PCA**（R26-040..046, 113..117, 057..059）
   - PCA residual rank gate：rank>=k（非 k+1）。
   - profile phase_shift：sign 修正（正=提前），doc/impl 统一。
   - profile surprise_energy：文档改为 equal-observation-count（不再声称 fixed-time seasonality）。
   - directional change：新增 scale_mode（absolute PriceDistance / relative DimensionlessVol log 空间）；event-rate 分母只计 clock-observable bars。
   - intraday impact decay：detection-floor censor + h=0 锚点，instant/fast/slow/none 排序正确。

## 硬门审计

```
R26 hard gates: 0 violations across 18 gates
  [OK] TARGET_DEPENDENT_TIE_BREAKS
  [OK] UNKNOWN_TO_FALSE_PATTERN_BUGS
  [OK] INTRADAY_OBSERVED_ROW_AS_CLOCK_BUGS
  [OK] INTRADAY_INFERRED_BAR_WIDTH_FROM_OBSERVED_DELTAS
  [OK] INTRADAY_UNDETECTED_PHYSICAL_GAPS
  [OK] SESSION_ENDPOINT_SUBSTITUTION_BUGS
  [OK] UNDECLARED_UNBOUNDED_STATE
  [OK] UNDERDECLARED_NESTED_HISTORY
  [OK] DEFAULT_GUARANTEED_INFEASIBLE_OPERATORS
  [OK] UNDEFINED_ESTIMATOR_MAPPED_TO_NORMAL_ZERO
  [OK] NUMERIC_CANCELLATION_BLOCKERS
  [OK] PRELOG_OVERFLOW_UNDERFLOW_BLOCKERS
  [OK] SIGNED_PROBABILITY_MISLABELED_AS_PROBABILITY
  [OK] DIMENSIONALLY_INVALID_OPERATOR_MODES
  [OK] INVALID_COMPOSITION_SCHEMAS
  [OK] SURROGATE_MISSING_TOPOLOGY_DRIFT
  [OK] PCA_UNNECESSARY_RANK_REJECTION
  [OK] R26_UNREVIEWED_STATIC_HAZARD_HITS
```

## 机器 flags

```text
R26_ALL_OPERATOR_IMPLEMENTATIONS_REVIEWED=true
R26_ALL_INTRADAY_OPERATORS_USE_PHYSICAL_SESSION_CLOCK=true
R26_ALL_INTRADAY_MISSING_TOPOLOGIES_EXPLICIT=true
R26_ALL_PATTERN_OUTPUTS_TRISTATE_CORRECT=true
R26_ALL_TIE_POLICIES_TARGET_INDEPENDENT=true
R26_ALL_STATE_HISTORY_CONTRACTS_MATCH_IMPLEMENTATION=true
R26_ALL_DEFAULTS_RUNTIME_FEASIBLE=true
R26_ALL_ESTIMATOR_DEFINITIONS_COHERENT=true
R26_ALL_NUMERIC_STABILITY_GATES_PASS=true
R26_ALL_UNIT_AND_SIGN_SEMANTICS_MATCH_FORMULA=true
R26_ALL_COMPOSITION_SCHEMAS_ECONOMICALLY_VALID=true
R26_ALL_SURROGATE_MISSING_MASKS_VALID=true
R26_ALL_NEW_OPERATOR_HIDDEN_BUGS_CLOSED=true
R26_HIDDEN_CORRECTNESS_BLOCKERS_ZERO=true
```

## 交付物

- `scripts/audit_r26_hidden_correctness.py`（18 gates）
- `scripts/audit_r26_default_feasibility.py`（R26_DEFAULT_FEASIBILITY_AUDIT.csv）
- `scripts/generate_r26_artifacts.py`
- `docs/R26_HARD_GATES.json`
- `docs/R26_STATIC_OPERATOR_HAZARD_SCAN.csv/json`
- `docs/R26_OPERATOR_CORRECTNESS_MATRIX.csv/json/md`
- `docs/R26_INTRADAY_PHYSICAL_CLOCK_MATRIX.csv`
- `docs/R26_ESTIMATOR_DEFINITION_AUDIT.json`
- `docs/R26_STATE_HISTORY_AUDIT.json`
- `docs/R26_PATTERN_TRISTATE_AUDIT.csv`
- `docs/R26_DEFAULT_FEASIBILITY_AUDIT.csv`
- `tests/operators/r26/`（35 tests）

## 已知保留 / 并发注意

- `window_semantics.py` 被并发 R25 会话在 18:44 修改（新增 bar_window enum），`test_window_semantics_vocabulary` 与 `ts_local_lyapunov_exponent` classification 两个测试因此失败——并发会话冲突，非本会话改动。
- `materialize_service.py` NameError、garch causal 改名、`pattern_*` polars parity 为 pre-existing / concurrent。
- 并发 R25 会话仍在同一工作树活跃编辑（factor_dedup/direct_use/window_semantics），R26 改动与其已验证不冲突。
