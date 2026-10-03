# FE Operator Reuse Matrix (R61-FI-040)

宽表到长表的对齐语义与性能核验见
[FE 结果回填性能](FE_PANEL_ALIGNMENT_PERFORMANCE.md)。

Audit of the FactorPreprocess `TransformRegistry` against the FactorEngine
canonical operator registry (via `factor_engine.api.static_analysis`
`operator_semantic_metadata` + direct `OperatorRegistry.get(...,
backend="pandas_numpy")` numeric probing).  Purpose (plan §26 F1-F4): remove
duplicated mathematical authority where FE is the stateless-math owner, while
keeping FP-native every transform whose semantics FE does not reproduce
exactly.  This matrix is the routing authority consumed by the
`implementation_origin` / `fe_operator_id` / `fit_kind` metadata stamped on
every registered transform (`registry/transforms.py`, R61-FI-041).

Columns:

- **transform** — FP registry transform name.
- **semantic id** — FP `TransformSemanticID` / metadata `semantic_id`.
- **math semantics** — one-line mathematical meaning.
- **FE equivalent operator** — canonical FE operator id, or `—`.
- **FE semantic metadata** — FE `OperatorSemanticMetadata` (tags /
  stage_hint / causality_class / fitted_or_stateless) when annotated.
- **confidence** — `exact` (numeric parity == 0.0 proven over randomized
  (T,N) panels incl. NaN / ties / Inf), `partial` (same family, differing
  degenerate/parameter/current-bar semantics), `none`.
- **fitted state?** — whether the FP transform requires fitted state.
- **causal?** — FP `causal_safe` + causality class.
- **FP-native reason** — why the transform stays FP (fitted / degenerate-row
  divergence / FE has no equivalent / offline-only / exposure-vs-fitted).
- **implementation_origin** — `FE_OPERATOR` (routed through
  `adapters/fe_operator.py`), `FE_COMPOSITE` (routed through a named FE recipe),
  or `FP_NATIVE`. FE-backed routes fail closed when FE is unavailable;
  FP-kernel fallback requires explicit research opt-in.

Parity evidence: `tests/test_fe_operator_parity.py` and
`tests/test_missing_indicator_fe.py` (per-FE-backed-transform parity tests;
tolerance contract explicit — cross-sectional, ffill, and missing indicator
exact `atol=0.0`, OLS `atol=1e-9`/`rtol=1e-7`). FP FeOperatorExecutor
selects FE `pandas_numpy` only and converts pandas long input to a wide panel,
then stacks the FE result back to a pandas Series. FE also has an `is_null`
Polars binding with NULL-or-NaN semantics, but FP does not execute it through
Polars; that binding is outside this FP route and its identity. FE_OPERATOR
execution identities bind the selected canonical/backend, catalog source and
semantic version, FE implementation and logical-contract hashes, adapter
implementation/call contract, and NumPy/Pandas versions. The coverage marker
excludes the full transitive runtime, native libraries, and data dependencies;
this is a scoped implementation identity, not a full runtime attestation.
Registry routing tests:
`tests/test_transform_metadata.py`.

The `cs_winsor` parity probe covers non-finite inputs: FP converts +/-Inf to
missing before quantile estimation, matching FE `winsorize`.

Identity regression tests: `tests/test_fe_operator_execution_identity.py`
checks all five FE_OPERATOR routes, live identity refresh after a FE backend
replacement, binding of the selected operator to actual execution, and
fail-closed behavior when FE provenance is incomplete.

## Cross-sectional

| transform | semantic id | math semantics | FE equivalent operator | FE semantic metadata | confidence | fitted? | causal? | FP-native reason | implementation_origin |
|---|---|---|---|---|---|---|---|---|---|
| `cs_rank` | `CS_RANK:pct` | 0-1 percentile rank, average ties, singleton → 0.5, NaN/±Inf excluded | `rank` (= `cs_rank_01` exact duplicate) | tags `(rank,)` stage `representation` causal `cross_sectional` stateless | **exact** (maxabs 0.0) | no | yes | — | **FE_OPERATOR** (`fe_operator_id=rank`) |
| `cs_zscore` | `ZSCORE:cs` | CS z-score `(x-mean)/std`, ddof=1, NaN out | `zscore` (pandas `mean/std ddof=1`; zero-std→`(x-mean)/1`) | tags `(rank,zscore)` stage `representation` causal `cross_sectional` stateless | **partial** — identical on normal rows (maxabs 0.0 over random panels) BUT **singleton-row divergence**: one finite value → FE emits all-NaN (std ddof=1 NaN not replaced), FP returns `constant_value` (0.0) at the finite cell (`np.where(std>0,…)` with NaN>0 False). Singleton rows are reachable in sparse/illiquid cross-sections | no | yes | degenerate-row semantics differ (FE all-NaN vs FP constant 0.0); FP must keep its documented `constant_value` policy | **FP_NATIVE** |
| `cs_demean` | `CROSS_SECTIONAL_DEMEAN:cs` | CS demean `x - nanmean(row)` | `cs_demean` | tags `(neutralize,)` stage `neutralize` causal `cross_sectional` stateless | **exact** (maxabs 0.0; singleton demean → 0.0 both) | no | yes | — | **FE_OPERATOR** (`fe_operator_id=cs_demean`) |
| `cs_winsor` | `WINSOR:cs` | CS quantile clip to `[lower,upper]`, linear interpolation, NaN out | `winsorize` | tags `(winsorize,)` stage `transform` causal `cross_sectional` stateless | **exact** (maxabs 0.0 across quantile grid; `np.nanquantile` linear == FE interpolation) | no | yes | — | **FE_OPERATOR** (`fe_operator_id=winsorize`) |
| `cs_scale` | `CROSS_SECTIONAL_SCALE:cs` | CS scaling to `target_std` (× target/std) | `scale` | unannotated for `cs_scale`; `scale` UNKNOWN tags/stage, causal `cross_sectional`, stateless | **partial** — FE `scale` is experimental with unknown quantile/scale semantics; FP multiplies by `target_std/std` where FE scale may min-max or divide differently. No FE semantic annotation → cannot certify | no | yes | FE operator unannotated (`scale` experimental, no semantic_tags/stage_hint); FP scale-by-target-std math is its own documented authority | **FP_NATIVE** |

## Missingness / freshness

| transform | semantic id | math semantics | FE equivalent operator | FE semantic metadata | confidence | fitted? | causal? | FP-native reason | implementation_origin |
|---|---|---|---|---|---|---|---|---|---|
| `forward_fill` | `FILL:forward` | bounded forward fill (`max_lag` consecutive NaNs) == pandas `ffill(limit)` | `ffill_limit` | data_cleaning, `max_periods` param; unannotated in semantic bridge | **exact** (maxabs 0.0 over random gap panels; FE `ffill_limit` == pandas `ffill(limit=)` == FP `ffill(limit=max_lag)`) | no | yes | — | **FE_OPERATOR** (`fe_operator_id=ffill_limit`) |
| `missing_indicator` | `MISSING_INDICATOR:binary` | binary 1/0 missingness | `is_null` | elementwise; pandas_numpy `x.isna().astype(float)` | **exact** for None/pd.NA/NaN and ±Inf (FP pandas adapter route) | no | yes | — | **FE_OPERATOR** (`fe_operator_id=is_null`; pandas_numpy only) |
| `missing_rate` | `MISSING_RATE:rolling` | rolling lagged missing fraction | — | — | none | no | yes | no FE rolling-missing-rate operator; FP lagged-rolling semantics | **FP_NATIVE** |
| `missing_run_length` | — | consecutive-missing run length | — | — | none | no | yes | no FE equivalent | **FP_NATIVE** |
| `impute_with_fallback` | — | ffill + fallback (zero/median/mean) | `fillna(method=…)` / `fillna_const` | elementwise, unannotated | none | no | **no** (RESEARCH_ONLY, non-causal full-sample median/mean) | research-only; full-sample median/mean fallback not production-causal; FE fillna has no per-asset median groupby parity | **FP_NATIVE** |
| `linear_interpolate` (registered? no — module export only) | — | bounded linear interpolation | — | — | none | no | yes | not in registry (module-level helper) | — |
| `time_weighted_interpolate` (module export only) | — | time-weighted interpolation | — | — | none | no | yes | not in registry | — |
| `days_since_update` | `FRESHNESS:days_since_update` | calendar days since last valid obs | — | — | none | no | yes | FE `ts_days_since` is a count of bars not calendar days; no equivalent | **FP_NATIVE** |
| `observation_age` | — | days between observation & decision date | — | — | none | no | yes | two-date-column PIT transform; no FE equivalent | **FP_NATIVE** |
| `freshness_score` | `FRESHNESS:score` | exp decay freshness score | — | — | none | no | yes | no FE equivalent | **FP_NATIVE** |
| `stale_data_indicator` | — | binary stale flag | — | — | none | no | yes | no FE equivalent | **FP_NATIVE** |
| `freshness_aware_fill` | `FILL:freshness_aware` | decayed forward fill | — | — | none | no | yes | FP-native decayed-carry; no FE operator | **FP_NATIVE** |

## Temporal smoothing (causal)

| transform | semantic id | math semantics | FE equivalent operator | FE semantic metadata | confidence | fitted? | causal? | FP-native reason | implementation_origin |
|---|---|---|---|---|---|---|---|---|---|
| `ewma` | `SMOOTH:ewma` | lagged EWMA on `shift(1)`, `adjust=False, ignore_nulls=False` | `FE_COMPOSITE:long_smoothing.lagged_ewma:v1` | native long-panel EWMA; half-life parameterization | tested finite-value and missing-input parity | no | yes | registry delegates to this distinct lagged recipe; current-inclusive `ts_ema` remains a different operator | **FE_COMPOSITE** |
| `trailing_sma` | `SMOOTH:trailing_sma` | lagged trailing SMA `[t-window, t-1]` over prior observed asset rows | composite `ts_mean(ts_delay(x, 1), min_periods=min_periods)` on long panels | stateless causal FE expression | **exact for sorted per-asset dates**; duplicate dates retain input order through positional ordinal | no | yes | empty/missing asset rows are NaN; null observed-asset dates reject, same as sort contract | **FE_COMPOSITE** |
| `trailing_median` | `SMOOTH:trailing_median` | finite-only lagged median over prior observed rows `[t-window, t-1]` | composite `ts_median(ts_delay(x, 1))` with finite-observation count gate | stateless causal FE expression | **exact for sorted per-asset dates**; interleaved assets and duplicate dates retain input row order through positional ordinal | no | yes | null asset rows omitted; null dates for observed assets reject; NaN and ±Inf are excluded from the median and `min_periods` count | **FE_COMPOSITE** |
| `rolling_mean` | `SMOOTH:trailing_sma` | lagged rolling mean over prior observed asset rows `[t-window, t-1]` | `FE_COMPOSITE:long_smoothing.lagged_mean:v1` (shared with `trailing_sma`) | stateless causal FE expression | **exact for sorted per-asset dates**; duplicate dates retain input order through positional ordinal | no | yes | same lagged-mean executor and recipe as `trailing_sma`; empty/missing asset rows are NaN and null observed-asset dates reject | **FE_COMPOSITE** |
| `rolling_std` | — | finite-only lagged rolling std over prior observed rows | `FE_COMPOSITE:long_smoothing.lagged_std:v1`: population `ts_std(ts_delay(x, 1))` plus finite count gate | stateless causal FE expression | **exact**; pandas-compatible `d=int(ddof)` coercion, then `sigma_d = sigma_0 * sqrt(n/(n-d))` for `n>d` and the required `min_periods` | no | yes | excludes current observation; missing assets remain NaN; positional restoration preserves interleaved assets and duplicate dates; NaN ddof rejects and infinite ddof raises overflow | **FE_COMPOSITE** |
| `rolling_zscore` | — | causal paired lagged z-score; numerator is current row, mean/std exclude it | `FE_COMPOSITE:long_smoothing.lagged_zscore:v1` | native Polars rolling expressions; bounded stable interval joins for hazardous magnitudes; not current-inclusive `ts_zscore` | independent 1200-digit Decimal tests validate exact binary64 inputs, including high offsets and extreme magnitudes; public/adapter/registry routes match FE bit-for-bit | no | yes | public function and production registry delegate to FE and fail closed if unavailable; legacy FP kernel remains an explicit research fallback, not the high-precision oracle | **FE_COMPOSITE** |
| `robust_ewma` | `SMOOTH:robust_ewma` | lagged finite-only rolling mean/sample std (10 rows, min 2), clip, then EWMA | `FE_COMPOSITE:long_robust_ewm.lagged_robust_ewma:v1` | pure Polars statistics and recurrence | tested masks, values and causal prefixes; bounded 240k-row A/B | no | yes | registry uses the distinct long-panel recipe; `ts_robust_ema` has different semantics; no DSL/export admission | **FE_COMPOSITE** |
| `kama` | `SMOOTH:kama` | Kaufman adaptive MA (recursive, ER lookback) | `KAMA` | technical | none/partial | no | yes | FE `KAMA` is the technical TA-Lib family with different period semantics; FP KAMA is recursive forward-only with explicit ER window and shift(1) causality — not numerically interchangeable | **FP_NATIVE** |
| `one_sided_iir_lowpass` | `SMOOTH:one_sided_iir_lowpass` | one-pole IIR `y_t=(1-α)y_{t-1}+αx_{t-1}`; non-finite lag resets state | native long-panel recipe `long_ewm.lagged_iir_lowpass` | `FE_COMPOSITE:long_ewm.lagged_iir_lowpass:v1`; exact alpha, prior-row timing, and gap reset | exact; stable per-asset order restores interleaved rows | no | yes | Production search domain is alpha ∈ [0.05, 0.5]; direct transform accepts (0, 1]. This recipe covers native long-panel execution, not FE DSL or export support. | **FE_COMPOSITE** |
| `kalman_local_level` | `SMOOTH:kalman_local_level` | scalar Kalman local-level filter (lagged); non-finite lag resets state | — | — | none | no | yes | no FE equivalent (stateful filter); FP fitted/state recursion | **FP_NATIVE** |
| `event_decay` | `EVENT_DECAY:short_halflife` | causal short-halflife event persistence | FE `event_*` decay family exists (e.g. `erd_recency_decay`, `ts_decay_exp_window`) | — | none | no | yes | FE event-decay operators are separate authoring surfaces with different halflife/event-response semantics; FP `event_decay` (DLIB-FP-014) is the eligibility-engine executable | **FP_NATIVE** |

## Volatility

| transform | semantic id | math semantics | FE equivalent operator | FE semantic metadata | confidence | fitted? | causal? | FP-native reason | implementation_origin |
|---|---|---|---|---|---|---|---|---|---|
| `volatility_scale` | `VOL_SCALE:realized` | scale values by lagged realized vol to target | — (FE `scale` is CS; no per-asset lagged-vol scale op) | — | none | no | yes | no FE equivalent (FE has no `x * target/lagged_std` per-asset operator); FP keeps lagged-vol authority | **FP_NATIVE** |
| `volatility_scale_returns` | `VOL_SCALE_RETURNS:realized` | scale returns by lagged vol | — | — | none | no | yes | no FE equivalent | **FP_NATIVE** |
| `realized_volatility` | — | lagged realized vol | `ts_std` (per-asset) | causal | partial — FE `ts_std` current-inclusive | no | yes | lagged semantics + FP annualization | **FP_NATIVE** |

## Neutralization

| transform | semantic id | math semantics | FE equivalent operator | FE semantic metadata | confidence | fitted? | causal? | FP-native reason | implementation_origin |
|---|---|---|---|---|---|---|---|---|---|
| `ols_neutralize` | `NEUTRAL:ols` | per-date finite-row least-squares residual, effective-rank residual DOF | `long_neutralization.ols_effective_rank` | recipe `FE_COMPOSITE:long_neutralization.ols_effective_rank:v1` | parity tests include rank deficiency, zero columns and ill conditioning | no (stateless per-date) | yes | — | **FE_COMPOSITE** |
| `compute_exposures` | `EXPOSURE:compute` | per-date exposure coefficients | `cs_regression` / `cs_neutralize` | — | none | no | yes | FE `cs_regression` returns mode-parameterized output (resid/beta/fit), not the coefficient-per-exposure-frame FP returns; no direct equivalent | **FP_NATIVE** |
| `industry_neutral` | `INDUSTRY_NEUTRAL:SW_L1` | semantic alias → `ols_neutralize` kernel | `long_neutralization.ols_effective_rank` | same OLS composite recipe | same kernel | no | yes | caller supplies exposure columns | **FE_COMPOSITE** |
| `size_neutral` | `SIZE_NEUTRAL:log_mktcap` | semantic alias → `ols_neutralize` | `long_neutralization.ols_effective_rank` | same OLS composite recipe | same kernel | no | yes | caller supplies exposure columns | **FE_COMPOSITE** |
| `dual_neutral` | `DUAL_NEUTRAL:industry_size` | semantic alias → `ols_neutralize` | `long_neutralization.ols_effective_rank` | same OLS composite recipe | same kernel | no | yes | caller supplies exposure columns | **FE_COMPOSITE** |

## Decomposition / regime / research-only

| transform | semantic id | math semantics | FE equivalent operator | FE semantic metadata | confidence | fitted? | causal? | FP-native reason | implementation_origin |
|---|---|---|---|---|---|---|---|---|---|
| `hp_filter` / `hp_decompose` | — | full-sample HP objective (symmetric, non-causal) | FE `ts_hp`-style? none in daily | — | none | no | **no** (OFFLINE_ONLY) | full-sample zero-phase; not production-causal; no FE daily equivalent | **FP_NATIVE** |
| `bandpass_filter` / `extract_cycle` / `christiano_fitzgerald_filter` / `wavelet_*` | — | full-series filter banks | FE spectral family is separate authoring | — | none | no | **no** (OFFLINE_ONLY) | offline research decompositions | **FP_NATIVE** |
| `detect_correlation_regime` | — | full-sample regime detector; thresholds are computed and consumed in one call | — | — | none | no reusable fit/apply state | **no** (RESEARCH_ONLY) | full-sample statistics are non-causal; no persisted fit state | **FP_NATIVE** |

`requires_fit` marks reusable fitted state consumed by a later apply call.
`detect_correlation_regime` and `impute_with_fallback` estimate full-sample
statistics inside each call. They remain `RESEARCH_ONLY` (`causal_safe=False`)
with `requires_fit=False`.

## Routing summary (R61-FI-041)

- **FE_OPERATOR (5):** `cs_rank`, `cs_demean`, `cs_winsor`, `forward_fill`, `missing_indicator`.
- **FE_COMPOSITE (12):** `trailing_sma`, `rolling_mean`, `trailing_median`,
  `rolling_std`, `rolling_zscore`, `ewma`, `one_sided_iir_lowpass`,
  `robust_ewma`, `ols_neutralize`, `industry_neutral`, `size_neutral`,
  `dual_neutral`.
- **FP_NATIVE (26):** everything else. FE-backed transforms retain their
  FP-native kernel for explicit research fallback (plan §26 F3). Production
  execution requires FE and fails closed when FE is unavailable; FE remains
  an optional dependency of FP.

Parity note on `cs_zscore` (the one *near*-exact that is deliberately NOT
routed): FE `zscore` equals FP `cs_zscore` with maxabs 0.0 on all normal
panels, but the singleton-row degenerate semantics differ (FE all-NaN vs FP
`constant_value` 0.0).  Because singleton rows are reachable, routing would
silently change production output on those rows; the transform therefore
stays FP-native and the divergence is documented (registry comment +
`test_cs_zscore_is_not_routed_singleton_semantics_differ`).
