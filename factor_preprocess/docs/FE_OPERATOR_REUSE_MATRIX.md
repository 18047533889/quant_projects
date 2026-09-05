# FE Operator Reuse Matrix (R61-FI-040)

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
  `adapters/fe_operator.py`; FP kernel retained as deprecated fallback) or
  `FP_NATIVE`.

Parity evidence: `tests/test_fe_operator_parity.py` (per-FE-backed-transform
parity test; tolerance contract explicit — cross-sectional and ffill exact
`atol=0.0`, OLS `atol=1e-9`/`rtol=1e-7`).  Registry routing tests:
`tests/test_transform_metadata.py`.

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
| `missing_indicator` | `MISSING_INDICATOR:binary` | binary 1/0 missingness | `is_null` | elementwise, unannotated tags/stage | none | no | yes | FE `is_null` is an elementwise *mask* operator (unannotated, no pandas_numpy production binding parity), while FP returns float 1.0/0.0 per the missingness contract — FP-native indicator carries its own output-channel semantics | **FP_NATIVE** |
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
| `ewma` | `SMOOTH:ewma` | lagged EWMA `ewm(halflife, adjust=False)` on `shift(1)` | `ts_ema` (`EMA`) | tags `(smoothing,)` stage `smoothing` causal `causal` stateless | **partial** — FE `ts_ema` is a **span-parameterized trailing-INCLUSIVE** EMA (span 3 → inclusive `ewm(span=3, adjust=False)`); FP is **halflife-parameterized and lagged** (current excluded). Different parameterization AND different current-bar inclusion → not a safe duplicate | no | yes | halflife vs span + lagged-vs-inclusive current-bar semantics; FP causal smoothing contract excludes current obs (`shift(1)`) | **FP_NATIVE** |
| `trailing_sma` | `SMOOTH:trailing_sma` | lagged trailing SMA `[t-window, t-1]` | `ts_mean` | stage transform causal `causal` stateless | **partial** — FE `ts_mean` window-3 output == FP `trailing_sma` **shifted +1** (FE includes current bar); same window/min_periods numeric but different alignment | no | yes | FE trailing-mean is current-inclusive; FP strictly-lagged (prefix-invariant) contract requires `shift(1)` | **FP_NATIVE** |
| `trailing_median` | `SMOOTH:trailing_median` | lagged trailing median | `ts_median` | causal | **partial** — current-inclusive vs lagged (same as SMA) | no | yes | lagged current-bar semantics | **FP_NATIVE** |
| `rolling_mean` | `SMOOTH:trailing_sma` | causal rolling mean per asset (long frame) | `ts_mean` | causal | **partial** — current-inclusive vs lagged | no | yes | lagged current-bar semantics | **FP_NATIVE** |
| `rolling_std` | — | causal rolling std (lagged) | `ts_std` | causal | **partial** — current-inclusive vs lagged | no | yes | lagged current-bar semantics | **FP_NATIVE** |
| `rolling_zscore` | — | causal rolling zscore (lagged stats) | `ts_zscore` | tags `(zscore,smoothing)` stage transform causal `causal` | **partial** — FE `ts_zscore` includes current bar by default (`includes_current_bar=True`); FP normalizes current against *lagged* stats | no | yes | current-bar inclusion differs; FP rolling_zscore is the lagged variant | **FP_NATIVE** |
| `robust_ewma` | `SMOOTH:robust_ewma` | EWMA on winsorized lagged values | `ts_robust_ema` | experimental | none/partial | no | yes | FP winsorizes against a lagged rolling mean/std window=10 then EWMA — FE `ts_robust_ema` semantics differ (robustness construction not the same); unannotated experimental | **FP_NATIVE** |
| `kama` | `SMOOTH:kama` | Kaufman adaptive MA (recursive, ER lookback) | `KAMA` | technical | none/partial | no | yes | FE `KAMA` is the technical TA-Lib family with different period semantics; FP KAMA is recursive forward-only with explicit ER window and shift(1) causality — not numerically interchangeable | **FP_NATIVE** |
| `one_sided_iir_lowpass` | `SMOOTH:one_sided_iir_lowpass` | one-pole IIR `y=(1-α)y+αx` lagged | `ts_super_smoother` / filters | — | none | no | yes | FE filter family differs (super-smoother is 2-pole); FP one-pole α-parameterized with shift(1) | **FP_NATIVE** |
| `kalman_local_level` | `SMOOTH:kalman_local_level` | scalar Kalman local-level filter (lagged) | — | — | none | no | yes | no FE equivalent (stateful filter); FP fitted/state recursion | **FP_NATIVE** |
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
| `ols_neutralize` | `NEUTRAL:ols` | per-date cross-sectional OLS residual against exposures (`lstsq`, `min_observations`, `add_intercept`) | `cs_neutralize` | overhaul `cross_sectional_regression`; params `y/exposures/group/weight/add_intercept/min_obs`; causal `cross_sectional` | **exact** for `add_intercept=False`, joint exposures (`maxabs ~2e-16` == LAPACK rounding) | no (stateless per-date) | yes | — | **FE_OPERATOR** (`fe_operator_id=cs_neutralize`) |
| `compute_exposures` | `EXPOSURE:compute` | per-date exposure coefficients | `cs_regression` / `cs_neutralize` | — | none | no | yes | FE `cs_regression` returns mode-parameterized output (resid/beta/fit), not the coefficient-per-exposure-frame FP returns; no direct equivalent | **FP_NATIVE** |
| `industry_neutral` | `INDUSTRY_NEUTRAL:SW_L1` | semantic alias → `ols_neutralize` kernel | `cs_neutralize` | — | **exact** (kernel == ols_neutralize) | no | yes | routed as alias of `ols_neutralize` | **FE_OPERATOR** (`fe_operator_id=cs_neutralize`) |
| `size_neutral` | `SIZE_NEUTRAL:log_mktcap` | semantic alias → `ols_neutralize` | `cs_neutralize` | — | **exact** | no | yes | routed as alias of `ols_neutralize` | **FE_OPERATOR** (`fe_operator_id=cs_neutralize`) |
| `dual_neutral` | `DUAL_NEUTRAL:industry_size` | semantic alias → `ols_neutralize` | `cs_neutralize` | — | **exact** | no | yes | routed as alias of `ols_neutralize` | **FE_OPERATOR** (`fe_operator_id=cs_neutralize`) |

## Decomposition / regime / research-only

| transform | semantic id | math semantics | FE equivalent operator | FE semantic metadata | confidence | fitted? | causal? | FP-native reason | implementation_origin |
|---|---|---|---|---|---|---|---|---|---|
| `hp_filter` / `hp_decompose` | — | full-sample HP objective (symmetric, non-causal) | FE `ts_hp`-style? none in daily | — | none | no | **no** (OFFLINE_ONLY) | full-sample zero-phase; not production-causal; no FE daily equivalent | **FP_NATIVE** |
| `bandpass_filter` / `extract_cycle` / `christiano_fitzgerald_filter` / `wavelet_*` | — | full-series filter banks | FE spectral family is separate authoring | — | none | no | **no** (OFFLINE_ONLY) | offline research decompositions | **FP_NATIVE** |
| `detect_correlation_regime` | — | full-sample regime detector | — | — | none | yes | **no** (RESEARCH_ONLY) | full-sample fitted; research-only | **FP_NATIVE** |

## Routing summary (R61-FI-041)

- **FE_OPERATOR (8):** `cs_rank`, `cs_demean`, `cs_winsor`, `forward_fill`,
  `ols_neutralize`, `industry_neutral`, `size_neutral`, `dual_neutral`.
- **FP_NATIVE (35):** everything else.  Each FE-backed transform keeps its
  FP-native kernel as a retained deprecated fallback (plan §26 F3) — no
  kernel was deleted.  The FE adapter (`adapters/fe_operator.py`) is lazy:
  FE is not a hard FP dependency; when FE is unavailable the registry
  executes the FP-native kernel.

Parity note on `cs_zscore` (the one *near*-exact that is deliberately NOT
routed): FE `zscore` equals FP `cs_zscore` with maxabs 0.0 on all normal
panels, but the singleton-row degenerate semantics differ (FE all-NaN vs FP
`constant_value` 0.0).  Because singleton rows are reachable, routing would
silently change production output on those rows; the transform therefore
stays FP-native and the divergence is documented (registry comment +
`test_cs_zscore_is_not_routed_singleton_semantics_differ`).
