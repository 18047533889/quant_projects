# FactorPreprocess legacy migration audit

Scope: read-only audit of the server tree under `/home/shw/quant_projects`; no legacy or production files were modified. The blueprint requires a split between `StatelessTransform` and fold-local `FittedTransform`, and explicitly forbids `fit_transform(full_sample)` before a train/test split.

## files found

Primary files (canonical non-worktree paths):

- `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/AlphaPurifier.py`
- `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/APr_utils.py`
- `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/Exposures.py`
- `/home/shw/quant_projects/toolkit/cross_sectional.py`
- `/home/shw/quant_projects/toolkit/alpha_tools/__init__.py`
- `/home/shw/quant_projects/toolkit/alpha_tools/library.py`
- `/home/shw/quant_projects/toolkit/alpha_tools/generated_library.py`
- `/home/shw/quant_projects/toolkit/alpha_tools/inactive_library.py`
- `/home/shw/quant_projects/toolkit/alpha_tools/registry.py`
- `/home/shw/quant_projects/factor_layer/factor_indicators_lysj/evaluation/preprocessing.py`
- `/home/shw/quant_projects/factor_layer/factor_indicators_lysj/evaluation/labels.py`
- `/home/shw/quant_projects/factor_layer/factor_indicators_lysj/evaluation/backtest.py`
- `/home/shw/quant_projects/factor_layer/factor_indicators_lysj/factor_vwap_reversion.py`
- `/home/shw/quant_projects/factor_layer/factor_indicators_lysj/factor_evaluation_framework.py`

The requested `find` also locates historical/corpus copies, which are not production candidates:

- `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/archive/origin_Alphapurify/alphapurify/AlphaPurifier.py`
- `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/archive/origin_Alphapurify/alphapurify/APr_utils.py`
- `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/archive/origin_Alphapurify/alphapurify/Exposures.py`
- `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/archive/origin_Alphapurify/examples/AlphaPurifier.py`
- `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/archive/origin_Alphapurify/examples/Exposures.py`
- `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/archive/origin_Alphapurify/alphapurify/APr_utils.py`
- `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/history_module/Exposures_0505_v_next.py`
- `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/history_module/Exposures_legacy.py`

`toolkit/alpha_tools` is primarily a factor-compute library, not a preprocessing package. `factor_indicators_lysj` is primarily factor computation/evaluation; only its `evaluation/preprocessing.py` is a direct preprocessing candidate.

## function-level migration matrix

| file | symbol / signature | decision | target FP module | stateless/fitted | reason | required tests | leakage risk |
|---|---|---|---|---|---|---|---|
| `toolkit/cross_sectional.py` | `cs_rank(factor_values)` | REUSE_AFTER_TEST | `cross_sectional` | stateless, cross-sectional | Pure pandas/numpy rank; same-date permutation invariant; explicit all-constant/no-signal NaN behavior | ties, NaNs, permutation, batch/chunk, reference/fast | low; validate no cross-date mixing |
| same | `cs_zscore`, `cs_robust_zscore` | REUSE_AFTER_TEST | `cross_sectional` | stateless, cross-sectional | Pure implementations; robust version is median/MAD with 1.4826 | ddof contract, zero variance/MAD, NaNs, parity | low |
| same | `cs_winsorize`, `cs_demean`, `cs_scale`, `cs_minmax` | REUSE_AFTER_TEST | `cross_sectional` | stateless, cross-sectional | Small pure transforms; preserve missingness only after explicit contract review | bounds, constants, NaNs, permutation | low |
| same | `cs_rank_gauss`, `cs_quantile_bucket`, `cs_signed_power` | REUSE_AFTER_TEST | `cross_sectional` | stateless, cross-sectional | Pure scipy/numpy rank-to-normal and buckets; useful multi-channel representations | ties, clipping, inverse-CDF bounds, bucket edges | low |
| same | `apply_cross_sectional_transform` and private helpers | REWRITE | `policy/registry` | stateless dispatcher | Coupled to `toolkit.registry.is_allowed_transform`; FP must own typed TransformSpec and metadata, not legacy registry | unknown transform, policy gating, metadata, extraction | low, but policy bypass risk |
| `AlphaPurifier.py` | `AlphaPurifier.__init__`, `get_methods`, `winsorize`, `neutralize`, `standardize`, `to_result` | DISCARD | `pipeline` (API concept only) | mutable wrapper; mixed | Method chaining mutates a Polars frame, hides config/state, and is coupled to legacy wildcard utilities; not a valid FP protocol | no direct port; extraction/API-boundary tests | high if called over full data; no fit boundary |
| `APr_utils.py` | `mean_std_winsorize`, `mad_winsorize`, `volatility_winsorize`, `iqr_winsorize`, `quantile_winsorize` | REWRITE | `cross_sectional.winsor` | stateless, cross-sectional | Algorithms are portable but Polars join/group implementation has edge-case/null/std contracts to establish | golden numeric cases, constant groups, NaN/null, parity | low |
| same | `rolling_quantile_winsorize(df, trade_date_col, symbol_col, factor_col, window, ...)` | REWRITE | `timeseries` | stateless causal-looking TS | Sorts by symbol/date and uses trailing rolling quantiles, but `min_periods=1` includes current value and needs an explicit causal convention | future-poison, first-window, irregular dates, symbol isolation | medium: current observation and unspecified closed boundary |
| same | `boxcox_compress_winsorize` | REWRITE / ADVANCED | `advanced/power` | stateless per-date, cross-sectional | Uses a per-date min shift plus fixed lambda; epsilon and positivity semantics are ad hoc; not learned Box-Cox | negative/constant data, lambda=0, parity | low; non-fitted here, but cross-sectional min is contemporaneous |
| same | `zscore_winsorize`, `rankgauss_winsorize`, `tanh_winsorize` | REWRITE | `cross_sectional.winsor` / `cross_sectional` | stateless, cross-sectional | Candidate algorithms; duplicate functionality exists in `toolkit/cross_sectional`; consolidate rather than port duplicates | reference equivalence, ties, clipping, constants | low |
| same | `huber_winsorize`, `ransac_winsorize` | REFERENCE_ONLY | `advanced/robust` | per-cross-section fitted model | sklearn robust regressors are fit inside each date and are not reusable fitted state; unusual cleaning semantics and heavy dependency | only if explicitly enabled: deterministic seed, small groups, outlier oracle | medium; fitting must remain date-local and label-free |
| same | `preprocess_for_neutralization(...)` | REWRITE | `neutralization/design_matrix` | stateless per invocation | Useful design-matrix idea, but `get_dummies` category mapping is not persisted and `scale_X` is described as global; drops rows and returns pandas | category stability, train/test vocabulary, missing rows, scaling scope | high for fitted/global scaling and category drift |
| same | `multiOLS_neutralize(...)` | REUSE_AFTER_TEST (reference first) | `neutralization/ols` | stateless, cross-sectional | Per-date residual OLS with pseudoinverse/lstsq and known exposures matches production core; requires explicit rank/conditioning and missing policy | residual exposure ~0, rank deficiency, same-day permutation, missingness | low when exposures are available at t; high if exposures are future/as-of unsafe |
| same | `lasso_neutralize`, `ridge_neutralize`, `elasticnet_neutralize`, `kernelridge_neutralize`, `bayesianridge_neutralize` | REWRITE / ADVANCED | `neutralization/regularized` | fitted per date, cross-sectional | sklearn model is fit separately on each date and discarded; no reusable state, scaling/category contracts, or solver audit | coefficient/residual oracle, alpha sensitivity, deterministic solver, rank deficiency | medium; labels are factor values `y`, not forward labels, but exposure timing must be checked |
| same | `polynomial_neutralize` | REWRITE / ADVANCED | `neutralization/design_matrix` | fitted per date, cross-sectional | `PolynomialFeatures.fit_transform(X)` is local design expansion, not full-sample leakage, but expansion config and feature order are not state metadata | feature-order hash, degree/interactions, residual sanity | medium if globally scaled/category learned |
| same | `huber_neutralize`, `rank_neutralize`, `theilsen_neutralize` | REFERENCE_ONLY | `neutralization/advanced` | fitted per date | Robust/regression variants are research-tier and sklearn-coupled; `rank_neutralize` uses RANSAC despite its name | robust regression oracle, small-group behavior, outlier influence | medium |
| same | `randomforest_neutralize`, `GBDT_neutralize` | REFERENCE_ONLY | `neutralization/nonlinear` | fitted per date | Supervised nonlinear residualization; expensive, model semantics are not FP production core, and sklearn is required | strict opt-in, seed, exposure residual sanity, cost | high: supervised `y=factor` is acceptable only for contemporaneous representation; never use forward returns/labels |
| same | `PCA_neutralize`, `ICA_neutralize` | REWRITE / ADVANCED | `fitted/decomposition` | fitted, cross-sectional in current code | `pca.fit_transform(X)` / `ica.fit_transform(X)` learns and discards components per date; cannot transform test data with preserved state; PCA/ICA are advanced and currently not clearly factor-input representations | train/test poison, component sign/order, feature hash, numerical parity | high if run across full sample; current per-date fit is not reusable |
| same | `partialcorrelation_neutralize` | REFERENCE_ONLY | `neutralization/diagnostics` | stateless per date | Computes correlation-style diagnostic, not a general residual channel; output shape/column behavior is inconsistent | compare with OLS residual and missing policy | medium if used as transform; no future labels allowed |
| same | `zscore_standardize`, `robust_zscore_standardize`, `rank_standardize`, `rank_gaussianize_standardize`, `normal_scores_standardize` | REUSE_AFTER_TEST | `cross_sectional` | stateless, cross-sectional | Production-core methods; duplicate implementations should be consolidated with `toolkit/cross_sectional`; NaN/zero-scale behavior differs | golden formulas, ties, zero scale, parity | low |
| same | `minmax_standardize`, `quantile_binning_standardize`, `log_zscore_standardize` | REWRITE | `cross_sectional` / `advanced/power` | stateless, cross-sectional | Potentially useful but contracts differ from FP production list; min/max and log shift need explicit semantics | monotonicity, constants, negatives, missingness | low |
| same | `rolling_standardize`, `rolling_robust_standardize`, `rolling_minmax_standardize` | REWRITE | `timeseries` | stateless causal TS | Trailing grouped windows; source does not consistently make “exclude current” explicit and may leave temporary columns/order changes | future-poison, closed boundary, warmup, symbol/date ordering | medium |
| same | `volatility_scaling_standardize(..., shift_vol=True)`, `EWMA_standardize(...)` | REUSE_AFTER_TEST | `timeseries/volatility` | stateless causal TS | Matches production-core causal scale channel when shift is enforced; EWMA implementation must be checked for initial value and decay convention | one-step lag, future-poison, zero vol, lambda bounds, parity | medium if `shift_vol=False`; low with strict lag |
| same | `yeo_johnson_standardize`, `boxcox_standardize` | REFERENCE_ONLY | `advanced/power` | effectively fitted if lambda estimated | Advanced transformations; learned lambda/state is not exposed or persisted, so current direct use cannot cross a train/test boundary | fold-local fit, lambda/state serialization, negatives, poison | high if estimator sees full sample |
| `Exposures.py` | `PortfolioExposures._standardize_exposures`, `PureExposures._standardize_frame` | REFERENCE_ONLY | `neutralization/exposure` | stateless, cross-sectional | Same-date standardization is portable but classes are evaluator/report wrappers; their future-return pipelines are QE/reporting, not FP | isolate pure standardizer and test separately | high at class level because future return is used |
| same | `PortfolioExposures._cross_section_ols`, `PureExposures._cross_section_ols` | REFERENCE_ONLY | `neutralization/diagnostics` | stateless per date, supervised | OLS predicts `fut_ret` from exposures, explicitly uses future labels; it is attribution/evaluation, not factor representation | must not enter FP; future-label sentinel | critical leakage if reused |
| same | `_prepare_future_return_frame`, `_build_beta_panel`, `_build_attribution_panel`, `calc_stats`, plotting methods | DISCARD | none (QE/report layer) | mixed | Future return construction, attribution, cumulative returns, Plotly and evaluator base-class coupling | retain only QE regression/report tests | critical: forward shifts and report outputs |
| `factor_indicators_lysj/evaluation/preprocessing.py` | `load_factor_series`, `load_vwap_series`, `align_on_overlap_reindex` | CORPUS_ONLY | none | stateless I/O preparation | Evaluation-specific factor/VWAP loading and overlap alignment; not a generic FP transform and coupled to `EvalConfig` | preserve as evaluation regression corpus only | low for alignment; not production FP |
| same | `rolling_zscore(series, window)` | REFERENCE_ONLY | `timeseries` | stateless TS | Clean pandas reference implementation; current window includes `x_t`, so causal contract must be changed or documented | future-poison, window closure, warmup, zero std | medium |
| same | `winsorize_by_day(series, date_index, q)` | CORPUS_ONLY | `cross_sectional.winsor` | stateless, cross-sectional/day | Evaluation-only per-day helper; useful numeric golden cases but not a public FP API | quantile/ties/NaN golden tests | low |
| same | `prepare_base`, `prepare_horizon`, `prepare_status` | CORPUS_ONLY | none | mixed | Evaluation pipeline computes forward returns and eligibility; labels/future values make it unsuitable for FP | forward-return direction and tail-mask regression only | critical in FP context |
| `toolkit/alpha_tools/generated_library.py` | `robust_zscore(series, window, clip)` and all rolling factor functions | CORPUS_ONLY | none | stateless TS factor compute | These are generated alpha operators, not representation transforms; retain formulas/corpus metadata, do not duplicate in FP | FE/operator parity only | low-to-medium depending on operator; out of FP scope |
| `toolkit/alpha_tools/library.py` | `classify_volume_regime`, `decompose_overnight_intraday` | CORPUS_ONLY | none | stateless TS factor compute | Factor features/regime classification, not model-input preprocessing | factor corpus tests | low; out of FP scope |
| `toolkit/alpha_tools/inactive_library.py` | `robust_volume_zscore` | DISCARD (inactive) | none | stateless TS | Explicitly deactivated alpha tool; no FP migration value | none beyond inactive registry integrity | low |
| `factor_indicators_lysj/evaluation/labels.py` | `MarketStatusLabel` | CORPUS_ONLY | none | stateless TS label generation | Produces market-state labels from realized volatility; labels are not representation transforms | label causality and boundary tests | high if accidentally used as an input transform |
| `factor_indicators_lysj/evaluation/backtest.py` | rolling rank/group helpers | CORPUS_ONLY | none | stateless TS | Backtest position/group logic, not FP | backtest regression only | high if reused as input preprocessing |

### Complete APr_utils family inventory

`APr_utils.py` contains 40+ functions. The matrix above groups only identical semantic families; the exact discovered symbols are: `map_freq`; `mean_std_winsorize`, `mad_winsorize`, `volatility_winsorize`, `iqr_winsorize`, `quantile_winsorize`, `rolling_quantile_winsorize`, `boxcox_compress_winsorize`, `zscore_winsorize`, `rankgauss_winsorize`, `tanh_winsorize`, `huber_winsorize`, `ransac_winsorize`; `preprocess_for_neutralization`; `multiOLS_neutralize`, `lasso_neutralize`, `ridge_neutralize`, `elasticnet_neutralize`, `polynomial_neutralize`, `kernelridge_neutralize`, `huber_neutralize`, `rank_neutralize`, `theilsen_neutralize`, `randomforest_neutralize`, `GBDT_neutralize`, `PCA_neutralize`, `ICA_neutralize`, `bayesianridge_neutralize`, `partialcorrelation_neutralize`; and `zscore_standardize`, `robust_zscore_standardize`, `minmax_standardize`, `rank_standardize`, `rank_gaussianize_standardize`, `rolling_standardize`, `rolling_robust_standardize`, `rolling_minmax_standardize`, `volatility_scaling_standardize`, `EWMA_standardize`, `normal_scores_standardize`, `quantile_binning_standardize`, `log_zscore_standardize`, `yeo_johnson_standardize`, `boxcox_standardize`. There are no discovered missing-indicator, imputation, freshness/age, or explicit decay-channel functions in these requested legacy files.

## stateless vs fitted inventory

**Stateless candidates:** all same-date rank/zscore/robust-zscore/winsor/demean/rank-Gauss transforms; fixed min-max/bucketing/power transforms; date-local OLS residualization with known, as-of-safe exposures; trailing rolling z/robust/quantile/min-max/volatility/EWMA transforms when the window boundary is explicit and causal; `map_freq` (metadata only).

**Fitted or model-fitting candidates:** every sklearn regression neutralizer (Lasso/Ridge/ElasticNet/KernelRidge/Huber/Theil-Sen/RANSAC/RandomForest/GBDT/BayesianRidge), PCA/ICA, and estimated Box-Cox/Yeo-Johnson. Legacy code generally fits and immediately emits residuals/components; it does not expose a reusable state. Per-date fitting is not a substitute for a train/test `FittedTransform` protocol.

## cross-sectional vs time-series inventory

- Cross-sectional: all `cs_*`; all ordinary winsorizers; rank Gaussianization; standardizers; neutralization/design matrices; `Exposures` standardization and OLS (but evaluator-only for the latter).
- Time-series: `rolling_quantile_winsorize`, rolling standardizers, volatility scaling, EWMA, `evaluation.preprocessing.rolling_zscore`, and alpha-tool rolling functions.
- Mixed/evaluation: `prepare_base` and horizon/status preparation align time series, then apply day-level cross-sectional clipping and forward labels.

## supervised vs unsupervised

- Unsupervised: rank, scale, winsor, Gaussianization, demean, rolling/volatility/EWMA, and exposure-only transformations.
- Supervised/model-fitting: regression neutralizers learn coefficients with factor values as `y`; nonlinear regressors are especially high-cost and research-only. This is not prediction-label supervision, but coefficients are still learned state and require fold-local/as-of contracts.
- Explicit future-label use: `Exposures._cross_section_ols` regresses `fut_ret`; `evaluation.preprocessing.make_forward_vwap_return`, `prepare_horizon`, `prepare_status`, and `MarketStatusLabel` produce/use labels. These must never be migrated as FP transforms.

## causal vs non-causal

- Same-date cross-sectional transforms are causal with respect to time only if their inputs are available at that timestamp; they are intentionally non-causal across assets within a date (permutation invariant).
- Trailing windows are intended causal but most legacy implementations include the current observation. `volatility_scaling_standardize` defaults to a one-period shift; `shift_vol=False` is unsafe for a production lane. `EWMA_standardize` requires explicit initial-state/lag tests.
- Non-causal/full-sample hazards: global `scale_X` in `preprocess_for_neutralization`; any estimated power transform if fit over the full frame; evaluator functions using forward returns; class-level `Exposures` calculations. The code does not establish a safe causal fitted-state boundary.

## legacy fit_transform safety findings

`rg` found no literal `fit_transform(full_sample)` call. It did find dangerous generic `fit_transform` calls:

- `APr_utils.py:1599`: `PolynomialFeatures.fit_transform(X)` inside each date group. This is not full-sample by itself, but learned feature metadata is discarded; rewrite with explicit deterministic design config/order.
- `APr_utils.py:2270`: `PCA.fit_transform(X)` inside each date group. State is discarded, components cannot be applied to test data, and component identity can vary date-to-date.
- `APr_utils.py:2355`: `FastICA.fit_transform(X)` inside each date group. Same state-loss and instability issue, plus advanced/research tier.

External `.fit(...).predict/transform` calls are also inside date-group helpers for RANSAC, Lasso, Ridge, ElasticNet, KernelRidge, Huber, Theil-Sen, RandomForest, GBDT, and BayesianRidge. The fitted objects are local variables and are discarded after producing residuals. No legacy function serializes or preserves fitted state for later test transformation.

## FittedState legacy fields

Observed legacy persistence is effectively **none**. `AlphaPurifier` stores only mutable `df`, `factor_name`, date/symbol column names, and original `cols`; sklearn estimators and decomposition objects are local. No `fit_start`, `fit_end`, feature IDs/order, transform version/config, training-universe definition, learned-parameter hash, fold ID, or state reference was found in the audited preprocessing path. `EvalConfig` stores operational windows/quantiles, not a fitted state.

FP must introduce at least:

- `fit_start`, `fit_end` (and fit/as-of timestamp semantics)
- feature IDs and exact feature order
- transform version and canonical config hash
- training-universe definition/snapshot reference
- learned-parameter hash (plus serialized parameter/state reference)

## coupling findings (DA/FE/report)

- `AlphaPurifier.py` imports `FactorAnalyzer`, legacy `Exposures`, Polars, and Plotly indirectly through the surrounding package; its mutable pipeline and registry are coupled to legacy runtime behavior. Keep only algorithmic formulas after extraction tests.
- `APr_utils.py` is numerically coupled to Polars/Pandas conversion, `joblib`, SciPy, and many sklearn estimators. It has no DA/FE calls, but `preprocess_for_neutralization` performs schema cleaning and dummy encoding that FP should replace with typed context/design-matrix contracts.
- `Exposures.py` subclasses legacy evaluator classes, computes forward returns, writes/plots reports with Plotly, and is QE/report infrastructure rather than FP. Its same-date exposure standardization and OLS are reusable only as isolated reference formulas.
- `toolkit/cross_sectional.py` is the cleanest pure implementation: pandas/numpy plus SciPy normal PPF, with a toolkit registry dependency only in the dispatcher.
- `factor_indicators_lysj/evaluation/preprocessing.py` is pandas/config/evaluation coupled and directly computes forward VWAP labels; preserve as corpus/reference, not production FP.
- `toolkit/alpha_tools` and `factor_indicators_lysj` mostly contain FE-style factor computation, labels, evaluation, and backtest code. Do not rebuild or import those runtimes into FP.

Migration priority: first extract and golden-test the pure cross-sectional functions and date-local OLS; then implement explicit causal rolling/volatility channels and missing/freshness channels (not present in legacy); leave nonlinear regressors, PCA/ICA, and estimated power transforms behind an advanced feature gate with a real `FittedState`.
