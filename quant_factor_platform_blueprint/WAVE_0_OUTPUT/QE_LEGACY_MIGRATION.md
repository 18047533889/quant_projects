# QuantEvaluator Legacy Migration Audit

Audit date: 2026-08-13

Scope: read-only mining of legacy evaluation, analyzer, exposure, preprocessing, admission, agent, and factor-pool code under `/home/shw/quant_projects`. This report is the only file written. Decisions use `REUSE_AFTER_TEST`, `REWRITE`, `REFERENCE_ONLY`, `CORPUS_ONLY`, and `DISCARD` from `AI_GUIDE/00_START_HERE.md`.

## Files Found

The required `find` command was run. Canonical legacy results are below; `.claude/worktrees`, `.replace_backup_*`, `.venv`, and generated `build/lib` copies are excluded from migration sources.

| Path | Size | Language | Role |
|---|---:|---|---|
| `/home/shw/quant_projects/toolkit/cross_sectional.py` | 5,863 B | Python | Pure cross-sectional transforms |
| `/home/shw/quant_projects/factor_layer/factor_evaluation/pipeline.py` | 19,001 B | Python | Single-factor evaluator |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/FactorAnalyzer.py` | 118,560 B | Python | IC, quantile, turnover, risk statistics, plotting |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/APr_utils.py` | 150,657 B | Python | Winsorization, neutralization, standardization |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/pipeline.py` | 24,565 B | Python | Parquet-to-report orchestration |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/Exposures.py` | 27,020 B | Python | OLS exposure attribution and plotting |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/AlphaPurifier.py` | 10,798 B | Python | Chainable preprocessing facade |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/evaluation/pipeline.py` | 15,852 B | Python | Three-stage file-oriented evaluation pipeline |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/evaluation/batch_metrics.py` | 18,443 B | Python | Leakage-aware reference metrics |
| `/home/shw/quant_projects/factor_layer/factor_indicators_lysj/evaluation/pipeline.py` | 7,910 B | Python | Intraday evaluator orchestration |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/pipeline.py` | 35,045 B | Python | Top-level legacy application pipeline; coupled |
| `/home/shw/quant_projects/factor_layer/factor_admission/pipeline.py` | 9,893 B | Python | Admission orchestration; outside QE ownership |
| `/home/shw/quant_projects/raw_data_layer/raw_data_fetching/pipeline.py` | 1,311 B | Python | Data-fetch pipeline; outside QE ownership |
| `/home/shw/quant_projects/factor_engine/pipeline.py` | 150 B | Python | FE compatibility module; outside QE ownership |
| `/home/shw/quant_projects/factor_engine/cleaned_operators/cross_sectional.py` | 161 B | Python | FE compatibility module; outside QE ownership |
| `/home/shw/quant_projects/factor_engine/cleaned_operators/common/cross_sectional.py` | 62,536 B | Python | FE-owned operator kernels; do not migrate to QE |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/factor_engine/pipeline.py` | 150 B | Python | Duplicate FE compatibility module |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/factor_engine/cleaned_operators/cross_sectional.py` | 161 B | Python | Duplicate FE compatibility module |
| `/home/shw/quant_projects/AutoFactorEvaluation-RECONSTRUCT/factor_engine/cleaned_operators/common/cross_sectional.py` | 55,175 B | Python | Duplicate FE implementation |

Archived exact copies, examples, and test corpus found by the same command:

| Path | Size | Decision |
|---|---:|---|
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/archive/origin_Alphapurify/alphapurify/FactorAnalyzer.py` | 118,560 B | CORPUS_ONLY (byte-size-identical canonical copy) |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/archive/origin_Alphapurify/alphapurify/APr_utils.py` | 150,657 B | CORPUS_ONLY |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/archive/origin_Alphapurify/alphapurify/Exposures.py` | 49,088 B | CORPUS_ONLY; older behavior oracle |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/archive/origin_Alphapurify/alphapurify/AlphaPurifier.py` | 10,809 B | CORPUS_ONLY |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/archive/origin_Alphapurify/examples/FactorAnalyzer.py` | 9,968 B | CORPUS_ONLY |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/archive/origin_Alphapurify/examples/Exposures.py` | 6,411 B | CORPUS_ONLY |
| `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/archive/origin_Alphapurify/examples/AlphaPurifier.py` | 8,067 B | CORPUS_ONLY |

No `factor_layer/factor_evaluation/alphapurify/` port exists. The only current implementation is `/home/shw/quant_projects/factor_layer/factor_evaluation_alphapurify/`.

Additional high-priority locations inspected:

- `/home/shw/quant_projects/toolkit/alpha_tools/registry.py` (7,069 B): generated-tool facade/metadata, not QE metric code.
- `/home/shw/quant_projects/factor_layer/factor_admission/catalog.py`: SQLite admission/evaluation ledger, FactorAssets ownership.
- `/home/shw/quant_projects/factor_layer/factor_agent/`: 58 files, LLM/tool orchestration, no portable QE kernels.
- `/home/shw/quant_projects/factor_layer/factor_indicators_lysj/`: 13 Python files, intraday evaluation reference implementation.
- `/home/shw/quant_projects/factor_layer/factor_pool/`: one README, no implementation.

## Function-Level Migration Matrix

Target names are proposed package boundaries, not instructions to preserve legacy APIs. `labels.*` means label adapters/contracts, not implicit label guessing inside metrics.

### Core reconstructed batch metrics

| File | Symbol and signature | Decision | Target module | Description / reason | Required tests |
|---|---|---|---|---|---|
| `evaluation/batch_metrics.py` | `normalize_market_frame(frame: pd.DataFrame) -> pd.DataFrame` | REWRITE | `quant_evaluator.adapters.pandas` | Useful key/finite validation, but hard-codes OHLCV and performs market-data normalization QE should not own. | duplicate/null keys; ordering; adapter contract |
| same | `split_boundaries(frame, config) -> SplitBoundaries` | REWRITE | `quant_evaluator.validation.splits` | Date split semantics useful; config is untyped and fixed fractional split is too narrow. | boundary edge cases; min dates; deterministic order |
| same | `apply_point_in_time_universe(market_frame, universe_frame, *, universe_id=None)` | DISCARD | DA/context adapter | PIT universe assembly is DataAccess ownership. QE consumes a supplied mask. | DA adapter integration only |
| same | `purify_series(series, *, mad_multiplier, min_assets)` | REUSE_AFTER_TEST | `factor_preprocess.reference.cross_sectional` | Clear MAD clipping plus z-score reference; belongs to FactorPreprocess, not QE. | NaN/Inf; zero MAD; min assets; order and dtype |
| same | `price_forward_returns(frame, horizons, price_field, *, entry_lag)` | REFERENCE_ONLY | `quant_evaluator.labels.reference` | Explicit entry/exit convention is valuable, but QE must consume `LabelBundle` and never infer price labels silently. | exact shift direction; irregular calendar; zero price; horizons |
| same | `purged_split_mask(index, split, bounds, trading_dates, *, entry_lag, horizon)` | REUSE_AFTER_TEST | `quant_evaluator.validation.purging` | Compact leakage-safe purge reference; needs typed label timing and calendar contract. | exit exactly on boundary; missing dates; all four splits |
| same | `_row_corr(left, right)` | REUSE_AFTER_TEST | `quant_evaluator.metrics.ic.reference` | Vectorized pairwise-finite row correlation with correct zero-variance NaN behavior. | NaN/Inf; constants; batch/chunk parity; float32/64 |
| same | `daily_ic(signal, returns, *, min_assets)` | REUSE_AFTER_TEST | `quant_evaluator.metrics.ic.reference` | Best compact legacy Pearson/average-tie RankIC seed; batch-first panel implementation. | ties; constants; min-assets; sign flip; monotonic transform; order |
| same | `_hac_variance(values, lags)` | REFERENCE_ONLY | `quant_evaluator.metrics.robustness.hac` | Bartlett/Newey-West long-run variance semantics useful; use statsmodels as golden oracle before a fast kernel. | statsmodels parity; lags 0/n; autocorrelated and constant series |
| same | `summary_stats(values, *, hac_lags=0)` | REWRITE | `quant_evaluator.metrics.ic_summary` | Mean/std/IR/sign/t/HAC bundle is useful, but normal approximation and nullable dict schema need explicit MetricSpecs. | ddof; non-annualized ICIR; HAC oracle; NaNs; small n |
| same | `long_short_series(signal, returns, *, min_assets, n_quantiles, cost_bps)` | REWRITE | `quant_evaluator.metrics.probe_portfolio` | Useful weight-turnover-cost semantics; ties can create unequal tails and missing realized returns are filled as zero. | ties; disappearing assets; missing returns; cost arithmetic; neutrality |
| same | `performance_stats(returns, *, annualization, hac_lags=0, holding_period=1)` | REWRITE | `quant_evaluator.metrics.portfolio_stats` | Sharpe/HAC Sharpe/max drawdown reference; sleeve-min drawdown is not a strategy path. | arithmetic/geometric annualization; overlapping sleeves; known drawdown |
| same | `route_factor(metrics, *, coverage)` | DISCARD | FactorAssets policy | Hard-coded admission thresholds and tier names are not QE evidence semantics. | policy tests in FactorAssets only |
| same | `_bh_qvalues(p_values)` | REUSE_AFTER_TEST | `quant_evaluator.metrics.multiple_testing` | Standard Benjamini-Hochberg kernel is portable. | statsmodels parity; ties; unsorted input; 0/1; empty |
| same | `apply_validation_fdr(records, *, alpha)` | REWRITE | FactorAssets policy + QE q-values | Mutates legacy records and downgrades admission route; separate evidence from decision. | q-value attachment; no mutation; policy boundary |
| same | `ranking_rows(records)` | DISCARD | FactorAssets ranking | Legacy tier-specific presentation, not QE math. | none in QE |

### Reconstructed time-series and indicator modules

| File | Symbol and signature | Decision | Target module | Description / reason | Required tests |
|---|---|---|---|---|---|
| `timeseries/scripts/ic.py` | `_safe_kendall_tau(x, y, eps)` | REUSE_AFTER_TEST | `quant_evaluator.metrics.ic.reference` | SciPy Kendall tau-b reference, correctly rejects constants. | SciPy parity; ties; NaN pre-mask; constants |
| same | `_group_corr(df, x_col, y_col, *, min_assets, eps)` | REUSE_AFTER_TEST | `quant_evaluator.metrics.ic.fast` | O(N) grouped sufficient-statistics correlation; suitable seed after parity tests. | cancellation/numerical stability; constants; chunks; shuffled order |
| same | `compute_daily_ic_series(aligned_panel, config, factor_id, eval_run_id, vb)` | REWRITE | `quant_evaluator.metrics.ic` | Good Pearson/RankIC/Kendall behavior, but hard-coded schema, warning buffer, and Python Kendall loop. | all RankIC contract tests; multi-horizon; warning contract |
| `timeseries/scripts/forward_returns.py` | `compute_forward_returns(market_data, horizons, price_col, vb)` | REFERENCE_ONLY | `quant_evaluator.labels.reference` | Explicit forward-label oracle; QE runtime should not fetch/cache or choose price fields. | timing table; symbol isolation; zero/Inf; horizons |
| same | `compute_or_load_forward_returns(...)` and cache helpers | DISCARD | DA | Direct parquet cache duplicates DataAccess/storage ownership. | none in QE |
| `timeseries/scripts/quantile.py` | `_sector_rank(values, industries)` | REFERENCE_ONLY | `quant_evaluator.metrics.exposure.reference` | Useful industry-relative ranking oracle. | missing industry; ties; singleton groups |
| same | `_assign_quantile_ids(sub, config)` | REWRITE | `quant_evaluator.metrics.quantile` | Contains sector-neutral/ranking policy but is DataFrame/config coupled. | average ties; sparse bins; industry mode; monotonicity |
| same | `_aggregate_quantile_returns(...)`, `_side_gross_from_weights(...)`, `_side_weights(...)` | REWRITE | `quant_evaluator.metrics.quantile` / `probe_portfolio` | Valuable equal-weight quantile and top/bottom mechanics; split from metadata and Python dict loops. | weight sum; missing return; tail sizes; sign convention |
| same | `compute_quantile_panels(...)` | REWRITE | `quant_evaluator.metrics.quantile` | Produces quantile returns, top-bottom, coverage, and weight rows, but is monolithic and schema-coupled. | Q1..Q10; coverage denominator; min assets; chunk parity |
| `timeseries/scripts/portfolio.py` | `compute_turnover_series(weight_rows, horizons, factor_id, eval_run_id)` | REUSE_AFTER_TEST | `quant_evaluator.metrics.turnover.reference` | Canonical `0.5 * sum(abs(delta weights))`; first observation intentionally NaN. | entries/exits; asset disappearance; first date; multiple horizons |
| same | `attach_cost_and_path(ls_df, turnover_df, config)` | REWRITE | `quant_evaluator.metrics.probe_portfolio` | Cost and drawdown path useful; imports unused `vectorbt`, fills missing turnover/returns with zero, and loops horizons. | missingness; cost timing; path oracle; multi-horizon |
| `indicator/.../calculator.py` | `_finite_series`, `_sample_stats`, `_safe_std`, `_max_drawdown`, `_t_stat`, `_t_p_value`, `_spearman_corr`, `_effective_annual_factor` | REFERENCE_ONLY | corresponding QE reference metrics | Formula seeds only; embedded in file/report contract and some alignment assumptions are unsafe. | SciPy/statsmodels parity; irregular dates; aligned indexes |
| same | `compute_metrics(series_bundle, method_config)` | REWRITE | `quant_evaluator.registry` + metric modules | Valuable metric inventory: IC/IR, Kendall, quantile monotonicity, top-bottom, Sharpe, drawdown, Calmar, turnover, coverage. Monolithic string-valued matrix and file schema are unsuitable. | MetricSpec per metric; missing inputs; horizon slices; annualization |
| same | `validate_series_bundle(...)` | REWRITE | `quant_evaluator.validation` | Useful required-series checks, but validates file-stage bundles rather than FactorBatch/LabelBundle. | missing columns; mixed horizons; minimum observations |
| same | `MetricBuilder`, scorecard/build/run/I/O helpers | DISCARD | QE output contracts / FactorAssets | Report writing, scorecard routing, and Dev4 adaptation are coupled presentation/policy. | none in QE runtime |
| `evaluation/pipeline.py` | `EvaluationPipelineResult`; `run_evaluation_pipeline(...)` | DISCARD | external orchestration | Reads/writes parquet and logs, patches net=gross, invokes label/admission stages. QE should expose in-memory evaluation only. | external adapter integration only |

### Factor-layer evaluator

| File | Symbol and signature | Decision | Target module | Description / reason | Required tests |
|---|---|---|---|---|---|
| `factor_evaluation/pipeline.py` | `_winsorize_cross_section(series, lower, upper)`; `_zscore_cross_section(series)` | REFERENCE_ONLY | `factor_preprocess.reference` | Straightforward preprocessing oracles; not QE ownership. | NaN/constant/quantile boundaries |
| same | `_build_market_returns(market, horizons)` | REFERENCE_ONLY | `quant_evaluator.labels.reference` | Clear `[t+1,t+h+1]` convention, but label construction must be explicit and external. | shift direction; per-symbol isolation |
| same | `_assign_quantiles(frame, n_quantiles, min_assets)` | REWRITE | `quant_evaluator.metrics.quantile` | Uses arbitrary `rank(method="first")`, contrary to QE average-tie contract. | ties/order invariance; sparse cross-sections |
| same | `_summary_from_series(series)` | REUSE_AFTER_TEST | `quant_evaluator.metrics.ic_summary.reference` | Clean non-annualized mean/std/IR/win-rate reference. | NaN, one value, constant, ddof |
| same | `_compute_daily_ic(frame, ret_col, min_assets)` | REFERENCE_ONLY | `quant_evaluator.metrics.ic.golden` | Slow per-date pandas oracle with correct pairwise drop and average RankIC ties. | external parity and metamorphic suite |
| same | `_compute_period_group_returns(frame, ret_col)` | REUSE_AFTER_TEST | `quant_evaluator.metrics.quantile.reference` | Simple equal-weight quantile-return oracle. | missing bins; NaNs; order |
| same | `_build_quantile_backtest(frame, *, horizon, n_quantiles)` | REWRITE | `quant_evaluator.metrics.probe_portfolio` | Layered holding idea is useful; Python loops and normalization over only nonmissing returns distort exposure. | overlapping holdings; delist/missing return; weight conservation |
| same | `_summarize_long_short(series, annualization_factor)` | REFERENCE_ONLY | `quant_evaluator.metrics.portfolio_stats.golden` | Standard geometric annual return, volatility, Sharpe, max drawdown oracle. | known path; irregular dates; return <= -1 |
| same | `evaluate_factor(config)` | REWRITE | `quant_evaluator.evaluate` | Valuable end-to-end behavior, but single-factor, loads data itself, inner-merges universe, and couples config/I/O. | in-memory contract; batch equivalence; masks; min dates |
| same | `save_results(result, config)` | DISCARD | caller-owned persistence | CSV/parquet/report output is not QE core responsibility. | none |

### AlphaPurify analyzer and exposure modules

| File | Symbol and signature | Decision | Target module | Description / reason | Required tests |
|---|---|---|---|---|---|
| `FactorAnalyzer.py` | `ResearchConfig`; `AnalysisConfig` | REFERENCE_ONLY | QE MetricSpec/config design | Legacy knobs identify horizons, bins, shifts, annualization, grouping; names and mixed concerns should not migrate. | config mapping test only |
| same | `FactorAnalyzer.__init__(base_df, trade_date_col, symbol_col, price_col, factor_name, research_cfg, analysis_cfg)`; `simple(...)` | DISCARD | `quant_evaluator.evaluate` | Stateful single-factor class mixes metric computation, multiprocessing, plotting, and mutable result panels. | golden output corpus only |
| same | `map_freq(td)` | REFERENCE_ONLY | calendar/annualization adapter | Frequency inference reference; calendar metadata should be explicit. | daily/intraday/irregular intervals |
| same | `map_symbol_to_industry(df, symbol_col, dummy_dict, industry_col="industry")` | REWRITE | context adapter | Useful mapping semantics but hard-coded default and Python mapping; context should already carry exposures. | unknown symbols; nulls; one-to-one mapping |
| same | `calc_stats_for_period(args)` | REWRITE | `metrics.quantile`, `turnover`, `portfolio_stats` | Contains quantile returns, set turnover, Sharpe/Sortino/Calmar and industry attribution, but has timing bugs, mutable globals, hard-coded columns and loops. | full golden corpus; timing; bins; risk metrics; industry slices |
| same | `calc_stats_for_horizon(args)` | REWRITE | `metrics.ic`, `decay`, `autocorrelation` | IC/RankIC/autocorrelation semantics useful; implementation uses Arrow temp files and hard-coded `code`. | horizons; shifts; tie/NaN; symbol name independence |
| same | `run_stats_parallel()`; `run()` | DISCARD | QE batch scheduler/evaluate | Joblib plus temporary Arrow memory map is legacy orchestration, not reusable batch architecture. | legacy corpus comparison only |
| same | `add_subtitle(...)`; five `create_*_sheet(...)` methods | DISCARD | optional external visualization | Plotly reporting is outside QE evidence core and spans ~1,600 lines. | none in QE |
| `Exposures.py` | `PortfolioExposures._cross_section_ols(sub_df, factor_cols)` | REFERENCE_ONLY | `quant_evaluator.metrics.exposure.reference` | Portable OLS/pseudoinverse math seed, but no finite/rank/conditioning checks. | statsmodels parity; collinearity; NaN; minimum observations |
| same | `PortfolioExposures.calc_stats/run` and inherited plot methods | REWRITE | `quant_evaluator.metrics.exposure` | Quantile portfolio exposure/attribution semantics useful; implementation inherits legacy base, has wrong future shift and hides failures with zeros. | exposure decomposition; return timing; missing/collinear matrices |
| same | `PureExposures.calc_stats/run` and inherited plot methods | REWRITE | `quant_evaluator.metrics.exposure` | Factor-weight exposure and correlation diagnostics are valuable but similarly coupled and zero-filling. | weight normalization; OLS attribution identity; missing contexts |
| `AlphaPurifier.py` | `AlphaPurifier` facade: `get_methods`, `winsorize`, `neutralize`, `standardize`, `to_result` | REWRITE | `factor_preprocess` public API | Useful chain/registry behavior, but wildcard imports, print-based discovery, pandas↔Polars conversion, and mixed fitted/stateless methods require a new typed API. Not QE runtime. | chain parity; input immutability; schema/order; fitted-window metadata |
| `factor_evaluation_alphapurify/pipeline.py` | `run_factor_case(...)`; `run_pipeline(config, *, config_path=None)`; `run_from_config(path)` | DISCARD | external orchestration | Scans factor directories, invokes Database/Analyzer/Exposures, monkeypatches Plotly, and writes JSON/PNG/config snapshots. | none in QE |

### AlphaPurify preprocessing families

All functions below operate on Polars frames and are candidates for **FactorPreprocess**, not QuantEvaluator. Grouping avoids repeating identical rationale while preserving every public symbol.

| File | Symbols and signature pattern | Decision | Target module | Description / reason | Required tests |
|---|---|---|---|---|---|
| `APr_utils.py` | `map_freq(td: timedelta) -> str | None` | REFERENCE_ONLY | context/calendar adapter | Frequency-name mapping only. | interval boundaries |
| same | `mean_std_winsorize`, `mad_winsorize`, `volatility_winsorize`, `iqr_winsorize`, `quantile_winsorize`: `(base_df, trade_date_col, factor_col, parameters) -> pl.DataFrame` | REUSE_AFTER_TEST | `factor_preprocess.reference.winsorize` | Cross-sectional clipping/compression kernels are mostly portable. | NaN/Inf; constants; small groups; hand-computed bounds |
| same | `rolling_quantile_winsorize(base_df, trade_date_col, symbol_col, factor_col, ...)` | REWRITE | `factor_preprocess.transforms.rolling` | Time-series fitted transform needs strict sorting, trailing-only window and fit metadata. | no lookahead; unsorted input; chunk parity |
| same | `boxcox_compress_winsorize`, `zscore_winsorize`, `rankgauss_winsorize`, `tanh_winsorize`, `huber_winsorize`, `ransac_winsorize` | REWRITE | `factor_preprocess.transforms.outlier` | Semantics worth retaining, but dependencies/domain shifts/model fitting and edge behavior need explicit contracts. | domain/positivity; deterministic RANSAC; ties; outliers; constants |
| same | `preprocess_for_neutralization(base_df, trade_date_col, factor_col, exposure_cols, ...)` | REWRITE | `factor_preprocess.neutralization.input` | Common finite/design-matrix preparation should be typed and reusable. | categorical exposures; missing rows; rank/conditioning |
| same | `multiOLS_neutralize`, `lasso_neutralize`, `ridge_neutralize`, `elasticnet_neutralize`, `polynomial_neutralize`, `kernelridge_neutralize`, `huber_neutralize`, `rank_neutralize`, `theilsen_neutralize`, `randomforest_neutralize`, `GBDT_neutralize`, `PCA_neutralize`, `ICA_neutralize`, `bayesianridge_neutralize`, `partialcorrelation_neutralize`: `(base_df, trade_date_col, factor_col, exposure_cols, ...) -> pl.DataFrame` | REWRITE | `factor_preprocess.neutralization` | Valuable residualization catalog, but per-date pandas/joblib, model-specific leakage risk, arbitrary residual definitions and no fitted-window identity preclude direct reuse. | sklearn/statsmodels oracle; same-date only; collinearity; deterministic seeds; missing exposures; residual orthogonality |
| same | `zscore_standardize`, `robust_zscore_standardize`, `minmax_standardize`, `rank_standardize`, `rank_gaussianize_standardize`, `normal_scores_standardize`, `quantile_binning_standardize`, `log_zscore_standardize`, `yeo_johnson_standardize`, `boxcox_standardize` | REUSE_AFTER_TEST | `factor_preprocess.reference.cross_sectional` | Mostly stateless same-date transform references; preserve only after domain and degeneracy tests. | ties; constants; NaN/Inf; domain shift; row order |
| same | `rolling_standardize`, `rolling_robust_standardize`, `rolling_minmax_standardize`, `volatility_scaling_standardize`, `EWMA_standardize` | REWRITE | `factor_preprocess.transforms.fitted` | Time-dependent transforms require explicit asset axis, sorting, trailing direction, fit window/end time, and streaming parity. | causality; fit metadata; chunks; minimum periods; state reset |

### Portable toolkit, intraday evaluator, registry/admission/agent/pool

| File/module | Symbol and signature | Decision | Target module | Description / reason | Required tests |
|---|---|---|---|---|---|
| `toolkit/cross_sectional.py` | `cs_rank(frame)`, `cs_zscore(frame)`, `cs_robust_zscore(frame)`, `cs_winsorize(frame, *, lower=.01, upper=.99)`, `cs_demean(frame)`, `cs_scale(frame)`, `cs_minmax(frame)`, `cs_rank_gauss(frame, *, clip=1e-4)`, `cs_quantile_bucket(frame, *, quantiles=10)`, `cs_signed_power(frame, *, exponent=.5)` | REUSE_AFTER_TEST | `factor_preprocess.reference.cross_sectional` | Cleanest pure-math panel transforms found; average ties and no-signal rows become NaN. Not QE metrics. | NaN/Inf; constants; ties; row/column order; dtype; SciPy rank-gauss parity |
| same | `apply_cross_sectional_transform(frame, transform="none")` | REWRITE | FactorPreprocess registry | Dispatch is coupled to `toolkit.registry`; replace with typed transform registry. | registry completeness; unavailable methods; no mutation |
| `toolkit/alpha_tools/registry.py` | `AlphaToolsFacade` and all `build/get/load/validate` functions | DISCARD | FactorOptimizer/FE metadata only | Loads generated function registries and hashes; no evidence math, duplicates FE/operator ownership. | none in QE |
| `factor_indicators_lysj/evaluation/preprocessing.py` | `rolling_zscore`, `make_forward_vwap_return`, `eligibility_mask_by_day`, `winsorize_by_day` | REFERENCE_ONLY | label/preprocess golden corpus | Intraday timing and per-session-tail semantics are useful oracles; implementation assumes one globally ordered series and date grouping. | session boundaries; multiple symbols; overnight gaps; timing |
| same | `load_factor_series`, `load_vwap_series`, `align_on_overlap_reindex`, `prepare_base/horizon/status` | DISCARD | adapters/caller | Loading/alignment/implicit label assembly are outside QE core. | adapter integration only |
| `factor_indicators_lysj/evaluation/predictivity.py` | `Predictivity.daily_corr`, `summary_from_daily_series`, `evaluate` | REFERENCE_ONLY | `metrics.ic.golden` | Clear pairwise-finite Pearson/Spearman oracle with average ties; Python group loop and intraday schema prevent runtime reuse. | standard IC suite; date ordering; constants |
| `factor_indicators_lysj/evaluation/backtest.py` | `_assign_quantile_groups`, `_capped_position_signal`, `layered_single_period_return`, `long_short_holding_pnl`, `apply_transaction_cost`, `holding_stats`, `evaluate` | REWRITE | `metrics.quantile`, `probe_portfolio`, `portfolio_stats` | Intraday rolling-quantile and holding/cost semantics provide a useful reference; time-series rolling rank differs from QE cross-sectional quantiles and fill-zero masks missingness. | session/multi-asset timing; costs; turnover; annualization; missing bars |
| `factor_indicators_lysj/evaluation/labels.py` | `MarketStatusLabel.volatility_regime`, `intraday_session`, `get_labels` | REFERENCE_ONLY | `quant_evaluator.slices.reference` | Regime/session slicing idea is useful, but rolling thresholds may embed full-series assumptions and labels are market-specific. | causality; session clock; boundary thresholds |
| `factor_indicators_lysj/evaluation/status_analysis.py` | `StatusAnalyzer.analyze` | REWRITE | `quant_evaluator.slicing` | Group/slice evaluation pattern maps to Metric × Slice × GroupBy, but delegates to coupled intraday classes. | slice/base consistency; small groups; no leakage |
| `factor_indicators_lysj/evaluation/pipeline.py` | `evaluate_horizon`, `run_evaluation` | DISCARD | `quant_evaluator.evaluate` reference corpus | Single-series orchestration with implicit preparation; not batch-first. | external golden corpus only |
| same | `serialize_results`, `print_brief` and `serialization.py` | DISCARD | caller/reporting | JSON conversion/downsampling/presentation is outside QE metric core. | none |
| `factor_admission/catalog.py` | `AdmissionCatalog` and CRUD methods | DISCARD | FactorAssets | SQLite ledger and approval status are explicitly outside QE; QE returns evidence references. | FactorAssets migration tests only |
| `factor_agent/` | `agent_loop`, evaluator/verifier/tools/hooks/skills and YAML scoring | DISCARD | FactorOptimizer/research control | LLM and filesystem orchestration; `_run_eval` executes scripts and logging writes files. No metric kernel. | none in QE |
| `factor_pool/` | README only | DISCARD | none | No code or formula corpus present. | none |
| `factor_indicators_lysj/factor_vwap_reversion.py` and archived AlphaPurify tests/examples | CORPUS_ONLY | QE/FactorPreprocess golden corpus | Real intraday factor and old usage patterns are suitable regression inputs, not runtime code. | corpus manifest; immutable expected outputs |

## Identified Legacy Bugs

| File | Line | Bug | Recommended fix in migration |
|---|---:|---|---|
| `factor_evaluation_alphapurify/Exposures.py` | 29-35 | `price_fut` uses positive `shift(rebalance_period)`, i.e. past price, while naming/result call it future return. | Do not port; define LabelBundle start/end and verify with a hand-computed timing table. |
| `factor_evaluation_alphapurify/FactorAnalyzer.py` | 403 | Assigns `pl.Datetime()` as an empty aggregation frame; this is a dtype, not a DataFrame constructor. | Rewrite aggregation with a typed empty result or skip aggregation explicitly. |
| same | 455-458 | Overnight `off`/`only` masks use `OR rebalance_mask`, which admits all rebalance rows and defeats the apparent filter semantics. | Specify truth table and test each overnight mode. |
| same | 530-532 | `-pl.col(...).rolling_mean(...).alias(...)` may alias before negation due expression binding, yielding fragile/incorrect names. | Parenthesize negation before alias and test output schema. |
| same | 564-578 | Set turnover divides intersection by current holdings only and concatenates long/short lists without dedup; not weight turnover and can misstate churn. | Use `0.5 * sum(abs(w_t-w_t-1))`; expose set-churn separately. |
| same | 892 | Industry mapping references hard-coded `pl.col("code")` instead of `self.symbol_col`. | Remove hard-coded names; context adapter supplies industry aligned by asset. |
| same | 420-426 vs `Exposures.py` 29-35 | Two legacy modules use opposite shift signs for nominal future returns. | Freeze one explicit label convention; retain both only as negative regression tests. |
| `factor_evaluation/pipeline.py` | 83-84 | Quantiles use `rank(method="first")`, so tied assets change bucket when row order changes. | QE default must use average ties and declare sparse-bin behavior. |
| same | 173 | `grouped_signal.shift(lag)` assumes complete ordered per-symbol rows; missing dates stretch holding periods by observations rather than calendar/session. | Use context calendar/label interval and test missing sessions. |
| same | 176-187 | Portfolio return renormalizes only over assets with non-null returns, silently changing intended weights. | Define missing-return policy; preserve denominator/exposure and emit diagnostics. |
| same | 258 | Inner universe merge silently drops rows with no coverage and does not report the denominator loss. | Consume explicit universe mask and publish coverage/missing-context diagnostics. |
| same | 340 | No `min_dates` gate before summary/risk metrics. | MetricSpec must declare minimum observations and return warning/NaN below threshold. |
| `batch_metrics.py` | 255-257 | Percentile thresholds with average ties can create unequal or empty tails. | Specify deterministic tie policy and report effective tail counts. |
| same | 267 | Missing realized returns are filled with zero, hiding missing labels and biasing gross return. | Pairwise validity plus explicit missing-label diagnostics; no silent zero imputation. |
| same | 288-292 | `annual_return` is arithmetic while some legacy modules use geometric; HAC Sharpe hard-codes 252 instead of the `annualization` argument. | Separate arithmetic/geometric metric names and use explicit annualization metadata. |
| same | 295-306 | Max drawdown is the minimum drawdown across offset sleeves, not drawdown of the represented strategy return path. | Construct the actual aggregate strategy path or name the sleeve statistic explicitly. |
| same | 311-324 | Admission thresholds and tier routing are hard-coded inside metrics module. | Move policy/versioning to FactorAssets; QE emits evidence only. |
| `timeseries/scripts/portfolio.py` | 11 | Imports `vectorbt` but does not use it, adding a heavy dependency. | Remove from migrated pure kernel. |
| same | 93, 97 | Missing turnover and net returns are filled with zero in cost/path calculation. | Preserve missingness and emit path-validity diagnostics. |
| `evaluation/pipeline.py` | 146 | Sets quantile `net_return = gross_return`, silently asserting zero costs. | Never synthesize net values; require cost model or mark unavailable. |
| same | 253 | Defaults `primary_horizon` to first configured horizon without semantic validation. | Require explicit primary horizon in policy/caller. |
| `indicator/.../calculator.py` | 307-318 | Annualized return assumes equally spaced observations. | Use frequency/context metadata or refuse annualization for irregular samples. |
| same | 359-365 | Spearman helper drops indexes before concatenation, allowing misaligned series to be correlated by position. | Align by key before finite pair filtering. |
| same | 751-753 | `survival_view` routing embeds arbitrary Sharpe/drawdown thresholds in calculator output. | Remove from QE; FactorAssets owns versioned decisions. |
| `factor_indicators_lysj/evaluation/preprocessing.py` | 95-97, 195 | Forward VWAP shifts are global; with multiple assets or concatenated sessions they cross symbol/day boundaries. | Group by asset and enforce session/LabelBundle boundaries. |
| same | 100-107 | Eligibility is grouped only by `date`, not asset, so multi-asset panels get incorrect tail masks. | Group by `(asset, session_date)` or consume validity mask. |
| `factor_indicators_lysj/evaluation/backtest.py` | 134-143 | Global shifts for triggers and VWAP can cross asset/session boundaries; fill-zero turns unavailable returns into flat PnL. | Block/asset-aware implementation and explicit missingness. |
| `factor_evaluation_alphapurify/AlphaPurifier.py` | 89-93 | Pandas input is not datetime-cast/sorted, while non-pandas input is; behavior depends on adapter type. | Normalize through one typed adapter and test representation parity. |
| same | 180-181 | `standardize("zscore", *args)` silently ignores supplied arguments. | Typed parameter validation; reject unexpected parameters. |
| `factor_evaluation_alphapurify/Database.py` | 13 | Hard-coded absolute Windows data root exists in production module. | Discard module; DataAccess owns storage and paths. |

No production-ready block bootstrap, purged K-fold/CPCV, DSR/PBO/SPA/Reality Check, or comprehensive null/Inf/zero/tie/unique diagnostic suite was found. `purged_split_mask` is a simple train/valid/test boundary purge, not purged K-fold or CPCV. No implementation named `hazing` was found.

## Data Coupling Findings

| Legacy code | Coupling | Migration action |
|---|---|---|
| `factor_layer/factor_evaluation/io.py` | Reads CSV/parquet factor, market, and universe files directly. | REMOVE from QE; caller/DA adapter supplies FactorBatch, LabelBundle, EvaluationContext. |
| `factor_layer/factor_evaluation/pipeline.py` | Calls the loaders, merges universe, and writes CSV/parquet reports. | Split pure reference kernels from I/O; discard persistence path. |
| `factor_evaluation_alphapurify/Database.py` | Reads/writes symbol parquet, uses DuckDB/joblib, scans directories. | DISCARD; duplicates DataAccess storage/query responsibilities. |
| `factor_evaluation_alphapurify/pipeline.py` | Detects years from parquet paths, invokes Database, writes PNG/JSON/config snapshots. | DISCARD orchestration. |
| `AutoFactorEvaluation-RECONSTRUCT/evaluation/timeseries/scripts/io.py`, `forward_returns.py`, `emit.py`, `service.py` | Reads factor/market/universe parquet/CSV, caches forward returns, writes artifacts/manifests. | REMOVE all I/O/cache from QE core; retain math only. |
| `AutoFactorEvaluation-RECONSTRUCT/evaluation/pipeline.py` and indicator calculator | Reads/writes parquet/JSON/log files and routes admission labels. | DISCARD pipeline; translate only formula kernels to MetricSpecs. |
| `factor_indicators_lysj/factor_evaluation_framework.py`, `factor_vwap_reversion.py` | Reads local parquet from hard-coded module-relative paths and writes output/report. | CORPUS_ONLY or discard runtime. |
| `factor_admission/catalog.py` | Writes SQLite evaluation/admission records. | FactorAssets ownership; QE returns evidence, never writes DB. |
| `factor_agent/` | Reads/writes YAML, runs evaluation scripts, writes audit logs; agent orchestration. | Outside QE; no runtime imports. |
| FE pipeline/cross-sectional copies | Factor computation/operator semantics. | REMOVE from QE; QE must not import or compute factors. |

No relevant canonical evaluator directly imports `dataaccess`; instead it bypasses DA by reading parquet/CSV directly. That is still coupling to remove. FE copies and agent YAML tooling compute/define factors and must not be pulled into QE.

## Pure Math (Portable) vs Coupled (Needs Decoupling)

### Portable after golden tests

- `batch_metrics._row_corr`, `daily_ic`, `_bh_qvalues`, and `purged_split_mask`.
- `timeseries/scripts/ic._group_corr` and `_safe_kendall_tau`.
- `timeseries/scripts/portfolio.compute_turnover_series` after removing metadata columns.
- `factor_evaluation._summary_from_series` and `_compute_period_group_returns` as slow references.
- `toolkit/cross_sectional.py` math kernels, targeted to FactorPreprocess.
- `Exposures._cross_section_ols` only as an OLS golden reference after finite/rank checks.

### Coupled and requiring rewrite

- `FactorAnalyzer`: mutable class state, hard-coded columns, temporary Arrow memory maps, joblib workers, plotting, aggregation and report concerns.
- `PortfolioExposures`/`PureExposures`: inheritance from history module, Plotly, return construction, quantile selection, regression, attribution and presentation in one object.
- `AlphaPurifier`: wildcard registry dispatch, conversions, fitted and stateless transforms in one mutable facade.
- All `pipeline.py`, loaders, emitters, serializers, catalog, and agent code.
- `indicator/calculator.compute_metrics`: useful formulas but one monolithic file-oriented metric matrix builder.
- Intraday LYSJ evaluator: single globally ordered series and session assumptions must become explicit asset/layout/context contracts.

## Suggested Reference Seeds for QE Modules

| QE target | Primary seed | Secondary/golden seed | Migration note |
|---|---|---|---|
| `metrics.ic.reference` | `batch_metrics.daily_ic` | `factor_evaluation._compute_daily_ic`, `Predictivity.daily_corr` | Freeze average ties, pairwise finite drop, constant→NaN, min-assets. |
| `metrics.ic.fast` | `timeseries/scripts/ic._group_corr` | `_row_corr` | Improve numerical stability and prove reference/fast parity. |
| `metrics.ic_summary` | `batch_metrics.summary_stats` | `factor_evaluation._summary_from_series` | Separate raw ICIR from explicitly annualized forms. |
| `metrics.quantile` | `timeseries/scripts/quantile.compute_quantile_panels` | `_compute_period_group_returns` | Rewrite around block arrays and declared tie/sparse-bin policy. |
| `metrics.turnover` | `timeseries/scripts/portfolio.compute_turnover_series` | `batch_metrics.long_short_series` | Canonical weight turnover; keep set churn as separately named diagnostic only. |
| `metrics.probe_portfolio` | `batch_metrics.long_short_series` | factor-layer layered backtests | Preserve missingness; separate weights, returns, costs, and path. |
| `metrics.portfolio_stats` | `factor_evaluation._summarize_long_short` | batch `performance_stats`, indicator calculator | Explicit frequency and arithmetic/geometric metric names. |
| `metrics.robustness.hac` | `batch_metrics._hac_variance` | statsmodels | Reference first; Newey-West/HAC parity before acceleration. |
| `metrics.multiple_testing` | `batch_metrics._bh_qvalues` | statsmodels | Return q-values only; no admission mutation. |
| `metrics.exposure` | `Exposures._cross_section_ols` | statsmodels OLS | Context supplies industry/size matrices; condition/rank diagnostics mandatory. |
| `labels.reference` | `batch_metrics.price_forward_returns` | factor-layer and LYSJ timing functions | Labels remain caller-built and carry start/end/execution convention. |
| `validation.purging` | `batch_metrics.purged_split_mask` | hand-built calendars | Extend only after typed label intervals; it is not CPCV. |
| FactorPreprocess cross-sectional | `toolkit/cross_sectional.py` | `APr_utils.py` stateless transforms | Separate package; do not place preprocessing in QE. |
| QE real corpus | archived AlphaPurify examples/tests, LYSJ VWAP factor, factor-layer runtime smoke test | reconstructed batch contract tests | Use as cross-check corpus, never absolute truth where legacy implementations disagree. |

Recommended P0 extraction order: (1) IC/RankIC reference and fast parity, (2) coverage/validity diagnostics, which are mostly missing and need fresh implementation, (3) quantile/top-bottom, (4) weight turnover and cost-free probe, (5) exposure OLS reference. P1 starts only after LabelBundle timing and annualization contracts are frozen. Block bootstrap and CPCV require new implementations because no suitable production legacy kernel was found.
