# Consolidated Migration Matrix — Wave 0

**Status:** DRAFT (no authorization for code changes or package initialization)

**Date:** 2026-08-13

**Scope:** Consolidated function-level migration matrix across all four proposed platform packages (QuantEvaluator, FactorOptimizer, FactorAssets, FactorPreprocess), drawn from the individual legacy audits. This document is the only file written. It does not modify production code, package structure, or any published contract.

**Source audits:**
- `WAVE_0_OUTPUT/QE_LEGACY_MIGRATION.md`
- `WAVE_0_OUTPUT/FO_LEGACY_MIGRATION.md`
- `WAVE_0_OUTPUT/FA_LEGACY_MIGRATION.md`
- `WAVE_0_OUTPUT/FP_LEGACY_MIGRATION.md`
- `WAVE_0_OUTPUT/PACKAGE_BOUNDARY_DRAFT.md`

**Taxonomy:**
- `REUSE_AFTER_TEST`: Portable formula/logic; golden tests required before runtime adoption.
- `REWRITE`: Semantically sound but coupled/incomplete; reimplement with typed contracts.
- `REFERENCE_ONLY`: Useful specification or oracle; not runtime code; used for validation/parity only.
- `CORPUS_ONLY`: Real data or historical examples; preserved as regression/test fixtures, never runtime dependencies.
- `DISCARD`: Duplicate, stub, unsafe, or out-of-scope; explicitly not migrated.

---

## 1. QuantEvaluator (QE): Evidence Engine

QE is a pure batch evaluator returning typed evidence bundles. It consumes `FactorBatch`, explicit `LabelBundle` and `EvaluationContext` from caller/DA. It never infers labels, assembles universe, calculates factors or chooses admission.

### 1.1 Core IC and correlation metrics

| Legacy source | Symbol | Decision | Target QE module | Reason | Required tests |
|---|---|---|---|---|---|
| `batch_metrics.py` | `_row_corr(left, right)` | REUSE_AFTER_TEST | `metrics.ic.reference` | Vectorized pairwise finite row correlation; correct NaN/constant behavior | NaN/Inf, constants, batch/chunk parity, float32/64 |
| same | `daily_ic(signal, returns, min_assets)` | REUSE_AFTER_TEST | `metrics.ic.reference` | Best compact Pearson/RankIC seed; batch-first panel | ties, constants, min-assets, sign, monotonic transform, order |
| `timeseries/ic.py` | `_safe_kendall_tau(x, y, eps)` | REUSE_AFTER_TEST | `metrics.ic.reference` | SciPy Kendall tau-b, correctly rejects constants | SciPy parity, ties, NaN pre-mask, constants |
| `timeseries/ic.py` | `_group_corr(df, x_col, y_col, min_assets, eps)` | REUSE_AFTER_TEST | `metrics.ic.fast` | O(N) grouped sufficient-statistics correlation | numerical stability, constants, chunks, shuffled order |
| `timeseries/scripts/ic.py` | `compute_daily_ic_series(aligned_panel, config, factor_id, eval_run_id)` | REWRITE | `metrics.ic` | Good Pearson/RankIC/Kendall semantics; hard-coded schema, warning buffer, Python Kendall loop | all RankIC contract tests, multi-horizon, warning contract |
| `factor_evaluation/pipeline.py` | `_compute_daily_ic(frame, ret_col, min_assets)` | REFERENCE_ONLY | `metrics.ic.golden` | Slow per-date pandas oracle with correct pairwise drop and average RankIC ties | external parity and metamorphic suite |
| `FactorAnalyzer.py` | `calc_stats_for_horizon(args)` | REWRITE | `metrics.ic`, `decay`, `autocorrelation` | IC/RankIC/autocorrelation semantics useful; Arrow temp files and hard-coded loops | full golden corpus, timing, bins, ties/NaN |

### 1.2 Summary statistics and intervals

| Legacy source | Symbol | Decision | Target QE module | Reason | Required tests |
|---|---|---|---|---|---|
| `batch_metrics.py` | `summary_stats(values, hac_lags=0)` | REWRITE | `metrics.ic_summary` | Mean/std/IR/sign/t/HAC bundle useful; normal approximation and nullable dict schema need explicit MetricSpecs | ddof, non-annualized ICIR, HAC oracle, NaNs, small n |
| `factor_evaluation/pipeline.py` | `_summary_from_series(series)` | REUSE_AFTER_TEST | `metrics.ic_summary.reference` | Clean non-annualized mean/std/IR/win-rate reference | NaN, one value, constant, ddof |
| `batch_metrics.py` | `_hac_variance(values, lags)` | REFERENCE_ONLY | `metrics.robustness.hac` | Bartlett/Newey-West long-run variance semantics; use statsmodels oracle first | statsmodels parity, lags 0/n, autocorrelated/constant series |

### 1.3 Quantile, top-bottom and portfolio metrics

| Legacy source | Symbol | Decision | Target QE module | Reason | Required tests |
|---|---|---|---|---|---|
| `timeseries/quantile.py` | `_assign_quantile_ids(sub, config)` | REWRITE | `metrics.quantile` | Contains sector-neutral/ranking policy but DataFrame/config coupled | average ties, sparse bins, industry mode, monotonicity |
| `timeseries/quantile.py` | `compute_quantile_panels(...)` | REWRITE | `metrics.quantile` | Produces quantile returns, top/bottom, coverage and weight rows; monolithic and schema-coupled | Q1..Q10, coverage denominator, min assets, chunk parity |
| `factor_evaluation/pipeline.py` | `_compute_period_group_returns(frame, ret_col)` | REUSE_AFTER_TEST | `metrics.quantile.reference` | Simple equal-weight quantile-return oracle | missing bins, NaNs, order |
| `batch_metrics.py` | `long_short_series(signal, returns, min_assets, n_quantiles, cost_bps)` | REWRITE | `metrics.probe_portfolio` | Weight-turnover-cost semantics useful; ties create unequal tails, missing returns filled as zero | ties, disappearing assets, missing returns, cost arithmetic, neutrality |
| `factor_evaluation/pipeline.py` | `_build_quantile_backtest(frame, horizon, n_quantiles)` | REWRITE | `metrics.probe_portfolio` | Layered holding idea useful; Python loops and normalization distort exposure | overlapping holdings, delist/missing return, weight conservation |
| `batch_metrics.py` | `performance_stats(returns, annualization, hac_lags=0, holding_period=1)` | REWRITE | `metrics.portfolio_stats` | Sharpe/HAC Sharpe/max drawdown reference; sleeve-min drawdown is not strategy path | arithmetic/geometric annualization, overlapping sleeves, known drawdown |
| `factor_evaluation/pipeline.py` | `_summarize_long_short(series, annualization_factor)` | REFERENCE_ONLY | `metrics.portfolio_stats.golden` | Standard geometric annual return, volatility, Sharpe, max drawdown oracle | known path, irregular dates, return <= -1 |

### 1.4 Turnover, costs and probe

| Legacy source | Symbol | Decision | Target QE module | Reason | Required tests |
|---|---|---|---|---|---|
| `timeseries/portfolio.py` | `compute_turnover_series(weight_rows, horizons, factor_id, eval_run_id)` | REUSE_AFTER_TEST | `metrics.turnover.reference` | Canonical `0.5 * sum(abs(delta weights))`; first observation intentionally NaN | entries/exits, asset disappearance, first date, multiple horizons |
| `timeseries/portfolio.py` | `attach_cost_and_path(ls_df, turnover_df, config)` | REWRITE | `metrics.probe_portfolio` | Cost and drawdown path useful; unused `vectorbt`, fills missing with zero, loops horizons | missingness, cost timing, path oracle, multi-horizon |

### 1.5 Robustness and multiple testing

| Legacy source | Symbol | Decision | Target QE module | Reason | Required tests |
|---|---|---|---|---|---|
| `batch_metrics.py` | `_bh_qvalues(p_values)` | REUSE_AFTER_TEST | `metrics.multiple_testing` | Standard Benjamini-Hochberg kernel; portable | statsmodels parity, ties, unsorted input, 0/1, empty |
| `batch_metrics.py` | `apply_validation_fdr(records, alpha)` | REWRITE | FactorAssets policy + QE q-values | Mutates records and downgrades admission; separate evidence from decision | q-value attachment, no mutation, policy boundary |

### 1.6 Labels and forward returns

| Legacy source | Symbol | Decision | Target QE module | Reason | Required tests |
|---|---|---|---|---|---|
| `batch_metrics.py` | `price_forward_returns(frame, horizons, price_field, entry_lag)` | REFERENCE_ONLY | `labels.reference` | Explicit entry/exit convention valuable; QE consumes `LabelBundle`, never infers silently | exact shift direction, irregular calendar, zero price, horizons |
| `timeseries/forward_returns.py` | `compute_forward_returns(market_data, horizons, price_col)` | REFERENCE_ONLY | `labels.reference` | Explicit forward-label oracle; QE runtime does not fetch/cache or choose price fields | timing table, symbol isolation, zero/Inf, horizons |
| `factor_indicators_lysj/evaluation/labels.py` | `MarketStatusLabel` | CORPUS_ONLY | none | Market-state labels from realized volatility; labels are not transforms | label causality and boundary tests |

### 1.7 Data validation and splits

| Legacy source | Symbol | Decision | Target QE module | Reason | Required tests |
|---|---|---|---|---|---|
| `batch_metrics.py` | `split_boundaries(frame, config)` | REWRITE | `validation.splits` | Date split semantics useful; config untyped, fixed fractional split too narrow | boundary edge cases, min dates, deterministic order |
| `batch_metrics.py` | `purged_split_mask(index, split, bounds, trading_dates, entry_lag, horizon)` | REUSE_AFTER_TEST | `validation.purging` | Compact leakage-safe purge reference; needs typed label timing and calendar contract | exit exactly on boundary, missing dates, all four splits |
| `batch_metrics.py` | `apply_point_in_time_universe(market_frame, universe_frame, universe_id)` | DISCARD | DA/context adapter | PIT universe assembly is DataAccess ownership; QE consumes supplied mask |

### 1.8 Coverage and diagnostics

| Legacy source | Symbol | Decision | Target QE module | Reason | Required tests |
|---|---|---|---|---|---|
| `batch_metrics.py` | `normalize_market_frame(frame)` | REWRITE | `adapters.pandas` | Key/finite validation useful; hard-codes OHLCV, performs market-data normalization QE should not own | duplicate/null keys, ordering, adapter contract |
| `batch_metrics.py` | `purify_series(series, mad_multiplier, min_assets)` | REUSE_AFTER_TEST | `factor_preprocess.reference.cross_sectional` | Clear MAD clipping + z-score reference; belongs to FactorPreprocess, not QE | NaN/Inf, zero MAD, min assets, order and dtype |

### 1.9 Exposure and attribution

| Legacy source | Symbol | Decision | Target QE module | Reason | Required tests |
|---|---|---|---|---|---|
| `Exposures.py` | `_cross_section_ols(sub_df, factor_cols)` | REFERENCE_ONLY | `metrics.exposure.reference` | Portable OLS/pseudoinverse math seed; no finite/rank/conditioning checks | statsmodels parity, collinearity, NaN, minimum observations |
| `timeseries/quantile.py` | `_sector_rank(values, industries)` | REFERENCE_ONLY | `metrics.exposure.reference` | Industry-relative ranking oracle | missing industry, ties, singleton groups |
| `Exposures.py` | `PortfolioExposures.calc_stats/run` and inherited plot methods | REWRITE | `metrics.exposure` | Quantile portfolio exposure/attribution semantics useful; wrong future shift, hides failures with zeros | exposure decomposition, return timing, missing/collinear matrices |

### 1.10 Unresolved QE items

- **LabelBundle contract:** Caller/modeling must supply explicit start/end/execution timing, horizon and return convention. QE validates, never guesses shifts.
- **EvaluationContext scope:** Caller/DA supplies universe mask, industry/size/liquidity/beta/volatility/tradability/status context. QE consumes only what is needed for evidence.
- **Diagnosis and warning schema:** Define `FactorDiagnosis` and `EvaluationBundle` with metric versions, warnings, slices and optional series refs. No admission decisions.
- **Annualization contract:** Separate raw ICIR from annualized forms. Freeze whether IAR is 252-day annualized or left raw.
- **Multi-horizon coordination:** Define horizon handling across IC/quantile/portfolio and whether primary horizon is selected or all returned.

---

## 2. FactorOptimizer (FO): Search and Feedback Loop

FO is optional and evidence-guided. It generates mutation proposals, orchestrates FE compilation, QE evaluation and research ledger entries. It does not own operators, metrics or any FE/QE implementation.

### 2.1 Mutation specs and grammar

| Legacy source | Symbol | Decision | Target FO module | Reason | Required tests |
|---|---|---|---|---|---|
| `toolkit/registry.py` | `TransformSpec` | REUSE_AFTER_TEST | `mutations.spec` | Good frozen metadata and exposure/leakage fields; extend to full MutationSpec | immutability, schema round-trip, required fields, no kernel ownership |
| `toolkit/alpha_tools/registry.py` | `SEED_TOOL_SPECS` | REUSE_AFTER_TEST | `mutations.spec` | Input/return contracts, typed parameters, bounds, defaults and version useful | bound edges, bool-vs-int, defaults, relational bounds |
| `toolkit/alpha_tools/registry.py` | `validate_tool_parameter_value(tool_name, parameter_name, value, bound_parameters)` | REWRITE | `mutations.validation` | Only int/float/min/max/max_ref and repeated global lookup; all kinds, enums, NaN, relational bounds, defaults | all kinds, enums, NaN, relational bounds, defaults |

### 2.2 Complexity profile

| Legacy source | Symbol | Decision | Target FO module | Reason | Required tests |
|---|---|---|---|---|---|
| `gateway/scripts/complexity.py` | `ComplexityReport` | REUSE_AFTER_TEST | `complexity.models` | Sound report concept; add AST depth, lookback, stateful/CS/nonlinear/interactions/domains/sources/latency | serialization, totals, budget boundary, FE parity |
| `gateway/scripts/complexity.py` | `ComplexityEvaluator.evaluate(expr)` | REWRITE | `complexity.evaluator` | Regex calls and raw parentheses; AST parser branch is `pass` | nested/malformed AST, aliases, string literals, FE parity |
| `gateway/scripts/complexity.py` | `suggest_optimization(expr)` | REFERENCE_ONLY | `policy.repair` | Generic heuristics, not diagnosis/evidence-driven | diagnosis mapping, unsupported mutation rejection |

### 2.3 Candidate and deduplication

| Legacy source | Symbol | Decision | Target FO module | Reason | Required tests |
|---|---|---|---|---|---|
| `gateway/scripts/models.py` | `Candidate` | REFERENCE_ONLY | `candidates.models` | Preserve IDs, timestamps, campaign/batch and stage evidence ideas; missing lineage and trial fields | lineage round-trip, parent requirement, immutable origin, versions |
| `gateway/scripts/deduplicator.py` | `DuplicateRecord`, `DeduplicationResult` | REWRITE | `seen.models` | Preserve first-seen/historical ref idea; add lifecycle scope, hash kind, neighbors and decision hints | lifecycle retention, serialization |
| `gateway/scripts/deduplicator.py` | `compute_candidate_hash(expr, config)` | REWRITE | `identity.service` | Arbitrary JSON config includes incidental fields and lacks parameter/sign canonicalization | key order, numeric normalization, relevant-field tests |
| `gateway/scripts/deduplicator.py` | `HashDeduplicator` persistence/cache | REWRITE | `seen.index` | 10k LRU evicts global history and corrupt JSON silently ignored | restart, >10k history, corrupt store fail-closed, concurrency |
| `factor_engine/mining/campaign.py` | `candidate_semantic_hash(Factor|Expr)` | REUSE_AFTER_TEST via adapter | `adapters.factor_engine` | Best AST/parameter-normalized structural identity; alias canonicalization and sorted mappings | aliases, kwarg order, arg order, nonfinite, FE versions |

### 2.4 Search and orchestration

| Legacy source | Symbol | Decision | Target FO module | Reason | Required tests |
|---|---|---|---|---|---|
| `gateway/scripts/gateway_core.py` | `run_gateway(manifest, campaign_config, init_result, dedup_cache, market_data_root, cache_dir)` | REFERENCE_ONLY | `search.runner` | Ordered cheap-to-expensive gates and fail-fast records useful; implementation couples IO, FE, LLM and DQ | gate order, halt, budget, complete trail |
| `gateway/scripts/gateway_core.py` | `_step3_dedup`, `_step6_complexity`, `_calc_nesting_depth` | REWRITE | `seen`, `complexity` | Replace strings/brackets with FE canonical AST/IR | FE identity/depth parity |
| `gateway/scripts/gateway_core.py` | `_step4_tiny_run(manifest)` | REFERENCE_ONLY | `adapters.factor_engine` | L0/L1 idea useful; execution belongs behind protocol | fake protocol, tiny budget, errors |
| `factor_engine/mining/campaign.py` | `MiningCampaignSnapshot`, `MiningCampaignSession` | REFERENCE_ONLY | `search.session` | Snapshot pinning/shared work useful; use protocols, not FE internals | pinned IDs, batch, cancellation/budget, cache |

### 2.5 FE and QE adapters

| Legacy source | Symbol | Decision | Target FO module | Reason | Required tests |
|---|---|---|---|---|---|
| `factor_engine/mining/operator_catalog.py` | `MiningOperator`, `AdmissionQuery`, `get_mining_operators` | REUSE_AFTER_TEST via adapter | `adapters.factor_engine` | Authoritative availability, roles, costs, types and sources | consistency audit, domain/source rejection, version binding |
| `factor_engine/mining/direct_use.py` | `DirectUseContract`, `InputSlotSpec` | REUSE_AFTER_TEST via adapter | `adapters.factor_engine` | Already-migrated eligibility and parameter-role semantics | parameter split, causal/timing gates, no mirror |
| `factor_engine/mining/campaign.py` | `NegativeCompileCache`, `DependencySignature`, `group_source_first` | REUSE_AFTER_TEST | `search.cache`, `search.batch` | Thread-safe cache keyed by candidate + compiler generation, batch-first grouping | races, invalidation, deterministic errors, stable grouping |

### 2.6 LLM and research control

| Legacy source | Symbol | Decision | Target FO module | Reason | Required tests |
|---|---|---|---|---|---|
| `factor_agent/core/agent_loop.py` | `agent_loop(query, create_message, system, max_rounds, max_tokens, model)` | REFERENCE_ONLY | `llm.researcher` | Structured tools, max rounds, verifier stop and audit hooks; reimplement around CandidateMutation | budgets, invalid tool, replay, structured output, no Python code |
| `factor_agent/tools/registry.py` | `register`, `run`, `get_api_tools` | REFERENCE_ONLY | `llm.tools` | Structured tool schema concept; mutable globals, file tools and string errors are unsafe | allowlist, typed errors, malformed input, no arbitrary code/files |
| `factor_agent/skills/registry.py` | `format_layer1_for_system`, `get_skill_content` | REFERENCE_ONLY | `llm.prompts` | Context layering useful; hard-coded text/imports need versioning | prompt hash, unknown skill, deterministic assembly |

### 2.7 Admission and policy

| Legacy source | Symbol | Decision | Target FO module | Reason | Required tests |
|---|---|---|---|---|---|
| `factor_admission/admission.py` | `_check_thresholds(summary_row, thresholds)` | REWRITE | `admission.policy.ConfiguredGate` | Diagnostics useful; fixed thresholds must be named/versioned policy config and only one gate | missing evidence fail-closed, boundary values, policy replay |
| `factor_admission/admission.py` | `admit_evaluation_run(config)` | REWRITE | `admission.service` | Must orchestrate safety gates, EvidenceRef, conditional novelty, Pareto/budget context, decision and lifecycle transaction | novelty adapter, Pareto cases, rollback, replay idempotency |

### 2.8 Unresolved FO items

- **MutationSpec finalization:** Define exact versioning, parameter role/kind semantics (causal, temporal, scaling), domain/source/type constraints.
- **Hypothesis and trial identity:** Define the idempotency key, parent/child lineage and mutation config serialization.
- **Search budget enforcement:** Define L0-L4 tiers, candidate/evaluation/compute/LLM budgets and stopping criteria (plateau, budget, iterations).
- **Evidence to repair logic:** Diagnosis vectors -> candidate generator contract; must avoid over-fitting repair to one metric.
- **Pareto and conditional novelty:** Define hard gates, Pareto front projection and conditional admission (e.g., admissible only if similar to core + better evidence).

---

## 3. FactorAssets (FA): Identity and Governance

FA is the identity, registry, lineage and lifecycle engine. **This consolidation awaits the FA migration report.** Placeholder structure based on PACKAGE_BOUNDARY_DRAFT guidance.

### 3.1 Registry and identity

| Legacy source | Symbol | Decision | Target FA module | Reason | Required tests |
|---|---|---|---|---|---|
| `factor_admission/catalog.py` | `AdmissionCatalog.__init__` | REUSE_AFTER_TEST | `storage.sqlite` | SQLite WAL suitable for first backend; migrations transactional | WAL concurrency, migration rollback, FK enforcement |
| `factor_admission/catalog.py` | `_BASE_SCHEMA_SQL`, `_ADMISSION_SCHEMA_SQL` | REFERENCE_ONLY | `storage.sqlite` migrations | Identity/run/status/decision separation useful; lacks versioned migrations, indexes, lifecycle, lineage, fingerprints and EvidenceRef | migration upgrade, FK integrity, index plan, no-delete invariant |
| `assetization/scripts/registry.py` | `generate_factor_id(coordinates, seed)` | REWRITE | `identity.service` | Stable seed idea useful; eight-hex suffix and coordinates insufficient | collision/property, concurrent registration |
| `assetization/scripts/registry.py` | `extract_coordinates(config)` | REUSE_AFTER_TEST | `origin.import_coordinates` | Four coordinates useful import metadata | missing field, enum validation, round-trip |

### 3.2 Asset lifecycle and decisions

| Legacy source | Symbol | Decision | Target FA module | Reason | Required tests |
|---|---|---|---|---|---|
| `factor_admission/catalog.py` | `update_factor_status(*, factor_id, latest_run_id, approved)` | REWRITE | `lifecycle.StateMachine.transition` | Blind overwrite collapses lifecycle to two states and writes no event | legal/illegal transition property tests, event/current consistency |
| `factor_admission/catalog.py` | `record_decision(*, factor_id, run_id, decision, ...)` | REWRITE | `decisions.DecisionLedger` | Append-only decision and policy snapshot valuable, but need typed evidence/context and transaction with state events | append-only, idempotency, rollback, audit ordering |
| `gateway/scripts/models.py` | `GatewayLabel` | REFERENCE_ONLY | `lifecycle.import_mapping` | Useful outcomes; not an asset lifecycle | mapping, duplicate-to-shadow/seen behavior |
| `gateway/scripts/models.py` | `GatewaySegment` | REWRITE | `decisions.GateEvent` | Preserve stage/run/timestamp/diagnostics; scores become evidence refs | JSON round-trip, immutable event |

### 3.3 Deduplication and seen history

| Legacy source | Symbol | Decision | Target FA module | Reason | Required tests |
|---|---|---|---|---|---|
| `gateway/scripts/deduplicator.py` | `DuplicateRecord`, `DeduplicationResult` | REWRITE | `seen.models` | Preserve first-seen/historical ref idea; add lifecycle scope, hash kind, neighbors and decision hints | lifecycle retention, serialization |
| `gateway/scripts/deduplicator.py` | `ExpressionNormalizer.normalize(expr)` | REFERENCE_ONLY | `seen.FEIdentityAdapter` | Whitespace/lowercase not canonical; FE must provide canonical AST hash | adversarial expressions, FE hash parity |
| `gateway/scripts/deduplicator.py` | `compute_expr_hash(expr)` | REFERENCE_ONLY | `seen.GlobalSeenIndex` | Exact hash/history concept sound; production consumes FE canonical representation | repeatability, collision handling |

### 3.4 Candidate models and manifests

| Legacy source | Symbol | Decision | Target FA module | Reason | Required tests |
|---|---|---|---|---|---|
| `gateway/scripts/models.py` | `Candidate` | REWRITE | `models.FactorCandidate` | Preserve candidate ID, campaign, batch, origin and timestamps; add FE definition ref and typed parents | disk.v1 import, provenance, FE identity parity |
| `assetization/scripts/models.py` | `FactorCandidate` | REWRITE | `models.FactorCandidate` | Minimal metadata reference; needs definition refs, origin, campaign and parents | legacy import, immutable identity |
| `assetization/scripts/models.py` | `FactorAsset` | REWRITE | `models.FactorAsset` | Concept useful; incomplete model embeds materialization path/state | full round-trip, lifecycle enum, no raw path |
| `gateway/scripts/models.py` | `GatewayResult` | REFERENCE_ONLY | `admission.GateResult` | Typed result pattern useful; statuses/evidence need redesign | serialization, exhaustive statuses |
| `gateway/scripts/models.py` | `Manifest` | REFERENCE_ONLY | `importers.disk_v1` | Import contract only; not new repository model | all 222 manifests, version rejection |

### 3.5 Catalog and queries

| Legacy source | Symbol | Decision | Target FA module | Reason | Required tests |
|---|---|---|---|---|---|
| `factor_admission/catalog.py` | `ensure_factor_registered(factor_id)` | REWRITE | `repository.require_asset` | Coupled to factor-lake registration; FA owns asset identity and references FE through public contracts | missing/existing asset, package boundary |
| `factor_admission/catalog.py` | `upsert_evaluation_run(payload)` | REWRITE | `evidence.attach_evidence_ref` | Stores filesystem paths, overwrites metadata; FA needs immutable QE bundle refs | idempotency, immutable bundle identity, stale ref rejection |
| `factor_admission/catalog.py` | `get_factor_status`, `get_evaluation_run` | REUSE_AFTER_TEST | `repository` | Parameterized read pattern reusable after typed models/repository abstraction | found/not-found, type round-trip |
| `factor_admission/catalog.py` | `list_factor_library()` | REWRITE | `queries.AssetQueryService` | Useful read-model intent, but joins QE metrics and factor-lake watermarks | pagination, state/family filters, no metric truth duplication |

### 3.6 Unresolved FA items

- **FA migration report pending.** This consolidation awaits full audit of legacy catalog/registry/lifecycle patterns.
- **Lifecycle state machine specification:** Define legal transitions, state semantics and certification/production-ready representation.
- **Identity reference format:** Freeze whether FA uses FE canonical AST digest, plan identity or a cross-package envelope.
- **Evidence reference schema:** Define `EvidenceRef` fields (metric version, run ID, factor IDs, time, universe), freshness, selected summary retention.
- **Lineage and parent identity:** Define how FA links to parent candidates, mutations, search trials and campaign contexts.
- **Seen history and dedup scope:** Define whether seen index includes all external/corpus candidates or only admitted factors.
- **Raw-value retention policy:** Explicitly forbid or bound any fingerprint/derived-value persistence; values must come from DA through provider.

---

## 4. FactorPreprocess (FP): Representation Boundary

FP is the fold-local representation layer, immediately before modeling. It transforms selected factor values into model-input features with explicit causal and state tracking.

### 4.1 Cross-sectional transforms

| Legacy source | Symbol | Decision | Target FP module | Reason | Required tests |
|---|---|---|---|---|---|
| `toolkit/cross_sectional.py` | `cs_rank`, `cs_zscore`, `cs_robust_zscore`, `cs_demean`, `cs_scale`, `cs_minmax` | REUSE_AFTER_TEST | `cross_sectional` | Pure pandas/numpy; same-date permutation invariant | ties, NaNs, permutation, batch/chunk parity |
| `toolkit/cross_sectional.py` | `cs_rank_gauss`, `cs_quantile_bucket`, `cs_signed_power` | REUSE_AFTER_TEST | `cross_sectional` | Pure scipy/numpy; multi-channel representations | ties, clipping, inverse-CDF bounds, bucket edges |
| `APr_utils.py` | `mean_std_winsorize`, `mad_winsorize`, `volatility_winsorize`, `iqr_winsorize`, `quantile_winsorize` | REWRITE | `cross_sectional.winsor` | Algorithms portable; Polars join/group edge-case/null/std contracts to establish | golden numeric cases, constant groups, NaN/null, parity |
| `APr_utils.py` | `zscore_winsorize`, `rankgauss_winsorize`, `tanh_winsorize` | REWRITE | `cross_sectional.winsor` | Candidate algorithms; consolidate rather than port duplicates | reference equivalence, ties, clipping, constants |
| `APr_utils.py` | `zscore_standardize`, `robust_zscore_standardize`, `rank_standardize`, `rank_gaussianize_standardize`, `normal_scores_standardize` | REUSE_AFTER_TEST | `cross_sectional` | Production-core methods; consolidate with `toolkit/cross_sectional` | golden formulas, ties, zero scale, parity |
| `toolkit/cross_sectional.py` | `apply_cross_sectional_transform(frame, transform)` | REWRITE | `policy.registry` | Coupled to `toolkit.registry`; FP must own typed TransformSpec | unknown transform, policy gating, metadata |
| `APr_utils.py` | `minmax_standardize`, `quantile_binning_standardize`, `log_zscore_standardize` | REWRITE | `cross_sectional` | Potentially useful; contracts differ from FP production list | monotonicity, constants, negatives, missingness |

### 4.2 Time-series and causal transforms

| Legacy source | Symbol | Decision | Target FP module | Reason | Required tests |
|---|---|---|---|---|---|
| `APr_utils.py` | `rolling_quantile_winsorize(...)` | REWRITE | `timeseries` | Trailing rolling quantiles; current value inclusion unclear | future-poison, first-window, irregular dates, symbol isolation |
| `APr_utils.py` | `rolling_standardize`, `rolling_robust_standardize`, `rolling_minmax_standardize` | REWRITE | `timeseries` | Trailing grouped windows; source does not consistently make "exclude current" explicit | future-poison, closed boundary, warmup, symbol/date ordering |
| `APr_utils.py` | `volatility_scaling_standardize(shift_vol=True)` | REUSE_AFTER_TEST | `timeseries.volatility` | Matches production causal scale when shift enforced | one-step lag, future-poison, zero vol, lambda bounds, parity |
| `APr_utils.py` | `EWMA_standardize(...)` | REUSE_AFTER_TEST | `timeseries.volatility` | Implementation must be checked for initial value and decay convention | one-step lag, future-poison, zero vol, lambda bounds, parity |
| `factor_indicators_lysj/evaluation/preprocessing.py` | `rolling_zscore(series, window)` | REFERENCE_ONLY | `timeseries` | Clean pandas reference; current window includes `x_t`, causal contract must be changed | future-poison, window closure, warmup, zero std |

### 4.3 Neutralization and exposure

| Legacy source | Symbol | Decision | Target FP module | Reason | Required tests |
|---|---|---|---|---|---|
| `APr_utils.py` | `preprocess_for_neutralization(...)` | REWRITE | `neutralization.design_matrix` | Useful design-matrix idea; `get_dummies` category mapping not persisted, `scale_X` described as global | category stability, train/test vocabulary, missing rows, scaling scope |
| `APr_utils.py` | `multiOLS_neutralize(...)` | REUSE_AFTER_TEST (reference first) | `neutralization.ols` | Per-date residual OLS with pseudoinverse/lstsq | residual exposure ~0, rank deficiency, same-day permutation, missingness |
| `Exposures.py` | `PortfolioExposures._cross_section_ols(sub_df, factor_cols)` | REFERENCE_ONLY | `neutralization.exposure.reference` | Portable OLS/pseudoinverse math seed | statsmodels parity, collinearity, NaN, minimum observations |
| `Exposures.py` | `_standardize_exposures`, `_standardize_frame` | REFERENCE_ONLY | `neutralization.exposure` | Same-date standardization portable; classes are evaluator/report wrappers | isolate pure standardizer, test separately |
| `APr_utils.py` | `lasso_neutralize`, `ridge_neutralize`, `elasticnet_neutralize`, `kernelridge_neutralize`, `bayesianridge_neutralize` | REWRITE / ADVANCED | `neutralization.regularized` | sklearn model fit separately per date; no reusable state, scaling/category contracts | coefficient/residual oracle, alpha sensitivity, deterministic solver, rank deficiency |
| `APr_utils.py` | `polynomial_neutralize` | REWRITE / ADVANCED | `neutralization.design_matrix` | `PolynomialFeatures.fit_transform(X)` local design expansion; feature order not state metadata | feature-order hash, degree/interactions, residual sanity |
| `APr_utils.py` | `huber_neutralize`, `rank_neutralize`, `theilsen_neutralize` | REFERENCE_ONLY | `neutralization.advanced` | Robust/regression variants research-tier; sklearn-coupled | robust regression oracle, small-group behavior, outlier influence |
| `APr_utils.py` | `randomforest_neutralize`, `GBDT_neutralize` | REFERENCE_ONLY | `neutralization.nonlinear` | Supervised nonlinear residualization; expensive, not FP production core | strict opt-in, seed, exposure residual sanity, cost |
| `APr_utils.py` | `PCA_neutralize`, `ICA_neutralize` | REWRITE / ADVANCED | `fitted.decomposition` | `fit_transform(X)` learns and discards components per date; cannot transform test data | train/test poison, component sign/order, feature hash, parity |
| `APr_utils.py` | `partialcorrelation_neutralize` | REFERENCE_ONLY | `neutralization.diagnostics` | Diagnostic, not general residual channel; output shape inconsistent | compare with OLS residual and missing policy |

### 4.4 Advanced and research

| Legacy source | Symbol | Decision | Target FP module | Reason | Required tests |
|---|---|---|---|---|---|
| `APr_utils.py` | `boxcox_compress_winsorize` | REWRITE / ADVANCED | `advanced.power` | Per-date min shift plus fixed lambda; epsilon and positivity ad hoc | negative/constant data, lambda=0, parity |
| `APr_utils.py` | `huber_winsorize`, `ransac_winsorize` | REFERENCE_ONLY | `advanced.robust` | sklearn robust regressors fit per date; fitting must remain date-local and label-free | deterministic seed, small groups, outlier oracle |
| `APr_utils.py` | `yeo_johnson_standardize`, `boxcox_standardize` | REFERENCE_ONLY | `advanced.power` | Advanced; learned lambda/state not exposed or persisted | fold-local fit, lambda/state serialization, negatives, poison |

### 4.5 Preprocessing orchestration

| Legacy source | Symbol | Decision | Target FP module | Reason | Required tests |
|---|---|---|---|---|---|
| `AlphaPurifier.py` | `AlphaPurifier.__init__`, `get_methods`, `winsorize`, `neutralize`, `standardize`, `to_result` | DISCARD | `pipeline` (API concept only) | Method chaining mutates Polars frame; not a valid FP protocol | no direct port, extraction/API-boundary tests |

### 4.6 Corpus and evaluation preprocessing

| Legacy source | Symbol | Decision | Target FP module | Reason | Required tests |
|---|---|---|---|---|---|
| `factor_indicators_lysj/evaluation/preprocessing.py` | `load_factor_series`, `load_vwap_series`, `align_on_overlap_reindex` | CORPUS_ONLY | none | Evaluation-specific factor/VWAP loading; coupled to `EvalConfig` | preserve as evaluation regression corpus only |
| `factor_indicators_lysj/evaluation/preprocessing.py` | `winsorize_by_day(series, date_index, q)` | CORPUS_ONLY | `cross_sectional.winsor` | Evaluation-only per-day helper; useful numeric golden cases | quantile/ties/NaN golden tests |
| `factor_indicators_lysj/evaluation/preprocessing.py` | `prepare_base`, `prepare_horizon`, `prepare_status` | CORPUS_ONLY | none | Evaluation pipeline computes forward returns; unsuitable for FP | forward-return direction and tail-mask regression only |
| `Exposures.py` | `_prepare_future_return_frame`, `_build_beta_panel`, `_build_attribution_panel`, `calc_stats`, plotting methods | DISCARD | none (QE/report layer) | Future return construction, attribution, cumulative returns, Plotly | retain only QE regression/report tests |

### 4.7 Unresolved FP items

- **Stateless vs fitted protocol:** Define `StatelessTransform` and `FittedTransform` boundaries, fit window semantics and serialization.
- **FittedState representation:** Freeze fit start/end, feature IDs/order, transform version/config hash, training-universe snapshot and learned-parameter hash/ref.
- **Causal conventions:** Explicit asset/date ordering, trailing closure, warmup, lag and future-poison tests for all time-series transforms.
- **Neutralization contract:** Typed exposure/design matrix from context; as-of-safe OLS first, then fold-local fitted transforms as advanced.
- **Multichannel output:** Define raw/rank/zscore/residual/exposure channels, freshness/age channel, missing indicators and auditable imputation policy.
- **FeatureBundle terminal type:** Decide canonical name and modeling handoff; keep training out of scope.
- **Cross-sectional consolidation:** Merge duplicates between `toolkit/cross_sectional.py` and `APr_utils.py` under unified FP ownership.

---

## 5. Cross-Package Dependency Matrix

### 5.1 Upward dependencies (data/compute providers)

```
FP <- FA <- FO <- QE <- FE <- DA
                    |       |
                    +-------+----> DA (labels/context)
```

- **DA:** All packages depend on for data, calendar, universe, PIT and context adapters.
- **FE:** QE depends on FactorBatch adapters. FO depends on operator metadata and execution adapters. FP indirectly (through FA).
- **QE:** FA depends on evidence references. FO depends on evaluator protocol. FP indirectly (for cold-start correlation metrics).
- **FA:** FP depends on selected factor set and novelty decisions (if admission needed before representation).
- **FO:** Optional feedback loop; no downward dependencies.

### 5.2 Coordination points

| Point | Ownership | Reason | Unresolved |
|---|---|---|---|
| LabelBundle contract | Caller/modeling + QE | QE validates, never infers label shifts | Exact timing fields and horizon convention |
| FactorBatch adapter | DA/FE -> QE | QE accepts Arrow/NumPy/Polars/Pandas | Adapter scope and optional dependency handling |
| EvaluationContext | Caller/DA -> QE | Universe, industry, size, liquidity, tradability, benchmark | Exactly which fields are required vs optional |
| EvaluationBundle | QE -> FA/FO/caller | Metric versions, warnings, slices, optional series refs | Schema, evidence-ref authority, freshness |
| FactorDefinitionRef | FE -> FA/FO | Canonical FE identity; versions and source metadata | Exact serialization and versioning scheme |
| Complexity profile DTO | FE -> FO | AST depth, weighted operator count, latency | FO-facing DTO fields and versioning |
| MutationSpec versioning | FO core | Parameter role/kind/domain/source/type metadata | Exact role taxonomy and inheritance rules |
| FactorSet/Artifact | FA -> FP/caller | Selected factors, references and policies | Terminal type name and selection/aggregation boundary |
| FeatureBundle | FP -> caller/modeling | Stateless/fitted policies, channels, output form | Terminal type name and training/model boundary |
| Research Control ledger | FO/FA -> ledger | Campaign/trial/event IDs, parent/child, idempotency | Repository path and event schema |

---

## 6. Known Coupling and Legacy Bugs

### 6.1 Data flow safety findings

| Legacy coupling | Risk | Mitigation in new packages |
|---|---|---|
| Admission thresholds in QE metrics | Biased evidence; hard-coded admission | FA owns decision policy; QE emits evidence only |
| Future returns in preprocessing | Leakage if used as train transform | FP consumes only caller-supplied labels; forbid estimation |
| Full-sample fit then split | Training bias | FittedTransform fit window must precede evaluation |
| Global scale/category learning | Train/test data leakage | Date-local or fold-local fit only; explicit contracts |
| Missing-return zero imputation | Bias and hidden failures | QE/FP emit diagnostics; preserve missingness |
| Correlate-and-delete dedup | Over-fits to individual metrics | FA orchestrates novelty through evidence vectors, Pareto |

### 6.2 Implementation-level bugs to avoid

| Bug | Consequence | Fix in migration |
|---|---|---|
| Shift direction inconsistency (Exposures.py vs evaluator) | Wrong return timing | One explicit label convention; both as negative regression tests |
| Rank ties with `method="first"` | Unstable bucketing | QE uses average ties; declare sparse-bin behavior |
| Inner universe merge silently drops rows | Hidden loss of coverage | Consume explicit universe mask; emit diagnostics |
| Current-observation rolling windows | Future-leakage in evaluation | Exclude `t` from windows; explicit lag/closed contracts |
| Per-date fitted models with discarded state | Cannot transform test data | FittedTransform must serialize state or use stateless fallback |
| Monolithic metric files | Hard to test and parallelize | Split by metric type; compose via registry |

---

## 7. Summary: Ownership and No-Build Zones

### 7.1 Ownership by package

| Package | Owns | Does not own |
|---|---|---|
| **QE** | Evidence metrics, diagnostics, contracts | Data fetch, factor compute, admission, representation |
| **FO** | Mutation specs, search logic, LLM interface | FE kernels, QE metrics, FA governance, training |
| **FA** | Identity, registry, lifecycle, decisions, lineage | Raw factor values, metric truth, data storage |
| **FP** | Cross-sectional/time-series transforms, neutralization | Factor compute, evaluation, model training |

### 7.2 Explicit no-build zones

- ❌ No second DSL parser or AST canonicalizer (FE owns).
- ❌ No second PIT/calendar/universe/snapshot manager (DA owns).
- ❌ No second cache platform or distributed DAG (FE/DA own).
- ❌ No metric recalculation or admission thresholds inside QE/FP (FA owns decisions).
- ❌ No factor-value storage in FA (DA owns; FA stores references only).
- ❌ No full-sample fit or model training (FP is fold-local; modeling is out of scope).
- ❌ No `sys.path`, legacy runtime imports or monorepo parent paths (explicit isolation).

---

## 8. Wave 0 Exit Criteria

This consolidation is complete when:

1. ✅ All four legacy audits (QE, FO, FA, FP) are read and function-level decisions recorded.
2. ✅ Cross-package dependencies and unresolved coordination points are listed.
3. ✅ No code changes have been made; draft status is preserved.
4. ✅ File exists at `/home/shw/quant_projects/quant_factor_platform_blueprint/WAVE_0_OUTPUT/MIGRATION_MATRIX_DRAFT.md`.

This consolidation is **not** approval for package implementation. Wave 1 must freeze:

- Public facades and adapter contracts (LabelBundle, FactorBatch, EvaluationContext, FeatureBundle).
- Exception taxonomy and error handling.
- Metric/transform/mutation tier definitions and minimum observations.
- Data-flow and dependency audit sign-off.
- File ownership and code review boundaries.

---

