# QuantEvaluator Changelog

All notable changes to the QuantEvaluator package will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.0.1a1] - 2026-08-14

### Added - Core Functionality

#### Contracts
- `FactorBatch`: Batch-first factor value container with explicit axes
- `LabelBundle`: Forward labels with strict timing contracts
- `AxisRef`: Explicit axis metadata for dimensions
- `EvaluationRequest`: Typed evaluation request contract
- `EvaluationBundle`: Versioned evaluation result bundle
- `MetricValue`: Individual metric result with metadata
- `FactorDiagnosis`: Per-factor health diagnostics

#### Error Taxonomy
- `QuantEvaluatorError`: Base exception class
- `ContractError`: Input validation errors
  - `SchemaVersionError`: Version mismatch
  - `MissingInputError`: Required input missing
  - `InvalidContractError`: Contract violation
  - `TimingContractError`: Timing invariant violated
  - `SnapshotMismatchError`: Snapshot mismatch
- `CapabilityError`: Unsupported feature errors
  - `UnsupportedMetricError`: Metric not available
  - `OptionalDependencyMissing`: Missing optional dependency
- `DataError`: Data quality errors
  - `InsufficientObservations`: Not enough valid data
  - `InvalidValidityMask`: Validity mask malformed
  - `MissingLabelError`: Labels not provided
  - `EvidenceUnavailableError`: Evidence not found
- `NumericalFailure`: Computation errors
  - `OverflowOrNonFiniteError`: Numerical instability

#### Metrics - Quality
- `compute_coverage`: Overall valid observation fraction
- `compute_per_time_coverage`: Per-time coverage array

#### Metrics - Information Coefficient
- `compute_daily_ic`: Daily Pearson/Spearman IC with pairwise-finite filtering
- `compute_mean_ic`: Mean IC with minimum period validation

#### Metrics - IC Summary
- `compute_icir`: Information Coefficient / Information Ratio
- `compute_ic_tstat`: T-statistic and p-value for IC
- `compute_ic_decay`: IC across multiple horizons
- `compute_ic_stability`: Rolling window IC stability

#### Metrics - Quantile Analysis
- `assign_quantiles`: Cross-sectional quantile binning
- `compute_quantile_returns`: Average returns per quantile
- `compute_top_bottom_spread`: Top minus bottom quantile spread

#### Metrics - Turnover
- `compute_turnover`: Canonical 0.5 * sum(|delta_weights|)
- `compute_turnover_series`: Time series of turnover
- `estimate_turnover_from_ranks`: Turnover proxy from rank correlation

#### Metrics - Temporal Analysis
- `compute_autocorrelation`: ACF of factor or IC series
- `compute_ic_autocorrelation`: IC decay over lags
- `compute_rank_stability`: Rank correlation stability
- `compute_mean_rank_stability`: Mean rank stability
- `compute_factor_turnover_rate`: Factor-based turnover rate
- `compute_half_life`: AR(1) half-life estimation

#### Metrics - Exposure Analysis
- `compute_factor_loadings`: Risk factor exposures via OLS
- `compute_sector_exposure`: Sector concentration metrics
- `compute_style_exposure`: Style factor loadings
- `compute_concentration_hhi`: Herfindahl-Hirschman Index

#### Metrics - Robustness
- `compute_subsample_ic`: Bootstrap subsample IC stability
- `compute_subsample_ic_std`: Subsample IC standard deviation
- `compute_hac_variance`: HAC variance (Newey-West)
- `compute_hac_tstat`: HAC-corrected t-statistics
- `compute_block_bootstrap_ci`: Block bootstrap confidence intervals

#### Metrics - Distribution Analysis
- `compute_skewness`: Distribution skewness
- `compute_kurtosis`: Excess kurtosis
- `detect_outliers_iqr`: IQR-based outlier detection
- `detect_outliers_zscore`: Z-score outlier detection
- `compute_outlier_ratio`: Outlier fraction
- `compute_higher_moments`: Mean, std, skew, kurtosis

#### Metrics - Portfolio Statistics
- `compute_long_short_returns`: Long-short portfolio returns
- `compute_sharpe_ratio`: Risk-adjusted return
- `compute_maximum_drawdown`: Largest peak-to-trough decline
- `compute_calmar_ratio`: Return / max drawdown
- `compute_sortino_ratio`: Downside-risk adjusted return
- `compute_win_rate`: Fraction of positive returns

#### Metrics - Multiple Testing
- `bonferroni_correction`: Bonferroni-adjusted p-values
- `benjamini_hochberg_correction`: FDR control (Benjamini-Hochberg)
- `holm_bonferroni_correction`: Holm's sequential method
- `sidak_correction`: Šidák correction
- `compute_fdr`: False discovery rate estimate

#### Runtime Engine
- `Evaluator`: Main evaluation orchestrator
  - Metric dependency resolution
  - Intermediate result caching
  - Memory-aware batch chunking
  - Resource budget tracking
- `IntermediateCache`: Caching layer for computed intermediates
- `ComputationBudget`: Resource limit specification
- `BudgetTracker`: Resource usage tracking
- `ResourceUsage`: Current resource snapshot

#### Planner
- `BatchPlan`: Memory-aware batch chunking plan
- `ChunkDescriptor`: Individual chunk specification
- `create_batch_plan`: Automatic batch planning
- `MetricDependencyGraph`: Metric dependency DAG
- `MetricNode`: Individual metric with dependencies
- `resolve_metric_dependencies`: Automatic dependency resolution
- `MetricKind`: Metric category enumeration

#### Registry
- `MetricRegistry`: Centralized metric catalog
- `MetricSpec`: Metric specification with metadata
- `MetricStatus`: STABLE, EXPERIMENTAL, DEPRECATED
- `MetricTier`: CORE, EXTENDED, RESEARCH
- Pre-registered metrics:
  - Core: mean_ic, ic_std, ic_ir, coverage, turnover, quantile_spread
  - Extended: hac_tstat, subsample_stability, ic_autocorr_lag1, rank_stability, half_life
  - Research: block_bootstrap_ci, factor_turnover_rate, quantile_returns_full
- `FACTOR_CORE`: Core factor evaluation preset
- `FACTOR_EXTENDED`: Extended metrics preset
- `PRODUCTION_DAILY`: Production daily evaluation preset

#### Diagnosis
- `diagnose_factor`: Per-factor health diagnostics
- `diagnose_all_factors`: Batch factor diagnostics

#### Adapters (Stubs)
- `DAAdapter`: DataAccess integration (optional)
- `FEAdapter`: FactorEngine integration (optional)

### Testing
- 276 test functions across 28 test files
- Unit tests for all metrics
- Integration tests for full pipeline
- Parity test infrastructure (ready for fast kernels)
- Contract validation tests
- Edge case coverage (NaN, empty, single observation)
- Golden value tests for regression detection

### Documentation
- README with quick start guide
- Package structure overview
- Core contracts documentation
- Error taxonomy
- Design principles documented

### Known Limitations

#### Not Yet Implemented
- Fast/optimized kernels (reference implementations only)
- Distributed execution
- Streaming evaluation
- Real-time monitoring
- Advanced portfolio backtesting
- Multi-asset class support

#### Performance
- Large batch memory usage not optimized
- No parallel evaluation across factors
- No GPU acceleration
- Chunking strategy is basic

#### Features
- No automatic label generation
- No data fetching integration
- No model training integration
- No automated report generation

### Breaking Changes

N/A - Initial alpha release

### Dependencies
- `numpy >= 1.24.0` (required)
- `scipy >= 1.10.0` (optional, for Spearman correlation)

### Deprecations

None

### Security

None

---

## [Unreleased]

### Planned for 0.1.0 (Wave 2)

#### Performance
- [ ] Fast IC kernel with Numba JIT
- [ ] Optimized quantile binning with Cython
- [ ] Parallel evaluation across factors
- [ ] Distributed execution with Ray/Dask
- [ ] GPU-accelerated kernels

#### Features
- [ ] Advanced HAC standard errors
- [ ] Cross-sectional dispersion metrics
- [ ] Factor timing analysis
- [ ] Tail risk metrics (CVaR, expected shortfall)
- [ ] Multi-horizon IC surface
- [ ] Regime-conditional analysis

#### Integration
- [ ] Complete DA adapter implementation
- [ ] Complete FE adapter implementation
- [ ] FA evidence bundle integration
- [ ] Automated report generation

#### Quality
- [ ] API freeze
- [ ] Full corpus regression tests
- [ ] Performance benchmarks
- [ ] Production readiness review

### Planned for 0.2.0 (Wave 3)

- [ ] Streaming evaluation
- [ ] Real-time monitoring dashboard
- [ ] Multi-asset class support
- [ ] AutoML integration
- [ ] Web API

---

## Version History

- **0.0.1a1** (2026-08-14): Initial alpha release

---

## Upgrade Guide

N/A - Initial release

---

## Contributors

Quant Platform Team

---

## Release Process

### Alpha Releases (0.0.x)

- Breaking changes allowed
- API not frozen
- Experimental features
- Internal use only

### Beta Releases (0.x.0)

- API stabilizing
- Backwards compatibility preferred
- External testing encouraged
- Production use at own risk

### Stable Releases (1.x.0)

- API frozen
- Backwards compatibility guaranteed (within major version)
- Production ready
- Long-term support

---

**Last Updated:** 2026-08-14
