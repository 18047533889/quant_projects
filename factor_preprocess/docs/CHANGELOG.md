# FactorPreprocess Changelog

All notable changes to FactorPreprocess will be documented in this file.

## [0.1.0] - 2026-08-14

### Added - Core Functionality

#### Contracts
- `PreprocessingPolicy`: Ordered transform specifications
- `TransformSpec`: Single transform specification (kind, mode, version, parameters)
- `FittedState`: Immutable fitted transform state with fit window metadata
- `FeatureBundle`: Model-ready features with channel metadata
- `TransformKind`: Enum (CROSS_SECTIONAL, ROLLING, NEUTRALIZATION, REPRESENTATION)
- `TransformMode`: Enum (STATELESS, FITTED)

#### Cross-Sectional Transforms (5 functions)
- `cs_rank`: Cross-sectional ranking (average, min, max tie-breaking)
- `cs_zscore`: Z-score standardization per time
- `cs_demean`: Demean per time
- `cs_winsor`: Winsorization by percentile
- `cs_scale`: Scale to target std

#### Rolling Transforms (4 functions)
- `rolling_mean`: Causal rolling mean (shift=1)
- `rolling_std`: Causal rolling standard deviation
- `rolling_zscore`: Causal rolling z-score
- `ewma`: Exponentially weighted moving average

#### Volatility Transforms (3 functions)
- `volatility_scale`: Scale by realized volatility
- `volatility_scale_returns`: Scale returns by volatility
- `realized_volatility`: Compute realized volatility

#### Missingness Transforms (4 functions)
- `forward_fill`: Forward fill with max lag
- `missing_indicator`: Binary missingness flag
- `missing_run_length`: Consecutive missing periods
- `missing_rate`: Rolling missingness rate

#### Freshness Transforms (4 functions)
- `days_since_update`: Age of last non-missing value
- `observation_age`: Age from observation date
- `freshness_score`: Exponential decay score
- `stale_data_indicator`: Binary staleness flag

#### Neutralization (5 neutralizers + 7 diagnostics)
- `ols_neutralize`: OLS residuals
- `ridge_neutralize`: Ridge regression residuals
- `lasso_neutralize`: Lasso regression residuals
- `elastic_net_neutralize`: Elastic net residuals
- `compute_exposures`: Exposure calculation
- Diagnostics: condition number, exposure correlation, residual exposures, variance reduction, etc.

#### Representation Builders
- `build_multichannel`: Multi-channel representation (raw, rank, zscore)
- `build_linear_ready`: Features for linear models (imputation, intercept, standardization)
- `build_tree_ready`: Features for tree models (missing indicators, outlier clipping)
- `MultichannelConfig`, `LinearReadyConfig`, `TreeReadyConfig`: Configuration dataclasses

#### Registry
- `TransformRegistry`: Centralized transform catalog
- `TransformMetadata`: Transform specification with category, version, parameters
- `TransformCategory`: Enum (CROSS_SECTIONAL, TEMPORAL, VOLATILITY, MISSINGNESS, FRESHNESS, NEUTRALIZATION, REPRESENTATION)
- `create_default_registry()`: Factory with 26 built-in transforms

#### Policy Registry
- `PolicyRegistry`: Preset transformation pipelines
- `PolicyPreset`: Named policy with steps
- `TransformStep`: Single step in pipeline
- `PolicyLevel`: Enum (RESEARCH, STAGING, PRODUCTION)
- 6 built-in policies: cs_only, causal_basic, production_full, returns_preprocessing, minimal, research_full

### Testing
- 27 test files, comprehensive coverage
- Transform correctness tests
- Neutralization quality tests
- Representation builder tests
- Parity tests (reference vs fast)
- Future poison tests (temporal leakage detection)
- Fold boundary tests (train/test split safety)

### Documentation
- README with scope and out-of-scope
- Design principles documented
- Causal safety emphasized

### Known Limitations

#### Not Implemented
- Regularized neutralization (ridge/lasso are basic)
- Multi-channel advanced representations
- Fitted state serialization
- Fast kernel implementations
- GPU acceleration

#### Performance
- Rolling transforms not optimized (pure pandas)
- No parallel processing
- Large dataset memory usage

### Dependencies
- `numpy>=1.24.0`, `pandas>=2.0.0`, `scipy>=1.10.0`

### Design Principles
1. **Batch-first API**: Multi-factor processing
2. **Explicit causality**: All rolling ops exclude current (shift=1)
3. **Fold-local fitting**: No full-sample stats before split
4. **Asset isolation**: Time-series ops grouped per asset
5. **NaN preservation**: Explicit validity, no silent fills

---

## [Unreleased]

### Planned for 0.2.0
- [ ] Fast kernel implementations
- [ ] Fitted state serialization
- [ ] Advanced multi-channel representations
- [ ] GPU acceleration
- [ ] Parallel processing

---

**Last Updated:** 2026-08-14
