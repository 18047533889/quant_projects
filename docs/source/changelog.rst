Changelog
=========

All notable changes to the Quant Projects platform will be documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_,
and this project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

[1.0.0] - 2026-08-14
--------------------

Initial release of the Quant Projects platform.

Added
~~~~~

**DataAccess Module**

* Unified data access layer for A-share and US markets
* Point-in-time correct fundamental data loading
* Multi-source data provider support (COS, local, remote)
* Intelligent caching with invalidation
* Schema validation and versioning
* Snapshot verification system
* Access control and audit trails

**Factor Engine Module**

* 1000+ built-in operators (technical, fundamental, statistical)
* Three execution backends: Pandas, Polars, DuckDB
* Declarative factor DSL with expression language
* Query optimization (CSE, fusion, cost-based planning)
* Automatic backend selection and query lowering
* Parameter certification and domain validation
* Timing contracts (intraday vs daily)
* Lane isolation (production vs research)
* Resource governance and OOM protection
* Semantic identity tracking for caching
* Operator admission control with hard gates

**Factor Preprocess Module**

* Outlier treatment (winsorization, truncation, MAD)
* Standardization (z-score, rank, quantile)
* Factor neutralization (industry, size, multi-factor)
* Missing data handling (forward fill, interpolation)
* Factor combination (equal, IC-weighted, optimal)
* Cross-sectional operations (demean, demedian)
* Time series smoothing (exponential, rolling)

**Factor Optimizer Module**

* Mean-variance optimization
* Risk parity optimization
* Factor-based portfolio construction
* Black-Litterman model support
* Constraint management system
* Covariance estimation (sample, shrinkage, factor model)
* Portfolio rebalancing utilities
* Risk decomposition

**Factor Assets Module**

* Universe management (HS300, ZZ500, ZZ1000, custom)
* Asset filtering by liquidity, market cap, listing status
* Listing/delisting tracking with PIT correctness
* Industry classification (CITIC, SW, custom)
* Multi-market support (A-share, US)

**Quant Evaluator Module**

* Factor evaluation (IC, rank IC, IC decay)
* Backtesting engine with transaction costs
* Performance metrics (Sharpe, Sortino, Calmar, max drawdown)
* Factor turnover analysis
* Autocorrelation computation
* Performance attribution
* Risk decomposition

**Research Control Module**

* Experiment tracking and versioning
* Artifact storage (factors, backtests, models)
* Governance validation
* Reproducibility support
* Experiment comparison tools

**Documentation**

* Complete API reference documentation
* Quickstart guide and tutorials
* Detailed usage examples
* Architecture overview
* Contributing guide

**Testing & Quality**

* Comprehensive test suite (1000+ tests)
* 80%+ code coverage
* Type hints throughout codebase
* CI/CD pipeline
* Multi-level audit system (R1-R47)

**Infrastructure**

* Multi-backend support (Pandas, Polars, DuckDB)
* Distributed execution support
* Memory-efficient streaming
* Spill-to-disk for large computations
* Resource autopilot
* Fail-closed error handling

Changed
~~~~~~~

* N/A (initial release)

Deprecated
~~~~~~~~~~

* N/A (initial release)

Removed
~~~~~~~

* N/A (initial release)

Fixed
~~~~~

* N/A (initial release)

Security
~~~~~~~~

* Credential management with access control
* Data access audit logging
* PIT validation to prevent look-ahead bias
* Schema validation to prevent data corruption
* Resource limits to prevent OOM
