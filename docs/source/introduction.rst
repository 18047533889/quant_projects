.. _introduction:

Introduction
============

The Quant Projects platform is an enterprise-grade quantitative research infrastructure designed for systematic factor research, backtesting, and portfolio construction.

Architecture
------------

The platform follows a layered architecture:

Data Layer (DataAccess)
~~~~~~~~~~~~~~~~~~~~~~~

Provides unified access to:

* Market data (OHLCV, volume, turnover)
* Fundamental data (financial statements, earnings)
* Corporate actions (splits, dividends, delistings)
* Alternative data sources

All data access is:

* **Point-in-time correct**: No look-ahead bias
* **Multi-market**: Unified API for A-share and US markets
* **Cached**: Intelligent caching with invalidation
* **Governed**: Access control and audit trails

Factor Layer (Factor Engine)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

High-performance factor computation with:

* 1000+ built-in operators (technical, fundamental, statistical)
* Three execution backends: Pandas, Polars, DuckDB
* Automatic query optimization and fusion
* Declarative factor DSL
* Semantic identity tracking for caching

The engine enforces:

* Parameter validation and certification
* Timing contracts (intraday vs daily)
* Lane isolation (production vs research)
* Resource governance

Research Layer
~~~~~~~~~~~~~~

Tools for research workflow:

* **Factor Preprocess**: Cleaning, winsorization, neutralization
* **Quant Evaluator**: IC analysis, turnover, factor decay
* **Factor Optimizer**: Mean-variance, risk parity, factor allocation
* **Research Control**: Experiment tracking, reproducibility

Governance & Production
~~~~~~~~~~~~~~~~~~~~~~~

Production-ready features:

* Hard gates for admission control
* Semantic continuity validation
* Evidence-based certification
* Multi-level audit system (R1-R47)
* Resource autopilot and OOM protection

Design Principles
-----------------

Fail-Closed by Default
~~~~~~~~~~~~~~~~~~~~~~~

The platform prefers raising errors over silent incorrectness:

* Missing data raises, doesn't return NaN
* Schema mismatches fail, don't coerce
* PIT violations are caught, not ignored

This ensures bugs surface during research, not in production.

Explicit Over Implicit
~~~~~~~~~~~~~~~~~~~~~~~

Parameters and behavior are explicit:

* No hidden lookback windows
* No implicit alignment or filling
* Timing contracts declared upfront
* Parameter domains certified

Composability
~~~~~~~~~~~~~

Operators compose cleanly:

* Pure functions where possible
* Stateful operators declare state explicitly
* Recursive kernels with checkpoint semantics
* Cost-based query planning for complex DAGs

Use Cases
---------

Factor Research
~~~~~~~~~~~~~~~

Discover and validate new alpha factors:

1. Load data from DataAccess
2. Define factor using engine operators
3. Compute over universe
4. Evaluate with Quant Evaluator
5. Track experiment in Research Control

Portfolio Construction
~~~~~~~~~~~~~~~~~~~~~~

Build and backtest portfolios:

1. Select factors based on IC/decay
2. Optimize weights with Factor Optimizer
3. Apply constraints and risk limits
4. Backtest with transaction costs
5. Generate performance reports

Production Deployment
~~~~~~~~~~~~~~~~~~~~~

Deploy factors to live trading:

1. Certify factor meets hard gates
2. Generate evidence ledger
3. Deploy with timing/lane contracts
4. Monitor resource usage
5. Audit semantic continuity

Target Audience
---------------

This platform is designed for:

* Quantitative researchers developing alpha factors
* Portfolio managers constructing systematic strategies
* Data engineers maintaining market data pipelines
* Risk managers validating factor exposures
* System architects designing production infrastructure

Prerequisites
-------------

* Python 3.10+
* Pandas, Polars, or DuckDB experience
* Understanding of financial markets
* Basic statistics and time series analysis

Next Steps
----------

* :ref:`installation` - Set up the platform
* :ref:`quickstart` - Run your first factor
* :ref:`usage-examples` - Detailed examples
