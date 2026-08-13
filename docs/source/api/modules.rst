.. _api-reference:

API Reference
=============

This section contains the complete API documentation for all modules.

.. toctree::
   :maxdepth: 2
   :caption: Modules:

   dataaccess
   factor_engine
   factor_preprocess
   factor_optimizer
   factor_assets
   quant_evaluator
   research_control

Module Overview
---------------

DataAccess
~~~~~~~~~~

Unified data access layer for market data, fundamentals, and alternative data.

Key classes:

* :class:`~dataaccess.DataAccessClient` - Main client interface
* :class:`~dataaccess.MarketDataProvider` - Market data provider
* :class:`~dataaccess.FundamentalDataProvider` - Fundamental data provider
* :class:`~dataaccess.CacheManager` - Data caching system

Factor Engine
~~~~~~~~~~~~~

High-performance factor computation engine with 1000+ operators.

Key classes:

* :class:`~factor_engine.FactorEngine` - Main engine interface
* :class:`~factor_engine.Operator` - Base operator class
* :class:`~factor_engine.QueryOptimizer` - Query optimization
* :class:`~factor_engine.ResourceGovernor` - Resource management

Factor Preprocess
~~~~~~~~~~~~~~~~~

Data preprocessing, cleaning, and normalization utilities.

Key functions:

* :func:`~factor_preprocess.winsorize` - Outlier treatment
* :func:`~factor_preprocess.standardize` - Standardization
* :func:`~factor_preprocess.neutralize` - Factor neutralization
* :func:`~factor_preprocess.combine_factors` - Factor combination

Factor Optimizer
~~~~~~~~~~~~~~~~

Portfolio construction and optimization tools.

Key classes:

* :class:`~factor_optimizer.MeanVarianceOptimizer` - Mean-variance optimization
* :class:`~factor_optimizer.RiskParityOptimizer` - Risk parity optimization
* :class:`~factor_optimizer.FactorPortfolioOptimizer` - Factor-based portfolios

Factor Assets
~~~~~~~~~~~~~

Asset universe management and filtering.

Key classes:

* :class:`~factor_assets.UniverseManager` - Universe management
* :class:`~factor_assets.AssetFilter` - Asset filtering
* :class:`~factor_assets.ListingManager` - Listing/delisting tracking

Quant Evaluator
~~~~~~~~~~~~~~~

Factor evaluation and backtesting framework.

Key classes:

* :class:`~quant_evaluator.FactorEvaluator` - Factor evaluation
* :class:`~quant_evaluator.Backtester` - Backtesting engine
* :class:`~quant_evaluator.PerformanceAnalyzer` - Performance analysis

Key functions:

* :func:`~quant_evaluator.compute_ic` - Information coefficient
* :func:`~quant_evaluator.compute_decay` - Factor decay analysis
* :func:`~quant_evaluator.compute_turnover` - Turnover calculation

Research Control
~~~~~~~~~~~~~~~~

Experiment tracking and research governance.

Key classes:

* :class:`~research_control.ExperimentTracker` - Experiment tracking
* :class:`~research_control.Experiment` - Experiment object
* :class:`~research_control.ArtifactStore` - Artifact storage
