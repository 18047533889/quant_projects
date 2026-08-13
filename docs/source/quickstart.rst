.. _quickstart:

Quickstart Guide
================

This guide walks through basic usage of the Quant Projects platform.

Your First Factor
-----------------

Let's compute a simple momentum factor on A-share stocks.

Step 1: Load Data
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from dataaccess import DataAccessClient

   # Initialize client
   client = DataAccessClient(market='ashare')

   # Get stock universe
   universe = client.get_universe('hs300', date='2024-01-31')
   print(f"Universe size: {len(universe)}")

   # Load price data
   prices = client.get_market_data(
       universe=universe,
       start_date='2023-01-01',
       end_date='2024-01-31',
       fields=['close', 'volume']
   )

Step 2: Compute Factor
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from factor_engine import FactorEngine

   # Initialize engine
   engine = FactorEngine(backend='polars')

   # Define momentum factor
   factor = engine.compute(
       operator='ts_returns',
       data=prices,
       params={'period': 20}
   )

   print(factor.head())

Step 3: Evaluate Performance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from quant_evaluator import FactorEvaluator

   # Initialize evaluator
   evaluator = FactorEvaluator()

   # Compute IC
   ic_stats = evaluator.compute_ic(
       factor=factor,
       forward_returns=1,  # 1-day forward return
       method='rank'
   )

   print(f"Mean IC: {ic_stats['mean_ic']:.4f}")
   print(f"IC Ratio: {ic_stats['ic_ratio']:.4f}")

Step 4: Save Results
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from research_control import ExperimentTracker

   # Track experiment
   tracker = ExperimentTracker()
   experiment = tracker.create_experiment(
       name='momentum_20d',
       description='20-day momentum factor'
   )

   # Log metrics
   experiment.log_metrics(ic_stats)
   experiment.save_factor(factor)

   print(f"Experiment saved: {experiment.id}")

Common Patterns
---------------

Working with Multiple Factors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Compute multiple factors efficiently:

.. code-block:: python

   from factor_engine import FactorEngine

   engine = FactorEngine()

   # Define factor specifications
   factor_specs = [
       {'name': 'mom_20', 'operator': 'ts_returns', 'params': {'period': 20}},
       {'name': 'mom_60', 'operator': 'ts_returns', 'params': {'period': 60}},
       {'name': 'vol_20', 'operator': 'ts_std', 'params': {'period': 20}},
   ]

   # Batch compute
   factors = engine.compute_batch(
       specs=factor_specs,
       data=prices
   )

   # Access individual factors
   for name, factor_data in factors.items():
       print(f"{name}: {factor_data.shape}")

Cross-Sectional Operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Apply cross-sectional standardization:

.. code-block:: python

   from factor_preprocess import standardize_cross_section

   # Z-score normalization
   normalized = standardize_cross_section(
       factor,
       method='zscore',
       clip=3.0  # Clip at ±3 std
   )

   # Rank normalization
   ranked = standardize_cross_section(
       factor,
       method='rank'
   )

Neutralization
~~~~~~~~~~~~~~

Neutralize factor against industry/size:

.. code-block:: python

   from factor_preprocess import neutralize

   # Load industry classifications
   industries = client.get_industry_class(universe, date='2024-01-31')

   # Load market cap
   market_cap = client.get_market_data(universe, fields=['market_cap'])

   # Neutralize
   neutral_factor = neutralize(
       factor,
       industries=industries,
       controls={'market_cap': market_cap},
       method='regression'
   )

Portfolio Construction
~~~~~~~~~~~~~~~~~~~~~~

Build optimal portfolio from factors:

.. code-block:: python

   from factor_optimizer import PortfolioOptimizer

   # Initialize optimizer
   optimizer = PortfolioOptimizer()

   # Combine factors
   combined_factor = (
       factors['mom_20'] * 0.5 +
       factors['mom_60'] * 0.3 +
       factors['vol_20'] * 0.2
   )

   # Optimize weights
   weights = optimizer.optimize(
       factor=combined_factor,
       method='mean_variance',
       constraints={
           'long_only': True,
           'max_position': 0.05,
           'target_volatility': 0.15
       }
   )

   print(f"Number of positions: {(weights > 0).sum()}")

Backtesting
~~~~~~~~~~~

Run historical backtest:

.. code-block:: python

   from quant_evaluator import Backtester

   # Initialize backtester
   backtester = Backtester(
       initial_capital=10_000_000,
       commission=0.001,  # 10 bps
       slippage=0.0005    # 5 bps
   )

   # Run backtest
   results = backtester.run(
       signals=weights,
       start_date='2023-01-01',
       end_date='2024-01-31',
       rebalance_freq='monthly'
   )

   # View metrics
   print(results.summary())
   print(f"Sharpe Ratio: {results.sharpe:.2f}")
   print(f"Max Drawdown: {results.max_drawdown:.2%}")

Advanced Usage
--------------

Custom Operators
~~~~~~~~~~~~~~~~

Register custom factor operators:

.. code-block:: python

   from factor_engine import register_operator
   import numpy as np

   @register_operator(
       name='custom_momentum',
       inputs=['close'],
       params={'fast': int, 'slow': int}
   )
   def custom_momentum(close, fast=10, slow=30):
       """Custom momentum indicator."""
       fast_ma = close.rolling(fast).mean()
       slow_ma = close.rolling(slow).mean()
       return (fast_ma - slow_ma) / slow_ma

   # Use custom operator
   factor = engine.compute(
       operator='custom_momentum',
       data=prices,
       params={'fast': 10, 'slow': 30}
   )

Query Optimization
~~~~~~~~~~~~~~~~~~

Enable query optimization for complex factor DAGs:

.. code-block:: python

   from factor_engine import FactorEngine, QueryOptimizer

   engine = FactorEngine(
       backend='duckdb',
       optimizer=QueryOptimizer(
           enable_cse=True,        # Common subexpression elimination
           enable_fusion=True,     # Operator fusion
           cost_threshold=1000     # Cost-based optimization
       )
   )

   # Complex factor tree will be optimized
   factor = engine.compute_expression("""
       ((close / ts_delay(close, 1) - 1) * volume) /
       ts_mean(volume, 20)
   """)

Resource Management
~~~~~~~~~~~~~~~~~~~

Control memory and compute resources:

.. code-block:: python

   from factor_engine import ResourceGovernor

   # Set resource limits
   governor = ResourceGovernor(
       max_memory_gb=16,
       max_cpu_cores=8,
       enable_spill=True,
       spill_threshold=0.8  # Spill when 80% full
   )

   engine = FactorEngine(resource_governor=governor)

   # Compute with resource monitoring
   with governor.monitor():
       factor = engine.compute_batch(specs=large_factor_list)
       print(f"Peak memory: {governor.peak_memory_gb:.2f} GB")

Next Steps
----------

* :ref:`usage-examples` - More detailed examples
* :ref:`api-reference` - Complete API documentation
* :ref:`contributing` - Contribute to the project
