.. _usage-examples:

Usage Examples
==============

This section provides detailed examples for common use cases.

Data Access Examples
--------------------

Loading Market Data
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from dataaccess import DataAccessClient
   from datetime import datetime, timedelta

   client = DataAccessClient(market='ashare')

   # Single stock
   df = client.get_market_data(
       symbols='000001.SZ',
       start_date='2024-01-01',
       end_date='2024-01-31',
       fields=['open', 'high', 'low', 'close', 'volume', 'amount']
   )

   # Multiple stocks
   symbols = ['000001.SZ', '000002.SZ', '600000.SH']
   df = client.get_market_data(
       symbols=symbols,
       start_date='2024-01-01',
       end_date='2024-01-31'
   )

   # Full universe
   universe = client.get_universe('all', date='2024-01-31')
   df = client.get_market_data(
       symbols=universe,
       start_date='2024-01-01',
       end_date='2024-01-31'
   )

Fundamental Data
~~~~~~~~~~~~~~~~

.. code-block:: python

   # Financial statements (point-in-time)
   financials = client.get_fundamentals(
       symbols='000001.SZ',
       start_date='2020-01-01',
       end_date='2024-01-31',
       fields=['revenue', 'net_income', 'total_assets', 'total_equity'],
       pit_correct=True  # Point-in-time correct
   )

   # Valuation metrics
   valuation = client.get_fundamentals(
       symbols=universe,
       date='2024-01-31',
       fields=['pe_ttm', 'pb_lf', 'ps_ttm', 'pcf_ttm']
   )

   # Earnings surprises
   earnings = client.get_earnings_surprises(
       symbols=universe,
       start_date='2023-01-01',
       end_date='2024-01-31'
   )

Alternative Data
~~~~~~~~~~~~~~~~

.. code-block:: python

   # Analyst ratings
   ratings = client.get_analyst_ratings(
       symbols=universe,
       start_date='2024-01-01',
       end_date='2024-01-31'
   )

   # Short interest
   short_interest = client.get_short_interest(
       symbols=universe,
       start_date='2024-01-01',
       end_date='2024-01-31'
   )

   # News sentiment
   sentiment = client.get_news_sentiment(
       symbols='000001.SZ',
       start_date='2024-01-01',
       end_date='2024-01-31'
   )

Factor Engine Examples
----------------------

Technical Indicators
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from factor_engine import FactorEngine

   engine = FactorEngine()

   # Moving averages
   sma_20 = engine.compute('ts_mean', data=prices, params={'period': 20})
   ema_20 = engine.compute('ts_ema', data=prices, params={'period': 20, 'adjust': False})

   # Momentum indicators
   roc = engine.compute('ts_roc', data=prices, params={'period': 20})
   rsi = engine.compute('ts_rsi', data=prices, params={'period': 14})
   macd = engine.compute('ts_macd', data=prices, params={'fast': 12, 'slow': 26, 'signal': 9})

   # Volatility indicators
   std = engine.compute('ts_std', data=prices, params={'period': 20})
   atr = engine.compute('ts_atr', data=prices, params={'period': 14})
   bbands = engine.compute('ts_bbands', data=prices, params={'period': 20, 'std_dev': 2})

Statistical Factors
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Time series statistics
   skew = engine.compute('ts_skewness', data=returns, params={'period': 60})
   kurt = engine.compute('ts_kurtosis', data=returns, params={'period': 60})

   # Cross-sectional statistics
   cs_rank = engine.compute('cs_rank', data=factor)
   cs_zscore = engine.compute('cs_zscore', data=factor)

   # Correlation measures
   corr = engine.compute('ts_corr',
                        data={'x': returns, 'y': market_returns},
                        params={'period': 60})

   # Beta calculation
   beta = engine.compute('ts_beta',
                        data={'returns': returns, 'benchmark': market_returns},
                        params={'period': 252})

Fundamental Factors
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Valuation factors
   ep_ratio = engine.compute('divide',
                            data={'numerator': earnings, 'denominator': price})

   bp_ratio = engine.compute('divide',
                            data={'numerator': book_value, 'denominator': market_cap})

   # Quality factors
   roa = engine.compute('divide',
                       data={'numerator': net_income, 'denominator': total_assets})

   roe = engine.compute('divide',
                       data={'numerator': net_income, 'denominator': equity})

   # Growth factors
   revenue_growth = engine.compute('ts_growth',
                                  data=revenue,
                                  params={'period': 4})  # YoY quarterly

   # Financial health
   debt_to_equity = engine.compute('divide',
                                  data={'numerator': total_debt, 'denominator': equity})

Complex Factor Expressions
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Use expression language
   factor = engine.compute_expression("""
       let price_mom = ts_returns(close, 20)
       let vol_adj = ts_std(returns, 20)
       let factor = price_mom / vol_adj
       cs_neutralize(factor, industry)
   """)

   # Composite alpha
   composite = engine.compute_expression("""
       let momentum = ts_returns(close, 20)
       let reversal = -ts_returns(close, 5)
       let value = 1 / pe_ratio
       let quality = roe

       cs_zscore(momentum * 0.3 + reversal * 0.2 + value * 0.3 + quality * 0.2)
   """)

Factor Preprocessing Examples
------------------------------

Outlier Treatment
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from factor_preprocess import winsorize, truncate

   # Winsorization (replace extremes)
   winsorized = winsorize(factor, lower=0.01, upper=0.99)

   # Truncation (remove extremes)
   truncated = truncate(factor, lower=-3, upper=3, method='zscore')

   # MAD-based (more robust)
   mad_winsorized = winsorize(factor, method='mad', threshold=3)

Standardization
~~~~~~~~~~~~~~~

.. code-block:: python

   from factor_preprocess import standardize

   # Z-score
   zscore = standardize(factor, method='zscore')

   # Rank (uniform distribution)
   rank = standardize(factor, method='rank')

   # Quantile (specified distribution)
   quantile = standardize(factor, method='quantile', target_dist='normal')

Neutralization
~~~~~~~~~~~~~~

.. code-block:: python

   from factor_preprocess import neutralize

   # Industry neutralization
   ind_neutral = neutralize(
       factor,
       industries=industry_codes,
       method='demean'
   )

   # Industry + size neutralization
   neutral = neutralize(
       factor,
       industries=industry_codes,
       controls={'market_cap': market_cap},
       method='regression'
   )

   # Multiple controls
   multi_neutral = neutralize(
       factor,
       industries=industry_codes,
       controls={
           'market_cap': market_cap,
           'book_to_market': btm,
           'momentum': mom
       }
   )

Factor Combination
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from factor_preprocess import combine_factors

   # Equal weight
   combined = combine_factors([factor1, factor2, factor3], method='equal')

   # IC-weighted
   combined = combine_factors(
       [factor1, factor2, factor3],
       method='ic_weight',
       ic_values=[0.05, 0.03, 0.04]
   )

   # Optimal (max IC)
   combined = combine_factors(
       [factor1, factor2, factor3],
       method='optimal',
       forward_returns=forward_ret,
       lookback=252
   )

Evaluation Examples
-------------------

Information Coefficient
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from quant_evaluator import compute_ic

   # Single period IC
   ic = compute_ic(factor, forward_returns, method='rank')

   # Time series of IC
   ic_ts = compute_ic(
       factor,
       forward_returns,
       method='rank',
       groupby='date'
   )

   # Statistics
   print(f"Mean IC: {ic_ts.mean():.4f}")
   print(f"IC Std: {ic_ts.std():.4f}")
   print(f"IC Ratio: {ic_ts.mean() / ic_ts.std():.4f}")
   print(f"IC > 0: {(ic_ts > 0).mean():.2%}")

Factor Decay
~~~~~~~~~~~~

.. code-block:: python

   from quant_evaluator import compute_decay

   # Compute IC at multiple horizons
   decay = compute_decay(
       factor,
       forward_returns_list=[1, 5, 10, 20],  # days
       method='rank'
   )

   # Plot decay curve
   import matplotlib.pyplot as plt
   plt.plot(decay['horizon'], decay['ic'])
   plt.xlabel('Horizon (days)')
   plt.ylabel('Rank IC')
   plt.title('Factor Decay')
   plt.show()

Turnover Analysis
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from quant_evaluator import compute_turnover

   # Compute turnover between periods
   turnover = compute_turnover(
       positions_t0=weights_t0,
       positions_t1=weights_t1
   )

   # Autocorrelation (proxy for turnover)
   autocorr = compute_autocorrelation(factor, lag=1)
   print(f"1-day autocorr: {autocorr:.4f}")

Performance Attribution
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from quant_evaluator import attribute_performance

   # Attribute returns to factors
   attribution = attribute_performance(
       returns=portfolio_returns,
       factors={
           'momentum': momentum_factor,
           'value': value_factor,
           'quality': quality_factor
       },
       benchmark=benchmark_returns
   )

   print(attribution.summary())

Portfolio Optimization Examples
--------------------------------

Mean-Variance Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from factor_optimizer import MeanVarianceOptimizer

   optimizer = MeanVarianceOptimizer()

   weights = optimizer.optimize(
       expected_returns=factor_scores,
       covariance_matrix=cov_matrix,
       constraints={
           'long_only': True,
           'max_weight': 0.05,
           'min_weight': 0.0,
           'leverage': 1.0
       }
   )

Risk Parity
~~~~~~~~~~~

.. code-block:: python

   from factor_optimizer import RiskParityOptimizer

   optimizer = RiskParityOptimizer()

   weights = optimizer.optimize(
       covariance_matrix=cov_matrix,
       target_risk_contribution='equal'  # or provide custom
   )

Factor Portfolio
~~~~~~~~~~~~~~~~

.. code-block:: python

   from factor_optimizer import FactorPortfolioOptimizer

   optimizer = FactorPortfolioOptimizer()

   # Long-short quintile portfolio
   weights = optimizer.optimize(
       factor=factor_scores,
       method='quantile',
       n_quantiles=5,
       long_quantile=5,   # Top quintile long
       short_quantile=1   # Bottom quintile short
   )

   # Market-neutral
   weights = optimizer.optimize(
       factor=factor_scores,
       method='ranking',
       neutralize={'market': True, 'industry': industry_codes}
   )

Backtesting Examples
--------------------

Simple Backtest
~~~~~~~~~~~~~~~

.. code-block:: python

   from quant_evaluator import Backtester

   backtester = Backtester(
       initial_capital=10_000_000,
       commission=0.001,
       slippage=0.0005
   )

   results = backtester.run(
       signals=factor_scores,
       prices=prices,
       start_date='2020-01-01',
       end_date='2024-01-31',
       rebalance_freq='monthly'
   )

   print(results.summary())

Advanced Backtest with Constraints
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   results = backtester.run(
       signals=factor_scores,
       prices=prices,
       start_date='2020-01-01',
       end_date='2024-01-31',
       rebalance_freq='monthly',
       constraints={
           'max_position': 0.05,
           'max_turnover': 0.5,
           'sector_neutral': True,
           'max_sector_deviation': 0.03
       }
   )

   # Access detailed results
   print(f"Annual Return: {results.annual_return:.2%}")
   print(f"Annual Volatility: {results.annual_volatility:.2%}")
   print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
   print(f"Max Drawdown: {results.max_drawdown:.2%}")
   print(f"Calmar Ratio: {results.calmar_ratio:.2f}")

Transaction Cost Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Detailed cost breakdown
   cost_analysis = backtester.analyze_costs(results)

   print(f"Total Commission: {cost_analysis['total_commission']:.2f}")
   print(f"Total Slippage: {cost_analysis['total_slippage']:.2f}")
   print(f"Cost as % of Returns: {cost_analysis['cost_ratio']:.2%}")

Research Control Examples
--------------------------

Experiment Tracking
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from research_control import ExperimentTracker

   tracker = ExperimentTracker()

   # Create experiment
   exp = tracker.create_experiment(
       name='momentum_strategy_v3',
       description='20-day momentum with vol adjustment',
       tags=['momentum', 'production-candidate']
   )

   # Log parameters
   exp.log_params({
       'lookback': 20,
       'universe': 'hs300',
       'rebalance_freq': 'monthly'
   })

   # Log metrics
   exp.log_metrics({
       'mean_ic': 0.045,
       'ic_ratio': 1.2,
       'sharpe': 1.8,
       'max_drawdown': 0.15
   })

   # Save artifacts
   exp.save_factor(factor, name='momentum_20d')
   exp.save_backtest_results(results)

Reproducibility
~~~~~~~~~~~~~~~

.. code-block:: python

   # Load previous experiment
   exp = tracker.load_experiment('momentum_strategy_v3')

   # Reproduce exact results
   factor = exp.load_factor('momentum_20d')
   params = exp.get_params()

   # Re-run with same parameters
   new_results = reproduce_experiment(exp.id)

Next Steps
----------

* :ref:`api-reference` - Detailed API documentation
* :ref:`contributing` - Contribute to the project
