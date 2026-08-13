"""
Example usage of structured logging in QuantEvaluator package.
"""

from pathlib import Path
from quant_evaluator.logging_config import setup_logging, get_logger, log_performance, PerformanceLogger


# Example 1: Strategy backtest with performance tracking
@log_performance("run_backtest")
def run_backtest(strategy_name, start_date, end_date):
    """Run strategy backtest with logging."""
    backtest_logger = get_logger("backtest")

    backtest_logger.info(
        "Starting backtest",
        extra={'context': {
            'strategy': strategy_name,
            'start_date': start_date,
            'end_date': end_date
        }}
    )

    # Simulate backtest
    import time
    time.sleep(0.1)

    metrics = {
        'total_return': 0.234,
        'sharpe_ratio': 1.45,
        'max_drawdown': -0.156,
        'win_rate': 0.58
    }

    backtest_logger.info(
        "Backtest completed",
        extra={'context': {
            'strategy': strategy_name,
            'metrics': metrics
        }}
    )

    return metrics


# Example 2: Performance metrics calculation
def calculate_performance_metrics(returns, benchmark_returns=None):
    """Calculate performance metrics with detailed logging."""
    metrics_logger = get_logger("metrics")

    metrics_logger.info(
        "Calculating performance metrics",
        extra={'context': {
            'returns_count': len(returns),
            'has_benchmark': benchmark_returns is not None
        }}
    )

    with PerformanceLogger(
        metrics_logger,
        "metrics_calculation",
        context={'data_points': len(returns)}
    ):
        metrics = {}

        # Calculate total return
        metrics_logger.debug("Calculating total return")
        metrics['total_return'] = sum(returns)

        # Calculate Sharpe ratio
        metrics_logger.debug("Calculating Sharpe ratio")
        import statistics
        if len(returns) > 1:
            mean_return = statistics.mean(returns)
            std_return = statistics.stdev(returns)
            metrics['sharpe_ratio'] = mean_return / std_return if std_return > 0 else 0
        else:
            metrics['sharpe_ratio'] = 0

        # Calculate max drawdown
        metrics_logger.debug("Calculating max drawdown")
        metrics['max_drawdown'] = -0.15  # Simplified

        metrics_logger.info(
            "Metrics calculated",
            extra={'context': {'metrics': metrics}}
        )

        return metrics


# Example 3: Portfolio optimization with logging
def optimize_portfolio(assets, constraints):
    """Optimize portfolio with detailed logging."""
    optimizer_logger = get_logger("optimizer")

    optimizer_logger.info(
        "Starting portfolio optimization",
        extra={'context': {
            'asset_count': len(assets),
            'constraints': constraints
        }}
    )

    with PerformanceLogger(
        optimizer_logger,
        "portfolio_optimization",
        context={'assets': len(assets)}
    ):
        try:
            # Simulate optimization iterations
            for iteration in range(1, 6):
                optimizer_logger.debug(
                    f"Optimization iteration {iteration}",
                    extra={'context': {
                        'iteration': iteration,
                        'objective_value': 0.95 + iteration * 0.01
                    }}
                )

                import time
                time.sleep(0.01)

            # Final weights
            weights = {asset: 1.0 / len(assets) for asset in assets}

            optimizer_logger.info(
                "Optimization completed",
                extra={'context': {
                    'asset_count': len(assets),
                    'weights': weights
                }}
            )

            return weights

        except Exception as e:
            optimizer_logger.error(
                "Optimization failed",
                extra={'context': {
                    'asset_count': len(assets),
                    'error_type': type(e).__name__
                }},
                exc_info=True
            )
            raise


# Example 4: Risk analysis with nested logging
def analyze_risk(portfolio, market_data):
    """Analyze portfolio risk with comprehensive logging."""
    risk_logger = get_logger("risk_analysis")

    risk_logger.info(
        "Starting risk analysis",
        extra={'context': {
            'portfolio_size': len(portfolio),
            'data_points': len(market_data)
        }}
    )

    with PerformanceLogger(
        risk_logger,
        "full_risk_analysis",
        context={'portfolio_size': len(portfolio)}
    ):
        risk_metrics = {}

        # Value at Risk (VaR)
        var_logger = get_logger("risk_analysis.var")
        with PerformanceLogger(var_logger, "var_calculation"):
            var_logger.debug("Calculating VaR at 95% confidence")
            risk_metrics['var_95'] = 0.05
            var_logger.info(
                "VaR calculated",
                extra={'context': {'var_95': risk_metrics['var_95']}}
            )

        # Conditional VaR (CVaR)
        cvar_logger = get_logger("risk_analysis.cvar")
        with PerformanceLogger(cvar_logger, "cvar_calculation"):
            cvar_logger.debug("Calculating CVaR")
            risk_metrics['cvar_95'] = 0.07
            cvar_logger.info(
                "CVaR calculated",
                extra={'context': {'cvar_95': risk_metrics['cvar_95']}}
            )

        # Beta calculation
        beta_logger = get_logger("risk_analysis.beta")
        with PerformanceLogger(beta_logger, "beta_calculation"):
            beta_logger.debug("Calculating portfolio beta")
            risk_metrics['beta'] = 1.2
            beta_logger.info(
                "Beta calculated",
                extra={'context': {'beta': risk_metrics['beta']}}
            )

        risk_logger.info(
            "Risk analysis completed",
            extra={'context': {'risk_metrics': risk_metrics}}
        )

        return risk_metrics


# Example 5: Signal generation with logging
def generate_signals(factors, rules):
    """Generate trading signals with logging."""
    signal_logger = get_logger("signals")

    signal_logger.info(
        "Generating trading signals",
        extra={'context': {
            'factor_count': len(factors),
            'rule_count': len(rules)
        }}
    )

    signals = []

    for rule in rules:
        rule_logger = get_logger(f"signals.{rule['name']}")

        rule_logger.debug(
            f"Applying rule: {rule['name']}",
            extra={'context': {'rule': rule['name']}}
        )

        try:
            # Simulate signal generation
            import time
            time.sleep(0.005)

            signal = {
                'rule': rule['name'],
                'signal': 'BUY',
                'strength': 0.75
            }
            signals.append(signal)

            rule_logger.info(
                f"Signal generated by {rule['name']}",
                extra={'context': signal}
            )

        except Exception as e:
            rule_logger.error(
                f"Failed to generate signal for {rule['name']}",
                extra={'context': {
                    'rule': rule['name'],
                    'error_type': type(e).__name__
                }},
                exc_info=True
            )

    signal_logger.info(
        "Signal generation completed",
        extra={'context': {
            'total_signals': len(signals),
            'buy_signals': sum(1 for s in signals if s['signal'] == 'BUY'),
            'sell_signals': sum(1 for s in signals if s['signal'] == 'SELL')
        }}
    )

    return signals


# Example 6: Trade execution with detailed logging
def execute_trade(symbol, quantity, price, order_type='MARKET'):
    """Execute trade with comprehensive logging."""
    execution_logger = get_logger("execution")

    order_id = f"ORD_{hash(symbol + str(quantity))}"

    execution_logger.info(
        "Submitting order",
        extra={'context': {
            'order_id': order_id,
            'symbol': symbol,
            'quantity': quantity,
            'price': price,
            'order_type': order_type
        }}
    )

    try:
        with PerformanceLogger(
            execution_logger,
            "trade_execution",
            context={'order_id': order_id, 'symbol': symbol}
        ):
            # Simulate order submission
            import time
            time.sleep(0.02)

            execution_logger.debug(
                "Order submitted to exchange",
                extra={'context': {'order_id': order_id}}
            )

            # Simulate order fill
            time.sleep(0.01)

            fill_price = price * 1.001  # Slight slippage

            execution_logger.info(
                "Order filled",
                extra={'context': {
                    'order_id': order_id,
                    'symbol': symbol,
                    'quantity': quantity,
                    'fill_price': fill_price,
                    'slippage': fill_price - price
                }}
            )

            return {
                'order_id': order_id,
                'status': 'FILLED',
                'fill_price': fill_price
            }

    except Exception as e:
        execution_logger.error(
            "Trade execution failed",
            extra={'context': {
                'order_id': order_id,
                'symbol': symbol,
                'quantity': quantity,
                'error_type': type(e).__name__
            }},
            exc_info=True
        )
        raise


# Example 7: Strategy evaluation pipeline
def evaluate_strategy_pipeline(strategy_config):
    """Run complete strategy evaluation pipeline."""
    pipeline_logger = get_logger("evaluation_pipeline")

    pipeline_logger.info(
        "Starting strategy evaluation pipeline",
        extra={'context': {
            'strategy': strategy_config['name'],
            'start_date': strategy_config['start_date'],
            'end_date': strategy_config['end_date']
        }}
    )

    with PerformanceLogger(
        pipeline_logger,
        "full_evaluation_pipeline",
        context={'strategy': strategy_config['name']}
    ):
        results = {}

        # Step 1: Run backtest
        pipeline_logger.info("Step 1: Running backtest")
        results['backtest'] = run_backtest(
            strategy_config['name'],
            strategy_config['start_date'],
            strategy_config['end_date']
        )

        # Step 2: Calculate metrics
        pipeline_logger.info("Step 2: Calculating metrics")
        returns = [0.01, -0.005, 0.015, 0.02, -0.01]  # Sample returns
        results['metrics'] = calculate_performance_metrics(returns)

        # Step 3: Risk analysis
        pipeline_logger.info("Step 3: Analyzing risk")
        portfolio = {'AAPL': 0.3, 'GOOGL': 0.3, 'MSFT': 0.4}
        market_data = list(range(100))
        results['risk'] = analyze_risk(portfolio, market_data)

        # Step 4: Generate report
        pipeline_logger.info("Step 4: Generating evaluation report")
        import time
        time.sleep(0.01)

        pipeline_logger.info(
            "Evaluation pipeline completed",
            extra={'context': {
                'strategy': strategy_config['name'],
                'results': results
            }}
        )

        return results


# Example 8: Portfolio rebalancing with logging
def rebalance_portfolio(current_weights, target_weights, threshold=0.05):
    """Rebalance portfolio with detailed logging."""
    rebalance_logger = get_logger("rebalance")

    rebalance_logger.info(
        "Starting portfolio rebalancing",
        extra={'context': {
            'assets': len(current_weights),
            'threshold': threshold
        }}
    )

    trades = []

    for asset in target_weights:
        current = current_weights.get(asset, 0)
        target = target_weights[asset]
        diff = target - current

        if abs(diff) > threshold:
            trade = {
                'asset': asset,
                'current_weight': current,
                'target_weight': target,
                'trade_amount': diff
            }
            trades.append(trade)

            rebalance_logger.info(
                f"Rebalance trade required for {asset}",
                extra={'context': trade}
            )

    if trades:
        rebalance_logger.info(
            "Rebalancing required",
            extra={'context': {
                'trade_count': len(trades),
                'trades': trades
            }}
        )
    else:
        rebalance_logger.info("Portfolio within threshold, no rebalancing needed")

    return trades


if __name__ == "__main__":
    print("Running QuantEvaluator logging examples...\n")

    # Setup logging
    Path("logs").mkdir(exist_ok=True)
    logger = setup_logging(
        level="DEBUG",
        log_file=Path("logs/quant_evaluator_example.log"),
        structured=True
    )

    print("1. Run backtest")
    backtest_results = run_backtest("momentum_strategy", "2023-01-01", "2023-12-31")

    print("\n2. Calculate performance metrics")
    returns = [0.01, -0.005, 0.015, 0.02, -0.01, 0.008, -0.003, 0.012]
    metrics = calculate_performance_metrics(returns)

    print("\n3. Portfolio optimization")
    assets = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
    constraints = {'max_weight': 0.4, 'min_weight': 0.1}
    optimal_weights = optimize_portfolio(assets, constraints)

    print("\n4. Risk analysis")
    portfolio = {'AAPL': 0.3, 'GOOGL': 0.3, 'MSFT': 0.4}
    market_data = list(range(100))
    risk_metrics = analyze_risk(portfolio, market_data)

    print("\n5. Signal generation")
    factors = ['momentum', 'value', 'quality']
    rules = [
        {'name': 'momentum_rule', 'threshold': 0.7},
        {'name': 'value_rule', 'threshold': 0.6}
    ]
    signals = generate_signals(factors, rules)

    print("\n6. Trade execution")
    trade_result = execute_trade('AAPL', 100, 150.50, 'MARKET')

    print("\n7. Full evaluation pipeline")
    strategy_cfg = {
        'name': 'momentum_v1',
        'start_date': '2023-01-01',
        'end_date': '2023-12-31'
    }
    pipeline_results = evaluate_strategy_pipeline(strategy_cfg)

    print("\n8. Portfolio rebalancing")
    current = {'AAPL': 0.25, 'GOOGL': 0.35, 'MSFT': 0.40}
    target = {'AAPL': 0.33, 'GOOGL': 0.33, 'MSFT': 0.34}
    rebalance_trades = rebalance_portfolio(current, target, threshold=0.05)

    print("\nExamples completed! Check logs/quant_evaluator_example.log for output.")
