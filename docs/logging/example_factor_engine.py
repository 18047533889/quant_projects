"""
Example usage of structured logging in FactorEngine package.
"""

from pathlib import Path
from factor_engine.logging_config import setup_logging, get_logger, log_performance, PerformanceLogger


# Example 1: Factor computation with performance tracking
@log_performance("compute_momentum_factor")
def compute_momentum(prices, window=20):
    """Compute momentum factor with automatic logging."""
    logger = get_logger("factors.momentum")

    logger.info(
        "Computing momentum factor",
        extra={'context': {
            'window': window,
            'data_points': len(prices)
        }}
    )

    # Simulate computation
    import time
    time.sleep(0.05)

    return prices  # Simplified for example


# Example 2: Batch factor computation
def compute_factor_batch(factor_definitions, universe, date_range):
    """Compute multiple factors with detailed logging."""
    batch_logger = get_logger("batch_processor")

    with PerformanceLogger(
        batch_logger,
        "batch_factor_computation",
        context={
            'factor_count': len(factor_definitions),
            'universe_size': len(universe),
            'date_range': date_range
        }
    ):
        results = {}

        for factor_id, definition in factor_definitions.items():
            factor_logger = get_logger(f"factors.{factor_id}")

            try:
                factor_logger.debug(
                    f"Starting computation for {factor_id}",
                    extra={'context': {'definition': definition}}
                )

                # Simulate computation
                import time
                time.sleep(0.01)

                results[factor_id] = {'status': 'success', 'rows': 1000}

                factor_logger.info(
                    f"Factor {factor_id} computed successfully",
                    extra={'context': {'rows': 1000}}
                )

            except Exception as e:
                factor_logger.error(
                    f"Factor {factor_id} computation failed",
                    extra={'context': {
                        'factor_id': factor_id,
                        'error_type': type(e).__name__
                    }},
                    exc_info=True
                )
                results[factor_id] = {'status': 'failed', 'error': str(e)}

        batch_logger.info(
            "Batch computation completed",
            extra={'context': {
                'total': len(factor_definitions),
                'success': sum(1 for r in results.values() if r['status'] == 'success'),
                'failed': sum(1 for r in results.values() if r['status'] == 'failed')
            }}
        )

        return results


# Example 3: Factor validation pipeline
def validate_factor_output(factor_id, factor_data):
    """Validate factor output with detailed logging."""
    validator_logger = get_logger("validator")

    validation_context = {
        'factor_id': factor_id,
        'rows': len(factor_data) if hasattr(factor_data, '__len__') else 0
    }

    with PerformanceLogger(validator_logger, "factor_validation", context=validation_context):
        issues = []

        # Check for NaN values
        validator_logger.debug("Checking for NaN values")
        # Simulate check
        nan_count = 0
        if nan_count > 0:
            issues.append('nan_values')
            validator_logger.warning(
                f"Factor {factor_id} contains NaN values",
                extra={'context': {'nan_count': nan_count}}
            )

        # Check for infinite values
        validator_logger.debug("Checking for infinite values")
        # Simulate check
        inf_count = 0
        if inf_count > 0:
            issues.append('infinite_values')
            validator_logger.warning(
                f"Factor {factor_id} contains infinite values",
                extra={'context': {'inf_count': inf_count}}
            )

        # Check value range
        validator_logger.debug("Checking value range")
        # Simulate check

        if not issues:
            validator_logger.info(
                f"Factor {factor_id} validation passed",
                extra={'context': validation_context}
            )
        else:
            validator_logger.error(
                f"Factor {factor_id} validation failed",
                extra={'context': {**validation_context, 'issues': issues}}
            )

        return len(issues) == 0


# Example 4: Data loading with error handling
def load_factor_data(factor_id, start_date, end_date):
    """Load factor data with comprehensive error logging."""
    loader_logger = get_logger("data_loader")

    loader_logger.info(
        "Loading factor data",
        extra={'context': {
            'factor_id': factor_id,
            'start_date': start_date,
            'end_date': end_date
        }}
    )

    try:
        with PerformanceLogger(loader_logger, "database_query", context={'factor_id': factor_id}):
            # Simulate data loading
            import time
            time.sleep(0.03)

            data = {'factor_id': factor_id, 'records': 5000}

            loader_logger.info(
                "Data loaded successfully",
                extra={'context': {
                    'factor_id': factor_id,
                    'records': data['records']
                }}
            )

            return data

    except Exception as e:
        loader_logger.error(
            "Failed to load factor data",
            extra={'context': {
                'factor_id': factor_id,
                'start_date': start_date,
                'end_date': end_date,
                'error_type': type(e).__name__
            }},
            exc_info=True
        )
        raise


# Example 5: Operator registration with logging
def register_operator(operator_name, operator_func, metadata):
    """Register a factor operator with logging."""
    registry_logger = get_logger("registry")

    registry_logger.info(
        "Registering operator",
        extra={'context': {
            'operator': operator_name,
            'metadata': metadata
        }}
    )

    try:
        # Simulate registration
        registry = {}
        registry[operator_name] = {
            'func': operator_func,
            'metadata': metadata
        }

        registry_logger.info(
            f"Operator {operator_name} registered successfully",
            extra={'context': {'operator': operator_name}}
        )

        return True

    except Exception as e:
        registry_logger.error(
            f"Failed to register operator {operator_name}",
            extra={'context': {
                'operator': operator_name,
                'error_type': type(e).__name__
            }},
            exc_info=True
        )
        return False


# Example 6: Multi-stage pipeline with nested logging
def execute_factor_pipeline(pipeline_config):
    """Execute a complete factor pipeline with detailed logging."""
    pipeline_logger = get_logger("pipeline")

    pipeline_logger.info(
        "Starting factor pipeline",
        extra={'context': {
            'pipeline_id': pipeline_config.get('id'),
            'stages': len(pipeline_config.get('stages', []))
        }}
    )

    with PerformanceLogger(
        pipeline_logger,
        "full_pipeline_execution",
        context={'pipeline_id': pipeline_config.get('id')}
    ):
        stage_results = []

        for i, stage in enumerate(pipeline_config.get('stages', [])):
            stage_logger = get_logger(f"pipeline.stage_{i}")

            stage_logger.info(
                f"Executing stage {i}: {stage['name']}",
                extra={'context': {
                    'stage_index': i,
                    'stage_name': stage['name']
                }}
            )

            try:
                with PerformanceLogger(
                    stage_logger,
                    f"stage_{stage['name']}",
                    context={'stage_index': i}
                ):
                    # Simulate stage execution
                    import time
                    time.sleep(0.02)

                    stage_results.append({
                        'stage': stage['name'],
                        'status': 'success'
                    })

                    stage_logger.info(
                        f"Stage {stage['name']} completed",
                        extra={'context': {'stage_index': i}}
                    )

            except Exception as e:
                stage_logger.error(
                    f"Stage {stage['name']} failed",
                    extra={'context': {
                        'stage_index': i,
                        'stage_name': stage['name'],
                        'error_type': type(e).__name__
                    }},
                    exc_info=True
                )
                stage_results.append({
                    'stage': stage['name'],
                    'status': 'failed',
                    'error': str(e)
                })

        success_count = sum(1 for r in stage_results if r['status'] == 'success')
        pipeline_logger.info(
            "Pipeline execution completed",
            extra={'context': {
                'total_stages': len(stage_results),
                'successful': success_count,
                'failed': len(stage_results) - success_count
            }}
        )

        return stage_results


# Example 7: Cache operations with logging
def cache_factor_result(factor_id, result, ttl=3600):
    """Cache factor result with logging."""
    cache_logger = get_logger("cache")

    cache_logger.debug(
        "Caching factor result",
        extra={'context': {
            'factor_id': factor_id,
            'result_size': len(str(result)),
            'ttl': ttl
        }}
    )

    try:
        # Simulate cache write
        import time
        time.sleep(0.005)

        cache_logger.info(
            f"Factor {factor_id} cached successfully",
            extra={'context': {
                'factor_id': factor_id,
                'ttl': ttl
            }}
        )
        return True

    except Exception as e:
        cache_logger.error(
            f"Failed to cache factor {factor_id}",
            extra={'context': {
                'factor_id': factor_id,
                'error_type': type(e).__name__
            }},
            exc_info=True
        )
        return False


if __name__ == "__main__":
    print("Running FactorEngine logging examples...\n")

    # Setup logging
    Path("logs").mkdir(exist_ok=True)
    logger = setup_logging(
        level="DEBUG",
        log_file=Path("logs/factor_engine_example.log"),
        structured=True
    )

    print("1. Simple factor computation")
    prices = list(range(100))
    result = compute_momentum(prices, window=20)

    print("\n2. Batch factor computation")
    factors = {
        'momentum_20d': {'window': 20},
        'volatility_30d': {'window': 30},
        'mean_reversion_10d': {'window': 10}
    }
    universe = ['AAPL', 'GOOGL', 'MSFT']
    batch_results = compute_factor_batch(factors, universe, "2023-01-01 to 2023-12-31")

    print("\n3. Factor validation")
    validate_factor_output('momentum_20d', list(range(1000)))

    print("\n4. Data loading")
    data = load_factor_data('momentum_20d', '2023-01-01', '2023-12-31')

    print("\n5. Operator registration")
    register_operator('custom_momentum', compute_momentum, {'version': '1.0'})

    print("\n6. Pipeline execution")
    pipeline_cfg = {
        'id': 'daily_factors',
        'stages': [
            {'name': 'data_load'},
            {'name': 'computation'},
            {'name': 'validation'},
            {'name': 'storage'}
        ]
    }
    pipeline_results = execute_factor_pipeline(pipeline_cfg)

    print("\n7. Cache operations")
    cache_factor_result('momentum_20d', {'data': [1, 2, 3]}, ttl=3600)

    print("\nExamples completed! Check logs/factor_engine_example.log for output.")
