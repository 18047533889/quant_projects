"""
Example usage of structured logging in DataAccess package.
"""

from pathlib import Path
from dataaccess.logging_config import setup_logging, get_logger, log_performance, PerformanceLogger


# Example 1: Basic setup and logging
def example_basic_logging():
    """Basic logging setup and usage."""
    # Setup logging for the package
    logger = setup_logging(
        level="INFO",
        log_file=Path("logs/dataaccess_example.log"),
        structured=True
    )

    # Simple log messages
    logger.info("Application started")
    logger.debug("This is debug info")
    logger.warning("This is a warning")

    # Log with context
    logger.info(
        "Processing market data",
        extra={'context': {
            'market': 'US',
            'symbols_count': 500,
            'date': '2023-12-31'
        }}
    )


# Example 2: Module-specific logger
def example_module_logger():
    """Using module-specific loggers."""
    # Get a logger for a specific module
    cache_logger = get_logger("cache")
    db_logger = get_logger("database")

    cache_logger.info("Cache hit", extra={'context': {'key': 'market_data_20231231'}})
    db_logger.info("Query executed", extra={'context': {'duration_ms': 145}})


# Example 3: Performance logging with decorator
@log_performance("fetch_market_data")
def fetch_market_data(symbol: str, start_date: str, end_date: str):
    """Fetch market data with automatic performance logging."""
    logger = get_logger("market_data")

    logger.info(
        f"Fetching data for {symbol}",
        extra={'context': {
            'symbol': symbol,
            'start_date': start_date,
            'end_date': end_date
        }}
    )

    # Simulate data fetching
    import time
    time.sleep(0.1)

    return {'symbol': symbol, 'records': 1000}


# Example 4: Performance logging with context manager
def example_context_manager():
    """Using PerformanceLogger as a context manager."""
    logger = get_logger("processor")

    data = {'records': 5000}

    with PerformanceLogger(logger, "data_processing", context={'records': data['records']}):
        # Process data
        import time
        time.sleep(0.05)

        logger.info("Processing batch", extra={'context': {'batch_size': 100}})


# Example 5: Error logging with context
def example_error_logging():
    """Error logging with full context and stack trace."""
    logger = get_logger("error_handler")

    try:
        # Simulate an error
        result = 10 / 0
    except Exception as e:
        logger.error(
            "Division error occurred",
            extra={'context': {
                'operation': 'calculate_ratio',
                'numerator': 10,
                'denominator': 0,
                'error_type': type(e).__name__
            }},
            exc_info=True
        )


# Example 6: Sensitive data sanitization
def example_sanitization():
    """Automatic sanitization of sensitive data."""
    logger = get_logger("auth")

    # These sensitive fields will be automatically redacted
    logger.info(
        "Database connection established",
        extra={'context': {
            'host': 'db.example.com',
            'port': 5432,
            'user': 'admin',
            'password': 'super_secret_password',  # Will be redacted
            'api_key': 'sk-1234567890abcdef',  # Will be redacted
            'database': 'market_data'
        }}
    )


# Example 7: Nested operations with multiple loggers
def example_nested_operations():
    """Complex operation with multiple nested components."""
    main_logger = get_logger("pipeline")

    with PerformanceLogger(main_logger, "full_pipeline", context={'pipeline_id': 'daily_update'}):

        # Step 1: Data extraction
        extract_logger = get_logger("extract")
        with PerformanceLogger(extract_logger, "data_extraction", context={'source': 'api'}):
            extract_logger.info("Extracting data from source")
            import time
            time.sleep(0.02)

        # Step 2: Data transformation
        transform_logger = get_logger("transform")
        with PerformanceLogger(transform_logger, "data_transformation", context={'rows': 1000}):
            transform_logger.info("Transforming data")
            time.sleep(0.03)

        # Step 3: Data loading
        load_logger = get_logger("load")
        with PerformanceLogger(load_logger, "data_loading", context={'destination': 'database'}):
            load_logger.info("Loading data to destination")
            time.sleep(0.02)

        main_logger.info("Pipeline completed successfully")


# Example 8: Different log levels
def example_log_levels():
    """Demonstrating different log levels."""
    logger = get_logger("levels_demo")

    # DEBUG: Detailed diagnostic information
    logger.debug("Cache lookup", extra={'context': {'key': 'user:123', 'hit': True}})

    # INFO: Confirmation that things are working
    logger.info("Request processed", extra={'context': {'request_id': 'req-456'}})

    # WARNING: Something unexpected but not critical
    logger.warning("Slow query detected", extra={'context': {'duration_ms': 5000}})

    # ERROR: Serious problem, function couldn't complete
    logger.error("Failed to connect to external service", extra={'context': {'service': 'api.example.com'}})

    # CRITICAL: Very serious error, system may not continue
    logger.critical("Database connection pool exhausted", extra={'context': {'active_connections': 100}})


# Example 9: Plain text logging for development
def example_plain_text():
    """Using plain text format for easier reading during development."""
    logger = setup_logging(
        level="DEBUG",
        structured=False  # Plain text format
    )

    logger.info("This is a plain text log message")
    logger.debug("Debug information in plain text")


# Example 10: Custom performance decorator with specific logger
def example_custom_decorator():
    """Using performance decorator with custom logger."""
    custom_logger = get_logger("custom")

    @log_performance("complex_calculation", logger=custom_logger)
    def calculate_risk_metrics(portfolio_data):
        """Calculate risk metrics with automatic timing."""
        custom_logger.info(
            "Starting risk calculation",
            extra={'context': {'positions': len(portfolio_data)}}
        )

        import time
        time.sleep(0.05)

        return {'var': 0.05, 'cvar': 0.07, 'sharpe': 1.5}

    portfolio = [{'symbol': 'AAPL', 'quantity': 100}]
    metrics = calculate_risk_metrics(portfolio)


if __name__ == "__main__":
    print("Running logging examples...\n")

    # Ensure logs directory exists
    Path("logs").mkdir(exist_ok=True)

    print("1. Basic logging")
    example_basic_logging()

    print("\n2. Module-specific loggers")
    example_module_logger()

    print("\n3. Performance logging with decorator")
    result = fetch_market_data("AAPL", "2023-01-01", "2023-12-31")

    print("\n4. Context manager")
    example_context_manager()

    print("\n5. Error logging")
    example_error_logging()

    print("\n6. Sensitive data sanitization")
    example_sanitization()

    print("\n7. Nested operations")
    example_nested_operations()

    print("\n8. Log levels")
    example_log_levels()

    print("\n9. Plain text logging")
    example_plain_text()

    print("\n10. Custom decorator")
    example_custom_decorator()

    print("\nExamples completed! Check logs/dataaccess_example.log for output.")
