# Structured Logging Guide

This guide demonstrates how to use the structured logging system across all packages (dataaccess, factor_engine, toolkit, quant_evaluator).

## Features

- **Structured JSON logging** with context and metadata
- **Performance logging** with automatic function timing
- **Error context capture** with full stack traces
- **Configurable log levels** (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- **Sensitive data sanitization** to prevent credential leaks

## Quick Start

### Basic Setup

```python
from pathlib import Path
from dataaccess.logging_config import setup_logging, get_logger

# Configure logging for the package
logger = setup_logging(
    level="INFO",
    log_file=Path("logs/dataaccess.log"),
    structured=True,
    include_context=True
)

# Get a module-specific logger
module_logger = get_logger("data_loader")
```

### Simple Logging

```python
# Basic log message
logger.info("Starting data fetch")

# Log with context
logger.info(
    "Data fetch completed",
    extra={'context': {
        'symbol': 'AAPL',
        'rows': 1000,
        'source': 'market_data'
    }}
)

# Log with different levels
logger.debug("Debug information")
logger.warning("Warning message")
logger.error("Error occurred", exc_info=True)
```

### JSON Output Example

```json
{
  "timestamp": "2026-08-14T01:23:45Z",
  "level": "INFO",
  "logger": "dataaccess.data_loader",
  "message": "Data fetch completed",
  "module": "data_loader",
  "function": "fetch_market_data",
  "line": 42,
  "context": {
    "symbol": "AAPL",
    "rows": 1000,
    "source": "market_data"
  }
}
```

## Performance Logging

### Using the Decorator

```python
from factor_engine.logging_config import log_performance, get_logger

logger = get_logger("factors")

@log_performance("compute_momentum")
def calculate_momentum(data, window=20):
    """Calculate momentum factor."""
    return data.pct_change(window)

# Automatically logs execution time
result = calculate_momentum(df, window=30)
```

### Using Context Manager

```python
from toolkit.logging_config import PerformanceLogger, get_logger

logger = get_logger("utils")

def process_data(df):
    with PerformanceLogger(logger, "data_normalization", context={'rows': len(df)}):
        # Your code here
        normalized = (df - df.mean()) / df.std()
        return normalized
```

### Performance Log Output

```json
{
  "timestamp": "2026-08-14T01:25:30Z",
  "level": "INFO",
  "logger": "factor_engine.factors",
  "message": "Function executed: compute_momentum",
  "module": "factors",
  "function": "calculate_momentum",
  "line": 15,
  "duration_ms": 245.67,
  "context": {
    "operation": "compute_momentum",
    "success": true
  }
}
```

## Error Logging with Context

```python
from quant_evaluator.logging_config import get_logger

logger = get_logger("backtest")

try:
    result = run_backtest(strategy)
except Exception as e:
    logger.error(
        "Backtest failed",
        extra={'context': {
            'strategy': strategy.name,
            'date_range': '2020-01-01 to 2023-12-31',
            'error_type': type(e).__name__
        }},
        exc_info=True
    )
    raise
```

### Error Log Output

```json
{
  "timestamp": "2026-08-14T01:27:15Z",
  "level": "ERROR",
  "logger": "quant_evaluator.backtest",
  "message": "Backtest failed",
  "module": "backtest",
  "function": "run_backtest",
  "line": 89,
  "context": {
    "strategy": "momentum_v1",
    "date_range": "2020-01-01 to 2023-12-31",
    "error_type": "ValueError"
  },
  "exception": {
    "type": "ValueError",
    "message": "Invalid date range",
    "traceback": ["Traceback (most recent call last):", "..."]
  }
}
```

## Sensitive Data Sanitization

The logging system automatically redacts sensitive information:

```python
from dataaccess.logging_config import get_logger

logger = get_logger("auth")

# This will be sanitized automatically
logger.info("Connecting to database", extra={'context': {
    'host': 'db.example.com',
    'user': 'admin',
    'password': 'secret123',  # Will be redacted
    'api_key': 'sk-1234567890'  # Will be redacted
}})
```

### Sanitized Output

```json
{
  "timestamp": "2026-08-14T01:28:00Z",
  "level": "INFO",
  "logger": "dataaccess.auth",
  "message": "Connecting to database",
  "context": {
    "host": "db.example.com",
    "user": "admin",
    "password": "***REDACTED***",
    "api_key": "***REDACTED***"
  }
}
```

## Configuration Options

### Log Levels

```python
# DEBUG: Detailed diagnostic information
logger = setup_logging(level="DEBUG")

# INFO: General informational messages
logger = setup_logging(level="INFO")

# WARNING: Warning messages for potentially problematic situations
logger = setup_logging(level="WARNING")

# ERROR: Error messages for serious problems
logger = setup_logging(level="ERROR")

# CRITICAL: Critical messages for very serious errors
logger = setup_logging(level="CRITICAL")
```

### Plain Text Format

For development or debugging, you can use plain text format:

```python
logger = setup_logging(
    level="DEBUG",
    structured=False  # Use plain text instead of JSON
)
```

Plain text output:
```
2026-08-14 01:30:45 - dataaccess.data_loader - INFO - Data fetch completed
```

### Multiple Handlers

```python
from pathlib import Path
from factor_engine.logging_config import setup_logging

# Log to both console and file
logger = setup_logging(
    level="INFO",
    log_file=Path("logs/factor_engine.log"),
    structured=True
)

# Console shows INFO and above
# File captures everything
```

## Package-Specific Examples

### DataAccess Package

```python
from pathlib import Path
from dataaccess.logging_config import setup_logging, get_logger, log_performance

# Setup
logger = setup_logging(level="INFO", log_file=Path("logs/dataaccess.log"))
module_logger = get_logger("market_data")

@log_performance("fetch_prices")
def fetch_stock_prices(symbols, start_date, end_date):
    module_logger.info(
        "Fetching stock prices",
        extra={'context': {
            'symbols': symbols,
            'date_range': f"{start_date} to {end_date}"
        }}
    )
    # Fetch logic here
    return prices

prices = fetch_stock_prices(['AAPL', 'GOOGL'], '2020-01-01', '2023-12-31')
```

### FactorEngine Package

```python
from factor_engine.logging_config import setup_logging, get_logger, PerformanceLogger

logger = setup_logging(level="DEBUG")
factor_logger = get_logger("computation")

def compute_factor_batch(factor_ids, universe, date_range):
    with PerformanceLogger(factor_logger, "batch_computation", 
                          context={'factor_count': len(factor_ids)}):
        results = {}
        for factor_id in factor_ids:
            factor_logger.debug(
                f"Computing factor {factor_id}",
                extra={'context': {'factor_id': factor_id}}
            )
            results[factor_id] = compute_single_factor(factor_id, universe, date_range)
        return results
```

### Toolkit Package

```python
from toolkit.logging_config import setup_logging, log_performance

logger = setup_logging(level="INFO")

@log_performance("data_validation")
def validate_dataframe(df, schema):
    """Validate DataFrame against schema."""
    errors = []
    for column, rules in schema.items():
        # validation logic
        pass
    return errors

errors = validate_dataframe(data, validation_schema)
```

### QuantEvaluator Package

```python
from quant_evaluator.logging_config import setup_logging, get_logger, PerformanceLogger

logger = setup_logging(level="INFO", log_file=Path("logs/evaluator.log"))
eval_logger = get_logger("metrics")

def evaluate_strategy(strategy, market_data):
    with PerformanceLogger(eval_logger, "strategy_evaluation",
                          context={'strategy_name': strategy.name}):
        
        eval_logger.info("Starting evaluation", extra={'context': {
            'strategy': strategy.name,
            'data_points': len(market_data)
        }})
        
        metrics = calculate_metrics(strategy, market_data)
        
        eval_logger.info("Evaluation complete", extra={'context': {
            'sharpe_ratio': metrics.sharpe,
            'max_drawdown': metrics.max_dd
        }})
        
        return metrics
```

## Best Practices

1. **Use module-specific loggers**: Always use `get_logger(module_name)` instead of the root logger
2. **Add context**: Include relevant context in the `extra` parameter
3. **Performance-critical paths**: Use `@log_performance` for functions you want to monitor
4. **Error handling**: Always use `exc_info=True` when logging exceptions
5. **Avoid over-logging**: Use DEBUG for verbose diagnostic info, INFO for normal operations
6. **Structured data**: Pass dictionaries in context rather than formatting strings
7. **Log levels**:
   - DEBUG: Detailed diagnostic information
   - INFO: Confirmation that things are working as expected
   - WARNING: Something unexpected happened but system continues
   - ERROR: Due to a more serious problem, function failed
   - CRITICAL: System might not be able to continue

## Log Analysis

### Parsing JSON Logs

```python
import json

# Read and analyze logs
with open('logs/dataaccess.log') as f:
    for line in f:
        log_entry = json.loads(line)
        if log_entry.get('duration_ms', 0) > 1000:
            print(f"Slow operation: {log_entry['context']['operation']} "
                  f"took {log_entry['duration_ms']}ms")
```

### Filtering by Context

```bash
# Find all logs for a specific symbol
cat logs/dataaccess.log | jq 'select(.context.symbol == "AAPL")'

# Find slow operations (> 500ms)
cat logs/factor_engine.log | jq 'select(.duration_ms > 500)'

# Find all errors
cat logs/quant_evaluator.log | jq 'select(.level == "ERROR")'
```

## Testing with Logging

```python
import logging
from io import StringIO

def test_function_with_logging(caplog):
    """Test that appropriate logs are generated."""
    with caplog.at_level(logging.INFO):
        result = my_function()
        
    # Check log messages
    assert "Operation completed" in caplog.text
    
    # Check log records
    records = [r for r in caplog.records if r.levelname == "INFO"]
    assert len(records) == 1
    assert records[0].context['operation'] == 'my_operation'
```

## Environment-Specific Configuration

```python
import os
from pathlib import Path
from dataaccess.logging_config import setup_logging

# Configure based on environment
env = os.getenv('ENVIRONMENT', 'development')

if env == 'production':
    logger = setup_logging(
        level="WARNING",
        log_file=Path("/var/log/dataaccess/app.log"),
        structured=True
    )
elif env == 'development':
    logger = setup_logging(
        level="DEBUG",
        structured=False  # Plain text for easier reading
    )
else:  # testing
    logger = setup_logging(
        level="ERROR",
        structured=True
    )
```
