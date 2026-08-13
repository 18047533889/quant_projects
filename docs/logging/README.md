# Structured Logging System

This directory contains the structured logging configuration and examples for all packages in the quantitative trading platform.

## Overview

Structured logging has been added to all four main packages:
- **dataaccess**: Data access and storage operations
- **factor_engine**: Factor computation and analysis
- **toolkit**: Utility functions and helpers
- **quant_evaluator**: Strategy evaluation and backtesting

## Features

Each package includes:

1. **Structured JSON Logging**: Machine-readable logs with full context
2. **Performance Logging**: Automatic function timing with decorators and context managers
3. **Error Context Capture**: Full stack traces and error context
4. **Configurable Log Levels**: DEBUG, INFO, WARNING, ERROR, CRITICAL
5. **Sensitive Data Sanitization**: Automatic redaction of passwords, tokens, API keys

## Files

### Configuration Files
- `/dataaccess/logging_config.py` - DataAccess logging configuration
- `/factor_engine/logging_config.py` - FactorEngine logging configuration
- `/toolkit/logging_config.py` - Toolkit logging configuration
- `/quant_evaluator/logging_config.py` - QuantEvaluator logging configuration

### Documentation
- `logging_guide.md` - Comprehensive guide with examples and best practices

### Example Scripts
- `example_dataaccess.py` - DataAccess logging examples
- `example_factor_engine.py` - FactorEngine logging examples
- `example_toolkit.py` - Toolkit logging examples
- `example_quant_evaluator.py` - QuantEvaluator logging examples

## Quick Start

```python
from pathlib import Path
from dataaccess.logging_config import setup_logging, get_logger, log_performance

# Setup logging
logger = setup_logging(
    level="INFO",
    log_file=Path("logs/myapp.log"),
    structured=True
)

# Get module-specific logger
module_logger = get_logger("my_module")

# Use decorator for performance tracking
@log_performance("my_operation")
def my_function():
    module_logger.info("Processing data", extra={'context': {'records': 1000}})
    # Your code here
```

## Running Examples

```bash
# Set PYTHONPATH to project root
export PYTHONPATH=/home/shw/quant_projects

# Run examples
python3 docs/logging/example_dataaccess.py
python3 docs/logging/example_factor_engine.py
python3 docs/logging/example_toolkit.py
python3 docs/logging/example_quant_evaluator.py
```

## Log Output Format

### JSON (Structured)
```json
{
  "timestamp": "2026-08-14T01:23:45Z",
  "level": "INFO",
  "logger": "dataaccess.data_loader",
  "message": "Data fetch completed",
  "module": "data_loader",
  "function": "fetch_market_data",
  "line": 42,
  "duration_ms": 245.67,
  "context": {
    "symbol": "AAPL",
    "rows": 1000
  }
}
```

### Plain Text (Development)
```
2026-08-14 01:30:45 - dataaccess.data_loader - INFO - Data fetch completed
```

## Key Components

### StructuredFormatter
Custom JSON formatter that:
- Adds timestamps, log levels, and metadata
- Includes context from `extra` parameter
- Captures exception information
- Sanitizes sensitive data

### PerformanceLogger
Context manager and decorator for timing operations:
- Automatically measures execution time
- Logs success/failure with duration
- Can be used as decorator or context manager

### Sanitization
Automatically redacts:
- Passwords
- API keys
- Tokens
- AWS secrets
- Authorization headers

## Best Practices

1. **Use module-specific loggers**: `get_logger("module_name")`
2. **Add context**: Include relevant data in `extra={'context': {...}}`
3. **Track performance**: Use `@log_performance` on critical functions
4. **Log exceptions**: Always use `exc_info=True` when logging errors
5. **Choose appropriate levels**:
   - DEBUG: Detailed diagnostic information
   - INFO: Normal operations
   - WARNING: Unexpected but handled situations
   - ERROR: Serious problems
   - CRITICAL: System may not continue

## Testing

All logging configuration files have been validated:
- Syntax check passed
- Example scripts execute successfully
- Log files created correctly
- Sensitive data sanitization works as expected

## Integration

To integrate into existing code:

```python
# At the top of your module
from <package>.logging_config import get_logger

logger = get_logger(__name__)

# In your functions
def my_function(data):
    logger.info("Starting processing", extra={'context': {'rows': len(data)}})
    try:
        # Your code
        result = process(data)
        logger.info("Processing complete", extra={'context': {'result_size': len(result)}})
        return result
    except Exception as e:
        logger.error("Processing failed", extra={'context': {'error': str(e)}}, exc_info=True)
        raise
```

## Performance Impact

Logging overhead is minimal:
- JSON serialization: ~0.1ms per log entry
- File I/O: Asynchronous, non-blocking
- Sanitization: Only on log creation, cached patterns

For high-frequency operations, use DEBUG level and filter in production.

## Log Analysis

### Command Line
```bash
# Find slow operations
cat logs/app.log | jq 'select(.duration_ms > 1000)'

# Filter by level
cat logs/app.log | jq 'select(.level == "ERROR")'

# Search context
cat logs/app.log | jq 'select(.context.symbol == "AAPL")'
```

### Python
```python
import json

with open('logs/app.log') as f:
    for line in f:
        entry = json.loads(line)
        if entry.get('duration_ms', 0) > 1000:
            print(f"Slow: {entry['message']} - {entry['duration_ms']}ms")
```

## Further Reading

See `logging_guide.md` for comprehensive documentation including:
- Advanced usage patterns
- Environment-specific configuration
- Testing with logging
- Multiple handler setups
- Package-specific examples
