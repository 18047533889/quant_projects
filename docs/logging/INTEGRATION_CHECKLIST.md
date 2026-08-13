# Logging Integration Checklist

Use this checklist when integrating structured logging into existing modules.

## Pre-Integration

- [ ] Review the logging guide: `docs/logging/logging_guide.md`
- [ ] Run example scripts to understand usage patterns
- [ ] Identify critical operations that need performance tracking
- [ ] Determine appropriate log levels for different operations

## Module Setup

### 1. Import Logging Utilities
- [ ] Add import at top of module:
  ```python
  from <package>.logging_config import get_logger
  logger = get_logger(__name__)
  ```

### 2. Configure at Application Startup (main/init modules only)
- [ ] Add setup call:
  ```python
  from <package>.logging_config import setup_logging
  from pathlib import Path
  
  setup_logging(
      level="INFO",
      log_file=Path("logs/<package>.log"),
      structured=True
  )
  ```

## Adding Logging Calls

### 3. Entry Points
- [ ] Log at function entry with context:
  ```python
  def process_data(symbol, start_date):
      logger.info(
          "Starting data processing",
          extra={'context': {
              'symbol': symbol,
              'start_date': start_date
          }}
      )
  ```

### 4. Success Cases
- [ ] Log successful completion with results:
  ```python
  logger.info(
      "Processing completed",
      extra={'context': {'rows_processed': len(result)}}
  )
  ```

### 5. Error Cases
- [ ] Log exceptions with full context:
  ```python
  except Exception as e:
      logger.error(
          "Processing failed",
          extra={'context': {
              'symbol': symbol,
              'error_type': type(e).__name__
          }},
          exc_info=True
      )
      raise
  ```

### 6. Warning Cases
- [ ] Log unexpected but handled situations:
  ```python
  if data_quality < threshold:
      logger.warning(
          "Data quality below threshold",
          extra={'context': {
              'quality': data_quality,
              'threshold': threshold
          }}
      )
  ```

### 7. Debug Information
- [ ] Add debug logs for troubleshooting:
  ```python
  logger.debug(
      f"Cache lookup for {key}",
      extra={'context': {'key': key, 'hit': cache_hit}}
  )
  ```

## Performance Tracking

### 8. Critical Operations
- [ ] Add performance decorator to important functions:
  ```python
  from <package>.logging_config import log_performance
  
  @log_performance("compute_factor")
  def compute_factor(data):
      # Function implementation
  ```

### 9. Long-Running Operations
- [ ] Use context manager for multi-step operations:
  ```python
  from <package>.logging_config import PerformanceLogger
  
  with PerformanceLogger(logger, "batch_processing", context={'batch_size': 100}):
      # Processing logic
  ```

### 10. Nested Operations
- [ ] Track sub-operations separately:
  ```python
  with PerformanceLogger(logger, "full_pipeline"):
      with PerformanceLogger(logger, "data_load"):
          data = load_data()
      with PerformanceLogger(logger, "processing"):
          result = process(data)
  ```

## Testing

### 11. Verify Logging Works
- [ ] Run module and check log output
- [ ] Verify JSON structure is correct
- [ ] Confirm sensitive data is sanitized
- [ ] Check timing information is captured

### 12. Test Error Paths
- [ ] Trigger errors and verify exception logging
- [ ] Confirm stack traces are captured
- [ ] Check error context is complete

### 13. Performance Validation
- [ ] Verify timing accuracy
- [ ] Check nested operation timing
- [ ] Confirm success/failure detection works

## Code Review

### 14. Review Log Levels
- [ ] DEBUG: Only for detailed diagnostics
- [ ] INFO: Normal operations and milestones
- [ ] WARNING: Unexpected but handled situations
- [ ] ERROR: Operation failures
- [ ] CRITICAL: System-level failures

### 15. Review Context
- [ ] All logs include relevant context
- [ ] Context keys are consistent across module
- [ ] No sensitive data in context (will be sanitized anyway)
- [ ] Context is structured (dicts, not formatted strings)

### 16. Review Messages
- [ ] Messages are clear and actionable
- [ ] Messages don't duplicate context (use context dict)
- [ ] Messages are consistent in style
- [ ] Messages include operation names

## Production Readiness

### 17. Log Level Configuration
- [ ] Set appropriate default level (INFO for production)
- [ ] Document how to change level via environment/config
- [ ] Ensure DEBUG logs won't overwhelm in production

### 18. File Rotation
- [ ] Configure log file rotation if needed
- [ ] Set appropriate retention policy
- [ ] Monitor disk space usage

### 19. Monitoring
- [ ] Set up alerts for ERROR/CRITICAL logs
- [ ] Configure log aggregation if using centralized logging
- [ ] Create dashboards for key metrics

### 20. Documentation
- [ ] Document logging strategy in module docstring
- [ ] Update README with logging information
- [ ] Add examples to module documentation

## Checklist for Each Package

### DataAccess Package
- [ ] Log data source connections
- [ ] Log query execution and timing
- [ ] Log cache hits/misses
- [ ] Log data validation results
- [ ] Track read/write operations

### FactorEngine Package
- [ ] Log factor computation start/end
- [ ] Track computation timing
- [ ] Log validation results
- [ ] Track operator registration
- [ ] Log pipeline execution steps

### Toolkit Package
- [ ] Log utility function calls
- [ ] Track data transformations
- [ ] Log configuration loading
- [ ] Track file operations
- [ ] Log retry attempts

### QuantEvaluator Package
- [ ] Log backtest execution
- [ ] Track performance metric calculations
- [ ] Log portfolio optimization steps
- [ ] Track risk analysis operations
- [ ] Log trade executions

## Common Patterns

### Pattern 1: Function with Error Handling
```python
@log_performance("my_operation")
def my_function(param1, param2):
    logger.info(
        "Starting operation",
        extra={'context': {'param1': param1, 'param2': param2}}
    )
    
    try:
        result = do_work(param1, param2)
        
        logger.info(
            "Operation completed",
            extra={'context': {'result_size': len(result)}}
        )
        
        return result
        
    except Exception as e:
        logger.error(
            "Operation failed",
            extra={'context': {
                'param1': param1,
                'param2': param2,
                'error_type': type(e).__name__
            }},
            exc_info=True
        )
        raise
```

### Pattern 2: Multi-Step Pipeline
```python
def process_pipeline(data):
    logger.info(
        "Starting pipeline",
        extra={'context': {'data_size': len(data)}}
    )
    
    with PerformanceLogger(logger, "full_pipeline", context={'stages': 3}):
        # Stage 1
        with PerformanceLogger(logger, "stage_1_extract"):
            extracted = extract(data)
            logger.debug(f"Extracted {len(extracted)} records")
        
        # Stage 2
        with PerformanceLogger(logger, "stage_2_transform"):
            transformed = transform(extracted)
            logger.debug(f"Transformed {len(transformed)} records")
        
        # Stage 3
        with PerformanceLogger(logger, "stage_3_load"):
            load(transformed)
            logger.debug(f"Loaded {len(transformed)} records")
    
    logger.info("Pipeline completed successfully")
    return transformed
```

### Pattern 3: Retry Logic
```python
def operation_with_retry(max_attempts=3):
    for attempt in range(1, max_attempts + 1):
        try:
            logger.debug(
                f"Attempt {attempt}",
                extra={'context': {'attempt': attempt, 'max': max_attempts}}
            )
            
            result = risky_operation()
            
            logger.info(
                "Operation succeeded",
                extra={'context': {'attempt': attempt}}
            )
            
            return result
            
        except Exception as e:
            if attempt < max_attempts:
                logger.warning(
                    f"Attempt {attempt} failed, retrying",
                    extra={'context': {
                        'attempt': attempt,
                        'error_type': type(e).__name__
                    }}
                )
            else:
                logger.error(
                    "All attempts failed",
                    extra={'context': {
                        'total_attempts': max_attempts,
                        'error_type': type(e).__name__
                    }},
                    exc_info=True
                )
                raise
```

## Anti-Patterns to Avoid

### ❌ Don't: Log in tight loops without throttling
```python
for item in huge_list:  # 1M items
    logger.debug(f"Processing {item}")  # 1M log entries!
```

### ✓ Do: Log batch summaries
```python
logger.info(f"Processing {len(huge_list)} items")
for batch in batches(huge_list, 1000):
    process_batch(batch)
logger.info(f"Processed {len(huge_list)} items")
```

### ❌ Don't: Format strings for context
```python
logger.info(f"Processed {symbol} with {rows} rows")
```

### ✓ Do: Use structured context
```python
logger.info(
    "Processing completed",
    extra={'context': {'symbol': symbol, 'rows': rows}}
)
```

### ❌ Don't: Log sensitive data directly
```python
logger.info(f"Connected with password: {password}")
```

### ✓ Do: Trust sanitization or omit sensitive data
```python
logger.info(
    "Database connected",
    extra={'context': {'host': host, 'user': user}}
    # password automatically sanitized if included
)
```

### ❌ Don't: Catch and log without context
```python
except Exception as e:
    logger.error(str(e))
```

### ✓ Do: Include context and stack trace
```python
except Exception as e:
    logger.error(
        "Operation failed",
        extra={'context': {'operation': 'process', 'input': input_data}},
        exc_info=True
    )
```

## Verification Commands

```bash
# Check syntax
python3 -m py_compile <module>.py

# Run with logging
PYTHONPATH=/home/shw/quant_projects python3 <module>.py

# View JSON logs
cat logs/<package>.log | jq '.'

# Find slow operations
cat logs/<package>.log | jq 'select(.duration_ms > 1000)'

# Find errors
cat logs/<package>.log | jq 'select(.level == "ERROR")'

# Check sanitization
cat logs/<package>.log | jq 'select(.context | tostring | contains("REDACTED"))'
```

## Sign-Off

- [ ] All logging implemented
- [ ] All tests passing
- [ ] Code review completed
- [ ] Documentation updated
- [ ] Ready for production

**Reviewer:** ________________  
**Date:** ________________
