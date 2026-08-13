"""
Example usage of structured logging in Toolkit package.
"""

from pathlib import Path
from toolkit.logging_config import setup_logging, get_logger, log_performance, PerformanceLogger


# Example 1: Data validation with logging
@log_performance("validate_dataframe")
def validate_dataframe(df_info, schema):
    """Validate DataFrame against schema with logging."""
    validator_logger = get_logger("validator")

    validator_logger.info(
        "Starting data validation",
        extra={'context': {
            'rows': df_info['rows'],
            'columns': df_info['columns'],
            'schema_fields': len(schema)
        }}
    )

    errors = []

    # Check required columns
    for column, rules in schema.items():
        if rules.get('required') and column not in df_info.get('column_names', []):
            errors.append(f"Missing required column: {column}")
            validator_logger.warning(
                f"Missing required column",
                extra={'context': {'column': column}}
            )

    if errors:
        validator_logger.error(
            "Validation failed",
            extra={'context': {
                'error_count': len(errors),
                'errors': errors
            }}
        )
    else:
        validator_logger.info("Validation passed")

    return errors


# Example 2: Data transformation pipeline
def transform_data(data, transformations):
    """Apply transformations with detailed logging."""
    transform_logger = get_logger("transformer")

    transform_logger.info(
        "Starting data transformation",
        extra={'context': {
            'data_size': len(data),
            'transformation_count': len(transformations)
        }}
    )

    with PerformanceLogger(
        transform_logger,
        "data_transformation",
        context={'transformations': len(transformations)}
    ):
        result = data.copy() if hasattr(data, 'copy') else data

        for i, transformation in enumerate(transformations):
            step_logger = get_logger(f"transformer.step_{i}")

            step_logger.debug(
                f"Applying transformation: {transformation['name']}",
                extra={'context': {
                    'step': i,
                    'transformation': transformation['name']
                }}
            )

            try:
                # Simulate transformation
                import time
                time.sleep(0.01)

                step_logger.info(
                    f"Transformation {transformation['name']} completed",
                    extra={'context': {'step': i}}
                )

            except Exception as e:
                step_logger.error(
                    f"Transformation {transformation['name']} failed",
                    extra={'context': {
                        'step': i,
                        'transformation': transformation['name'],
                        'error_type': type(e).__name__
                    }},
                    exc_info=True
                )
                raise

        transform_logger.info("All transformations completed")
        return result


# Example 3: File operations with logging
def read_file_with_logging(file_path):
    """Read file with comprehensive logging."""
    io_logger = get_logger("io")

    io_logger.info(
        "Reading file",
        extra={'context': {'file_path': str(file_path)}}
    )

    try:
        with PerformanceLogger(io_logger, "file_read", context={'file': str(file_path)}):
            # Simulate file read
            import time
            time.sleep(0.02)

            content = "file content here"
            size_bytes = len(content)

            io_logger.info(
                "File read successfully",
                extra={'context': {
                    'file_path': str(file_path),
                    'size_bytes': size_bytes
                }}
            )

            return content

    except FileNotFoundError:
        io_logger.error(
            "File not found",
            extra={'context': {'file_path': str(file_path)}},
            exc_info=True
        )
        raise
    except Exception as e:
        io_logger.error(
            "Failed to read file",
            extra={'context': {
                'file_path': str(file_path),
                'error_type': type(e).__name__
            }},
            exc_info=True
        )
        raise


# Example 4: Configuration management
def load_config(config_path):
    """Load configuration with logging."""
    config_logger = get_logger("config")

    config_logger.info(
        "Loading configuration",
        extra={'context': {'config_path': str(config_path)}}
    )

    try:
        # Simulate config loading
        config = {
            'database': {
                'host': 'localhost',
                'port': 5432,
                'user': 'admin',
                'password': 'secret123'  # Will be sanitized in logs
            },
            'api': {
                'endpoint': 'https://api.example.com',
                'api_key': 'sk-123456'  # Will be sanitized
            }
        }

        config_logger.info(
            "Configuration loaded",
            extra={'context': {
                'config_path': str(config_path),
                'sections': list(config.keys()),
                'config': config  # Sensitive data will be sanitized
            }}
        )

        return config

    except Exception as e:
        config_logger.error(
            "Failed to load configuration",
            extra={'context': {
                'config_path': str(config_path),
                'error_type': type(e).__name__
            }},
            exc_info=True
        )
        raise


# Example 5: Retry logic with logging
def retry_with_logging(operation_name, func, max_attempts=3):
    """Execute function with retry logic and detailed logging."""
    retry_logger = get_logger("retry")

    retry_logger.info(
        f"Starting operation with retry: {operation_name}",
        extra={'context': {
            'operation': operation_name,
            'max_attempts': max_attempts
        }}
    )

    for attempt in range(1, max_attempts + 1):
        try:
            retry_logger.debug(
                f"Attempt {attempt} for {operation_name}",
                extra={'context': {
                    'operation': operation_name,
                    'attempt': attempt,
                    'max_attempts': max_attempts
                }}
            )

            with PerformanceLogger(
                retry_logger,
                f"{operation_name}_attempt_{attempt}",
                context={'attempt': attempt}
            ):
                result = func()

                retry_logger.info(
                    f"Operation {operation_name} succeeded",
                    extra={'context': {
                        'operation': operation_name,
                        'attempt': attempt
                    }}
                )

                return result

        except Exception as e:
            if attempt < max_attempts:
                retry_logger.warning(
                    f"Attempt {attempt} failed, retrying",
                    extra={'context': {
                        'operation': operation_name,
                        'attempt': attempt,
                        'max_attempts': max_attempts,
                        'error_type': type(e).__name__,
                        'error_message': str(e)
                    }}
                )
            else:
                retry_logger.error(
                    f"All attempts failed for {operation_name}",
                    extra={'context': {
                        'operation': operation_name,
                        'total_attempts': max_attempts,
                        'error_type': type(e).__name__
                    }},
                    exc_info=True
                )
                raise


# Example 6: Connection pool management
class ConnectionPool:
    """Connection pool with logging."""

    def __init__(self, max_connections=10):
        self.logger = get_logger("connection_pool")
        self.max_connections = max_connections
        self.active_connections = 0

        self.logger.info(
            "Connection pool initialized",
            extra={'context': {'max_connections': max_connections}}
        )

    def acquire(self):
        """Acquire a connection from the pool."""
        self.logger.debug(
            "Acquiring connection",
            extra={'context': {
                'active': self.active_connections,
                'max': self.max_connections
            }}
        )

        if self.active_connections >= self.max_connections:
            self.logger.warning(
                "Connection pool exhausted",
                extra={'context': {
                    'active': self.active_connections,
                    'max': self.max_connections
                }}
            )
            raise Exception("No connections available")

        self.active_connections += 1

        self.logger.info(
            "Connection acquired",
            extra={'context': {
                'active': self.active_connections,
                'available': self.max_connections - self.active_connections
            }}
        )

        return f"connection_{self.active_connections}"

    def release(self, connection):
        """Release a connection back to the pool."""
        self.logger.debug(
            "Releasing connection",
            extra={'context': {'connection': connection}}
        )

        self.active_connections -= 1

        self.logger.info(
            "Connection released",
            extra={'context': {
                'active': self.active_connections,
                'available': self.max_connections - self.active_connections
            }}
        )


# Example 7: Batch processing with progress logging
def process_batch(items, batch_size=100):
    """Process items in batches with progress logging."""
    batch_logger = get_logger("batch_processor")

    total_items = len(items)
    total_batches = (total_items + batch_size - 1) // batch_size

    batch_logger.info(
        "Starting batch processing",
        extra={'context': {
            'total_items': total_items,
            'batch_size': batch_size,
            'total_batches': total_batches
        }}
    )

    with PerformanceLogger(
        batch_logger,
        "batch_processing",
        context={'total_items': total_items}
    ):
        results = []

        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, total_items)
            batch = items[start_idx:end_idx]

            batch_logger.debug(
                f"Processing batch {batch_num + 1}/{total_batches}",
                extra={'context': {
                    'batch_num': batch_num + 1,
                    'batch_size': len(batch),
                    'progress_pct': round((batch_num + 1) / total_batches * 100, 2)
                }}
            )

            # Simulate processing
            import time
            time.sleep(0.01)

            results.extend(batch)

        batch_logger.info(
            "Batch processing completed",
            extra={'context': {
                'total_items': total_items,
                'processed': len(results)
            }}
        )

        return results


# Example 8: Utility function with conditional logging
def normalize_data(data, method='standard'):
    """Normalize data with conditional logging based on method."""
    norm_logger = get_logger("normalizer")

    norm_logger.info(
        "Normalizing data",
        extra={'context': {
            'method': method,
            'data_size': len(data) if hasattr(data, '__len__') else 0
        }}
    )

    if method not in ['standard', 'minmax', 'robust']:
        norm_logger.error(
            f"Invalid normalization method: {method}",
            extra={'context': {'method': method, 'valid_methods': ['standard', 'minmax', 'robust']}}
        )
        raise ValueError(f"Invalid method: {method}")

    with PerformanceLogger(norm_logger, f"normalize_{method}", context={'method': method}):
        # Simulate normalization
        import time
        time.sleep(0.02)

        norm_logger.debug(f"Applied {method} normalization")

        return data  # Return normalized data


if __name__ == "__main__":
    print("Running Toolkit logging examples...\n")

    # Setup logging
    Path("logs").mkdir(exist_ok=True)
    logger = setup_logging(
        level="DEBUG",
        log_file=Path("logs/toolkit_example.log"),
        structured=True
    )

    print("1. Data validation")
    df_info = {'rows': 1000, 'columns': 5, 'column_names': ['a', 'b', 'c']}
    schema = {'a': {'required': True}, 'b': {'required': True}, 'd': {'required': True}}
    errors = validate_dataframe(df_info, schema)

    print("\n2. Data transformation")
    data = [1, 2, 3, 4, 5]
    transformations = [
        {'name': 'normalize'},
        {'name': 'scale'},
        {'name': 'clip'}
    ]
    result = transform_data(data, transformations)

    print("\n3. File operations")
    read_file_with_logging(Path("example.txt"))

    print("\n4. Configuration loading")
    config = load_config(Path("config.yaml"))

    print("\n5. Retry logic")
    def flaky_operation():
        import random
        if random.random() < 0.5:
            raise Exception("Random failure")
        return "success"

    try:
        retry_with_logging("flaky_api_call", flaky_operation, max_attempts=3)
    except:
        pass

    print("\n6. Connection pool")
    pool = ConnectionPool(max_connections=5)
    conn1 = pool.acquire()
    conn2 = pool.acquire()
    pool.release(conn1)

    print("\n7. Batch processing")
    items = list(range(250))
    processed = process_batch(items, batch_size=100)

    print("\n8. Data normalization")
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    normalized = normalize_data(data, method='standard')

    print("\nExamples completed! Check logs/toolkit_example.log for output.")
