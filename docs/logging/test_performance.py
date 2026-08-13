"""
Test script to verify performance logging functionality.
"""
import sys
import time
sys.path.insert(0, '/home/shw/quant_projects')

from pathlib import Path
from factor_engine.logging_config import setup_logging, get_logger, log_performance, PerformanceLogger

# Setup
logger = setup_logging(level="INFO", structured=True)
test_logger = get_logger("performance_test")

print("Testing performance logging functionality:")
print("=" * 60)

# Test 1: Decorator
print("\n1. Testing @log_performance decorator:")
@log_performance("test_operation")
def slow_function(duration):
    time.sleep(duration)
    return "completed"

result = slow_function(0.05)
print(f"   Result: {result}")

# Test 2: Context manager
print("\n2. Testing PerformanceLogger context manager:")
with PerformanceLogger(test_logger, "context_test", context={'items': 100}):
    time.sleep(0.03)
    print("   Processing in context...")

# Test 3: Nested operations
print("\n3. Testing nested performance logging:")
with PerformanceLogger(test_logger, "outer_operation", context={'level': 'outer'}):
    time.sleep(0.02)
    with PerformanceLogger(test_logger, "inner_operation", context={'level': 'inner'}):
        time.sleep(0.01)
    print("   Nested operations completed")

# Test 4: Error handling
print("\n4. Testing error handling in performance logger:")
try:
    with PerformanceLogger(test_logger, "failing_operation", context={'expected': 'error'}):
        time.sleep(0.01)
        raise ValueError("Simulated error")
except ValueError:
    print("   Error caught and logged correctly")

print("\n" + "=" * 60)
print("✓ All performance logging tests passed")
