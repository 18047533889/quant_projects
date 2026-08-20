"""
Verification script for parallel batch evaluation implementation.
"""

import sys
import numpy as np
from quant_evaluator.contracts.factor_batch import FactorBatch, AxisRef
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.runtime.parallel_executor import (
    ParallelBatchExecutor,
    ParallelConfig,
    ParallelEvaluationResult,
    BatchEvaluationTask,
    BatchEvaluationTaskResult,
    evaluate_batches_parallel,
    register_metric_for_parallel,
)

print("=" * 70)
print("Parallel Batch Evaluation - Implementation Verification")
print("=" * 70)

checks_passed = 0
checks_total = 0

def check(name, condition):
    global checks_passed, checks_total
    checks_total += 1
    if condition:
        print(f"✅ {name}")
        checks_passed += 1
        return True
    else:
        print(f"❌ {name}")
        return False

# Check 1: Imports
check("All imports successful", True)

# Check 2: ParallelConfig creation
config = ParallelConfig(num_workers=2)
check("ParallelConfig creation", config is not None)
check("ParallelConfig worker count", config.num_workers == 2)

# Check 3: ParallelBatchExecutor creation
executor = ParallelBatchExecutor(config=config)
check("ParallelBatchExecutor creation", executor is not None)

# Check 4: Metric registration
def test_metric(factor_batch, **kwargs):
    return float(np.mean(factor_batch.values))

executor.register_metric("test", test_metric)
check("Metric registration", True)

# Check 5: Create test data
T, N, F = 10, 20, 2
batches = []
for i in range(10):
    # Local Generator — no global RNG state mutation.
    rng = np.random.default_rng(i)
    time_axis = AxisRef(name="date", dtype="datetime64", size=T)
    asset_axis = AxisRef(name="ticker", dtype="int64", size=N)
    values = rng.standard_normal((T, N, F))
    batch = FactorBatch(
        factor_ids=("f1", "f2"),
        time_axis=time_axis,
        asset_axis=asset_axis,
        values=values,
    )
    labels = LabelBundle(
        target_id="return",
        values=rng.standard_normal((T, N)),
        horizon=1,
        decision_time=tuple(range(T)),
        label_start_time=tuple(range(T)),
        label_end_time=tuple(range(1, T + 1)),
    )
    batches.append((batch, labels, [{"metric_id": "test", "metric_kind": "custom"}]))

check("Test data creation", len(batches) == 10)

# Check 6: Parallel execution
result = executor.execute_parallel(batches)
check("Parallel execution", result is not None)
check("Result type", isinstance(result, ParallelEvaluationResult))
check("All batches successful", result.successful_batches == 10)
check("No failures", result.failed_batches == 0)
check("Speedup calculated", result.speedup_factor > 0)
check("Throughput calculated", result.throughput_batches_per_second > 0)

# Check 7: Result access
first_result = result.get_result(0)
check("Result access by index", first_result is not None)
check("Result has metrics", first_result.result is not None)
check("Metric value exists", "test" in first_result.result.metrics)

# Check 8: Convenience function
metric_functions = {"test": test_metric}
result2 = evaluate_batches_parallel(batches, metric_functions, num_workers=2)
check("Convenience function", result2 is not None)
check("Convenience function success", result2.successful_batches == 10)

# Check 9: Empty batch handling
result_empty = executor.execute_parallel([])
check("Empty batch handling", result_empty.total_batches == 0)

# Summary
print("\n" + "=" * 70)
print(f"Verification Summary: {checks_passed}/{checks_total} checks passed")
print("=" * 70)

if checks_passed == checks_total:
    print("\n✅ All checks passed - Implementation verified successfully!")
    sys.exit(0)
else:
    print(f"\n❌ {checks_total - checks_passed} check(s) failed")
    sys.exit(1)
