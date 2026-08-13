#!/usr/bin/env python3
"""Test that all operators in ts_batch1.py are properly structured"""

import sys
import importlib.util

# Load the module directly
spec = importlib.util.spec_from_file_location(
    "ts_batch1",
    "/home/shw/quant_projects/factor_engine/cleaned_operators/polars_native/ts_batch1.py"
)
module = importlib.util.module_from_spec(spec)

# Mock the dependencies
class MockSeriesOperator:
    pass

class MockOperatorMetadata:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.param_specs = {}

class MockParamSpec:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

class MockParamRole:
    HORIZON = "horizon"
    SCALAR = "scalar"
    ECONOMIC = "economic"
    ESTIMATOR_RESOLUTION = "estimator_resolution"
    STATE_THRESHOLD = "state_threshold"

def mock_register_operator(*args, **kwargs):
    def decorator(cls):
        return cls
    return decorator

# Inject mocks
sys.modules['cleaned_operators.base'] = type('module', (), {
    'SeriesOperator': MockSeriesOperator,
    'register_operator': mock_register_operator,
    'OperatorMetadata': MockOperatorMetadata,
    'ParamSpec': MockParamSpec,
    'ParamRole': MockParamRole,
})()

# Try to load
try:
    spec.loader.exec_module(module)

    # Count operators
    operators = [name for name in dir(module) if name.endswith('PolarsNative')]

    # Verify each has metadata
    operators_with_metadata = []
    operators_with_param_specs = []
    operators_with_method = []

    for op_name in operators:
        op_class = getattr(module, op_name)
        if hasattr(op_class, 'metadata'):
            operators_with_metadata.append(op_name)
            if hasattr(op_class.metadata, 'param_specs'):
                operators_with_param_specs.append(op_name)
        if hasattr(op_class, '_calculate_series'):
            operators_with_method.append(op_name)

    print(f"✓ Total operators: {len(operators)}")
    print(f"✓ Operators with metadata: {len(operators_with_metadata)}")
    print(f"✓ Operators with param_specs: {len(operators_with_param_specs)}")
    print(f"✓ Operators with _calculate_series: {len(operators_with_method)}")

    if len(operators) == len(operators_with_metadata) == len(operators_with_method):
        print("\n✓ SUCCESS: All operators have complete metadata structure!")
        sys.exit(0)
    else:
        print("\n✗ FAILURE: Some operators missing metadata or methods")
        sys.exit(1)

except Exception as e:
    print(f"✗ FAILURE: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
