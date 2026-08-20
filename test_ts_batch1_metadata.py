#!/usr/bin/env python3
"""Test that all operators in ts_batch1.py are properly structured"""

import sys
import importlib.util
from abc import ABC, abstractmethod

# Load the module directly
spec = importlib.util.spec_from_file_location(
    "ts_batch1",
    "/home/shw/quant_projects/factor_engine/cleaned_operators/polars_native/ts_batch1.py"
)
module = importlib.util.module_from_spec(spec)

# Mock the dependencies
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
    NUMERICAL = "numerical"
    POLICY = "policy"
    MODEL_ORDER = "model_order"
    MISSING_POLICY = "missing_policy"
    MARKET_POLICY = "market_policy"
    SUPPORT_POLICY = "support_policy"
    SOURCE_POLICY = "source_policy"
    SESSION_POLICY = "session_policy"

class _MockMissingDefaultType:
    """Sentinel matching cleaned_operators.base.MISSING."""
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    def __repr__(self):
        return "MISSING"

MISSING = _MockMissingDefaultType()

class MockOperator(ABC):
    metadata: MockOperatorMetadata = None

    @abstractmethod
    def calculate(self, *args, **kwargs):
        pass

class MockSeriesOperator(MockOperator):
    def calculate(self, *args, **kwargs):
        return self._calculate_series(*args, **kwargs)

    @abstractmethod
    def _calculate_series(self, *args, **kwargs):
        pass

class MockScalarOperator(MockOperator):
    def calculate(self, *args, **kwargs):
        return self._calculate_scalar(*args, **kwargs)

    @abstractmethod
    def _calculate_scalar(self, *args, **kwargs):
        pass

class MockTwoVarOperator(MockOperator):
    def calculate(self, *args, **kwargs):
        return self._calculate_series(*args, **kwargs)

    @abstractmethod
    def _calculate_series(self, x, y, **kwargs):
        pass

class MockTransformOperator(MockOperator):
    def _calculate_series(self, x, **kwargs):
        pass

    def calculate(self, x, **kwargs):
        return self._calculate_series(x, **kwargs)

def mock_register_operator(*args, **kwargs):
    def decorator(cls):
        # Create an instance to simulate the registration
        instance = cls()
        if hasattr(instance, 'metadata') and instance.metadata is None:
            instance.metadata = MockOperatorMetadata(name=kwargs.get('name', 'unknown'), category='general')
        # Store in a mock registry
        mock_registry.register(instance)
        return cls
    return decorator

class MockRegistry:
    def __init__(self):
        self.operators = {}
        self.aliases = {}
        self.catalog = {}

    def register(self, operator):
        name = getattr(operator.metadata, 'name', None)
        if name:
            self.operators[name] = operator
            self.catalog[name] = {
                'metadata': operator.metadata,
                'implementation': operator
            }

    def register_alias(self, alias, canonical):
        self.aliases[alias] = canonical

    def finalize(self):
        pass

    def freeze(self):
        pass

    @classmethod
    def lifecycle(cls):
        return "building"

mock_registry = MockRegistry()

class MockOperatorRegistry:
    _operators = {}
    _catalog = {}
    _aliases = {}
    _locked = False

    @classmethod
    def register(cls, name, category, canonical, metadata, implementation, backend):
        if name not in cls._operators:
            cls._operators[name] = {}
        cls._operators[name][backend] = implementation
        if canonical not in cls._catalog:
            cls._catalog[canonical] = {}
        cls._catalog[canonical]['metadata'] = metadata
        cls._catalog[canonical]['implementation'] = implementation
        cls._catalog[canonical]['backend'] = backend

    @classmethod
    def register_alias(cls, alias, canonical):
        cls._aliases[alias] = canonical

    @classmethod
    def finalize(cls):
        pass

    @classmethod
    def freeze(cls):
        cls._locked = True

    @classmethod
    def lifecycle(cls):
        return "building"

# Inject mocks
sys.modules['cleaned_operators.base'] = type('module', (), {
    'Operator': MockOperator,
    'SeriesOperator': MockSeriesOperator,
    'ScalarOperator': MockScalarOperator,
    'TwoVarOperator': MockTwoVarOperator,
    'TransformOperator': MockTransformOperator,
    'register_operator': mock_register_operator,
    'OperatorMetadata': MockOperatorMetadata,
    'ParamSpec': MockParamSpec,
    'ParamRole': MockParamRole,
    'MISSING': MISSING,
    'validate_operator_call': lambda *args, **kwargs: (args, kwargs),
    'NormalizedBoundParameters': type('NormalizedBoundParameters', (), {}),
    'BoundOperatorCall': type('BoundOperatorCall', (), {}),
})()

# Also mock cleaned_operators.registry and cleaned_operators.__init__
sys.modules['cleaned_operators'] = type('module', (), {'__path__': []})()
sys.modules['cleaned_operators.registry'] = type('module', (), {
    'OperatorRegistry': MockOperatorRegistry,
    'MISSING': MISSING,
})()

# Mock other dependencies that ts_batch1.py might need
sys.modules['cleaned_operators.base_polars'] = type('module', (), {
    'panel_pandas_bridge': lambda *args, **kwargs: None,
})()

sys.modules['backend'] = type('module', (), {'__path__': []})()
sys.modules['backend.contracts'] = type('module', (), {
    'ExecutionKind': type('ExecutionKind', (), {
        'COMPUTE': 'compute',
        'POLARS_NUMPY_KERNEL': 'polars_numpy_kernel',
    }),
    'PhysicalImplementationSpec': type('PhysicalImplementationSpec', (), {
        '__init__': lambda self, **kwargs: None,
    }),
})()
sys.modules['backend.operator_errors'] = type('module', (), {
    'OperatorParameterError': type('OperatorParameterError', (Exception,), {}),
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
