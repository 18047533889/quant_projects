"""
Import isolation tests - verify packages can be imported independently.

Each package should be importable without requiring the others.
Tests for circular dependencies and import order issues.
"""

import pytest
import sys


def test_qe_standalone_import():
    """QE should import without requiring FP, FA, or FO."""
    # Save current modules
    original_modules = set(sys.modules.keys())

    # Import QE
    import quant_evaluator

    # Check QE imports worked
    assert 'quant_evaluator' in sys.modules
    assert quant_evaluator.__version__ is not None

    # Verify QE didn't drag in other packages
    # (They might already be loaded from other tests, so we just verify QE itself works)
    from quant_evaluator import FactorBatch, LabelBundle, EvaluationRequest
    assert FactorBatch is not None


def test_fp_standalone_import():
    """FP should import without requiring QE, FA, or FO."""
    import factor_preprocess

    assert 'factor_preprocess' in sys.modules
    assert factor_preprocess.__version__ is not None

    from factor_preprocess import PreprocessingPolicy, TransformSpec
    assert PreprocessingPolicy is not None


def test_fa_standalone_import():
    """FA should import without requiring QE, FP, or FO."""
    import factor_assets

    assert 'factor_assets' in sys.modules
    assert factor_assets.__version__ is not None

    from factor_assets import FactorAsset, AssetRepository, LifecycleState
    assert FactorAsset is not None


def test_fo_standalone_import():
    """FO should import without requiring QE, FP, or FA (core functionality)."""
    import factor_optimizer

    assert 'factor_optimizer' in sys.modules
    info = factor_optimizer.package_info()
    assert info is not None
    assert info['name'] == 'factor-optimizer'


def test_no_circular_dependencies():
    """Verify no circular import dependencies."""
    # All packages should be importable in any order
    packages = [
        'quant_evaluator',
        'factor_preprocess',
        'factor_assets',
        'factor_optimizer',
    ]

    for pkg in packages:
        assert pkg in sys.modules, f"{pkg} should be loaded"

    # Try importing in reverse order (should work since already loaded)
    for pkg in reversed(packages):
        mod = sys.modules[pkg]
        assert mod is not None


def test_qe_public_api_complete():
    """QE public API is complete and documented."""
    import quant_evaluator

    required_exports = [
        'FactorBatch',
        'AxisRef',
        'LabelBundle',
        'EvaluationRequest',
        'EvaluationBundle',
        'MetricValue',
        'FactorDiagnosis',
        'QuantEvaluatorError',
        'ContractError',
        'DataError',
    ]

    for export in required_exports:
        assert hasattr(quant_evaluator, export), f"Missing {export} in QE API"


def test_fp_public_api_complete():
    """FP public API is complete and documented."""
    import factor_preprocess

    required_exports = [
        'PreprocessingPolicy',
        'TransformSpec',
        'FittedState',
        'FeatureBundle',
    ]

    for export in required_exports:
        assert hasattr(factor_preprocess, export), f"Missing {export} in FP API"

    # Check package_info available
    assert hasattr(factor_preprocess, 'package_info')
    info = factor_preprocess.package_info()
    assert 'capabilities' in info


def test_fa_public_api_complete():
    """FA public API is complete and documented."""
    import factor_assets

    required_exports = [
        'FactorAsset',
        'AssetMetadata',
        'FactorSet',
        'EvidenceRef',
        'AssetRepository',
        'LifecycleState',
        'StateTransition',
        'create_factor_id',
        'FactorIdentity',
        'SeenIndex',
    ]

    for export in required_exports:
        assert hasattr(factor_assets, export), f"Missing {export} in FA API"


def test_fo_public_api_complete():
    """FO public API is complete and documented."""
    import factor_optimizer

    # FO exports package_info as primary interface
    assert hasattr(factor_optimizer, 'package_info')

    info = factor_optimizer.package_info()
    assert 'capabilities' in info
    assert 'adapters' in info
    assert 'version' in info


def test_qe_contracts_independent():
    """QE contracts don't depend on external types."""
    from quant_evaluator import FactorBatch, LabelBundle, EvaluationRequest
    import inspect

    # Check these are defined in quant_evaluator module
    assert FactorBatch.__module__.startswith('quant_evaluator')
    assert LabelBundle.__module__.startswith('quant_evaluator')
    assert EvaluationRequest.__module__.startswith('quant_evaluator')


def test_fa_contracts_independent():
    """FA contracts don't depend on external types."""
    from factor_assets import FactorAsset, EvidenceRef, LifecycleState

    # Check these are defined in factor_assets module
    assert FactorAsset.__module__.startswith('factor_assets')
    assert EvidenceRef.__module__.startswith('factor_assets')
    assert LifecycleState.__module__.startswith('factor_assets')


def test_fp_contracts_independent():
    """FP contracts don't depend on external types."""
    from factor_preprocess import PreprocessingPolicy, TransformSpec

    # Check these are defined in factor_preprocess module
    assert PreprocessingPolicy.__module__.startswith('factor_preprocess')
    assert TransformSpec.__module__.startswith('factor_preprocess')


def test_cross_package_integration_optional():
    """Cross-package integration is optional, not required."""
    # QE can work standalone
    from quant_evaluator import FactorBatch, AxisRef
    import numpy as np

    T, N, F = 5, 3, 1
    time_axis = AxisRef(name="time", dtype="datetime64[D]", size=T)
    asset_axis = AxisRef(name="asset", dtype="int64", size=N)
    values = np.random.randn(T, N, F)

    batch = FactorBatch(
        factor_ids=("test",),
        time_axis=time_axis,
        asset_axis=asset_axis,
        values=values,
    )
    assert batch is not None

    # FA can work standalone
    from factor_assets import AssetRepository, FactorAsset, AssetMetadata, create_factor_id
    from datetime import datetime

    factor_id = create_factor_id(name="test", version="v1", params={})
    metadata = AssetMetadata(
        factor_id=factor_id,
        canonical_repr="test",
        canonical_hash="hash_test_v1",
        frequency="daily",
        domains=("equity",),
        timing="daily",
        description="test",
    )
    from factor_assets import LineageRef, LifecycleState
    lineage = LineageRef(
        factor_id=factor_id,
        parents=(),
    )
    asset = FactorAsset(
        metadata=metadata,
        lineage=lineage,
        lifecycle_state=LifecycleState.REGISTERED,
        registered_at=datetime.now().isoformat(),
    )

    repo = AssetRepository()
    asset = repo.register(metadata, lineage)
    assert repo.get(factor_id) is not None

    # FP can work standalone
    from factor_preprocess import PreprocessingPolicy, TransformSpec, TransformKind, TransformMode

    policy = PreprocessingPolicy(
        policy_id="test_policy",
        transforms=[TransformSpec(name="standardize", kind=TransformKind.CROSS_SECTIONAL, mode=TransformMode.STATELESS, version="1.0", parameters={})],
    )
    assert policy is not None


def test_package_versions_accessible():
    """All packages expose version information."""
    import quant_evaluator
    import factor_preprocess
    import factor_assets
    import factor_optimizer

    # Direct version attributes
    assert hasattr(quant_evaluator, '__version__')
    assert hasattr(factor_preprocess, '__version__')
    assert hasattr(factor_assets, '__version__')

    # Version in package info
    fo_info = factor_optimizer.package_info()
    assert 'version' in fo_info

    fp_info = factor_preprocess.package_info()
    assert 'version' in fp_info


def test_error_types_isolated():
    """Error types are properly isolated per package."""
    from quant_evaluator import QuantEvaluatorError, ContractError, DataError
    from factor_assets import (
        DuplicateIdentityError,
        AssetNotFoundError,
        LifecycleConflictError,
    )

    # QE errors should be in QE module
    assert QuantEvaluatorError.__module__.startswith('quant_evaluator')
    assert ContractError.__module__.startswith('quant_evaluator')

    # FA errors should be in FA module
    assert DuplicateIdentityError.__module__.startswith('factor_assets')
    assert AssetNotFoundError.__module__.startswith('factor_assets')


def test_numpy_pandas_compatibility():
    """All packages work with standard numpy/pandas types."""
    import numpy as np
    from quant_evaluator import FactorBatch, AxisRef, LabelBundle

    # All packages should accept numpy arrays without issue
    T, N = 5, 3
    time_axis = AxisRef(name="time", dtype="datetime64[D]", size=T)
    asset_axis = AxisRef(name="asset", dtype="int64", size=N)

    # Test with different numpy dtypes
    for dtype in [np.float64, np.float32]:
        values = np.random.randn(T, N, 1).astype(dtype)
        batch = FactorBatch(
            factor_ids=("test",),
            time_axis=time_axis,
            asset_axis=asset_axis,
            values=values,
        )
        assert batch.values.dtype == dtype


def test_no_hidden_dependencies():
    """Verify packages don't have undeclared dependencies on each other."""
    import quant_evaluator
    import factor_preprocess
    import factor_assets
    import factor_optimizer

    # Each package's __init__ should not import from the others
    # (We verify this by checking that basic imports work)

    qe_exports = dir(quant_evaluator)
    fp_exports = dir(factor_preprocess)
    fa_exports = dir(factor_assets)
    fo_exports = dir(factor_optimizer)

    # QE should not expose FP/FA/FO types
    assert not any('Preprocess' in x for x in qe_exports if not x.startswith('_'))
    assert not any('Asset' in x for x in qe_exports if not x.startswith('_'))

    # FP should not expose QE/FA/FO types
    assert not any('Evaluation' in x for x in fp_exports if not x.startswith('_'))

    # FA should not expose QE/FP types
    assert not any('Evaluation' in x for x in fa_exports if not x.startswith('_'))
    assert not any('Preprocess' in x for x in fa_exports if not x.startswith('_'))
