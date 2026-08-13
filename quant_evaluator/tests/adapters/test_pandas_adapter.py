"""
Tests for pandas adapter (reference/debug only).

Tests verify:
- Lazy import and OptionalDependencyMissing
- DataFrame <-> FactorBatch conversions
- DataFrame <-> LabelBundle conversions
- Reference/debug only semantics (not production fallback)
"""

import pytest
import numpy as np

from quant_evaluator.contracts.errors import OptionalDependencyMissing
from quant_evaluator.contracts.factor_batch import FactorBatch, AxisRef
from quant_evaluator.contracts.label_bundle import LabelBundle


def test_pandas_adapter_import_fails_without_pandas():
    """Test that PandasAdapter raises OptionalDependencyMissing when pandas not installed."""
    # This test assumes pandas is installed (common dependency)
    # If pandas is not installed, this will pass
    try:
        import pandas  # noqa: F401
        pytest.skip("pandas is installed, skipping missing dependency test")
    except ImportError:
        pass

    from quant_evaluator.adapters.pandas import PandasAdapter

    with pytest.raises(OptionalDependencyMissing) as exc_info:
        PandasAdapter()

    assert "Pandas is not installed" in str(exc_info.value)


def test_pandas_adapter_not_imported_by_core():
    """Test that core modules do not import pandas adapter."""
    import sys

    # Import core modules
    import quant_evaluator
    import quant_evaluator.contracts
    import quant_evaluator.api

    # Verify adapter module is not loaded
    adapter_modules = [
        name for name in sys.modules.keys()
        if "quant_evaluator.adapters.pandas" in name
    ]

    assert len(adapter_modules) == 0, \
        f"Core import should not load pandas adapter, but found: {adapter_modules}"


@pytest.fixture
def pandas_adapter():
    """Fixture providing PandasAdapter if pandas is available."""
    try:
        import pandas  # noqa: F401
    except ImportError:
        pytest.skip("pandas not installed")

    from quant_evaluator.adapters.pandas import PandasAdapter
    return PandasAdapter()


@pytest.fixture
def sample_dataframe(pandas_adapter):
    """Fixture providing sample DataFrame for testing."""
    pd = pandas_adapter._pd

    data = {
        "date": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
        "code": ["000001", "000002", "000001", "000002"],
        "momentum": [0.05, 0.03, 0.06, 0.04],
        "value": [-0.02, 0.01, -0.01, 0.02],
    }
    return pd.DataFrame(data)


def test_dataframe_to_factor_batch_basic(pandas_adapter, sample_dataframe):
    """Test basic DataFrame to FactorBatch conversion."""
    batch = pandas_adapter.dataframe_to_factor_batch(
        sample_dataframe,
        factor_cols=["momentum", "value"],
        time_col="date",
        asset_col="code",
    )

    # Verify structure
    assert isinstance(batch, FactorBatch)
    assert batch.num_factors == 2
    assert batch.num_times == 2
    assert batch.num_assets == 2
    assert batch.factor_ids == ("momentum", "value")

    # Verify axes
    assert batch.time_axis.size == 2
    assert batch.asset_axis.size == 2
    assert batch.time_axis.values is not None
    assert batch.asset_axis.values is not None

    # Verify values shape
    assert batch.values.shape == (2, 2, 2)

    # Verify specific values
    # date=2024-01-01, code=000001, momentum=0.05
    assert batch.values[0, 0, 0] == pytest.approx(0.05)
    # date=2024-01-01, code=000001, value=-0.02
    assert batch.values[0, 0, 1] == pytest.approx(-0.02)


def test_dataframe_to_factor_batch_missing_columns(pandas_adapter, sample_dataframe):
    """Test that missing columns raise ValueError."""
    with pytest.raises(ValueError, match="Missing required columns"):
        pandas_adapter.dataframe_to_factor_batch(
            sample_dataframe,
            factor_cols=["momentum", "nonexistent"],
            time_col="date",
            asset_col="code",
        )


def test_dataframe_to_factor_batch_duplicates(pandas_adapter):
    """Test that duplicate (time, asset) pairs raise ValueError."""
    pd = pandas_adapter._pd

    data = {
        "date": ["2024-01-01", "2024-01-01"],  # Duplicate
        "code": ["000001", "000001"],  # Duplicate
        "momentum": [0.05, 0.06],
    }
    df = pd.DataFrame(data)

    with pytest.raises(ValueError, match="Duplicate"):
        pandas_adapter.dataframe_to_factor_batch(
            df,
            factor_cols=["momentum"],
            time_col="date",
            asset_col="code",
        )


def test_factor_batch_to_dataframe_basic(pandas_adapter):
    """Test basic FactorBatch to DataFrame conversion."""
    time_axis = AxisRef(
        name="time",
        dtype="object",
        size=2,
        values=np.array(["2024-01-01", "2024-01-02"]),
    )
    asset_axis = AxisRef(
        name="asset",
        dtype="object",
        size=2,
        values=np.array(["000001", "000002"]),
    )
    values = np.array([
        [[0.05, -0.02], [0.03, 0.01]],
        [[0.06, -0.01], [0.04, 0.02]],
    ])

    batch = FactorBatch(
        factor_ids=("momentum", "value"),
        time_axis=time_axis,
        asset_axis=asset_axis,
        values=values,
        layout="wide",
        dtype="float64",
    )

    df = pandas_adapter.factor_batch_to_dataframe(batch, time_col="date", asset_col="code")

    # Verify structure
    assert len(df) == 4  # 2 times × 2 assets
    assert "date" in df.columns
    assert "code" in df.columns
    assert "momentum" in df.columns
    assert "value" in df.columns

    # Verify specific values
    row = df[(df["date"] == "2024-01-01") & (df["code"] == "000001")].iloc[0]
    assert row["momentum"] == pytest.approx(0.05)
    assert row["value"] == pytest.approx(-0.02)


def test_dataframe_to_label_bundle_basic(pandas_adapter):
    """Test basic DataFrame to LabelBundle conversion."""
    pd = pandas_adapter._pd

    data = {
        "decision_time": ["2024-01-01", "2024-01-02"],
        "label_start": ["2024-01-02", "2024-01-03"],
        "label_end": ["2024-01-06", "2024-01-07"],
        "label_value": [0.02, 0.03],
    }
    df = pd.DataFrame(data)

    bundle = pandas_adapter.dataframe_to_label_bundle(
        df,
        target_id="ret_5d_vwap",
        horizon=5,
        value_col="label_value",
        decision_time_col="decision_time",
        label_start_col="label_start",
        label_end_col="label_end",
    )

    # Verify structure
    assert isinstance(bundle, LabelBundle)
    assert bundle.target_id == "ret_5d_vwap"
    assert bundle.horizon == 5
    assert bundle.execution_delay == 0
    assert len(bundle.values) == 2

    # Verify timing
    assert len(bundle.decision_time) == 2
    assert len(bundle.label_start_time) == 2
    assert len(bundle.label_end_time) == 2

    # Verify values
    assert bundle.values[0] == pytest.approx(0.02)
    assert bundle.values[1] == pytest.approx(0.03)


def test_dataframe_to_label_bundle_missing_columns(pandas_adapter):
    """Test that missing timing columns raise ValueError."""
    pd = pandas_adapter._pd

    data = {
        "decision_time": ["2024-01-01"],
        "label_value": [0.02],
        # Missing label_start and label_end
    }
    df = pd.DataFrame(data)

    with pytest.raises(ValueError, match="Missing required columns"):
        pandas_adapter.dataframe_to_label_bundle(
            df,
            target_id="ret_5d_vwap",
            horizon=5,
        )


def test_label_bundle_to_dataframe_basic(pandas_adapter):
    """Test basic LabelBundle to DataFrame conversion."""
    bundle = LabelBundle(
        target_id="ret_5d_vwap",
        values=np.array([0.02, 0.03]),
        horizon=5,
        execution_delay=0,
        decision_time=("2024-01-01", "2024-01-02"),
        execution_time=("2024-01-01", "2024-01-02"),
        label_start_time=("2024-01-02", "2024-01-03"),
        label_end_time=("2024-01-06", "2024-01-07"),
    )

    df = pandas_adapter.label_bundle_to_dataframe(bundle, include_all_timing=True)

    # Verify structure
    assert len(df) == 2
    assert "decision_time" in df.columns
    assert "label_value" in df.columns
    assert "execution_time" in df.columns
    assert "label_start_time" in df.columns
    assert "label_end_time" in df.columns
    assert "horizon" in df.columns

    # Verify values
    assert df["label_value"].iloc[0] == pytest.approx(0.02)
    assert df["horizon"].iloc[0] == 5


def test_pandas_adapter_is_reference_debug_only():
    """Test that pandas adapter is documented as reference/debug only."""
    from quant_evaluator.adapters import pandas as pandas_module
    import inspect

    source = inspect.getsource(pandas_module)

    # Verify documentation emphasizes reference/debug only
    assert "reference/debug" in source.lower() or "reference/debug only" in source.lower()
    assert "NOT" in source and "production" in source.lower()

    # Verify adapter class has explicit warning
    assert "reference" in source.lower() or "debug" in source.lower()


def test_roundtrip_dataframe_factor_batch(pandas_adapter, sample_dataframe):
    """Test roundtrip conversion: DataFrame -> FactorBatch -> DataFrame."""
    # Convert to FactorBatch
    batch = pandas_adapter.dataframe_to_factor_batch(
        sample_dataframe,
        factor_cols=["momentum", "value"],
        time_col="date",
        asset_col="code",
    )

    # Convert back to DataFrame
    df_result = pandas_adapter.factor_batch_to_dataframe(
        batch,
        time_col="date",
        asset_col="code",
    )

    # Verify row count
    assert len(df_result) == len(sample_dataframe)

    # Verify columns exist
    assert "date" in df_result.columns
    assert "code" in df_result.columns
    assert "momentum" in df_result.columns
    assert "value" in df_result.columns

    # Verify values match (within tolerance)
    pd = pandas_adapter._pd
    merged = pd.merge(
        sample_dataframe,
        df_result,
        on=["date", "code"],
        suffixes=("_orig", "_result")
    )

    for factor in ["momentum", "value"]:
        orig_col = f"{factor}_orig"
        result_col = f"{factor}_result"
        if orig_col in merged.columns and result_col in merged.columns:
            assert np.allclose(merged[orig_col], merged[result_col], rtol=1e-6)
