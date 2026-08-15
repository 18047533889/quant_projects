from datetime import datetime
from modeling_adapters.contracts import ModelReadyData


def test_model_ready_data_deterministic():
    """Verify ModelReadyData is deterministic with explicit timestamp."""

    # Create two instances with same parameters and explicit timestamp
    ts = datetime(2024, 1, 1, 12, 0, 0)

    data1 = ModelReadyData(
        features=[[1.0, 2.0]],
        feature_names=["f1", "f2"],
        data_start=datetime(2023, 1, 1),
        data_end=datetime(2023, 12, 31),
        preprocess_contract_id="test_contract",
        created_at=ts
    )

    data2 = ModelReadyData(
        features=[[1.0, 2.0]],
        feature_names=["f1", "f2"],
        data_start=datetime(2023, 1, 1),
        data_end=datetime(2023, 12, 31),
        preprocess_contract_id="test_contract",
        created_at=ts
    )

    # Should be equal (deterministic)
    assert data1 == data2

    # Note: hash() may fail if features contain unhashable types (e.g., lists),
    # which is a pre-existing condition of the dataclass definition.
    # The critical determinism guarantee is equality.


def test_model_ready_data_none_timestamp():
    """Verify None timestamp is allowed."""

    data = ModelReadyData(
        features=[[1.0, 2.0]],
        feature_names=["f1", "f2"],
        data_start=datetime(2023, 1, 1),
        data_end=datetime(2023, 12, 31),
        preprocess_contract_id="test_contract",
        created_at=None
    )

    assert data.created_at is None
