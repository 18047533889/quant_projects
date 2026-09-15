"""Malformed memory contracts must not turn a large task into a zero-byte task."""
import pytest
from factor_engine.runtime.task_resource_contract import TaskResourceContract


@pytest.mark.parametrize("value", [0.0, -1.0, 0.5, float("nan"), float("inf"), True])
def test_invalid_uncertainty_cannot_bypass_memory_admission(value):
    with pytest.raises((TypeError, ValueError)):
        TaskResourceContract(peak_memory_bytes=10**12, uncertainty=value)


@pytest.mark.parametrize("field", ["input_bytes", "peak_memory_bytes", "output_bytes", "spill_bytes"])
@pytest.mark.parametrize("value", [-1, 0.5, True, float("nan"), float("inf")])
def test_invalid_byte_quantities_rejected(field, value):
    with pytest.raises((TypeError, ValueError)):
        TaskResourceContract(**{field: value})


def test_numpy_quantities_and_uncertainty_one_remain_supported():
    import numpy as np
    contract = TaskResourceContract(input_bytes=np.int64(10), peak_memory_bytes=np.int64(20),
        output_bytes=np.int64(10), spill_bytes=np.int64(0), uncertainty=np.float64(1.0))
    assert contract.admissible_peak_bytes == 20
    for field in ("input_bytes", "peak_memory_bytes", "output_bytes", "spill_bytes"):
        assert type(getattr(contract, field)) is int
    assert contract.with_uncertainty(1.5).admissible_peak_bytes == 30
    with pytest.raises(ValueError):
        contract.with_uncertainty(0)
