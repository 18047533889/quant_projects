"""Kernel workspace metadata must fail closed before any device allocation."""
import numpy as np
import pytest

from quant_evaluator.kernels.gpu.quantile_numeric import _local_size_bytes
from quant_evaluator.kernels.gpu.quantile_finance_guard import local_workspace_bytes


class Kernel:
    def __init__(self, attributes):
        self.attributes = attributes
        self.compiled = False

    def compile(self):
        self.compiled = True


@pytest.mark.parametrize("value", [-1, True, False, 1.5, "1024", float("nan"), float("inf")])
def test_invalid_local_memory_metadata_rejected(value):
    kernel = Kernel({"local_size_bytes": value})
    with pytest.raises(RuntimeError, match="invalid"):
        _local_size_bytes(kernel)
    assert kernel.compiled


@pytest.mark.parametrize("key", ["local_size_bytes", "localSizeBytes"])
@pytest.mark.parametrize("value", [0, 8192, np.int64(1024)])
def test_valid_compiled_local_memory_metadata(key, value):
    kernel = Kernel({key: value})
    assert _local_size_bytes(kernel) == int(value)
    assert kernel.compiled


def test_missing_local_memory_metadata_rejected():
    with pytest.raises(RuntimeError, match="did not report"):
        _local_size_bytes(Kernel({}))


def test_invalid_primary_attribute_does_not_fall_back_to_alias():
    with pytest.raises(RuntimeError, match="invalid"):
        _local_size_bytes(Kernel({"local_size_bytes": -1, "localSizeBytes": 0}))


@pytest.mark.parametrize("value", [-1, True, False, 1.5, "1024", float("nan"), float("inf")])
def test_finance_workspace_invalid_metadata_rejected(value):
    with pytest.raises(RuntimeError, match="invalid"):
        local_workspace_bytes(Kernel({"local_size_bytes": value}), rows=129)


@pytest.mark.parametrize("rows,threads", [(0, 0), (1, 128), (128, 128), (129, 256)])
def test_finance_workspace_rounds_up_launch_and_accounts_actual_local_bytes(rows, threads):
    kernel = Kernel({"local_size_bytes": np.int64(1024)})
    actual = local_workspace_bytes(kernel, rows=rows)
    assert actual == {
        "local_size_bytes_per_thread": 1024,
        "launched_threads": threads,
        "local_workspace_bytes": 1024 * threads,
    }
    assert kernel.compiled
