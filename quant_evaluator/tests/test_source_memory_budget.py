from quant_evaluator.runtime.source_memory_budget import (
    PREFETCH_BYTES_PER_WORKER, estimate_source_memory,
)
import numpy as np
import pytest
import subprocess
import sys
from quant_evaluator.adapters.cos_factor_tile_source import BoundCosFactor, CosFactorTileSource
from quant_evaluator.contracts.factor_batch import AxisRef


def test_shared_source_estimator_preserves_existing_f48_width4_formula():
    cells = 2586 * 5461
    extra = (256 * 1024**2 + cells - 1) // cells
    estimate = estimate_source_memory(
        time_size=2586, asset_size=5461, factor_count=48, max_tile_size=4,
        dtype_itemsize=8, prefetch_workers=2, prefetch_enabled=True,
        extra_assembly_bytes_per_cell=extra)
    assert estimate.tile_factors == 4
    assert estimate.assembly_bytes == (
        3 * cells * 8 * 4 + 2 * cells * 4 + extra * cells * 4)
    assert estimate.prefetch_bytes == 2 * PREFETCH_BYTES_PER_WORKER
    assert estimate.total_bytes == estimate.assembly_bytes + estimate.prefetch_bytes
    assert estimate.total_bytes < 4096 * 1024**2

@pytest.mark.parametrize("dtype_itemsize", [4, 8])
@pytest.mark.parametrize("prefetch_enabled", [False, True])
@pytest.mark.parametrize("factor_count,max_tile_size", [(3, 8), (8, 4)])
@pytest.mark.parametrize("scratch", [0, 7])
def test_shared_estimator_matches_legacy_formula(dtype_itemsize, prefetch_enabled,
                                                  factor_count, max_tile_size, scratch):
    time_size, asset_size, workers = 7, 11, 2
    cells = time_size * asset_size
    tile_factors = min(factor_count, max_tile_size)
    legacy_assembly = (3 * cells * dtype_itemsize * tile_factors
                       + 2 * cells * tile_factors
                       + scratch * cells * tile_factors)
    legacy_prefetch = workers * PREFETCH_BYTES_PER_WORKER if prefetch_enabled else 0
    estimate = estimate_source_memory(
        time_size=time_size, asset_size=asset_size, factor_count=factor_count,
        max_tile_size=max_tile_size, dtype_itemsize=dtype_itemsize,
        prefetch_workers=workers, prefetch_enabled=prefetch_enabled,
        extra_assembly_bytes_per_cell=scratch)
    assert estimate.tile_factors == tile_factors
    assert estimate.assembly_bytes == legacy_assembly
    assert estimate.prefetch_bytes == legacy_prefetch
    assert estimate.total_bytes == legacy_assembly + legacy_prefetch

def test_shared_estimator_accepts_numpy_integer_axis_sizes():
    estimate = estimate_source_memory(
        time_size=np.int64(7), asset_size=np.int64(11), factor_count=3,
        max_tile_size=4, dtype_itemsize=8, prefetch_workers=0,
        prefetch_enabled=False)
    assert estimate.assembly_bytes == 3 * 7 * 11 * 8 * 3 + 2 * 7 * 11 * 3

def test_cos_source_accepts_numpy_integer_axis_sizes():
    time_axis = AxisRef("time", "int64", np.int64(2), np.array([1, 2], dtype=np.int64))
    asset_axis = AxisRef("asset", "int64", np.int64(3), np.array([1, 2, 3], dtype=np.int64))
    with CosFactorTileSource(
            records=(BoundCosFactor("f", "cos://b/f", "a" * 64, 1),),
            time_axis=time_axis, asset_axis=asset_axis, dtype="float64",
            manifest_snapshot={}, manifest_sha256="m" * 64, max_tile_size=1,
            read_factor=lambda *_: None, verify_manifest=lambda *_: None,
            make_tile=lambda *_: None, prefetch="off") as source:
        assert source.estimated_assembly_bytes == 2 * 3 * (3 * 8 + 2)

@pytest.mark.parametrize("field,value", [
    ("time_size", True), ("asset_size", 3.0), ("factor_count", -1),
    ("max_tile_size", 0), ("dtype_itemsize", False),
    ("prefetch_workers", -1), ("extra_assembly_bytes_per_cell", -1),
])
def test_shared_estimator_rejects_invalid_numeric_inputs(field, value):
    params = dict(time_size=1, asset_size=1, factor_count=1, max_tile_size=1,
                  dtype_itemsize=8, prefetch_workers=0, prefetch_enabled=False,
                  extra_assembly_bytes_per_cell=0)
    params[field] = value
    with pytest.raises(ValueError):
        estimate_source_memory(**params)

def test_fresh_interpreter_imports_adapter_with_shared_estimator():
    result = subprocess.run(
        [sys.executable, "-c", "import quant_evaluator.adapters.cos_factor_tile_source; "
         "import quant_evaluator.runtime.source_memory_budget"],
        capture_output=True, text=True, timeout=30, check=False)
    assert result.returncode == 0, result.stderr

@pytest.mark.parametrize("field", ["time_size", "asset_size", "factor_count", "max_tile_size", "dtype_itemsize"])
def test_shared_estimator_rejects_empty_dimensions(field):
    params = dict(time_size=1, asset_size=1, factor_count=1, max_tile_size=1,
                  dtype_itemsize=8, prefetch_workers=0, prefetch_enabled=False,
                  extra_assembly_bytes_per_cell=0)
    params[field] = 0
    with pytest.raises(ValueError):
        estimate_source_memory(**params)
