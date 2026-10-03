from __future__ import annotations

import os
import subprocess
import sys

from quant_evaluator.runtime.gpu_quantile_shape_adapter import GPU_LINEAR_QUANTILE_METRICS
from quant_evaluator.runtime.source_auto_evidence import SOURCE_AUTO_METRICS
from quant_evaluator.runtime.source_metric_catalog import (
    SOURCE_IMPLEMENTED_METRICS,
    SOURCE_SERIES_METRICS,
)


def test_source_metric_catalog_is_immutable_and_includes_only_supported_union():
    assert isinstance(SOURCE_IMPLEMENTED_METRICS, frozenset)
    assert isinstance(SOURCE_SERIES_METRICS, frozenset)
    assert SOURCE_IMPLEMENTED_METRICS == SOURCE_AUTO_METRICS | GPU_LINEAR_QUANTILE_METRICS
    assert SOURCE_SERIES_METRICS == frozenset({"rank_ic_series", "pearson_ic_series"})


def test_shape_metrics_do_not_change_source_auto_certification():
    original_auto = SOURCE_AUTO_METRICS
    assert len(GPU_LINEAR_QUANTILE_METRICS) == 6
    assert GPU_LINEAR_QUANTILE_METRICS <= SOURCE_IMPLEMENTED_METRICS
    assert not GPU_LINEAR_QUANTILE_METRICS.intersection(SOURCE_AUTO_METRICS)
    from quant_evaluator.runtime import source_auto_evidence

    assert source_auto_evidence.SOURCE_AUTO_METRICS is original_auto
    assert SOURCE_AUTO_METRICS is original_auto


def test_catalog_import_does_not_import_cupy_or_gpu_kernels():
    code = """
import sys
from quant_evaluator.runtime.source_metric_catalog import SOURCE_IMPLEMENTED_METRICS
assert 'cupy' not in sys.modules
assert 'quant_evaluator.kernels.gpu.quantile_shape' not in sys.modules
assert len(SOURCE_IMPLEMENTED_METRICS) > 0
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=os.getcwd(),
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_vector_series_membership_rejects_non_series_and_unsupported_metrics():
    assert "rank_ic_series" in SOURCE_SERIES_METRICS
    assert "pearson_ic_series" in SOURCE_SERIES_METRICS
    assert "rank_ic" not in SOURCE_SERIES_METRICS
    assert "quantile_curvature" not in SOURCE_SERIES_METRICS
    assert "quantile_monotonicity" not in SOURCE_SERIES_METRICS
    assert "coverage" not in SOURCE_SERIES_METRICS
