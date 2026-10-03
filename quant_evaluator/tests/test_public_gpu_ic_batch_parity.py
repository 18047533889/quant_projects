"""Bounded real-CUDA public multi-metric IC parity for a 32-factor batch."""
import numpy as np
import pytest

cp = pytest.importorskip("cupy")
try:
    _cuda_devices = cp.cuda.runtime.getDeviceCount()
except cp.cuda.runtime.CUDARuntimeError:
    pytest.skip("CUDA driver unavailable", allow_module_level=True)
if _cuda_devices < 1:
    pytest.skip("CUDA device required", allow_module_level=True)

from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.runtime.evaluator import evaluate


def test_public_cuda_ic_metrics_match_cpu_for_32_factor_batch():
    # Tiny resident input, but enough factors to exercise the batched route.
    rng = np.random.default_rng(20261003)
    t, n, f = 24, 48, 32
    values = rng.normal(size=(t, n, f))
    labels = rng.normal(size=(t, n))
    validity = rng.random((t, n, f)) > 0.12
    label_validity = rng.random((t, n)) > 0.08
    values[1, :24, 0] = np.nan
    values[2, 18:, 1] = np.nan
    labels[3, :25] = np.nan
    times = np.arange(t, dtype=np.int64)
    assets = np.arange(n, dtype=np.int64)
    time_axis = AxisRef("time", str(times.dtype), t, times)
    asset_axis = AxisRef("asset", str(assets.dtype), n, assets)
    factor_ids = tuple(f"f{i}" for i in range(f))
    batch = FactorBatch(
        factor_ids, time_axis, asset_axis,
        values, validity=validity,
    )
    label = LabelBundle(
        "ret", labels, 1, decision_time=times, label_start_time=times,
        label_end_time=tuple(int(i) + 1 for i in times), asset_axis=asset_axis,
        validity=label_validity,
    )
    metrics = (
        "rank_ic_series", "rank_ic", "ic_std", "ic_ir",
        "pearson_ic_series", "pearson_ic", "pearson_ic_std", "pearson_ic_ir",
    )
    metric_parameters = {"ic_ir": {"min_periods": 20},
                         "pearson_ic_ir": {"min_periods": 20}}
    cpu = evaluate(batch, label, backend="cpu", metrics=metrics,
                   metric_parameters=metric_parameters)
    gpu = evaluate(batch, label, backend="cuda_strict", metrics=metrics,
                   metric_parameters=metric_parameters)

    for metric in metrics:
        cpu_artifact = cpu.artifacts[metric]
        gpu_artifact = gpu.artifacts[metric]
        np.testing.assert_array_equal(
            np.isfinite(gpu_artifact.values), np.isfinite(cpu_artifact.values),
        )
        np.testing.assert_allclose(
            gpu_artifact.values, cpu_artifact.values,
            rtol=1e-8, atol=1e-10, equal_nan=True,
        )
        cpu_counts = cpu_artifact.provenance.get("observation_counts")
        gpu_counts = gpu_artifact.provenance.get("observation_counts")
        if cpu_counts is not None or gpu_counts is not None:
            assert cpu_counts is not None and gpu_counts is not None
            np.testing.assert_array_equal(gpu_counts, cpu_counts)
    for metric in ("rank_ic", "ic_std", "ic_ir", "pearson_ic",
                   "pearson_ic_std", "pearson_ic_ir"):
        np.testing.assert_array_equal(
            gpu.artifacts[metric].provenance["observation_counts"],
            cpu.artifacts[metric].provenance["observation_counts"],
        )
