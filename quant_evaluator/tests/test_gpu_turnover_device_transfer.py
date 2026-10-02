"""Device-resident turnover return path; public default remains compatible."""

import ast
import inspect
import numpy as np
import textwrap
import pytest

cp = pytest.importorskip("cupy")
if cp.cuda.runtime.getDeviceCount() < 1:
    pytest.skip("CUDA device required", allow_module_level=True)

from quant_evaluator.kernels.gpu import turnover
from quant_evaluator.kernels.gpu.tradability import batched_factor_turnover_rate
from quant_evaluator.kernels.gpu import tradability
from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.runtime.evaluator import evaluate


def test_turnover_device_return_avoids_wrapper_host_reupload(monkeypatch):
    rng = np.random.default_rng(20261002)
    values = rng.normal(size=(7, 3, 24)).astype(np.float64)
    values[1, 0, :4] = np.nan
    values[3, 2, :14] = 2.0  # tie-heavy cutoff
    factors = cp.asarray(values)

    # The lower-level public kernel keeps its established host return by default.
    host = batched_factor_turnover_rate(factors, quantile=.9)
    assert isinstance(host, np.ndarray)

    direct_device = batched_factor_turnover_rate(
        factors, quantile=.9, return_device=True)
    assert isinstance(direct_device, cp.ndarray)
    np.testing.assert_allclose(cp.asnumpy(direct_device), host, equal_nan=True)

    # The wrapper's legacy default still returns its usual device array.
    legacy = turnover.batched_membership_turnover(factors, quantile=.9)
    assert isinstance(legacy, cp.ndarray)
    np.testing.assert_allclose(cp.asnumpy(legacy), host, equal_nan=True)

    class NoUploadProxy:
        def __getattr__(self, name):
            return getattr(cp, name)

        def asarray(self, *args, **kwargs):
            raise AssertionError("device-return wrapper must not re-upload host data")

    monkeypatch.setattr(turnover, "_import_cp", lambda: NoUploadProxy())
    wrapped_device = turnover.batched_membership_turnover(
        factors, quantile=.9, return_device=True)
    assert isinstance(wrapped_device, cp.ndarray)
    np.testing.assert_allclose(cp.asnumpy(wrapped_device), host, equal_nan=True)


@pytest.mark.parametrize("shape", [(1, 2, 12), (3, 2, 9)])
def test_turnover_device_return_handles_empty_early_paths(shape):
    factors = cp.ones(shape, dtype=cp.float64)
    device_result = batched_factor_turnover_rate(factors, return_device=True)
    host_result = batched_factor_turnover_rate(factors)
    assert isinstance(device_result, cp.ndarray)
    assert isinstance(host_result, np.ndarray)
    assert device_result.shape == host_result.shape
    np.testing.assert_allclose(cp.asnumpy(device_result), host_result, equal_nan=True)


@pytest.mark.parametrize("bad", [None, 0, 1, "yes", np.bool_(True)])
def test_turnover_device_return_flag_requires_boolean(bad, monkeypatch):
    factors = cp.ones((2, 1, 12), dtype=cp.float64)
    monkeypatch.setattr(tradability, "_as_tfn",
                        lambda *_: pytest.fail("validated flag after input allocation"))
    with pytest.raises(TypeError, match="return_device must be a boolean"):
        batched_factor_turnover_rate(factors, return_device=bad)
    with pytest.raises(TypeError, match="return_device must be a boolean"):
        turnover.batched_membership_turnover(factors, return_device=bad)



def test_device_return_source_guards_every_host_get_behind_flag():
    source = textwrap.dedent(inspect.getsource(batched_factor_turnover_rate))
    function = ast.parse(source).body[0]
    parents = {}
    for node in ast.walk(function):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    get_calls = [
        node for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "get"
    ]
    assert len(get_calls) == 3
    for call in get_calls:
        parent = parents[call]
        assert isinstance(parent, ast.IfExp)
        assert isinstance(parent.test, ast.Name)
        assert parent.test.id == "return_device"
        assert parent.orelse is call


def test_executor_materializes_and_accounts_only_final_turnover_outputs():
    times = np.arange("2024-01-01", "2024-01-04", dtype="datetime64[D]")
    assets = np.arange(12, dtype=np.int64)
    time_axis = AxisRef("time", str(times.dtype), len(times), times)
    asset_axis = AxisRef("asset", str(assets.dtype), len(assets), assets)
    rng = np.random.default_rng(20261003)
    values = rng.normal(size=(len(times), len(assets), 2))
    labels = rng.normal(size=(len(times), len(assets)))
    batch = FactorBatch(("f0", "f1"), time_axis, asset_axis, values)
    label_bundle = LabelBundle(
        "ret", labels, 1, decision_time=tuple(times),
        label_start_time=tuple(times),
        label_end_time=tuple(times + np.timedelta64(1, "D")),
        asset_axis=asset_axis,
    )

    result = evaluate(
        batch, label_bundle, backend="cuda_strict",
        metrics=("factor_turnover_rate",))
    assert result.metadata["d2h_bytes"] == 2 * len(batch.factor_ids) * 8
    assert result.metadata["h2d_bytes"] == values.nbytes + labels.nbytes

