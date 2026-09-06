import numpy as np
import pandas as pd
import pytest

from factor_engine.api.columns import col
from factor_engine.api.factor import Factor
from factor_engine.backend.factory import build_backend
from factor_engine.runtime.config import BackendConfig
from factor_engine.runtime.engine import FactorEngine, PhysicalPlanRequiredError
from factor_engine.runtime.perf_config import PerfConfig
from benchmarks.benchmark_run_many_streaming_20260906 import Source


def exercise(mode, backend, run_mode="research"):
    values = pd.Series(np.arange(6, dtype=float), index=pd.MultiIndex.from_product(
        [pd.date_range("2024-01-01", periods=3), ["A", "B"]], names=["timestamp", "instrument"]))
    engine = FactorEngine(backend=backend, data_source=Source(values), run_mode=run_mode)
    factors = [Factor(name=f"f{i}", expr=col("close") + i) for i in range(3)]
    perf = PerfConfig(max_workers=2, native_fusion=False)
    if mode == "run":
        results = {factor.name: engine.run(factor)["result"] for factor in factors}
    elif mode == "iter":
        results = {name: value for name, value, _ in engine.run_many_iter(factors, perf=perf)}
    elif mode == "stream":
        results = {}
        out = engine.run_many_stream(iter(factors), sink=lambda name, value: results.__setitem__(name, value),
                                     wave_size=2, n_jobs=2, perf=perf)
        assert out["completed_factors"] == 3
    elif mode == "parallel":
        results = engine.run_many_parallel(factors, n_jobs=2, perf=perf)["results"]
    else:
        results = engine.run_many(factors, perf=perf)["results"]
    assert set(results) == {"f0", "f1", "f2"}
    for name, result in results.items():
        pd.testing.assert_series_equal(result, values + int(name[1:]), check_names=False)


def test_config_default_is_supported_public_backend():
    assert BackendConfig().type == "pandas"


def test_loaded_config_default_and_explicit_auto(tmp_path):
    from factor_engine.runtime.config import load_config
    path = tmp_path / "config.yaml"
    base = "factor: {name: f, expr: close}\ndata_source: {type: parquet}\n"
    path.write_text(base)
    assert load_config(path).backend.type == "pandas"
    path.write_text(base + "backend: {type: auto}\n")
    assert load_config(path).backend.type == "auto"


@pytest.mark.parametrize("mode", ["run", "many", "iter", "parallel", "stream"])
def test_default_backend_public_values(mode):
    backend = build_backend()
    assert type(backend).__name__ == "PandasBackend"
    exercise(mode, backend)


@pytest.mark.parametrize("mode", ["run", "many", "iter", "parallel", "stream"])
def test_explicit_auto_still_typed_rejects(mode):
    with pytest.raises(PhysicalPlanRequiredError):
        exercise(mode, build_backend("auto"))


def test_default_does_not_relax_production_gate():
    with pytest.raises(Exception, match="production|market|input_dq_check|auto_warmup"):
        exercise("many", build_backend(), run_mode="production")
