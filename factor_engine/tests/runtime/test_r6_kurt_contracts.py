"""Full finite window Fisher estimator: independent oracle and backend contract."""
import numpy as np
import pandas as pd
import polars as pl
import pytest
from scipy.stats import kurtosis
from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.runtime.operator_snapshot import _parameter_contract

@pytest.fixture(scope="module", autouse=True)
def registry():
    load_all()

@pytest.mark.parametrize("backend", ["pandas_numpy", "polars"])
def test_full_finite_kurtosis_oracle_and_domains(backend):
    op = OperatorRegistry.get("ts_kurt", backend, mode="research")
    assert _parameter_contract(op, op.metadata.panel_params)[2]
    dates = pd.date_range("2025-01-01", periods=24)
    values = np.arange(1.,25.)
    frame = pd.DataFrame({"A": values, "B": np.ones(24)}, index=dates)
    def call(pdf, **kw):
        data = pl.from_pandas(pdf.rename_axis("date").reset_index()) if backend == "polars" else pdf
        out = op.calculate(data, **kw)
        return out.to_pandas().set_index("date") if backend == "polars" else out
    result = call(frame)
    assert result.A.iloc[:19].isna().all()
    assert result.B.isna().all()
    np.testing.assert_allclose(result.A.iloc[19:], [kurtosis(values[t-19:t+1], fisher=True, bias=False) for t in range(19,24)], atol=1e-12)
    for scale in (1e-300, 1e300):
        np.testing.assert_allclose(call(frame*scale,window=4), call(frame,window=4), equal_nan=True, atol=1e-12)
    for invalid in (np.nan, np.inf, -np.inf):
        bad = frame.copy(); bad.iloc[20,0] = invalid
        assert call(bad,window=4).A.iloc[20:24].isna().all()
    for bad in (True, 3, 4.5):
        with pytest.raises((ValueError,TypeError)):
            call(frame,window=bad)
    np.testing.assert_allclose(call(frame.iloc[:22]), result.iloc[:22], equal_nan=True)

def test_direct_twins_use_exact_kernel():
    import ast
    from pathlib import Path
    from factor_engine.cleaned_operators.base_polars import SeriesOperator
    from factor_engine.backend.contracts import ExecutionKind
    root = Path(__file__).resolve().parents[2] / "cleaned_operators"
    for file, name in (("common/time_series.py", "TSKurtosisPolars"), ("polars_native/ts_batch1.py", "TSKurtPolarsNative")):
        # Load the actual class without importing unrelated package registrations
        # into the sealed runtime registry.
        node = next(n for n in ast.parse((root/file).read_text()).body if isinstance(n, ast.ClassDef) and n.name == name)
        node.decorator_list = []
        ns = {"SeriesOperator": SeriesOperator}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(root/file), "exec"), ns)
        cls = ns[name]
        op = cls()
        out = op._calculate_series(pl.DataFrame({"A":[1.,2.,3.,4.],"B":[1.,1.,1.,1.]}),4)
        assert out["A"][-1] == pytest.approx(-1.2)
        assert np.isnan(out["B"][-1])
        spec = op.physical_spec()
        assert spec.execution_kind == ExecutionKind.POLARS_NUMPY_KERNEL
