from types import SimpleNamespace

import numpy as np
import pandas as pd
import polars as pl
import pytest

from factor_engine.cleaned_operators import rolling_pack as bridge
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.backend.operator_errors import OperatorShapeError


@pytest.fixture
def panels():
    dates = pd.date_range("2024-01-01", periods=3)
    return pl.DataFrame({"date": dates, "A": [1., 2., 3.]}), pl.DataFrame({"date": dates, "A": [4., 5., 6.]})


def install_reference(monkeypatch, fn):
    monkeypatch.setattr(OperatorRegistry, "get", lambda *args, **kwargs: SimpleNamespace(calculate=fn))


def test_actual_polars_keyword_operands_do_not_swap(monkeypatch, panels):
    x, y = panels
    install_reference(monkeypatch, lambda x, y: x-y)
    a = bridge._call_pandas_delegate("subtract", (), {"x": x, "y": y})
    b = bridge._call_pandas_delegate("subtract", (), {"y": y, "x": x})
    assert a.equals(b)
    assert a["A"].to_list() == [-3.] * 3
    assert a["date"].equals(x["date"])


def test_mixed_binding_scalar_stays_in_place(monkeypatch, panels):
    x, y = panels
    install_reference(monkeypatch, lambda x, scale, *, y: (x-y)*scale)
    out = bridge._call_pandas_delegate("scaled_subtract", (x, 3), {"y": y})
    assert out["A"].to_list() == [-9.] * 3


def test_duplicate_binding_rejected_by_reference(monkeypatch, panels):
    x, y = panels
    install_reference(monkeypatch, lambda x, y: x-y)
    with pytest.raises(TypeError):
        bridge._call_pandas_delegate("subtract", (x,), {"x": x, "y": y})


def test_true_time_axis_mismatch_rejected(monkeypatch, panels):
    x, y = panels
    install_reference(monkeypatch, lambda x, y: x-y)
    y = y.with_columns(pl.col("date") + pl.duration(days=1))
    with pytest.raises(ValueError):
        bridge._call_pandas_delegate("subtract", (x, y), {})


@pytest.mark.parametrize("mutation", [
    lambda frame: frame.iloc[::-1],
    lambda frame: frame.rename(columns={"A": "B"}),
    lambda frame: frame.reset_index(drop=True),
    lambda frame: frame.to_numpy(),
    lambda frame: frame.iloc[:1],
])
def test_result_shape_and_axis_not_relabelled(monkeypatch, panels, mutation):
    x, _ = panels
    install_reference(monkeypatch, mutation)
    with pytest.raises(OperatorShapeError):
        bridge._call_pandas_delegate("bad_axis", (x,), {})


def test_reference_mutation_cannot_change_shared_source(monkeypatch, panels):
    x, _ = panels
    original = x.clone()

    def mutate(frame):
        frame.iloc[:] = 99
        return frame

    install_reference(monkeypatch, mutate)
    assert bridge._call_pandas_delegate("mutate", (x,), {})["A"].to_list() == [99.] * 3
    assert x.equals(original)


def test_shared_input_converts_once(monkeypatch, panels):
    x, _ = panels
    install_reference(monkeypatch, lambda x, y: x-y)
    convert = bridge._pl_to_pd
    calls = []

    def tracked(frame):
        calls.append(frame)
        return convert(frame)

    monkeypatch.setattr(bridge, "_pl_to_pd", tracked)
    bridge._call_pandas_delegate("subtract", (x, x), {})
    assert len(calls) == 1
