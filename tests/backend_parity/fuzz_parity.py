# -*- coding: utf-8
"""算子 fuzz parity 框架（§二十）：固定种子随机 panel vs Pandas oracle。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("polars")

from factor_engine.api.cleaned_ops import make_cleaned_call_factory
from factor_engine.api.columns import col
from factor_engine.api.factor import Factor
from factor_engine.backend.factory import build_backend
from factor_engine.cleaned_operators import load_all
from factor_engine.runtime.engine import FactorEngine
from tests.helpers import InMemorySeriesSource

F = make_cleaned_call_factory
FUZZ_SEED = 20260711
FAILURE_DIR = Path(__file__).resolve().parents[2] / "evidence" / "fuzz_failures"


def _random_panel(rng: np.random.Generator, *, n_ts: int, n_inst: int, miss: float) -> pd.Series:
    dates = pd.date_range("2020-01-01", periods=n_ts, freq="B")
    insts = [f"I{i}" for i in range(n_inst)]
    idx = pd.MultiIndex.from_product([dates, insts], names=["timestamp", "instrument"])
    arr = rng.normal(0, 1, len(idx))
    mask = rng.random(len(idx)) < miss
    arr[mask] = np.nan
    if rng.random() < 0.1:
        arr[rng.integers(0, len(arr))] = np.inf
    return pd.Series(arr, index=idx)


def _run_both(source: InMemorySeriesSource, expr):
    pd_out = (
        FactorEngine(backend=build_backend("pandas"), data_source=source)
        .run(Factor(name="t", expr=expr))["result"]
        .sort_index()
    )
    pl_out = (
        FactorEngine(backend=build_backend("polars_long"), data_source=source)
        .run(Factor(name="t", expr=expr))["result"]
        .sort_index()
    )
    return pd_out, pl_out


def _save_failure(name: str, payload: dict) -> None:
    FAILURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FAILURE_DIR / f"{name}.json"
    path.write_text(json.dumps(payload, default=str, indent=2), encoding="utf-8")


@pytest.fixture(scope="module")
def _loaded():
    load_all()


@pytest.mark.parametrize(
    "op_builder",
    [
        pytest.param(lambda: F("abs")(col("x")), id="abs"),
        pytest.param(lambda: F("neg")(col("x")), id="neg"),
        pytest.param(lambda: F("ts_delay")(col("x"), 2), id="ts_delay"),
        pytest.param(lambda: F("ts_mean")(col("x"), 3), id="ts_mean"),
        pytest.param(lambda: F("add")(col("x"), 0.0), id="add_scalar"),
        pytest.param(lambda: F("rank_pct")(col("x")), id="rank_pct"),
    ],
)
def test_fuzz_pandas_polars_parity(_loaded, op_builder):
    rng = np.random.default_rng(FUZZ_SEED)
    for trial in range(5):
        n_ts = int(rng.integers(3, 30))
        n_inst = int(rng.integers(1, 8))
        miss = float(rng.random() * 0.5)
        x = _random_panel(rng, n_ts=n_ts, n_inst=n_inst, miss=miss)
        src = InMemorySeriesSource(data={"x": x})
        expr = op_builder()
        pd_out, pl_out = _run_both(src, expr)
        try:
            pd.testing.assert_series_equal(
                pd_out, pl_out, check_names=False, rtol=1e-6, atol=1e-6
            )
        except AssertionError as exc:
            _save_failure(
                f"{op_builder.__name__}_trial{trial}",
                {"n_ts": n_ts, "n_inst": n_inst, "miss": miss, "error": str(exc)},
            )
            raise
