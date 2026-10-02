# -*- coding: utf-8 -*-
"""R69 batch N1: strict pure-Polars conversion tests (wave 1: 6 canonicals).

Covers, per protocol §每算子流程 7:
  * numeric regression vs the pandas authority (<=1e-12, NaN masks);
  * kernel purity (no pandas / numpy / UDF in the r69 batch-N1 module);
  * registry wiring (polars slot == POLARS_NATIVE_EXPR, r69-N1 source wins);
  * ``run_many`` auto path: no pandas fallback, polars panel execution;
  * medium-panel A/B speedup vs the pandas authority.
"""
from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import polars as pl
import pytest

_NATIVE_MODULE = "factor_engine.cleaned_operators.polars_native.r69_native_batchN1"

CANONICALS = (
    "open_to_vwap_return",
    "ts_path_efficiency",
    "ts_days_since",
    "ts_min_if",
    "ts_max_if",
    "ts_quantile_if",
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _panels(seed=20261002, rows=120, cols=5):
    rng = np.random.default_rng(seed)
    a = rng.normal(0.0, 0.02, size=(rows, cols))
    a[3:9, 0] = np.nan
    a[40:42, 1] = np.nan
    a[80, 2] = np.inf
    a[100:104, 3] = np.nan
    a[:, 4][::7] = np.nan
    cols_n = [f"i{j}" for j in range(cols)]
    xp = pd.DataFrame(a, columns=cols_n)
    xl = pl.DataFrame({"__fe_time__": np.arange(rows, dtype=np.int64),
                       **{c: a[:, j] for j, c in enumerate(cols_n)}})
    c = (rng.random(size=(rows, cols)) > 0.55).astype(float)
    c[rng.random(size=(rows, cols)) < 0.10] = np.nan
    cp = pd.DataFrame(c, columns=cols_n)
    cl = pl.DataFrame({"__fe_time__": np.arange(rows, dtype=np.int64),
                       **{c2: c[:, j] for j, c2 in enumerate(cols_n)}})
    return cols_n, xp, xl, cp, cl


def _registry():
    from factor_engine.cleaned_operators import load_all
    from factor_engine.cleaned_operators.registry import OperatorRegistry
    load_all()
    return OperatorRegistry


def _assert_parity(name, expected, actual, cols):
    e = expected[cols].to_numpy(dtype=float)
    a = actual.select(cols).to_numpy()
    with np.errstate(invalid="ignore"):
        assert np.array_equal(np.isnan(e), np.isnan(a)), name
    both = ~np.isnan(e)
    if both.any():
        d = np.abs(e[both] - a[both])
        rel = d / np.maximum(np.abs(e[both]), 1e-300)
        assert float(np.max(np.minimum(d, rel))) <= 1e-12, name


# ---------------------------------------------------------------------------
# numeric regression vs pandas authority
# ---------------------------------------------------------------------------
_CASES = [
    ("ts_path_efficiency", dict(window=20, min_periods=2), "x"),
    ("ts_days_since", dict(max_lookback=None), "condition"),
    ("ts_days_since", dict(max_lookback=10), "condition"),
    ("ts_min_if", dict(window=20, min_periods=3), "x"),
    ("ts_max_if", dict(window=20, min_periods=3), "x"),
    ("ts_quantile_if", dict(window=20, q=0.25, min_periods=3), "x"),
]


@pytest.mark.parametrize("name,kw,panel_key", _CASES)
def test_authority_parity(name, kw, panel_key):
    R = _registry()
    cols_n, xp, xl, cp, cl = _panels()
    ref = R.get(name, "pandas_numpy", mode="any")
    pd_args, pl_args = {}, {}
    for p in ref.metadata.param_names:
        if p in kw:
            pd_args[p], pl_args[p] = kw[p], kw[p]
        elif p == "condition":
            pd_args[p], pl_args[p] = cp, cl
        elif p in ("x", "ret", "y"):
            pd_args[p], pl_args[p] = xp, xl
    expected = ref.calculate(**pd_args)
    live = R.get(name, "polars", mode="any")
    actual = live.calculate(**pl_args)
    _assert_parity(name, expected, actual, cols_n)


def test_open_to_vwap_return_p0_61_gate():
    """vwap/open - 1 with the PositivePrice gate, vs the deterministic authority."""
    rng = np.random.default_rng(11)
    rows = 100
    o = np.abs(rng.normal(10.0, 1.0, size=rows))
    v = o * (1.0 + rng.normal(0.0, 0.01, size=rows))
    o[10] = np.nan; v[20] = 0.0; o[30] = -1.0; v[40] = np.inf
    with np.errstate(divide="ignore", invalid="ignore"):
        exp = v / o - 1.0
    exp[(np.isfinite(o) & (o <= 0)) | (np.isfinite(v) & (v <= 0))] = np.nan
    exp[~np.isfinite(exp)] = np.nan
    R = _registry()
    live = R.get("open_to_vwap_return", "polars", mode="any")
    got = live.calculate(
        open_px=pl.DataFrame({"__fe_time__": np.arange(rows, dtype=np.int64), "close": o}),
        vwap=pl.DataFrame({"__fe_time__": np.arange(rows, dtype=np.int64), "close": v}),
    )
    a = got["close"].to_numpy()
    with np.errstate(invalid="ignore"):
        assert np.array_equal(np.isnan(exp), np.isnan(a))
    m = ~np.isnan(exp)
    assert np.allclose(exp[m], a[m], rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# purity + registry wiring
# ---------------------------------------------------------------------------
def test_batch_n1_module_is_pandas_and_numpy_free():
    import ast

    import factor_engine.cleaned_operators.polars_native.r69_native_batchN1 as mod

    tree = ast.parse(open(mod.__file__).read())
    banned_modules = {"numpy", "pandas"}
    banned_calls = {
        "to_pandas", "from_pandas", "map_elements", "map_rows",
        "iterrows", "apply", "itertuples", "rolling_map",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(alias.name.split(".")[0] not in banned_modules for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            assert root not in banned_modules, node.module
        elif isinstance(node, ast.Call):
            func = node.func
            name = getattr(func, "attr", getattr(func, "id", ""))
            assert name not in banned_calls, name


def test_polars_slots_are_expression_native_from_r69_batch_n1():
    R = _registry()
    for name in CANONICALS:
        op = R.get(name, "polars", mode="any")
        assert op._physical_spec.execution_kind.value == "polars_native_expr", name
        meta = R._catalog[name]["backend_meta"]["polars"]
        assert meta["execution_kind"] == "polars_native_expr", name
        assert meta["source"] == _NATIVE_MODULE, name


# ---------------------------------------------------------------------------
# run_many: auto path, no pandas fallback
# ---------------------------------------------------------------------------
def _source_frame(rows=60, instruments=5, seed=77):
    rng = np.random.default_rng(seed)
    idx = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-01", periods=rows), [f"S{j}" for j in range(instruments)]],
        names=["timestamp", "instrument"],
    )
    n = len(idx)
    return pd.DataFrame({
        "ret": rng.normal(0.0, 0.02, size=n),
        "close": 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, size=n))),
        "cond": (rng.random(size=n) > 0.5).astype(float),
    }, index=idx)


class _LongSource:
    def __init__(self, frame):
        self.frame = frame
        self.instrument_filter = tuple(sorted(set(frame.index.get_level_values("instrument"))))
        self.start_date = frame.index.get_level_values("timestamp").min().tz_localize("UTC")
        self.end_date = frame.index.get_level_values("timestamp").max().tz_localize("UTC")
        self.full_history_start = self.start_date.strftime("%Y-%m-%d")
        self.schema = {c: "float64" for c in frame.columns}

    def scan_polars_long(self, columns):
        out = self.frame[list(columns)].reset_index().rename(
            columns={"timestamp": "ts", "instrument": "inst"}
        )
        return pl.from_pandas(out).lazy()

    def scan_index_long(self):
        return self.scan_polars_long(["close"]).select("ts", "inst")

    def load_column(self, name):
        return self.frame[name]

    def load_columns(self, names):
        return {n: self.frame[n] for n in names}

    def estimate_scan_cost(self, *, fields, time_range=None, instruments=None):
        rows = len(self.frame)
        return SimpleNamespace(
            selected_bytes=rows * 16, projection_bytes=rows * 16,
            estimated_rows=rows, instrument_count=len(self.instrument_filter),
            file_count=0, remote=False,
        )


def _factors():
    from factor_engine.api.columns import col
    from factor_engine.api.cleaned_ops import make_cleaned_call_factory
    from factor_engine.api.factor import Factor, FactorExecutionScopeHint

    build = make_cleaned_call_factory
    scope = FactorExecutionScopeHint(
        market="ashare",
        universe_id="ashare_stock_daily_adj",
        frequency="1d",
        calendar_id="SSE",
        decision_time_policy="close_to_close",
    )
    c = col("ret")
    x = col("close")
    cond = col("cond")
    def F(name, expr, src):
        return Factor(name=name, expr=expr, source_expr=src, semantic_identity=scope)
    return [
        F("ts_path_efficiency", build("ts_path_efficiency")(x, window=20),
          "ts_path_efficiency(field('close'), window=20)"),
        F("ts_days_since", build("ts_days_since")(cond),
          "ts_days_since(field('cond'))"),
        F("ts_min_if", build("ts_min_if")(x, cond, window=20),
          "ts_min_if(field('close'), field('cond'), window=20)"),
        F("ts_max_if", build("ts_max_if")(x, cond, window=20),
          "ts_max_if(field('close'), field('cond'), window=20)"),
        F("ts_quantile_if", build("ts_quantile_if")(x, cond, window=20, q=0.25),
          "ts_quantile_if(field('close'), field('cond'), window=20, q=0.25)"),
    ]


def test_run_many_auto_paths_are_polars_native_without_pandas_fallback():
    from factor_engine.backend import build_backend
    from factor_engine.runtime.engine import FactorEngine

    R = _registry()
    frame = _source_frame()
    factors = _factors()

    def run(backend_name):
        source = _LongSource(frame)
        engine = FactorEngine(
            backend=build_backend(backend_name), data_source=source,
            run_mode="research",
        )
        return engine.run_many(factors)

    expected = run("pandas")
    actual = run("auto")
    for name in [f.name for f in factors]:
        want = expected["results"][name].sort_index()
        got = actual["results"][name].sort_index()
        assert want.index.equals(got.index), name
        np.testing.assert_allclose(
            got.to_numpy(), want.to_numpy(), rtol=1e-12, atol=1e-12,
            equal_nan=True, err_msg=name,
        )
        path = actual["backend_paths"][name]
        assert path["physical_plan"]["actual_backend"] == "polars_panel", name
        assert path["backend_path_summary"]["pandas_fallback_ops"] == [], name
        op = R.get(name, "polars", mode="any")
        assert op._physical_spec.execution_kind.value == "polars_native_expr", name


# ---------------------------------------------------------------------------
# A/B speedup (medium panel, kernel fn vs pandas authority)
# ---------------------------------------------------------------------------
def test_medium_panel_ab_speedup():
    cols_n, xp, xl, _, _ = _panels(rows=600, cols=120, seed=99)
    R = _registry()

    start = time.perf_counter()
    expected = R.get("ts_path_efficiency", "pandas_numpy", mode="any").calculate(
        x=xp, window=20, min_periods=2
    )
    pandas_seconds = time.perf_counter() - start

    start = time.perf_counter()
    live = R.get("ts_path_efficiency", "polars", mode="any")
    actual = live.calculate(x=xl, window=20, min_periods=2)
    native_seconds = time.perf_counter() - start

    _assert_parity("ts_path_efficiency", expected, actual, cols_n)
    speedup = pandas_seconds / max(native_seconds, 1e-9)
    print(
        f"r69-batchN1 ts_path_efficiency A/B 600x120: pandas={pandas_seconds:.4f}s "
        f"polars_native={native_seconds:.4f}s speedup={speedup:.2f}x"
    )
    assert actual.height == 600
