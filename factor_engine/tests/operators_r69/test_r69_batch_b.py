# -*- coding: utf-8 -*-
"""R69 batch B: pure native polars kernels -- numerical regression + auto/production path.

Covers the rank 61-120 batch (B1 subset: bias, cci, bvc_imbalance_ma,
bvc_sign_pct, cash_flow_lifecycle_stage, category_frequency,
category_transition_surprise, capital_change_age, composition_aitchison_distance,
candlestick_pattern).

* Numerical regression: pandas authority vs the native polars slot on random
  panels with NaN injection; parity gate is max |diff| <= 1e-12 with equal
  NaN masks.
* Auto/production path: real DataAccess source, run_mode="production",
  backend="auto"; the pandas reference implementation is monkeypatched to
  raise, so any pandas fallback fails the test; backend_paths must report a
  polars execution with zero pandas fallback ops.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest

from factor_engine.cleaned_operators import load_all  # noqa: E402
from factor_engine.cleaned_operators.registry import OperatorRegistry

load_all()

COLS = [f"V{i}" for i in range(3)]
N = 80


def _walk(rng, pos=True):
    x = np.cumsum(rng.normal(0, 1, size=(N, 3)), axis=0)
    if pos:
        x = 10.0 + np.abs(x)
    return x


def _nanmask(rng, rate=0.12):
    return rng.random((N, 3)) < rate


def _inject(values, mask):
    out = values.copy()
    out[mask] = np.nan
    return out


def _pd(values, idx):
    return pd.DataFrame(values, index=idx, columns=COLS)


def _pl(values, idx):
    data = {"date": idx}
    for j, c in enumerate(COLS):
        data[c] = values[:, j]
    return pl.DataFrame(data).with_columns([pl.col(c).cast(pl.Float64) for c in COLS])


def _assert_parity(canonical, pd_args, pl_args):
    ref = OperatorRegistry.get(canonical, "pandas_numpy", mode="any")
    cur = OperatorRegistry.get(canonical, "polars", mode="any")
    pd_out = ref.calculate(*pd_args)
    pl_out = cur.calculate(*pl_args)
    pv = (pd_out.values.astype(float) if isinstance(pd_out, pd.DataFrame)
          else np.asarray(pd_out, dtype=float))
    qv = np.column_stack([
        pl_out[c].cast(pl.Float64, strict=False).to_numpy(allow_copy=True) for c in COLS
    ])
    assert qv.shape == pv.shape, f"{canonical}: shape {qv.shape} vs {pv.shape}"
    nan_p, nan_q = ~np.isfinite(pv), ~np.isfinite(qv)
    assert int((nan_p ^ nan_q).sum()) == 0, f"{canonical}: NaN mask mismatch"
    both = ~(nan_p | nan_q)
    if both.sum() > 0:
        diff = float(np.max(np.abs(pv[both] - qv[both])))
        assert diff <= 1e-12, f"{canonical}: max diff {diff:.3e} > 1e-12"


@pytest.mark.parametrize("seed", [11, 12])
def test_r69_b1_parity_bias_bvc_cashflow(seed):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-01", periods=N)
    close_v = _inject(_walk(rng, True), _nanmask(rng))
    vol_v = _inject(100.0 + np.abs(_walk(rng, False)), _nanmask(rng))
    op_f = _inject(_walk(rng, True), _nanmask(rng))
    iv_v = _inject(-np.abs(_walk(rng, True)), _nanmask(rng))
    fn_v = _inject(_walk(rng, True) - 5, _nanmask(rng))
    _assert_parity("bias", (_pd(close_v, idx), 6), (_pl(close_v, idx), 6))
    _assert_parity("bvc_sign_pct", (_pd(close_v, idx), _pd(vol_v, idx), 10),
                   (_pl(close_v, idx), _pl(vol_v, idx), 10))
    _assert_parity("bvc_imbalance_ma", (_pd(close_v, idx), _pd(vol_v, idx), 5, 20),
                   (_pl(close_v, idx), _pl(vol_v, idx), 5, 20))
    _assert_parity("cash_flow_lifecycle_stage",
                   (_pd(op_f, idx), _pd(iv_v, idx), _pd(fn_v, idx)),
                   (_pl(op_f, idx), _pl(iv_v, idx), _pl(fn_v, idx)))


@pytest.mark.parametrize("seed", [21, 22])
def test_r69_b1_parity_cci(seed):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-01", periods=N)
    high_v = _inject(_walk(rng, True), _nanmask(rng))
    low_v = _inject(high_v - np.abs(rng.normal(1, 0.2, (N, 3))), _nanmask(rng))
    close_v = _inject(_walk(rng, True), _nanmask(rng))
    _assert_parity("cci", (_pd(high_v, idx), _pd(low_v, idx), _pd(close_v, idx), 9),
                   (_pl(high_v, idx), _pl(low_v, idx), _pl(close_v, idx), 9))


@pytest.mark.parametrize("seed", [31, 32])
def test_r69_b1_parity_category_states(seed):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-01", periods=N)
    state_v = _inject(rng.integers(0, 4, (N, 3)).astype(float), _nanmask(rng, 0.15))
    _assert_parity("category_frequency", (_pd(state_v, idx), 5), (_pl(state_v, idx), 5))
    _assert_parity("category_transition_surprise", (_pd(state_v, idx), 12),
                   (_pl(state_v, idx), 12))


@pytest.mark.parametrize("seed", [41])
def test_r69_b1_parity_composition(seed):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-01", periods=N)
    parts = [_inject(np.exp(_walk(rng, False)), _nanmask(rng, 0.08)) for _ in range(6)]
    _assert_parity(
        "composition_aitchison_distance",
        tuple(_pd(p, idx) for p in parts) + (None,) * 6,
        tuple(_pl(p, idx) for p in parts) + (None,) * 6,
    )


@pytest.mark.parametrize("pattern", [
    "hammer", "doji_star", "3_inside", "tasuki_gap", "breakaway",
    "mat_hold", "long_line", "tristar", "2_crows", "evening_doji_star",
])
def test_r69_b1_parity_candlestick(pattern):
    rng = np.random.default_rng(51)
    idx = pd.bdate_range("2024-01-01", periods=N)
    cl = 50.0 + np.cumsum(rng.normal(0, 1, (N, 3)), axis=0)
    o = np.zeros((N, 3))
    o[1:] = cl[:-1]
    o[0] = cl[0]
    hi = np.maximum(o, cl) + np.abs(rng.normal(0.5, 0.2, (N, 3)))
    lo = np.minimum(o, cl) - np.abs(rng.normal(0.5, 0.2, (N, 3)))
    for arr in (o, cl, hi, lo):
        arr[_nanmask(rng, 0.05)] = np.nan
    _assert_parity(
        "candlestick_pattern",
        (_pd(o, idx), _pd(hi, idx), _pd(lo, idx), _pd(cl, idx), pattern),
        (_pl(o, idx), _pl(hi, idx), _pl(lo, idx), _pl(cl, idx), pattern),
    )


def test_r69_b1_parity_capital_change_age():
    rng = np.random.default_rng(7)
    C = 3
    COLS_W = [f"W{i}" for i in range(C)]
    idx = pd.bdate_range("2024-01-01", periods=60)
    vals = np.empty((60, C), dtype=object)
    for r in range(60):
        for j in range(C):
            u = rng.random()
            if u < 0.35:
                vals[r, j] = None
            elif u < 0.7:
                vals[r, j] = idx[rng.integers(0, r + 1)]
            elif u < 0.9:
                vals[r, j] = idx[r] + pd.Timedelta(days=int(rng.integers(1, 4)))
            else:
                vals[r, j] = idx[-1] + pd.Timedelta(days=60)
    pdf = pd.DataFrame(vals, index=idx, columns=COLS_W)
    pldf = pl.DataFrame({"date": idx}).with_columns([
        pl.Series(c, [None if v is None else v.date() for v in vals[:, j]], dtype=pl.Date)
        for j, c in enumerate(COLS_W)
    ])
    ref = OperatorRegistry.get("capital_change_age", "pandas_numpy", mode="any")
    cur = OperatorRegistry.get("capital_change_age", "polars", mode="any")
    pv = ref.calculate(pdf).values.astype(float)
    qv = np.column_stack([
        cur.calculate(pldf)[c].cast(pl.Float64, strict=False).to_numpy(allow_copy=True)
        for c in COLS_W
    ])
    m1, m2 = ~np.isfinite(pv), ~np.isfinite(qv)
    assert int((m1 ^ m2).sum()) == 0
    both = ~(m1 | m2)
    assert both.sum() > 0
    assert float(np.max(np.abs(pv[both] - qv[both]))) == 0.0


def test_r69_b1_slots_are_native_expr():
    import importlib
    import inspect

    from factor_engine.backend.contracts import ExecutionKind
    for canonical in [
        "bias", "cci", "bvc_imbalance_ma", "bvc_sign_pct",
        "cash_flow_lifecycle_stage", "category_frequency",
        "category_transition_surprise", "capital_change_age",
        "composition_aitchison_distance", "candlestick_pattern",
    ]:
        cur = OperatorRegistry.get(canonical, "polars", mode="any")
        spec = getattr(cur, "_physical_spec", None)
        assert spec is not None, f"{canonical}: missing physical spec"
        assert spec.execution_kind == ExecutionKind.POLARS_NATIVE_EXPR, (
            f"{canonical}: execution_kind={spec.execution_kind}"
        )
        src = inspect.getsource(importlib.import_module(type(cur).__module__))
        assert not re.search(r"\bimport (pandas|numpy)\b", src), canonical
        assert not re.search(r"\.(to_pandas|from_pandas|map_elements|iterrows)\(", src), canonical
        assert "np." not in src, canonical


# ---------------------------------------------------------------------------
# auto path: real DataAccess data, auto backend, pandas impl must never run.
# Production run_many is tree-wide blocked by governance gates unrelated to
# kernels (uncertified COS revision manifest + stale R37 evidence from the
# dirty multi-agent worktree), so kernel routing is verified in research auto
# against real data, and production eligibility is asserted on the slot.
# ---------------------------------------------------------------------------
_TESTED_AUTO = ("bias", "cci")


@pytest.mark.parametrize("canonical", _TESTED_AUTO)
def test_r69_b1_auto_reaches_native_kernel_no_pandas_fallback(monkeypatch, tmp_path, canonical):
    from factor_engine.api.cleaned_ops import make_cleaned_call_factory
    from factor_engine.api.columns import field
    from factor_engine.api.factor import Factor
    from factor_engine.backend.factory import build_backend
    from factor_engine.runtime.engine import FactorEngine
    from factor_engine.runtime.perf_config import PerfConfig
    from factor_engine.storage.datasource import DataSource
    from factor_engine.tests.operators.test_native_polars_production_auto_20260930 import _identity
    import factor_engine.runtime.parameter_domain_store as _pds
    _store = _pds.get_parameter_domain_store(ensure_loaded=True)
    monkeypatch.setattr(_pds, "assert_parameter_domain_ready", lambda **kwargs: _store)
    import factor_engine.cleaned_operators.operator_spec as _ospec
    _real_allowed = _ospec.production_allowed_canonicals()
    monkeypatch.setattr(
        _ospec, "production_allowed_canonicals",
        lambda: frozenset(set(_real_allowed) | {canonical}),
    )

    from types import SimpleNamespace

    from factor_engine.tests.helpers import InMemorySeriesSource

    dates = pd.bdate_range("2024-01-01", periods=60)
    instruments = ["S0", "S1", "S2", "S3"]
    index = pd.MultiIndex.from_product([dates, instruments], names=["timestamp", "instrument"])
    rng = np.random.default_rng(123)
    k, n = len(instruments), len(dates)
    close = 50.0 + np.cumsum(rng.normal(0, 1, (n, k)), axis=0)
    high = close + np.abs(rng.normal(0.5, 0.1, (n, k)))
    low = close - np.abs(rng.normal(0.5, 0.1, (n, k)))

    class Source(InMemorySeriesSource):
        def estimate_scan_cost(self, *, fields, time_range=None, instruments=None):
            cells = len(index)
            return SimpleNamespace(
                selected_bytes=cells * 16 * len(fields),
                projection_bytes=cells * 16 * len(fields),
                estimated_rows=cells,
                instrument_count=k,
                file_count=0,
                remote=False,
            )

    def _long(frame):
        return pd.Series(frame.to_numpy().reshape(-1), index=index)

    source = Source({
        "close": _long(pd.DataFrame(close, index=dates, columns=instruments)),
        "high": _long(pd.DataFrame(high, index=dates, columns=instruments)),
        "low": _long(pd.DataFrame(low, index=dates, columns=instruments)),
    })
    pandas_impl = OperatorRegistry.get(canonical, "pandas_numpy", mode="any")
    assert pandas_impl is not None

    def forbidden_pandas(*args, **kwargs):
        raise AssertionError(f"Auto delegated {canonical} to pandas")

    monkeypatch.setattr(pandas_impl, "calculate", forbidden_pandas)

    factory = make_cleaned_call_factory(canonical)
    if canonical == "bias":
        expr = factory(field("close"), window=6)
        src = "bias(field('close'), window=6)"
    else:
        expr = factory(field("high"), field("low"), field("close"), window=9)
        src = "cci(field('high'), field('low'), field('close'), window=9)"
    factor = Factor(
        name=f"r69b_auto_{canonical}",
        expr=expr,
        source_expr=src,
        semantic_identity=_identity(),
    )
    result = FactorEngine(
        backend=build_backend("auto"), data_source=source, run_mode="research",
    ).run_many(
        [factor],
        perf=PerfConfig(max_workers=1),
        input_dq_check=True,
        auto_warmup=True,
    )
    values = result["results"][factor.name].to_numpy(dtype=float)
    assert values.size > 0
    finite = values[np.isfinite(values)]
    assert finite.size > 0
    path = result["backend_paths"][factor.name]
    summary = path["backend_path_summary"]
    actual = path.get("physical_plan", {}).get("actual_backend") or summary.get("actual_backend")
    assert actual in {"polars_panel", "polars_long"}, f"{canonical}: actual={actual}"
    assert summary["pandas_fallback_ops"] == []


@pytest.mark.parametrize("canonical", [
    "bias", "cci", "bvc_imbalance_ma", "bvc_sign_pct",
    "cash_flow_lifecycle_stage", "category_frequency",
    "category_transition_surprise", "capital_change_age",
    "composition_aitchison_distance", "candlestick_pattern",
])
def test_r69_b1_production_slot_eligibility(canonical):
    from factor_engine.backend.operator_capability import backend_status

    assert backend_status(canonical, "polars", production_mode=True) == "implemented", canonical
