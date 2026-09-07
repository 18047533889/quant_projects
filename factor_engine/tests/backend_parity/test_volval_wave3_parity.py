# -*- coding: utf-8
"""wave3c vol/valuation window family: pandas / polars-long / DuckDB parity.

21 operators across four thematic groups (pandas authority = the
``cleaned_operators`` registry kernels):

1. volatility-of-volatility / regime statistics (``vv1_*``):
   vol_of_vol, downside_vol_share, fractional_share, long_short_vol_beta,
   vol_acceleration, dispersion_vol, regime_change_ratio, vol_level_score.
2. OHLC high-frequency volatility estimators (``vr1_*``):
   parkinson_close_scale, garman_klass_ext, rogers_satchell,
   range_to_close_eff, range_everage, ewma_range_vol.
3. valuation trailing statistics (``val1_*``):
   valuation_z_own, valuation_percentile_own, earnings_yield_ma_diff,
   valuations_lag_component, earnings_yield_slope.
4. liquidity exposure (``vax_*``):
   liquidity_penalty_exposure, ret_per_liquidity_unit.

Semantics replicated exactly by both native branches:

* trailing windows ending at t (``.over(_INST, order_by=_TS)`` / DuckDB
  ``ROWS BETWEEN w-1 PRECEDING AND CURRENT ROW``); a NaN/Inf row never
  counts toward min_periods nor contributes a sample;
* std = ddof=1 (pandas rolling default) / DuckDB ``STDDEV_SAMP``;
* the compressed-finite-sequence semantics (fractional_share /
  long_short_vol_beta / lag_component / earnings_yield_slope) compress the
  window's finite values before the statistic (polars: list-eval; SQL:
  ``list_filter`` + ``list_aggregate`` / ``list_transform``);
* mp > window fails closed to NaN (the reference ``ValueError`` contract is
  carried by validation; the emitters return None / NULL outputs).

Two parity layers per case:

* engine level — pandas vs polars_long through ``FactorEngine``;
* emitter level — raw (unlowered) PlanNode through the polars-long emitter
  and the DuckDB SQL emitter executed on a real in-memory DuckDB table,
  compared against the pandas registry operator as authority.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("polars")
pytest.importorskip("duckdb")

import duckdb

from factor_engine.api.cleaned_ops import make_cleaned_call_factory
from factor_engine.api.columns import col
from factor_engine.api.factor import Factor
from factor_engine.backend.factory import build_backend
from factor_engine.backend.polars_expr_emitter import compile_plan_to_polars
from factor_engine.backend.sql_pushdown.emitter import (
    SqlDialect,
    compile_plan_to_sql,
    reset_sql_template_cache,
)
from factor_engine.backend.sql_pushdown.plan_fixtures import column as sql_column
from factor_engine.backend.sql_pushdown.plan_fixtures import literal as sql_literal
from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.planner.logical_plan import PlanNode
from factor_engine.runtime.engine import FactorEngine
from tests.helpers import InMemorySeriesSource

load_all()


# ---------------------------------------------------------------------------
# Engine-level fixtures: 60 business days x 6 instruments, NaN holes +
# zero/negative returns + zero-amount rows for the liquidity family.
# ---------------------------------------------------------------------------

N_DAYS = 60
INSTRUMENTS = ["A", "B", "C", "D", "E", "F"]


def _panel(seed: int = 11):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-02", periods=N_DAYS, freq="B")
    idx = pd.MultiIndex.from_product(
        [dates, INSTRUMENTS], names=["timestamp", "instrument"]
    )
    n = len(idx)
    close = 100.0 * np.exp(pd.Series(np.cumsum(rng.normal(0, 0.02, n)), index=idx))
    open_ = close * (1 + rng.normal(0, 0.005, n))
    high = np.maximum(close, open_) * (1 + np.abs(rng.normal(0, 0.005, n)))
    low = np.minimum(close, open_) * (1 - np.abs(rng.normal(0, 0.005, n)))
    ret = close.pct_change()
    ret.iloc[: 2 * len(INSTRUMENTS)] = 0.0
    amount = pd.Series(np.abs(rng.normal(1e6, 2e5, n)), index=idx)
    # a zero-amount row (a>0 gate of ret_per_liquidity_unit)
    amount.iloc[7 * len(INSTRUMENTS)] = 0.0
    amihud = ret.abs() / amount
    vm = pd.Series(rng.normal(1.0, 0.3, n), index=idx)
    ey = pd.Series(0.05 + rng.normal(0, 0.01, n), index=idx)
    for s in (close, open_, high, low, ret, amount, amihud, vm, ey):
        holes = rng.choice(n, n // 40, replace=False)
        s.iloc[holes] = np.nan
    return InMemorySeriesSource(data={
        "close": close, "open": open_, "high": high, "low": low,
        "ret": ret, "amount": amount, "amihud": amihud, "vm": vm, "ey": ey,
    })


@pytest.fixture(scope="module")
def source():
    return _panel()


def _run(source, expr, backend_name: str):
    return FactorEngine(
        backend=build_backend(backend_name), data_source=source, run_mode="research"
    ).run(Factor(name="t", expr=expr))


def _result(run_out) -> pd.Series:
    return run_out["result"].sort_index()


def _assert_engine_parity(source, expr, *, rtol=1e-6, atol=1e-9):
    pandas_out = _result(_run(source, expr, "pandas"))
    polars_out = _result(_run(source, expr, "polars_long"))
    pd.testing.assert_series_equal(
        pandas_out, polars_out, check_names=False, rtol=rtol, atol=atol
    )


# ---------------------------------------------------------------------------
# Cases: (canonical, arg columns, kwargs)
# ---------------------------------------------------------------------------

ENGINE_CASES = [
    ("vv1_vol_of_vol", ("ret",), dict(window=20, min_periods=5)),
    ("vv1_downside_vol_share", ("ret",), dict(window=40, min_periods=10)),
    ("vv1_fractional_share", ("ret",), dict(short_window=5, long_window=30, min_periods=15)),
    ("vv1_long_short_vol_beta", ("ret",), dict(short_window=5, long_window=30, min_periods=15)),
    ("vv1_vol_acceleration", ("ret",), dict(vol_window=10, accel_window=10, min_periods=4)),
    ("vv1_dispersion_vol", ("ret",), dict(vol_window=10, min_breadth=6)),
    ("vv1_regime_change_ratio", ("ret",), dict(short_window=5, long_window=30, band=0.5, min_periods=4)),
    ("vv1_vol_level_score", ("ret",), dict(short_window=5, history_window=30, min_periods=3)),
    ("vr1_parkinson_close_scale", ("high", "low", "close"), dict(window=20, min_periods=5)),
    ("vr1_garman_klass_ext", ("open", "high", "low", "close"), dict(window=20, min_periods=5)),
    ("vr1_rogers_satchell", ("open", "high", "low", "close"), dict(window=20, min_periods=5)),
    ("vr1_range_to_close_eff", ("open", "high", "low", "close"), dict(window=20, min_periods=5)),
    ("vr1_range_everage", ("high", "low", "close", "ret"), dict(window=20, min_periods=5)),
    ("vr1_ewma_range_vol", ("open", "high", "low", "close"), dict(decay_scale=10.0)),
    ("val1_valuation_z_own", ("vm",), dict(history_window=30, min_periods=10)),
    ("val1_valuation_percentile_own", ("vm",), dict(history_window=30, min_periods=10)),
    ("val1_earnings_yield_ma_diff", ("ey",), dict(baseline_window=12, min_periods=4)),
    ("val1_valuations_lag_component", ("vm",), dict(fast_window=5, slow_window=20, min_periods=4)),
    ("val1_earnings_yield_slope", ("ey",), dict(window=12, min_periods=5)),
    ("vax_liquidity_penalty_exposure", ("amihud",), dict(window=20, min_periods=5)),
    ("vax_ret_per_liquidity_unit", ("ret", "amount"), dict(window=20, min_periods=10)),
]


@pytest.mark.parametrize(
    ("name", "cols", "kwargs"),
    ENGINE_CASES,
    ids=[name for name, _, _ in ENGINE_CASES],
)
def test_volval_engine_polars_matches_pandas(source, name, cols, kwargs):
    """polars_long native branch vs the pandas authority (engine level)."""
    expr = make_cleaned_call_factory(name)(*[col(c) for c in cols], **kwargs)
    _assert_engine_parity(source, expr, rtol=1e-5, atol=1e-9)


def test_volval_ops_registered_in_sql_and_polars_native():
    from factor_engine.backend.polars_long_policy import POLARS_LONG_NATIVE
    from factor_engine.backend.sql_tiers import SQL_IMPLEMENTED_CANONICALS

    for name, _, _ in ENGINE_CASES:
        assert name in SQL_IMPLEMENTED_CANONICALS, f"{name} missing from SQL_IMPLEMENTED"
        assert name in POLARS_LONG_NATIVE, f"{name} missing from POLARS_LONG_NATIVE"


def test_volval_window_contract_mp_gt_window_fails_closed():
    """mp > window: the pandas reference raises ValueError (strict contract),
    so the emitters must NOT silently approximate — the polars branch returns
    None and the plan keeps failing closed rather than producing values."""
    source = _panel()
    expr = make_cleaned_call_factory("vv1_vol_of_vol")(
        col("ret"), window=5, min_periods=9
    )
    from factor_engine.backend.polars_long_policy import POLARS_LONG_NATIVE
    from factor_engine.backend.polars_expr_emitter import compile_plan_to_polars
    import polars as pl

    # the branch declines to compile the invalid parameterisation
    base = source.scan_polars_long(["ret"])
    plan_out = None
    try:
        plan = _node("vv1_vol_of_vol", ("ret",), [5, 9])
        plan_out = compile_plan_to_polars(plan, base)
    except ValueError:
        plan_out = None
    assert plan_out is None


# ---------------------------------------------------------------------------
# Emitter level: raw PlanNode -> polars-long emitter & DuckDB SQL emitter.
# 30 days x 2 instruments with NaN holes; the pandas registry operator is the
# authority for both emitters.
# ---------------------------------------------------------------------------

EMITTER_DAYS = 30
EMITTER_INSTS = ["A", "B", "C"]


def _emitter_frame(seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = list(pd.date_range("2024-01-02", periods=EMITTER_DAYS, freq="B"))
    idx = pd.MultiIndex.from_product(
        [ts, EMITTER_INSTS], names=["timestamp", "instrument"]
    )
    n = len(idx)
    close = 100.0 * np.exp(pd.Series(np.cumsum(rng.normal(0, 0.02, n)), index=idx))
    open_ = close * (1 + rng.normal(0, 0.005, n))
    high = np.maximum(close, open_) * (1 + np.abs(rng.normal(0, 0.005, n)))
    low = np.minimum(close, open_) * (1 - np.abs(rng.normal(0, 0.005, n)))
    low.iloc[7] = np.nan
    close.iloc[13] = np.nan
    ret = close.pct_change()
    ret.iloc[3] = np.nan
    amount = pd.Series(np.abs(rng.normal(1e6, 2e5, n)), index=idx)
    vm = pd.Series(rng.normal(1.0, 0.3, n), index=idx)
    vm.iloc[10] = np.nan
    ey = pd.Series(0.05 + rng.normal(0, 0.01, n), index=idx)
    amihud = ret.abs() / amount
    out = pd.DataFrame({
        "ts": [t for t, _ in idx],
        "inst": [i for _, i in idx],
        "ret": ret.to_numpy(),
        "vm": vm.to_numpy(),
        "ey": ey.to_numpy(),
        "amihud": amihud.to_numpy(),
        "amount": amount.to_numpy(),
        "close": close.to_numpy(),
        "open": open_.to_numpy(),
        "high": high.to_numpy(),
        "low": low.to_numpy(),
    })
    return out


@pytest.fixture(scope="module")
def emitter_frame():
    return _emitter_frame()


def _node(op: str, cols, lits) -> PlanNode:
    return PlanNode(
        op=op,
        inputs=[sql_column(c) for c in cols]
        + [sql_literal(v) for v in lits],
        attrs={},
    )


def _sql_emitter_out(plan: PlanNode, frame: pd.DataFrame) -> pd.Series:
    reset_sql_template_cache()
    compiled = compile_plan_to_sql(
        plan,
        dataset="panel",
        table="panel",
        time_column="ts",
        instrument_column="inst",
        dialect=SqlDialect.DUCKDB,
    )
    assert compiled is not None, f"SQL emitter returned None for {plan.op}"
    con = duckdb.connect()
    try:
        con.register("panel", frame)
        out = con.execute(compiled.query.replace("{{panel}}", "panel")).df()
    finally:
        con.close()
    s = out.set_index(["ts", "inst"])["value"].sort_index()
    s.index = s.index.set_names(["timestamp", "instrument"])
    return s


def _polars_emitter_out(plan: PlanNode, frame: pd.DataFrame) -> pd.Series:
    import polars as pl

    base = pl.LazyFrame(frame)
    res = compile_plan_to_polars(plan, base)
    assert res is not None, f"polars emitter returned None for {plan.op}"
    out = res.frame.collect().to_pandas()
    s = out.set_index(["ts", "inst"])["_v"].sort_index()
    s.index = s.index.set_names(["timestamp", "instrument"])
    return s


def _pandas_reference(name: str, frame: pd.DataFrame, cols, kwargs) -> pd.Series:
    op = OperatorRegistry.get(name, backend="pandas_numpy")
    panels = [frame.pivot(index="ts", columns="inst", values=c) for c in cols]
    ref = op.calculate(*panels, **kwargs)
    if isinstance(ref, pd.DataFrame):
        ref = ref.stack(future_stack=True)
    ref.index = ref.index.set_names(["timestamp", "instrument"])
    return ref.sort_index()


SQL_CASES = [
    ("vv1_vol_of_vol", ("ret",), [20, 5], dict(window=20, min_periods=5)),
    ("vv1_downside_vol_share", ("ret",), [40, 10], dict(window=40, min_periods=10)),
    (
        "vv1_vol_level_score",
        ("ret",),
        [5, 30, 3],
        dict(short_window=5, history_window=30, min_periods=3),
    ),
    (
        "vv1_regime_change_ratio",
        ("ret",),
        [5, 30, 0.5, 4],
        dict(short_window=5, long_window=30, band=0.5, min_periods=4),
    ),
    ("val1_valuation_z_own", ("vm",), [30, 10], dict(history_window=30, min_periods=10)),
    (
        "val1_valuation_percentile_own",
        ("vm",),
        [30, 10],
        dict(history_window=30, min_periods=10),
    ),
    (
        "val1_earnings_yield_ma_diff",
        ("ey",),
        [12, 4],
        dict(baseline_window=12, min_periods=4),
    ),
    (
        "vax_liquidity_penalty_exposure",
        ("amihud",),
        [20, 5],
        dict(window=20, min_periods=5),
    ),
    (
        "vv1_vol_acceleration",
        ("ret",),
        [10, 10, 4],
        dict(vol_window=10, accel_window=10, min_periods=4),
    ),
    ("vr1_ewma_range_vol", ("open", "high", "low", "close"), [10.0], dict(decay_scale=10.0)),
    (
        "vv1_fractional_share",
        ("ret",),
        [5, 30, 15],
        dict(short_window=5, long_window=30, min_periods=15),
    ),
    (
        "vv1_long_short_vol_beta",
        ("ret",),
        [5, 30, 15],
        dict(short_window=5, long_window=30, min_periods=15),
    ),
    (
        "val1_valuations_lag_component",
        ("vm",),
        [5, 20, 4],
        dict(fast_window=5, slow_window=20, min_periods=4),
    ),
    ("val1_earnings_yield_slope", ("ey",), [12, 5], dict(window=12, min_periods=5)),
    ("vv1_dispersion_vol", ("ret",), [10, 3], dict(vol_window=10, min_breadth=3)),
    (
        "vr1_parkinson_close_scale",
        ("high", "low", "close"),
        [20, 5],
        dict(window=20, min_periods=5),
    ),
    (
        "vr1_garman_klass_ext",
        ("open", "high", "low", "close"),
        [20, 5],
        dict(window=20, min_periods=5),
    ),
    (
        "vr1_rogers_satchell",
        ("open", "high", "low", "close"),
        [20, 5],
        dict(window=20, min_periods=5),
    ),
    (
        "vr1_range_to_close_eff",
        ("open", "high", "low", "close"),
        [20, 5],
        dict(window=20, min_periods=5),
    ),
    (
        "vr1_range_everage",
        ("high", "low", "close", "ret"),
        [20, 5],
        dict(window=20, min_periods=5),
    ),
    (
        "vax_ret_per_liquidity_unit",
        ("ret", "amount"),
        [20, 10],
        dict(window=20, min_periods=10),
    ),
]


@pytest.mark.parametrize(
    ("name", "cols", "lits", "kwargs"),
    SQL_CASES,
    ids=[name for name, *_ in SQL_CASES],
)
def test_volval_emitter_sql_matches_pandas(emitter_frame, name, cols, lits, kwargs):
    """DuckDB SQL emitter vs the pandas authority (emitter level)."""
    plan = _node(name, cols, lits)
    sql_out = _sql_emitter_out(plan, emitter_frame)
    ref = _pandas_reference(name, emitter_frame, cols, kwargs)
    df = pd.DataFrame({"sql": sql_out, "pd": ref}).dropna()
    assert len(df) > 0, f"{name}: SQL output has no finite values"
    np.testing.assert_allclose(
        df["sql"].to_numpy(),
        df["pd"].to_numpy(),
        rtol=1e-6,
        atol=1e-9,
        err_msg=f"{name}: DuckDB SQL does not match the pandas reference",
    )


@pytest.mark.parametrize(
    ("name", "cols", "lits", "kwargs"),
    SQL_CASES,
    ids=[name for name, *_ in SQL_CASES],
)
def test_volval_emitter_polars_matches_pandas(emitter_frame, name, cols, lits, kwargs):
    """polars-long emitter vs the pandas authority (emitter level, raw plan)."""
    plan = _node(name, cols, lits)
    pl_out = _polars_emitter_out(plan, emitter_frame)
    ref = _pandas_reference(name, emitter_frame, cols, kwargs)
    df = pd.DataFrame({"pl": pl_out, "pd": ref}).dropna()
    assert len(df) > 0, f"{name}: polars output has no finite values"
    np.testing.assert_allclose(
        df["pl"].to_numpy(),
        df["pd"].to_numpy(),
        rtol=1e-5,
        atol=1e-9,
        err_msg=f"{name}: polars long does not match the pandas reference",
    )


def test_vol_level_score_sigmoid_bounds():
    """The level score is a sigmoid of (cur/hmean - 1): bounded in (0, 1)."""
    frame = _emitter_frame()
    plan = _node("vv1_vol_level_score", ("ret",), [5, 30, 3])
    pl_out = _polars_emitter_out(plan, frame)
    sql_out = _sql_emitter_out(plan, frame)
    for out in (pl_out, sql_out):
        finite = out.dropna()
        assert len(finite) > 0
        assert float(finite.min()) > 0.0
        assert float(finite.max()) < 1.0


def test_downside_vol_share_bounded_0_1():
    """downside share = sd/(sd+su) lies strictly within (0, 1)."""
    frame = _emitter_frame()
    plan = _node("vv1_downside_vol_share", ("ret",), [40, 10])
    pl_out = _polars_emitter_out(plan, frame)
    sql_out = _sql_emitter_out(plan, frame)
    for out in (pl_out, sql_out):
        finite = out.dropna()
        assert len(finite) > 0
        assert float(finite.min()) > 0.0
        assert float(finite.max()) < 1.0
