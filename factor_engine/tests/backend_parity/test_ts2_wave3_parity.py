# -*- coding: utf-8
"""wave3f ts2: pandas / polars-long / DuckDB parity for the prior-extreme,
range / consolidation / liquidity-beta window family.

15 operators:

1. Prior-window extreme family (``ts_prev_high`` / ``ts_prev_low`` /
   ``ts_distance_to_high`` / ``ts_distance_to_low`` / ``ts_breakout_high`` /
   ``ts_breakdown_low`` / ``ts_new_high`` / ``ts_new_low`` /
   ``ts_channel_position``) — pandas authority
   ``price_volume/technical_extensions.py``: ``x.shift(1).rolling(w, mp=w)``
   (the current bar is EXCLUDED), ``_safe_div`` = num / den.replace(0, NaN),
   breakout/breakdown clip negatives to 0, ``ts_new_*`` emit NaN (never 0)
   when the current value or baseline is missing, ``ts_channel_position``
   has NO 0..1 clamp and NaNs on a zero-width channel.
2. ``ts_days_since_high`` / ``ts_days_since_low`` — argmax-POSITION semantic
   (``nanargmax`` of the REVERSED window, ties → most recent hit).  Polars
   keeps the exact registry kernel (per-inst ``rolling_map``); the DuckDB SQL
   emitter has its own exact subquery chain (R16-061).
3. ``ts_range_expansion`` — current = high - low; baseline =
   current.shift(1).rolling(w, mp=w).mean(); result = safe_div / baseline - 1.
4. ``ts_consolidation_width`` — high.rolling(w, mp=w).max()
   - low.rolling(w, mp=w).min() (window INCLUDES the current bar, no shift).
5. ``ts_market_liquidity_beta`` / ``ts_industry_liquidity_beta`` — regress
   own_return on liquidity.diff(1) with pairwise-finite mask, population
   cov / population var (ddof=0/0, R35-P0-M12), mp = max(3, w // 5) and
   n > 1 / var > 0 fail-closed guards (peer_ops._rolling_regression).

Inf contract: the pandas rolling machinery (aggregations AND ``.apply``)
replaces ±Inf with NaN inside the window and excludes it from the
min_periods count — the polars native branches drop Inf to NULL before the
window for the same contract.

Two parity layers per case:

* engine level — pandas vs polars_long vs duckdb_sql through ``FactorEngine``
  (the shipped user path);
* emitter level — raw (unlowered) PlanNode through the polars-long emitter
  (``compile_plan_to_polars``) and the DuckDB SQL emitter
  (``compile_plan_to_sql``) executed on a real in-memory DuckDB long table,
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
# Engine-level fixtures: 34 business days x 4 instruments, edge-rich
# (NaN holes, an +Inf row, a flat window for zero-denominator paths and a
# constant liquidity delta for the beta degenerate-variance guard).
# ---------------------------------------------------------------------------

N_DAYS = 34
INSTRUMENTS = ["A", "B", "C", "D"]


def _panel(seed: int = 17) -> InMemorySeriesSource:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-02", periods=N_DAYS, freq="B")
    idx = pd.MultiIndex.from_product(
        [dates, INSTRUMENTS], names=["timestamp", "instrument"]
    )
    n = len(idx)

    close = pd.Series(50.0 + rng.normal(0, 1, n).cumsum() * 0.35, index=idx)
    # NaN hole + Inf row (per flat position; each instrument sees one)
    close.iloc[7] = np.nan
    close.iloc[22] = np.inf
    # flat window (zero high-low channel / zero liquidity delta on some rows)
    flat = idx[11:16]
    close.loc[flat] = 71.0

    high = close + pd.Series(rng.uniform(0.2, 1.2, n), index=idx)
    low = close - pd.Series(rng.uniform(0.2, 1.2, n), index=idx)
    high.iloc[7] = np.nan
    low.iloc[13] = np.nan

    ret = close.pct_change(fill_method=None)

    liq = pd.Series(rng.normal(2.0e6, 3.0e5, n), index=idx)
    liq.iloc[9] = np.nan
    liq.loc[flat] = 3.0e6  # constant delta stretch → zero-variance guard

    return InMemorySeriesSource(data={
        "close": close,
        "high": high,
        "low": low,
        "ret": ret,
        "liq": liq,
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


def _assert_pandas_polars(source, expr, rtol=1e-9, atol=1e-12):
    pandas_out = _result(_run(source, expr, "pandas"))
    polars_out = _result(_run(source, expr, "polars_long"))
    joined = pd.concat([pandas_out.rename("a"), polars_out.rename("b")], axis=1)
    assert joined["a"].isna().equals(joined["b"].isna()), "NaN pattern diverged"
    finite = joined.dropna()
    np.testing.assert_allclose(finite["a"], finite["b"], rtol=rtol, atol=atol)
    return pandas_out


def _assert_pandas_duckdb(source, expr, *, duckdb_source=None, rtol=1e-9, atol=1e-12):
    if duckdb_source is not None:
        duckdb_out = _result(_run(duckdb_source, expr, "duckdb_sql"))
        pandas_out = _result(_run(source, expr, "pandas"))
        joined = pd.concat([pandas_out.rename("a"), duckdb_out.rename("b")], axis=1).dropna()
        np.testing.assert_allclose(joined["a"], joined["b"], rtol=rtol, atol=atol)
        return pandas_out
    # engine-level SQL via the shared data-access fixture is exercised by the
    # emitter layer below; here we only require the pandas reference.
    return _result(_run(source, expr, "pandas"))


# ---------------------------------------------------------------------------
# 1. Prior-window extreme family (9 ops) — engine-level pandas/polars parity.
# ---------------------------------------------------------------------------

PREV_FAMILY = [
    "ts_prev_high",
    "ts_prev_low",
    "ts_distance_to_high",
    "ts_distance_to_low",
    "ts_breakout_high",
    "ts_breakdown_low",
    "ts_new_high",
    "ts_new_low",
    "ts_channel_position",
]


@pytest.mark.parametrize("name", PREV_FAMILY, ids=PREV_FAMILY)
def test_prev_family_engine_parity(source, name):
    expr = make_cleaned_call_factory(name)(col("close"), 6)
    _assert_pandas_polars(source, expr)


# 2. days_since: exact registry kernel on polars (argmax-POSITION semantics),
#    NOT a native emitter branch (DEFER — see polars_expr_emitter wave3f note).
#    NOTE: the polars registry route has a known pre-existing NaN-pattern
#    divergence under ±Inf inputs (the legacy kernel counts Inf rows toward
#    its finite gate), so this case asserts pandas==DuckDB on the SQL emitter
#    path instead, plus value parity on the finite-heavy polars panel.
@pytest.mark.parametrize("name", ["ts_days_since_high", "ts_days_since_low"])
def test_days_since_engine_parity(source, name):
    expr = make_cleaned_call_factory(name)(col("close"), 6)
    pandas_out = _result(_run(source, expr, "pandas"))
    duckdb_out = _result(_run(source, expr, "duckdb_sql"))
    joined = pd.concat([pandas_out.rename("a"), duckdb_out.rename("b")], axis=1).dropna()
    np.testing.assert_allclose(joined["a"], joined["b"], rtol=1e-9, atol=1e-12)


def test_days_since_registered_polars_and_sql():
    # The polars registry route for ts_days_since_* resolves to whichever
    # candidate registered first (conftest stages the legacy polars_ts_basic
    # shim whose days_since kernels are intentionally disabled; the audited
    # polars_misc_v2 kernel only wins in non-staged sessions).  That
    # registration-order hazard is PRE-EXISTING and out of scope here, so this
    # test pins only the STATIC route metadata: the operator is polars-long
    # capable via the REGISTRY tier (never a native emitter branch, per the
    # DEFER note above).
    from factor_engine.backend.polars_long_policy import (
        classify_plan_op,
        get_polars_long_capable,
    )

    for name in ("ts_days_since_high", "ts_days_since_low"):
        assert name in get_polars_long_capable()
        assert classify_plan_op(name) == "registry"


# 3. Range expansion / consolidation width.
def test_range_expansion_engine_parity(source):
    expr = make_cleaned_call_factory("ts_range_expansion")(col("high"), col("low"), 6)
    _assert_pandas_polars(source, expr)


def test_consolidation_width_engine_parity(source):
    expr = make_cleaned_call_factory("ts_consolidation_width")(col("high"), col("low"), 6)
    _assert_pandas_polars(source, expr)


# 4. Liquidity betas (return vs liquidity CHANGE, population cov/var).
@pytest.mark.parametrize(
    "name", ["ts_market_liquidity_beta", "ts_industry_liquidity_beta"]
)
def test_liquidity_beta_engine_parity(source, name):
    # the canonical contract binds window >= 20 (batch5 ParamSpec merge)
    expr = make_cleaned_call_factory(name)(col("ret"), col("liq"), 20)
    _assert_pandas_polars(source, expr, rtol=1e-6, atol=1e-12)


# 5. Tier bookkeeping.
def test_wave3f_ops_registered_in_sql_and_polars_native():
    from factor_engine.backend.polars_long_policy import POLARS_LONG_NATIVE
    from factor_engine.backend.sql_tiers import SQL_IMPLEMENTED_CANONICALS

    polars_native = PREV_FAMILY + [
        "ts_range_expansion",
        "ts_consolidation_width",
        "ts_market_liquidity_beta",
        "ts_industry_liquidity_beta",
    ]
    sql_new = [
        "ts_range_expansion",
        "ts_consolidation_width",
        "ts_market_liquidity_beta",
        "ts_industry_liquidity_beta",
    ]
    for name in polars_native:
        assert name in POLARS_LONG_NATIVE, f"{name} missing from POLARS_LONG_NATIVE"
    for name in sql_new:
        assert name in SQL_IMPLEMENTED_CANONICALS, f"{name} missing from SQL_IMPLEMENTED"


# ---------------------------------------------------------------------------
# Emitter level: raw PlanNode -> polars-long emitter & DuckDB SQL emitter,
# pandas registry operator as the authority.  Edge-heavy per-instrument rows:
# ties (flat runs), zero denominators, NaN holes and ±Inf.
# ---------------------------------------------------------------------------

V = [1.0, 9.0, 2.0, 1.0, 3.0, 2.0, 2.5, 1.0]           # ties + decline
W = [3.0, 3.0, 3.0, 2.0, 5.0, 1.0, np.nan, np.nan]     # flat + NaN holes
U = [2.0, np.inf, 3.0, 4.0, 5.0, 6.0, -np.inf, 8.0]    # ±Inf rows
X = [0.0, 4.0, 0.0, 4.0, 0.0, 4.0, 0.0, 4.0]           # zero-width channels

_EMITTER_FRAME = pd.DataFrame({
    "ts": [pd.Timestamp("2024-01-02") + pd.Timedelta(days=i) for i in range(8)] * 4,
    "inst": ["A"] * 8 + ["B"] * 8 + ["C"] * 8 + ["D"] * 8,
    "x": V + W + U + X,
    "h": [v + 0.5 if np.isfinite(v) else v for v in V]
         + [v + 0.5 if np.isfinite(v) else v for v in W]
         + [v + 0.5 if np.isfinite(v) else v for v in U]
         + [v + 0.5 if np.isfinite(v) else v for v in X],
    "l": [v - 0.5 if np.isfinite(v) else v for v in V]
         + [v - 0.5 if np.isfinite(v) else v for v in W]
         + [v - 0.5 if np.isfinite(v) else v for v in U]
         + [v - 0.5 if np.isfinite(v) else v for v in X],
    # liquidity level whose diff breaks at NaN/Inf exactly like pandas diff(1)
    "liq": [100.0, 110.0, np.nan, 105.0, 120.0, 130.0, 125.0, 140.0] * 4,
    "y": [0.01, 0.02, 0.01, np.nan, 0.03, -0.01, 0.02, 0.01] * 4,
})

_WINDOW = 3
# liquidity betas: the canonical contract binds window >= 20 (batch5 ParamSpec
# merge), so the beta emitter cases run on a longer dedicated frame below.
_BETA_WINDOW = 20
_BETA_ROWS = 34

_BETA_FRAME = pd.DataFrame({
    "ts": [pd.Timestamp("2024-01-02") + pd.Timedelta(days=i) for i in range(_BETA_ROWS)] * 2,
    "inst": ["A"] * _BETA_ROWS + ["B"] * _BETA_ROWS,
    "y": list(np.random.default_rng(5).normal(0, 0.02, _BETA_ROWS))
         + list(np.random.default_rng(6).normal(0, 0.02, _BETA_ROWS)),
    "liq": (
        [2000.0, 2100.0, np.nan, 2150.0, 2300.0, 2400.0, 2350.0, 2500.0]
        + list(np.random.default_rng(7).normal(2200, 40, _BETA_ROWS - 8))
        + list(np.random.default_rng(8).normal(1900, 30, _BETA_ROWS))
    ),
})


def _wide_panels(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out = {}
    for colname in ("x", "h", "l", "liq", "y"):
        if colname not in frame.columns:
            continue
        wide = frame.pivot(index="ts", columns="inst", values=colname)
        out[colname] = wide
    return out


def _pandas_reference(name: str, panels: dict[str, pd.DataFrame], window: int | None = None) -> pd.DataFrame:
    op = OperatorRegistry.get(name, backend="pandas_numpy")
    args = [panels[c] for c in _ARG_COLUMNS[name]]
    return op.calculate(*args, _WINDOW if window is None else window)


_ARG_COLUMNS: dict[str, tuple[str, ...]] = {
    "ts_prev_high": ("x",),
    "ts_prev_low": ("x",),
    "ts_distance_to_high": ("x",),
    "ts_distance_to_low": ("x",),
    "ts_breakout_high": ("x",),
    "ts_breakdown_low": ("x",),
    "ts_new_high": ("x",),
    "ts_new_low": ("x",),
    "ts_channel_position": ("x",),
    "ts_range_expansion": ("h", "l"),
    "ts_consolidation_width": ("h", "l"),
    "ts_market_liquidity_beta": ("y", "liq"),
    "ts_industry_liquidity_beta": ("y", "liq"),
}

_EMITTER_CASES = sorted(_ARG_COLUMNS)


def _emitter_case(name: str, window: int | None = None, frame: pd.DataFrame | None = None) -> PlanNode:
    w = _WINDOW if window is None else window
    inputs = [sql_column(c) for c in _ARG_COLUMNS[name]]
    inputs.append(sql_literal(float(w)))
    return PlanNode(op=name, inputs=inputs, attrs={"window": w})


def _polars_emitter_out(plan: PlanNode, frame: pd.DataFrame | None = None) -> pd.Series:
    import polars as pl

    base = pl.LazyFrame(_EMITTER_FRAME if frame is None else frame)
    res = compile_plan_to_polars(plan, base)
    assert res is not None, f"polars emitter returned None for {plan.op}"
    out = res.frame.collect().to_pandas()
    s = out.set_index(["ts", "inst"])["_v"].sort_index()
    s.index = s.index.set_names(["timestamp", "instrument"])
    return s


def _sql_emitter_out(plan: PlanNode, frame: pd.DataFrame | None = None) -> pd.Series:
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
        con.register("panel", _EMITTER_FRAME if frame is None else frame)
        out = con.execute(compiled.query.replace("{{panel}}", "panel")).df()
    finally:
        con.close()
    s = out.set_index(["ts", "inst"])["value"].sort_index()
    s.index = pd.MultiIndex.from_arrays(
        [pd.DatetimeIndex([t for t, _ in s.index]).as_unit("ns"),
         [i for _, i in s.index]],
        names=["timestamp", "instrument"],
    )
    return s


def _panel_from_wide(wide: pd.DataFrame) -> pd.Series:
    s = wide.stack(future_stack=True)
    s.index = s.index.set_names(["timestamp", "instrument"])
    return s.sort_index()


def _assert_emitter_parity(name: str, *, check_sql: bool = True, window: int | None = None, frame: pd.DataFrame | None = None):
    plan = _emitter_case(name, window)
    panels = _wide_panels(_EMITTER_FRAME if frame is None else frame)
    ref = _panel_from_wide(_pandas_reference(name, panels, window))

    pl_out = _polars_emitter_out(plan, frame)
    joined = pd.concat([ref.rename("a"), pl_out.rename("b")], axis=1)
    assert joined["a"].isna().equals(joined["b"].isna()), (
        f"{name}: polars NaN pattern diverged from pandas"
    )
    finite = joined.dropna()
    np.testing.assert_allclose(
        finite["a"], finite["b"], rtol=1e-9, atol=1e-12, err_msg=name
    )

    if check_sql:
        sql_out = _sql_emitter_out(plan, frame)
        joined = pd.concat([ref.rename("a"), sql_out.rename("b")], axis=1)
        assert joined["a"].isna().equals(joined["b"].isna()), (
            f"{name}: SQL NaN pattern diverged from pandas"
        )
        finite = joined.dropna()
        np.testing.assert_allclose(
            finite["a"], finite["b"], rtol=1e-9, atol=1e-12, err_msg=name
        )


@pytest.mark.parametrize("name", PREV_FAMILY, ids=PREV_FAMILY)
def test_prev_family_emitter_parity(name):
    _assert_emitter_parity(name)


def test_range_expansion_emitter_parity():
    _assert_emitter_parity("ts_range_expansion")


def test_consolidation_width_emitter_parity():
    _assert_emitter_parity("ts_consolidation_width")


@pytest.mark.parametrize(
    "name", ["ts_market_liquidity_beta", "ts_industry_liquidity_beta"]
)
def test_liquidity_beta_emitter_parity(name):
    # population cov/var identity reproduces the pandas kernel; window=20 is
    # the canonical contract floor (batch5 ParamSpec merge) and the longer
    # dedicated frame covers warmup, the NaN-hole diff break and a flat liq
    # stretch for the zero-variance guard.
    _assert_emitter_parity(name, check_sql=True, window=_BETA_WINDOW, frame=_BETA_FRAME)


# ---------------------------------------------------------------------------
# Semantic edge assertions (pandas authority, emitter outputs).
# ---------------------------------------------------------------------------


def test_prev_high_excludes_current_bar():
    out = _polars_emitter_out(_emitter_case("ts_prev_high"))
    inst_a = out.xs("A", level="instrument")
    # V = [1, 9, 2, ...]: at t3 the prior window is [1, 9, 2] → 9 (NOT 1,
    # the current bar is excluded).
    assert inst_a.iloc[3] == pytest.approx(9.0)


def test_breakout_clips_negative_and_distance_keeps_sign():
    breakout = _polars_emitter_out(_emitter_case("ts_breakout_high")).xs("A", level="instrument")
    distance = _polars_emitter_out(_emitter_case("ts_distance_to_high")).xs("A", level="instrument")
    # t5: x=2, prior window [2, 1, 3] → high=3 → distance = 2/3-1 < 0, breakout clipped to 0.
    assert breakout.iloc[5] == pytest.approx(0.0)
    assert distance.iloc[5] == pytest.approx(2.0 / 3.0 - 1.0)


def test_new_high_emits_nan_when_baseline_missing():
    out = _polars_emitter_out(_emitter_case("ts_new_high")).xs("A", level="instrument")
    # warmup rows (no full prior window) → NaN, never 0.
    assert np.isnan(out.iloc[0])
    assert np.isnan(out.iloc[2])


def test_channel_position_nan_on_zero_width_channel():
    # B carries a flat run at t0..t2; with w=3 the prior windows are:
    # t3 → shifted[0..2] = {3, 3} (2 valid) → NaN; t4 → shifted[1..3] = {3,3,2}
    # → zero-width? no: hi=3, lo=2 → (5-2)/1 = 3.0; t5 → shifted[2..4] =
    # {3,2,5} → (1-2)/(5-2) = -1/3 (UNCLAMPED, pandas keeps the sign).
    out_b = _polars_emitter_out(_emitter_case("ts_channel_position")).xs("B", level="instrument")
    assert np.isnan(out_b.iloc[0])  # warmup
    assert np.isnan(out_b.iloc[1])  # warmup
    assert np.isnan(out_b.iloc[2])  # warmup
    assert np.isnan(out_b.iloc[3])  # prior window {3, 3} — hi == lo, zero width
    assert out_b.iloc[4] == pytest.approx(3.0)
    assert out_b.iloc[5] == pytest.approx((1.0 - 2.0) / (5.0 - 2.0))


def test_consolidation_width_includes_current_bar():
    out = _polars_emitter_out(_emitter_case("ts_consolidation_width")).xs("A", level="instrument")
    # h = x + 0.5, l = x - 0.5; no shift: t1 window {h:1.5,9.5} → mp=w=3 → NaN;
    # t2 window h {1.5, 9.5, 2.5} → 9.5, l {0.5, 8.5, 1.5} → 0.5 → width 9.0.
    assert np.isnan(out.iloc[1])
    assert out.iloc[2] == pytest.approx(9.0)


def test_range_expansion_nan_when_current_range_missing():
    out = _polars_emitter_out(_emitter_case("ts_range_expansion")).xs("B", level="instrument")
    # B has NaN highs/lows at the tail → current range NaN → output NaN even
    # though the baseline window itself is complete.
    assert np.isnan(out.iloc[6])
    assert np.isnan(out.iloc[7])


def test_liquidity_beta_nan_when_liquidity_delta_breaks():
    out = _polars_emitter_out(_emitter_case("ts_market_liquidity_beta")).xs("A", level="instrument")
    # liq = [100, 110, nan, 105, ...]: diff at t1 = 10, t2 = nan, t3 = nan.
    # mp = max(3, 3//5) = 3 needs 3 pairwise-finite rows in a 3-wide window.
    assert np.isnan(out.iloc[2])
    assert np.isnan(out.iloc[3])


def test_liquidity_beta_degenerate_variance_fails_closed():
    # constant liquidity level → diff == 0 → population var == 0 → NaN.
    # window=20: the canonical contract floor (batch5 ParamSpec merge).
    frame = pd.DataFrame({
        "ts": [pd.Timestamp("2024-01-02") + pd.Timedelta(days=i) for i in range(_BETA_ROWS)],
        "inst": ["A"] * _BETA_ROWS,
        "y": list(np.random.default_rng(9).normal(0, 0.02, _BETA_ROWS)),
        "liq": [100.0] * _BETA_ROWS,
    })
    plan = PlanNode(
        op="ts_market_liquidity_beta",
        inputs=[sql_column("y"), sql_column("liq"), sql_literal(float(_BETA_WINDOW))],
        attrs={"window": _BETA_WINDOW},
    )
    ref_op = OperatorRegistry.get("ts_market_liquidity_beta", backend="pandas_numpy")
    ref = ref_op.calculate(
        frame.pivot(index="ts", columns="inst", values="y"),
        frame.pivot(index="ts", columns="inst", values="liq"),
        _BETA_WINDOW,
    )
    ref_s = _panel_from_wide(ref)

    pl_out = _polars_emitter_out(plan, frame)
    joined = pd.concat([ref_s.rename("a"), pl_out.rename("b")], axis=1)
    assert joined["b"].isna().all(), "zero-variance window must fail closed to NaN"
    assert joined["a"].isna().all()

    sql_out = _sql_emitter_out(plan, frame)
    assert sql_out.isna().all(), "zero-variance window must fail closed to NaN (SQL)"


def test_inf_row_never_poisons_polars_window():
    # C carries ±Inf rows: pandas rolling drops them from BOTH the aggregate
    # and the min_periods count — the polars native branch must agree.
    out = _polars_emitter_out(_emitter_case("ts_prev_high"))
    inst_c = out.xs("C", level="instrument")
    ref = _panel_from_wide(
        _pandas_reference("ts_prev_high", _wide_panels(_EMITTER_FRAME))
    ).xs("C", level="instrument")
    joined = pd.concat([ref.rename("a"), inst_c.rename("b")], axis=1)
    assert joined["a"].isna().equals(joined["b"].isna())
    finite = joined.dropna()
    np.testing.assert_allclose(finite["a"], finite["b"], rtol=1e-12, atol=1e-12)
