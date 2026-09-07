# -*- coding: utf-8
"""wave3 valuation/shareholder family: pandas / polars-long / DuckDB parity.

15 operators across three shapes:

1. Pure elementwise ratios (``a_share_cap_ratio`` / ``free_float_ratio``) —
   ``valuation/ops_v2._safe_div`` (zero denominator → NaN, Inf quotient → NaN).
2. Bounded share-count ratios (``holder_pledge_ratio`` / ``holder_freeze_ratio``
   / ``holder_locked_share_ratio``) — the pandas authority is
   ``shareholder/churn_network._bounded_ratio``: num >= 0, den > 0 and
   ratio <= 1.0; any violation is a data error and fails closed to NaN
   (NOT the polars_churn_network ``_safe_div`` variant).
3. Elementwise gaps (``holder_float_concentration_gap`` /
   ``val1_relative_valuation_gap``) — x - y, both sides finite.
4. Log gaps (``valuation_pe_ttm_lyr_gap`` / ``valuation_pcf_definition_gap``) —
   log|x| - log|y|.
5. Sign-aware gaps (``valuation_pe_gap_signed_log`` /
   ``valuation_pcf_gap_signed_log``) — sign(x)*log1p|x| - sign(y)*log1p|y|.
6. Positive-only gaps (``valuation_pe_gap_positive`` /
   ``valuation_pcf_gap_positive``) — log(x) - log(y) only when both > 0.
7. Shift branches (``holder_pledge_change`` — pledge_ratio - shift(lag);
   ``circulating_cap_ratio_change`` — safe_div(cc,tc) - shift(1), the shift
   applies AFTER the ratio).

Two parity layers per case:

* engine level — pandas vs polars_long vs duckdb_sql through ``FactorEngine``
  (the shipped user path, ``test_fin_backend_gap_parity`` pattern);
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
# Engine-level fixtures: 30 business days x 6 instruments, edge-rich.
# ---------------------------------------------------------------------------

N_DAYS = 30
INSTRUMENTS = ["A", "B", "C", "D", "E", "F"]


def _panel(seed: int = 7):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-02", periods=N_DAYS, freq="B")
    idx = pd.MultiIndex.from_product(
        [dates, INSTRUMENTS], names=["timestamp", "instrument"]
    )
    n = len(idx)

    def _series(base, scale, *, zero_frac=0.0, nan_frac=0.0, neg_frac=0.0):
        s = pd.Series(rng.normal(base, scale, n), index=idx)
        if zero_frac:
            z = rng.choice(n, int(n * zero_frac), replace=False)
            s.iloc[z] = 0.0
        if nan_frac:
            z = rng.choice(n, int(n * nan_frac), replace=False)
            s.iloc[z] = np.nan
        if neg_frac:
            z = rng.choice(n, int(n * neg_frac), replace=False)
            s.iloc[z] = -np.abs(s.iloc[z])
        return s

    return InMemorySeriesSource(data={
        # valuation multiples (negative = loss-making, 0 = undefined)
        "pe_ratio": _series(25, 12, zero_frac=0.06, nan_frac=0.06, neg_frac=0.18),
        "pe_ratio_lyr": _series(24, 11, zero_frac=0.06, nan_frac=0.06, neg_frac=0.18),
        "pcf_ratio": _series(10, 5, zero_frac=0.06, nan_frac=0.06, neg_frac=0.18),
        "pcf_ratio2": _series(9, 4, zero_frac=0.06, nan_frac=0.06, neg_frac=0.18),
        # relative-valuation anchor pair
        "own_metric": _series(1.0, 0.4, nan_frac=0.08),
        "reference_metric": _series(1.0, 0.4, nan_frac=0.08),
        # share counts (share-capital family)
        "a_cap": _series(1e6, 2e5, zero_frac=0.04, nan_frac=0.04, neg_frac=0.10),
        "capitalization": _series(2e6, 4e5, zero_frac=0.04, nan_frac=0.04),
        "free_cap": _series(8e5, 2e5, zero_frac=0.04, nan_frac=0.04, neg_frac=0.10),
        "circulating_capital": _series(1.2e6, 3e5, zero_frac=0.04, nan_frac=0.04, neg_frac=0.10),
        "total_capital": _series(3e6, 6e5, zero_frac=0.03, nan_frac=0.04),
        "pledge_shares": _series(8e5, 3e5, zero_frac=0.04, nan_frac=0.04, neg_frac=0.12),
        "freeze_shares": _series(6e5, 3e5, zero_frac=0.04, nan_frac=0.04, neg_frac=0.12),
        "locked_shares": _series(9e5, 3e5, zero_frac=0.04, nan_frac=0.04, neg_frac=0.12),
        # concentration percentages (can legitimately be negative after demeaning
        # in synthetic data; keeps the elementwise-gap NaN contract exercised)
        "top10_concentration": _series(0.45, 0.12, nan_frac=0.06),
        "top10_float_concentration": _series(0.40, 0.12, nan_frac=0.06),
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


def _assert_engine_parity(source, expr, rtol=1e-9, atol=1e-12):
    pandas_out = _result(_run(source, expr, "pandas"))
    polars_out = _result(_run(source, expr, "polars_long"))
    pd.testing.assert_series_equal(
        pandas_out, polars_out, check_names=False, rtol=rtol, atol=atol
    )
    duckdb_out = _result(_run(source, expr, "duckdb_sql"))
    pd.testing.assert_series_equal(
        pandas_out, duckdb_out, check_names=False, rtol=rtol, atol=atol
    )


# ---------------------------------------------------------------------------
# Cases: (canonical, arg columns, extra literal/attrs)
# ---------------------------------------------------------------------------

TWO_COL_CASES = [
    ("a_share_cap_ratio", ("a_cap", "capitalization")),
    ("free_float_ratio", ("free_cap", "capitalization")),
    ("holder_pledge_ratio", ("pledge_shares", "total_capital")),
    ("holder_freeze_ratio", ("freeze_shares", "total_capital")),
    ("holder_locked_share_ratio", ("locked_shares", "total_capital")),
    ("holder_float_concentration_gap", ("top10_concentration", "top10_float_concentration")),
    ("val1_relative_valuation_gap", ("own_metric", "reference_metric")),
    ("valuation_pe_ttm_lyr_gap", ("pe_ratio", "pe_ratio_lyr")),
    ("valuation_pcf_definition_gap", ("pcf_ratio", "pcf_ratio2")),
    ("valuation_pe_gap_signed_log", ("pe_ratio", "pe_ratio_lyr")),
    ("valuation_pcf_gap_signed_log", ("pcf_ratio", "pcf_ratio2")),
    ("valuation_pe_gap_positive", ("pe_ratio", "pe_ratio_lyr")),
    ("valuation_pcf_gap_positive", ("pcf_ratio", "pcf_ratio2")),
]


@pytest.mark.parametrize(
    ("name", "cols"),
    TWO_COL_CASES,
    ids=[name for name, _ in TWO_COL_CASES],
)
def test_wave3_two_col_engine_parity(source, name, cols):
    expr = make_cleaned_call_factory(name)(col(cols[0]), col(cols[1]))
    _assert_engine_parity(source, expr)


def test_holder_pledge_change_engine_parity(source):
    expr = make_cleaned_call_factory("holder_pledge_change")(
        col("pledge_shares"), lag=2
    )
    _assert_engine_parity(source, expr, rtol=1e-8, atol=1e-12)


def test_circulating_cap_ratio_change_engine_parity(source):
    expr = make_cleaned_call_factory("circulating_cap_ratio_change")(
        col("circulating_capital"), col("total_capital")
    )
    _assert_engine_parity(source, expr, rtol=1e-8, atol=1e-12)


def test_wave3_ops_registered_in_sql_and_polars_native():
    from factor_engine.backend.polars_long_policy import POLARS_LONG_NATIVE
    from factor_engine.backend.sql_tiers import SQL_IMPLEMENTED_CANONICALS

    names = [name for name, _ in TWO_COL_CASES] + [
        "holder_pledge_change",
        "circulating_cap_ratio_change",
    ]
    for name in names:
        assert name in SQL_IMPLEMENTED_CANONICALS, f"{name} missing from SQL_IMPLEMENTED"
        assert name in POLARS_LONG_NATIVE, f"{name} missing from POLARS_LONG_NATIVE"


# ---------------------------------------------------------------------------
# Emitter level: raw PlanNode -> polars-long emitter & DuckDB SQL emitter,
# pandas registry operator as the authority.  Data is deliberately edge-heavy:
# num < 0 rows, den == 0 rows and ratio > 1 rows for the bounded family,
# non-positive values for the signed/positive log families.
# ---------------------------------------------------------------------------

EMITTER_ROWS = [
    # ts, inst, x, y, u, v
    (0, "A", 20.0, 19.0, 1e6, 2e6),      # plain positives
    (0, "B", -5.0, -4.0, -1e6, 2e6),     # negatives (positive-family -> NaN)
    (1, "A", 0.0, 3.0, 0.0, 2e6),        # zero numerator / x=0
    (1, "B", 5.0, 0.0, 5e5, 0.0),        # zero denominator (safe_div -> NaN)
    (2, "A", 7.0, 2.0, 3e6, 2e6),        # ratio > 1 (bounded -> NaN)
    (2, "B", np.nan, 4.0, np.nan, 2e6),  # NaN left side
    (3, "A", 4.0, np.nan, 1e6, np.nan),  # NaN right side
    (3, "B", -2.0, 6.0, 1e6, 4e6),       # mixed sign gap
    (4, "A", 0.5, 2.0, 5e5, 2e6),        # ratio <= 1 valid
    (4, "B", 12.0, 3.0, 1.5e6, 1.0e6),   # ratio > 1 bounded NaN, log family valid
]

_EMITTER_FRAME = pd.DataFrame(
    {
        "ts": [pd.Timestamp("2024-01-02") + pd.Timedelta(days=r[0]) for r in EMITTER_ROWS],
        "inst": [r[1] for r in EMITTER_ROWS],
        "x": [r[2] for r in EMITTER_ROWS],
        "y": [r[3] for r in EMITTER_ROWS],
        "u": [r[4] for r in EMITTER_ROWS],
        "v": [r[5] for r in EMITTER_ROWS],
    }
)


def _wide_panels(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out = {}
    for colname in ("x", "y", "u", "v"):
        out[colname] = frame.pivot(index="ts", columns="inst", values=colname)
    return out


def _pandas_reference(name: str, panels: dict[str, pd.DataFrame], lag: int | None = None):
    op = OperatorRegistry.get(name, backend="pandas_numpy")
    param_names = list(op.metadata.param_names)
    args = []
    kwargs = {}
    operand_idx = 0
    for p in param_names:
        if p == "lag":
            if lag is not None:
                kwargs["lag"] = lag
            continue
        args.append(panels[_COLUMN_MAP[name][operand_idx]])
        operand_idx += 1
    return op.calculate(*args, **kwargs)


def _emitter_case(name: str, cols: tuple[str, ...], lag: int | None = None) -> PlanNode:
    inputs = [sql_column(c) for c in cols]
    attrs: dict = {}
    if lag is not None:
        inputs.append(sql_literal(float(lag)))
        attrs["lag"] = lag
    return PlanNode(op=name, inputs=inputs, attrs=attrs)


def _polars_emitter_out(plan: PlanNode) -> pd.Series:
    import polars as pl

    base = pl.LazyFrame(_EMITTER_FRAME)
    res = compile_plan_to_polars(plan, base)
    assert res is not None, f"polars emitter returned None for {plan.op}"
    out = res.frame.collect().to_pandas()
    s = out.set_index(["ts", "inst"])["_v"].sort_index()
    s.index = s.index.set_names(["timestamp", "instrument"])
    return s


def _sql_emitter_out(plan: PlanNode) -> pd.Series:
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
        con.register("panel", _EMITTER_FRAME)
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


EMITTER_CASES = [
    (name, cols, None)
    for name, cols in TWO_COL_CASES
] + [
    ("holder_pledge_change", ("x",), 2),
    ("circulating_cap_ratio_change", ("u", "v"), None),
]

# Column mapping for the emitter edge frame: every case consumes an alias
# column (x / y / u / v).  TWO_COL_CASES map to the (x, y) pair, except the
# share-count ratios which map to (u, v) — num/denom need share-scale values
# plus num<0 / den==0 / ratio>1 edge rows.  The pandas reference consumes the
# same aliases via the same mapping.
_COLUMN_MAP = {
    "a_share_cap_ratio": ("u", "v"),
    "free_float_ratio": ("u", "v"),
    "holder_pledge_ratio": ("u", "v"),
    "holder_freeze_ratio": ("u", "v"),
    "holder_locked_share_ratio": ("u", "v"),
    "holder_float_concentration_gap": ("x", "y"),
    "val1_relative_valuation_gap": ("x", "y"),
    "valuation_pe_ttm_lyr_gap": ("x", "y"),
    "valuation_pcf_definition_gap": ("x", "y"),
    "valuation_pe_gap_signed_log": ("x", "y"),
    "valuation_pcf_gap_signed_log": ("x", "y"),
    "valuation_pe_gap_positive": ("x", "y"),
    "valuation_pcf_gap_positive": ("x", "y"),
    "holder_pledge_change": ("x",),
    "circulating_cap_ratio_change": ("u", "v"),
}


@pytest.mark.parametrize(
    ("name", "cols", "lag"),
    EMITTER_CASES,
    ids=[c[0] for c in EMITTER_CASES],
)
def test_wave3_emitter_polars_sql_parity(name, cols, lag):
    emit_cols = _COLUMN_MAP[name]
    plan = _emitter_case(name, emit_cols, lag)
    panels = _wide_panels(_EMITTER_FRAME)

    ref = _pandas_reference(name, panels, lag)
    ref_series = _panel_from_wide(ref)

    pl_out = _polars_emitter_out(plan)
    sql_out = _sql_emitter_out(plan)

    pd.testing.assert_series_equal(
        ref_series, pl_out, check_names=False, rtol=1e-9, atol=1e-12
    )
    pd.testing.assert_series_equal(
        ref_series, sql_out, check_names=False, rtol=1e-9, atol=1e-12
    )


def test_bounded_ratio_fail_closed_edges():
    """num<0 / den==0 / ratio>1 must all fail closed to NaN (pandas authority:
    shareholder/churn_network._bounded_ratio), unlike the non-authoritative
    polars _safe_div variant."""
    plan = _emitter_case("holder_pledge_ratio", ("u", "v"))
    pl_out = _polars_emitter_out(plan)
    sql_out = _sql_emitter_out(plan)

    key_neg = (pd.Timestamp("2024-01-02"), "B")     # num < 0
    key_den0 = (pd.Timestamp("2024-01-03"), "B")    # den == 0
    key_gt1 = (pd.Timestamp("2024-01-04"), "A")     # ratio > 1
    for out in (pl_out, sql_out):
        assert np.isnan(out.loc[key_neg])
        assert np.isnan(out.loc[key_den0])
        assert np.isnan(out.loc[key_gt1])
        assert out.loc[(pd.Timestamp("2024-01-02"), "A")] == pytest.approx(0.5)
        # ratio <= 1 valid row: num=5e5 / den=2e6
        assert out.loc[(pd.Timestamp("2024-01-06"), "A")] == pytest.approx(0.25)


def test_positive_log_gap_requires_strictly_positive():
    plan = _emitter_case("valuation_pe_gap_positive", ("x", "y"))
    pl_out = _polars_emitter_out(plan)
    sql_out = _sql_emitter_out(plan)
    for out in (pl_out, sql_out):
        assert np.isnan(out.loc[(pd.Timestamp("2024-01-02"), "B")])   # x < 0
        assert np.isnan(out.loc[(pd.Timestamp("2024-01-03"), "A")])   # x == 0
        assert np.isnan(out.loc[(pd.Timestamp("2024-01-03"), "B")])   # y == 0
        expected = np.log(20.0) - np.log(19.0)
        assert out.loc[(pd.Timestamp("2024-01-02"), "A")] == pytest.approx(expected)


def test_signed_log_gap_keeps_sign():
    plan = _emitter_case("valuation_pe_gap_signed_log", ("x", "y"))
    pl_out = _polars_emitter_out(plan)
    sql_out = _sql_emitter_out(plan)
    for out in (pl_out, sql_out):
        expected = np.sign(20.0) * np.log1p(20.0) - np.sign(19.0) * np.log1p(19.0)
        assert out.loc[(pd.Timestamp("2024-01-02"), "A")] == pytest.approx(expected)
        # (-5, -4) row: sign-preserving gap must be small negative, NOT the
        # sign-collapsed log|ratio| (which would be +0.223).
        expected_neg = np.sign(-5.0) * np.log1p(5.0) - np.sign(-4.0) * np.log1p(4.0)
        assert out.loc[(pd.Timestamp("2024-01-02"), "B")] == pytest.approx(expected_neg)
        # x = 0 row: sign(0)*log1p(0) = 0 → gap = 0 - sign(3)*log1p(3)
        assert out.loc[(pd.Timestamp("2024-01-03"), "A")] == pytest.approx(
            -np.log1p(3.0)
        )


def test_holder_pledge_change_lag_is_second_operand():
    plan = _emitter_case("holder_pledge_change", ("x",), lag=2)
    pl_out = _polars_emitter_out(plan)
    sql_out = _sql_emitter_out(plan)
    ts = sorted({t for t, _ in pl_out.index})
    # warmup rows: lag rows have no predecessor -> NaN
    for out in (pl_out, sql_out):
        for i in range(2):
            assert np.isnan(out.loc[(ts[i], "A")])
        # x[0]=20, x[2]=7 -> 7-20 = -13
        assert out.loc[(ts[2], "A")] == pytest.approx(7.0 - 20.0)
