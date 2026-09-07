# -*- coding: utf-8
"""wave3 flow/momentum/quality window family: pandas / polars-long / DuckDB parity.

22 operators across three pandas authorities (``cleaned_operators``):

1. ``wave1_orderflow`` — OFI imbalance statistics (``ofi_*``) and signed-volume
   dynamics (``sv_*``) over the ``signed_volume`` / ``amount`` inputs.
2. ``wave1_cs_momentum`` — momentum strength / stability / speed / volume
   weighting over the ``ret`` (+ ``turnover``) inputs.
3. ``wave1_earnings`` — cash-flow volatility / accrual quality over the
   ``operating_cash_flow`` / ``working_capital`` / ``earnings`` inputs.

Polars-long tiers (pandas registry operator is the authority):

* ``POLARS_LONG_NATIVE`` — pure trailing-window Expr branches in
  ``backend/polars_expr_emitter.py`` (``_FLOWMOM_WINDOW_OPS`` block): rolling
  sums over sign-decomposed volumes, population/sample std ratios, regime
  thresholds, prior-window self-relative change, consecutive-OK accrual
  kernels that ARE expressible per-row.
* ``POLARS_LONG_PYTHON_ROLLING`` / ``POLARS_LONG_MAP_GROUPS`` — COMPACTED
  finite-slice kernels (``ofi_reversal_rate``, ``ofi_imbalance_persistence``,
  ``m1_momentum_strength``, ``m1_momentum_speed_change``,
  ``aq1_accrual_stability``, ``aq1_accrual_ratio_dispersion``): pairs span NaN
  gaps (adjacent-FINITE pairing), and Π(1+r) can go negative (r < -1), so a
  pure per-row Expr cannot express the reference exactly.

Two parity layers per case:

* engine level — pandas vs polars_long vs duckdb_sql through ``FactorEngine``;
* emitter level — raw (unlowered) PlanNode through the polars-long emitter
  (``compile_plan_to_polars``) and the DuckDB SQL emitter
  (``compile_plan_to_sql``) executed on a real in-memory DuckDB long table.

Data is edge-rich: NaN holes, zeros (neutral signed volume / zero earnings),
sign flips and negative cash flows.
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


def _panel(seed: int = 11):
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
        # signed volume: positive = buy, negative = sell, 0 = neutral
        "signed_volume": _series(0.0, 5.0, zero_frac=0.08, nan_frac=0.08),
        "amount": _series(2e6, 8e5, zero_frac=0.04, nan_frac=0.06),
        "ret": _series(0.0, 0.03, nan_frac=0.07, neg_frac=0.2),
        "turnover": _series(1.5, 0.6, zero_frac=0.05, nan_frac=0.06),
        "operating_cash_flow": _series(2.0, 8.0, zero_frac=0.05, nan_frac=0.08, neg_frac=0.25),
        "earnings": _series(-1.0, 9.0, zero_frac=0.05, nan_frac=0.08, neg_frac=0.3),
        "working_capital": _series(12.0, 5.0, nan_frac=0.08),
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


def _assert_engine_parity(source, expr, rtol=1e-7, atol=1e-12):
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
# Cases: (canonical, arg columns, kwargs)
# ---------------------------------------------------------------------------

W = 8
MP = 3

FLOWMOM_CASES = [
    # ofi_* / sv_* over signed_volume / amount
    ("ofi_volume_imbalance", ("signed_volume",), {"window": W, "min_periods": MP}),
    ("ofi_abs_imbalance_trend", ("signed_volume",), {"window": W, "min_periods": MP}),
    ("ofi_dominant_direction", ("signed_volume",), {"window": W, "min_periods": MP, "threshold": 0.25}),
    ("ofi_imbalance_agreement", ("signed_volume",), {"window": W, "min_periods": MP}),
    ("ofi_imbalance_cv", ("signed_volume",), {"window": W, "min_periods": MP}),
    ("ofi_imbalance_persistence", ("signed_volume",), {"window": W, "min_periods": MP}),
    ("ofi_reversal_rate", ("signed_volume",), {"window": W, "min_periods": MP}),
    ("ofi_volume_flow_regime", ("signed_volume",), {"window": W, "min_periods": MP, "threshold": 0.3}),
    ("ofi_zero_flow_balance", ("signed_volume",), {"window": W, "min_periods": MP}),
    ("sv_net_flow_direction", ("signed_volume",), {"window": W, "min_periods": MP}),
    ("sv_own_flow_fraction", ("amount",), {"window": W, "min_periods": MP}),
    ("sv_signed_volume_volatility", ("signed_volume",), {"window": W, "min_periods": MP}),
    ("sv_self_relative_change", ("amount",), {"window": 6}),
    # m1_* over ret (+ turnover)
    ("m1_momentum_strength", ("ret",), {"window": W, "min_periods": 4}),
    ("m1_momentum_stability", ("ret",), {"window": W, "min_periods": 4}),
    ("m1_momentum_speed_change", ("ret",), {"fast_window": 4, "slow_window": 12, "min_periods": 3}),
    ("m1_volume_adjusted_momentum", ("ret", "turnover"), {"window": W, "min_periods": 3}),
    # aq1_* over ocf / wc / earnings
    ("aq1_cash_flow_volatility", ("operating_cash_flow",), {"window": W, "min_periods": 3}),
    ("aq1_accrual_stability", ("working_capital", "earnings"), {"window": W, "min_periods": 3}),
    ("aq1_cash_conversion_strength", ("operating_cash_flow", "earnings"), {"window": W, "min_periods": 3}),
    ("aq1_accrual_ratio_dispersion", ("working_capital", "earnings"), {"window": W, "min_periods": 3}),
    ("aq1_working_capital_accrual", ("working_capital", "earnings"), {"min_periods": 2}),
]


@pytest.mark.parametrize(
    ("name", "cols", "kwargs"),
    FLOWMOM_CASES,
    ids=[name for name, _, _ in FLOWMOM_CASES],
)
def test_wave3_flowmom_engine_parity(source, name, cols, kwargs):
    expr = make_cleaned_call_factory(name)(*[col(c) for c in cols], **kwargs)
    _assert_engine_parity(source, expr)


def test_wave3_flowmom_ops_registered_in_sql_and_polars():
    from factor_engine.backend.polars_long_policy import (
        POLARS_LONG_NATIVE,
        POLARS_LONG_PYTHON_ROLLING,
        POLARS_LONG_MAP_GROUPS,
        infer_polars_long_tier,
    )
    from factor_engine.backend.sql_tiers import SQL_IMPLEMENTED_CANONICALS

    for name, _, _ in FLOWMOM_CASES:
        assert name in SQL_IMPLEMENTED_CANONICALS, f"{name} missing from SQL_IMPLEMENTED"
        tier = infer_polars_long_tier(name)
        assert tier in {"native", "python_rolling", "map_groups"}, (
            f"{name} unexpected polars_long tier {tier!r}"
        )
    # the four compacted-slice kernels must NOT claim native
    for name in (
        "ofi_reversal_rate",
        "ofi_imbalance_persistence",
        "m1_momentum_strength",
        "m1_momentum_speed_change",
    ):
        assert name in POLARS_LONG_PYTHON_ROLLING
        assert name not in POLARS_LONG_NATIVE
    for name in ("aq1_accrual_stability", "aq1_accrual_ratio_dispersion"):
        assert name in POLARS_LONG_MAP_GROUPS
        assert name not in POLARS_LONG_NATIVE


# ---------------------------------------------------------------------------
# Emitter level: raw PlanNode -> polars-long emitter & DuckDB SQL emitter,
# pandas registry operator as the authority.  20 business days x 2
# instruments with NaN holes, zeros and sign flips.
# ---------------------------------------------------------------------------

def _emitter_frame(seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = 20
    ts = [pd.Timestamp("2024-01-02") + pd.Timedelta(days=i) for i in range(n)]

    def _vals(base, scale, zero=(), nan=(), neg=()):
        v = rng.normal(base, scale, n).round(3)
        for i in zero:
            v[i] = 0.0
        for i in nan:
            v[i] = np.nan
        for i in neg:
            v[i] = -abs(v[i])
        return list(v)

    frame = pd.DataFrame({"ts": np.repeat(ts, 2), "inst": ["A", "B"] * n})
    sv = _vals(0.0, 5.0, zero=(3, 7, 12), nan=(5, 16), neg=(2, 9, 14))
    frame["sv"] = sv + sv[::-1] if False else sv + list(reversed(sv))
    amt = _vals(2e6, 8e5, zero=(4, 11), nan=(6, 17))
    frame["amt"] = amt + list(reversed(amt))
    ret = _vals(0.0, 0.03, nan=(5, 13), neg=(2, 6, 9, 14, 18))
    frame["ret"] = ret + list(reversed(ret))
    turn = _vals(1.5, 0.6, zero=(4,), nan=(9,))
    frame["turn"] = turn + list(reversed(turn))
    ocf = _vals(2.0, 8.0, zero=(5,), nan=(6, 15), neg=(3, 8, 13))
    frame["ocf"] = ocf + list(reversed(ocf))
    ear = _vals(-1.0, 9.0, zero=(2, 12), nan=(8,), neg=(4, 10))
    frame["earn"] = ear + list(reversed(ear))
    wc = _vals(12.0, 5.0, nan=(5,))
    frame["wc"] = wc + list(reversed(wc))
    return frame


_EMITTER_FRAME = _emitter_frame()

_FLOWMOM_EMITTER_CASES = [
    ("ofi_volume_imbalance", ("sv",), {"window": 6, "min_periods": 3}),
    ("ofi_abs_imbalance_trend", ("sv",), {"window": 6, "min_periods": 3}),
    ("ofi_dominant_direction", ("sv",), {"window": 6, "min_periods": 3, "threshold": 0.25}),
    ("ofi_imbalance_agreement", ("sv",), {"window": 6, "min_periods": 3}),
    ("ofi_imbalance_cv", ("sv",), {"window": 6, "min_periods": 3}),
    ("ofi_imbalance_persistence", ("sv",), {"window": 6, "min_periods": 3}),
    ("ofi_reversal_rate", ("sv",), {"window": 6, "min_periods": 3}),
    ("ofi_volume_flow_regime", ("sv",), {"window": 6, "min_periods": 3, "threshold": 0.3}),
    ("ofi_zero_flow_balance", ("sv",), {"window": 6, "min_periods": 3}),
    ("sv_net_flow_direction", ("sv",), {"window": 6, "min_periods": 3}),
    ("sv_own_flow_fraction", ("amt",), {"window": 6, "min_periods": 3}),
    ("sv_signed_volume_volatility", ("sv",), {"window": 6, "min_periods": 3}),
    ("sv_self_relative_change", ("amt",), {"window": 5}),
    ("m1_momentum_strength", ("ret",), {"window": 6, "min_periods": 4}),
    ("m1_momentum_stability", ("ret",), {"window": 6, "min_periods": 4}),
    ("m1_momentum_speed_change", ("ret",), {"fast_window": 3, "slow_window": 8, "min_periods": 2}),
    ("m1_volume_adjusted_momentum", ("ret", "turn"), {"window": 6, "min_periods": 3}),
    ("aq1_cash_flow_volatility", ("ocf",), {"window": 6, "min_periods": 3}),
    ("aq1_accrual_stability", ("wc", "earn"), {"window": 6, "min_periods": 3}),
    ("aq1_cash_conversion_strength", ("ocf", "earn"), {"window": 6, "min_periods": 3}),
    ("aq1_accrual_ratio_dispersion", ("wc", "earn"), {"window": 6, "min_periods": 3}),
    ("aq1_working_capital_accrual", ("wc", "earn"), {"min_periods": 2}),
]


def _pandas_reference(name: str, cols: tuple[str, ...], kwargs: dict):
    op = OperatorRegistry.get(name, backend="pandas_numpy")
    panels = []
    for c in cols:
        panels.append(_EMITTER_FRAME.pivot(index="ts", columns="inst", values=c))
    ref = op.calculate(*panels, **kwargs)
    s = ref.stack(future_stack=True)
    s.index = s.index.set_names(["ts", "inst"])
    return s.sort_index()


def _emitter_plan(name: str, cols: tuple[str, ...], kwargs: dict) -> PlanNode:
    return PlanNode(
        op=name,
        inputs=[sql_column(c) for c in cols],
        attrs=dict(kwargs),
    )


def _polars_emitter_out(plan: PlanNode) -> pd.Series:
    import polars as pl

    base = pl.LazyFrame(_EMITTER_FRAME)
    res = compile_plan_to_polars(plan, base)
    assert res is not None, f"polars emitter returned None for {plan.op}"
    out = res.frame.collect().to_pandas()
    s = out.set_index(["ts", "inst"])["_v"].sort_index()
    s.index = s.index.set_names(["ts", "inst"])
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
    s.index = s.index.set_names(["ts", "inst"])
    return s


# compacted-slice kernels: DuckDB has no exact window-sum SQL (pair
# membership depends on per-window partner positions) — the SQL backend
# intentionally falls back to the polars long path, which the ENGINE-level
# parity test covers (pandas == duckdb_sql there).  Emitter-level SQL parity
# is asserted for the remaining ops.
_SQL_FALLBACK_FLOWMOM_OPS = frozenset({
    "ofi_imbalance_persistence", "ofi_reversal_rate",
    "ofi_abs_imbalance_trend",
    "aq1_accrual_stability", "aq1_accrual_ratio_dispersion",
})


@pytest.mark.parametrize(
    ("name", "cols", "kwargs"),
    _FLOWMOM_EMITTER_CASES,
    ids=[c[0] for c in _FLOWMOM_EMITTER_CASES],
)
def test_wave3_flowmom_emitter_polars_sql_parity(name, cols, kwargs):
    plan = _emitter_plan(name, cols, kwargs)
    ref = _pandas_reference(name, cols, kwargs)
    pl_out = _polars_emitter_out(plan)

    pd.testing.assert_series_equal(
        ref, pl_out, check_names=False, rtol=1e-7, atol=1e-10
    )

    if name in _SQL_FALLBACK_FLOWMOM_OPS:
        reset_sql_template_cache()
        from factor_engine.backend.sql_pushdown.emitter import _compile_layer_impl

        assert (
            _compile_layer_impl(plan, dialect=SqlDialect.DUCKDB) is None
        ), f"{name} should stay on the SQL fallback list"
        return

    sql_out = _sql_emitter_out(plan)
    pd.testing.assert_series_equal(
        ref, sql_out, check_names=False, rtol=1e-7, atol=1e-10
    )


def test_ofi_volume_imbalance_zero_total_fail_closed():
    """A window whose signed volume is ALL zero must fail closed to NaN."""
    kwargs = {"window": 4, "min_periods": 2}
    plan = _emitter_plan("ofi_volume_imbalance", ("sv",), kwargs)
    pl_out = _polars_emitter_out(plan)
    sql_out = _sql_emitter_out(plan)
    # column B: rows 0..3 = values from the reversed half — construct the
    # expectation from the pandas reference instead of hard-coding.
    ref = _pandas_reference("ofi_volume_imbalance", ("sv",), kwargs)
    # rows where the whole window is zero: B col contains sv rows 0..3
    # reversed: [nan, 13.x?, ...] — just assert consistency with reference
    # on the all-zero window of A at t=4 (sv A row4=0, but window has
    # non-zeros) — instead directly probe a synthetic all-zero panel.
    zero_frame = pd.DataFrame({
        "ts": np.repeat(pd.date_range("2024-01-02", periods=5), 1),
        "inst": ["A"] * 5,
        "sv": [0.0, 0.0, 0.0, 0.0, 0.0],
    })
    global _EMITTER_FRAME  # noqa: PLW0603
    original = _EMITTER_FRAME
    try:
        globals()["_EMITTER_FRAME"] = zero_frame
        plan_zero = _emitter_plan("ofi_volume_imbalance", ("sv",), {"window": 4, "min_periods": 2})
        pl_zero = _polars_emitter_out(plan_zero)
        sql_zero = _sql_emitter_out(plan_zero)
        assert pl_zero.dropna().empty
        assert sql_zero.dropna().empty
    finally:
        globals()["_EMITTER_FRAME"] = original
    # and the main panel parity still holds
    pd.testing.assert_series_equal(ref, pl_out, check_names=False, rtol=1e-7, atol=1e-10)
    pd.testing.assert_series_equal(ref, sql_out, check_names=False, rtol=1e-7, atol=1e-10)


def test_ofi_dominant_direction_threshold_bounds():
    for bad in (-0.1, 1.5):
        plan = _emitter_plan(
            "ofi_dominant_direction", ("sv",),
            {"window": 6, "min_periods": 3, "threshold": bad},
        )
        with pytest.raises(Exception):
            _polars_emitter_out(plan)
        with pytest.raises(Exception):
            _sql_emitter_out(plan)


def test_sv_net_flow_direction_zero_volume_window_is_zero():
    """mean|sv| <= 1e-12 windows emit 0.0 (pandas contract), not NaN."""
    zero_frame = pd.DataFrame({
        "ts": pd.date_range("2024-01-02", periods=5),
        "inst": ["A"] * 5,
        "sv": [0.0, 0.0, 0.0, 0.0, 0.0],
    })
    global _EMITTER_FRAME  # noqa: PLW0603
    original = _EMITTER_FRAME
    try:
        globals()["_EMITTER_FRAME"] = zero_frame
        plan = _emitter_plan("sv_net_flow_direction", ("sv",), {"window": 4, "min_periods": 2})
        pl_out = _polars_emitter_out(plan)
        sql_out = _sql_emitter_out(plan)
        assert (pl_out.dropna() == 0.0).all()
        assert (sql_out.dropna() == 0.0).all()
        assert not pl_out.dropna().empty
    finally:
        globals()["_EMITTER_FRAME"] = original


def test_m1_momentum_speed_change_fast_must_be_smaller():
    plan = _emitter_plan(
        "m1_momentum_speed_change", ("ret",),
        {"fast_window": 8, "slow_window": 4, "min_periods": 2},
    )
    with pytest.raises(Exception):
        _polars_emitter_out(plan)
    with pytest.raises(Exception):
        _sql_emitter_out(plan)


def test_aq1_working_capital_accrual_delta_over_abs_earnings():
    kwargs = {"min_periods": 2}
    plan = _emitter_plan("aq1_working_capital_accrual", ("wc", "earn"), kwargs)
    ref = _pandas_reference("aq1_working_capital_accrual", ("wc", "earn"), kwargs)
    pl_out = _polars_emitter_out(plan)
    sql_out = _sql_emitter_out(plan)
    # warmup row has no predecessor (ΔWC needs a finite prior row)
    ts0 = _EMITTER_FRAME["ts"].iloc[0]
    assert np.isnan(pl_out.loc[(ts0, "A")])
    assert np.isnan(sql_out.loc[(ts0, "A")])
    # zero-earnings rows fail closed (|e| <= 1e-12 -> NaN), matching ref
    a_rows = _EMITTER_FRAME[_EMITTER_FRAME["inst"] == "A"].reset_index(drop=True)
    for i in range(1, len(a_rows)):
        if abs(a_rows["earn"].iloc[i]) <= 1e-12:
            key = (a_rows["ts"].iloc[i], "A")
            assert np.isnan(pl_out.loc[key])
            assert np.isnan(sql_out.loc[key])
    pd.testing.assert_series_equal(ref, pl_out, check_names=False, rtol=1e-9, atol=1e-12)
    pd.testing.assert_series_equal(ref, sql_out, check_names=False, rtol=1e-9, atol=1e-12)
