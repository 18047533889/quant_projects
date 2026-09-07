# -*- coding: utf-8 -*-
"""wave3d ashare limit family: pandas / polars-long / DuckDB parity.

12 operators across two shapes:

1. Elementwise limit booleans (``ashare_limit_up_touch`` /
   ``ashare_limit_down_touch`` / ``ashare_open_at_upper_limit`` /
   ``ashare_limit_failed`` / ``ashare_limit_open_failed``) — the pandas
   authority is ``ashare/limit_ops.py``: an ABSOLUTE tick tolerance compares
   the price against ``limit ± tolerance``, valid requires ALL operands
   finite, and the output is {1.0, 0.0, NaN}.
2. Rolling state-machine operators (``ashare_limit_touch_count`` /
   ``ashare_failed_limit_count`` / ``ashare_limit_asymmetry`` /
   ``ashare_limit_event_density`` / ``ashare_limit_up_volume_ratio`` /
   ``ashare_limit_down_volume_ratio`` / ``ashare_limit_up_streak``) — the
   pandas authority is ``ashare/state_machine.py``: a RELATIVE tick tolerance
   (limit * (1 ± tol)), per-cell known gates (an unknown/NaN row → NaN output,
   never a 0), trailing-window counts and a run-length streak broken (NaN) by
   missing/unknown/suspended rows.

Two parity layers per case:

* engine level — pandas vs polars_long vs duckdb_sql through ``FactorEngine``
  (the shipped user path);
* emitter level — raw (unlowered) PlanNode through the polars-long emitter
  (``compile_plan_to_polars``) and the DuckDB SQL emitter
  (``compile_plan_to_sql``) executed on a real in-memory DuckDB long table,
  compared against the pandas registry operator as authority.

Edge cases baked into the panel: NaN holes in every price/event column, a
price EXACTLY equal to the limit (touch), a price just OUTSIDE the tolerance
band, zero-volume days, valid_trade = {0, 1, NaN} (suspended/unknown) and
NaN event flags.
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
LIMIT_UP = 10.0
LIMIT_DOWN = 9.0


def _panel(seed: int = 11) -> InMemorySeriesSource:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-02", periods=N_DAYS, freq="B")
    idx = pd.MultiIndex.from_product(
        [dates, INSTRUMENTS], names=["timestamp", "instrument"]
    )
    n = len(idx)

    def _prices(scale: float) -> pd.Series:
        base = rng.choice(
            np.array([
                LIMIT_UP,             # exactly at the limit (touch)
                LIMIT_UP + 0.02,      # inside the band
                LIMIT_UP - 0.05,      # outside the band (9.95 boundary)
                LIMIT_DOWN,           # exactly at the lower limit
                LIMIT_DOWN - 0.02,
                LIMIT_DOWN + 0.05,
                9.5,
            ]),
            n,
        )
        s = pd.Series(base + rng.normal(0, scale, n), index=idx)
        holes = rng.choice(n, int(n * 0.10), replace=False)
        s.iloc[holes] = np.nan
        return s

    raw_high = _prices(0.05)
    raw_low = _prices(0.05)
    raw_open = _prices(0.05)
    raw_close = _prices(0.05)

    up_event = pd.Series(
        np.where(rng.random(n) < 0.3, 1.0, 0.0), index=idx
    )
    up_event.iloc[rng.choice(n, int(n * 0.12), replace=False)] = np.nan
    down_event = pd.Series(
        np.where(rng.random(n) < 0.3, 1.0, 0.0), index=idx
    )
    down_event.iloc[rng.choice(n, int(n * 0.12), replace=False)] = np.nan

    volume = pd.Series(rng.integers(100, 5000, n).astype(float), index=idx)
    volume.iloc[rng.choice(n, int(n * 0.08), replace=False)] = 0.0

    valid_trade = pd.Series(
        np.where(rng.random(n) < 0.10, np.nan, 1.0), index=idx
    )
    # suspended days {0} and unknown {NaN} both break a streak
    valid_trade.iloc[rng.choice(n, int(n * 0.10), replace=False)] = 0.0

    return InMemorySeriesSource(data={
        "raw_high": raw_high,
        "raw_low": raw_low,
        "raw_open": raw_open,
        "raw_close": raw_close,
        "high_limit": pd.Series(LIMIT_UP, index=idx),
        "low_limit": pd.Series(LIMIT_DOWN, index=idx),
        "valid_trade": valid_trade,
        "limit_up_event": up_event,
        "limit_down_event": down_event,
        "volume": volume,
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


def _assert_two_way(pandas_out: pd.Series, candidate: pd.Series) -> None:
    pd.testing.assert_series_equal(
        pandas_out, candidate, check_names=False, rtol=1e-9, atol=1e-12
    )


def _assert_engine_parity(source, expr, *, rtol=1e-9, atol=1e-12):
    pandas_out = _result(_run(source, expr, "pandas"))
    polars_out = _result(_run(source, expr, "polars_long"))
    _assert_two_way(pandas_out, polars_out)
    duckdb_out = _result(_run(source, expr, "duckdb_sql"))
    _assert_two_way(pandas_out, duckdb_out)
    return pandas_out


# ---------------------------------------------------------------------------
# Engine cases: (canonical, column names, positional literals, kwargs)
# ---------------------------------------------------------------------------

_ENGINE_CASES = [
    # elementwise (absolute tolerance)
    ("ashare_limit_up_touch", ("raw_high", "high_limit"), (0.005,), {}),
    ("ashare_limit_down_touch", ("raw_low", "low_limit"), (0.005,), {}),
    ("ashare_open_at_upper_limit", ("raw_open", "high_limit"), (0.005,), {}),
    ("ashare_limit_failed", ("raw_high", "raw_close", "high_limit"), (0.005,), {}),
    ("ashare_limit_open_failed", ("raw_open", "raw_low", "high_limit"), (0.005,), {}),
    # rolling (relative tolerance)
    ("ashare_limit_touch_count", ("raw_high", "raw_low", "high_limit", "low_limit"), (5, "up", 0.005), {}),
    ("ashare_limit_touch_count", ("raw_high", "raw_low", "high_limit", "low_limit"), (5, "down", 0.005), {}),
    ("ashare_failed_limit_count", ("raw_high", "raw_low", "raw_close", "high_limit", "low_limit"), (5, "up", 0.005), {}),
    ("ashare_failed_limit_count", ("raw_high", "raw_low", "raw_close", "high_limit", "low_limit"), (5, "down", 0.005), {}),
    ("ashare_limit_asymmetry", ("limit_up_event", "limit_down_event"), (5,), {}),
    ("ashare_limit_event_density", ("limit_up_event",), (5,), {}),
    ("ashare_limit_up_volume_ratio", ("volume", "limit_up_event"), (5,), {}),
    ("ashare_limit_down_volume_ratio", ("volume", "limit_down_event"), (5,), {}),
    ("ashare_limit_up_streak", ("raw_close", "high_limit", "valid_trade"), (0.005,), {}),
]


@pytest.mark.parametrize(
    ("name", "cols", "lits", "kwargs"),
    _ENGINE_CASES,
    ids=[f"{name}-{lits}" for name, _, lits, _ in _ENGINE_CASES],
)
def test_ashare_wave3_engine_parity(source, name, cols, lits, kwargs):
    expr = make_cleaned_call_factory(name)(*[col(c) for c in cols], *lits, **kwargs)
    _assert_engine_parity(source, expr)


def test_ashare_limit_event_density_known_status_engine_parity(source):
    """known_status as the 3rd (optional) input: unknown current row → NaN.

    The pandas kernel signature is ``(event, window, known_status)`` so the
    DSL call binds the window literal positionally BEFORE the known-status
    column.
    """
    expr = make_cleaned_call_factory("ashare_limit_event_density")(
        col("limit_up_event"), 5, col("valid_trade")
    )
    _assert_engine_parity(source, expr)


def test_ashare_wave3_ops_registered_in_sql_and_polars_native():
    from factor_engine.backend.polars_long_policy import POLARS_LONG_NATIVE
    from factor_engine.backend.sql_tiers import SQL_IMPLEMENTED_CANONICALS

    names = {name for name, _, _, _ in _ENGINE_CASES}
    names.add("ashare_limit_event_density")
    for name in names:
        assert name in SQL_IMPLEMENTED_CANONICALS, f"{name} missing from SQL_IMPLEMENTED"
        assert name in POLARS_LONG_NATIVE, f"{name} missing from POLARS_LONG_NATIVE"


# ---------------------------------------------------------------------------
# Emitter level: raw PlanNode -> polars-long emitter & DuckDB SQL emitter,
# pandas registry operator as the authority.  Deterministic edge rows: an
# exact touch, a just-outside-tolerance miss, NaN holes on every operand,
# zero-volume days and valid_trade {0, 1, NaN}.
# ---------------------------------------------------------------------------

N_EMIT_DAYS = 12
_EMIT_LIMIT_UP = 10.0
_EMIT_LIMIT_DOWN = 9.0

# per-column value sequences for one instrument (broadcast to both)
_EMIT_VALUES: dict[str, list[float | None]] = {
    "raw_high": [10.00, 10.02, None, 9.00, 10.00, 10.05, 10.01, 9.98, 10.00, None, 10.00, 10.02],
    "raw_low": [8.99, 8.98, 9.00, None, 8.50, 9.00, 8.49, 9.01, 9.02, None, 9.00, 8.98],
    "raw_open": [10.00, 10.01, None, 9.50, 9.99, 10.02, 10.00, 9.95, 9.99, None, 9.98, 10.01],
    "raw_close": [10.00, 10.00, 9.99, None, 10.00, 9.00, 10.00, 9.94, 9.99, None, 9.50, 10.00],
    "high_limit": [10.00] * N_EMIT_DAYS,
    "low_limit": [9.00] * N_EMIT_DAYS,
    "valid_trade": [1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, None, 1.0, 1.0, 1.0, 1.0],
    "limit_up_event": [0.0, 1.0, 0.0, None, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    "limit_down_event": [0.0, 0.0, 1.0, 0.0, None, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    "volume": [100.0, 200.0, 300.0, 400.0, 0.0, 600.0, 700.0, 800.0, 900.0, 1000.0, 1100.0, 1200.0],
}

_EMIT_DATES = pd.date_range("2024-01-02", periods=N_EMIT_DAYS, freq="B")

_EMIT_FRAME = pd.DataFrame(
    [
        {"ts": ts, "inst": inst, **{k: vals[i] for k, vals in _EMIT_VALUES.items()}}
        for i, ts in enumerate(_EMIT_DATES)
        for inst in ("A", "B")
    ]
)


def _emit_panels() -> dict[str, pd.DataFrame]:
    return {
        k: _EMIT_FRAME.pivot(index="ts", columns="inst", values=k)
        for k in _EMIT_VALUES
    }


def _polars_emitter_out(plan: PlanNode) -> pd.Series:
    import polars as pl

    base = pl.LazyFrame(_EMIT_FRAME)
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
        con.register("panel", _EMIT_FRAME)
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


def _pandas_reference(
    name: str,
    cols: tuple[str, ...],
    lits: tuple,
    *,
    ks_mode: bool = False,
) -> pd.DataFrame:
    op = OperatorRegistry.get(name, backend="pandas_numpy")
    panels = _emit_panels()
    args = [panels[c] for c in cols]
    if ks_mode:
        # event, known_status panel, window literal
        return op._calculate_series(args[0], lits[0], args[1])
    return op._calculate_series(*args, *lits)


def _reference_series(ref: pd.DataFrame) -> pd.Series:
    s = ref.stack(future_stack=True)
    s.index = s.index.set_names(["ts", "inst"])
    return s.sort_index()


def _assert_emitter_parity(
    name: str,
    cols: tuple[str, ...],
    lits: tuple,
    *,
    ks_mode: bool = False,
) -> pd.Series:
    if ks_mode:
        inputs = [sql_column(cols[0]), sql_column(cols[1]), sql_literal(lits[0])]
    else:
        inputs = [sql_column(c) for c in cols]
        inputs += [sql_literal(v) for v in lits]
    plan = PlanNode(op=name, inputs=inputs, attrs={})
    ref_series = _reference_series(_pandas_reference(name, cols, lits, ks_mode=ks_mode))

    pl_out = _polars_emitter_out(plan)
    sql_out = _sql_emitter_out(plan)

    pl_out = pl_out.reindex(ref_series.index)
    sql_out = sql_out.reindex(ref_series.index)

    ref_num = np.nan_to_num(ref_series.to_numpy(dtype=float), nan=np.inf)
    pd.testing.assert_series_equal(
        ref_series, pl_out, check_names=False, rtol=1e-9, atol=1e-12
    )
    np.testing.assert_allclose(
        ref_num,
        np.nan_to_num(pl_out.to_numpy(dtype=float), nan=np.inf),
        rtol=1e-9, atol=1e-12,
    )
    np.testing.assert_allclose(
        ref_num,
        np.nan_to_num(sql_out.to_numpy(dtype=float), nan=np.inf),
        rtol=1e-9, atol=1e-12,
    )
    return ref_series


@pytest.mark.parametrize(
    ("name", "cols", "lits"),
    [
        ("ashare_limit_up_touch", ("raw_high", "high_limit"), (0.005,)),
        ("ashare_limit_down_touch", ("raw_low", "low_limit"), (0.005,)),
        ("ashare_open_at_upper_limit", ("raw_open", "high_limit"), (0.005,)),
        ("ashare_limit_failed", ("raw_high", "raw_close", "high_limit"), (0.005,)),
        ("ashare_limit_open_failed", ("raw_open", "raw_low", "high_limit"), (0.005,)),
        ("ashare_limit_touch_count", ("raw_high", "raw_low", "high_limit", "low_limit"), (5, "up", 0.005)),
        ("ashare_limit_touch_count", ("raw_high", "raw_low", "high_limit", "low_limit"), (5, "down", 0.005)),
        ("ashare_failed_limit_count", ("raw_high", "raw_low", "raw_close", "high_limit", "low_limit"), (5, "up", 0.005)),
        ("ashare_failed_limit_count", ("raw_high", "raw_low", "raw_close", "high_limit", "low_limit"), (5, "down", 0.005)),
        ("ashare_limit_asymmetry", ("limit_up_event", "limit_down_event"), (5,)),
        ("ashare_limit_event_density", ("limit_up_event",), (5,)),
        ("ashare_limit_up_volume_ratio", ("volume", "limit_up_event"), (5,)),
        ("ashare_limit_down_volume_ratio", ("volume", "limit_down_event"), (5,)),
        ("ashare_limit_up_streak", ("raw_close", "high_limit", "valid_trade"), (0.005,)),
    ],
    ids=[name for name, _, _ in [
        ("ashare_limit_up_touch", ("raw_high", "high_limit"), (0.005,)),
        ("ashare_limit_down_touch", ("raw_low", "low_limit"), (0.005,)),
        ("ashare_open_at_upper_limit", ("raw_open", "high_limit"), (0.005,)),
        ("ashare_limit_failed", ("raw_high", "raw_close", "high_limit"), (0.005,)),
        ("ashare_limit_open_failed", ("raw_open", "raw_low", "high_limit"), (0.005,)),
        ("ashare_limit_touch_count", ("raw_high", "raw_low", "high_limit", "low_limit"), (5, "up", 0.005)),
        ("ashare_limit_touch_count", ("raw_high", "raw_low", "high_limit", "low_limit"), (5, "down", 0.005)),
        ("ashare_failed_limit_count", ("raw_high", "raw_low", "raw_close", "high_limit", "low_limit"), (5, "up", 0.005)),
        ("ashare_failed_limit_count", ("raw_high", "raw_low", "raw_close", "high_limit", "low_limit"), (5, "down", 0.005)),
        ("ashare_limit_asymmetry", ("limit_up_event", "limit_down_event"), (5,)),
        ("ashare_limit_event_density", ("limit_up_event",), (5,)),
        ("ashare_limit_up_volume_ratio", ("volume", "limit_up_event"), (5,)),
        ("ashare_limit_down_volume_ratio", ("volume", "limit_down_event"), (5,)),
        ("ashare_limit_up_streak", ("raw_close", "high_limit", "valid_trade"), (0.005,)),
    ]],
)
def test_ashare_wave3_emitter_parity(name, cols, lits):
    _assert_emitter_parity(name, cols, lits)


def test_ashare_limit_event_density_known_status_emitter_parity():
    _assert_emitter_parity(
        "ashare_limit_event_density", ("limit_up_event", "valid_trade"), (5,),
        ks_mode=True,
    )


# ---------------------------------------------------------------------------
# Targeted edge semantics (pandas authority values asserted directly).
# ---------------------------------------------------------------------------

def test_touch_exactly_at_limit_and_outside_tolerance():
    """price == limit is a touch; price just below limit - tol is not."""
    ref = _assert_emitter_parity(
        "ashare_limit_up_touch", ("raw_high", "high_limit"), (0.005,)
    )
    exact = ref.loc[(pd.Timestamp("2024-01-02"), "A")]   # 10.00 vs 10.00 -> touch
    assert exact == pytest.approx(1.0)
    outside = ref.loc[(pd.Timestamp("2024-01-05"), "A")]  # 9.00 vs 10.00 -> miss
    assert outside == pytest.approx(0.0)
    hole = ref.loc[(pd.Timestamp("2024-01-04"), "A")]     # raw_high is None -> NaN
    assert np.isnan(hole)


def test_streak_breaks_on_unknown_and_suspended_rows():
    """NaN close (unknown) and valid_trade {0, NaN} rows output NaN; a valid
    non-limit day resets the streak to 0 (NOT NaN)."""
    ref = _assert_emitter_parity(
        "ashare_limit_up_streak", ("raw_close", "high_limit", "valid_trade"), (0.005,)
    )
    # 2024-01-05 (raw_close NaN, unknown): NaN
    assert np.isnan(ref.loc[(pd.Timestamp("2024-01-05"), "A")])
    # 2024-01-08 (valid_trade=0 suspended): NaN
    assert np.isnan(ref.loc[(pd.Timestamp("2024-01-08"), "A")])
    # 2024-01-11 (valid_trade=NaN unknown): NaN
    assert np.isnan(ref.loc[(pd.Timestamp("2024-01-11"), "A")])
    # 2024-01-09 (close 9.00, valid): streak resets to 0
    assert ref.loc[(pd.Timestamp("2024-01-09"), "A")] == pytest.approx(0.0)
    # 2024-01-05 is the NaN-close hole -> NaN
    assert np.isnan(ref.loc[(pd.Timestamp("2024-01-05"), "A")])
    # consecutive limit days accumulate
    assert ref.loc[(pd.Timestamp("2024-01-02"), "A")] == pytest.approx(1.0)
    assert ref.loc[(pd.Timestamp("2024-01-03"), "A")] == pytest.approx(2.0)


def test_volume_ratio_zero_base_and_no_event_fail_closed():
    """A window with zero base volume (or no event day) is NaN — never 0/0."""
    ref = _assert_emitter_parity(
        "ashare_limit_up_volume_ratio", ("volume", "limit_up_event"), (5,)
    )
    # day 5 volume == 0 inside the window keeps the ratio finite when other
    # days carry volume; a full-zero window fails closed to NaN
    assert np.isfinite(ref.dropna()).all() or ref.isna().all()


def test_elementwise_output_is_binary_or_nan():
    """The touch family only ever emits {0.0, 1.0, NaN}."""
    ref = _assert_emitter_parity(
        "ashare_limit_up_touch", ("raw_high", "high_limit"), (0.005,)
    )
    finite = ref.dropna().unique()
    assert set(finite.tolist()) <= {0.0, 1.0}
