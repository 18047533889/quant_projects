# -*- coding: utf-8
"""wave3e cs/group family: pandas / polars-long / DuckDB parity.

5 operators promoted to native polars_long + DuckDB SQL branches, plus one
honest DEFER:

1. ``cs_bucket_fixed`` — fixed-boundary bucketing (``overhaul.daily.pd_cs_bucket_fixed``):
   ``1 + count(breaks < x)`` elementwise; non-finite x -> NaN; non-strictly
   increasing breaks raise in the reference and the branches fall back.
2. ``cs_empirical_bayes_shrinkage`` — row-wise EB shrinkage toward the
   cross-section mean (``cs_batch1.CsEmpiricalBayesShrinkage``):
   valid = finite(est) & finite(se) & se > 0; breadth < ``_MIN_BREADTH`` (10)
   -> whole row NaN; ``shrinkage_factor < 0`` -> fail-closed all-NaN;
   ``cs_var`` (ddof=1) <= 0 -> shrink to the mean; else
   ``mean + (x - mean) * clip(1 / (1 + lambda * se^2 / cs_var), 0, 1)``.
3. ``group_ex_self_weighted_mean`` — group weighted mean excluding self
   (``group_ext.GroupExSelfWeightedMean``): ``(Σw·x - w_j·x_j) / (Σw - w_j)``
   per (ts, group) over members with finite x, finite w, w >= 0.  The
   group-constant sums make this a pure over/window expression — no
   self-join needed.
4. ``group_feature_valid_member_count`` — per (ts, group) count of members
   whose ALL THREE features are finite, broadcast to every member
   (``group_spectrum``); invalid group label -> NaN.
5. ``group_peer_deviation_index`` — sum of per-frame cross-sectional
   z-scores over the whole ts partition (the signature has NO group input),
   each frame with its OWN finite mask; frames with < 2 finite values or
   sd <= 1e-12 are skipped; cells missing from every contributing frame
   stay NaN — never a manufactured 0 (``cross_section.peer_ops``).

DEFERRED (stays on the pandas reference / audited bridge):
- ``cs_quantile_resid`` — the pandas authority
  (``cross_section.robust_cs._cs_quantile_resid``) solves an exact pinball-loss
  quantile regression via an LP (scipy ``linprog``/HiGHS) per row.  Neither a
  pure polars over-expression nor DuckDB window SQL can reproduce an interior-
  point LP solution faithfully, so no backend branch is claimed.

Two parity layers per case:

* engine level — pandas vs polars_long vs duckdb_sql through ``FactorEngine``
  (data_access-backed DuckDB fixture, ``test_csg_backend_gap_parity`` pattern);
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
from factor_engine.storage.factory import build_data_source
from tests.helpers import InMemorySeriesSource

load_all()


# ---------------------------------------------------------------------------
# Engine-level fixtures: 30 business days x 6 instruments, 2 groups, edge-rich
# (NaN holes, zero weights, negative weights, a NaN group label, single-member
# groups and a constant frame for the sd=0 skip path).
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

    def _series(base, scale, *, nan_frac=0.0, zero_frac=0.0):
        s = pd.Series(rng.normal(base, scale, n), index=idx)
        if nan_frac:
            holes = rng.choice(n, int(n * nan_frac), replace=False)
            s.iloc[holes] = np.nan
        if zero_frac:
            zeros = rng.choice(n, int(n * zero_frac), replace=False)
            s.iloc[zeros] = 0.0
        return s

    # two groups, three members each per day; group label 0.0 is INVALID
    # (matches a NaN label through the long-table float conversion) so the
    # NaN-label fail-closed contract is exercised on every backend.
    group = pd.Series(
        np.tile([1.0, 1.0, 1.0, 2.0, 2.0, 2.0], N_DAYS), index=idx
    )
    group.iloc[6:12] = 0.0  # one whole day with invalid labels for A..F block 1
    group.iloc[13] = 0.0  # single invalid cell (E, day 3)

    return InMemorySeriesSource(data={
        "x": _series(0.0, 3.0, nan_frac=0.05),
        "weight": _series(1.0, 0.3, nan_frac=0.04, zero_frac=0.05),
        "group_id": group,
        "f1": _series(0.0, 1.0, nan_frac=0.05),
        "f2": _series(0.0, 1.0, nan_frac=0.05),
        "f3": _series(0.0, 1.0, nan_frac=0.05),
        "d1": _series(0.0, 1.0, nan_frac=0.05),
        "d2": _series(0.0, 1.0, nan_frac=0.05),
        "d3": _series(0.0, 1.0, nan_frac=0.05),
        "d4": _series(0.0, 1.0, nan_frac=0.05),
        "d5": _series(0.0, 1.0, nan_frac=0.05),
        "xb": _series(0.0, 3.0, nan_frac=0.04),
        "est": _series(0.0, 1.0, nan_frac=0.05),
        "se": _series(0.5, 0.15, nan_frac=0.05),
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


def _assert_parity(source, expr, rtol=1e-9, atol=1e-12):
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
# Emitter-level fixtures: edge-heavy wide panel driven straight through the
# emitters, pandas registry operator as authority.
# ---------------------------------------------------------------------------

EMITTER_ROWS = [
    # ts, inst, x, w, g, f1, f2, f3, d1, d2, d3, d4, d5, est, se
    # day 0: plain positive row
    (0, "A", 2.0, 1.0, 1.0, 1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.5, 0.4),
    (0, "B", -3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 0.4, 0.5, 0.6, 0.7, 0.8, -0.5, 0.4),
    (0, "C", 0.0, 0.0, 1.0, 3.0, 4.0, 5.0, 0.7, 0.8, 0.9, 1.0, 1.1, 0.9, 0.4),
    (0, "D", 5.0, 1.5, 2.0, 4.0, 5.0, 6.0, 1.0, 1.1, 1.2, 1.3, 1.4, 0.2, 0.6),
    (0, "E", 7.0, 0.5, 2.0, 5.0, 6.0, 7.0, 1.3, 1.4, 1.5, 1.6, 1.7, -0.2, 0.6),
    (0, "F", -1.0, 2.5, 2.0, 6.0, 7.0, 8.0, 1.6, 1.7, 1.8, 1.9, 2.0, 1.1, 0.6),
    # day 1: NaN / Inf / zero-weight / invalid-group edges
    (1, "A", np.nan, 1.0, 1.0, np.nan, 2.0, 3.0, np.nan, 0.2, 0.3, 0.4, 0.5, np.nan, 0.4),
    (1, "B", 4.0, np.nan, 1.0, 2.0, np.nan, 4.0, 0.4, np.nan, 0.6, 0.7, 0.8, 0.3, np.nan),
    (1, "C", np.inf, 1.0, 1.0, 3.0, 4.0, np.nan, 0.7, 0.8, np.nan, 1.0, 1.1, 0.1, 0.4),
    (1, "D", -2.0, 0.0, 2.0, 4.0, 5.0, 6.0, 1.0, 1.1, 1.2, np.nan, 1.4, 0.8, 0.6),
    (1, "E", 1.0, -0.5, 2.0, 5.0, 6.0, 7.0, 1.3, 1.4, 1.5, 1.6, np.nan, -0.8, 0.6),
    (1, "F", 6.0, 1.0, 2.0, 6.0, 7.0, 8.0, 1.6, 1.7, 1.8, 1.9, 2.0, 0.0, 0.6),
    # day 2: constant cross-section in one frame (sd == 0 -> skip) and a
    # single-finite frame (< 2 finite -> skip)
    (2, "A", 3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 1.0, np.nan, 0.2, 0.4, 0.4, 0.3),
    (2, "B", 3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 1.0, np.nan, 0.6, 0.2, -0.4, 0.3),
    (2, "C", 3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 1.0, np.nan, 0.1, 0.9, 0.6, 0.3),
    (2, "D", 3.0, 1.0, 2.0, 1.0, 1.0, 1.0, 0.5, 1.0, np.nan, 0.3, 0.7, 0.2, 0.3),
    (2, "E", 3.0, 1.0, 2.0, 1.0, 1.0, 1.0, 0.5, 1.0, np.nan, 0.8, 0.1, -0.6, 0.3),
    (2, "F", 3.0, 1.0, 2.0, 1.0, 1.0, 1.0, 0.5, 1.0, np.nan, 0.5, 0.5, 0.9, 0.3),
]

_EMITTER_TS = [pd.Timestamp("2024-01-02") + pd.Timedelta(days=r[0]) for r in EMITTER_ROWS]
_EMITTER_FRAME = pd.DataFrame(
    {"ts": _EMITTER_TS,
     "inst": [r[1] for r in EMITTER_ROWS],
     **{name: [r[i] for r in EMITTER_ROWS] for i, name in enumerate(
         ["x", "w", "g", "f1", "f2", "f3", "d1", "d2", "d3", "d4", "d5", "est", "se"], start=2)}}
)


def _wide(frame: pd.DataFrame, colname: str) -> pd.DataFrame:
    return frame.pivot(index="ts", columns="inst", values=colname)


def _panel_from_wide(wide: pd.DataFrame) -> pd.Series:
    s = wide.stack(future_stack=True)
    s.index = s.index.set_names(["timestamp", "instrument"])
    return s.sort_index()


def _pandas_reference(name: str, columns: tuple[str, ...], extra: dict | None = None):
    op = OperatorRegistry.get(name, backend="pandas_numpy")
    panels = [_wide(_EMITTER_FRAME, c) for c in columns]
    return op.calculate(*panels, **(extra or {}))


def _emitter_plan(name: str, columns: tuple[str, ...], extra_literal=None) -> PlanNode:
    inputs = [sql_column(c) for c in columns]
    if extra_literal is not None:
        inputs.append(sql_literal(extra_literal))
    return PlanNode(op=name, inputs=inputs, attrs={})


def _polars_emitter_out(plan: PlanNode) -> pd.Series:
    base = _EMITTER_FRAME[["ts", "inst"] + [c for c in _EMITTER_FRAME.columns
                                            if c not in {"ts", "inst"}]].copy()
    res = compile_plan_to_polars(plan, pl.LazyFrame(_EMITTER_FRAME))
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


def _assert_emitter_parity(name, columns, extra=None, extra_literal=None):
    ref = _panel_from_wide(_pandas_reference(name, columns, extra))
    plan = _emitter_plan(name, columns, extra_literal)
    pl_out = _polars_emitter_out(plan)
    sql_out = _sql_emitter_out(plan)
    pd.testing.assert_series_equal(ref, pl_out, check_names=False, rtol=1e-9, atol=1e-12)
    pd.testing.assert_series_equal(ref, sql_out, check_names=False, rtol=1e-9, atol=1e-12)


import polars as pl  # noqa: E402  (after importorskip guard)


# ---------------------------------------------------------------------------
# Engine-level 3-way parity.
# ---------------------------------------------------------------------------

ENGINE_CASES = [
    ("cs_bucket_fixed", lambda: make_cleaned_call_factory("cs_bucket_fixed")(col("xb"), breaks=[-1.0, 1.0, 5.0])),
    ("cs_empirical_bayes_shrinkage", lambda: make_cleaned_call_factory("cs_empirical_bayes_shrinkage")(col("est"), col("se"), 0.7)),
    ("group_ex_self_weighted_mean", lambda: make_cleaned_call_factory("group_ex_self_weighted_mean")(col("x"), col("weight"), col("group_id"))),
    ("group_feature_valid_member_count", lambda: make_cleaned_call_factory("group_feature_valid_member_count")(col("f1"), col("f2"), col("f3"), col("group_id"))),
    ("group_peer_deviation_index", lambda: make_cleaned_call_factory("group_peer_deviation_index")(col("d1"), col("d2"), col("d3"), col("d4"), col("d5"))),
]


@pytest.mark.parametrize(
    ("name", "expr_builder"),
    ENGINE_CASES,
    ids=[name for name, _ in ENGINE_CASES],
)
def test_csgrp_wave3_engine_parity(source, name, expr_builder):
    _assert_parity(source, expr_builder())


# ---------------------------------------------------------------------------
# Emitter-level parity with edge rows (NaN/Inf/zero-weight/invalid-group/
# constant and single-finite frames).
# ---------------------------------------------------------------------------


def test_emitter_parity_bucket_fixed():
    _assert_emitter_parity("cs_bucket_fixed", ("x",), extra={"breaks": [-1.0, 1.0, 5.0]}, extra_literal=[-1.0, 1.0, 5.0])


def test_emitter_parity_empirical_bayes():
    _assert_emitter_parity("cs_empirical_bayes_shrinkage", ("est", "se"), extra={"shrinkage_factor": 0.7}, extra_literal=0.7)


def test_emitter_parity_ex_self_weighted_mean():
    _assert_emitter_parity("group_ex_self_weighted_mean", ("x", "w", "g"))


def test_emitter_parity_valid_member_count():
    _assert_emitter_parity("group_feature_valid_member_count", ("f1", "f2", "f3", "g"))


def test_emitter_parity_peer_deviation_index():
    _assert_emitter_parity("group_peer_deviation_index", ("d1", "d2", "d3", "d4", "d5"))


# ---------------------------------------------------------------------------
# Semantic edge contracts.
# ---------------------------------------------------------------------------


def test_bucket_fixed_boundary_semantics():
    """x <= b[0] -> 1, (b[i], b[i+1]] -> i+2, x > b[-1] -> len+1, NaN/Inf -> NaN."""
    plan = _emitter_plan("cs_bucket_fixed", ("x",), extra_literal=[-1.0, 1.0, 5.0])
    pl_out = _polars_emitter_out(plan)
    sql_out = _sql_emitter_out(plan)
    day0 = pd.Timestamp("2024-01-02")
    for out in (pl_out, sql_out):
        # A x=2.0: searchsorted([-1,1,5], 2, right)=2 -> bucket 3
        assert out.loc[(day0, "A")] == 3.0
        # C x=0.0: (-1, 1] -> bucket 2
        assert out.loc[(day0, "C")] == 2.0
        # D x=5.0: exactly on the last break -> NEXT bucket (4); right-bisect
        # places equal elements after the break.
        assert out.loc[(day0, "D")] == 4.0
        # E x=7.0: > 5 -> bucket 4
        assert out.loc[(day0, "E")] == 4.0
        # B x=-3.0: <= -1 -> bucket 1
        assert out.loc[(day0, "B")] == 1.0
        # F x=-1.0: exactly ON the break -> NEXT bucket (2); bucket i holds
        # (b[i-1], b[i]] in reference terms — (b[i-1], b[i]] via right-bisect.
        assert out.loc[(day0, "F")] == 2.0


def test_bucket_fixed_nonfinite_is_nan():
    plan = _emitter_plan("cs_bucket_fixed", ("x",), extra_literal=[-1.0, 1.0, 5.0])
    day1 = pd.Timestamp("2024-01-03")
    pl_out = _polars_emitter_out(plan)
    sql_out = _sql_emitter_out(plan)
    for out in (pl_out, sql_out):
        assert np.isnan(out.loc[(day1, "A")])  # x NaN
        assert np.isnan(out.loc[(day1, "C")])  # x +Inf


def test_bucket_fixed_invalid_breaks_fall_back():
    """Non-strictly-increasing breaks raise in the reference -> no native branch."""
    plan = PlanNode(op="cs_bucket_fixed", inputs=[sql_column("x"), sql_literal([1.0, 1.0])], attrs={})
    res = compile_plan_to_polars(plan, pl.LazyFrame(_EMITTER_FRAME))
    assert res is None, "non-increasing breaks must not compile natively"


def test_eb_negative_shrinkage_fail_closed():
    """shrinkage_factor < 0 -> the whole panel is NaN (fail-closed)."""
    plan = _emitter_plan("cs_empirical_bayes_shrinkage", ("est", "se"), extra_literal=-1.0)
    pl_out = _polars_emitter_out(plan)
    sql_out = _sql_emitter_out(plan)
    for out in (pl_out, sql_out):
        assert out.notna().sum() == 0


def test_eb_min_breadth_gate():
    """A cross-section with < 10 valid (est, se>0) pairs -> whole row NaN."""
    tiny = _EMITTER_FRAME.iloc[:6].copy()  # day 0 only: 6 instruments < 10
    op = OperatorRegistry.get("cs_empirical_bayes_shrinkage", backend="pandas_numpy")
    ref = op.calculate(
        _wide(tiny, "est"), _wide(tiny, "se"), shrinkage_factor=0.7
    )
    assert ref.isna().all().all()
    plan = _emitter_plan("cs_empirical_bayes_shrinkage", ("est", "se"), extra_literal=0.7)
    pl_out = _polars_emitter_out(plan)
    day0 = pd.Timestamp("2024-01-02")
    assert pl_out.loc[[k for k in pl_out.index if k[0] == day0]].isna().all()


def test_eb_degenerate_variance_shrinks_to_mean():
    """cs_var == 0 -> every valid cell shrinks exactly to the cross-section mean."""
    plan = _emitter_plan("cs_empirical_bayes_shrinkage", ("est", "se"), extra_literal=0.7)
    pl_out = _polars_emitter_out(plan)
    day2 = pd.Timestamp("2024-01-04")
    vals = pl_out.loc[[k for k in pl_out.index if k[0] == day2]]
    # est on day 2 is a constant 0.5 cross-section (5 finite + 1 NaN... all finite
    # here) -> shrunk value == the mean.
    assert vals.dropna().nunique() <= 1


def test_valid_member_count_invalid_group_label_is_nan():
    """A cell with an invalid group label keeps NaN (never joins a group)."""
    plan = _emitter_plan("group_feature_valid_member_count", ("f1", "f2", "f3", "g"))
    frame = _EMITTER_FRAME.copy()
    frame.loc[frame["ts"] == pd.Timestamp("2024-01-03"), "g"] = np.nan
    op = OperatorRegistry.get("group_feature_valid_member_count", backend="pandas_numpy")
    ref = op.calculate(
        frame.pivot(index="ts", columns="inst", values="f1"),
        frame.pivot(index="ts", columns="inst", values="f2"),
        frame.pivot(index="ts", columns="inst", values="f3"),
        frame.pivot(index="ts", columns="inst", values="g"),
    )
    assert ref.loc[pd.Timestamp("2024-01-03")].isna().all()

    # The polars emitter agrees: invalid labels are NULL in the long table.
    res = compile_plan_to_polars(plan, pl.LazyFrame(frame))
    out = res.frame.collect().to_pandas()
    s = out.set_index(["ts", "inst"])["_v"].sort_index()
    day1 = pd.Timestamp("2024-01-03")
    assert s.loc[[k for k in s.index if k[0] == day1]].isna().all()


def test_peer_deviation_index_all_missing_cell_stays_nan():
    """A cell missing from EVERY contributing frame stays NaN (no fake 0)."""
    plan = _emitter_plan("group_peer_deviation_index", ("d1", "d2", "d3", "d4", "d5"))
    frame = _EMITTER_FRAME.copy()
    day1 = pd.Timestamp("2024-01-03")
    mask = (frame["ts"] == day1) & (frame["inst"] == "C")
    for c in ("d1", "d2", "d3", "d4", "d5"):
        frame.loc[mask, c] = np.nan
    op = OperatorRegistry.get("group_peer_deviation_index", backend="pandas_numpy")
    ref = op.calculate(*[frame.pivot(index="ts", columns="inst", values=c)
                         for c in ("d1", "d2", "d3", "d4", "d5")])
    assert np.isnan(ref.loc[day1, "C"])

    # The polars emitter agrees — and the full panel still matches.
    res = compile_plan_to_polars(plan, pl.LazyFrame(frame))
    out = res.frame.collect().to_pandas()
    s = out.set_index(["ts", "inst"])["_v"].sort_index()
    ref_s = _panel_from_wide(ref)
    assert np.isnan(s.loc[(day1, "C")])
    pd.testing.assert_series_equal(ref_s, s, check_names=False, rtol=1e-9, atol=1e-12)


# ---------------------------------------------------------------------------
# Tier honesty: the 5 promoted ops are claimed, cs_quantile_resid is not.
# ---------------------------------------------------------------------------


def test_csgrp_wave3_ops_registered_in_sql_and_polars_native():
    from factor_engine.backend.polars_long_policy import POLARS_LONG_NATIVE
    from factor_engine.backend.sql_tiers import SQL_IMPLEMENTED_CANONICALS

    for name, _ in ENGINE_CASES:
        assert name in SQL_IMPLEMENTED_CANONICALS, f"{name} missing from SQL_IMPLEMENTED"
        assert name in POLARS_LONG_NATIVE, f"{name} missing from POLARS_LONG_NATIVE"


def test_cs_quantile_resid_honestly_deferred():
    """cs_quantile_resid is an exact pinball-loss LP per row — no native claim."""
    from factor_engine.backend.polars_long_policy import POLARS_LONG_NATIVE, infer_polars_long_tier
    from factor_engine.backend.sql_tiers import SQL_IMPLEMENTED_CANONICALS

    name = "cs_quantile_resid"
    assert name not in SQL_IMPLEMENTED_CANONICALS
    assert name not in POLARS_LONG_NATIVE
    assert infer_polars_long_tier(name) != "native"


# ---------------------------------------------------------------------------
# Engine-level DuckDB (data_access-backed) parity — same fixture pattern as
# tests/backend_parity/test_csg_backend_gap_parity.py.
# ---------------------------------------------------------------------------


def _write_duckdb_registry(tmp_path: Path, root: Path) -> Path:
    content = f"""
test_csgrp_wave3:
  kind: static
  access_mode: published
  layout: plain
  hive_partitioning: false
  union_by_name: true
  root: {root}
  glob: "**/*.parquet"
  time_column: TradeDate
  instrument_column: Symbol
  schema:
    TradeDate: date
    Symbol: string
    xb: double
    est: double
    se: double
    x: double
    weight: double
    group_id: double
    f1: double
    f2: double
    f3: double
    d1: double
    d2: double
    d3: double
    d4: double
    d5: double
"""
    path = tmp_path / "datasets.yaml"
    path.write_text(content.strip() + "\n", encoding="utf-8")
    return path


def _seed_duckdb_panel(root: Path, mem: InMemorySeriesSource) -> None:
    root.mkdir(parents=True, exist_ok=True)
    index = mem.data["x"].index
    rows = []
    for (ts, sym) in index:
        row = {"TradeDate": ts.date(), "Symbol": sym}
        for name, column_name in (
            ("x", "x"), ("weight", "weight"), ("group_id", "group_id"),
            ("f1", "f1"), ("f2", "f2"), ("f3", "f3"),
            ("d1", "d1"), ("d2", "d2"), ("d3", "d3"), ("d4", "d4"), ("d5", "d5"),
            ("xb", "xb"), ("est", "est"), ("se", "se"),
        ):
            value = mem.data[name].loc[(ts, sym)]
            row[column_name] = float(value) if pd.notna(value) else None
        rows.append(row)
    pd.DataFrame(rows).to_parquet(root / "panel.parquet")


from pathlib import Path  # noqa: E402


@pytest.fixture
def duckdb_source(tmp_path, monkeypatch, source):
    monkeypatch.setenv("DATA_ACCESS_SKIP_COS_MIRROR", "1")
    monkeypatch.setenv(
        "DATA_ACCESS_CONFIG", str(_write_duckdb_registry(tmp_path, tmp_path / "data"))
    )
    _seed_duckdb_panel(tmp_path / "data", source)
    try:
        from data_access import reset_store

        reset_store()
    except ImportError:
        pass
    return build_data_source({"type": "data_access", "dataset": "test_csgrp_wave3"})


@pytest.mark.parametrize(
    ("name", "expr_builder"),
    ENGINE_CASES,
    ids=[name for name, _ in ENGINE_CASES],
)
def test_csgrp_wave3_duckdb_matches_pandas(source, duckdb_source, name, expr_builder):
    expr = expr_builder()
    pd_out = _result(_run(source, expr, "pandas"))
    sql_run = _run(duckdb_source, expr, "duckdb_sql")
    from tests.backend_parity.duckdb_parity_helpers import assert_duckdb_real_sql_execution

    assert_duckdb_real_sql_execution(sql_run)
    sql_out = _result(sql_run)
    pd.testing.assert_series_equal(pd_out, sql_out, check_names=False, rtol=1e-6, atol=1e-6)
