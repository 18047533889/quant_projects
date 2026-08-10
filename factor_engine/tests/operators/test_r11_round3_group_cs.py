# -*- coding: utf-8 -*-
"""R11 audit round-3 regression tests for group / cross-sectional operators.

Covers the review items owned by this fixer:
- #6  group_decay_linear.window is a dead parameter -> new canonical
     ``group_rank_weighted_value`` (no window); group_decay_linear keeps
     ``window`` declared ``searchable=False / ParamRole.POLICY``.
- #7  group_decay_linear ties depend on column order -> average tie rank
     (ties within the same group get the same weight).
- #8  rename: group_decay_linear is really group_rank_weighted_value
     (cross-sectional rank weighting, not time decay); backward-compat alias.
- #9  aggr_top_n classified time_series but is cross-sectional routing ->
     category ``cross_sectional`` / business ``cross_sectional_routing``.
- #10 aggr_top_n invalid aggr_func must raise (never fall through to sum).
- #11 ts_argmax / ts_argmin dual parameter authority -> canonical ``window``
     only; ``d`` is a parser-level alias (param_aliases).

These tests import the owned modules directly instead of the full
``load_all()`` so they stay runnable while an unrelated concurrent session is
mid-edit on ``polars_statistics.py`` (a ``residual`` logical-contract
divergence that blocks the full registry load).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# Register the owned modules directly (order matters for ts_argmax: gtja_compat
# replaces the time_series pandas backend with the GTJA-compatible semantic).
import cleaned_operators.common.group  # noqa: F401
import cleaned_operators.common.group_polars  # noqa: F401
import cleaned_operators.common.time_series  # noqa: F401
import cleaned_operators.common.gtja_compat  # noqa: F401
import cleaned_operators.common.polars_ops  # noqa: F401
import cleaned_operators.common.polars_batch_mirror  # noqa: F401

from cleaned_operators.base import ParamRole
from cleaned_operators.common.gtja_compat import GTJATSArgmax, GTJATSArgmin
from cleaned_operators.common.polars_batch_mirror import AggrTopNPolars
from cleaned_operators.common.time_series import AggrTopN
from cleaned_operators.registry import OperatorRegistry


def _panel(values, columns, dates=None):
    if dates is None:
        dates = pd.date_range("2024-01-01", periods=len(values))
    return pd.DataFrame(values, index=dates, columns=columns)


# ---------------------------------------------------------------------------
# #6 / #8  group_rank_weighted_value canonical + group_decay_linear compat
# ---------------------------------------------------------------------------

def test_group_rank_weighted_value_is_canonical_without_window() -> None:
    op = OperatorRegistry.get("group_rank_weighted_value")
    # R24-094: fallback_policy is now a DECLARED behavioral param.
    assert op.metadata.param_names == ["x", "group", "fallback_policy"]
    assert "window" not in op.metadata.param_names


def test_group_decay_linear_window_dead_param_is_policy() -> None:
    op = OperatorRegistry.get("group_decay_linear")
    assert op.metadata.param_names == ["x", "group", "window", "fallback_policy"]
    spec = op.metadata.param_specs["window"]
    assert spec.searchable is False
    assert spec.param_role is ParamRole.POLICY


def test_group_decay_linear_backward_compat_alias_preserved() -> None:
    # The pre-existing honest alias must keep pointing at group_decay_linear.
    assert OperatorRegistry._aliases["group_rank_linear_weighted_value"] == "group_decay_linear"


def test_group_rank_weighted_value_matches_group_decay_linear() -> None:
    x = _panel([[10.0, 20.0, 30.0], [1.0, 2.0, 3.0]], ["A", "B", "C"])
    grp = _panel([["g1", "g1", "g2"], ["g1", "g1", "g2"]], ["A", "B", "C"], dates=x.index).astype(object)
    rw = OperatorRegistry.get("group_rank_weighted_value").calculate(x, grp)
    dl = OperatorRegistry.get("group_decay_linear").calculate(x, grp)
    pd.testing.assert_frame_equal(rw, dl, check_dtype=False)


def test_group_decay_linear_window_does_not_change_result() -> None:
    x = _panel([[10.0, 20.0, 30.0, 40.0], [1.0, 2.0, 3.0, 4.0]], ["A", "B", "C", "D"])
    grp = _panel([["g1", "g1", "g1", "g1"], ["g1", "g1", "g1", "g1"]], ["A", "B", "C", "D"], dates=x.index).astype(object)
    base = OperatorRegistry.get("group_decay_linear").calculate(x, grp, 5)
    for window in (10, 60, 120):
        other = OperatorRegistry.get("group_decay_linear").calculate(x, grp, window)
        pd.testing.assert_frame_equal(base, other, check_dtype=False)


def test_group_decay_linear_3arg_dsl_compat() -> None:
    # Existing formulas call group_decay_linear(x, group, 2); the extra window
    # positional is validated (integer) but does not change the result.
    x = _panel([[10.0, 20.0, 30.0], [1.0, 2.0, 3.0]], ["A", "B", "C"])
    grp = _panel([[1, 1, 2], [1, 1, 2]], ["A", "B", "C"], dates=x.index)
    out = OperatorRegistry.get("group_decay_linear").calculate(x, grp, 2)
    assert out.notna().any().any()


# ---------------------------------------------------------------------------
# #7  ties get the same weight (average tie rank)
# ---------------------------------------------------------------------------

def test_group_rank_weighted_value_ties_get_same_weight() -> None:
    # B and C have the same value 20 -> identical average rank -> identical
    # weight -> identical output (independent of column position).
    x = _panel([[10.0, 20.0, 20.0, 30.0]], ["A", "B", "C", "D"])
    grp = _panel([["g1", "g1", "g1", "g1"]], ["A", "B", "C", "D"], dates=x.index).astype(object)
    out = OperatorRegistry.get("group_rank_weighted_value").calculate(x, grp)
    row = out.iloc[0]
    assert row["B"] == pytest.approx(row["C"], abs=1e-12)
    # average ranks A=1, B=C=2.5, D=4; sum=10 -> weights 0.1 / 0.25 / 0.25 / 0.4
    assert row["A"] == pytest.approx(10.0 * 0.1, abs=1e-12)
    assert row["B"] == pytest.approx(20.0 * 0.25, abs=1e-12)
    assert row["D"] == pytest.approx(30.0 * 0.4, abs=1e-12)


def test_group_decay_linear_ties_same_weight() -> None:
    x = _panel([[10.0, 20.0, 20.0, 30.0]], ["A", "B", "C", "D"])
    grp = _panel([["g1", "g1", "g1", "g1"]], ["A", "B", "C", "D"], dates=x.index).astype(object)
    out = OperatorRegistry.get("group_decay_linear").calculate(x, grp, 5)
    assert out.iloc[0]["B"] == pytest.approx(out.iloc[0]["C"], abs=1e-12)


def test_group_rank_weighted_value_group_scoped() -> None:
    # Group g1 weights only within g1; g2 independently.
    x = _panel([[10.0, 20.0, 30.0, 60.0]], ["A", "B", "C", "D"])
    grp = _panel([["g1", "g1", "g2", "g2"]], ["A", "B", "C", "D"], dates=x.index).astype(object)
    out = OperatorRegistry.get("group_rank_weighted_value").calculate(x, grp)
    row = out.iloc[0]
    # g1: ranks 1,2 sum 3 -> weights 1/3, 2/3 -> 10/3, 40/3
    assert row["A"] == pytest.approx(10.0 / 3.0, abs=1e-12)
    assert row["B"] == pytest.approx(40.0 / 3.0, abs=1e-12)
    # g2: ranks 1,2 -> 30/3, 120/3
    assert row["C"] == pytest.approx(10.0, abs=1e-12)
    assert row["D"] == pytest.approx(40.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Polars twin parity for #6/#7/#8
# ---------------------------------------------------------------------------

pl = pytest.importorskip("polars")


def _pl_pair(pdf, gpdf):
    dates = pdf.index
    xpl = pl.DataFrame({"date": dates}).with_columns(
        [pl.Series(c, pdf[c].to_numpy()) for c in pdf.columns]
    )
    gpl = pl.DataFrame({"date": dates}).with_columns(
        [pl.Series(c, gpdf[c].to_numpy()) for c in gpdf.columns]
    )
    return xpl, gpl


def test_polars_group_rank_weighted_value_parity() -> None:
    x = _panel([[10.0, 20.0, 20.0, 30.0], [1.0, 2.0, 3.0, 4.0]], ["A", "B", "C", "D"])
    grp = _panel([[1, 1, 1, 1], [1, 1, 2, 2]], ["A", "B", "C", "D"], dates=x.index)
    pd_out = OperatorRegistry.get("group_rank_weighted_value").calculate(x, grp)
    xpl, gpl = _pl_pair(x, grp)
    pl_out = OperatorRegistry.get("group_rank_weighted_value", backend="polars").calculate(xpl, gpl)
    np.testing.assert_allclose(
        pd_out.to_numpy(),
        pl_out.select(["A", "B", "C", "D"]).to_numpy(),
        rtol=1e-12,
        atol=1e-12,
    )
    # ties -> same weight in polars too
    pl_row = pl_out.select(["A", "B", "C", "D"]).row(0)
    assert pl_row[1] == pytest.approx(pl_row[2], abs=1e-12)


def test_polars_group_decay_linear_window_invariance() -> None:
    x = _panel([[10.0, 20.0, 20.0, 30.0]], ["A", "B", "C", "D"])
    grp = _panel([[1, 1, 1, 1]], ["A", "B", "C", "D"], dates=x.index)
    xpl, gpl = _pl_pair(x, grp)
    op = OperatorRegistry.get("group_decay_linear", backend="polars")
    base = op.calculate(xpl, gpl, 5)
    for window in (10, 60):
        other = op.calculate(xpl, gpl, window)
        assert base.select(["A", "B", "C", "D"]).to_numpy().tolist() == \
            other.select(["A", "B", "C", "D"]).to_numpy().tolist()


def test_polars_group_decay_linear_window_is_policy() -> None:
    op = OperatorRegistry.get("group_decay_linear", backend="polars")
    spec = op.metadata.param_specs["window"]
    assert spec.searchable is False
    assert spec.param_role is ParamRole.POLICY


# ---------------------------------------------------------------------------
# #9 / #10  aggr_top_n cross-sectional routing + strict aggr_func
# ---------------------------------------------------------------------------

def test_aggr_top_n_category_is_cross_sectional_routing() -> None:
    # Test the class definition directly (layer_governance blocks aggr_top_n from
    # the final registry after a full load_all, so a registry lookup would be
    # None there; the owned-file contract is the class metadata).
    op = AggrTopN()
    assert op.metadata.category == "cross_sectional"
    assert op.metadata.business_category == "cross_sectional_routing"
    assert "routing" in op.metadata.tags or "cross_sectional" in op.metadata.tags


def test_aggr_top_n_invalid_aggr_func_raises() -> None:
    op = AggrTopN()
    x = _panel([[1.0, 3.0, 2.0]], ["A", "B", "C"])
    with pytest.raises(ValueError, match="aggr_func"):
        op.calculate("bogus", x, x, 2, True)


def test_aggr_top_n_routing_is_global_per_day() -> None:
    op = AggrTopN()
    x = _panel(
        [[1.0, 3.0, 2.0], [4.0, 2.0, 6.0]],
        ["A", "B", "C"],
    )
    sort_col = _panel(
        [[0.5, 0.2, 0.8], [1.0, 0.3, 0.7]],
        ["A", "B", "C"],
        dates=x.index,
    )
    out = op.calculate("sum", x, sort_col, 2, True)
    # day 0: ascending sort -> B(0.2), A(0.5) -> sum = 3 + 1 = 4 broadcast to B,A
    assert out.iloc[0]["A"] == pytest.approx(4.0)
    assert out.iloc[0]["B"] == pytest.approx(4.0)
    assert np.isnan(out.iloc[0]["C"])
    # day 1: ascending sort -> B(0.3), C(0.7) -> sum = 2 + 6 = 8 broadcast to B,C
    assert out.iloc[1]["B"] == pytest.approx(8.0)
    assert out.iloc[1]["C"] == pytest.approx(8.0)
    assert np.isnan(out.iloc[1]["A"])


@pytest.mark.parametrize("func", ["sum", "avg", "mean", "max", "min", "std", "count"])
def test_aggr_top_n_supported_funcs_run(func) -> None:
    op = AggrTopN()
    x = _panel([[1.0, 3.0, 2.0]], ["A", "B", "C"])
    out = op.calculate(func, x, x, 3, True)
    assert out.notna().any().any()


def test_aggr_top_n_polars_backend_parity_and_raise() -> None:
    x = _panel([[1.0, 3.0, 2.0], [4.0, 2.0, 6.0]], ["A", "B", "C"])
    sort_col = _panel([[0.5, 0.2, 0.8], [1.0, 0.3, 0.7]], ["A", "B", "C"], dates=x.index)
    xpl, spl = _pl_pair(x, sort_col)
    op = AggrTopNPolars()
    pd_out = AggrTopN().calculate("sum", x, sort_col, 2, True)
    pl_out = op.calculate("sum", xpl, spl, 2, True)
    np.testing.assert_allclose(
        pd_out.to_numpy(),
        pl_out.select(["A", "B", "C"]).to_numpy(),
        rtol=1e-12,
        atol=1e-12,
    )
    with pytest.raises(ValueError, match="aggr_func"):
        op.calculate("bogus", xpl, spl, 2, True)


# ---------------------------------------------------------------------------
# #11  ts_argmax / ts_argmin: canonical window + d alias
# ---------------------------------------------------------------------------

def test_ts_argmax_canonical_param_is_window() -> None:
    op = OperatorRegistry.get("ts_argmax", backend="pandas_numpy")
    # Canonical parameter is ``window``; ``d`` must not be an independent
    # parameter (the overhaul layer may add ``min_periods``, which is fine).
    assert "window" in op.metadata.param_names
    assert "d" not in op.metadata.param_names
    assert op.metadata.param_aliases.get("d") == "window"


def test_ts_argmin_canonical_param_is_window() -> None:
    op = OperatorRegistry.get("ts_argmin", backend="pandas_numpy")
    assert "window" in op.metadata.param_names
    assert "d" not in op.metadata.param_names
    assert op.metadata.param_aliases.get("d") == "window"


def test_ts_argmax_d_alias_maps_to_window() -> None:
    # Test the owned GTJA-compat class directly (the overhaul layer owns the
    # full-load ts_argmax kernel, which is outside this fixer's files).  Data
    # distinguishes window sizes: max=5 at idx1 -> window=3 sees max=3 (dist 1),
    # window=5 sees max=5 (dist 3).
    op = GTJATSArgmax()
    x = _panel([1.0, 5.0, 2.0, 3.0, 1.0], ["A"])
    by_position = op.calculate(x, 3)
    by_d = op.calculate(x, d=3)
    by_window = op.calculate(x, window=3)
    pd.testing.assert_frame_equal(by_position, by_d, check_dtype=False)
    pd.testing.assert_frame_equal(by_position, by_window, check_dtype=False)
    by_d5 = op.calculate(x, d=5)
    pd.testing.assert_frame_equal(by_d5, op.calculate(x, 5), check_dtype=False)
    assert by_position["A"].iloc[-1] != by_d5["A"].iloc[-1]


def test_ts_argmin_d_alias_maps_to_window() -> None:
    op = GTJATSArgmin()
    x = _panel([3.0, 1.0, 5.0, 4.0, 2.0], ["A"])
    by_position = op.calculate(x, 3)
    by_d = op.calculate(x, d=3)
    pd.testing.assert_frame_equal(by_position, by_d, check_dtype=False)


def test_ts_argmax_polars_twin_uses_window() -> None:
    op = OperatorRegistry.get("ts_argmax", backend="polars")
    assert "window" in op.metadata.param_names
    assert "d" not in op.metadata.param_names
    assert op.metadata.param_aliases.get("d") == "window"


def test_ts_argmin_polars_twin_uses_window() -> None:
    op = OperatorRegistry.get("ts_argmin", backend="polars")
    assert "window" in op.metadata.param_names
    assert "d" not in op.metadata.param_names
    assert op.metadata.param_aliases.get("d") == "window"


def test_ts_argmax_no_hidden_window_override() -> None:
    # Passing window positionally AND as a keyword must be rejected (Python-level
    # multiple-values TypeError) — the dual authority is gone.
    op = OperatorRegistry.get("ts_argmax", backend="pandas_numpy")
    x = _panel([1.0, 3.0, 2.0, 3.0], ["A"])
    with pytest.raises(TypeError):
        op.calculate(x, 3, window=9)
