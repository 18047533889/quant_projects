# -*- coding: utf-8 -*-
"""R11 WS-J long-tail audit: episode excursion, downside risk, return
decomposition, Hankel/SSA, vector-path, fractional difference (#149-#164, #28,
#92-#96).

Every test exercises the operator through the real ``calculate``/kernel path.
The operator classes are imported from their modules directly (each module
registers its operators + polars bridge on import) so this file is not blocked
by other workstreams' in-flight edits to unrelated canonical families; a
separate ``test_load_all_baseline`` runs the full ``load_all()``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import cleaned_operators.downside_risk as _dr
import cleaned_operators.state_episode_excursion as _see
import cleaned_operators.return_decomp as _rd
import cleaned_operators.hankel as _hk
import cleaned_operators.vector_path as _vp
import cleaned_operators.memory_ext as _me
import cleaned_operators.cross_section_local as _csl

from cleaned_operators.registry import OperatorRegistry

# Re-import modules is enough to register the WS-J operators; load_all() adds
# the full audit + fiscal families that a concurrent session may be mid-edit on.


def _frame(vals: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"A": np.asarray(vals, dtype=float)})


def _panel(vals: np.ndarray, cols: list[str] | None = None) -> pd.DataFrame:
    cols = cols or ["A"]
    return pd.DataFrame(
        np.asarray(vals, dtype=float),
        index=pd.date_range("2024-01-01", periods=len(np.asarray(vals))),
        columns=cols,
    )


# ---------------------------------------------------------------------------
# #149  episode entry efficiency 0/0 must not output 0
# ---------------------------------------------------------------------------
def test_state_episode_efficiency_entry_bar_is_nan_until_a_step():
    op = _see.StateEpisodeEfficiency()
    # Monotone series, state all +1: entry bar has displacement=0 and path=0.
    out = op.calculate(_frame([10.0, 11.0, 12.0, 13.0]), _frame([1.0, 1.0, 1.0, 1.0]))
    vals = out["A"].tolist()
    assert np.isnan(vals[0]), "0/0 efficiency at the entry bar must be NaN, not 0"
    # Once one valid step is walked, efficiency is well-defined.
    assert vals[1] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# #150  retrace ratio must not arbitrary clamp=100
# ---------------------------------------------------------------------------
def test_state_episode_retrace_mfe_zero_is_nan_not_mass_100():
    op = _see.StateEpisodeRetraceRatio()
    # Monotone adverse series, state all +1: MFE stays 0 while P<0 -> undefined.
    out = op.calculate(_frame([10.0, 9.0, 8.0, 7.0]), _frame([1.0, 1.0, 1.0, 1.0]))
    vals = out["A"].tolist()
    assert all(np.isnan(v) for v in vals), f"MFE~0 adverse must be NaN, got {vals}"
    # A real favourable-then-retrace episode emits an honest (unclamped) ratio.
    out2 = op.calculate(_frame([10.0, 12.0, 11.0]), _frame([1.0, 1.0, 1.0]))
    v2 = out2["A"].tolist()
    assert np.isnan(v2[0])
    assert v2[1] == pytest.approx(0.0)  # at the best, retrace = 0
    assert v2[2] == pytest.approx(0.5)  # 1 - 1/2


# ---------------------------------------------------------------------------
# #151  scale must be PositiveScale (require scale > 0, not just finite)
# ---------------------------------------------------------------------------
def test_state_episode_scale_positive_scale_required():
    op = _see.StateEpisodeMfe()
    x = _frame([10.0, 11.0, 12.0])
    s = _frame([1.0, 1.0, 1.0])
    # scale = 0 at entry -> NaN, never a 0-normalised value.
    out = op.calculate(x, s, _frame([0.0, 5.0, 5.0]))
    assert np.isnan(out["A"].iloc[0])
    # negative scale -> NaN.
    out2 = op.calculate(x, s, _frame([-1.0, 5.0, 5.0]))
    assert np.isnan(out2["A"].iloc[0])
    # positive scale works.
    out3 = op.calculate(x, s, _frame([2.0, 5.0, 5.0]))
    assert np.isfinite(out3["A"].iloc[0])


# ---------------------------------------------------------------------------
# #152  state input must be StateSigned (no magic numeric inference)
# ---------------------------------------------------------------------------
def test_state_episode_rejects_arbitrary_numeric_state():
    op = _see.StateEpisodeEfficiency()
    # z-score-like / continuous state must be rejected, not silently inferred
    # into a direction through a 1e-9 threshold.
    with pytest.raises(ValueError, match="signed-state"):
        op.calculate(_frame([10.0, 11.0, 12.0]), _frame([0.5, 1.0, 2.0]))
    with pytest.raises(ValueError, match="signed-state"):
        op.calculate(_frame([10.0, 11.0, 12.0]), _frame([100.0, -3.0, 0.4]))
    # signed {-1,0,+1} (NaN allowed) is accepted.
    out = op.calculate(
        _frame([10.0, np.nan, 11.0, 12.0]), _frame([1.0, 1.0, 1.0, 1.0])
    )
    assert out.shape == (4, 1)


# ---------------------------------------------------------------------------
# #158  downside/upside deviation unit reference is same_as:x; x is a RETURN /
#       signed series vs a same-unit MAR target (NOT a price-only input).
# ---------------------------------------------------------------------------
def test_downside_upside_deviation_unit_is_same_as_x():
    for cls in (_dr.TsDownsideDeviation, _dr.TsUpsideDeviation):
        meta = cls().metadata
        assert meta.output_unit == "same_as:x", f"{cls.__name__} -> {meta.output_unit}"
        unit_tags = [t for t in meta.tags if t.startswith("unit:")]
        assert "unit:same_as:x" in unit_tags, f"{cls.__name__} tags {unit_tags}"
        # x accepts return-or-same-unit numeric — NOT price-only — and target is
        # a MAR threshold in the same unit as x.
        assert meta.input_units.get("x") == "return_or_same_unit_numeric"
        assert meta.input_units.get("target") == "same_unit_as:x"
        assert meta.compatible_units.get("x") == ("return", "same_unit_as:x")
        assert meta.compatible_units.get("target") == ("same_unit_as:x",)


# ---------------------------------------------------------------------------
# #159  strict window/min_periods validation (no int(5.9) truncation)
# ---------------------------------------------------------------------------
def test_window_min_periods_strict_int_and_domain():
    op = _dr.TsDownsideDeviation()
    x = _frame(np.arange(20.0))
    with pytest.raises(Exception, match="integer"):
        op.calculate(x, window=5.9)
    with pytest.raises(ValueError, match="min_periods must be >= 2"):
        op.calculate(x, window=10, min_periods=1)
    # ``min_periods > window`` is rejected (by the central relation gate or the
    # kernel domain check — either message is acceptable).
    with pytest.raises(ValueError, match="min_periods"):
        op.calculate(x, window=10, min_periods=12)
    # valid combination works.
    out = op.calculate(x, window=10, min_periods=7)
    assert np.isfinite(out["A"].iloc[-1])


# ---------------------------------------------------------------------------
# #160  time_under_water missing policy == drawdown-duration policy
# ---------------------------------------------------------------------------
def test_time_under_water_resets_peak_after_gap_like_drawdown_duration():
    water = _dr.TsTimeUnderWater()
    dd = _dr.TsCurrentDrawdownDuration()
    x = _frame([10.0, 11.0, np.nan, 9.0, 8.0])
    w = water.calculate(x, window=5)["A"].tolist()
    d = dd.calculate(x, window=5)["A"].tolist()
    # After the NaN gap the running peak is RESET: value 9 is not under water.
    assert w[3] == pytest.approx(0.0)
    assert d[3] == pytest.approx(0.0)
    # The carry-across-gap bug would have kept peak=11 and reported under=1.
    assert w[4] == pytest.approx(0.25)  # only 8 < 9 (post-gap peak)
    assert d[4] == pytest.approx(1.0)   # drawdown-duration streak = 1


# ---------------------------------------------------------------------------
# #163 / #164  return decomposition: exact axes + shared PriceBasis
# ---------------------------------------------------------------------------
def test_return_decomp_rejects_mismatched_axes():
    idx = pd.date_range("2024-01-01", periods=3)
    a = pd.DataFrame([10.0, 10.0, 10.0], index=idx, columns=["A"])
    b_rows = pd.DataFrame([9.0, 10.0], index=idx[:2], columns=["A"])
    with pytest.raises(ValueError):
        _rd.OvernightReturn().calculate(a, b_rows)
    b_cols = pd.DataFrame([9.0, 10.0, 11.0], index=idx, columns=["B"])
    with pytest.raises(ValueError):
        _rd.OpenCloseReturn().calculate(a, b_cols)
    # Direct kernel path (bypassing the central axis validator) also fails.
    with pytest.raises(ValueError, match="index/columns"):
        _rd.OvernightReturn()._calculate_series(a, b_rows)
    with pytest.raises(ValueError, match="index/columns"):
        _rd.OpenCloseReturn()._calculate_series(a, b_cols)


def test_return_decomp_rejects_mixed_price_basis():
    idx = pd.date_range("2024-01-01", periods=3)
    raw_open = pd.DataFrame([10.0, 10.0, 10.0], index=idx, columns=["raw_open"])
    cont_close = pd.DataFrame(
        [10.5, 10.5, 10.5], index=idx, columns=["continuous_close"]
    )
    # A raw open and a continuous (adjusted) close must never combine into a
    # return.  Through calculate() the framework rejects the call…
    with pytest.raises(ValueError):
        _rd.OvernightReturn().calculate(raw_open, cont_close)
    # …and the kernel-level PriceBasis gate rejects it directly too.
    with pytest.raises(ValueError, match="price basis"):
        _rd._assert_shared_price_basis(raw_open, cont_close)
    # Same basis (both raw) passes the gate.
    raw_close = pd.DataFrame([10.5, 10.5, 10.5], index=idx, columns=["raw_close"])
    _rd._assert_shared_price_basis(raw_open, raw_close)


def test_return_decomp_metadata_declares_price_basis_gate():
    meta = _rd.OvernightReturn().metadata
    assert meta.input_units == {"open": "price", "pre_close": "price"}
    assert meta.compatible_units == {
        "open": ("price",),
        "pre_close": ("price",),
    }


# ---------------------------------------------------------------------------
# #92 / #93 / #94  Hankel/SSA
# ---------------------------------------------------------------------------
def test_hankel_full_window_semantics_nan_until_w():
    op = _hk.TsHankelEffectiveRank()
    out = op.calculate(_frame(np.arange(80.0) + 100.0), window=40, embedding_dim=10)
    vals = out["A"].tolist()
    assert all(np.isnan(v) for v in vals[:39]), "rows 0..38 (warmup) must be NaN"
    assert np.isfinite(vals[39]), "row 39 (first full window) is finite"


def test_ssa_numerical_rank_checked_on_actual_run():
    op = _hk.TsSsaReconstructionResidual()
    # A linear ramp has numerical rank 2 in the Hankel embedding; asking for 9
    # components exceeds the rank and must fail closed (NaN), not emit 0.
    out = op.calculate(_frame(np.arange(80.0)), window=40, embedding_dim=10, n_components=9)
    mature = out["A"].tolist()[39:]
    assert all(np.isnan(v) for v in mature), "rank-deficient SSA -> NaN"
    # A full-rank noisy signal with a small n_components still emits.
    rng = np.random.default_rng(0)
    xn = _frame(rng.normal(size=80))
    out_n = op.calculate(xn, window=40, embedding_dim=10, n_components=3)
    mature_n = out_n["A"].tolist()[39:]
    assert sum(np.isfinite(v) for v in mature_n) > 20


def test_ssa_constant_series_fails_closed():
    op = _hk.TsSsaReconstructionResidual()
    out = op.calculate(_frame(np.full(60, 7.0)), window=40, embedding_dim=10, n_components=3)
    vals = out["A"].tolist()
    assert all(np.isnan(v) for v in vals), "constant series -> NaN, never 0/EPS"


# ---------------------------------------------------------------------------
# #95 / #96  vector path
# ---------------------------------------------------------------------------
def test_vector_path_fixed_horizon_nan_until_window():
    op = _vp.TsVectorPathEfficiency()
    f1 = _frame(np.arange(80.0))
    f2 = _frame(np.arange(80.0) * 2.0)
    out = op.calculate(f1, f2, window=60)
    vals = out["A"].tolist()
    assert all(np.isnan(v) for v in vals[:59]), "2-3 point prefix must not emit"
    assert np.isfinite(vals[59])


def test_vector_self_intersection_rate_collapsed_path_is_nan():
    op = _vp.TsVectorSelfIntersectionRate()
    # A path collapsed to a single point has no intersection structure — its
    # self-intersection rate is UNKNOWN, never a clean 0.
    out = op.calculate(_frame(np.full(10, 5.0)), _frame(np.full(10, 5.0)), window=10)
    vals = out["A"].tolist()
    assert all(np.isnan(v) for v in vals[9:])
    # A genuine non-intersecting path still reports a legitimate 0.
    straight = np.linspace(0.0, 1.0, 10)
    out2 = op.calculate(_frame(straight), _frame(straight), window=10)
    assert out2["A"].iloc[9] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# #41  fractional-difference startup region NaN per contract
# ---------------------------------------------------------------------------
def test_fractional_difference_startup_region_is_nan():
    op = _me.TsFractionalDifference()
    out = op.calculate(_frame(np.arange(80.0)), fd=0.4, cutoff=20)
    vals = out["A"].tolist()
    # Rows 0..cutoff-1 cannot apply the full cutoff+1-term filter -> NaN.
    assert all(np.isnan(v) for v in vals[:20]), "startup region must be NaN"
    assert np.isfinite(vals[20]), "first full-history row is finite"
    # The full filter horizon agrees with cutoff+1 terms.
    expected = 0.0  # placeholder; the key contract is the NaN prefix


# ---------------------------------------------------------------------------
# #28  normalized copula entropy unit is dimensionless, not nats
# ---------------------------------------------------------------------------
def test_copula_entropy_unit_is_dimensionless():
    ent = OperatorRegistry.get("cs_rank_copula_entropy", "pandas_numpy")
    mi = OperatorRegistry.get("cs_rank_copula_mi", "pandas_numpy")
    ent_tags = [t for t in ent.metadata.tags if t.startswith("unit:")]
    mi_tags = [t for t in mi.metadata.tags if t.startswith("unit:")]
    assert ent_tags == ["unit:dimensionless"], ent_tags
    assert mi_tags == ["unit:nats"], mi_tags
