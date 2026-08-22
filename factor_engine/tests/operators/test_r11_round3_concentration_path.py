# -*- coding: utf-8 -*-
"""R11 round-3 (P1-I direction/concentration + P1-J alpha-language path) fixes.

Covers:
- 124 threshold-relative ops declare return/ratio input semantics and exclude
  raw price/volume at the metadata level;
- 125 ts_abs_entropy split into fixed-unit ts_abs_entropy_normalized /
  ts_abs_entropy_nats (no unit-switching boolean) with a back-compat
  ts_abs_entropy that rejects normalize=False;
- 126 raw HHI (ts_abs_concentration) is demoted to extended while the
  normalized excess-HHI (ts_mass_concentration) stays on the daily surface;
- 127 ts_effective_turning_rate (Definition B: active-pairs denominator) vs
  ts_turning_rate (Definition A: all-pairs denominator);
- 128 ts_turning_intensity returns NaN (not an EPS explosion) when MAD(delta)=0;
- 129 ts_path_efficiency declares constant_path_policy=ZERO (0, not 0/(0+eps));
- 130 ts_trend_break_score returns NaN (not an EPS explosion) when the recent
  residual std is 0 but the two slopes differ;
- 131 ts_endpoint_deviation uses trailing-contiguous (no drop-finite
  compression that would reconnect points across a missing-value gap).

The operator modules register their operators on import, so this file imports
them directly and is not blocked by other workstreams' in-flight edits.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import cleaned_operators.direction_concentration  # noqa: F401  (registers on import)
import cleaned_operators.alpha_language_shape  # noqa: F401  (registers on import)
from cleaned_operators.operator_surface import classify_canonical
from cleaned_operators.registry import OperatorRegistry


def _panel(values) -> pd.DataFrame:
    return pd.DataFrame(np.asarray(values, dtype=float).reshape(-1, 1), columns=["A"])


def _run(name: str, values, **kw) -> np.ndarray:
    op = OperatorRegistry.get(name, "pandas_numpy")
    assert op is not None, f"{name} missing pandas_numpy runtime"
    return op.calculate(_panel(values), **kw)["A"].to_numpy()


# ---------------------------------------------------------------------------
# 124 — threshold-relative operators: input semantics + no raw price/volume
# ---------------------------------------------------------------------------
def test_threshold_relative_ops_declare_return_ratio_input_semantics():
    for name in ("ts_positive_ratio", "ts_negative_ratio", "ts_zero_ratio"):
        meta = OperatorRegistry.get(name, "pandas_numpy").metadata
        tags = {str(t) for t in meta.tags}
        assert any(t.startswith("input_semantics:") for t in tags), (
            f"{name} missing input_semantics tag"
        )
        # The compatible-unit contract explicitly excludes raw price/volume
        # levels, whose >0 threshold is ~always true (meaningless signal).
        compat = meta.compatible_units.get("x", ())
        assert compat, f"{name} must declare compatible_units for x"
        assert "return" in compat, f"{name} x must accept return, got {compat!r}"
        assert not any("price" in u or "volume" in u for u in compat), (
            f"{name} x must NOT accept raw price/volume, got {compat!r}"
        )
        assert meta.input_units.get("x"), f"{name} missing input_units for x"


def test_threshold_relative_ops_still_compute_on_signed_input():
    # metadata-level contract only: the math runs on a signed series.
    r = np.array([0.01, -0.02, 0.03, -0.04, 0.05])
    out = _run("ts_positive_ratio", r, window=5, min_periods=5, threshold=0.0)
    assert np.isfinite(out[-1])
    assert out[-1] == 3 / 5


# ---------------------------------------------------------------------------
# 125 — ts_abs_entropy split into fixed-unit canonicals
# ---------------------------------------------------------------------------
def test_abs_entropy_split_canonicals_have_fixed_units():
    for name, unit in (("ts_abs_entropy_normalized", "dimensionless"),
                       ("ts_abs_entropy_nats", "nats")):
        op = OperatorRegistry.get(name, "pandas_numpy")
        assert op is not None, f"{name} missing"
        assert "normalize" not in op.metadata.param_names, (
            f"{name} must not carry a unit-switching normalize boolean, got "
            f"{op.metadata.param_names}"
        )
        assert op.metadata.output_unit == unit, (
            f"{name} output_unit -> {op.metadata.output_unit!r}"
        )


def test_abs_entropy_normalized_is_legacy_default_and_nats_is_log_scaled():
    x = [1.0, 2.0, 4.0, 8.0, 16.0]
    norm = _run("ts_abs_entropy_normalized", x, window=5, min_periods=5)[-1]
    nats = _run("ts_abs_entropy_nats", x, window=5, min_periods=5)[-1]
    legacy = _run("ts_abs_entropy", x, window=5, min_periods=5)[-1]
    # legacy default (normalize=True) is the dimensionless/normalized unit.
    assert np.isclose(legacy, norm)
    # normalized = nats / log(n) for a fully-finite window.
    assert abs(norm - nats / np.log(5.0)) < 1e-9
    # nats >= normalized (log(5) > 1) for a concentrated distribution.
    assert nats > norm


def test_abs_entropy_legacy_rejects_normalize_false():
    import pytest

    op = OperatorRegistry.get("ts_abs_entropy", "pandas_numpy")
    with pytest.raises(ValueError):
        op.calculate(_panel([1.0, 2.0, 3.0]), window=3, normalize=False)


def test_abs_entropy_split_surfaces():
    assert classify_canonical("ts_abs_entropy_normalized") == "daily", (
        "the fixed-unit normalized successor belongs on the default surface"
    )
    assert classify_canonical("ts_abs_entropy_nats") == "extended", (
        "the nats variant is a niche fixed-unit canonical -> extended"
    )


# ---------------------------------------------------------------------------
# 126 — raw HHI stays extended; normalized excess-HHI is the default surface
# ---------------------------------------------------------------------------
def test_raw_hhi_extended_and_normalized_excess_hhi_daily():
    assert classify_canonical("ts_abs_concentration") == "extended", (
        "raw HHI (1/N mechanical baseline) stays on extended"
    )
    assert classify_canonical("ts_mass_concentration") == "daily", (
        "normalized excess-HHI is preferred on the default mining surface"
    )


# ---------------------------------------------------------------------------
# 127 — ts_effective_turning_rate (Definition B) vs ts_turning_rate (A)
# ---------------------------------------------------------------------------
def test_turning_rate_definition_a_vs_effective_definition_b():
    # Flat run (+ ,0,0,0,-): Definition A divides by all pairs (0/4 -> 0);
    # Definition B has zero active (both-delta-nonzero) pairs -> NaN.
    flat = [1.0, 2.0, 2.0, 2.0, 2.0, 1.0]
    a_flat = _run("ts_turning_rate", flat, window=6, min_periods=2)
    b_flat = _run("ts_effective_turning_rate", flat, window=6, min_periods=2)
    assert a_flat[-1] == 0.0
    assert np.isnan(b_flat[-1])
    # Mixed (+,-,0,0,+): Definition A counts 1 flip / 4 pairs = 0.25; Definition
    # B counts the single active flip / 1 active pair = 1.0.
    mix = [1.0, 2.0, 1.0, 1.0, 1.0, 2.0]
    a_mix = _run("ts_turning_rate", mix, window=6, min_periods=2)
    b_mix = _run("ts_effective_turning_rate", mix, window=6, min_periods=2)
    assert a_mix[-1] == 0.25
    assert b_mix[-1] == 1.0


# ---------------------------------------------------------------------------
# 128 — ts_turning_intensity: MAD(delta)=0 -> NaN, never EPS explosion
# ---------------------------------------------------------------------------
def test_turning_intensity_mad_zero_returns_nan():
    # Constant-magnitude alternating path: |delta| constant -> MAD(delta)=0,
    # so the ratio is degenerate.  Must be NaN, not ~4e12.
    alt = [1.0, 3.0, 1.0, 3.0, 1.0, 3.0]
    out = _run("ts_turning_intensity", alt, window=6, min_periods=3)
    assert np.isnan(out[-1])
    assert np.all(np.isfinite(out) | np.isnan(out))


# ---------------------------------------------------------------------------
# 129 — ts_path_efficiency: constant_path_policy=ZERO
# ---------------------------------------------------------------------------
def test_path_efficiency_constant_path_is_zero_policy():
    op = OperatorRegistry.get("ts_path_efficiency", "pandas_numpy")
    assert any("constant_path_policy:zero" in str(t) for t in op.metadata.tags), (
        "constant-path policy must be declared in metadata"
    )
    out = _run("ts_path_efficiency", [5.0, 5.0, 5.0, 5.0, 5.0], window=5, min_periods=2)
    assert out[-1] == 0.0  # deliberate ZERO policy, not an 0/(0+eps) accident


# ---------------------------------------------------------------------------
# 130 — ts_trend_break_score: recent residual std=0 -> NaN
# ---------------------------------------------------------------------------
def test_trend_break_score_sigma_zero_different_slopes_is_nan():
    # Recent segment is perfectly linear (sigma=0) with a DIFFERENT slope from
    # the old segment: (10-1)/(0+eps) would explode -> must be NaN.
    tb = [1.0, 2.0, 3.0, 10.0, 20.0, 30.0]
    out = _run("ts_trend_break_score", tb, window=6, split=0.5, min_periods=4)
    assert np.isnan(out[-1])


def test_trend_break_score_fully_linear_still_zero():
    # Fully linear window: slopes equal AND sigma=0 -> policy 0.0 (kept).
    lin = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    out = _run("ts_trend_break_score", lin, window=6, split=0.5, min_periods=4)
    assert out[-1] == 0.0


# ---------------------------------------------------------------------------
# 131 — ts_endpoint_deviation: trailing-contiguous, no drop-finite compression
# ---------------------------------------------------------------------------
def test_endpoint_deviation_uses_trailing_contiguous_not_drop_finite():
    # Interior NaN must break the run, not be compressed away: dropping
    # (day1,day2,NaN,day10) to (day1,day2,day10) would reset the OLS x-axis to
    # 0,1,2 and treat a gap as contiguous geometry.
    gap = [1.0, 2.0, 3.0, np.nan, 10.0, 11.0]
    out = _run("ts_endpoint_deviation", gap, window=6, min_periods=3)
    # Rows 3..5 sit on/after the gap; the trailing-contiguous run is too short.
    assert np.all(np.isnan(out[3:])), f"gap rows must be NaN, got {out[3:]}"


def test_endpoint_deviation_linear_window_is_zero():
    lin = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    out = _run("ts_endpoint_deviation", lin, window=6, min_periods=3)
    assert out[-1] == 0.0
