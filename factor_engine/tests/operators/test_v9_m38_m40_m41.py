from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest


def _load(name: str):
    return importlib.import_module(f"factor_engine.cleaned_operators.{name}")


def test_m38_public_first_passage_classes_share_complete_horizon_contract():
    fp = _load("first_passage")
    idx = pd.RangeIndex(3)
    x = pd.DataFrame({"A": [0.0, 2.0, np.nan]}, index=idx)
    scale = pd.DataFrame({"A": [1.0, 1.0, 1.0]}, index=idx)
    common = dict(window=2, barrier=1.0, horizon=2, min_anchors=1)

    bias = fp.TsFirstPassageBias().calculate(x, scale, **common)
    hit = fp.TsFirstPassageHitProbability().calculate(
        x, scale, side="upper", **common
    )
    cond = fp.TsFirstPassageConditionalTime().calculate(
        x, scale, side="upper", **common
    )

    assert np.isnan(bias.iloc[-1, 0])
    assert np.isnan(hit.iloc[-1, 0])
    assert np.isnan(cond.iloc[-1, 0])


@pytest.mark.parametrize(
    ("path", "expected_bias", "expected_up", "expected_dn"),
    [
        ([0.0, 2.0, 2.0], 1.0, 1.0, 0.0),
        ([0.0, -2.0, -2.0], -1.0, 0.0, 1.0),
        ([0.0, 0.25, -0.25], 0.0, 0.0, 0.0),
    ],
)
def test_m38_complete_anchor_hit_and_non_hit_decomposition(
    path, expected_bias, expected_up, expected_dn
):
    fp = _load("first_passage")
    x = pd.DataFrame({"A": path})
    scale = pd.DataFrame({"A": [1.0] * 3})
    common = dict(window=2, barrier=1.0, horizon=2, min_anchors=1)
    bias = fp.TsFirstPassageBias().calculate(x, scale, **common).iloc[-1, 0]
    up = fp.TsFirstPassageHitProbability().calculate(
        x, scale, side="upper", **common
    ).iloc[-1, 0]
    dn = fp.TsFirstPassageHitProbability().calculate(
        x, scale, side="lower", **common
    ).iloc[-1, 0]
    assert bias == pytest.approx(expected_bias)
    assert up == pytest.approx(expected_up)
    assert dn == pytest.approx(expected_dn)
    assert up + dn <= 1.0


@pytest.mark.parametrize("path", [[0.0, np.nan, 2.0], [0.0, 2.0, np.nan]])
def test_m38_missing_before_or_after_apparent_hit_invalidates_anchor(path):
    fp = _load("first_passage")
    x = pd.DataFrame({"A": path})
    scale = pd.DataFrame({"A": [1.0] * 3})
    kwargs = dict(window=2, barrier=1.0, horizon=2, min_anchors=1)
    for cls in (
        fp.TsFirstPassageBias,
        fp.TsFirstPassageHitProbability,
        fp.TsFirstPassageConditionalTime,
    ):
        assert np.isnan(cls().calculate(x, scale, **kwargs).iloc[-1, 0])


def test_m40_second_localization_requires_both_adjacent_spectral_gaps():
    gs = _load("group_spectrum")
    rng = np.random.default_rng(0)
    n = 1000
    common = rng.normal(size=n)
    # One dominant common mode; the remaining exchangeable noise modes are
    # nearly degenerate, so mode 1 is identifiable but mode 2 is not.
    raw = np.column_stack(
        [common + 0.7 * rng.normal(size=n) for _ in range(3)]
    )
    med = np.median(raw, axis=0)
    mad = 1.4826 * np.median(np.abs(raw - med), axis=0)
    Z = (raw - med) / mad
    stats = gs._spectrum_stats(Z, eigen_gap=0.10)
    assert stats is not None
    assert np.isfinite(stats[2])
    assert np.isnan(stats[4])

    idx = pd.RangeIndex(1)
    cols = [f"S{i}" for i in range(n)]
    frames = [pd.DataFrame([raw[:, j]], index=idx, columns=cols) for j in range(3)]
    group = pd.DataFrame([["G"] * n], index=idx, columns=cols)
    first = gs.GroupFeatureModeLocalization().calculate(*frames, group)
    second = gs.GroupFeatureSecondModeLocalization().calculate(*frames, group)
    assert np.isfinite(first.iloc[0, 0])
    assert np.isnan(second.iloc[0, 0])


def test_m40_localization_gap_rules_are_permutation_invariant():
    gs = _load("group_spectrum")
    rng = np.random.default_rng(17)
    Z = rng.normal(size=(40, 3))
    base = gs._spectrum_stats(Z, eigen_gap=0.10)
    rows = gs._spectrum_stats(Z[rng.permutation(40)], eigen_gap=0.10)
    cols = gs._spectrum_stats(Z[:, [2, 0, 1]], eigen_gap=0.10)
    assert base is not None and rows is not None and cols is not None
    np.testing.assert_allclose(base, rows, equal_nan=True)
    np.testing.assert_allclose(base, cols, equal_nan=True)


def test_m40_two_feature_second_mode_uses_implicit_zero_third_value():
    gs = _load("group_spectrum")
    rng = np.random.default_rng(18)
    Z = rng.normal(size=(100, 2))
    Z[:, 1] += 0.3 * Z[:, 0]
    stats = gs._spectrum_stats(Z, eigen_gap=0.05)
    assert stats is not None
    assert np.isfinite(stats[4])


def test_m41_breadth_history_ages_on_actual_bar_clock_without_zero_fill():
    gs = _load("group_spectrum")
    rng = np.random.default_rng(412)
    rows, cols = 9, 20
    idx = pd.RangeIndex(rows)
    names = [f"S{i}" for i in range(cols)]
    arrays = [rng.normal(size=(rows, cols)) for _ in range(3)]
    # Bars 3..7 are unknown/insufficient.  At bar 8 exactly eight members are
    # known: enough for d=3, and the old 20-member counts have expired in a
    # three-bar window.  Missing members remain missing, never known zeros.
    for arr in arrays:
        arr[3:8, 7:] = np.nan
        arr[8, 8:] = np.nan
    frames = [pd.DataFrame(a, index=idx, columns=names) for a in arrays]
    group = pd.DataFrame([["G"] * cols] * rows, index=idx, columns=names)

    out = gs.GroupFeatureModeShare().calculate(
        *frames, group, breadth_window=3
    )
    assert out.iloc[3:8].isna().all().all()
    assert out.iloc[8, :8].notna().all()
    assert out.iloc[8, 8:].isna().all()


def test_m41_exact_breadth_window_warmup_matches_full_history():
    gs = _load("group_spectrum")
    rng = np.random.default_rng(413)
    rows, cols, bw = 12, 20, 3
    names = [f"S{i}" for i in range(cols)]
    arrays = [rng.normal(size=(rows, cols)) for _ in range(3)]
    frames = [pd.DataFrame(a, columns=names) for a in arrays]
    group = pd.DataFrame([["G"] * cols] * rows, columns=names)
    full = gs.GroupFeatureModeShare().calculate(*frames, group, breadth_window=bw)
    warm = gs.GroupFeatureModeShare().calculate(
        *(f.iloc[-(bw + 1):] for f in frames),
        group.iloc[-(bw + 1):],
        breadth_window=bw,
    )
    np.testing.assert_allclose(full.iloc[-1], warm.iloc[-1], equal_nan=True)
