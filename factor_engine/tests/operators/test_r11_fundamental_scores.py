# -*- coding: utf-8 -*-
"""Round-11 score-semantics regression tests for the composite fundamental scores.

Covers four R11 operator-semantic corrections in
``cleaned_operators/fundamental/accruals_scores.py``:
- ISSUE 1  fundamental-strength cash component: ``cash_yield := OCF / AverageAssets``
           (new ``avg_assets`` input), not the OCF/|ROA| size-dominated artifact.
- ISSUE 2  fundamental-strength coverage: score is the MEAN over the OBSERVED
           components and fails closed below ``MIN_FUNDAMENTAL_COMPONENTS``;
           ``fin_fundamental_strength_coverage`` exposes the observed count.
- ISSUE 3  Piotroski split: ``piotroski_f_score`` (full, NaN unless all 9 observed),
           ``piotroski_partial_score`` (passed/observed), ``piotroski_observed_count``.
- ISSUE 4  Altman Z / Zmijewski applicability mask is REQUIRED (fail closed).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# NOTE: importing ``cleaned_operators.fundamental.accruals_scores`` registers the
# operators under test via ``_mk(...)``, so this module does NOT need the full
# ``ensure_cleaned_loaded()`` load_all() (which is currently blocked by the
# concurrent session's in-flight R4-100 ``ts_extremal_index/polars`` edit).
from cleaned_operators.registry import OperatorRegistry
from cleaned_operators.fundamental.accruals_scores import (
    MIN_FUNDAMENTAL_COMPONENTS,
    _fin_altman_z_score,
    _fin_fundamental_strength_coverage,
    _fin_fundamental_strength_score,
    _fin_piotroski_f_score,
    _fin_piotroski_observed_count,
    _fin_piotroski_partial_score,
    _fin_zmijewski_score,
)


def _pid(periods, idx):
    return pd.DataFrame(np.asarray(periods, dtype=object)[:, None], index=idx, columns=["S0"])


# ---------------------------------------------------------------------------
# ISSUE 1 — cash_yield := OCF / AverageAssets
# ---------------------------------------------------------------------------

def test_fundamental_strength_cash_yield_is_ocf_over_avg_assets():
    idx = pd.date_range("2025-01-01", periods=2, freq="D")
    cols = ["A", "B", "C"]
    pid = pd.DataFrame(np.array([["2024Q1"] * 3, ["2024Q1"] * 3], dtype=object),
                       index=idx, columns=cols)

    def panel(a, b, c):
        return pd.DataFrame([[a, b, c], [a, b, c]], index=idx, columns=cols)

    # A and B have the SAME ROA and the same OCF, so the OLD cash component
    # (OCF/|ROA|) ties at 2000 for both; only avg_assets differs, so the NEW
    # cash component (OCF/AverageAssets) is 0.20 vs 0.02.
    roa = panel(0.05, 0.05, 0.10)
    ocf = panel(100.0, 100.0, 300.0)
    gm = panel(0.30, 0.30, 0.25)
    at = panel(0.6, 0.6, 0.5)
    lev = panel(0.4, 0.4, 0.5)
    rt = panel(1.5, 1.5, 1.0)
    it = panel(2.0, 2.0, 1.5)
    avg_assets = panel(500.0, 5000.0, 3000.0)  # A: 0.20, B: 0.02, C: 0.10
    rg = panel(0.1, 0.1, 0.05)

    # The new avg_assets parameter is accepted by the signature.
    score = _fin_fundamental_strength_score(roa, ocf, gm, at, lev, rt, it, avg_assets, rg, pid)
    assert np.isfinite(score["A"].iloc[0])
    # A (OCF/AvgAssets = 0.20) must rank above B (0.02).  Under the OLD cash
    # component (OCF/|ROA|) A and B tie, so this ordering is the regression.
    assert score["A"].iloc[0] > score["B"].iloc[0]


# ---------------------------------------------------------------------------
# ISSUE 2 — fundamental-strength coverage: mean over observed + MIN gate
# ---------------------------------------------------------------------------

def test_fundamental_strength_requires_min_observed_components():
    idx = pd.date_range("2025-01-01", periods=2, freq="D")
    cols = ["A", "B"]
    pid = pd.DataFrame(np.array([["2024Q1"] * 2, ["2024Q1"] * 2], dtype=object),
                       index=idx, columns=cols)

    def panel(a, b):
        return pd.DataFrame([[a, b], [a, b]], index=idx, columns=cols)

    # A: all 8 components finite.  B: only roa/ocf/leverage/revenue_growth finite
    # (4 observed) — gross_margin, asset_turnover, receivable_turnover and
    # inventory_turnover are missing.
    roa = panel(0.06, 0.05)
    ocf = panel(50.0, 40.0)
    gm = panel(0.30, np.nan)
    at = panel(0.6, np.nan)
    lev = panel(0.4, 0.45)
    rt = panel(1.5, np.nan)
    it = panel(2.0, np.nan)
    avg_assets = panel(500.0, 400.0)
    rg = panel(0.1, 0.08)

    score = _fin_fundamental_strength_score(roa, ocf, gm, at, lev, rt, it, avg_assets, rg, pid)
    cov = _fin_fundamental_strength_coverage(roa, ocf, gm, at, lev, rt, it, avg_assets, rg, pid)

    # A is the mean over its 8 observed components -> finite.
    assert np.isfinite(score["A"].iloc[0])
    # B has only 4 observed components (< MIN_FUNDAMENTAL_COMPONENTS) -> NaN.
    assert cov["A"].iloc[0] == 8
    assert cov["B"].iloc[0] == 4
    assert np.isnan(score["B"].iloc[0])
    assert MIN_FUNDAMENTAL_COMPONENTS == 6


def test_fundamental_strength_coverage_registered():
    op = OperatorRegistry.get("fin_fundamental_strength_coverage", "pandas_numpy")
    assert op is not None
    assert "flow_type:SinglePeriodFlow" in set(op.metadata.tags)


# ---------------------------------------------------------------------------
# ISSUE 3 — Piotroski split: full / partial / observed count
# ---------------------------------------------------------------------------

def _piotroski_split_panels():
    idx = pd.date_range("2025-01-01", periods=4, freq="B")
    pid = _pid(["2024Q1", "2024Q1", "2024Q2", "2024Q2"], idx)
    roa = pd.DataFrame([0.05, 0.05, 0.06, 0.06], index=idx, columns=["S0"])
    ocf = pd.DataFrame([0.03, 0.03, 0.03, 0.03], index=idx, columns=["S0"])
    npf = pd.DataFrame([0.02, 0.02, 0.02, 0.02], index=idx, columns=["S0"])
    nan = pd.DataFrame([np.nan] * 4, index=idx, columns=["S0"])
    lev = pd.DataFrame([0.4] * 4, index=idx, columns=["S0"])
    cr = pd.DataFrame([1.5] * 4, index=idx, columns=["S0"])
    tc = pd.DataFrame([100.0] * 4, index=idx, columns=["S0"])
    gm = pd.DataFrame([0.3] * 4, index=idx, columns=["S0"])
    at = pd.DataFrame([0.5] * 4, index=idx, columns=["S0"])
    return idx, pid, roa, ocf, npf, nan, lev, cr, tc, gm, at


def test_piotroski_partial_cell_fails_closed_on_full_score():
    _, pid, roa, ocf, npf, nan, _lev, _cr, _tc, _gm, _at = _piotroski_split_panels()
    # Only roa / ocf / net_profit are observable -> 4 signals on the 2024Q2 rows
    # (roa, ocf>0, roa_delta>0, ocf>net_profit).
    full = _fin_piotroski_f_score(roa, ocf, npf, nan, nan, nan, nan, nan, pid)
    partial = _fin_piotroski_partial_score(roa, ocf, npf, nan, nan, nan, nan, nan, pid)
    observed = _fin_piotroski_observed_count(roa, ocf, npf, nan, nan, nan, nan, nan, pid)

    assert observed.iloc[-1, 0] == 4
    # A genuine full F-score requires all 9 signals -> NaN, never a smaller count.
    assert np.isnan(full.iloc[-1, 0])
    # The partial score is passed / observed in [0, 1]; here 4/4 == 1.0.
    assert 0.0 <= partial.iloc[-1, 0] <= 1.0
    assert partial.iloc[-1, 0] == pytest.approx(1.0, abs=1e-9)


def test_piotroski_full_score_is_integer_when_all_observed():
    _, pid, roa, ocf, npf, nan, lev, cr, tc, gm, at = _piotroski_split_panels()
    full = _fin_piotroski_f_score(roa, ocf, npf, lev, cr, tc, gm, at, pid)
    observed = _fin_piotroski_observed_count(roa, ocf, npf, lev, cr, tc, gm, at, pid)

    assert observed.iloc[-1, 0] == 9
    fs = full.iloc[-1, 0]
    assert np.isfinite(fs)
    assert fs == int(fs)  # integer passed count
    assert 0 <= fs <= 9


def test_piotroski_three_operators_registered():
    for name in ("piotroski_f_score", "piotroski_partial_score", "piotroski_observed_count"):
        op = OperatorRegistry.get(name, "pandas_numpy")
        assert op is not None, name
        tags = set(op.metadata.tags)
        assert "flow_type:SinglePeriodFlow" in tags, name
        assert "applicable_universe:non_financial" in tags, name


# ---------------------------------------------------------------------------
# ISSUE 4 — Altman Z / Zmijewski applicability mask is REQUIRED (fail closed)
# ---------------------------------------------------------------------------

def test_altman_zmijewski_raise_without_mask():
    idx = pd.date_range("2025-01-01", periods=2, freq="D")
    cols = ["A", "B"]
    pid = pd.DataFrame(np.array([["2024Q1"] * 2, ["2024Q1"] * 2], dtype=object),
                       index=idx, columns=cols)

    def panel(a, b):
        return pd.DataFrame([[a, b], [a, b]], index=idx, columns=cols)

    with pytest.raises(ValueError, match="mask"):
        _fin_altman_z_score(panel(1, 2), panel(3, 4), panel(5, 6), panel(100, 100),
                            panel(50, 50), panel(40, 40), panel(80, 80), pid, None)
    with pytest.raises(ValueError, match="mask"):
        _fin_zmijewski_score(panel(5, 6), panel(100, 100), panel(40, 40),
                             panel(60, 60), panel(30, 30), pid, None)


def test_altman_zmijewski_mask_gates_inapplicable_universe():
    idx = pd.date_range("2025-01-01", periods=2, freq="D")
    cols = ["A", "B"]
    pid = pd.DataFrame(np.array([["2024Q1"] * 2, ["2024Q1"] * 2], dtype=object),
                       index=idx, columns=cols)

    def panel(a, b):
        return pd.DataFrame([[a, b], [a, b]], index=idx, columns=cols)

    mask = pd.DataFrame([[1.0, 0.0], [1.0, 0.0]], index=idx, columns=cols)  # B inapplicable
    z = _fin_altman_z_score(panel(1, 2), panel(3, 4), panel(5, 6), panel(100, 100),
                            panel(50, 50), panel(40, 40), panel(80, 80), pid, mask)
    zm = _fin_zmijewski_score(panel(5, 6), panel(100, 100), panel(40, 40),
                              panel(60, 60), panel(30, 30), pid, mask)
    assert np.isfinite(z["A"].iloc[0])
    assert np.isnan(z["B"].iloc[0])
    assert np.isfinite(zm["A"].iloc[0])
    assert np.isnan(zm["B"].iloc[0])
