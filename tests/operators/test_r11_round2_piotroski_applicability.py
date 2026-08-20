# -*- coding: utf-8 -*-
"""Round-11 round-2 regression tests for the Piotroski strict/tolerant split and
the ApplicabilityBool mask contract.

Covers two P0 fixes in ``cleaned_operators/fundamental/accruals_scores.py``:

- ISSUE 1 (review section 十四): ``piotroski_f_score`` was branded a "genuine
  full F-score" while its equity-issuance signal passed at ``cap_g <= 5%`` — a
  5%-tolerant proxy, not the strict "no share issuance" criterion.  Split into
  two genuinely different economic definitions:
    * ``piotroski_f_score``          — strict: issuance passes only at cap_g <= 0.
    * ``piotroski_f_score_tolerant`` — 5% tolerance (cap_g <= 0.05).
  The old ``piotroski_f_score`` spelling keeps resolving to the strict canonical.
  Both keep the R11 nine-component completeness semantics (all 9 observed else
  NaN), and the ``piotroski_partial_score`` / ``piotroski_observed_count`` paths
  are untouched (split applies to the full-score path only).

- ISSUE 2 (review section 十五): the fundamental applicability mask must be a
  strict bool.  ``_masked()`` accepted any finite non-zero value (-1/0.4/2) as
  "applicable"; it now enforces an ApplicabilityBool contract: valid values are
  {0, 1, NaN} only — a finite value that is neither 0 nor 1 raises (fail closed);
  NaN and 0 mean not applicable (masked out); 1 means applicable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# NOTE: importing ``cleaned_operators.fundamental.accruals_scores`` registers the
# operators under test via ``_mk(...)``, so this module does NOT need the full
# ``ensure_cleaned_loaded()`` load_all() (which is currently blocked by the
# concurrent session's in-flight edits).
from cleaned_operators.registry import OperatorRegistry
from cleaned_operators.fundamental.accruals_scores import (
    _fin_altman_z_score,
    _fin_piotroski_f_score,
    _fin_piotroski_f_score_tolerant,
    _masked,
)


def _pid(periods, idx):
    return pd.DataFrame(np.asarray(periods, dtype=object)[:, None], index=idx, columns=["S0"])


# ---------------------------------------------------------------------------
# ISSUE 1 — Piotroski strict vs tolerant equity-issuance definition
# ---------------------------------------------------------------------------

def _full_panels(tc_values):
    """All 9 F-score signals observed and all 8 non-issuance signals passing on
    the 2024Q2 rows; only ``total_capital`` varies to control cap_g."""
    idx = pd.date_range("2025-01-01", periods=4, freq="B")
    pid = _pid(["2024Q1", "2024Q1", "2024Q2", "2024Q2"], idx)

    def panel(vals):
        return pd.DataFrame([float(v) for v in vals], index=idx, columns=["S0"])

    roa = panel([0.05, 0.05, 0.06, 0.06])        # roa > 0 and roa_delta > 0
    ocf = panel([0.03] * 4)                      # ocf > 0
    npf = panel([0.02] * 4)                      # ocf > net_profit
    lev = panel([0.40, 0.40, 0.35, 0.35])        # leverage_delta < 0
    cr = panel([1.5, 1.5, 1.6, 1.6])             # current_ratio_delta > 0
    tc = panel(tc_values)
    gm = panel([0.30, 0.30, 0.31, 0.31])         # gross_margin_delta > 0
    at = panel([0.50, 0.50, 0.55, 0.55])         # asset_turnover_delta > 0
    return pid, roa, ocf, npf, lev, cr, tc, gm, at


# (prior, current, expected_strict, expected_tolerant).  cap_g = (cur-prior)/|prior|.
# With all other 8 signals passing, the full score is 9 when issuance passes and
# 8 when it fails.  Strict and tolerant must differ EXACTLY on 0 < cap_g <= 0.05
# (strict fails -> 8, tolerant passes -> 9) and be equal everywhere else.
_CAP_G_CASES = [
    (100.0, 90.0, 9.0, 9.0),    # cap_g = -0.10  -> both pass
    (100.0, 95.0, 9.0, 9.0),    # cap_g = -0.05  -> both pass (buyback)
    (100.0, 100.0, 9.0, 9.0),   # cap_g =  0.00  -> both pass (no issuance)
    (100.0, 102.0, 8.0, 9.0),   # cap_g =  0.02  -> STRICT fails, tolerant passes
    (100.0, 105.0, 8.0, 9.0),   # cap_g =  0.05  -> STRICT fails, tolerant passes (boundary)
    (100.0, 108.0, 8.0, 8.0),   # cap_g =  0.08  -> both fail
    (100.0, 112.0, 8.0, 8.0),   # cap_g =  0.12  -> both fail
]


def test_piotroski_strict_vs_tolerant_differ_exactly_on_small_issuance():
    for prior, cur, exp_strict, exp_tolerant in _CAP_G_CASES:
        cap_g = (cur - prior) / abs(prior)
        pid, roa, ocf, npf, lev, cr, tc, gm, at = _full_panels([prior, prior, cur, cur])
        strict = _fin_piotroski_f_score(roa, ocf, npf, lev, cr, tc, gm, at, pid)
        tolerant = _fin_piotroski_f_score_tolerant(roa, ocf, npf, lev, cr, tc, gm, at, pid)

        s = strict.iloc[-1, 0]
        t = tolerant.iloc[-1, 0]
        assert s == exp_strict, f"cap_g={cap_g:+.2f} strict full={s}, expected {exp_strict}"
        assert t == exp_tolerant, f"cap_g={cap_g:+.2f} tolerant full={t}, expected {exp_tolerant}"

        if 0.0 < cap_g <= 0.05:
            assert s != t and s < t, (
                f"cap_g={cap_g:+.2f} must differ (strict fails, tolerant passes): {s} vs {t}"
            )
        else:
            assert s == t, (
                f"cap_g={cap_g:+.2f} must be equal outside (0, 0.05]: {s} vs {t}"
            )


def test_piotroski_both_enforce_nine_component_completeness():
    idx = pd.date_range("2025-01-01", periods=4, freq="B")
    pid = _pid(["2024Q1", "2024Q1", "2024Q2", "2024Q2"], idx)

    def panel(vals):
        return pd.DataFrame([float(v) for v in vals], index=idx, columns=["S0"])

    nan = panel([np.nan] * 4)
    roa = panel([0.05, 0.05, 0.06, 0.06])
    ocf = panel([0.03] * 4)
    npf = panel([0.02] * 4)
    lev = panel([0.40, 0.40, 0.35, 0.35])
    cr = panel([1.5, 1.5, 1.6, 1.6])
    tc = panel([100.0, 100.0, 102.0, 102.0])
    gm = panel([0.30, 0.30, 0.31, 0.31])
    at = panel([0.50, 0.50, 0.55, 0.55])

    # Only 8 of 9 observed (gross_margin missing) -> a genuine full F-score is
    # NOT comparable with a partial one -> both strict and tolerant fail closed.
    strict = _fin_piotroski_f_score(roa, ocf, npf, lev, cr, tc, nan, at, pid)
    tolerant = _fin_piotroski_f_score_tolerant(roa, ocf, npf, lev, cr, tc, nan, at, pid)
    assert np.isnan(strict.iloc[-1, 0])
    assert np.isnan(tolerant.iloc[-1, 0])

    # Even fewer observed (only roa/ocf/net_profit) -> NaN for both.
    strict2 = _fin_piotroski_f_score(roa, ocf, npf, nan, nan, nan, nan, nan, pid)
    tolerant2 = _fin_piotroski_f_score_tolerant(roa, ocf, npf, nan, nan, nan, nan, nan, pid)
    assert np.isnan(strict2.iloc[-1, 0])
    assert np.isnan(tolerant2.iloc[-1, 0])


def test_piotroski_old_name_resolves_to_strict_canonical():
    strict = OperatorRegistry.get("piotroski_f_score", "pandas_numpy")
    tolerant = OperatorRegistry.get("piotroski_f_score_tolerant", "pandas_numpy")
    assert strict is not None, "piotroski_f_score must stay registered"
    assert tolerant is not None, "piotroski_f_score_tolerant must be registered"

    # The old spelling resolves to the strict canonical, and the two are distinct
    # registry entries (a genuine economic split, not a parameterized tweak).
    assert strict.metadata.name == "piotroski_f_score"
    assert tolerant.metadata.name == "piotroski_f_score_tolerant"
    assert strict.metadata.name != tolerant.metadata.name

    for op in (strict, tolerant):
        tags = set(op.metadata.tags)
        assert "flow_type:SinglePeriodFlow" in tags
        assert "applicable_universe:non_financial" in tags


def test_piotroski_registered_strict_is_strict_and_tolerant_is_tolerant():
    # Drive the REGISTERED operators through the registry on a 2% issuer and
    # confirm strict == 8 (issuance fails) while tolerant == 9 (issuance passes).
    prior, cur = 100.0, 102.0
    pid, roa, ocf, npf, lev, cr, tc, gm, at = _full_panels([prior, prior, cur, cur])
    strict_op = OperatorRegistry.get("piotroski_f_score", "pandas_numpy")
    tolerant_op = OperatorRegistry.get("piotroski_f_score_tolerant", "pandas_numpy")
    strict = strict_op.calculate(roa, ocf, npf, lev, cr, tc, gm, at, pid)
    tolerant = tolerant_op.calculate(roa, ocf, npf, lev, cr, tc, gm, at, pid)
    assert strict.iloc[-1, 0] == 8.0
    assert tolerant.iloc[-1, 0] == 9.0


# ---------------------------------------------------------------------------
# ISSUE 2 — ApplicabilityBool strict mask contract
# ---------------------------------------------------------------------------

def test_masked_raises_on_non_bool_finite_values():
    score = pd.DataFrame([[1.0], [2.0], [3.0]])
    for bad in (2.0, 0.4, -1.0):
        with pytest.raises(ValueError, match="bool-valued"):
            _masked(score, pd.DataFrame([[bad]]))


def test_masked_bool_contract():
    score = pd.DataFrame([[10.0], [20.0], [30.0], [40.0]])
    mask = pd.DataFrame([[1.0], [0.0], [np.nan], [1.0]])
    out = _masked(score, mask)
    assert out.iloc[0, 0] == 10.0   # 1   -> applicable
    assert np.isnan(out.iloc[1, 0])  # 0   -> not applicable (masked out)
    assert np.isnan(out.iloc[2, 0])  # NaN -> not applicable (unknown, never truthy)
    assert out.iloc[3, 0] == 40.0   # 1   -> applicable


def test_masked_none_returns_score_unchanged():
    score = pd.DataFrame([[1.0], [2.0]])
    out = _masked(score, None)
    assert out.equals(score)


def test_altman_z_mask_strict_bool_fails_closed():
    idx = pd.date_range("2025-01-01", periods=2, freq="D")
    cols = ["A", "B"]
    pid = pd.DataFrame(np.array([["2024Q1"] * 2, ["2024Q1"] * 2], dtype=object),
                       index=idx, columns=cols)

    def panel(a, b):
        return pd.DataFrame([[a, b], [a, b]], index=idx, columns=cols)

    # A finite 0.4 applicability is a contract violation -> fail closed.
    bad_mask = pd.DataFrame([[1.0, 0.4], [1.0, 1.0]], index=idx, columns=cols)
    with pytest.raises(ValueError, match="bool-valued"):
        _fin_altman_z_score(panel(1, 2), panel(3, 4), panel(5, 6), panel(100, 100),
                            panel(50, 50), panel(40, 40), panel(80, 80), pid, bad_mask)

    # A valid {0,1} mask still gates exactly: A applicable, B inapplicable.
    ok_mask = pd.DataFrame([[1.0, 0.0], [1.0, 0.0]], index=idx, columns=cols)
    z = _fin_altman_z_score(panel(1, 2), panel(3, 4), panel(5, 6), panel(100, 100),
                            panel(50, 50), panel(40, 40), panel(80, 80), pid, ok_mask)
    assert np.isfinite(z["A"].iloc[0])
    assert np.isnan(z["B"].iloc[0])
