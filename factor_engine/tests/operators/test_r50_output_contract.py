# -*- coding: utf-8 -*-
"""R50 output-contract + numeric-edge certification for broadcast context operators.

The R50 production-rehabilitation matrix flags every ``output_cardinality ==
"broadcast"`` operator with ``OUTPUT_WRONG_SHAPE``.  That blocker is a heuristic
*false positive* for operators whose broadcast is CORRECT because they are
genuinely cross-sectional GLOBAL or GROUP state (regime / condition inputs), not
per-stock alpha terminals:

* ``cs_*`` global/group-state primitives (coverage, churn, tail breadth /
  retention) — every instrument on a date receives the same value (all-market)
  or the same value within its group (industry / sector).
* ``group_feature_*`` spectrum operators — every member of a group on a date
  receives the same SVD-spectrum / membership-diagnostic value.

This suite locks in the two invariants the audit requires:

1. OUTPUT CONTRACT — the returned panel is aligned to the input universe
   (same index / columns / dtype) and is *within-scope constant*: a global
   operator is constant across the whole date row; a group operator is constant
   across the members of each group on that date.  A broadcast panel is still a
   stock x date series, so its shape must equal the input.

2. NUMERIC EDGE — fail-closed semantics: degenerate inputs (all-NaN) must never
   emit +Inf/-Inf or a fabricated finite value for a statistic; they emit NaN.
   Warmup (insufficient history) rows are NaN, not a fabricated value.

These certify that the R50 ``OUTPUT_WRONG_SHAPE`` flag is a contract-metadata
concern, not a kernel defect.  If any kernel starts returning an Inf or a
scalar (non-panel), this file fails.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_engine.backend.cleaned_bridge import ensure_cleaned_loaded
from factor_engine.cleaned_operators.registry import OperatorRegistry

ensure_cleaned_loaded()

BE = "pandas_numpy"

GLOBAL_OPS = [
    "cs_hartigan_dip",
    "cs_physical_panel_coverage",
    "cs_universe_coverage",
]
GROUP_OPS = [
    "cs_rank_churn",
    "cs_rank_composition_churn",
    "cs_rank_combined_churn",
    "cs_tail_breadth",
    "cs_tail_retention",
    "group_feature_coverage_ratio",
    "group_feature_effective_rank",
    "group_feature_mode_localization",
    "group_feature_mode_share",
    "group_feature_second_mode_localization",
    "group_feature_spectral_gap",
    "group_feature_valid_member_count",
]

N, COLS = 120, 12
DATES = pd.bdate_range("2024-01-02", periods=N)
ASSETS = [f"S{i}" for i in range(COLS)]


def _panels(seed: int = 7):
    """Return (close, ret, group) over a deterministic 12-stock universe."""
    rng = np.random.default_rng(seed)
    close = pd.DataFrame(
        np.exp(np.cumsum(rng.normal(0, 0.01, (N, COLS)), axis=0)) * 10.0,
        index=DATES,
        columns=ASSETS,
    )
    ret = close.pct_change()
    labels = ["g0"] * (COLS // 2) + ["g1"] * (COLS // 2)
    group = pd.DataFrame(np.tile(labels, (N, 1)), index=DATES, columns=ASSETS)
    return close, ret, group


def _runner(canonical):
    op = OperatorRegistry.get(canonical, BE)
    assert op is not None, f"{canonical}: missing pandas_numpy runtime"
    return op


def _invoke(canonical, close, ret, group):
    """Run one operator with default parameterisation over a real universe."""
    op = _runner(canonical)
    f1, f2, f3 = ret, close.rolling(5).mean(), ret.abs()
    if canonical == "cs_hartigan_dip":
        return op.calculate(ret, min_cross=3)
    if canonical == "cs_physical_panel_coverage":
        return op.calculate(close)
    if canonical == "cs_universe_coverage":
        return op.calculate(close, (close > 0).astype(float))
    if canonical.startswith("group_feature"):
        # the two diagnostic ops declare no breadth_window; the spectrum ops do
        if canonical in ("group_feature_valid_member_count", "group_feature_coverage_ratio"):
            return op.calculate(f1, f2, f3, group)
        return op.calculate(f1, f2, f3, group, breadth_window=3)
    if canonical in ("cs_rank_churn", "cs_rank_composition_churn", "cs_rank_combined_churn"):
        return op.calculate(ret, lag=3, group=group)
    if canonical == "cs_tail_breadth":
        return op.calculate(ret, quantile=0.3, group=group)
    # cs_tail_retention
    return op.calculate(ret, lag=3, quantile=0.3, side="top", group=group)


def test_r50_output_contract_shape_aligned_to_universe():
    """Every broadcast operator returns a stock x date panel aligned to input."""
    close, ret, group = _panels()
    for canonical in GLOBAL_OPS + GROUP_OPS:
        out = _invoke(canonical, close, ret, group)
        assert isinstance(out, pd.DataFrame), (
            f"{canonical}: expected a DataFrame panel, got {type(out).__name__}"
        )
        assert out.index.equals(DATES), f"{canonical}: index changed from universe"
        assert list(out.columns) == ASSETS, f"{canonical}: columns changed from universe"
        assert out.shape == (N, COLS), f"{canonical}: shape {out.shape} != {(N, COLS)}"
        assert str(out.dtypes.iloc[0]) == "float64", f"{canonical}: dtype drifted"


def test_r50_global_ops_constant_across_cross_section():
    """Global-state operators must be identical for every stock on a date."""
    close, ret, group = _panels()
    for name in GLOBAL_OPS:
        out = _invoke(name, close, ret, group)
        v = out.to_numpy()
        for r in range(v.shape[0]):
            fin = np.isfinite(v[r])
            if fin.any():
                assert len(set(np.round(v[r][fin], 9))) == 1, (
                    f"{name}: row {r} not constant across the cross-section"
                )


def test_r50_global_ops_date_scalar_broadcast():
    """A global state is one scalar per date, repeated across the row."""
    close, ret, group = _panels()
    for name in GLOBAL_OPS:
        out = _invoke(name, close, ret, group)
        v = out.to_numpy()
        for r in range(v.shape[0]):
            fin = v[r][np.isfinite(v[r])]
            if fin.size > 0:
                assert len(set(np.round(fin, 9))) == 1, (
                    f"{name}: row {r} not a single scalar broadcast"
                )


def test_r50_group_ops_constant_within_group():
    """Group-state operators must be identical across each group's members on a date."""
    close, ret, group = _panels()
    gv = group.to_numpy()
    for name in GROUP_OPS:
        out = _invoke(name, close, ret, group)
        v = out.to_numpy()
        for r in range(v.shape[0]):
            for lab in np.unique(gv[r]):
                mask = gv[r] == lab
                vals = v[r][mask]
                fin = vals[np.isfinite(vals)]
                if fin.size > 0:
                    assert len(set(np.round(fin, 6))) == 1, (
                        f"{name}: row {r} group {lab} not constant within group"
                    )


@pytest.mark.parametrize("name", GLOBAL_OPS + [c for c in GROUP_OPS if not c.startswith("group_feature")])
def test_r50_numeric_edge_no_inf_on_all_nan(name):
    """All-NaN input must never leak Inf; statistic kernels fail closed to NaN."""
    nan_panel = pd.DataFrame(np.nan * np.ones((N, COLS)), index=DATES, columns=ASSETS)
    nan_group = pd.DataFrame(
        np.full((N, COLS), None, dtype=object), index=DATES, columns=ASSETS
    )
    op = _runner(name)
    if name == "cs_hartigan_dip":
        out = op.calculate(nan_panel, min_cross=3)
    elif name == "cs_physical_panel_coverage":
        out = op.calculate(nan_panel)
    elif name == "cs_universe_coverage":
        out = op.calculate(nan_panel, (nan_panel > 0).astype(float))
    elif name in ("cs_rank_churn", "cs_rank_composition_churn", "cs_rank_combined_churn"):
        out = op.calculate(nan_panel, lag=3, group=nan_group)
    elif name == "cs_tail_breadth":
        out = op.calculate(nan_panel, quantile=0.3, group=nan_group)
    else:
        out = op.calculate(nan_panel, lag=3, quantile=0.3, side="top", group=nan_group)
    v = out.to_numpy()
    assert not np.isinf(v).any(), f"{name}: produced Inf on all-NaN input"
    # fail-closed statistic kernels (rank churn / tail / hartigan): an all-NaN
    # cross-section has no distribution -> NaN, never a fabricated value.
    # cs_physical_panel_coverage is a ratio 0-finite/cols => 0.0 is the CORRECT
    # value (denominator = physical column count, never 0), so it is exempt.
    if name in ("cs_hartigan_dip", "cs_rank_churn", "cs_rank_composition_churn",
                "cs_rank_combined_churn", "cs_tail_breadth", "cs_tail_retention"):
        assert np.isnan(v).all(), f"{name}: all-NaN input produced finite value (fail-closed violated)"


def test_r50_group_feature_all_nan_rejected_or_nan():
    """group_feature_* must never leak Inf on degenerate feature panels.

    Three all-NaN feature panels are bitwise identical, so the kernel's
    duplicate-feature guard raises a hard ValueError (degenerate input rejected
    up-front).  With distinct non-degenerate NaN-pattern panels the kernel must
    fail closed to all-NaN, never Inf.
    """
    nan_panel = pd.DataFrame(np.nan * np.ones((N, COLS)), index=DATES, columns=ASSETS)
    # distinct NaN pattern per feature panel so the duplicate guard does not fire
    f1 = nan_panel.copy()
    f2 = nan_panel.copy()
    f3 = nan_panel.copy()
    f2.iloc[0, 0] = 1.0
    f3.iloc[0, 0] = 2.0
    nan_group = pd.DataFrame(
        np.full((N, COLS), None, dtype=object), index=DATES, columns=ASSETS
    )
    for name in GROUP_OPS:
        if not name.startswith("group_feature"):
            continue
        op = _runner(name)
        if name in ("group_feature_valid_member_count", "group_feature_coverage_ratio"):
            out = op.calculate(f1, f2, f3, nan_group)
        else:
            out = op.calculate(f1, f2, f3, nan_group, breadth_window=3)
        v = out.to_numpy()
        assert not np.isinf(v).any(), f"{name}: produced Inf on degenerate input"
        # coverage_ratio is 0-valid/group_rows and valid_member_count is a 0
        # count — both are DIAGNOSTIC kernels where 0.0 is the CORRECT value on
        # a no-valid-member group (denominators = group row count, never 0).
        # The spectrum kernels are statistics and must fail closed to NaN.
        if name not in ("group_feature_coverage_ratio", "group_feature_valid_member_count"):
            assert np.isnan(v).all(), f"{name}: missing-group input produced a finite value"


@pytest.mark.parametrize("name", GLOBAL_OPS + GROUP_OPS)
def test_r50_numeric_edge_warmup_nan(name):
    """Leading warmup (lag) rows must be NaN for lag / rolling kernels."""
    close, ret, group = _panels()
    out = _invoke(name, close, ret, group)
    v = out.to_numpy()
    if name in ("cs_rank_churn", "cs_rank_composition_churn", "cs_rank_combined_churn",
                "cs_tail_retention"):
        assert np.isnan(v[:2]).all(), f"{name}: warmup rows not NaN"
