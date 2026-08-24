# -*- coding: utf-8 -*-
"""Review-10 #21 / #22 / #23 — multi-regime semantic dedup.

* R10 #21 — dedup is MULTI-REGIME: per-day Spearman, top/bottom-decile overlap,
  position overlap, aggregated to median / p10 / p90 / duplicate-day ratio per
  regime.  A pair must agree on the vast majority of regimes AND days; two
  factors that are identical on Gaussian normal days but diverge on extreme days
  are NOT merged (a flattened date×stock rho would merge them).
* R10 #22 — every dedup group is scoped by a FactorKind (ALPHA / CONDITION /
  EVENT / GLOBAL_STATE); different kinds never collide.
* R10 #23 — signed / categorical state factors get a transition / state-label
  signature, and CONDITION / EVENT duplicates are gated by state labels, not
  value correlation.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from factor_engine.cleaned_operators.search.factor_dedup import (
    DedupPolicy,
    FactorKind,
    aggregate_day_metrics,
    are_rank_duplicates,
    build_ashare_fixture,
    build_typed_fixtures,
    day_is_duplicate,
    dedup_bucket,
    per_date_metrics,
    probe_panel,
    regime_similarity_report,
    state_label_signature,
    transition_signature,
)


# --------------------------------------------------------------------------- #
# R10 #21 — regime fixture generator
# --------------------------------------------------------------------------- #
_REQUIRED_REGIMES = {
    "gaussian", "heavy_tail", "trend", "mean_revert", "ties", "gaps",
    "positive_only", "event_bool", "event_signed_intensity", "group", "ohlc",
}


def test_regime_fixture_generator_spans_required_regimes():
    fixtures = build_typed_fixtures()
    assert _REQUIRED_REGIMES <= set(fixtures)
    # gaps models MISSING ROWS (whole no-trading days), not just NaN cells.
    gaps = fixtures["gaps"].to_numpy()
    assert np.isnan(gaps[2]).all()
    assert np.isnan(gaps[-2]).all()


def test_ashare_fixture_is_realistic_trading_with_jumps():
    ash = build_ashare_fixture()
    assert ash.shape == (240, 128)
    assert ash.index.is_monotonic_increasing
    arr = ash.to_numpy(dtype=float)
    assert np.isfinite(arr).all() and (arr > 0).all()
    # log-returns carry at least one big jump (>5% day) and vol clustering is
    # visible in the largest |return| far exceeding a plain Gaussian daily move.
    logret = np.diff(np.log(arr), axis=0)
    assert float(np.nanmax(np.abs(logret))) > 0.05


# --------------------------------------------------------------------------- #
# R10 #21 — tail-regime divergence must NOT be merged
# --------------------------------------------------------------------------- #
def _regime_flip_factor(threshold: float = 5.0):
    """On days any column is extreme (|v| > threshold) flip every column."""
    def _fn(p: pd.DataFrame) -> pd.DataFrame:
        arr = p.to_numpy(dtype=float)
        with warnings.catch_warnings():
            # the gaps regime has whole missing rows -> nanmax warns harmlessly
            warnings.simplefilter("ignore", RuntimeWarning)
            extreme = np.nanmax(np.abs(arr), axis=1) > threshold
        arr[extreme] = -arr[extreme]
        return pd.DataFrame(arr, index=p.index, columns=p.columns)
    return _fn


def test_gaussian_identical_but_tail_regime_differing_not_removed():
    fixtures = build_typed_fixtures()
    fa = lambda p: p
    fb = _regime_flip_factor(threshold=5.0)
    # identical under the Gaussian draw ...
    assert are_rank_duplicates(fa, fb, panel=fixtures["gaussian"]) is True
    # ... but the heavy-tail regime flips enough extreme days that the pair is
    # NOT a duplicate when the decision spans gaussian + heavy-tail.
    assert (
        are_rank_duplicates(
            fa, fb, regimes=("gaussian", "heavy_tail"), min_regime_agreement=1.0
        )
        is False
    )


def test_flattened_rho_passes_but_extreme_days_not_removed():
    """99% of days identical, 1% extreme day opposite — flattened rho passes,
    the per-date decision rejects it."""
    rows, cols = 100, 50
    rng = np.random.default_rng(13)
    base = rng.normal(size=(rows, cols))
    flipped = base.copy()
    flipped[-1] = -base[-1]  # the one extreme day is anti-correlated

    # A lax flattened date x stock correlation would happily merge this pair.
    flat = float(np.corrcoef(base.ravel(), flipped.ravel())[0, 1])
    assert flat >= 0.95

    idx = pd.date_range("2023-01-02", periods=rows, freq="D")
    cols_list = [f"C{i}" for i in range(cols)]
    panel = pd.DataFrame(base, index=idx, columns=cols_list)
    fa = lambda p: pd.DataFrame(base, index=p.index, columns=p.columns)
    fb = lambda p: pd.DataFrame(flipped, index=p.index, columns=p.columns)

    # Per-date metrics path: duplicate-day ratio is 0.99 but the worst day is
    # anti-correlated, which is exactly the alpha that must be kept.
    ra = pd.DataFrame(base).rank(axis=1).to_numpy()
    rb = pd.DataFrame(flipped).rank(axis=1).to_numpy()
    summary = aggregate_day_metrics(
        per_date_metrics(ra, rb, min_peers=cols),
        rho_threshold=0.99999,
        overlap_threshold=0.9,
    )
    assert summary.n_days == rows
    assert summary.duplicate_day_ratio == pytest.approx(0.99)
    assert summary.spearman_min < 0.0  # the extreme day is visible

    # The decision rejects the pair (not removed).
    assert are_rank_duplicates(fa, fb, panel=panel) is False


def test_x_vs_2x_terminal_rank_vs_compositional():
    """x vs 2*x: merged under terminal-rank-equivalence, kept under
    compositional policy (R10 #21)."""
    panel = probe_panel()
    f = lambda p: p
    two = lambda p: 2.0 * p

    assert are_rank_duplicates(
        f, two, panel=panel, policy=DedupPolicy(rank_equivalence=True)
    ) is True
    assert are_rank_duplicates(
        f, two, panel=panel, policy=DedupPolicy(rank_equivalence=False)
    ) is False
    # a factor is trivially its own duplicate under both policies
    assert are_rank_duplicates(
        f, f, panel=panel, policy=DedupPolicy(rank_equivalence=False)
    ) is True


# --------------------------------------------------------------------------- #
# R10 #22 — FactorKind groups never collide
# --------------------------------------------------------------------------- #
def test_different_kinds_never_collide_in_dedup_groups():
    f = lambda p: p
    groups = {
        dedup_bucket(f, factor_kind=kind)
        for kind in (
            FactorKind.ALPHA,
            FactorKind.CONDITION,
            FactorKind.EVENT,
            FactorKind.GLOBAL_STATE,
        )
    }
    assert len(groups) == 4  # every kind lives in its own group


def test_global_state_kind_uses_time_series_rank():
    """The same pair reaches different decisions under different kinds: the two
    factors have IDENTICAL cross-sectional ranks each day (ALPHA merges them)
    but DIFFERENT per-instrument time paths (GLOBAL_STATE keeps them)."""
    panel = probe_panel()

    def levels(p: pd.DataFrame) -> pd.DataFrame:
        arr = np.tile(np.arange(p.shape[1], dtype=float), (p.shape[0], 1))
        return pd.DataFrame(arr, index=p.index, columns=p.columns)

    def levels_plus_trend(p: pd.DataFrame) -> pd.DataFrame:
        arr = np.tile(np.arange(p.shape[1], dtype=float), (p.shape[0], 1))
        arr = arr + np.arange(p.shape[0], dtype=float)[:, None] * 0.001
        return pd.DataFrame(arr, index=p.index, columns=p.columns)

    # same per-date cross-sectional ordering (level / level + common trend)
    assert are_rank_duplicates(levels, levels_plus_trend, panel=panel, factor_kind=FactorKind.ALPHA) is True
    # different per-instrument time-series path (flat vs trending)
    assert are_rank_duplicates(levels, levels_plus_trend, panel=panel, factor_kind=FactorKind.GLOBAL_STATE) is False


# --------------------------------------------------------------------------- #
# R10 #22 / #23 — CONDITION / EVENT use state labels, not value correlation
# --------------------------------------------------------------------------- #
def _state_mask(p: pd.DataFrame, thr: float) -> pd.DataFrame:
    return pd.DataFrame((p.to_numpy() > thr).astype(float), index=p.index, columns=p.columns)


def test_condition_kind_uses_state_semantics_not_alpha_rank():
    panel = probe_panel()
    ca = lambda p: _state_mask(p, 0.0)
    cb = lambda p: _state_mask(p, 0.01)

    # The two states are ~the same active set: duplicates as CONDITION.
    assert are_rank_duplicates(ca, cb, panel=panel, factor_kind=FactorKind.CONDITION) is True
    # But their cross-sectional RANKS are not near-exact, so ALPHA keeps both.
    assert are_rank_duplicates(ca, cb, panel=panel, factor_kind=FactorKind.ALPHA) is False


def test_event_kind_uses_state_labels():
    panel = probe_panel()
    sparse = lambda p: _state_mask(p, 0.0)      # fires on ~50% of instruments
    rarer = lambda p: _state_mask(p, 1.0)       # fires on a disjoint-er ~16%

    # disjoint event sets are not duplicates ...
    assert are_rank_duplicates(sparse, rarer, panel=panel, factor_kind=FactorKind.EVENT) is False
    # ... and an event factor is its own duplicate.
    assert are_rank_duplicates(sparse, sparse, panel=panel, factor_kind=FactorKind.EVENT) is True


# --------------------------------------------------------------------------- #
# R10 #21 — per-day metrics path is exercised and reported
# --------------------------------------------------------------------------- #
def test_per_date_metrics_fields_reported():
    rng = np.random.default_rng(21)
    a = rng.normal(size=(30, 10))
    ra = pd.DataFrame(a).rank(axis=1, method="average").to_numpy()
    metrics = per_date_metrics(ra, ra, min_peers=10)
    assert len(metrics) == 30
    m = metrics[0]
    assert m.spearman == pytest.approx(1.0)
    assert m.top_decile_overlap == pytest.approx(1.0)
    assert m.bottom_decile_overlap == pytest.approx(1.0)
    assert m.position_overlap == pytest.approx(1.0)
    assert day_is_duplicate(m, rho_threshold=0.99999, overlap_threshold=0.9) is True

    summary = aggregate_day_metrics(metrics, rho_threshold=0.99999, overlap_threshold=0.9)
    assert summary.spearman_median == pytest.approx(1.0)
    assert summary.spearman_p10 == pytest.approx(1.0)
    assert summary.spearman_p90 == pytest.approx(1.0)
    assert summary.duplicate_day_ratio == pytest.approx(1.0)
    assert summary.n_duplicate_days == 30


def test_regime_similarity_report_reports_duplicate_day_ratio():
    fa = lambda p: p
    fb = _regime_flip_factor(threshold=5.0)

    report = regime_similarity_report(fa, fa, factor_kind=FactorKind.ALPHA)
    # every regime is covered, including the A-share trading regime
    for regime in ("gaussian", "heavy_tail", "ashare", "event_bool", "group", "ohlc"):
        assert regime in report
        s = report[regime]
        assert s.n_days > 0
        assert s.duplicate_day_ratio == pytest.approx(1.0)
        assert s.spearman_median == pytest.approx(1.0)

    # where they diverge is visible per-regime: heavy-tail drops, gaussian stays.
    divergent = regime_similarity_report(fa, fb, factor_kind=FactorKind.ALPHA)
    assert divergent["gaussian"].duplicate_day_ratio == pytest.approx(1.0)
    assert divergent["heavy_tail"].duplicate_day_ratio < 0.9


# --------------------------------------------------------------------------- #
# R10 #23 — transition / state-label signatures
# --------------------------------------------------------------------------- #
def test_transition_signature_distinguishes_timing():
    p = probe_panel()
    fast = lambda _p: (_p > 0).astype(float)  # flips with the sign every day

    def slow(_p: pd.DataFrame) -> pd.DataFrame:
        arr = np.zeros((_p.shape[0], _p.shape[1]))
        arr[_p.shape[0] // 2:] = 1.0
        return pd.DataFrame(arr, index=_p.index, columns=_p.columns)

    ta = transition_signature(fast, panel=p)
    tb = transition_signature(slow, panel=p)
    assert ta != tb                      # same ~50% on-rate, different timing
    assert transition_signature(fast, panel=p) == ta  # deterministic


def test_state_label_signature_is_deterministic_and_label_sensitive():
    p = probe_panel()
    fl = lambda _p: pd.DataFrame(np.floor(_p.to_numpy()).astype(float), index=_p.index, columns=_p.columns)
    ce = lambda _p: pd.DataFrame(np.ceil(_p.to_numpy()).astype(float), index=_p.index, columns=_p.columns)
    assert state_label_signature(fl, panel=p) == state_label_signature(fl, panel=p)
    assert state_label_signature(fl, panel=p) != state_label_signature(ce, panel=p)
