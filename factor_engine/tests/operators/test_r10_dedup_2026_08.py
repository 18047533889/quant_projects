# -*- coding: utf-8 -*-
"""Review-10 dedup overhaul (R10-P0-022/023/031/032/033).

* R10-P0-031 — ``dedup_bucket`` requires an explicit ``factor_kind``.
* R10-P0-022 — the default fingerprint spans MULTIPLE regimes; a factor that
  collides on a Gaussian probe but diverges under heavy-tail / trend regimes is
  kept, not merged.
* R10-P0-023 — the duplicate decision is PER-DATE, never one flattened
  date×stock correlation (10% of dates being opposite can be the alpha).
* R10-P0-032 — the OHLC fixture leaves no uninitialized columns when
  ``cols % 4 != 0``.
* R10-P0-033 — event fixtures are typed (bool / signed / positive intensity).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cleaned_operators.search.factor_dedup import (
    FactorBehaviorSignature,
    DedupPolicy,
    _per_date_duplicates,
    are_rank_duplicates,
    build_typed_fixtures,
    dedup_bucket,
    factor_signatures,
    multi_regime_signature,
    probe_panel,
    signature_for,
)
from cleaned_operators.search.factor_dedup import FactorKind


# ---------------------------------------------------------------------------
# R10-P0-031: dedup_bucket requires factor_kind
# ---------------------------------------------------------------------------

def test_dedup_bucket_requires_factor_kind():
    f = lambda _p: _p
    with pytest.raises(TypeError):
        dedup_bucket(f)  # missing factor_kind
    bucket = dedup_bucket(f, factor_kind=FactorKind.ALPHA)
    assert isinstance(bucket, str) and bucket


def test_dedup_bucket_dispatches_per_kind():
    f = lambda _p: _p
    a = dedup_bucket(f, factor_kind=FactorKind.ALPHA)
    c = dedup_bucket(f, factor_kind=FactorKind.CONDITION)
    assert a != c  # ALPHA vs CONDITION use different equivalence semantics


# ---------------------------------------------------------------------------
# R10-P0-022: multi-regime fingerprint
# ---------------------------------------------------------------------------

def _regime_flip_factor(threshold: float = 5.0):
    """On days when any column is extreme (|v| > threshold), flip every column.

    Under a Gaussian probe (std 1) extreme days are vanishingly rare -> the
    per-date duplicate decision says "identical"; under a heavy-tailed regime
    extreme days are common -> those dates anti-correlate.  Exactly the
    "identical on Gaussian, different in extreme regimes" pair the review says
    a cold-start library must keep.  (Exact per-hash equality on the Gaussian
    regime is NOT asserted — even one flipped row changes a rank hash; the
    per-date decision is the robust claim.)
    """
    def _fn(p: pd.DataFrame) -> pd.DataFrame:
        arr = p.to_numpy(dtype=float)
        abs_arr = np.abs(arr)
        finite_rows = np.isfinite(abs_arr).any(axis=1)
        extreme = np.zeros(arr.shape[0], dtype=bool)
        extreme[finite_rows] = np.nanmax(abs_arr[finite_rows], axis=1) > threshold
        arr[extreme] = -arr[extreme]
        return pd.DataFrame(arr, index=p.index, columns=p.columns)
    return _fn


def test_gaussian_collision_but_heavy_tail_divergence():
    fixtures = build_typed_fixtures()
    fa = lambda _p: _p
    fb = _regime_flip_factor(threshold=5.0)

    # single Gaussian probe says: duplicate (extreme days too rare to matter)
    assert are_rank_duplicates(fa, fb, panel=fixtures["gaussian"]) is True
    # the same pair under gaussian + heavy_tail with full agreement required
    # is NOT a duplicate: the heavy-tail regime flips too many dates
    assert (
        are_rank_duplicates(
            fa, fb, regimes=("gaussian", "heavy_tail"), min_regime_agreement=1.0
        )
        is False
    )


def test_multi_regime_signature_structured():
    fa = lambda _p: _p
    fb = _regime_flip_factor(threshold=5.0)
    sa = multi_regime_signature(fa, kind=FactorKind.ALPHA)
    sb = multi_regime_signature(fb, kind=FactorKind.ALPHA)
    assert isinstance(sa, FactorBehaviorSignature)
    # the heavy-tail regime robustly differs (the factor flips extreme days);
    # therefore the two fingerprints are not identical across every regime
    assert sa.heavy_tail != sb.heavy_tail
    matched, total = sa.matched_regimes(sb)
    assert matched < total
    assert total == 6


def test_dedup_bucket_default_is_multi_regime():
    fa = lambda _p: _p
    fb = _regime_flip_factor(threshold=5.0)
    # the two buckets differ under the default multi-regime fingerprint
    assert dedup_bucket(fa, factor_kind=FactorKind.ALPHA) != dedup_bucket(
        fb, factor_kind=FactorKind.ALPHA
    )


# ---------------------------------------------------------------------------
# R10-P0-023: per-date, not flattened
# ---------------------------------------------------------------------------

def test_per_date_rejects_dates_that_are_opposite():
    rng = np.random.default_rng(7)
    rows, cols = 40, 8
    base = rng.normal(size=(rows, cols))
    # 36 dates identical, last 4 dates opposite
    flipped = base.copy()
    flipped[-4:] = -base[-4:]
    ra = pd.DataFrame(base).rank(axis=1, method="average").to_numpy()
    rb = pd.DataFrame(flipped).rank(axis=1, method="average").to_numpy()

    # the naive flattened correlation (what the old dedup used) is ~0.8
    flat = np.corrcoef(ra.ravel(), rb.ravel())[0, 1]
    assert flat >= 0.8 - 1e-6  # would pass a lax flattened threshold

    # per-date decision rejects it: the bottom decile is anti-correlated
    assert _per_date_duplicates(ra, rb, rho_threshold=0.8, min_peers=8) is False


def test_per_date_identical_dates_are_duplicates():
    rng = np.random.default_rng(8)
    arr = rng.normal(size=(30, 8))
    ra = pd.DataFrame(arr).rank(axis=1, method="average").to_numpy()
    assert _per_date_duplicates(ra, ra.copy(), rho_threshold=0.99999, min_peers=8) is True


# ---------------------------------------------------------------------------
# R10-P0-032 / 033: typed fixtures
# ---------------------------------------------------------------------------

def test_ohlc_fixture_no_uninitialized_columns():
    fx = build_typed_fixtures(cols=9)  # 9 % 4 == 1 -> leftover column
    ohlc = fx["ohlc"].to_numpy()
    # leftover column (index 8) must be NaN, not garbage
    assert np.isnan(ohlc[:, 8]).all()
    assert np.isfinite(ohlc[:, :8]).all()


def test_event_fixtures_typed():
    fx = build_typed_fixtures()
    eb = fx["event_bool"].to_numpy()
    assert set(np.unique(eb) <= {0.0, 1.0}  # EventBool is 0/1
    si = fx["event_signed_intensity"].to_numpy()
    # signed intensity has negative values where events fire
    assert (si < 0).any() and (si == 0).any()
    pi = fx["event_positive_intensity"].to_numpy()
    # positive intensity is strictly non-negative, non-binary
    assert (pi > 0).any() and (pi > 1.0).any()


# ---------------------------------------------------------------------------
# factor_signatures (R10-P0-028 companion)
# ---------------------------------------------------------------------------

def test_factor_signatures_requires_ast_hash():
    f = lambda _p: _p
    with pytest.raises((TypeError, ValueError)):
        factor_signatures(f)  # ast_hash is a required keyword
    with pytest.raises(ValueError, match="non-empty ast_hash"):
        factor_signatures(f, ast_hash="")
    sigs = factor_signatures(f, ast_hash="sha1")
    assert sigs["ast_hash"] == "sha1"
