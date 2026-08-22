# -*- coding: utf-8 -*-
"""R39 PERF-029 regression tests: AxisIdentityCertificate cache + verify fast path.

The certificate cache must:
  (a) compute the (expensive) axis fingerprint ONCE per frame and reuse it on
      subsequent verifies (hit counter grows, compute_count stays put);
  (b) make equal-axis frames verify TRUE entirely through the cached
      certificates;
  (c) make axis-changed frames verify FALSE and RE-COMPUTE the certificate
      instead of trusting a stale one;
  (d) be verdict-parity with the pre-R39 slow path (``PanelIdentity.from_frame``
      equality) on random same/different-axis pairs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cleaned_operators.axis_identity_certificate import (
    AxisIdentityCertificate,
    axis_identity_certificate_hit,
    axis_identity_compute_count,
    axis_identity_counters,
    axis_identity_ms,
    certificate_for_frame,
    clear_certificates,
    reset_axis_identity_counters,
)
from cleaned_operators.common._polars_bridge import (
    PanelIdentity,
    verify_frames_share_identity,
)

pytest.importorskip("pandas")
try:
    import polars as pl

    HAS_POLARS = True
except ImportError:  # pragma: no cover
    pl = None
    HAS_POLARS = False


def _panel(n: int = 20, seed: int = 0, cols: tuple[str, ...] = ("A", "B"), start="2024-01-02") -> pd.DataFrame:
    dates = pd.bdate_range(start, periods=n)
    return pd.DataFrame(
        np.random.default_rng(seed).normal(0, 1, (n, len(cols))),
        index=dates,
        columns=list(cols),
    )


def _pl_wide(pdf: pd.DataFrame, time_col: str = "__fe_time__") -> "pl.DataFrame":
    return pl.DataFrame(
        {time_col: pl.Series(pd.DatetimeIndex(pdf.index)), **{c: pdf[c].to_numpy() for c in pdf.columns}}
    )


@pytest.fixture(autouse=True)
def _reset_counters():
    reset_axis_identity_counters()
    yield
    reset_axis_identity_counters()


# ---------------------------------------------------------------------------
# (a) compute once per frame, then cache hits
# ---------------------------------------------------------------------------
def test_certificate_computed_once_then_verify_hits_cache():
    a = _panel(seed=1)
    b = _panel(seed=2)  # same date axis, different values -> same identity
    verify_frames_share_identity("t", a, b)
    assert axis_identity_compute_count() == 2
    assert axis_identity_certificate_hit() == 0
    # second verify must reuse both cached certificates, not re-hash.
    verify_frames_share_identity("t", a, b)
    assert axis_identity_compute_count() == 2
    assert axis_identity_certificate_hit() == 2


def test_certificate_for_frame_computes_once():
    a = _panel(seed=1)
    c1 = certificate_for_frame(a)
    c2 = certificate_for_frame(a)
    assert c1.axis_hash == c2.axis_hash
    assert axis_identity_compute_count() == 1
    assert axis_identity_certificate_hit() == 1
    assert c1.datetime_count == len(a)
    assert c1.instrument_count == 2


def test_certificate_fields_match_spec():
    from dataclasses import fields

    names = {f.name for f in fields(AxisIdentityCertificate)}
    for required in (
        "source_snapshot_id",
        "axis_hash",
        "datetime_count",
        "instrument_count",
        "sortedness",
    ):
        assert required in names, required


def test_axis_identity_ms_proxy_tracks_compute():
    a = _panel(seed=1)
    certificate_for_frame(a)
    compute = axis_identity_compute_count()
    assert compute == 1
    assert axis_identity_ms() == pytest.approx(float(compute) * 1.0)
    counters = axis_identity_counters()
    assert counters["axis_identity_compute_count"] == compute


# ---------------------------------------------------------------------------
# (b) equal-axis frames verify TRUE via the cached certificates
# ---------------------------------------------------------------------------
def test_equal_axis_verify_true_via_cache():
    a = _panel(seed=1)
    b = _panel(seed=2)
    verify_frames_share_identity("t", a, b)  # prime cache (2 computes)
    assert axis_identity_compute_count() == 2
    hits_before = axis_identity_certificate_hit()
    verify_frames_share_identity("t", a, b)  # must be the fast path
    assert axis_identity_compute_count() == 2  # no recompute
    assert axis_identity_certificate_hit() == hits_before + 2
    assert PanelIdentity.from_frame(a) == PanelIdentity.from_frame(b)


def test_equal_axis_three_panels_fast():
    a = _panel(seed=1)
    b = _panel(seed=2)
    c = _panel(seed=3)
    verify_frames_share_identity("t", a, b, c)
    verify_frames_share_identity("t", a, b, c)
    assert axis_identity_compute_count() == 3
    assert axis_identity_certificate_hit() == 3  # second verify: 3 cache hits


# ---------------------------------------------------------------------------
# (c) axis-changed frames verify FALSE; the certificate is recomputed, not stale
# ---------------------------------------------------------------------------
def test_axis_changed_fresh_frame_verify_false_and_recomputed():
    a = _panel(seed=1)
    shifted = _panel(seed=1)
    shifted.index = a.index + pd.Timedelta(days=1)  # different FRAME object
    verify_frames_share_identity("t", a, a)  # prime a's certificate
    with pytest.raises(ValueError, match="PanelIdentity"):
        verify_frames_share_identity("t", a, shifted)
    # shifted never carried a certificate -> computed fresh (not stale).
    cert_a = certificate_for_frame(a)
    cert_shifted = certificate_for_frame(shifted)
    assert cert_shifted.axis_hash != cert_a.axis_hash


def test_axis_changed_in_place_recomputes_stale_certificate():
    a = _panel(seed=1)
    b = _panel(seed=2)
    verify_frames_share_identity("t", a, b)  # both get certificates (compute=2)
    assert axis_identity_compute_count() == 2
    # Mutate b IN PLACE: shift its date axis one day.  The cached certificate
    # is now stale (first/last time value changed) and must be recomputed.
    b.index = b.index + pd.Timedelta(days=1)
    with pytest.raises(ValueError, match="PanelIdentity"):
        verify_frames_share_identity("t", a, b)
    assert axis_identity_compute_count() >= 3  # b's stale cert recomputed
    cert_a = certificate_for_frame(a)
    cert_b = certificate_for_frame(b)
    assert cert_b.axis_hash != cert_a.axis_hash
    assert cert_b.first_time_value == b.index[0]


def test_known_frequency_mismatch_verify_false():
    """R9-P0-014: same date values but a KNOWN frequency mismatch must reject.

    The certificate's axis_hash folds grain/frequency so the fast path never
    declares two frames equal when ``PanelIdentity.__eq__`` would reject a known
    metadata mismatch (a daily panel and a business-day panel sharing dates by
    accident must loud-fail).
    """
    d_daily = pd.DatetimeIndex(["2024-01-02", "2024-01-03", "2024-01-04"], freq="D")
    d_b = pd.DatetimeIndex(["2024-01-02", "2024-01-03", "2024-01-04"], freq="B")
    a = pd.DataFrame(np.ones((3, 2)), index=d_daily, columns=["A", "B"])
    b = pd.DataFrame(np.ones((3, 2)), index=d_b, columns=["A", "B"])
    assert PanelIdentity.from_frame(a) != PanelIdentity.from_frame(b)
    verify_frames_share_identity("t", a, a)  # prime a's certificate
    with pytest.raises(ValueError, match="PanelIdentity"):
        verify_frames_share_identity("t", a, b)


def test_axis_changed_permuted_columns_verify_false():
    a = _panel(seed=1)
    permuted = a[["B", "A"]]
    verify_frames_share_identity("t", a, a)
    with pytest.raises(ValueError, match="PanelIdentity"):
        verify_frames_share_identity("t", a, permuted)


def test_certificate_not_propagated_through_transform():
    """An axis-preserving polars transform returns a NEW frame with no cert."""
    a = _panel(seed=1)
    pa = _pl_wide(a)
    certificate_for_frame(pa)
    assert getattr(pa, "_r39_axis_certificate", None) is not None
    out = pa.with_columns((pl.col("A") * 2.0).alias("A"))
    assert getattr(out, "_r39_axis_certificate", None) is None
    # The new frame's axis is unchanged, so it still verifies equal to pa.
    verify_frames_share_identity("t", pa, out)
    assert axis_identity_compute_count() == 2  # pa cached + out computed fresh


# ---------------------------------------------------------------------------
# (d) verdict parity: fast path == slow path on 50 random pairs
# ---------------------------------------------------------------------------
def test_verdict_parity_fast_vs_slow_50_random():
    rng = np.random.default_rng(12345)
    fast_fast_hits = 0
    for trial in range(50):
        n = int(rng.integers(5, 40))
        cols = list("ABC")
        idx = pd.bdate_range("2024-01-02", periods=n)
        a = pd.DataFrame(rng.normal(0, 1, (n, 3)), index=idx, columns=cols)
        if rng.integers(0, 2) == 0:
            # same axis -> today's slow path says EQUAL
            b = pd.DataFrame(rng.normal(0, 1, (n, 3)), index=idx, columns=cols)
        else:
            # shifted axis -> today's slow path says DIFFERENT
            b = pd.DataFrame(
                rng.normal(0, 1, (n, 3)),
                index=idx + pd.Timedelta(days=int(rng.integers(1, 5))),
                columns=cols,
            )
        slow_equal = bool(PanelIdentity.from_frame(a) == PanelIdentity.from_frame(b))

        hits_before = axis_identity_certificate_hit()
        compute_before = axis_identity_compute_count()
        certificate_for_frame(a)
        certificate_for_frame(b)
        try:
            verify_frames_share_identity("parity", a, b)
            fast_equal = True
        except ValueError:
            fast_equal = False
        assert fast_equal == slow_equal, f"parity mismatch on trial {trial}"

        # When the pair is genuinely equal, the second verify must have been
        # served entirely from cached certificates (no re-hash).
        if slow_equal:
            assert axis_identity_certificate_hit() >= hits_before + 2
            assert axis_identity_compute_count() == compute_before + 2
            fast_fast_hits += 1
        else:
            # Mismatch falls back to the authoritative slow path (never trusts
            # the certificates for a "different" verdict on their own).
            assert axis_identity_compute_count() > compute_before
    assert fast_fast_hits >= 10  # both classes well represented


# ---------------------------------------------------------------------------
# polars path (optional dependency)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not HAS_POLARS, reason="polars not installed")
def test_polars_verify_fast_path():
    a = _panel(seed=1)
    b = _panel(seed=2)
    pa, pb = _pl_wide(a), _pl_wide(b)
    verify_frames_share_identity("t", pa, pb)
    assert axis_identity_compute_count() == 2
    verify_frames_share_identity("t", pa, pb)
    assert axis_identity_compute_count() == 2
    assert axis_identity_certificate_hit() == 2


@pytest.mark.skipif(not HAS_POLARS, reason="polars not installed")
def test_polars_axis_changed_false():
    a = _panel(seed=1)
    shifted = _panel(seed=1)
    shifted.index = a.index + pd.Timedelta(days=1)
    pa, ps = _pl_wide(a), _pl_wide(shifted)
    verify_frames_share_identity("t", pa, pa)
    with pytest.raises(ValueError, match="PanelIdentity"):
        verify_frames_share_identity("t", pa, ps)


# ---------------------------------------------------------------------------
# clear_certificates helper
# ---------------------------------------------------------------------------
def test_clear_certificates_forces_recompute():
    a = _panel(seed=1)
    b = _panel(seed=2)
    verify_frames_share_identity("t", a, b)
    assert axis_identity_compute_count() == 2
    clear_certificates(a, b)
    verify_frames_share_identity("t", a, b)
    assert axis_identity_compute_count() == 4  # recomputed after clear
