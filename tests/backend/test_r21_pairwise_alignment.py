# -*- coding: utf-8 -*-
"""R21-P0-PAIRWISE-ALIGNMENT: regression tests for the pairwise panel alignment contract.

Four tests that must all pass before the contract is considered wired:

1. Wide-panel exact alignment (pandas) — mismatched columns raise.
2. Long-panel exact alignment (Polars) — missing counterpart column raises.
3. Missing counterpart column raises PanelSchemaMismatchError (fail-closed).
4. Exact match passes through without error.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import polars as pl

from backend.pairwise_alignment import (
    AlignmentMode,
    PanelSchemaMismatchError,
    PairwiseAlignmentSpec,
    assert_long_frames_exact,
    assert_wide_pairwise_aligned,
    pairwise_alignment_evidence,
    pairwise_alignment_spec_for,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_IDX_A = pd.date_range("2024-01-01", periods=5, freq="B")
_COLS_A = ["AAPL", "MSFT", "GOOG"]
_COLS_B = ["AAPL", "MSFT"]  # missing GOOG

_RNG = np.random.default_rng(42)


def _wide_a() -> pd.DataFrame:
    return pd.DataFrame(
        _RNG.standard_normal((len(_IDX_A), len(_COLS_A))),
        index=_IDX_A,
        columns=_COLS_A,
    )


def _wide_b() -> pd.DataFrame:
    return pd.DataFrame(
        _RNG.standard_normal((len(_IDX_A), len(_COLS_B))),
        index=_IDX_A,
        columns=_COLS_B,
    )


def _wide_b_shifted() -> pd.DataFrame:
    """Same columns but shifted date axis."""
    idx = _IDX_A + pd.Timedelta(days=1)
    return pd.DataFrame(
        _RNG.standard_normal((len(idx), len(_COLS_A))),
        index=idx,
        columns=_COLS_A,
    )


def _long_base(lf: pl.LazyFrame | None = None) -> pl.LazyFrame:
    """Return a standard 5-date x 3-instrument long frame."""
    if lf is None:
        lf = pl.DataFrame(
            {
                "ts": [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5],
                "inst": ["A", "B", "C"] * 5,
                "_v": list(range(15)),
            }
        ).lazy()
    return lf


def _long_missing_c() -> pl.LazyFrame:
    """Long frame missing instrument C on all dates."""
    return pl.DataFrame(
        {
            "ts": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
            "inst": ["A", "B"] * 5,
            "_v": list(range(10)),
        }
    ).lazy()


# ---------------------------------------------------------------------------
# 1. Wide-panel exact alignment — mismatched columns raise
# ---------------------------------------------------------------------------


class TestWidePanelExactAlignment:
    def test_mismatched_columns_raise(self):
        """Missing counterpart column (GOOG only in left) → PanelSchemaMismatchError."""
        left = _wide_a()
        right = _wide_b()
        with pytest.raises(PanelSchemaMismatchError, match="instrument axis mismatch"):
            assert_wide_pairwise_aligned(
                left, right, canonical="ts_corr",
            )

    def test_mismatched_date_axis_raise(self):
        """Shifted date axis → PanelSchemaMismatchError."""
        left = _wide_a()
        right = _wide_a().copy()
        right.index = right.index + pd.Timedelta(days=1)
        with pytest.raises(PanelSchemaMismatchError, match="date axis mismatch"):
            assert_wide_pairwise_aligned(
                left, right, canonical="ts_corr",
            )

    def test_exact_match_passes(self):
        """Identical panels pass through without error."""
        left = _wide_a()
        right = _wide_a()
        # should not raise
        assert_wide_pairwise_aligned(left, right, canonical="ts_corr")


# ---------------------------------------------------------------------------
# 2. Long-panel exact alignment — missing counterpart column raises
# ---------------------------------------------------------------------------


class TestLongPanelExactAlignment:
    def test_missing_counterpart_column_raises(self):
        """Left has inst C, right does not → PanelSchemaMismatchError."""
        left = _long_base()
        right = _long_missing_c()
        with pytest.raises(PanelSchemaMismatchError, match="missing"):
            assert_long_frames_exact(
                left, right, canonical="ts_corr",
            )

    def test_exact_key_set_passes(self):
        """Identical key sets pass through."""
        left = _long_base()
        right = _long_base()
        # should not raise
        assert_long_frames_exact(left, right, canonical="ts_corr")


# ---------------------------------------------------------------------------
# 3. Missing counterpart column → PanelSchemaMismatchError (fail-closed)
# ---------------------------------------------------------------------------


class TestFailClosedOnMissing:
    def test_left_join_alignment_research_only(self):
        """LEFT_JOIN_ALIGNMENT mode skips the check (research only)."""
        spec = PairwiseAlignmentSpec(mode=AlignmentMode.LEFT_JOIN_ALIGNMENT)
        left = _long_base()
        right = _long_missing_c()
        # should not raise under research mode
        assert_long_frames_exact(
            left, right, canonical="ts_corr", spec=spec,
        )

    def test_exact_alignment_is_default(self):
        """Default spec for ts_corr is EXACT_ALIGNMENT."""
        spec = pairwise_alignment_spec_for("ts_corr")
        assert spec.mode == AlignmentMode.EXACT_ALIGNMENT

    def test_panel_schema_mismatch_error_is_value_error(self):
        """PanelSchemaMismatchError is a ValueError (fail-closed, not a warning)."""
        assert issubclass(PanelSchemaMismatchError, ValueError)


# ---------------------------------------------------------------------------
# 4. Exact match passes through without error
# ---------------------------------------------------------------------------


class TestExactMatchPasses:
    def test_checked_axes_includes_all_six(self):
        """The default EXACT spec enables all six checks."""
        spec = PairwiseAlignmentSpec()
        axes = spec.checked_axes()
        for name in ("date_axis", "instrument_axis", "ordering",
                     "universe_snapshot", "grain", "session"):
            assert name in axes

    def test_evidence_payload_is_dict(self):
        """Evidence helper returns a JSON-serialisable dict."""
        ev = pairwise_alignment_evidence()
        assert isinstance(ev, dict)
        assert ev["mode"] == "exact_alignment"
        assert "ts_corr" in ev["canon_map"]
