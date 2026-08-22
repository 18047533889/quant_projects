# -*- coding: utf-8 -*-
"""Regression tests for intraday profile operator availability declarations.

Verifies that all intraday profile operators in polars_intraday_profile.py
correctly declare `available_at="session_close"` and `same_session_usable=False`.

Uses the OperatorRegistry catalog (pre-loaded by conftest.py) to avoid
double-registration issues from re-importing the module.
"""
from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# The canonical list of operators that MUST have availability declared.
# This acts as a regression guard: if a new operator is added without
# availability fields, the test will fail.
# ---------------------------------------------------------------------------
EXPECTED_INTRADAY_PROFILE_OPERATORS = [
    "intra_volume_profile_cosine",
    "intra_volume_profile_jsd",
    "intra_amount_profile_cosine",
    "intra_amount_profile_jsd",
    "intra_return_profile_cosine",
    "intra_signed_return_profile_cosine",
    "intra_abs_return_profile_cosine",
    "intra_volume_profile_peak_geometry",
    "intra_volume_profile_supply_structure",
    "intra_volume_profile_value_area",
    "intra_volume_at_price_profile",
    "intra_round_price_clustering_share",
    "intra_round_price_barrier_response",
    "intra_bar_range_persistence",
    "intra_bar_range_deviation",
    "intra_consolidation_quality",
    "intra_extreme_bar_return",
    "intra_tail_event_count",
    "intra_tail_volume_share",
    "intra_negative_tail_variation",
    "intra_positive_tail_variation",
    "intra_signed_tail_variation_ratio",
    "intra_price_vwap_max_positive_excursion",
    "intra_price_vwap_max_negative_excursion",
    "intraday_profile_pca_residual",
    "intraday_profile_phase_shift",
    "intraday_profile_surprise_energy",
]


@pytest.fixture(scope="module")
def registry():
    from cleaned_operators.registry import OperatorRegistry
    return OperatorRegistry


def _get_polars_op(registry, name):
    """Get the polars-backend operator, falling back to any backend if polars not found."""
    op = registry.get(name, "polars")
    if op is not None:
        return op
    # Fallback: try without backend filter
    op = registry.get(name)
    return op


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestIntradayProfileAvailability:
    """Verify availability declarations on intraday profile operators."""

    def test_all_expected_operators_present(self, registry):
        """Every expected operator must exist in the registry."""
        for name in EXPECTED_INTRADAY_PROFILE_OPERATORS:
            op = _get_polars_op(registry, name)
            assert op is not None, (
                f"Operator {name!r} not found in registry"
            )

    def test_available_at_session_close(self, registry):
        """Every intraday profile operator must declare available_at='session_close'."""
        for name in EXPECTED_INTRADAY_PROFILE_OPERATORS:
            op = _get_polars_op(registry, name)
            assert op is not None, f"{name} not in registry"
            meta = op.metadata
            assert meta.available_at == "session_close", (
                f"{name}.available_at = {meta.available_at!r}, expected 'session_close'"
            )

    def test_same_session_usable_false(self, registry):
        """Every intraday profile operator must declare same_session_usable=False."""
        for name in EXPECTED_INTRADAY_PROFILE_OPERATORS:
            op = _get_polars_op(registry, name)
            assert op is not None, f"{name} not in registry"
            meta = op.metadata
            assert meta.same_session_usable is False, (
                f"{name}.same_session_usable = {meta.same_session_usable!r}, expected False"
            )

    def test_metadata_fields_exist_on_polars_base(self):
        """OperatorMetadata in base_polars must have available_at and same_session_usable fields."""
        from cleaned_operators.base_polars import OperatorMetadata
        # Should be able to construct with the new fields
        m = OperatorMetadata(
            name="test_op",
            category="intraday",
            available_at="session_close",
            same_session_usable=False,
        )
        assert m.available_at == "session_close"
        assert m.same_session_usable is False
