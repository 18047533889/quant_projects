# -*- coding: utf-8 -*-
"""Regression tests for intraday profile operator availability declarations.

Verifies that all intraday profile operators in polars_intraday_profile.py
correctly declare `available_at="session_close"` and `same_session_usable=False`.
"""
from __future__ import annotations

import importlib
import inspect
import sys

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_intraday_profile_module():
    """Import the intraday profile module, forcing a fresh import if cached."""
    mod_name = "cleaned_operators.common.polars_intraday_profile"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    return importlib.import_module(mod_name)


def _collect_intraday_profile_operators():
    """Return list of (name, operator_instance) for all registered intraday profile ops."""
    from cleaned_operators.registry import OperatorRegistry
    mod = _load_intraday_profile_module()
    # Collect class names defined in the module
    classes = [
        obj for name, obj in inspect.getmembers(mod, inspect.isclass)
        if obj.__module__ == mod.__name__
    ]
    results = []
    for cls in classes:
        try:
            instance = cls()
        except Exception:
            continue
        meta = getattr(instance, "metadata", None)
        if meta is None:
            continue
        # Filter to intraday category operators
        if getattr(meta, "category", None) == "intraday":
            results.append((meta.name, instance))
    return results


# The canonical list of operators that MUST have availability declared.
# This acts as a regression guard: if a new operator is added without
# availability fields, the test will fail.
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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestIntradayProfileAvailability:
    """Verify availability declarations on intraday profile operators."""

    def test_all_expected_operators_present(self):
        """Every expected operator must be importable from the module."""
        ops = _collect_intraday_profile_operators()
        found_names = {name for name, _ in ops}
        missing = set(EXPECTED_INTRADAY_PROFILE_OPERATORS) - found_names
        assert not missing, f"Expected operators not found in registry: {missing}"

    def test_available_at_session_close(self):
        """Every intraday profile operator must declare available_at='session_close'."""
        ops = _collect_intraday_profile_operators()
        ops_dict = dict(ops)
        for name in EXPECTED_INTRADAY_PROFILE_OPERATORS:
            instance = ops_dict[name]
            meta = instance.metadata
            assert meta.available_at == "session_close", (
                f"{name}.available_at = {meta.available_at!r}, expected 'session_close'"
            )

    def test_same_session_usable_false(self):
        """Every intraday profile operator must declare same_session_usable=False."""
        ops = _collect_intraday_profile_operators()
        ops_dict = dict(ops)
        for name in EXPECTED_INTRADAY_PROFILE_OPERATORS:
            instance = ops_dict[name]
            meta = instance.metadata
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

    def test_no_unexpected_operators_missing_availability(self):
        """Catch any NEW intraday operator that was added without availability fields."""
        ops = _collect_intraday_profile_operators()
        for name, instance in ops:
            meta = instance.metadata
            if meta.name in EXPECTED_INTRADAY_PROFILE_OPERATORS:
                # Already checked above
                continue
            # Any new intraday operator: availability fields must be set
            assert meta.available_at is not None, (
                f"New intraday operator {name!r} missing available_at"
            )
            assert meta.same_session_usable is not None, (
                f"New intraday operator {name!r} missing same_session_usable"
            )
