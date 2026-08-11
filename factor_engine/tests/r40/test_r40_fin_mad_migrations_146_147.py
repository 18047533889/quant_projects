# -*- coding: utf-8 -*-
"""R40 #146/#147: fin_mad is a deprecated alias only (removed from the mining
surface), and the versioned operator migration table covers every deprecated
compat alias."""
from __future__ import annotations

import pytest


class TestFinMadDeprecatedAlias:
    def test_fin_mad_is_deprecated_alias_only(self):
        # fin_mad must no longer appear in the operator surface (mining search space).
        from cleaned_operators import operator_surface

        assert "fin_mad" not in operator_surface.DAILY_CANONICALS
        assert "fin_mad" not in operator_surface.EXTENDED_ONLY_CANONICALS
        assert "fin_mad" not in operator_surface.DAILY_FACTOR_MIGRATED
        # canonical survives on the surface
        assert "fin_mean_abs_deviation" in operator_surface.EXTENDED_ONLY_CANONICALS

    def test_fin_mad_spec_no_longer_a_canonical(self):
        # fin_mad must NOT be registered as a standalone canonical implementation.
        from cleaned_operators.fundamental.transforms_v2 import _SPECS as pandas_specs

        pandas_names = {name for name, *_ in pandas_specs}
        assert "fin_mad" not in pandas_names
        assert "fin_mean_abs_deviation" in pandas_names

        from cleaned_operators.fundamental.polars_fundamental import _SPECS as polars_specs

        polars_names = {name for name, *_ in polars_specs}
        assert "fin_mad" not in polars_names
        assert "fin_mean_abs_deviation" in polars_names

    def test_fin_mad_formula_resolves_to_canonical(self):
        # Importing transforms_v2 registers the compat alias fin_mad ->
        # fin_mean_abs_deviation (formula parsing keeps working even though
        # fin_mad is no longer a canonical).
        import cleaned_operators.fundamental.transforms_v2 as _t  # noqa: F401

        from cleaned_operators._aliases import MIGRATION_TABLE
        from cleaned_operators.registry import OperatorRegistry

        assert OperatorRegistry._aliases.get("fin_mad") == "fin_mean_abs_deviation"
        assert OperatorRegistry.resolve_canonical("fin_mad") == "fin_mean_abs_deviation"
        rec = next(r for r in MIGRATION_TABLE if r.old_canonical == "fin_mad")
        assert rec.new_canonical == "fin_mean_abs_deviation"
        assert rec.semantic_equivalent is True


class TestMigrationTable:
    def test_migration_table_completeness(self):
        from cleaned_operators._aliases import MIGRATION_TABLE, OperatorMigrationRecord

        old_names = {r.old_canonical for r in MIGRATION_TABLE}
        # Every alias registered via register_compat_alias in the codebase must
        # have a versioned record.
        expected = {
            "fin_mad",
            "ts_matrix_profile_motif_distance",
            "ts_weighted_semivariance_sqrt",
            "ts_garch_vol_forecast",
            "ts_har_rv_next_forecast",
            "ts_har_rv_forecast",
            "ts_har_rv_innovation_z",
        }
        assert old_names == expected, f"missing: {expected - old_names}"

    def test_migration_table_records_well_formed(self):
        from cleaned_operators._aliases import MIGRATION_TABLE

        for rec in MIGRATION_TABLE:
            assert rec.old_canonical and rec.new_canonical
            assert rec.old_canonical != rec.new_canonical
            assert rec.deprecated_since and rec.remove_after
            assert rec.effective_dsl_version
            assert rec.migration_rule in {"rename", "merge", "split"}
