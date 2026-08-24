# -*- coding: utf-8 -*-
"""R40 axis-truth items #167/#168/#169/#170/#171/#172/#173/#204."""
from __future__ import annotations

import datetime

import numpy as np
import pandas as pd
import polars as pl
import pytest

from factor_engine.cleaned_operators.common import _polars_bridge as pb


class TestColumnNameInjective:
    def test_collision_rejected(self):
        cols = pd.Index([1, "1"])
        with pytest.raises(pb.ColumnNameCollisionError):
            pb.PhysicalColumnNameMap.validate_injective(cols)

    def test_distinct_names_pass(self):
        out = pb.PhysicalColumnNameMap.validate_injective(pd.Index(["a", "b"]))
        assert out == ("a", "b")


class TestAxisColumnRole:
    def test_wide_value_column_named_date_rejected(self):
        idx = pd.date_range("2024-01-01", periods=3)
        df = pd.DataFrame({"date": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]}, index=idx)
        with pytest.raises(pb.ValueColumnNamedAxisError):
            pb.classify_panel_columns(df)

    def test_wide_panel_columns_are_value(self):
        idx = pd.date_range("2024-01-01", periods=3)
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]}, index=idx)
        roles = pb.classify_panel_columns(df)
        assert set(roles.values()) == {pb.AxisColumnRole.VALUE}

    def test_long_panel_roles(self):
        idx = pd.MultiIndex.from_product(
            [pd.date_range("2024-01-01", periods=3), ["A", "B"]],
            names=["date", "stock_code"],
        )
        df = pd.DataFrame({"x": np.arange(6.0)}, index=idx)
        roles = pb.classify_panel_columns(df)
        assert roles["x"] == pb.AxisColumnRole.VALUE


class TestPanelCompatibilityPolicy:
    def _mk(self, grain="daily", freq="D"):
        return pb.PanelIdentity(
            time_index_hash=1, instrument_axis_hash=2, grain=grain, frequency=freq
        )

    def test_production_rejects_unknown_grain(self):
        a = self._mk(grain="daily")
        b = self._mk(grain="unknown")
        assert not pb.PanelCompatibilityPolicy.compatible(a, b, production=True)
        # research tolerates unknown
        assert pb.PanelCompatibilityPolicy.compatible(a, b, production=False)

    def test_production_requires_equal_grain(self):
        a = self._mk(grain="daily")
        b = self._mk(grain="minute")
        assert not pb.PanelCompatibilityPolicy.compatible(a, b, production=True)
        assert not pb.PanelCompatibilityPolicy.compatible(a, b, production=False)

    def test_production_equal_proven_axes(self):
        a = self._mk(grain="daily", freq="D")
        b = self._mk(grain="daily", freq="D")
        assert pb.PanelCompatibilityPolicy.compatible(a, b, production=True)


class TestPanelCompatibilityPolicyProduction:
    def test_production_panel_unknown_grain_rejected(self):
        # R40 #169: production requires PROVEN+equal grain; unknown is rejected
        # through the policy (not through __eq__, which keeps historical
        # unknown-tolerant semantics for existing identity-contract tests).
        a = pb.PanelIdentity(time_index_hash=1, instrument_axis_hash=2, grain="daily", frequency="D")
        b = pb.PanelIdentity(time_index_hash=1, instrument_axis_hash=2, grain="unknown", frequency="unknown")
        assert not pb.PanelCompatibilityPolicy.compatible(a, b, production=True)
        # research tolerance
        assert pb.PanelCompatibilityPolicy.compatible(a, b, production=False)

    def test_known_grain_mismatch_rejected_both_modes(self):
        a = pb.PanelIdentity(time_index_hash=1, instrument_axis_hash=2, grain="daily", frequency="D")
        b = pb.PanelIdentity(time_index_hash=1, instrument_axis_hash=2, grain="minute", frequency="min")
        assert not pb.PanelCompatibilityPolicy.compatible(a, b, production=True)
        assert not pb.PanelCompatibilityPolicy.compatible(a, b, production=False)


class TestCanonicalAxisLabel:
    def test_supported_types(self):
        assert pb.canonical_axis_label("ABC")[0] == "s"
        assert pb.canonical_axis_label(123)[0] == "i"
        assert pb.canonical_axis_label(datetime.date(2024, 1, 1))[0] == "d"
        assert pb.canonical_axis_label(pd.Timestamp("2024-01-01"))[0] == "t"

    def test_unknown_label_type_rejected(self):
        with pytest.raises(TypeError):
            pb.canonical_axis_label(1.5)  # float is not a canonical axis label
        with pytest.raises(TypeError):
            pb.canonical_axis_label(True)  # bool is not a canonical label
        with pytest.raises(TypeError):
            pb.canonical_axis_label(object())


class TestInstrumentCountUniqueCardinality:
    def test_multiindex_instrument_count_is_unique(self):
        idx = pd.MultiIndex.from_product(
            [pd.date_range("2024-01-01", periods=4), ["A", "B", "C"]],
            names=["date", "stock_code"],
        )
        df = pd.DataFrame({"x": np.arange(12.0)}, index=idx)
        comps = pb._frame_axis_components(df)
        # 7th component is instrument_count (unique instrument cardinality = 3)
        assert comps[6] == 3


class TestPolarsBulkAxisVerification:
    def test_missing_fe_time_hard_fails_in_production(self):
        from factor_engine.backend import panel_polars as pp

        idx = pd.date_range("2024-01-01", periods=3)
        template = pd.DataFrame({"a": [0.0] * 3}, index=idx)
        res = pl.DataFrame({"a": [1.0, 2.0, 3.0]})
        with pytest.raises(pp.AxisMismatchError):
            pp.polars_to_panel(res, template=template, strict=True)
        # research degrades
        out = pp.polars_to_panel(res, template=template, strict=False)
        assert list(out.index) == list(idx)

    def test_reordered_time_axis_detected(self):
        from factor_engine.backend import panel_polars as pp

        idx = pd.date_range("2024-01-01", periods=3)
        template = pd.DataFrame({"a": [0.0] * 3}, index=idx)
        # __fe_time__ re-ordered
        res = pl.DataFrame({"__fe_time__": idx[::-1].to_list(), "a": [1.0, 2.0, 3.0]})
        with pytest.raises(pp.AxisMismatchError):
            pp.polars_to_panel(res, template=template)

    def test_both_paths_share_axis_verification(self):
        from factor_engine.backend import panel_polars as pp

        idx = pd.date_range("2024-01-01", periods=3)
        template = pd.DataFrame({"a": [0.0] * 3}, index=idx)
        # Both bulk and column-loop reject a missing __fe_time__ in strict mode.
        res = pl.DataFrame({"a": [1.0, 2.0, 3.0]})
        with pytest.raises(pp.AxisMismatchError):
            pp._polars_to_panel_column_loop(res, template, strict=True)


class TestFromPandasPanelRowOrder:
    def test_kernel_index_reorder_detected(self):
        idx = pd.date_range("2024-01-01", periods=3)
        base = pl.DataFrame({"__fe_time__": idx.to_list(), "a": [0.0, 0.0, 0.0]})
        out = pd.DataFrame({"a": [3.0, 2.0, 1.0]}, index=idx[::-1])
        with pytest.raises(ValueError):
            pb.from_pandas_panel(base, out)
        # matching index passes
        ok = pd.DataFrame({"a": [1.0, 2.0, 3.0]}, index=idx)
        result = pb.from_pandas_panel(base, ok)
        assert result["a"].to_list() == [1.0, 2.0, 3.0]


class TestExtraOutputColumns:
    def test_extra_output_columns_rejected(self):
        from factor_engine.backend import cleaned_bridge as cb

        idx = pd.date_range("2024-01-01", periods=3)
        template = pd.DataFrame({"a": [0.0] * 3, "b": [0.0] * 3}, index=idx)
        res = pl.DataFrame({"a": [1.0, 2.0, 3.0], "b": [1.0, 2.0, 3.0], "c": [9.0, 9.0, 9.0]})
        with pytest.raises(Exception) as ei:
            cb._validate_no_extra_output_columns(res, template)
        assert "extra output column" in str(ei.value)


class TestGrainFromContract:
    def test_ashare_daily_panel_gets_contract_grain(self):
        idx = pd.date_range("2024-01-01", periods=5)
        # Index inference may or may not guess "daily" depending on the pandas
        # freq inference; the CONTRACT is the authority and always wins.
        df = pd.DataFrame(np.ones((5, 2)), index=idx, columns=["a", "b"])
        df2 = pb.attach_contract_axis(df.copy(), grain="daily", frequency="D")
        grain, freq = pb._pandas_grain_frequency(df2)
        assert grain == "daily"
        assert freq == "D"

    def test_grain_from_contract_not_observed_index(self):
        # A datetime index with freq "D" is inferred "daily", but the CONTRACT
        # overrides it to "minute".
        idx = pd.date_range("2024-01-01", periods=5, freq="D")
        df = pd.DataFrame(np.ones((5, 2)), index=idx, columns=["a", "b"])
        assert pb._pandas_grain_frequency(df)[0] == "daily"
        df2 = pb.attach_contract_axis(df, grain="minute", frequency="min")
        assert pb._pandas_grain_frequency(df2)[0] == "minute"
