from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def test_yaml_dialect_version_is_validated_and_bridged(tmp_path: Path) -> None:
    from runtime.config import load_config
    path = tmp_path / "factor.yaml"
    path.write_text(
        """
factor:
  name: x
  expr: safe_log(close)
  surface: daily
  dialect: lqtp
  dialect_version: '2026-07-19'
data_source:
  type: data_access
  dataset: ashare_stock_daily
""",
        encoding="utf-8",
    )
    cfg = load_config(path)
    assert cfg.factor.dialect == "lqtp"
    assert cfg.factor.dialect_version == "2026-07-19"
    assert cfg.factor.surface == "lqtp"  # bridge for existing from_loaded_config API

    path.write_text(path.read_text().replace("2026-07-19", "2025-01-01"), encoding="utf-8")
    with pytest.raises(ValueError, match="dialect_version"):
        load_config(path)


def test_source_ref_disables_full_sql_pushdown() -> None:
    from api.source_ref import source_col
    from planner.logical_plan import PlanNode
    from planner.sql_lowerer import lower_to_physical_plan

    ref = source_col("BenchmarkIndexDailyBar", "Close", index="000985.SH")
    name = ref.name
    node = PlanNode(
        op="add",
        inputs=(
            PlanNode(op="column", inputs=(), attrs={"name": name}),
            PlanNode(op="literal", inputs=(), attrs={"value": 1.0}),
        ),
        attrs={},
    )
    physical = lower_to_physical_plan(node, mode="research")
    assert physical.fully_sql is False


def test_financial_lag_updates_when_old_quarter_is_revised() -> None:
    from storage.sources.lqtp_logical_source_v2 import LQTPLogicalDataSource

    class DummyInner:
        start_date = None
        end_date = None

    class FixtureSource(LQTPLogicalDataSource):
        def _anchor_index(self):
            return pd.MultiIndex.from_tuples(
                [
                    (pd.Timestamp("2024-08-01"), "A"),
                    (pd.Timestamp("2024-08-20"), "A"),
                ],
                names=["timestamp", "instrument"],
            )

        def _financial_raw(self, dataset: str, field: str):
            return pd.DataFrame(
                {
                    "Symbol": ["A", "A", "A"],
                    "ReportPeriodEndDate": ["2024-03-31", "2024-06-30", "2024-03-31"],
                    "PubDate": ["2024-04-30", "2024-07-31", "2024-08-15"],
                    field: [10.0, 20.0, 11.0],
                }
            )

    source = FixtureSource(DummyInner())
    out = source._financial("dummy", "Metric", "financial_lag", {"quarters": 1})
    np.testing.assert_allclose(out.to_numpy(), [10.0, 11.0], equal_nan=True)


def test_minute_resample_does_not_silently_collapse_to_daily() -> None:
    from storage.sources.lqtp_logical_source_v2 import LQTPLogicalDataSource
    from storage.sources.data_access_source import MissingDataDependencyError

    class DummyInner:
        pass

    source = LQTPLogicalDataSource(DummyInner(), factor_freq="1d")
    with pytest.raises(MissingDataDependencyError, match="intraday bar sequence"):
        source._minute_daily("Close", "minute_resample", {"period": 5})


def test_multiminute_vwap_is_fail_closed_until_weighted_aggregation() -> None:
    from storage.sources.lqtp_logical_source_v2 import LQTPLogicalDataSource
    from storage.sources.data_access_source import MissingDataDependencyError

    class DummyInner:
        pass

    source = LQTPLogicalDataSource(DummyInner(), factor_freq="1d")
    with pytest.raises(MissingDataDependencyError, match="weighted aggregation"):
        source._minute_daily("Vwap", "minute_bar", {"period": 5, "index": 0})


def test_blocked_lqtp_names_are_classified_not_unknown() -> None:
    from api.dsl_parser import DSLParseError, parse_expr
    for formula in [
        "l2_sum(close)",
        "l2_count(close)",
        "group_minmax(close, industry)",
        "group_quantile_mask(close, industry, 0.2, 0.8)",
    ]:
        with pytest.raises(DSLParseError, match="recognized but not executable"):
            parse_expr(formula, surface="lqtp")


def test_machine_readable_manifest_distinguishes_blocked_from_production() -> None:
    from api.lqtp_capabilities import build_lqtp_capability_manifest
    manifest = build_lqtp_capability_manifest()
    assert manifest["dialect_version"] == "2026-07-19"
    assert "ts_sumac" in manifest["recognized_blocked"]
    assert "l2_sum" in manifest["recognized_blocked"]
    assert manifest["canonical_operators"]["ts_mean"]["production_allowed"] is True
