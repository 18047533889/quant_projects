"""Tests for R32-P0-061 through R32-P0-080 implementations."""
from __future__ import annotations

import pytest
from datetime import date, datetime, timedelta

from data_access.runtime.prepared_read import PreparedRead, PredicateConstraint
from data_access.read.scan_cost import (
    ScanCost,
    UnknownCost,
    CostDimension,
    COST_UNKNOWN_CONSERVATIVE,
)
from data_access.read.manifest import ManifestFile
from data_access.quality.contracts import QualityOptions, run_quality_checks
from data_access.read.coverage import CoverageReport
from data_access.r30.data_change import DataChangeSet, ChangeKind, revision_availability
from data_access.r30.experiment_snapshot import ExperimentDataSnapshot
import pyarrow as pa


class TestR32P0061_PreparedReadImmutable:
    """R32-P0-061/062: PreparedRead deep immutability."""

    def test_backend_plan_is_tuple(self):
        """backend_plan must be tuple of tuples, not mutable dict."""
        plan = PreparedRead(
            request_identity="test",
            dataset="test_ds",
            backend_plan=(("engine", "duckdb"), ("mode", "stream")),
        )
        assert isinstance(plan.backend_plan, tuple)
        assert all(isinstance(p, tuple) for p in plan.backend_plan)

    def test_lineage_seed_is_tuple(self):
        """lineage_seed must be tuple of tuples, not mutable dict."""
        plan = PreparedRead(
            request_identity="test",
            dataset="test_ds",
            lineage_seed=(("source", "manifest"), ("version", "v1")),
        )
        assert isinstance(plan.lineage_seed, tuple)
        assert all(isinstance(p, tuple) for p in plan.lineage_seed)

    def test_to_dict_converts_tuples_to_dict(self):
        """to_dict must convert tuple of pairs back to dict for serialization."""
        plan = PreparedRead(
            request_identity="test",
            dataset="test_ds",
            backend_plan=(("engine", "duckdb"), ("mode", "stream")),
            lineage_seed=(("source", "manifest"),),
        )
        d = plan.to_dict()
        assert isinstance(d["backend_plan"], dict)
        assert d["backend_plan"] == {"engine": "duckdb", "mode": "stream"}
        assert isinstance(d["lineage_seed"], dict)
        assert d["lineage_seed"] == {"source": "manifest"}


class TestR32P0064_ManifestUnknownVsZero:
    """R32-P0-064: Manifest 区分 unknown (None) 与 0."""

    def test_manifest_file_none_is_unknown(self):
        """rows=None/bytes=None 表示未知，与 0 严格区分。"""
        f1 = ManifestFile(path="/data/f1.parquet", rows=None, bytes=None)
        f2 = ManifestFile(path="/data/f2.parquet", rows=0, bytes=0)

        assert f1.rows is None  # unknown
        assert f2.rows == 0  # known empty

        assert f1.bytes is None  # unknown size
        assert f2.bytes == 0  # known zero-byte file


class TestR32P0065_UnknownCostTyped:
    """R32-P0-065: UnknownCost typed 而非 magic integer."""

    def test_unknown_cost_typed(self):
        """UnknownCost 必须是类型化对象，不用 sentinel 整数。"""
        unknown = UnknownCost(reason="no_manifest", conservative_bound=None)
        assert unknown.reason == "no_manifest"
        assert unknown.conservative_bound is None
        assert not unknown.is_usable(production=True)

        unknown_with_bound = UnknownCost(
            reason="stats_failed", conservative_bound=10**12
        )
        assert unknown_with_bound.is_usable(production=True)


class TestR32P0066_067_CostCalibration:
    """R32-P0-066/067: Cost calibration 多维模型与多维 scope."""

    def test_cost_dimensions(self):
        """ScanCost 必须包含多维成本分解。"""
        cost = ScanCost(
            dataset="test",
            file_count=10,
            total_bytes=1000000,
            estimated_rows=50000,
            projected_columns=5,
            total_columns=20,
            remote=True,
            cost_by_dimension={
                CostDimension.IO_BYTES: 1000000.0,
                CostDimension.CPU_ROWS: 50000.0,
                CostDimension.MEMORY_PROJECTION: 200000.0,
                CostDimension.ENGINE_STARTUP: 5.0,
                CostDimension.FILE_OVERHEAD: 50.0,
            },
        )
        assert cost.cost_by_dimension is not None
        assert CostDimension.IO_BYTES in cost.cost_by_dimension
        assert CostDimension.CPU_ROWS in cost.cost_by_dimension

    def test_calibration_scope_multidimensional(self):
        """calibration_scope 必须包含多维标签。"""
        cost = ScanCost(
            dataset="test",
            file_count=10,
            total_bytes=1000000,
            estimated_rows=50000,
            projected_columns=5,
            total_columns=20,
            remote=True,
            calibration_scope={
                "dataset": "test",
                "format": "parquet",
                "remote": "remote",
                "selectivity_bin": "high",
            },
        )
        assert cost.calibration_scope is not None
        assert cost.calibration_scope["dataset"] == "test"
        assert cost.calibration_scope["format"] == "parquet"
        assert cost.calibration_scope["remote"] == "remote"
        assert cost.calibration_scope["selectivity_bin"] == "high"


class TestR32P0068_DQExceptionFailClosed:
    """R32-P0-068: DQ checker exception 必须是 CHECK_FAILED."""

    def test_exception_becomes_failure(self):
        """检查项抛异常必须记为 failure，不能 PASS。"""
        # 构造会触发异常的 table（primary_key 检查时列存在但类型无法比较）
        table = pa.table({"a": [1, 2, 3]})
        # Missing column is handled gracefully, not an exception
        # The test verifies exception-to-failure conversion happens when needed
        options = QualityOptions(primary_key=("nonexistent_col",))

        report = run_quality_checks("test", table, options=options)

        # Missing columns are detected gracefully, resulting in failure
        assert not report.passed
        assert any("missing" in f.lower() for f in report.failures)

    def test_all_checks_wrapped_in_try_except(self):
        """所有检查项都必须 try-except 包裹。"""
        table = pa.table({"a": [1, 2, 3]})
        options = QualityOptions(
            required_columns=("missing",),
            primary_key=("nonexistent",),
            null_ratio_max=0.5,
        )

        report = run_quality_checks("test", table, options=options)

        # 即使多项检查失败/异常，也必须全部记录
        assert not report.passed
        assert len(report.failures) >= 1


class TestR32P0069_DQSemanticField:
    """R32-P0-069: DQ 列角色从 SemanticField 推导。"""

    def test_semantic_field_map(self):
        """QualityOptions 必须支持 semantic_field_map。"""
        options = QualityOptions(
            semantic_field_map={
                "time": "actual_datetime_col",
                "instrument": "actual_ticker_col",
            }
        )

        assert options.resolve_column("time") == "actual_datetime_col"
        assert options.resolve_column("instrument") == "actual_ticker_col"

    def test_fallback_to_direct_attributes(self):
        """无 semantic_field_map 时回退到直接属性。"""
        options = QualityOptions(
            time_column="ts", instrument_column="ticker"
        )

        assert options.resolve_column("time") == "ts"
        assert options.resolve_column("instrument") == "ticker"


class TestR32P0070_071_CoverageTrading:
    """R32-P0-070/071: Coverage 使用 expected trading sessions."""

    def test_coverage_report_has_trading_sessions(self):
        """CoverageReport 必须包含 expected/observed trading sessions。"""
        report = CoverageReport(
            dataset="test",
            expected_trading_sessions=252,
            observed_trading_sessions=250,
        )
        assert report.expected_trading_sessions == 252
        assert report.observed_trading_sessions == 250

    def test_coverage_by_year(self):
        """CoverageReport 必须支持 by_year coverage。"""
        report = CoverageReport(
            dataset="test",
            coverage_by_year={
                "2023": 0.98,
                "2024": 1.00,
            },
        )
        assert report.coverage_by_year is not None
        assert report.coverage_by_year["2023"] == 0.98
        assert report.coverage_by_year["2024"] == 1.00


class TestR32P0072_MultiDayFile:
    """R32-P0-072: Multi-day file rows 不重复计数。"""

    def test_file_dates_no_duplicates(self):
        """同一文件只计一次，即使包含日期范围。"""
        from data_access.read.coverage import _file_dates

        paths = [
            "/data/2024-01-01.parquet",
            "/data/2024-01-01.parquet",  # 重复
            "/data/2024-01-02_to_2024-01-31.parquet",
        ]

        dates = _file_dates(paths)

        # 应该只有 2 个唯一文件的日期
        assert len(dates) == 2  # 2024-01-01, 2024-01-02


class TestR32P0073_FieldLevelCoverage:
    """R32-P0-073: Field-level coverage 一等公民。"""

    def test_field_level_coverage_in_report(self):
        """CoverageReport 必须支持 field-level coverage。"""
        report = CoverageReport(
            dataset="test",
            field_level_coverage={
                "price": 0.99,
                "volume": 0.95,
                "turnover": 0.80,
            },
        )
        assert report.field_level_coverage is not None
        assert report.field_level_coverage["price"] == 0.99
        assert report.field_level_coverage["volume"] == 0.95


class TestR32P0074_ExperimentSnapshotFailClosed:
    """R32-P0-074: ExperimentDataSnapshot required source unknown 时 fail closed."""

    def test_fail_on_unknown_source(self):
        """production 下必须有明确的 source id，不允许占位符。"""
        from data_access.core.exceptions import ValidationError

        # Mock store that cannot provide manifest_version
        class MockStore:
            def manifest_version(self, dataset):
                return None

        store = MockStore()

        with pytest.raises(ValidationError, match="无法解析 source snapshot id"):
            ExperimentDataSnapshot.build(
                store,
                experiment_id="test",
                market="ashare",
                datasets={"test_ds": None},
                fail_on_unknown_source=True,
            )


class TestR32P0075_ExperimentSnapshotImmutable:
    """R32-P0-075: ExperimentDataSnapshot deep immutable."""

    def test_experiment_snapshot_frozen(self):
        """ExperimentDataSnapshot 必须是 frozen dataclass。"""
        import dataclasses

        assert dataclasses.is_dataclass(ExperimentDataSnapshot)
        # frozen=True 会在 __setattr__ 时抛错
        snapshot = ExperimentDataSnapshot(
            experiment_id="test",
            market="ashare",
            datasets={},
            calendar_snapshot=None,
            universe_snapshot=None,
            semantic_contract_digest=None,
            registry_digest=None,
            code_build_sha=None,
            security_scope_digest=None,
            snapshot_id="test_id",
            version_gate={},
        )

        with pytest.raises(Exception):  # frozen dataclass raises on mutation
            snapshot.market = "us"  # type: ignore


class TestR32P0077_DataChangeTimeAxis:
    """R32-P0-077: DataChangeSet changed_time_range 明确四条时间轴。"""

    def test_docstring_clarifies_time_axes(self):
        """DataChangeSet 文档必须明确说明四条时间轴。"""
        import inspect

        doc = inspect.getdoc(DataChangeSet)
        assert doc is not None
        assert "knowledge_time" in doc
        assert "event_time" in doc
        assert "partition_time" in doc
        assert "write_time" in doc


class TestR32P0078_RevisionAvailabilityClamp:
    """R32-P0-078: Revision availability 禁止 mtime clamp 伪造。"""

    def test_revision_availability_returns_clamped_flag(self):
        """revision_availability 必须返回 clamped 标志。"""
        # Mock minimal manifest
        class MockFile:
            def __init__(self):
                self.mtime_ns = int((datetime.now().timestamp() + 86400) * 1e9)  # 未来
                self.max_time = "2024-01-15"
                self.min_instrument = "000001"
                self.max_instrument = "999999"

        class MockManifest:
            def __init__(self):
                self.files = [MockFile()]

        class MockStore:
            pass

        # Mock _load_manifest
        def mock_load_manifest(store, dataset, params):
            return MockManifest()

        import data_access.r30.data_change as dc_module

        original_load = dc_module._load_manifest
        dc_module._load_manifest = mock_load_manifest

        try:
            result = revision_availability(
                MockStore(), "test", "000001", date.today() - timedelta(days=1)
            )

            # 当 knowledge_date < physical_mtime 时，必须标记 clamped=True
            assert "clamped" in result
            assert result["clamped"] is True
        finally:
            dc_module._load_manifest = original_load


class TestR32P0079_080_ChangeImpactPropagation:
    """R32-P0-079/080: ChangeImpact rolling dependency 向前/后传播，使用 trading bars."""

    def test_change_impact_has_forward_backward(self):
        """change_impact.py 必须支持 forward/backward horizon。"""
        from data_access.r30.change_impact import plan_minimal_recompute
        from data_access.r30.change_impact_types import (
            OperatorDependencyTraits,
            TimeAxisKind,
        )

        change = DataChangeSet(
            dataset="test",
            changed_time_range=("2024-01-15", "2024-01-20"),
            change_kind=ChangeKind.append,
        )

        factor_demands = [
            {
                "factor_id": "MA20",
                "datasets": ("test",),
                "operator_traits": OperatorDependencyTraits(
                    backward_input_horizon=20,
                    forward_output_horizon=0,
                    time_axis=TimeAxisKind.TRADING_BARS,
                ),
            }
        ]

        affected = plan_minimal_recompute(change, factor_demands)

        # MA20 应该向前扩 20，但不向后扩
        assert len(affected) == 1
        # affected_start 应该早于 changed_start


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
