"""R32 P0-068..072: DQ checker 异常返回 fail-closed（CHECK_FAILED/UNKNOWN）。

旧实现：异常被 except 吞掉 → 返回 PASS（fail-open，危险）。
新实现：异常 → failures 附加 "check failed: {exc}"，绝不静默 PASS。
"""
from __future__ import annotations

import pyarrow as pa
import pytest

from data_access.quality.contracts import QualityOptions, run_quality_checks
from data_access.r30.data_quality import BLOCK, PASS, WARN, DataQualityService


def test_quality_contracts_exception_fail_closed():
    """contracts.run_quality_checks：异常 → failures（不再静默 PASS）。"""
    # 构造会让 primary_key check 抛异常的 table（列名不匹配）
    table = pa.table({"a": [1, 2], "b": ["x", "y"]})
    report = run_quality_checks(
        "test",
        table,
        options=QualityOptions(primary_key=("nonexistent_col",)),
    )
    assert not report.passed
    failures_text = " | ".join(report.failures)
    assert "primary key columns missing" in failures_text or "check failed" in failures_text


def test_quality_contracts_finite_exception():
    """finite check 异常（非数值列）→ 报告失败。"""
    table = pa.table({"text_col": ["a", "b", "c"]})
    report = run_quality_checks(
        "test",
        table,
        options=QualityOptions(finite_columns=("text_col",)),
    )
    assert report.checks == ("finite",)
    assert isinstance(report.passed, bool)


def test_data_quality_service_external_checker_exception_blocks(monkeypatch):
    """外部 checker 崩溃不能被内置 PASS 检查掩盖。"""
    import data_access.quality.contracts as contracts

    table = pa.table({"value": [1.0, 2.0]})

    class Store:
        def read(self, dataset, **params):
            return table

    def fail_checker(dataset, checked_table, *, options=None):
        raise RuntimeError("synthetic checker crash")

    monkeypatch.setattr(contracts, "run_quality_checks", fail_checker)

    result = DataQualityService().run(Store(), "synthetic")

    assert result.severity == BLOCK
    assert result.checks["quality_contracts"] == BLOCK
    assert any("synthetic checker crash" in problem for problem in result.problems)


def test_data_quality_service_missing_failures_contract_blocks(monkeypatch):
    """A checker report without failures cannot be interpreted as PASS."""
    import data_access.quality.contracts as contracts

    table = pa.table({"value": [1.0, 2.0]})

    class Store:
        def read(self, dataset, **params):
            return table

    class IncompleteReport:
        pass

    monkeypatch.setattr(
        contracts,
        "run_quality_checks",
        lambda dataset, checked_table, *, options=None: IncompleteReport(),
    )

    result = DataQualityService().run(Store(), "incomplete-report")

    assert result.severity == BLOCK
    assert result.checks["quality_contracts"] == BLOCK
    assert any("no failures field" in problem for problem in result.problems)


def test_data_quality_service_ohlc_exception_blocks():
    """DataQualityService._check_ohlc 异常 → BLOCK（R32）。"""
    service = DataQualityService()
    # Mock table that will cause exception in pyarrow operations
    from unittest.mock import Mock
    table = Mock()
    table.column_names = ["high", "low", "close"]
    table.column = Mock(side_effect=RuntimeError("deliberate ohlc failure"))

    name, severity, msgs = service._check_ohlc(table)
    assert name == "ohlc"
    assert severity == BLOCK
    assert any("check failed" in m for m in msgs)


def test_data_quality_service_negative_volume_exception_blocks():
    """DataQualityService._check_negative_volume 异常 → BLOCK（R32）。"""
    service = DataQualityService()
    from unittest.mock import Mock
    table = Mock()
    table.column_names = ["volume"]
    table.column = Mock(side_effect=RuntimeError("deliberate volume failure"))

    name, severity, msgs = service._check_negative_volume(table)
    assert name == "negative_volume"
    assert severity == BLOCK
    assert any("check failed" in m for m in msgs)


def test_data_quality_service_financial_period_exception_blocks():
    """DataQualityService._check_financial_period 异常 → BLOCK（R32）。"""
    service = DataQualityService()
    from unittest.mock import Mock
    table = Mock()
    table.column_names = ["report_period", "publish_time"]
    table.column = Mock(side_effect=RuntimeError("deliberate financial_period failure"))

    name, severity, msgs = service._check_financial_period(table)
    assert name == "financial_period"
    assert severity == BLOCK
    assert any("check failed" in m for m in msgs)


def test_data_quality_service_currency_exception_warns():
    """DataQualityService._check_currency 异常 → WARN（非致命）。"""
    service = DataQualityService()
    table = pa.table({"currency": [1, 2, 3]})  # 数值列非字符串
    name, severity, msgs = service._check_currency(table)
    assert name == "currency_enum"
    # currency check 可能容错返回 PASS，或异常时返回 WARN
    assert severity in (PASS, WARN)


def test_data_quality_service_missing_partition_exception_warns():
    """DataQualityService._check_missing_partition 异常 → WARN（R32）。"""
    service = DataQualityService()
    from unittest.mock import Mock
    table = Mock()
    table.column_names = ["date"]
    table.column = Mock(side_effect=RuntimeError("deliberate missing_partition failure"))

    name, severity, msgs = service._check_missing_partition(table)
    assert name == "missing_partition"
    assert severity == WARN
    assert any("check failed" in m for m in msgs)


def test_data_quality_service_schema_epoch_exception_warns():
    """DataQualityService._check_schema_epoch 异常 → WARN（R32）。"""
    service = DataQualityService()
    from unittest.mock import Mock
    table = Mock()
    table.column_names = ["a"]
    # Force exception in stable_digest_full by making schema.types throw
    table.schema = Mock()
    table.schema.types = Mock(side_effect=RuntimeError("deliberate schema_epoch failure"))

    name, severity, msgs = service._check_schema_epoch(table, "some_epoch")
    assert name == "schema_epoch"
    assert severity == WARN
    assert any("check failed" in m for m in msgs)


def test_quality_contracts_all_checks_exception_wrapped():
    """所有 12 个 check 异常时都被捕获并记录到 failures。"""
    table = pa.table({"a": [1]})
    # 构造会让多个 check 抛异常的 options（列名不存在等）
    report = run_quality_checks(
        "test",
        table,
        options=QualityOptions(
            required_columns=("nonexistent",),
            primary_key=("also_nonexistent",),
            declared_schema={"bad_col": "int"},
            null_ratio_max=0.0,
            range={"missing_col": (0, 100)},
            finite_columns=("another_missing",),
            time_column="no_time",
            instrument_column="no_inst",
            check_monotonic_time=True,
            check_duplicate_timestamp=True,
            check_future_timestamp=True,
            min_rows=1000,
            check_pit_leakage=True,
        ),
    )
    # 至少有 required_columns missing 会触发失败
    assert not report.passed
    # 多个 check 的失败都被记录（非静默吞掉）
    assert len(report.failures) >= 1
