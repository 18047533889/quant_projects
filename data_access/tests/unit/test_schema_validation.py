"""
data_access/tests/unit/test_schema_validation —— PR8 schema 校验单元测试

覆盖：
- _is_compatible 的类型别名匹配
- check_schema 对缺列 / 类型不匹配的检测
- enforce_schema_or_raise 的 strict / warn / off 三态
- SchemaCheckResult.format_message 有用户可读信息
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import pytest

from data_access.core.engine import DuckDBEngine
from data_access.core.exceptions import ValidationError
from data_access.registry import StaticDataset
from data_access.registry.schema_validation import (
    SchemaCheckResult,
    check_schema,
    enforce_schema_or_raise,
    _is_compatible,
    reset_validated_cache,
)


# ---- _is_compatible ---------------------------------------------------------


@pytest.mark.parametrize(
    "declared,actual,expected",
    [
        # 别名：timestamp
        ("timestamp", "TIMESTAMP", True),
        ("timestamp", "TIMESTAMP WITH TIME ZONE", True),
        ("timestamp", "TIMESTAMP_NS", True),
        ("datetime", "TIMESTAMP WITH TIME ZONE", True),
        # 别名：int
        ("int", "BIGINT", True),
        ("int", "INTEGER", True),
        ("int", "HUGEINT", True),
        ("int64", "BIGINT", True),
        ("int64", "INTEGER", False),
        # 别名：string
        ("string", "VARCHAR", True),
        ("str", "VARCHAR", True),
        # 别名：double
        ("double", "DOUBLE", True),
        ("double", "DECIMAL(18,4)", True),
        ("float", "DOUBLE", True),
        # 原生类型：精确 / 参数化
        ("DECIMAL", "DECIMAL(10,2)", True),
        ("VARCHAR", "VARCHAR", True),
        # 不兼容
        ("int", "DOUBLE", False),
        ("string", "BIGINT", False),
        ("double", "BIGINT", False),
        ("timestamp", "VARCHAR", False),
        # 边界
        ("", "VARCHAR", False),
        ("string", "", False),
    ],
)
def test_is_compatible(declared: str, actual: str, expected: bool):
    assert _is_compatible(declared, actual) is expected


# ---- check_schema / enforce_schema_or_raise --------------------------------


@pytest.fixture
def tiny_parquet_dataset(tmp_path: Path) -> StaticDataset:
    """造一个微型 parquet 目录，返回对应的 StaticDataset。

    schema：ticker (VARCHAR), close (DOUBLE), volume (BIGINT)
    """
    root = tmp_path / "tiny"
    root.mkdir()
    pd.DataFrame(
        {
            "ticker": ["AAPL", "MSFT"],
            "close": [190.0, 380.0],
            "volume": [1000, 2000],
        }
    ).to_parquet(root / "data.parquet")
    return StaticDataset(
        name="tiny",
        access_mode="published",
        layout="plain",
        time_column="ticker",  # 无所谓，schema 校验不关心
        instrument_column="ticker",
        hive_partitioning=False,
        union_by_name=True,
        root=root,
        glob="*.parquet",
        schema={"ticker": "string", "close": "double", "volume": "int"},
    )


@pytest.fixture
def engine():
    eng = DuckDBEngine(threads=2)
    yield eng
    eng.close()
    reset_validated_cache()


def test_check_schema_ok(engine, tiny_parquet_dataset):
    paths = tiny_parquet_dataset.resolve_paths()
    result = check_schema(engine, tiny_parquet_dataset, paths)
    assert result.ok is True
    assert result.missing == ()
    assert result.type_mismatch == ()


def test_check_schema_detects_missing_column(engine, tiny_parquet_dataset):
    # 把 schema 改成带不存在的列
    bad_ds = StaticDataset(
        name=tiny_parquet_dataset.name,
        access_mode=tiny_parquet_dataset.access_mode,
        layout=tiny_parquet_dataset.layout,
        time_column=tiny_parquet_dataset.time_column,
        instrument_column=tiny_parquet_dataset.instrument_column,
        hive_partitioning=tiny_parquet_dataset.hive_partitioning,
        union_by_name=tiny_parquet_dataset.union_by_name,
        root=tiny_parquet_dataset.root,
        glob=tiny_parquet_dataset.glob,
        schema={"ticker": "string", "not_there": "double"},
    )
    result = check_schema(engine, bad_ds, bad_ds.resolve_paths())
    assert result.ok is False
    assert "not_there" in result.missing
    assert result.type_mismatch == ()


def test_check_schema_detects_type_mismatch(engine, tiny_parquet_dataset):
    bad_ds = StaticDataset(
        name=tiny_parquet_dataset.name,
        access_mode=tiny_parquet_dataset.access_mode,
        layout=tiny_parquet_dataset.layout,
        time_column=tiny_parquet_dataset.time_column,
        instrument_column=tiny_parquet_dataset.instrument_column,
        hive_partitioning=tiny_parquet_dataset.hive_partitioning,
        union_by_name=tiny_parquet_dataset.union_by_name,
        root=tiny_parquet_dataset.root,
        glob=tiny_parquet_dataset.glob,
        schema={"ticker": "string", "close": "int"},  # close 实际是 DOUBLE
    )
    result = check_schema(engine, bad_ds, bad_ds.resolve_paths())
    assert result.ok is False
    mismatch_cols = [m[0] for m in result.type_mismatch]
    assert "close" in mismatch_cols


def test_check_schema_empty_declaration_is_ok(engine, tiny_parquet_dataset):
    """schema 未声明 → 直接 ok，不读 parquet。"""
    ds_no_schema = StaticDataset(
        name="no_schema",
        access_mode="published",
        layout="plain",
        time_column="ticker",
        instrument_column="ticker",
        hive_partitioning=False,
        union_by_name=True,
        root=tiny_parquet_dataset.root,
        glob="*.parquet",
        schema={},
    )
    result = check_schema(engine, ds_no_schema, ds_no_schema.resolve_paths())
    assert result.ok is True


# ---- enforce_schema_or_raise ------------------------------------------------


def test_enforce_ok_result_no_op():
    ok = SchemaCheckResult(ok=True, dataset="x")
    # 不抛不打
    enforce_schema_or_raise(ok, mode="strict")
    enforce_schema_or_raise(ok, mode="warn")
    enforce_schema_or_raise(ok, mode="off")


def test_enforce_strict_raises():
    bad = SchemaCheckResult(
        ok=False,
        dataset="x",
        missing=("col_x",),
        actual_schema=(("foo", "BIGINT"),),
    )
    with pytest.raises(ValidationError, match="col_x"):
        enforce_schema_or_raise(bad, mode="strict")


def test_enforce_warn_logs_does_not_raise(caplog):
    bad = SchemaCheckResult(
        ok=False,
        dataset="x",
        missing=("col_x",),
        actual_schema=(("foo", "BIGINT"),),
    )
    caplog.set_level(logging.WARNING, logger="data_access.schema_validation")
    enforce_schema_or_raise(bad, mode="warn")
    assert any("col_x" in r.getMessage() for r in caplog.records)


def test_enforce_off_silent(caplog):
    bad = SchemaCheckResult(ok=False, dataset="x", missing=("col_x",))
    caplog.set_level(logging.WARNING, logger="data_access.schema_validation")
    enforce_schema_or_raise(bad, mode="off")
    assert not any("col_x" in r.getMessage() for r in caplog.records)


def test_format_message_mentions_fix_hint():
    bad = SchemaCheckResult(
        ok=False,
        dataset="factor_lake",
        missing=("value",),
        type_mismatch=(("close", "int", "DOUBLE"),),
        actual_schema=(("close", "DOUBLE"),),
    )
    msg = bad.format_message()
    assert "factor_lake" in msg
    assert "value" in msg
    assert "close" in msg and "DOUBLE" in msg
    assert "datasets.yaml" in msg  # 修复指引
