"""
contract 测试：PR8 store 首访 schema 自检

覆盖：
- schema 正确时 read_arrow / read_arrow_stream / scan_polars 正常
- schema 声明漂移（strict）→ ValidationError
- schema 声明漂移（warn）→ 日志告警 + 正常返回数据
- schema 声明漂移（off）→ 完全静默
- 同一 dataset 校验只跑一次（读第二次不再 DESCRIBE）
- 空 schema 声明不触发校验
"""
from __future__ import annotations

import logging
from pathlib import Path
from textwrap import dedent
from unittest.mock import patch

import pandas as pd
import pytest

from data_access import reset_store
from data_access.engine import DuckDBEngine
from data_access.exceptions import ValidationError
from data_access.registry import load_registry
from data_access.schema_validation import reset_validated_cache
from data_access.store import DataAccessStore


def _write_yaml(path: Path, content: str) -> Path:
    path.write_text(dedent(content).strip() + "\n", encoding="utf-8")
    return path


def _make_data(root: Path) -> None:
    """造一个 parquet 目录：ticker (str), close (double), volume (int64)。"""
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "ticker": ["AAPL", "MSFT", "GOOG"],
            "close": [190.0, 380.0, 145.0],
            "volume": [1_000_000, 800_000, 2_000_000],
            "align_time": pd.to_datetime(
                ["2024-01-02 09:30", "2024-01-02 09:30", "2024-01-02 09:30"]
            ),
        }
    ).to_parquet(root / "data.parquet")


@pytest.fixture
def store_with_correct_schema(tmp_path, monkeypatch):
    """schema 声明与实际 parquet 一致。"""
    data_root = tmp_path / "data"
    _make_data(data_root)
    monkeypatch.setenv("TEST_DATA_ROOT", str(data_root))

    yaml_path = _write_yaml(
        tmp_path / "datasets.yaml",
        """
        stocks:
          kind: static
          access_mode: published
          layout: plain
          root: ${TEST_DATA_ROOT}
          glob: "*.parquet"
          time_column: align_time
          instrument_column: ticker
          hive_partitioning: false
          union_by_name: true
          schema:
            ticker: string
            close: double
            volume: int
            align_time: timestamp
        """,
    )
    reset_store()
    reset_validated_cache()
    reg = load_registry(yaml_path)
    engine = DuckDBEngine(threads=2)
    store = DataAccessStore(registry=reg, engine=engine)
    yield store
    engine.close()
    reset_store()


@pytest.fixture
def store_with_wrong_schema(tmp_path, monkeypatch):
    """schema 声明列名错了，触发首访失败。"""
    data_root = tmp_path / "data"
    _make_data(data_root)
    monkeypatch.setenv("TEST_DATA_ROOT", str(data_root))

    yaml_path = _write_yaml(
        tmp_path / "datasets.yaml",
        """
        stocks:
          kind: static
          access_mode: published
          layout: plain
          root: ${TEST_DATA_ROOT}
          glob: "*.parquet"
          time_column: align_time
          instrument_column: ticker
          hive_partitioning: false
          union_by_name: true
          schema:
            ticker: string
            close: int                 # 实际是 DOUBLE
            bogus_column: string       # 实际不存在
        """,
    )
    reset_store()
    reset_validated_cache()
    reg = load_registry(yaml_path)
    engine = DuckDBEngine(threads=2)
    store = DataAccessStore(registry=reg, engine=engine)
    yield store
    engine.close()
    reset_store()


# ---- 正常路径 ---------------------------------------------------------------


def test_read_arrow_with_correct_schema(store_with_correct_schema):
    tbl = store_with_correct_schema.read_arrow("stocks", columns=["ticker", "close"])
    assert tbl.num_rows == 3


def test_stream_with_correct_schema(store_with_correct_schema):
    batches = list(
        store_with_correct_schema.read_arrow_stream(
            "stocks", columns=["ticker", "close"], batch_size=100
        )
    )
    assert sum(b.num_rows for b in batches) == 3


def test_scan_polars_with_correct_schema(store_with_correct_schema):
    lf = store_with_correct_schema.scan_polars(
        "stocks", columns=["ticker", "close"]
    )
    df = lf.collect().to_pandas()
    assert len(df) == 3


# ---- strict 模式：不匹配 → 报错 --------------------------------------------


def test_strict_mode_raises_on_mismatch(store_with_wrong_schema, monkeypatch):
    monkeypatch.setenv("QUANT_SCHEMA_CHECK", "strict")
    with pytest.raises(ValidationError, match="schema 校验失败"):
        store_with_wrong_schema.read_arrow("stocks", columns=["ticker"])


def test_strict_mode_also_blocks_stream(store_with_wrong_schema, monkeypatch):
    monkeypatch.setenv("QUANT_SCHEMA_CHECK", "strict")
    with pytest.raises(ValidationError, match="schema 校验失败"):
        # 生成器要消费才会触发 _ensure_schema；next() 一次就够
        gen = store_with_wrong_schema.read_arrow_stream("stocks", batch_size=100)
        next(gen)


def test_strict_mode_also_blocks_polars(store_with_wrong_schema, monkeypatch):
    monkeypatch.setenv("QUANT_SCHEMA_CHECK", "strict")
    with pytest.raises(ValidationError, match="schema 校验失败"):
        store_with_wrong_schema.scan_polars("stocks", columns=["ticker"])


# ---- warn 模式：不匹配但不阻塞 ----------------------------------------------


def test_warn_mode_logs_but_returns_data(
    store_with_wrong_schema, monkeypatch, caplog
):
    monkeypatch.setenv("QUANT_SCHEMA_CHECK", "warn")
    caplog.set_level(logging.WARNING, logger="data_access.schema_validation")
    tbl = store_with_wrong_schema.read_arrow("stocks", columns=["ticker"])
    assert tbl.num_rows == 3
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("bogus_column" in m or "close" in m for m in warnings)


def test_warn_mode_only_logs_once(store_with_wrong_schema, monkeypatch, caplog):
    """同一 dataset 多次读，warn 只打一次——避免刷屏。"""
    monkeypatch.setenv("QUANT_SCHEMA_CHECK", "warn")
    caplog.set_level(logging.WARNING, logger="data_access.schema_validation")
    store_with_wrong_schema.read_arrow("stocks", columns=["ticker"])
    store_with_wrong_schema.read_arrow("stocks", columns=["close"])
    store_with_wrong_schema.read_arrow("stocks", columns=["volume"])
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING
                and "schema 校验未通过" in r.getMessage()]
    assert len(warnings) == 1, f"期望只告警一次，实际 {len(warnings)} 次"


# ---- off 模式：完全静默 -----------------------------------------------------


def test_off_mode_silent(store_with_wrong_schema, monkeypatch, caplog):
    monkeypatch.setenv("QUANT_SCHEMA_CHECK", "off")
    caplog.set_level(logging.WARNING, logger="data_access.schema_validation")
    tbl = store_with_wrong_schema.read_arrow("stocks", columns=["ticker"])
    assert tbl.num_rows == 3
    assert not any(
        "schema 校验" in r.getMessage() for r in caplog.records
    )


# ---- 缓存行为 ---------------------------------------------------------------


def test_schema_check_runs_only_once(store_with_correct_schema):
    """第二次读同一 dataset 不应再调用 check_schema（进程缓存命中）。"""
    # 第一次读：走 _ensure_schema → check_schema
    store_with_correct_schema.read_arrow("stocks", columns=["ticker"])
    # 之后 patch check_schema 使其抛错；如果第二次读还触发它，测试会失败
    with patch("data_access.store.check_schema", side_effect=AssertionError("不应调用")):
        store_with_correct_schema.read_arrow("stocks", columns=["close"])
        store_with_correct_schema.read_arrow("stocks", columns=["volume"])


def test_no_schema_declaration_skips_check(tmp_path, monkeypatch):
    """schema 块未填 → _ensure_schema 直接返回，不 DESCRIBE。"""
    data_root = tmp_path / "data"
    _make_data(data_root)
    monkeypatch.setenv("TEST_DATA_ROOT", str(data_root))

    yaml_path = _write_yaml(
        tmp_path / "datasets.yaml",
        """
        stocks:
          kind: static
          access_mode: published
          layout: plain
          root: ${TEST_DATA_ROOT}
          glob: "*.parquet"
          time_column: align_time
          instrument_column: ticker
          hive_partitioning: false
          union_by_name: true
          # 故意不写 schema
        """,
    )
    reset_store()
    reset_validated_cache()
    reg = load_registry(yaml_path)
    engine = DuckDBEngine(threads=2)
    store = DataAccessStore(registry=reg, engine=engine)
    try:
        # patch check_schema 抛错；没声明 schema 时根本不应调用它
        with patch(
            "data_access.store.check_schema", side_effect=AssertionError("不应调用")
        ):
            tbl = store.read_arrow("stocks", columns=["ticker"])
            assert tbl.num_rows == 3
    finally:
        engine.close()
        reset_store()
