# -*- coding: utf-8 -*-
"""R25 T-PHY-001..005 —— 物理布局测试（P0-001/002/016/015）。

    T-PHY-001  US finance period file：物理 StockIncome/2024-03-31.parquet，
               request knowledge window=2024-05-10 → 读到该行（四模式 local/
               mirror/remote/auto 一致，不用 request 日期拼文件名）。
    T-PHY-002  StockCapital shares：只访问 shares_2024-01-01.parquet，
               不访问 2024-01-01.parquet。
    T-PHY-003  StockCapital split：反过来，只访问 {date}.parquet。
    T-PHY-004  event sparse：某天无 event object → EMPTY_OK（不 hard fail）。
    T-PHY-005  dense D1 missing：交易日行情 object missing → hard error。
"""
from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import patch

import pytest

from data_access.contract.physical_partition import (
    MissingPartitionSemantics,
    PhysicalLayout,
    PhysicalPartitionSpec,
)
from data_access.contract.runtime_contract import compile_runtime_contract
from data_access.cos.mirror import (
    _daily_filename,
    _sync_cos_file,
    expected_partitions,
    physical_partition_for,
)
from data_access.registry import load_registry


# ---------------------------------------------------------------------------
# T-PHY-001 — US finance period file
# ---------------------------------------------------------------------------
def test_tphy001_us_finance_period_files_layout():
    """contract storage_layout=period_files → PhysicalPartitionSpec PERIOD_END_FILE。"""
    rc = compile_runtime_contract("us_stock_income", load_registry())
    assert rc.physical_partition.layout == PhysicalLayout.PERIOD_END_FILE
    assert rc.physical_partition.partition_clock == "period_end"
    assert rc.physical_partition.filename_template == "{period_end}.parquet"
    assert rc.physical_partition.query_clock == "filing_date"
    # 不能按 request time_range 展开文件名（partition_clock != predicate_clock）
    assert not rc.physical_partition.calendar_enumerable
    assert rc.physical_partition.requires_index_mapping


def test_tphy001b_expected_partitions_period_files_empty():
    """period_files 布局 expected_partitions 返回空——不按 request 日期枚举。"""
    parts = expected_partitions(
        "us_stock_income", date(2024, 5, 10), date(2024, 5, 10), layout="period_files"
    )
    assert parts == []


def test_tphy001c_mirror_spec_period_files():
    """mirror spec 对 US finance 用 period_files（P0-001 根因修复）。"""
    from data_access.cos.mirror import mirror_spec_for_dataset

    spec = mirror_spec_for_dataset("us_stock_income")
    assert spec is not None
    assert spec.layout == "period_files"
    assert spec.filename_template == "{period_end}.parquet"


def test_tphy001d_physical_partition_locator_shared():
    """mirror/remote 用同一个 physical_partition_for locator。"""
    pp = physical_partition_for("us_stock_income")
    assert pp.layout == PhysicalLayout.PERIOD_END_FILE
    assert pp.filename_template == "{period_end}.parquet"
    # StockIncome/2024-03-31.parquet（period_end）可渲染
    assert pp.filename_for(period_end="2024-03-31") == "2024-03-31.parquet"


# ---------------------------------------------------------------------------
# T-PHY-002 / T-PHY-003 — StockCapital split/shares
# ---------------------------------------------------------------------------
def test_tphy002_shares_prefix_locator():
    """shares 用 file_selector=shares_ 只生成 shares_{date}.parquet。"""
    from data_access.cos.mirror import MirrorSpec

    spec = MirrorSpec(
        cos_prefix="cos://t",
        local_root=Path("/tmp/x"),
        table="StockCapitalDaily",
        layout="prefixed_date_file",
        file_selector="shares_",
        filename_template="shares_{date}.parquet",
    )
    assert _daily_filename(spec, date(2024, 1, 1)) == "shares_2024-01-01.parquet"


def test_tphy002b_shares_remote_path_only_selector():
    """remote 只访问 shares_{date}.parquet，不访问 {date}.parquet（P0-002/R26-P0-010）。

    R26-P0-010：不再用「shares_*.parquet」这种宽 glob（会命中任意前缀），改用
    FileSelector 精确 date 文件名 glob（``shares_[0-9]{4}-...``）。
    """
    from data_access.cos.mirror import mirror_spec_for_dataset
    from data_access.cos.remote import _remote_prefixed_date

    spec = mirror_spec_for_dataset("us_stock_capital_shares")
    assert spec is not None
    assert spec.file_selector == "shares_"
    paths = _remote_prefixed_date(spec, "us_stock_capital_shares", None)
    assert len(paths) == 1
    assert paths[0].endswith("shares_[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9].parquet")
    # 精确 glob 不得退化成裸 *.parquet（P0-010：不能把 shares_ 排除逻辑当空前缀）。
    assert not paths[0].endswith("*.parquet")


def test_tphy003_split_default_prefix():
    """split 用默认 {date}.parquet（file_selector=None）。"""
    from data_access.cos.mirror import MirrorSpec

    spec = MirrorSpec(
        cos_prefix="cos://t",
        local_root=Path("/tmp/x"),
        table="StockCapitalDaily",
        layout="prefixed_date_file",
    )
    assert _daily_filename(spec, date(2024, 1, 1)) == "2024-01-01.parquet"
    from data_access.cos.mirror import mirror_spec_for_dataset

    split_spec = mirror_spec_for_dataset("us_stock_capital_split")
    assert split_spec.file_selector is None
    assert split_spec.filename_template == "{date}.parquet"


# ---------------------------------------------------------------------------
# T-PHY-004 — event sparse EMPTY_OK
# ---------------------------------------------------------------------------
def test_tphy004_event_sparse_empty_ok():
    """某天无 event object → missing_semantics=empty_ok 静默跳过。"""
    import subprocess

    dest = Path("/tmp/r25_evt") / "2024-01-02.parquet"

    def fake_run(cmd, **kwargs):
        raise subprocess.CalledProcessError(
            1, cmd, stderr="cos object not found: x"
        )

    with patch("data_access.cos.mirror.subprocess.run", side_effect=fake_run):
        _sync_cos_file(
            "cos://t/StockDividend/2024-01-02.parquet",
            dest,
            missing_semantics="empty_ok",
        )
    assert not dest.exists()  # empty_ok 跳过，不 hard fail


# ---------------------------------------------------------------------------
# T-PHY-005 — dense D1 missing ERROR
# ---------------------------------------------------------------------------
def test_tphy005_dense_missing_error():
    """交易日行情 object missing → MissingRequiredPartition（fail-closed）。"""
    import subprocess

    from data_access.core.exceptions import MissingRequiredPartition

    dest = Path("/tmp/r25_dense") / "2024-01-02.parquet"

    def fake_run(cmd, **kwargs):
        raise subprocess.CalledProcessError(
            1, cmd, stderr="cos object not found: x"
        )

    with patch("data_access.cos.mirror.subprocess.run", side_effect=fake_run):
        with pytest.raises(MissingRequiredPartition):
            _sync_cos_file(
                "cos://t/StockDailyBar/2024-01-02.parquet",
                dest,
                missing_semantics="error",
                dataset_name="ashare_stock_daily",
            )
