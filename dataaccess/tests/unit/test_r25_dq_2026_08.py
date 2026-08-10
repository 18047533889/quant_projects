# -*- coding: utf-8 -*-
"""R25 §100 / §64-66 —— DQ golden 测试。

    A Return bp -> decimal、US Ret decimal unchanged、ROE、CNY vs USD money、
    单位漂移 sentinel、negative volume、duplicate semantic keys。
"""
from __future__ import annotations

import pyarrow as pa

from data_access.quality.contracts import QualityOptions, run_quality_checks


def test_dq_unit_sentinel_detects_scaling():
    """R25 §65：Return 分布突然缩放（bp→decimal 10000x）→ BLOCK。"""
    import pyarrow as pa

    table = pa.table(
        {
            "Return": [0.0001, 0.0002, 0.0003, 0.0004, 0.0005],  # decimal semantics
        }
    )
    opts = QualityOptions(
        unit_sentinel={"Return": ("decimal", (0.0001, 0.01))},
    )
    report = run_quality_checks("t", table, options=opts)
    assert report.passed is True  # decimal 在典型区间

    # bp semantics 误标 decimal：10000x 缩放 → BLOCK
    table_bp = pa.table({"Return": [1.0, 2.0, 3.0, 4.0, 5.0]})  # bp values
    opts_bp = QualityOptions(
        unit_sentinel={"Return": ("decimal", (0.0001, 0.01))},
    )
    report_bp = run_quality_checks("t", table_bp, options=opts_bp)
    assert report_bp.passed is False
    assert "unit sentinel" in " ".join(report_bp.failures)


def test_dq_unit_sentinel_warn_only():
    """warn_only=True：超界只 WARN 不 BLOCK。"""
    table = pa.table({"Return": [1.0, 2.0, 3.0, 4.0, 5.0]})
    opts = QualityOptions(
        unit_sentinel={"Return": ("decimal", (0.0001, 0.01))},
        unit_sentinel_warn_only=True,
    )
    report = run_quality_checks("t", table, options=opts)
    assert report.passed is True  # warn_only 不 fail
    assert "unit_sentinel" in report.details


def test_dq_negative_volume():
    """R25 §100：negative volume blocked/flagged。"""
    table = pa.table(
        {"TradeDate": ["2024-01-02"] * 3, "Symbol": ["A", "B", "C"], "Volume": [100, -5, 200]}
    )
    opts = QualityOptions(
        range={"Volume": (0.0, 1e15)},  # volume 必须 >= 0
        finite_columns=("Volume",),
    )
    report = run_quality_checks("t", table, options=opts)
    assert report.passed is False
    assert "range violations" in " ".join(report.failures)


def test_dq_duplicate_semantic_keys():
    """R25 §100：duplicate semantic keys detected。"""
    table = pa.table(
        {
            "TradeDate": ["2024-01-02", "2024-01-02"],
            "Symbol": ["A", "A"],
            "Close": [1.0, 2.0],
        }
    )
    opts = QualityOptions(primary_key=("TradeDate", "Symbol"))
    report = run_quality_checks("t", table, options=opts)
    assert report.passed is False
    assert "duplicate primary keys" in " ".join(report.failures)


def test_dq_pit_leakage():
    """R25 §100：finance period > knowledge → PIT leakage。"""
    from datetime import datetime, timezone

    def _ts(s: str):
        return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    table = pa.table(
        {
            "report_period": pa.array(
                [_ts("2024-03-31"), _ts("2024-06-30")], type=pa.timestamp("us")
            ),
            "publish_time": pa.array(
                [_ts("2024-05-10"), _ts("2024-06-25")], type=pa.timestamp("us")
            ),
        }
    )
    # 2024-06-30 的 report_period > 2024-06-25 的 publish → leakage
    opts = QualityOptions(check_pit_leakage=True)
    report = run_quality_checks("t", table, options=opts)
    assert report.passed is False
