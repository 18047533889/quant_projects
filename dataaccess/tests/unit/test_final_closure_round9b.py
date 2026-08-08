"""round9b 补充：0.9.5 收口中 round9.py 未覆盖的审计项。

round9.py（并发会话）覆盖：atomic 多进程 / publish-audit / read_auto / scan local
revalidation / mirror 三态 / parametric validate_params / empty universe / PIT typed /
PIT 并发 builder。本文件补齐：
    - item 4  远程（s3://）ScanHandle snapshot revalidation（round9 只测了本地）
    - item 6  共享 expected-partitions（trade-day 跨周末不生成周末对象 / 本地完整性
              用交易日历 + verified 三态，不再裸 .exists()）
    - item 8  coverage manifest fast path 贯穿同一份 validated params
    - item 9  strict 5t 无日历 fail-closed（authority=approximate / 降级 partial）
    - item 1  atomic full-write loop + durable fsync 失败传播 + 异常清理
    - item 12 只有 authoritative 的 PIT 构建才切 current（不完整保留上一代）
"""
from __future__ import annotations

import datetime as dt
import json
import os
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data_access.core.engine import DuckDBEngine
from data_access.core.exceptions import ValidationError
from data_access.registry import load_registry
from data_access.store import DataAccessStore


def _no_cos(monkeypatch):
    monkeypatch.setenv("DATA_ACCESS_SKIP_COS_MIRROR", "1")
    monkeypatch.setenv("DATA_ACCESS_COS_READ_MODE", "mirror")
    monkeypatch.delenv("DATA_ACCESS_STRICT_READ", raising=False)
    monkeypatch.delenv("QUANT_PRODUCTION_MODE", raising=False)


def _store(tmp_path, datasets: dict[str, dict]) -> DataAccessStore:
    cfg = tmp_path / "datasets.yaml"
    lines = []
    for name, body in datasets.items():
        lines.append(f"{name}:")
        for k, v in body.items():
            if isinstance(v, dict):
                lines.append(f"  {k}:")
                for sk, sv in v.items():
                    if isinstance(sv, dict):
                        lines.append(f"    {sk}:")
                        for ssk, ssv in sv.items():
                            lines.append(f"      {ssk}: {json.dumps(str(ssv))}")
                    else:
                        lines.append(f"    {sk}: {json.dumps(str(sv))}")
            else:
                lines.append(f"  {k}: {json.dumps(str(v))}")
    cfg.write_text("\n".join(lines), encoding="utf-8")
    return DataAccessStore(load_registry(cfg), DuckDBEngine(threads=2))


def _weekdays(start: dt.date, end: dt.date) -> list[dt.date]:
    out = []
    cur = start
    while cur <= end:
        if cur.weekday() < 5:
            out.append(cur)
        cur += dt.timedelta(days=1)
    return out


# ---------------------------------------------------------------------------
# 1) atomic：full-write loop + durable fsync 失败传播 + 异常清理
# ---------------------------------------------------------------------------

def test_atomic_full_write_loop(tmp_path, monkeypatch):
    import data_access.core.atomic as atomic_mod

    real_write = os.write
    calls = {"n": 0}

    def short_write(fd, data):
        calls["n"] += 1
        if calls["n"] == 1:
            half = len(data) // 2
            return real_write(fd, data[:half])
        return real_write(fd, data)

    monkeypatch.setattr(os, "write", short_write)
    target = tmp_path / "w.bin"
    atomic_mod.atomic_write_bytes(target, b"0123456789")
    assert target.read_bytes() == b"0123456789"
    assert calls["n"] >= 2  # 第一次短写触发了 full-write loop


def test_atomic_fsync_failure_propagates_and_cleans_up(tmp_path, monkeypatch):
    import data_access.core.atomic as atomic_mod

    target = tmp_path / "f.bin"

    def failing_fsync(fd):
        raise OSError("injected fsync failure")

    monkeypatch.setattr(os, "fsync", failing_fsync)
    with pytest.raises(OSError, match="injected"):
        atomic_mod.atomic_write_bytes(target, b"data")
    assert not target.exists()  # 目标未被写坏
    assert [p for p in tmp_path.iterdir() if ".tmp." in p.name] == []
    # durable=False：fsync 失败吞掉（best-effort），但仍完整写出
    atomic_mod.atomic_write_bytes(target, b"data", durable=False)
    assert target.exists() and target.read_bytes() == b"data"


# ---------------------------------------------------------------------------
# 4) 远程 s3:// ScanHandle snapshot revalidation
# ---------------------------------------------------------------------------

def _scan_handle_with_remote(snapshot):
    from data_access.read.query_budget import QueryBudget
    from data_access.read.read_contract import ReadLineage
    from data_access.read.scan_handle import ScanHandle

    return ScanHandle(
        _lf=None,
        snapshot=snapshot,
        budget=QueryBudget(),
        lineage=ReadLineage(dataset="d"),
        _store=object(),  # 非 None → revalidation 真正执行
    )


def test_scan_remote_revalidate_strict_rejects_changed(monkeypatch):
    from data_access.read.read_contract import FileVersion, build_data_snapshot
    import data_access.read.scan_handle as scan_mod

    snap = build_data_snapshot(
        dataset="d", registry_hash="r", schema=None,
        paths=["s3://b/k.parquet"],
        files=[FileVersion(path="s3://b/k.parquet", etag="old", content_length=100)],
    )
    monkeypatch.setattr(scan_mod, "_remote_snapshot_meta_enabled", lambda: True)
    monkeypatch.setattr(
        scan_mod, "_remote_object_meta",
        lambda uri, fresh=False: {"etag": "new", "content_length": 100},
    )
    scan = _scan_handle_with_remote(snap)
    os.environ["DATA_ACCESS_STRICT_READ"] = "1"
    try:
        with pytest.raises(ValidationError, match="已失效"):
            scan._revalidate_snapshot()
    finally:
        del os.environ["DATA_ACCESS_STRICT_READ"]


def test_scan_remote_revalidate_strict_unverifiable_fail_closed(monkeypatch):
    from data_access.read.read_contract import FileVersion, build_data_snapshot
    import data_access.read.scan_handle as scan_mod

    snap = build_data_snapshot(
        dataset="d", registry_hash="r", schema=None,
        paths=["s3://b/k.parquet"],
        files=[FileVersion(path="s3://b/k.parquet", etag="old", content_length=100)],
    )
    monkeypatch.setattr(scan_mod, "_remote_snapshot_meta_enabled", lambda: True)
    monkeypatch.setattr(scan_mod, "_remote_object_meta", lambda uri, fresh=False: None)
    scan = _scan_handle_with_remote(snap)
    os.environ["DATA_ACCESS_STRICT_READ"] = "1"
    try:
        with pytest.raises(ValidationError, match="已失效"):
            scan._revalidate_snapshot()
    finally:
        del os.environ["DATA_ACCESS_STRICT_READ"]


def test_scan_remote_revalidate_unchanged_ok(monkeypatch):
    from data_access.read.read_contract import FileVersion, build_data_snapshot
    import data_access.read.scan_handle as scan_mod

    snap = build_data_snapshot(
        dataset="d", registry_hash="r", schema=None,
        paths=["s3://b/k.parquet"],
        files=[FileVersion(path="s3://b/k.parquet", etag="old", content_length=100)],
    )
    monkeypatch.setattr(scan_mod, "_remote_snapshot_meta_enabled", lambda: True)
    monkeypatch.setattr(
        scan_mod, "_remote_object_meta",
        lambda uri, fresh=False: {"etag": "old", "content_length": 100},
    )
    scan = _scan_handle_with_remote(snap)
    os.environ["DATA_ACCESS_STRICT_READ"] = "1"
    try:
        assert scan._revalidate_snapshot() is snap
    finally:
        del os.environ["DATA_ACCESS_STRICT_READ"]


# ---------------------------------------------------------------------------
# 6) 共享 expected-partitions（trade-day 跨周末不生成周末对象）
# ---------------------------------------------------------------------------

def test_remote_daily_paths_skip_weekends_for_trade_day(tmp_path, monkeypatch):
    """直接测 ``_remote_daily_paths``（本轮的 per-day expected-partitions 编译器）。

    不走 ``build_remote_paths``：concurrent session 的 cos_storage_runtime 会把它
    patch 成 month-glob（daily 用 ``2026-08-*.parquet`` 由 SQL 谓词裁剪，天然不含
    day 枚举）——per-day 的 trade-day 跳过逻辑只活在 ``_remote_daily_paths`` 里。
    """
    from data_access.cos.mirror import MirrorSpec
    import data_access.cos.mirror as mirror_mod
    import data_access.cos.remote as remote_mod

    spec = MirrorSpec(
        cos_prefix="cos://qs-cold/clean_data/ashare/lqtp_data",
        local_root=tmp_path / "local",
        table="StockDailyBar",
        layout="daily_parquet",
    )
    monkeypatch.setattr(mirror_mod, "_calendar_domain_of", lambda name: "trade_day")
    monkeypatch.setattr(
        mirror_mod,
        "_market_trading_days",
        lambda name, s, e: (_weekdays(s, e), True),
    )
    paths = remote_mod._remote_daily_paths(spec, "ashare_stock_daily", ("2026-08-03", "2026-08-10"))
    texts = "\n".join(paths)
    assert "2026-08-08.parquet" not in texts  # 周六
    assert "2026-08-09.parquet" not in texts  # 周日
    assert "2026-08-03.parquet" in texts and "2026-08-07.parquet" in texts
    # 自然日退化路径仍可用（无真实日历 → _market_trading_days 返回自然日 + used=False）
    monkeypatch.setattr(
        mirror_mod,
        "_market_trading_days",
        lambda name, s, e: (mirror_mod._iter_dates(s, e), False),
    )
    paths2 = remote_mod._remote_daily_paths(spec, "ashare_stock_daily", ("2026-08-08", "2026-08-09"))
    assert len(paths2) == 2  # 退化：自然日（含周末）


def test_local_daily_complete_uses_expected_and_usable(tmp_path, monkeypatch):
    from data_access.cos.mirror import MirrorSpec, _write_download_manifest
    import data_access.cos.mirror as mirror_mod
    import data_access.cos.remote as remote_mod

    local = tmp_path / "local"
    table = local / "T"
    table.mkdir(parents=True)
    expected = _weekdays(dt.date(2026, 8, 3), dt.date(2026, 8, 10))
    for day in expected:
        f = table / f"{day.isoformat()}.parquet"
        pq.write_table(pa.table({"v": [1.0]}), str(f))
        _write_download_manifest(f)
    spec = MirrorSpec(
        cos_prefix="cos://qs-cold/clean_data/x",
        local_root=local,
        table="T",
        layout="daily_parquet",
    )
    monkeypatch.setattr(remote_mod, "mirror_spec_for_dataset", lambda name: spec)
    monkeypatch.setattr(mirror_mod, "_calendar_domain_of", lambda name: "trade_day")
    monkeypatch.setattr(
        mirror_mod,
        "_market_trading_days",
        lambda name, s, e: (_weekdays(s, e), True),
    )
    # 跨周末（08-03..08-10）：期望只有 6 个交易日，全部 verified → 齐全。
    # 旧逻辑按自然日枚举 8 天会把 08-08/09 当缺失 → 误判不完整。
    assert remote_mod.local_mirror_complete_for_range(
        "d", time_range=("2026-08-03", "2026-08-10")
    ) is True


def test_hybrid_cos_read_paths_uses_usable_state(tmp_path, monkeypatch):
    from data_access.cos.mirror import MirrorSpec, _write_download_manifest
    import data_access.cos.mirror as mirror_mod
    import data_access.cos.remote as remote_mod

    local = tmp_path / "local"
    table = local / "T"
    table.mkdir(parents=True)
    good = table / "2026-08-03.parquet"
    bad = table / "2026-08-04.parquet"
    pq.write_table(pa.table({"v": [1.0]}), str(good))
    pq.write_table(pa.table({"v": [2.0]}), str(bad))
    _write_download_manifest(good)
    bad.with_suffix(bad.suffix + ".manifest.json").write_text(
        json.dumps({"size": 1, "mtime_ns": 1, "verified": True})
    )
    spec = MirrorSpec(
        cos_prefix="cos://qs-cold/clean_data/ashare/lqtp_data",
        local_root=local,
        table="T",
        layout="daily_parquet",
    )
    monkeypatch.setattr(remote_mod, "mirror_spec_for_dataset", lambda name: spec)
    monkeypatch.setattr(remote_mod, "cos_read_mode", lambda: "auto")
    monkeypatch.setattr(remote_mod, "cos_remote_backend", lambda: "httpfs")
    monkeypatch.setattr(remote_mod, "authorize_s3_path", lambda p, extra_prefixes=None: None)
    monkeypatch.setattr(
        mirror_mod,
        "_expected_dates",
        lambda name, s, e: [dt.date(2026, 8, 3), dt.date(2026, 8, 4)],
    )
    # hybrid 要求真 StaticDataset（SimpleNamespace 不是，会被拒）
    store = _store(
        tmp_path,
        {
            "d": {
                "kind": "static", "access_mode": "published", "layout": "plain",
                "root": str(local), "glob": "*.parquet",
                "time_column": "t", "instrument_column": "s",
            }
        },
    )
    merged = remote_mod.hybrid_cos_read_paths(
        store._registry.get("d"),
        time_range=("2026-08-03", "2026-08-04"),
    )
    assert merged is not None
    locals_only = [p for p in merged if p.startswith(str(local))]
    remotes = [p for p in merged if p.startswith("s3://")]
    # verified 的 08-03 走本地；corrupt 的 08-04 绝不走本地，走 remote
    assert any("2026-08-03.parquet" in p for p in locals_only)
    assert not any("2026-08-04.parquet" in p for p in locals_only)
    assert any("2026-08-04.parquet" in p for p in remotes)


# ---------------------------------------------------------------------------
# 8) coverage manifest fast path 贯穿 validated params
# ---------------------------------------------------------------------------

def test_coverage_parametric_manifest_identity(tmp_path):
    from data_access.read.coverage import compute_coverage

    lake = tmp_path / "lake"
    store = _store(
        tmp_path,
        {
            "fl_stg": {
                "kind": "parametric", "access_mode": "staging", "layout": "plain",
                "root_template": str(lake / "factors" / "{factor_id}"),
                "glob_template": "part-*.parquet",
                "params_schema": {"factor_id": "str"},
                "time_column": "datetime", "instrument_column": "asset",
                "schema": {"datetime": "timestamp", "asset": "string", "value": "double"},
            }
        },
    )
    store.write_arrow(
        "fl_stg",
        pa.table(
            {
                "datetime": [dt.datetime(2024, 1, 5), dt.datetime(2024, 1, 20)],
                "asset": ["A", "A"],
                "value": [1.0, 2.0],
            }
        ),
        factor_id="f1",
        mode="overwrite",
    )
    # 显式构建 manifest（写路径不自动创建首个 manifest sidecar）
    from data_access.read.manifest import build_manifest_for_dataset

    assert build_manifest_for_dataset(
        store, "fl_stg", params={"factor_id": "f1"}
    ) is not None
    report = compute_coverage(store, "fl_stg", params={"factor_id": "f1"})
    # manifest lookup 必须用同一份 params 定位到 f1 实例（不是空 params）
    assert report.observed_start is not None
    assert report.observed_start.startswith("2024-01")
    assert report.observed_end.startswith("2024-01")
    assert report.status in {"complete", "partial"}


# ---------------------------------------------------------------------------
# 9) strict 5t 无日历 fail-closed
# ---------------------------------------------------------------------------

def test_coverage_5t_strict_no_calendar_downgrades_partial(monkeypatch):
    import data_access.read.coverage as cov

    report = cov.CoverageReport(
        dataset="us_stock_daily", observed_end="2026-08-05", max_staleness="5t"
    )
    monkeypatch.setattr(cov, "_infer_market", lambda ds: "us")
    os.environ["DATA_ACCESS_STRICT_READ"] = "1"
    try:
        cov._maybe_mark_stale(report, store=None, dataset="us_stock_daily")
    finally:
        del os.environ["DATA_ACCESS_STRICT_READ"]
    assert report.status == "partial"
    assert report.authority == "approximate"
    assert any("日历不可用" in p for p in report.problems)


def test_coverage_5t_research_approximates(monkeypatch):
    import data_access.read.coverage as cov

    report = cov.CoverageReport(
        dataset="us_stock_daily", observed_end="2026-08-05", max_staleness="5t"
    )
    monkeypatch.setattr(cov, "_infer_market", lambda ds: "us")
    cov._maybe_mark_stale(report, store=None, dataset="us_stock_daily")
    assert report.authority == "approximate"  # 显式标记近似
    assert report.status != "partial"          # research 不降级


def test_coverage_5t_authoritative_with_calendar(monkeypatch):
    import data_access.read.coverage as cov
    import data_access.read.session_calendar as sc

    report = cov.CoverageReport(
        dataset="us_stock_daily", observed_end="2026-08-05", max_staleness="30t"
    )
    monkeypatch.setattr(cov, "_infer_market", lambda ds: "us")

    class _Cal:
        has_data = True
        trading_days = list(_weekdays(dt.date(2026, 8, 5), dt.date(2026, 8, 9)))

    monkeypatch.setattr(sc, "get_market_calendar", lambda market, store=None: _Cal())
    cov._maybe_mark_stale(report, store=object(), dataset="us_stock_daily")
    assert report.authority == "authoritative"
    assert report.status != "partial"


# ---------------------------------------------------------------------------
# 12) 只有 authoritative 的 PIT 构建才切 current（不完整保留上一代）
# ---------------------------------------------------------------------------

def test_pit_incomplete_build_keeps_previous_generation(tmp_path):
    import data_access.read.pit_event_index as pei

    d = tmp_path / "d"
    d.mkdir()
    pq.write_table(
        pa.table({"Ticker": ["A"], "filing_date": [dt.date(2024, 1, 1)],
                  "period_end": [dt.date(2023, 12, 31)]}),
        str(d / "2023-12-31.parquet"),
    )
    store = _store(
        tmp_path,
        {
            "us_balance": {
                "kind": "static", "access_mode": "published", "layout": "plain",
                "root": str(d), "glob": "*.parquet",
                "time_column": "filing_date", "instrument_column": "Ticker",
                "schema": {"Ticker": "string", "filing_date": "date",
                           "period_end": "date"},
            }
        },
    )
    idx = pei.build_pit_event_index(store, "us_balance", force=True)
    assert idx.metadata.complete is True
    first_gen = pei._current_generation(d)
    assert first_gen is not None
    # 引入损坏文件（glob 仍枚举到，但读取失败）→ 构建不完整
    (d / "bad.parquet").write_bytes(b"not a parquet file at all........")
    idx2 = pei.build_pit_event_index(store, "us_balance", force=True)
    assert idx2.metadata.complete is False
    # current 仍指向上一代好索引（不完整构建不覆盖）
    assert pei._current_generation(d) == first_gen
    p = pei._current_index_parquet(d)
    assert p is not None and p.exists()
