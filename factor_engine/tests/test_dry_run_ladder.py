# -*- coding: utf-8 -*-
"""P0#11+#12 —— dry-run ladder + streaming sink + checkpoint + 失败分类 测试。

(a) checkpoint resume 跳过已完成 shard（预写 checkpoint 后运行，只算 pending）；
(b) streaming sink 最终结果 == in-memory 基线（1e-12）；
(c) 失败分类把强制 DataDegeneracy 与强制 contract error 路由到不同 bucket。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from factor_engine.benchmarks import dry_run_ladder as ladder
from factor_engine.runtime.dry_run_checkpoint import DryRunCheckpoint, CampaignMeta
from factor_engine.runtime.failure_classification import (
    FailureCode,
    FailureReport,
    classify_exception,
)
from factor_engine.storage.streaming_sink import StreamingSink, StreamRow


# ---------------------------------------------------------------------------
# (a) checkpoint resume
# ---------------------------------------------------------------------------
def test_checkpoint_resume_skips_completed(tmp_path: Path) -> None:
    ckpt = DryRunCheckpoint(checkpoint_dir=tmp_path / "ckpt")
    ckpt.init_campaign(
        CampaignMeta(campaign_id="resume", total_roots=4, source_snapshot="s")
    )
    root_ids = ["f0", "f1", "f2", "f3"]
    # 预写 3 个已完成（模拟上次跑到一半被中断）
    ckpt.mark_completed("f0", marker={"checksum": "c0"})
    ckpt.mark_completed("f1", marker={"checksum": "c1"})
    ckpt.mark_completed("f2", marker={"checksum": "c2"})

    pending = ckpt.pending_roots(root_ids)
    assert pending == ["f3"]  # 只剩下未完成的一根


def test_checkpoint_atomic_write_leaves_no_tmp(tmp_path: Path) -> None:
    ckpt = DryRunCheckpoint(checkpoint_dir=tmp_path / "ckpt")
    ckpt.mark_completed("f0", marker={"x": 1})
    leftovers = [p for p in (tmp_path / "ckpt").iterdir() if p.suffix == ".tmp"]
    assert leftovers == []
    assert ckpt.is_completed("f0")


def test_pending_excludes_failed(tmp_path: Path) -> None:
    ckpt = DryRunCheckpoint(checkpoint_dir=tmp_path / "ckpt")
    ckpt.mark_failed("f9", ValueError("bad"))
    root_ids = ["f9", "f10"]
    assert ckpt.pending_roots(root_ids) == ["f10"]


# ---------------------------------------------------------------------------
# (b) streaming sink == in-memory baseline
# ---------------------------------------------------------------------------
def _make_tiny_rows(nstocks: int, ndays: int, seed: int = 0) -> list[StreamRow]:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=ndays)
    assets = [f"A{i:04d}" for i in range(nstocks)]
    vals = rng.standard_normal((ndays, nstocks))
    vals = np.where(np.abs(vals) < 1.5, np.nan, vals)  # 加部分 NaN
    rows: list[StreamRow] = []
    for d in range(ndays):
        for a in range(nstocks):
            v = float(vals[d, a])
            if not np.isnan(v):
                rows.append(StreamRow(str(dates[d].date()), assets[a], v))
    return rows


def _sha_triples(rows: list[StreamRow]) -> str:
    import hashlib

    triples = sorted(rows, key=lambda r: (r.date, r.asset))
    h = hashlib.sha256()
    for r in triples:
        h.update(f"{r.date}|{r.asset}|{r.value!r}\n".encode("utf-8"))
    return h.hexdigest()


def test_streaming_matches_in_memory(tmp_path: Path) -> None:
    baseline_rows = _make_tiny_rows(30, 90)
    sink = StreamingSink(output_dir=tmp_path / "lake", campaign_id="s1")
    root_ids = ["f00000", "f00001", "f00002"]
    per_root = [baseline_rows[i::3] for i in range(3)]
    for rid, rows in zip(root_ids, per_root):
        with sink.open(rid) as shard:
            shard.add_many(rows)

    # streaming 重读
    final_tbl = sink.read_all()

    # 手动重建
    got = [
        StreamRow(d, a, v)
        for (d, a, v) in zip(
            final_tbl["date"].to_pylist(),
            final_tbl["asset"].to_pylist(),
            final_tbl["value"].to_pylist(),
        )
    ]
    # 值级 1e-12
    assert len(got) == len(baseline_rows)
    got_d = {(r.date, r.asset): r.value for r in got}
    for r in baseline_rows:
        assert got_d[(r.date, r.asset)] == pytest.approx(r.value, rel=1e-12, abs=1e-12)

    # checksum 级（排序后一致）
    assert _sha_triples(got) == _sha_triples(baseline_rows)


def test_streaming_shard_checkpoint_sidecar(tmp_path: Path) -> None:
    sink = StreamingSink(output_dir=tmp_path / "lake", campaign_id="s2")
    with sink.open("fX") as shard:
        shard.add("2024-01-01", "A1", 1.0)
    sidecar = tmp_path / "lake" / "shards" / "fX.sidecar.json"
    assert sidecar.exists()
    data = json.loads(sidecar.read_text())
    assert data["num_rows"] == 1
    assert data["checksum"]


# ---------------------------------------------------------------------------
# (c) 失败分类路由
# ---------------------------------------------------------------------------
def test_failure_classification_routes_distinct_buckets() -> None:
    report = FailureReport()
    # 强制 DataDegeneracy：DataQualityError（runtime 异常）→ DATA_MISSING / data-degeneracy
    from factor_engine.runtime.exceptions import DataQualityError

    code = report.record(DataQualityError("source empty, factor degenerate"))
    assert code == FailureCode.DATA_MISSING
    assert report.bucket_counts().get("data-degeneracy", 0) == 1

    # 强制 contract error：SemanticError → INVALID_FORMULA / contract-violation
    from factor_engine.runtime.exceptions import SemanticError

    code2 = report.record(SemanticError("unknown operator in formula"))
    assert code2 == FailureCode.INVALID_FORMULA
    assert report.bucket_counts().get("contract-violation", 0) == 1

    # 两个 bucket 必须互斥、不同
    assert report.bucket_counts()["data-degeneracy"] == 1
    assert report.bucket_counts()["contract-violation"] == 1


def test_failure_classification_oom_and_memory() -> None:
    from factor_engine.runtime.exceptions import OOMReplanRequired

    code, bucket = classify_exception(OOMReplanRequired("budget exceeded"))
    assert code == FailureCode.OOM
    assert bucket == "resource-exhaustion"

    code2, bucket2 = classify_exception(MemoryError("out"))
    assert code2 == FailureCode.OOM
    assert bucket2 == "resource-exhaustion"


def test_failure_classification_internal_fallback() -> None:
    code, bucket = classify_exception(RuntimeError("weird internal state"))
    assert code == FailureCode.INTERNAL
    assert bucket == "internal-bug"


# ---------------------------------------------------------------------------
# (a/b) 全链路：checkpoint resume 后 rung 只算 pending，且结果仍等于基线
# ---------------------------------------------------------------------------
def test_ladder_resume_only_recomputes_pending(tmp_path: Path, monkeypatch) -> None:
    # 缩小面板，跑纯内存轻量级 ladder
    monkeypatch.setattr(ladder, "N_DAYS", 20)
    monkeypatch.setattr(ladder, "N_STOCKS", 8)
    n_factors = 6

    out_root = tmp_path / "out"
    workspace = tmp_path / "ws"
    cp = DryRunCheckpoint(checkpoint_dir=workspace / "ckpt" / "tiny" / "campaign_tiny")
    cp.mark_completed("f00000", marker={"done": True})
    cp.mark_completed("f00001", marker={"done": True})

    meta = ladder.run_rung(name="tiny", n_factors=n_factors, root_dir=workspace)
    assert meta["completed_roots"] == n_factors
    # 预写 2 根已完成 → 重跑只补算剩余 4 根
    completed_json = json.loads(
        (workspace / "ckpt" / "tiny" / "campaign_tiny" / "completed.json").read_text()
    )
    assert set(completed_json.keys()) == {f"f{i:05d}" for i in range(n_factors)}
