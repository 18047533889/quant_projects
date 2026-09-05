# -*- coding: utf-8 -*-
"""R61-P1 #56 —— 跨进程 cache single-flight（真实 COS / 本地 object-store 双轨）。

证明：K 个进程同时请求同一个未缓存对象时，per-key flock（``_sync_cos_file``
持有，覆盖 fresh 复检 + cp + os.replace + manifest 完整事务）保证**恰好 1 次
远程 fetch**，其余 K-1 个进程等锁后命中缓存。

两层验证：
1. ``test_single_flight_exactly_one_fetch_real_cos``（真实 COS，环境可达才跑）：
   4 个进程并发 ``_sync_cos_file`` 同一对象 → fetch 日志恰好 1 行。
2. ``test_single_flight_exactly_one_fetch_local_object_store``（无 COS 环境跑）：
   用本地对象存储模拟（同一 ``data_access.cos.mirror`` 读链路 + 本地 parquet
   源 + 同 per-key flock），证明与网络无关的锁语义本身。

真实 COS 判定：``_cos_reason()`` 返回 None = 可达。不可达时真实 COS 测试
**如实 SKIP**（不伪造 PASS），本地模拟测试始终跑。

fetch 计数：每个真正执行 COS CLI cp 的进程写一行到 ``DATA_ACCESS_COS_FETCH_LOG``
（JSONL）。测试断言该文件行数 == 1。
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[3]  # .../quant_projects
_COS_CLI = os.environ.get("DATA_ACCESS_COS_CLI", "/usr/local/bin/research-cos")
_COS_PREFIX = os.environ.get(
    "ASHARE_LQTP_COS_PREFIX", "cos://qs-cold/clean_data/ashare/lqtp_data"
)


def _cos_reason() -> str | None:
    """None = 真实 COS 可达；否则返回 SKIP 原因。"""
    if os.environ.get("DATA_ACCESS_SKIP_COS_MIRROR", "").strip() == "1":
        return "DATA_ACCESS_SKIP_COS_MIRROR=1"
    try:
        out = subprocess.run(
            [str(_COS_CLI), "ls", f"{_COS_PREFIX}/StockDailyBar/2026-06-15.parquet"],
            capture_output=True,
            text=True,
            timeout=20,
        )
        if out.returncode != 0:
            return f"COS CLI ls rc={out.returncode}: {(out.stderr or out.stdout)[:150]}"
    except Exception as exc:  # noqa: BLE001
        return f"COS CLI 不可达：{str(exc)[:150]}"
    return None


# cache root 必须落在 research-cos 批准的 local root（/home/{user}）下，否则 CLI cp rc=77。
_CACHE_ROOT = Path(
    os.environ.get("DA_SF_CACHE_ROOT", "/home/sunhaiwei/cos_data/da_sf_cache")
)

# 本测试会改 DATA_ACCESS_COS_* env（子进程 fork 继承）。pytest 全量跑时这些
# 改动会污染同 session 的其它 COS 测试 → 用 autouse fixture 在每个测试后还原。
_POLLUTED_ENV_KEYS = (
    "DATA_ACCESS_COS_CLI",
    "DATA_ACCESS_COS_CACHE_ROOT",
    "DATA_ACCESS_COS_SINGLE_FLIGHT",
    "DATA_ACCESS_COS_FETCH_LOG",
    "DATA_ACCESS_COS_DOWNLOAD_LOG",
    "ASHARE_COS_CLI",
)


@pytest.fixture(autouse=True)
def _restore_cos_env():
    saved = {k: os.environ.get(k) for k in _POLLUTED_ENV_KEYS}
    yield
    for k in _POLLUTED_ENV_KEYS:
        v = saved.get(k)
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def _child_body(cos_uri: str, dest: str, cache_root: str, fetch_log: str, dl_log: str, barrier: mp.Barrier) -> str:
    """子进程：等 barrier → 对同一对象执行 _sync_cos_file → 返回状态 JSON。"""
    import sys

    sys.path.insert(0, str(_REPO))
    os.environ["DATA_ACCESS_COS_CACHE_ROOT"] = cache_root
    os.environ["DATA_ACCESS_COS_SINGLE_FLIGHT"] = "1"
    os.environ["DATA_ACCESS_COS_FETCH_LOG"] = fetch_log
    os.environ["DATA_ACCESS_COS_DOWNLOAD_LOG"] = dl_log
    os.environ["DATA_ACCESS_COS_CLI"] = str(_COS_CLI)
    from data_access.cos.mirror import _local_file_fresh, _sync_cos_file
    from pathlib import Path as P

    try:
        barrier.wait(timeout=30)
    except Exception:  # noqa: BLE001
        pass
    pre = _local_file_fresh(P(dest))
    t0 = time.time()
    _sync_cos_file(cos_uri, P(dest), missing_semantics="error", dataset_name="ashare_stock_daily")
    after = _local_file_fresh(P(dest))
    return json.dumps(
        {"pre": bool(pre), "after": bool(after), "wall": round(time.time() - t0, 3)}
    )


def _run_k_concurrent(k: int) -> dict:
    """K 进程 barrier 同步同对象下载；返回 fetch/hit 统计。"""
    import shutil

    shutil.rmtree(_CACHE_ROOT, ignore_errors=True)
    _CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    fetch_log = _CACHE_ROOT / "fetch.jsonl"
    dl_log = _CACHE_ROOT / "download.jsonl"
    for f in (fetch_log, dl_log):
        f.unlink(missing_ok=True)

    # 计算 cache dest（与 remote.materialize_remote_via_cli 相同路径）。
    # 先设好 cache-root env，让 _cache_mirror_spec 基于 /home/... 批准根计算。
    os.environ["DATA_ACCESS_COS_CACHE_ROOT"] = str(_CACHE_ROOT)
    os.environ["DATA_ACCESS_COS_CLI"] = str(_COS_CLI)
    os.environ["DATA_ACCESS_COS_SINGLE_FLIGHT"] = "1"
    from data_access.cos.mirror import _daily_filename, _local_table_dir
    from data_access.cos.remote import _cache_mirror_spec, mirror_spec_for_dataset

    spec = mirror_spec_for_dataset("ashare_stock_daily")
    cache_spec = _cache_mirror_spec(spec)
    import datetime as dt

    fname = _daily_filename(cache_spec, dt.date.fromisoformat("2026-06-15"))
    dest = _local_table_dir(cache_spec) / fname
    cos_uri = f"{spec.cos_prefix}/StockDailyBar/{fname}"

    ctx = mp.get_context("fork")
    barrier = ctx.Barrier(k)
    procs = [
        ctx.Process(
            target=_child_body,
            args=(cos_uri, str(dest), str(_CACHE_ROOT), str(fetch_log), str(dl_log), barrier),
        )
        for _ in range(k)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=120)
        assert p.exitcode == 0, f"child exit={p.exitcode}"

    n_fetch = sum(1 for _ in fetch_log.open(encoding="utf-8")) if fetch_log.exists() else 0
    kinds = {"fetch": 0, "hit": 0}
    if dl_log.exists():
        for line in dl_log.open(encoding="utf-8"):
            try:
                kinds[json.loads(line)["kind"]] += 1
            except Exception:  # noqa: BLE001
                pass
    assert dest.exists(), "对象应已被下载"
    return {"fetch": n_fetch, "hit": kinds["hit"], "dest": str(dest)}


def test_single_flight_exactly_one_fetch_real_cos():
    """真实 COS：4 进程并发同对象 → 恰好 1 fetch + 3 hit。"""
    reason = _cos_reason()
    if reason is not None:
        pytest.skip(f"真实 COS 不可达，如实 SKIP：{reason}")
    stat = _run_k_concurrent(4)
    assert stat["fetch"] == 1, f"期望恰好 1 次远程 fetch，实际 {stat['fetch']}"
    assert stat["hit"] == 3, f"期望 3 次缓存命中，实际 {stat['hit']}"


def test_single_flight_exactly_one_fetch_local_object_store():
    """本地对象存储模拟（无网络依赖）：同一锁语义必须成立。

    ``DATA_ACCESS_SINGLE_FLIGHT_FAKE_CLI`` 指向一个本地「远程」parquet 源 +
    模拟 CLI 脚本：该脚本每执行一次就 append 一行到 fetch log（模拟真实 COS CLI
    的网络 fetch），再把源文件 cat 到目标。mirror 读链路 + per-key flock 完全不变
    → 4 进程并发 → fetch 日志恰好 1 行。
    """
    import shutil
    import stat as _stat

    reason = _cos_reason()
    if reason is None:
        # 真实 COS 可达时本测试也照跑（本地模拟，不触网），但为避免与真实 COS
        # 测试共用同一 cache root 相互污染，用独立 root。
        pass
    import tempfile

    tmp = Path(tempfile.mkdtemp(prefix="da_sf_local_"))
    cache_root = tmp / "cache"
    cache_root.mkdir()
    src_dir = tmp / "remote"
    src_dir.mkdir()
    # 本地「远程」源对象（与真实 COS 同布局）。
    src_file = src_dir / "2026-06-15.parquet"
    src_file.write_bytes(b"FAKE-PARQUET-CONTENT-SF-TEST-20260615")
    fetch_log = tmp / "fetch.jsonl"
    dl_log = tmp / "download.jsonl"

    # 模拟 CLI：每次调用把源 cat 到目标（$3）。fetch 行由 ``_run_cos_cli``
    # 的 ``_log_cos_fetch`` 统一追加（与真实 COS 同一计数通道），fake CLI
    # **不**自己写 fetch log，避免双计。
    cli = tmp / "fake-cos-cli"
    cli.write_text(
        "#!/bin/sh\n"
        "if [ \"$1\" = \"ls\" ]; then echo 'FAKE LS OK'; exit 0; fi\n"
        f"cp {src_file} \"$3\"\n"
        "exit 0\n",
        encoding="utf-8",
    )
    cli.chmod(cli.stat().st_mode | _stat.S_IXUSR)

    os.environ["DATA_ACCESS_COS_CLI"] = str(cli)
    os.environ["DATA_ACCESS_COS_CACHE_ROOT"] = str(cache_root)
    os.environ["DATA_ACCESS_COS_SINGLE_FLIGHT"] = "1"
    os.environ["DATA_ACCESS_COS_FETCH_LOG"] = str(fetch_log)
    os.environ["DATA_ACCESS_COS_DOWNLOAD_LOG"] = str(dl_log)

    ctx = mp.get_context("fork")
    k = 4
    barrier = ctx.Barrier(k)

    def _child(cache_root_s: str, dest_s: str, cos_uri_s: str, fl: str, dl: str, bar):
        import sys

        sys.path.insert(0, str(_REPO))
        os.environ["DATA_ACCESS_COS_CACHE_ROOT"] = cache_root_s
        os.environ["DATA_ACCESS_COS_SINGLE_FLIGHT"] = "1"
        os.environ["DATA_ACCESS_COS_FETCH_LOG"] = fl
        os.environ["DATA_ACCESS_COS_DOWNLOAD_LOG"] = dl
        os.environ["DATA_ACCESS_COS_CLI"] = str(cli)
        from data_access.cos.mirror import _sync_cos_file
        from pathlib import Path as P

        try:
            bar.wait(timeout=30)
        except Exception:  # noqa: BLE001
            pass
        _sync_cos_file(cos_uri_s, P(dest_s), missing_semantics="error", dataset_name="x")

    procs = [
        ctx.Process(
            target=_child,
            args=(
                str(cache_root),
                str(tmp / "dest_2026-06-15.parquet"),
                f"cos://fake-bucket/table/2026-06-15.parquet",
                str(fetch_log),
                str(dl_log),
                barrier,
            ),
        )
        for _ in range(k)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=60)
        assert p.exitcode == 0, f"child exit={p.exitcode}"

    n_fetch = sum(1 for _ in fetch_log.open(encoding="utf-8")) if fetch_log.exists() else 0
    assert n_fetch == 1, f"本地模拟：期望恰好 1 次 fetch，实际 {n_fetch}"
    kinds = {"fetch": 0, "hit": 0}
    if dl_log.exists():
        for line in dl_log.open(encoding="utf-8"):
            try:
                kinds[json.loads(line)["kind"]] += 1
            except Exception:  # noqa: BLE001
                pass
    assert kinds == {"fetch": 1, "hit": 3}, f"本地模拟 kinds={kinds}"
    shutil.rmtree(tmp, ignore_errors=True)
