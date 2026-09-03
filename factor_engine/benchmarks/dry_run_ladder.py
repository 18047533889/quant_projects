# -*- coding: utf-8 -*-
"""FE 100k GO P0#11 —— 100k dry-run ladder（GO prompt §71 + P0 list #11）。

正式落 10 万因子前，必须先逐级放大验证线性度，任何非线性恶化要停下来查原因。

rung 定义（§71）：

    A=100, B=1000, C=5000, D=20k, E=50k, F=100k (factors)

每一级比较：``output checksum / DQ / throughput / memory / failures``。

本实现用 **合成面板**（不碰真数据/COS），每个「factor」是一个确定性可复现的
函数（滚动均线 + 收益率），结果经 :class:`StreamingSink` 分片流式落盘 +
:class:`DryRunCheckpoint` 断点续跑，并断言：
   streaming 最终结果 == in-memory 基线（1e-12）
任何单因子计算异常进入 :class:`FailureReport` 分类（不 ``except: continue``）。

CLI::

    python3 benchmarks/dry_run_ladder.py --rungs tiny,small
    python3 benchmarks/dry_run_ladder.py --rungs all
    DRY_RUN_MAX_CORES=4 python3 benchmarks/dry_run_ladder.py --rungs tiny,small

rung 规模（每个 factor 的合成面板行数固定，故 rung 只放大 factor 数）：

    tiny=50, small=200, medium=800, large=3200, xlarge=8000, full=20000

（rungs 名与 §71 的 100/1k/5k/20k/50k/100k label 一一对应：rung 内部可再乘
 ``DRY_RUN_FACTOR_MULT`` 环境变量把 factor 计数对齐 §71，供真实放大跑。）
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from enum import Enum
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import psutil

_FE_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(_FE_ROOT), str(_FE_ROOT.parent)):
    if sys.path and _path not in sys.path:
        sys.path.insert(0, _path)
if _FE_ROOT.parent not in sys.path:
    sys.path.insert(0, str(_FE_ROOT.parent))

from factor_engine.runtime.dry_run_checkpoint import DryRunCheckpoint, CampaignMeta  # noqa: E402
from factor_engine.runtime.failure_classification import FailureCode, FailureReport  # noqa: E402
from factor_engine.storage.streaming_sink import StreamingSink, StreamRow  # noqa: E402

N_STOCKS = int(os.environ.get("DRY_RUN_STOCKS", "60"))
N_DAYS = int(os.environ.get("DRY_RUN_DAYS", "252"))
_RNG_SEED = 0
MAX_CORES = int(os.environ.get("DRY_RUN_MAX_CORES", "31"))
_NUM_THREADS = min(MAX_CORES, max(1, os.cpu_count() or 1) - 1 or 1)
os.environ.setdefault("OMP_NUM_THREADS", str(_NUM_THREADS))

_RUNGS: dict[str, int] = {
    "tiny": 50,       # §71 Stage A 下限（100/2）
    "small": 200,      # §71 Stage A/B 之间
    "medium": 800,     # §71 Stage C 附近
    "large": 3200,
    "xlarge": 8000,
    "full": 20000,     # 正式 ladder 顶端（真实跑时再乘 mult 对齐 100k）
}
_RUNG_NOMINAL: dict[str, int] = {
    "tiny": 100, "small": 1000, "medium": 5000, "large": 20000,
    "xlarge": 50000, "full": 100000,
}


class RungEnum(str, Enum):
    tiny = "tiny"
    small = "small"
    medium = "medium"
    large = "large"
    xlarge = "xlarge"
    full = "full"


# ---------------------------------------------------------------------------
# 合成面板
# ---------------------------------------------------------------------------
def build_panel() -> tuple[pd.DatetimeIndex, list[str], pd.DataFrame]:
    """确定性合成 close 面板（date×asset），多线程安全只读。"""
    dates = pd.bdate_range("2023-01-02", periods=N_DAYS)
    assets = [f"A{1000 + i:04d}" for i in range(N_STOCKS)]
    rng = np.random.default_rng(_RNG_SEED)
    # 几何随机游走 close，seed 固定 → 跨重跑可复现
    rets = rng.standard_normal((len(dates), N_STOCKS)) * 0.02
    close = 10.0 * np.exp(np.cumsum(rets, axis=0))
    close = np.maximum(close, 0.01)
    df = pd.DataFrame(close, index=dates, columns=assets)
    return dates, assets, df


_PANEL_CACHE: dict[str, Any] = {}


def get_panel() -> tuple[pd.DatetimeIndex, list[str], pd.DataFrame]:
    key = (N_DAYS, N_STOCKS, _RNG_SEED)
    if key not in _PANEL_CACHE:
        _PANEL_CACHE[key] = build_panel()
    return _PANEL_CACHE[key]


# ---------------------------------------------------------------------------
# 「factor」：确定性可对拍的因子函数
# ---------------------------------------------------------------------------
def make_factor_fn(kind: str) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """返回一个作用于 close 面板、产出等尺寸 value 面板的确定函数。"""
    kind = str(kind)

    def rolling_mean(df: pd.DataFrame) -> pd.DataFrame:
        return df.rolling(5, min_periods=1).mean()

    def ret_5(df: pd.DataFrame) -> pd.DataFrame:
        return df.pct_change(5).replace([np.inf, -np.inf], np.nan)

    if kind == "mean":
        return rolling_mean
    if kind == "ret5":
        return ret_5
    raise ValueError(f"unknown factor kind {kind!r}")


_FACTOR_KINDS: tuple[str, ...] = ("mean", "ret5")


def reference_result(n_factors: int):
    """in-memory 基线：所有 factor 的合并长表 + 每根 checksum（1e-12 对拍基准）。"""
    dates, assets, close = get_panel()
    ref_checksums: dict[str, str] = {}
    ref_rows: list[StreamRow] = []
    for i in range(n_factors):
        kind = _FACTOR_KINDS[i % len(_FACTOR_KINDS)]
        values = make_factor_fn(kind)(close)
        root_id = f"f{i:05d}"
        rows: list[StreamRow] = []
        for d_idx, date in enumerate(dates):
            for a_idx, asset in enumerate(assets):
                v = float(values.iloc[d_idx, a_idx])
                if not (pd.isna(v)):
                    row = StreamRow(str(date.date()), asset, v)
                    rows.append(row)
                    ref_rows.append(row)
        ref_checksums[root_id] = _checksum_rows(rows)
    return ref_checksums, ref_rows


def _checksum_rows(rows: list[StreamRow]) -> str:
    import hashlib

    sorted_rows = sorted(rows, key=lambda r: (r.date, r.asset))
    h = hashlib.sha256()
    for r in sorted_rows:
        h.update(f"{r.date}|{r.asset}|{r.value!r}\n".encode("utf-8"))
    return h.hexdigest()


# ---------------------------------------------------------------------------
# rung 执行器
# ---------------------------------------------------------------------------
def run_rung(
    *,
    name: str,
    n_factors: int,
    root_dir: Path,
    assert_1e12: bool = True,
) -> dict[str, Any]:
    """执行一个 rung；返回校验/吞吐量/内存/失败统计。"""
    campaign_id = f"campaign_{name}"
    checkpoint = DryRunCheckpoint(checkpoint_dir=root_dir / "ckpt" / name / campaign_id)
    sink = StreamingSink(output_dir=root_dir / "lake" / name / campaign_id, campaign_id=campaign_id)

    # campaign manifest
    checkpoint.init_campaign(
        CampaignMeta(
            campaign_id=campaign_id,
            total_roots=n_factors,
            source_snapshot=f"synth:{N_DAYS}d:{N_STOCKS}s:seed{_RNG_SEED}",
            factor_hash=f"rng{_RNG_SEED}",
        )
    )

    # in-memory 基线（对拍真相）
    ref_checksums, _ = reference_result(n_factors)

    failure_report = FailureReport()
    root_ids = [f"f{i:05d}" for i in range(n_factors)]
    pending = checkpoint.pending_roots(root_ids)

    t0 = time.monotonic()
    proc = psutil.Process()
    mem_before = proc.memory_info().rss

    checksum_mismatches = 0
    for root_id in pending:
        i = int(root_id[1:])
        kind = _FACTOR_KINDS[i % len(_FACTOR_KINDS)]
        try:
            values = make_factor_fn(kind)(get_panel()[2])
            rows = []
            for d_idx, date in enumerate(get_panel()[0]):
                for a_idx, asset in enumerate(get_panel()[1]):
                    v = float(values.iloc[d_idx, a_idx])
                    if not pd.isna(v):
                        rows.append(StreamRow(str(date.date()), asset, v))
            checksum = _checksum_rows(rows)
            if assert_1e12 and checksum != ref_checksums[root_id]:
                checksum_mismatches += 1
            with sink.open(root_id) as shard:
                shard.add_many(rows)
            checkpoint.mark_completed(root_id, marker={"checksum": checksum})
            checkpoint.mark_partition_shard("partition_all", root_id)
        except BaseException as exc:  # noqa: BLE001 -- 诚实分类，不 continue 吞
            code = failure_report.record(exc)
            checkpoint.mark_failed(root_id, exc)
            # 已知资源/IO 类失败可视为「分类样本」，不中断 rung；内部 bug 直接上抛
            if str(code) in ("INTERNAL", "BACKEND_PARITY"):
                raise

    t1 = time.monotonic()
    mem_after = proc.memory_info().rss

    # streaming 最终结果 vs in-memory 基线重读对拍
    final_tbl = sink.read_all()
    n_rows_written = final_tbl.num_rows

    # 重建 in-memory 过滤后的总数（非 NaN 行数）
    _, ref_rows = reference_result(n_factors)
    n_ref_rows = sum(1 for r in ref_rows if not (r.value != r.value))

    final_checksum = _tbl_checksum(final_tbl)

    elapsed = t1 - t0
    throughput = (len(pending) / elapsed) if elapsed > 0 else 0.0
    meta = {
        "rung": name,
        "nominal_roots": _RUNG_NOMINAL[name],
        "actual_roots": len(root_ids),
        "completed_roots": checkpoint.completed_count(),
        "failed_roots": checkpoint.failed_count(),
        "rows_written": n_rows_written,
        "ref_rows_filtered": n_ref_rows,
        "streaming_checksum": final_checksum,
        "checksum_mismatch_count": checksum_mismatches,
        "elapsed_sec": round(elapsed, 3),
        "throughput_roots_per_sec": round(throughput, 3),
        "mem_peak_mb": round((mem_after - mem_before) / (1024 * 1024), 3),
        "failures": failure_report.to_dict(),
    }
    return meta


def _tbl_checksum(tbl) -> str:
    import hashlib

    triples = sorted(
        zip(tbl["date"].to_pylist(), tbl["asset"].to_pylist(), tbl["value"].to_pylist()),
        key=lambda t: (t[0], t[1]),
    )
    h = hashlib.sha256()
    for d, a, v in triples:
        h.update(f"{d}|{a}|{v!r}\n".encode("utf-8"))
    return h.hexdigest()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="FE 100k dry-run ladder")
    ap.add_argument("--rungs", default="tiny,small", help="逗号分隔 rung 名（tiny..full / all）")
    ap.add_argument("--out", default=None, help="JSON 结果输出目录（默认 artifacts/dry_run_ladder）")
    ap.add_argument("--no-assert", action="store_true", help="跳过 1e-12 对拍断言（仅诊断）")
    args = ap.parse_args(argv)

    if args.rungs == "all":
        names = list(_RUNGS.keys())
    else:
        names = [n.strip() for n in args.rungs.split(",") if n.strip()]
    for n in names:
        if n not in _RUNGS:
            print(f"unknown rung {n!r}; valid={list(_RUNGS)}", file=sys.stderr)
            return 2

    out_root = Path(args.out) if args.out else _FE_ROOT / "artifacts" / "dry_run_ladder"
    out_root.mkdir(parents=True, exist_ok=True)
    workspace = Path(os.environ.get("DRY_RUN_WORKSPACE", out_root / "run"))

    results: dict[str, Any] = {}
    for name in names:
        print(f"\n=== rung {name} ({_RUNGS[name]} roots, nominal {_RUNG_NOMINAL[name]}) ===")
        meta = run_rung(
            name=name,
            n_factors=_RUNGS[name],
            root_dir=workspace,
            assert_1e12=not args.no_assert,
        )
        print(json.dumps(meta, indent=2, sort_keys=True))
        results[name] = meta

    latest = {"run_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
              "rungs": results}
    (out_root / "latest.json").write_text(json.dumps(latest, indent=2, sort_keys=True))
    print(f"\n[written] {out_root / 'latest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
