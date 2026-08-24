#!/usr/bin/env python3
"""R30-P0-001 —— B07/B08 固定 workload：COS cold / warm 读。

需要真实 COS 凭证 + boto3 + 已注册的 cos:// 数据集。无真实 COS 时**如实 SKIP**
（记录 status=SKIP + 原因），绝不假通过。

用法
    python -m data_access.benchmarks.benchmark_cos --scale tiny --out /tmp/b07b08
"""
from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from typing import Any

from data_access.benchmarks.fixtures import build_fixture_store
from data_access.benchmarks.report import BenchmarkReport

WORKLOAD = "B07/B08 COS cold/warm read (SKIP without real COS)"


def _cos_reason() -> str | None:
    """返回 None = COS 可用；否则返回 SKIP 原因字符串。"""
    if os.environ.get("DATA_ACCESS_SKIP_COS_MIRROR", "").strip() == "1":
        return "DATA_ACCESS_SKIP_COS_MIRROR=1（镜像关闭）"
    mode = os.environ.get("DATA_ACCESS_COS_READ_MODE", "mirror")
    if mode == "mirror":
        return "DATA_ACCESS_COS_READ_MODE=mirror（本地镜像模式，非真实 COS）"
    try:
        import boto3  # noqa: F401
    except Exception:
        return "boto3 未安装"
    try:
        from data_access.cos.remote import resolve_s3_credentials

        creds = resolve_s3_credentials()
    except Exception as exc:  # noqa: BLE001
        return f"COS 凭证解析失败：{str(exc)[:120]}"
    if not creds:
        return "未配置 COS/S3 凭证"
    return None


def _cos_cold_warm(store, reason: str, scale: str, label: str):
    report = BenchmarkReport(label, f"{label} COS read", scale)
    if reason is not None:
        report.mark_skipped(reason)
        report.finish()
        return report
    # 有真实 COS 时：best-effort 冷/热读（需要已注册 cos:// 数据集，这里没有 → SKIP）
    report.mark_skipped("存在 COS 凭证但 benchmark 无固定 cos:// 数据集（B07/B08 需接入真实数据集）")
    report.finish()
    return report


def run_workload(store, paths, *, scale="small", repeat=1, out_dir=None):
    """返回 [B07, B08] 报告。无真实 COS → 两个都 SKIP。"""
    reason = _cos_reason()
    reports = [
        _cos_cold_warm(store, reason, scale, "B07"),
        _cos_cold_warm(store, reason, scale, "B08"),
    ]
    if out_dir:
        base = Path(out_dir)
        for r in reports:
            r.write_evidence(base / r.name)
    return reports


def _arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=WORKLOAD)
    p.add_argument("--scale", choices=["tiny", "small", "full"], default="tiny")
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--tmp", type=str, default=None)
    return p


def main(argv=None) -> int:
    args: Any = _arg_parser().parse_args(argv)
    reason = _cos_reason()
    if reason is not None:
        # 无真实 COS → 打印 SKIP + 原因；run_workload 负责记录 SKIP 证据。
        print(f"COS workload SKIP：{reason}")
        if args.out:
            base = Path(args.out)
            for label in ("B07", "B08"):
                r = _cos_cold_warm(None, reason, args.scale, label)
                r.write_evidence(base / label)
        return 0
    tmp = Path(args.tmp) if args.tmp else Path(tempfile.mkdtemp(prefix="bcos_"))
    store, paths = build_fixture_store(tmp, scale=args.scale, freq="daily")
    reports = run_workload(store, paths, scale=args.scale, repeat=args.repeat, out_dir=args.out)
    for r in reports:
        print(f"{r.name} verdict={r.gate_verdict()} status={r.metrics.get('status')} "
              f"reason={r.extra.get('skip_reason')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
