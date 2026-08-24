#!/usr/bin/env python3
"""R30-P0-001 —— Benchmark 编排入口。

默认跑 B01/B02/B03/B04（small scale），输出 evidence 目录：
    BENCHMARK_ENV.json / BENCHMARK_RESULTS.parquet / BENCHMARK_SUMMARY.md

用法
    python -m data_access.benchmarks.run_benchmarks --scale small --out /tmp/bench_ev
    python -m data_access.benchmarks.run_benchmarks --workloads B01 B02 B03 B04 B05 B06 B09 \
        --scale small --out /tmp/bench_ev
    python -m data_access.benchmarks.run_benchmarks --compare /tmp/bench_prev --out /tmp/bench_ev
"""
from __future__ import annotations

import argparse
import datetime as dt
import tempfile
from pathlib import Path
from typing import Any

from data_access.benchmarks.fixtures import build_fixture_store
from data_access.benchmarks.report import BenchmarkReport, benchmark_env, compare


def _run_one(name: str, store, paths, scale: str, out_dir: Path | None):
    if name == "B01":
        from data_access.benchmarks import benchmark_local_daily as m

        return m.run_workload(store, paths, scale=scale, out_dir=str(out_dir) if out_dir else None)
    if name == "B02":
        from data_access.benchmarks import benchmark_local_minute as m

        return m.run_workload(store, paths, scale=scale, out_dir=str(out_dir) if out_dir else None)
    if name == "B03":
        from data_access.benchmarks import benchmark_join as m

        return m.run_workload(store, paths, scale=scale, out_dir=str(out_dir) if out_dir else None)
    if name in ("B04", "B05", "B06"):
        from data_access.benchmarks import benchmark_factor_batch as m

        counts = m._FACTOR_COUNTS.get(scale, m._FACTOR_COUNTS["small"])
        return m.run_factor_batch(
            store, paths, factor_count=counts[name], scale=scale,
            label=name, out_dir=str(out_dir) if out_dir else None,
        )
    if name == "B09":
        from data_access.benchmarks import benchmark_incremental as m

        return m.run_workload(store, paths, scale=scale, out_dir=str(out_dir) if out_dir else None)
    if name == "B-session":
        from data_access.benchmarks import benchmark_session_reuse as m

        return m.run_workload(store, paths, scale=scale, out_dir=str(out_dir) if out_dir else None)
    raise ValueError(f"未知 workload: {name}")


def _load_previous(out_dir: Path, name: str) -> BenchmarkReport | None:
    """从上一个 evidence 目录载入 report（parquet 优先，json 兜底）。"""
    for ext in (".parquet", ".json"):
        p = out_dir / name / f"BENCHMARK_RESULTS{ext}"
        if not p.exists():
            continue
        try:
            if ext == ".parquet":
                import pandas as pd

                row = pd.read_parquet(p).iloc[0].to_dict()
            else:
                import json

                row = json.loads(p.read_text(encoding="utf-8"))
            rep = BenchmarkReport(str(row.get("name") or name), "baseline", str(row.get("scale") or "?"))
            rep.metrics = dict(row.get("metrics") or {})
            rep.extra = dict(row.get("extra") or {})
            rep._gates = list(row.get("gates") or [])
            return rep
        except Exception:
            return None
    return None


def _arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="R30-P0-001 benchmark orchestration")
    p.add_argument("--scale", choices=["tiny", "small", "full"], default="small")
    p.add_argument("--out", type=str, default=None, help="evidence 输出目录")
    p.add_argument("--tmp", type=str, default=None)
    p.add_argument(
        "--workloads",
        type=str,
        nargs="*",
        default=["B01", "B02", "B03", "B04"],
        help="要跑的 workload（默认 B01 B02 B03 B04）",
    )
    p.add_argument("--compare", type=str, default=None, help="与上一个 evidence 目录对比")
    return p


def main(argv=None) -> int:
    args: Any = _arg_parser().parse_args(argv)
    out = Path(args.out) if args.out else Path(tempfile.mkdtemp(prefix="bench_ev_"))
    out.mkdir(parents=True, exist_ok=True)
    tmp = Path(args.tmp) if args.tmp else Path(tempfile.mkdtemp(prefix="bench_fix_"))

    need_minute = any(w in ("B02",) for w in args.workloads)
    store, paths = build_fixture_store(
        tmp, scale=args.scale, freq="both" if need_minute else "daily"
    )

    reports: dict[str, BenchmarkReport] = {}
    for name in args.workloads:
        sub = out / name
        reports[name] = _run_one(name, store, paths, args.scale, sub)

    # 对比
    compare_rows: list[dict[str, Any]] = []
    if args.compare:
        prev = Path(args.compare)
        for name, rep in reports.items():
            base = _load_previous(prev, name)
            if base is not None:
                cmp = compare(base, rep)
                compare_rows.append(
                    {
                        "workload": name,
                        "verdict": cmp["verdict"],
                        "delta_wall_ms": (cmp["deltas"] or {}).get("wall_ms"),
                    }
                )

    env = benchmark_env()
    summary = _render_summary(reports, compare_rows, env)
    (out / "BENCHMARK_SUMMARY.md").write_text(summary, encoding="utf-8")
    _write_combined(out, reports, env)
    print(f"evidence → {out}")
    for name, rep in reports.items():
        print(f"  {name}: verdict={rep.gate_verdict()} wall_ms={rep.metrics.get('wall_ms')}")
    for c in compare_rows:
        print(f"  compare {c['workload']}: {c['verdict']} delta_wall_ms={c['delta_wall_ms']}")
    return 0


def _render_summary(reports, compare_rows, env) -> str:
    lines = [
        "# R30-P0-001 Benchmark Suite",
        "",
        f"- scale: `{next(iter(reports.values())).scale if reports else '?'}`",
        f"- commit: {env.get('commit_sha') or 'n/a'}  version: {env.get('version') or 'n/a'}",
        f"- platform: {env.get('platform')}  cpu: {env.get('cpu_model') or env.get('cpu_count')}",
        f"- ram_gb: {env.get('ram_gb')}  python: {env.get('python')}",
        "",
        "| workload | verdict | wall_ms | rows/s | scan_count |",
        "|---|---|---|---|---|",
    ]
    for name, rep in reports.items():
        lines.append(
            f"| {name} | {rep.gate_verdict()} | {rep.metrics.get('wall_ms')} | "
            f"{rep.metrics.get('rows_per_sec')} | {rep.metrics.get('physical_scan_count')} |"
        )
    if compare_rows:
        lines.append("")
        lines.append("## vs baseline")
        lines.append("")
        for c in compare_rows:
            lines.append(f"- {c['workload']}: {c['verdict']} delta_wall_ms={c['delta_wall_ms']}")
    lines.append("")
    return "\n".join(lines)


def _write_combined(out: Path, reports, env) -> None:
    try:
        import pandas as pd

        rows = [rep.to_dict() for rep in reports.values()]
        pd.DataFrame(rows).to_parquet(out / "BENCHMARK_RESULTS.parquet", index=False)
    except Exception:
        import json

        (out / "BENCHMARK_RESULTS.json").write_text(
            json.dumps([rep.to_dict() for rep in reports.values()], indent=2, default=str),
            encoding="utf-8",
        )
    (out / "BENCHMARK_ENV.json").write_text(
        __import__("json").dumps(env, indent=2, default=str), encoding="utf-8"
    )


if __name__ == "__main__":
    raise SystemExit(main())
