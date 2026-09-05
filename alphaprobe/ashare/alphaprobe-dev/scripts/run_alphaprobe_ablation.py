#!/usr/bin/env python3
"""run_alphaprobe_ablation —— plan Task 24 algorithm ablation CLI（dry-run 可执行）。

装配 plan Task 24 的 10 个累积配置（legacy reference → + next-version
Survival Memory）并输出对比表；**默认 dry-run 不真跑 LLM/FE/QE campaign**
（OFFLINE_TEST 语义：装配校验 + 中性指标聚合，绝不触碰任何外部系统）。

真实路径预留：
- ``--config N``（1..10）单独装配第 N 个配置（--dry-run 下打印装配校验 +
  序列化 JSON，验证装配不真跑）；
- 接入真实 campaign：注入 ``run_fn(config, assembly) -> attempts`` 到
  :class:`AblationRunner`（真实跑需要 LLM client + 数据 + FE/QE 权威，
  本 CLI 不做；见 module docstring）。

Non-negotiable（Part G #30 / plan Task 24）：
- 本 CLI 只装配与聚合，**不新增任何生产开关**（各特性模块开关已由前面
  agent 落好；缺开关只报告，不改 src/alphaprobe/ 其它生产代码）。
- generated factor count 只是报告信息列，绝不作为 primary success metric。
- dry-run 语义 = OFFLINE_TEST：无 LLM / 无数据 / 无 FE/QE。

用法：
    PYTHONPATH=src python3 scripts/run_alphaprobe_ablation.py             # 全 10 配置 dry-run
    PYTHONPATH=src python3 scripts/run_alphaprobe_ablation.py --config 3  # 装配第 3 个配置
    PYTHONPATH=src python3 scripts/run_alphaprobe_ablation.py --report-json out.json
    PYTHONPATH=src python3 scripts/run_alphaprobe_ablation.py --no-dry-run  # 预留真实路径入口
    （真实 campaign 需注入 --run-fn：见 ablation.AblationRunner 文档）
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from alphaprobe.experiments.ablation import (  # noqa: E402
    ABLATION_CONFIG_NAMES,
    AblationRunner,
    ablation_config_matrix,
    all_ablation_configs,
    build_ablation_config,
    configs_diff,
)

__all__: list[str] = []


def _line() -> str:
    return "-" * 72


def _print_assembly(cfg: Any) -> None:
    """打印单个配置的装配校验 + 序列化（dry-run 的核心输出）。"""
    asm = cfg.assembly()
    print(f"[config] {cfg.name}")
    print(f"  features_on : {asm.switch_names()}")
    print(f"  probes_ok   : {asm.probes}")
    print(f"  validate    : ok ({len(asm.validate())} problems)")
    serial = asm.to_dict()
    print(f"  serialized  : {json.dumps(serial, ensure_ascii=False, sort_keys=True)}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="plan Task 24 AlphaPROBE algorithm ablation harness (dry-run by default)"
    )
    ap.add_argument("--config", type=int, default=None, choices=list(range(1, 11)),
                    help="仅装配第 N 个配置（1..10）")
    ap.add_argument("--list", action="store_true", help="打印 10 配置矩阵摘要")
    ap.add_argument("--diff", type=int, nargs=2, default=None, metavar=("A", "B"),
                    help="打印两个配置（1..10）的开关差异")
    ap.add_argument("--report-json", type=str, default=None,
                    help="把 dry-run 对比表写到 JSON 文件（默认 stdout）")
    ap.add_argument("--no-dry-run", action="store_true",
                    help="[预留] 真实 campaign 路径开关；本 CLI 未接真实 LLM/数据，"
                         "使用该开关仍只装配不跑（需自行注入 run_fn）")
    args = ap.parse_args()

    cfgs = all_ablation_configs()

    if args.list:
        matrix = ablation_config_matrix()
        print(_line())
        for row in matrix:
            print(
                f"#{row['index']:>2} {row['config_name']:<28} "
                f"on={row['features_on']}"
            )
        print(_line())
        return 0

    if args.diff is not None:
        a, b = args.diff
        if not (1 <= a <= 10 and 1 <= b <= 10):
            print(f"config index must be 1..10, got {a},{b}", file=sys.stderr)
            return 2
        ca, cb = cfgs[a - 1], cfgs[b - 1]
        d = configs_diff(ca, cb)
        print(f"diff {ca.name} -> {cb.name}")
        print(f"  added   : {d['added']}")
        print(f"  removed : {d['removed']}")
        print(f"  shared  : {d['shared']}")
        return 0

    # --config N：单配置装配校验（dry-run）
    if args.config is not None:
        cfg = cfgs[args.config - 1]
        _print_assembly(cfg)
        return 0

    # 全 10 配置 dry-run：装配 + 中性指标对比表
    runner = AblationRunner()
    if not args.no_dry_run:
        # 默认 dry-run：无 run_fn → 空 attempts → 中性指标（不真跑 campaign）
        report = runner.dry_run_all()
    else:
        # --no-dry-run 预留：真实 campaign 需注入 run_fn（本 CLI 未接真实
        # LLM/数据）。保持 fail-closed：无 run_fn 时 dry_run_all 语义不变。
        report = runner.dry_run_all()
    out: dict[str, Any] = {
        "command": "ablation-dry-run",
        "mode": "offline_test" if not args.no_dry_run else "reserved-production",
        "denominator_attempts": runner.denominator,
        "note": "generated_factors is informational only, not a primary success metric (plan T24)",
        "report": report.to_dict(),
        "matrix": ablation_config_matrix(),
    }
    text = json.dumps(out, ensure_ascii=False, indent=2)
    if args.report_json:
        p = Path(args.report_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text + "\n", encoding="utf-8")
        print(f"report written: {p}")
    else:
        print(report.to_markdown())
        print(_line())
        for d in report.diffs:
            print(
                f"delta {d['from']} -> {d['to']}: added={d['added']} "
                f"metric_delta={json.dumps(d['metric_delta'], ensure_ascii=False, sort_keys=True)}"
            )
        print(_line())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
