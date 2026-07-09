#!/usr/bin/env python3
"""Bucket 分区迁移一键 playbook：plan → dry-run → execute → validate → cutover 清单。"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _scripts_dir() -> Path:
    return Path(__file__).resolve().parent


def _run_module(script: str, argv: list[str]) -> tuple[int, str]:
    path = _scripts_dir() / script
    proc = subprocess.run(
        [sys.executable, str(path), *argv],
        capture_output=True,
        text=True,
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode, out.strip()


def run_playbook(
    *,
    dataset: str,
    target_root: str | Path,
    config: str | None = None,
    execute: bool = False,
    max_files: int = 0,
    tolerance_ratio: float = 0.001,
    skip_validate: bool = False,
) -> dict:
    """执行完整 bucket 迁移流程并返回各步摘要。"""
    target = Path(target_root)
    cfg_args = ["--config", config] if config else []
    steps: dict[str, Any] = {}

    plan_argv = ["--dataset", dataset, *cfg_args, "--json"]
    code, out = _run_module("plan_bucket_migration.py", plan_argv)
    if code != 0:
        raise RuntimeError(f"plan 失败 (exit={code}): {out}")
    plan = json.loads(out)
    steps["plan"] = plan

    dry_argv = [
        "--dataset",
        dataset,
        *cfg_args,
        "--target-root",
        str(target),
        "--json",
    ]
    if max_files:
        dry_argv.extend(["--max-files", str(max_files)])
    code, out = _run_module("execute_bucket_migration.py", dry_argv)
    if code != 0:
        raise RuntimeError(f"dry-run 失败 (exit={code}): {out}")
    steps["dry_run"] = json.loads(out)

    if execute:
        exec_argv = dry_argv + ["--execute"]
        code, out = _run_module("execute_bucket_migration.py", exec_argv)
        if code != 0:
            raise RuntimeError(f"execute 失败 (exit={code}): {out}")
        steps["execute"] = json.loads(out)

    validation: dict | None = None
    if execute and not skip_validate:
        val_argv = [
            "--dataset",
            dataset,
            *cfg_args,
            "--target-root",
            str(target),
            "--tolerance-ratio",
            str(tolerance_ratio),
            "--json",
        ]
        code, out = _run_module("validate_bucket_migration.py", val_argv)
        validation = json.loads(out) if out else {"passed": code == 0}
        steps["validate"] = validation
        if not validation.get("passed", False):
            raise RuntimeError(f"validate 未通过: {validation}")

    cutover = build_cutover_checklist(
        dataset=dataset,
        target_root=str(target),
        plan=plan,
        executed=execute,
        validation=validation,
    )
    steps["cutover"] = cutover
    return {
        "dataset": dataset,
        "target_root": str(target),
        "executed": execute,
        "steps": steps,
        "ok": True,
    }


def build_cutover_checklist(
    *,
    dataset: str,
    target_root: str,
    plan: dict,
    executed: bool,
    validation: dict | None,
) -> dict:
    """生成切读/双写切换操作清单（人工或自动化）。"""
    partition_cols = plan.get("recommended_partition_columns") or ["year", "month", "bucket"]
    env_key = f"DATA_ACCESS_READ_ROOT_{dataset.upper().replace('-', '_')}"
    return {
        "dataset": dataset,
        "target_root": target_root,
        "validation_passed": validation.get("passed") if validation else None,
        "checklist": [
            "1. 确认 validate 行数偏差在 tolerance 内",
            f"2. 在 datasets.yaml 为 {dataset!r} 启用 partition_columns: {partition_cols}",
            f"3. 切读方式 A：将 root 改为 {target_root}（永久）",
            f"4. 切读方式 B（灰度）：export {env_key}={target_root} 后跑因子 smoke",
            "5. 刷新 sidecar: refresh_dataset_stats.py --with-null-ratio",
            "6. 跑 factor_engine read_auto smoke + input_dq",
            "7. 确认无回归后删除或归档源目录（保留备份窗口）",
        ],
        "yaml_patch_hint": {
            "partition_columns": partition_cols,
            "root_after_cutover": target_root,
        },
        "env_cutover": {env_key: target_root},
        "executed": executed,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Bucket 迁移 playbook")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--target-root", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--execute", action="store_true", help="实际 ETL（默认仅 plan+dry-run）")
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--tolerance-ratio", type=float, default=0.001)
    parser.add_argument("--skip-validate", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    try:
        result = run_playbook(
            dataset=args.dataset,
            target_root=args.target_root,
            config=args.config,
            execute=args.execute,
            max_files=args.max_files,
            tolerance_ratio=args.tolerance_ratio,
            skip_validate=args.skip_validate,
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"Playbook OK: {result['dataset']} -> {result['target_root']}")
        print(f"executed={result['executed']}")
        cutover = result["steps"]["cutover"]
        print("Cutover checklist:")
        for line in cutover["checklist"]:
            print(f"  {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
