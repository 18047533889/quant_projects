#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""本周新挖因子：把待办候选切成 N 个互不重叠的分片，并行跑独立 orchestrator。

每个分片是自己的工作目录（NEWMINING_WORK），拥有独立的 candidates.json /
state.json / waves/ / factors/，因此 orchestrator 之间没有任何共享写冲突。
落值矩阵、落值 receipt、评估 artifact 都由主链路按“因子名/波浪目录”隔离，
分片并行不会互相覆盖（矩阵写盘为 tmp + os.replace 原子替换）。

分片跑完后用 `--merge` 把各分片的逐因子 manifest 与 state 合并回主目录。

子命令
------
--plan    构造/刷新分片目录（幂等：只纳入尚无 manifest 的因子）
--launch  后台启动各分片 orchestrator
--status  打印各分片与总体进度
--merge   合并分片结果回主工作目录
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path("/home/sunhaiwei/quant_projects")
JOBS = ROOT / "jobs"
PY = ROOT / ".venv/bin/python"
MAIN = ROOT / "work/newmining_20260919"
SHARDS = MAIN / "shards"
NAMES = ["a", "b"]


def _has_manifest(factors_dir: Path, name: str) -> bool:
    manifest = factors_dir / name / "report_manifest.json"
    return manifest.exists() and manifest.stat().st_size > 2


def done_names() -> set:
    """Factors with a usable manifest in the main tree or any shard tree."""
    done = set()
    for root in [MAIN / "factors"] + [SHARDS / f"shard_{n}" / "factors" for n in NAMES]:
        if not root.exists():
            continue
        for manifest in root.glob("*/report_manifest.json"):
            if manifest.stat().st_size > 2:
                done.add(manifest.parent.name)
    return done


def plan(shards: int):
    payload = json.loads((MAIN / "candidates.json").read_text())
    candidates = payload["candidates"]
    done = done_names()
    pending = [c for c in candidates if c["page_name"] not in done]
    print(f"[plan] candidates={len(candidates)} done={len(done)} pending={len(pending)} shards={shards}")
    if not pending:
        return
    active = NAMES[:shards]
    SHARDS.mkdir(parents=True, exist_ok=True)
    for i, shard in enumerate(active):
        folder = SHARDS / f"shard_{shard}"
        (folder / "factors").mkdir(parents=True, exist_ok=True)
        subset = pending[i::shards]
        (folder / "candidates.json").write_text(json.dumps(
            dict(payload, candidates=subset, rejected=[],
                 shard=shard, generated_at=payload.get("generated_at")),
            ensure_ascii=False, indent=1))
        # drop stale state entries for the subset so re-runs recompute cleanly
        state_path = folder / "state.json"
        state = json.loads(state_path.read_text()) if state_path.exists() else {"factors": {}}
        state.setdefault("factors", {})
        for c in subset:
            state["factors"].pop(c["page_name"], None)
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=1))
        print(f"        shard_{shard}: {len(subset)} factors")


def launch(shards: int, batch_size: int, memory_gib: int, threads: int, root_workers: int,
           window_years: int):
    active = NAMES[:shards]
    for shard in active:
        folder = SHARDS / f"shard_{shard}"
        log = folder / "waves_run.log"
        cmd = (f"cd {ROOT} && NEWMINING_WORK={folder} PYTHONWARNINGS=ignore nohup {PY} "
               f"jobs/new_mining_intake.py --waves --batch-size {batch_size} "
               f"--root-workers {root_workers} --threads {threads} --memory-gib {memory_gib} "
               f"--window-years {window_years} > {log} 2>&1 < /dev/null &")
        subprocess.run(["bash", "-lc", cmd], check=True)
        print(f"[launch] shard_{shard} started -> {log}")


def status(shards: int):
    total_candidates = len(json.loads((MAIN / "candidates.json").read_text())["candidates"])
    grand_done = 0
    for shard in NAMES[:shards]:
        folder = SHARDS / f"shard_{shard}"
        if not (folder / "candidates.json").exists():
            continue
        subset = json.loads((folder / "candidates.json").read_text())["candidates"]
        done = sum(1 for c in subset if _has_manifest(folder / "factors", c["page_name"]))
        running = 0
        for proc in os.popen("ps -eo cmd").read().splitlines():
            if "new_mining_intake.py --run-wave" in proc and f"shard_{shard}" in proc:
                running += 1
        grand_done += done
        print(f"  shard_{shard}: done={done}/{len(subset)} wave_running={running}")
    main_done = sum(1 for _ in (MAIN / "factors").glob("*/report_manifest.json"))
    print(f"[status] main_manifests={main_done} shard_manifests={grand_done} "
          f"total_candidates={total_candidates}")


def merge(shards: int):
    import shutil
    main_state_path = MAIN / "state.json"
    main_state = json.loads(main_state_path.read_text()) if main_state_path.exists() else {"factors": {}}
    main_state.setdefault("factors", {})
    copied = 0
    for shard in NAMES[:shards]:
        folder = SHARDS / f"shard_{shard}"
        src_root = folder / "factors"
        if src_root.exists():
            for src in sorted(src_root.glob("*/report_manifest.json")):
                name = src.parent.name
                dst = MAIN / "factors" / name
                dst.mkdir(parents=True, exist_ok=True)
                for f in src.parent.iterdir():
                    if f.is_file():
                        shutil.copy2(f, dst / f.name)
                # The shard wrote an absolute artifact path; repoint it at the
                # merged copy so the main tree stays self-contained after the
                # shard directories are cleaned up.
                try:
                    merged = dst / "report_manifest.json"
                    payload = json.loads(merged.read_text())
                    entry = (payload.get("factors") or {}).get(name) or {}
                    artifact = entry.get("artifact")
                    if artifact:
                        local = dst / Path(artifact).name
                        if local.exists():
                            payload["factors"][name]["artifact"] = str(local)
                            merged.write_text(json.dumps(payload, ensure_ascii=False, indent=1))
                except Exception:
                    pass
                copied += 1
        state_path = folder / "state.json"
        if state_path.exists():
            shard_state = json.loads(state_path.read_text()).get("factors", {})
            for name, info in shard_state.items():
                main_state["factors"][name] = info
    main_state_path.write_text(json.dumps(main_state, ensure_ascii=False, indent=1))
    print(f"[merge] copied {copied} per-factor manifests; state merged")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--launch", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--shards", type=int, default=2, choices=(1, 2, 3))
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--memory-gib", type=int, default=20)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--root-workers", type=int, default=6)
    ap.add_argument("--window-years", type=int, default=3)
    args = ap.parse_args()
    if args.plan:
        plan(args.shards)
    if args.launch:
        launch(args.shards, args.batch_size, args.memory_gib, args.threads,
               args.root_workers, args.window_years)
    if args.status:
        status(args.shards)
    if args.merge:
        merge(args.shards)
    if not any([args.plan, args.launch, args.status, args.merge]):
        ap.print_help()


if __name__ == "__main__":
    main()
