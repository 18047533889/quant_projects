#!/usr/bin/env python3
# -*- coding: utf-8
"""企业级文档一致性检查（CI 可调用）。"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    checks = [
        [
            sys.executable,
            str(ROOT / "scripts" / "sync_factor_engine_llm_prompt_txt.py"),
            "--check",
        ],
    ]
    failed = 0
    for cmd in checks:
        print("RUN:", " ".join(cmd))
        rc = subprocess.call(cmd, cwd=str(ROOT))
        if rc != 0:
            failed += 1
    roadmap = ROOT / "docs" / "enterprise_factor_engine_roadmap.md"
    text = roadmap.read_text(encoding="utf-8")
    if "Phase 14 — 企业级加固（已完成）" not in text:
        print(f"FAIL: {roadmap} missing Phase 14 completed marker", file=sys.stderr)
        failed += 1
    if "Phase 15 — 批量编排与写目标收敛（已完成）" not in text:
        print(f"FAIL: {roadmap} missing Phase 15 completed marker", file=sys.stderr)
        failed += 1
    if "Phase 16 — 物化批量与配置映射（已完成）" not in text:
        print(f"FAIL: {roadmap} missing Phase 16 completed marker", file=sys.stderr)
        failed += 1
    if "Phase 17 — pipeline 收敛与 allowlist（已完成）" not in text:
        print(f"FAIL: {roadmap} missing Phase 17 completed marker", file=sys.stderr)
        failed += 1
    if "Phase 18 — Pipeline 覆盖与增量批量（已完成）" not in text:
        print(f"FAIL: {roadmap} missing Phase 18 completed marker", file=sys.stderr)
        failed += 1
    if failed:
        print(f"{failed} enterprise doc check(s) failed", file=sys.stderr)
        return 1
    print("OK: enterprise docs checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
