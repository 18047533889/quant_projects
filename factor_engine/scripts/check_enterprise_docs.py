#!/usr/bin/env python3
# -*- coding: utf-8
"""文档一致性检查（CI 可调用）。"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    """运行文档一致性检查（LLM prompt md/txt 同步等）。"""
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
    if failed:
        print(f"{failed} doc check(s) failed", file=sys.stderr)
        return 1
    print("OK: docs checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
