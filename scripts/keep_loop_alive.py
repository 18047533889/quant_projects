#!/usr/bin/env python3
"""Keep the coordinator looping after any subagent/turn ends.

Allow a real stop only if loop/HALT exists (user said stop).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HALT = ROOT / "loop" / "HALT"

REASON = (
    "KEEP GOING. A finished subagent is not session-complete. "
    "Immediately spawn the next role in the same cycle, or if the cycle just closed, "
    "start the next cycle now: Finder → Dispatcher → ≤2 disjoint Writers → Tester → Reviewer. "
    "Update loop/running.md. One status line. Then spawn again. "
    "Do not write a wrap-up and idle. Do not wait for the user. "
    "Stop only if loop/HALT exists or the user said 停 / stop / /clear."
)


def main() -> int:
    raw = sys.stdin.read()
    try:
        data = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError:
        data = {}

    if HALT.is_file():
        sys.stdout.write("{}\n")
        return 0

    # Claude Code: decision=block continues the parent. Cursor: followup_message.
    payload = {
        "decision": "block",
        "reason": REASON,
        "followup_message": REASON,
    }
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    # Claude Code also treats exit 2 as block-stop; keep 0 + JSON so Cursor stays valid.
    _ = data
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
