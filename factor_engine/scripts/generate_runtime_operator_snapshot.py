#!/usr/bin/env python3
"""Generate the v3 runtime operator snapshot from a loaded registry."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from factor_engine.runtime.operator_snapshot import (
    build_runtime_operator_snapshot,
    load_evidence_records,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, action="append", default=[])
    parser.add_argument("--profile", default="runtime")
    args = parser.parse_args(argv)
    records = []
    for path in args.evidence:
        records.extend(load_evidence_records(path))
    snapshot = build_runtime_operator_snapshot(evidence_records=records, profile=args.profile)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
