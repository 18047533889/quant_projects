#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run test_all_canonicals_execute in memory-bounded batches.

The single parametrized file (1436 canonicals) accumulates too much memory in
one process on this small server.  We split the canonical list into chunks and
run ``pytest`` per chunk (each chunk is a fresh subprocess whose memory is
reclaimed).  Each batch passes ``R28_EXECUTE_SUBSET`` so the test only collects
that chunk; node ids are identical to the unbatched run.

Run:  python3 scripts/run_r28_execute_batches.py [chunk_size]
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def main() -> None:
    chunk = int(sys.argv[1]) if len(sys.argv) > 1 else 250
    os.chdir(REPO)
    sys.path.insert(0, str(REPO))
    from cleaned_operators import load_all
    from cleaned_operators.registry import OperatorRegistry

    load_all()
    canonicals = sorted(OperatorRegistry.list_canonical())
    chunks = [canonicals[i : i + chunk] for i in range(0, len(canonicals), chunk)]
    print(f"total={len(canonicals)} chunks={len(chunks)} chunk_size={chunk}", flush=True)

    failed_batches = 0
    for idx, batch in enumerate(chunks):
        env = dict(os.environ)
        env["R28_EXECUTE_SUBSET"] = ",".join(batch)
        cmd = [
            sys.executable, "-m", "pytest",
            "tests/operators/r28/test_all_canonicals_execute.py",
            "-q",
        ]
        print(f"chunk {idx + 1}/{len(chunks)}: {len(batch)} canonicals", flush=True)
        result = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True)
        tail = result.stdout.strip().splitlines()
        print("\n".join(tail[-5:]), flush=True)
        if result.returncode != 0:
            failed_batches += 1
            print(f"  !! batch {idx + 1} had failures/errors", flush=True)
            (Path("/tmp") / f"r28_batch_fail_{idx:02d}.txt").write_text(
                result.stdout + "\n" + result.stderr, encoding="utf-8"
            )

    print(f"\nDONE: {len(chunks)} batches, {failed_batches} with failures/errors")


if __name__ == "__main__":
    main()
