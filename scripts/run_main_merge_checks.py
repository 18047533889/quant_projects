"""Run bounded pytest scopes and retain actual output plus source identities.

This does not commit, publish, create source copies, or certify unrun scopes.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def source_identity():
    paths = subprocess.check_output(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT).decode().split("\0")
    hashes = {}
    for relative in sorted(set(paths)):
        path = ROOT / relative
        if path.suffix not in (".py", ".toml", ".yml", ".yaml") or not path.is_file():
            continue
        if relative.startswith(("evidence/", "logs/", "loop/")):
            continue
        hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return digest, hashes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("tests", nargs="+")
    args = parser.parse_args()
    if not args.name.replace("_", "").isalnum():
        raise ValueError("name must be alphanumeric/underscore")
    for test in args.tests:
        path = (ROOT / test).resolve()
        if not path.is_relative_to(ROOT) or not path.exists():
            raise ValueError("test path must exist in formal checkout")
    folder = ROOT / "evidence/r2/main_merge_20260909"
    folder.mkdir(exist_ok=True)
    before, sources = source_identity()
    log = folder / (args.name + ".log")
    command = [sys.executable, "-m", "pytest", "-q", *args.tests]
    started = time.time()
    with log.open("w") as output:
        run = subprocess.run(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT)
    after, after_sources = source_identity()
    changes = {path: {"before": sources.get(path), "after": after_sources.get(path)}
               for path in sorted(sources.keys() | after_sources.keys())
               if sources.get(path) != after_sources.get(path)}
    payload = {
        "scope": args.name, "command": command, "executable": sys.executable,
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "working_tree_modified": True,
        "source_digest_before": before, "source_digest_after": after,
        "source_changes_during_run": changes,
        "source_hashes": sources, "exit_code": run.returncode,
        "status": "SOURCE_CHANGED_DURING_RUN" if before != after else "PASS" if run.returncode == 0 else "FAIL",
        "wall_seconds": time.time() - started, "started_unix": started,
        "log": str(log.relative_to(ROOT)), "log_sha256": hashlib.sha256(log.read_bytes()).hexdigest(),
        "limits": "Only named pytest scope; skipped/xfail tests are not executed PASS. No production or scale certification inferred.",
    }
    (folder / (args.name + ".json")).write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({k: v for k, v in payload.items() if k != "source_hashes"}))
    print("\n".join(log.read_text().splitlines()[-8:]))
    return run.returncode


if __name__ == "__main__":
    raise SystemExit(main())
