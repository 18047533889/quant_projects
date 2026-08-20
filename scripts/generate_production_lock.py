#!/usr/bin/env python
"""R21-119..123: generate an exact production dependency lock.

pyproject keeps library-compatible ``>=`` ranges; production deployment must
resolve exact constraints.  This script freezes the *currently installed*
versions of the production-critical packages into ``requirements-production.lock``
and records a lock digest for build artifacts.

Usage:
    python scripts/generate_production_lock.py [--out requirements-production.lock]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from importlib.metadata import PackageNotFoundError, version

PRODUCTION_PACKAGES = [
    "python", "numpy", "pandas", "pyarrow", "scipy", "numba", "polars",
    "duckdb", "clickhouse-connect", "data-access", "factor-engine", "fastapi",
    "pydantic", "uvicorn", "yaml",
]

_IMPORT_MODS = {
    "python": "sys",
    "yaml": "yaml",
    "data-access": "data_access",
    "factor-engine": "factor_engine",
    "clickhouse-connect": "clickhouse_connect",
    "fastapi": "fastapi",
    "pydantic": "pydantic",
    "uvicorn": "uvicorn",
}


def _version_of(pkg: str) -> str:
    if pkg == "python":
        return sys.version.split()[0]
    mod = _IMPORT_MODS.get(pkg, pkg.replace("-", "_"))
    try:
        return version(pkg)
    except PackageNotFoundError:
        pass
    try:
        __import__(mod)
        import importlib

        m = importlib.import_module(mod)
        return str(getattr(m, "__version__", "unknown"))
    except Exception:
        return "missing"


def generate_lock() -> tuple[dict[str, str], str]:
    entries: dict[str, str] = {pkg: _version_of(pkg) for pkg in PRODUCTION_PACKAGES}
    digest = hashlib.sha256(
        json.dumps(entries, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return entries, digest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate production dependency lock")
    parser.add_argument("--out", default="requirements-production.lock")
    args = parser.parse_args(argv)
    entries, digest = generate_lock()
    lines = [f"# R21-119..122 production exact dependency lock", f"# lock_digest={digest}", ""]
    for pkg in PRODUCTION_PACKAGES:
        lines.append(f"{pkg}=={entries[pkg]}")
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {args.out} (digest={digest})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
