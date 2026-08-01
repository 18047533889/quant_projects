#!/usr/bin/env python3
"""Regenerate committed cold-start catalogs and coverage reports."""
from __future__ import annotations

import json

from _bootstrap import REPO_ROOT  # noqa: F401
from factor_cold_start.generator import write_catalogs
from factor_cold_start.scripts.report_coverage import write_reports


def main() -> int:
    counts = write_catalogs(REPO_ROOT)
    report = write_reports(REPO_ROOT)
    print(json.dumps({"counts": counts, "coverage": report["summary"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
