#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate deterministic direct, recipe and expanded-primitive usage report."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FACTOR_ENGINE = ROOT / "factor_engine"
if str(FACTOR_ENGINE) not in sys.path:
    sys.path.insert(0, str(FACTOR_ENGINE))

from factor_engine.cleaned_operators import load_all  # noqa: E402
from operator_usage_expanded import build_usage_report, scan_manifest_files  # noqa: E402


def main() -> None:
    load_all()
    report = build_usage_report(scan_manifest_files(ROOT), lookback_days=90)
    output = FACTOR_ENGINE / "docs" / "operator_usage_report.json"
    output.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
