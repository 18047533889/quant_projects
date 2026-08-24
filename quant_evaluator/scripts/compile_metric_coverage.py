#!/usr/bin/env python3
"""Generate quant_evaluator/docs/METRIC_COVERAGE_COMPILER.csv.

This is a thin executable wrapper around the real compiler in
``quant_evaluator.metrics.coverage_compiler``.  All scanning logic lives in
the library module so tests can exercise it directly.

Usage:
    /tmp/fe2/bin/python quant_evaluator/scripts/compile_metric_coverage.py
"""

import os
import sys


def main() -> int:
    # Ensure the repo root is importable (script may run from anywhere).
    _here = os.path.dirname(os.path.abspath(__file__))
    _root = os.path.dirname(_here)  # quant_evaluator/
    if _root not in sys.path:
        sys.path.insert(0, _root)
    parent = os.path.dirname(_root)
    if parent not in sys.path:
        sys.path.insert(0, parent)

    from quant_evaluator.metrics.coverage_compiler import compile_metric_coverage

    out = compile_metric_coverage()
    print(f"METRIC_COVERAGE_COMPILER.csv -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
