#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate the intraday vector-coverage report (GO_PROMPT §6.1 / §6.3).

Writes:
  - artifacts/perf_vec/intraday_vector_coverage.json  (machine snapshot)
  - artifacts/perf_vec/INTRADAY_VECTOR_COVERAGE.md     (human table)

Usage:
  python3 scripts/generate_intraday_vector_coverage.py [--output PATH]

The ``--output`` path is the .json file; the .md is written alongside it (same
stem with ``.md``).  Both are produced only when the fail-closed assertion
(``count_bound() == len(bind_whitelist())``) holds.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# CLI self-bootstrap: when invoked directly (not via an installed package) the
# repo root and its parent (quant_projects) must be on sys.path before we can
# import ``factor_engine.cleaned_operators.intraday``.
for _root in (("..", os.path.dirname(os.path.abspath(__file__))), ("..", "..")):
    _p = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), *_root))
    if _p not in sys.path:
        sys.path.insert(0, _p)

from factor_engine.cleaned_operators.intraday import perf_vec_coverage as pvc
from factor_engine.cleaned_operators.intraday.perf_vec_telemetry import (
    get_telemetry_snapshot,
    set_vector_kernels_bound,
)

_DEFAULT_OUT = Path("artifacts/perf_vec/intraday_vector_coverage.json")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate intraday vector-coverage report")
    parser.add_argument(
        "--output",
        default=str(_DEFAULT_OUT),
        help="output .json path (default: %(default)s)",
    )
    args = parser.parse_args(argv)

    out_json = Path(args.output)
    out_md = out_json.with_suffix(".md")
    out_md = out_md.parent / f"{out_json.stem.upper()}.md"  # INTRADAY_VECTOR_COVERAGE.md
    out_json.parent.mkdir(parents=True, exist_ok=True)

    cov = pvc.build_vector_coverage()

    # Publish bound-kernel count to the §6.3 telemetry snapshot before rendering.
    set_vector_kernels_bound(cov["bound_count"])
    snapshot = get_telemetry_snapshot()

    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(cov, fh, ensure_ascii=False, indent=2)
    with open(out_md, "w", encoding="utf-8") as fh:
        fh.write(pvc.render_intraday_markdown(cov))

    s = cov["summary"]
    print(f"[perf_vec] wrote {out_json}")
    print(f"[perf_vec] wrote {out_md}")
    print(
        "[perf_vec] summary total={total} vectorized={vectorized} scalar_only={scalar_only}".format(
            total=s["total"], vectorized=s["vectorized"], scalar_only=s["scalar_only"]
        )
    )
    print(f"[perf_vec] telemetry={snapshot}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
