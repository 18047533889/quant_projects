#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLI: factor_engine DSL / Python → LQTP formula converter.

Examples
--------
Convert one DSL formula::

    python -m scripts.cogalpha_lqtp.convert_to_lqtp \\
        --dsl 'tanh(clip(cs_rank(close), -3, 3))'

Convert a Python factor file::

    python -m scripts.cogalpha_lqtp.convert_to_lqtp --python-file factor.py

Batch-convert a report::

    python -m scripts.cogalpha_lqtp.convert_to_lqtp \\
        --report-js /home/shw/reports/report_data.js \\
        --out-json /home/shw/reports/lqtp_submission_formulas.json \\
        --out-md /home/shw/reports/lqtp_submission_formulas.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FE_ROOT = ROOT / "factor_engine"
for p in (ROOT, FE_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from scripts.cogalpha_lqtp.lqtp_converter import (  # noqa: E402
    convert_auto,
    convert_dsl,
    convert_python,
    convert_report_factors,
    render_markdown,
)


def _print_one(result, *, as_json: bool) -> int:
    if as_json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(f"status : {result.status}")
        print(f"native : {result.lqtp_native}")
        if result.fe_formula:
            print(f"fe     : {result.fe_formula}")
        print(f"lqtp   : {result.lqtp_formula}")
        if result.approximations:
            print("approx :")
            for a in result.approximations:
                print(f"  - {a}")
        if result.notes:
            print("notes  :")
            for n in result.notes:
                print(f"  - {n}")
        if result.fe_only_ops:
            print(f"fe_only: {', '.join(result.fe_only_ops)}")
        if result.unknown_ops:
            print(f"unknown: {', '.join(result.unknown_ops)}")
    return 0 if result.status in {"ready", "review"} else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert factor_engine DSL or Python factors to LQTP formulas",
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--dsl", help="factor_engine DSL formula string")
    src.add_argument("--python", dest="python_code", help="Python factor source string")
    src.add_argument("--python-file", type=Path, help="Path to Python factor file")
    src.add_argument("--auto", help="Auto-detect DSL vs Python from text")
    src.add_argument(
        "--report-js",
        type=Path,
        help="Batch convert report_data.js factors",
    )
    parser.add_argument("--name", default="", help="Optional factor name for python mode")
    parser.add_argument("--out-json", type=Path, help="Write batch JSON result")
    parser.add_argument("--out-md", type=Path, help="Write batch markdown result")
    parser.add_argument("--json", action="store_true", help="Print single result as JSON")
    parser.add_argument(
        "--validate-fe",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Validate via factor_engine parse (default: on for single, off for report batch)",
    )
    args = parser.parse_args(argv)

    if args.report_js is not None:
        validate = False if args.validate_fe is None else bool(args.validate_fe)
        batch = convert_report_factors(args.report_js, validate_fe=validate)
        if args.out_json:
            args.out_json.write_text(
                json.dumps(batch, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(f"wrote {args.out_json}")
        if args.out_md:
            args.out_md.write_text(render_markdown(batch), encoding="utf-8")
            print(f"wrote {args.out_md}")
        if not args.out_json and not args.out_md:
            print(json.dumps(batch["summary"], ensure_ascii=False, indent=2))
            for row in batch["factors"]:
                print(
                    f"{row['status']:7} {row.get('factor_key') or '':20} "
                    f"{(row.get('lqtp_formula') or '')[:90]}"
                )
        summary = batch["summary"]
        return 0 if summary.get("blocked", 0) == 0 else 2

    validate = True if args.validate_fe is None else bool(args.validate_fe)
    if args.dsl is not None:
        result = convert_dsl(args.dsl, validate_fe=validate)
    elif args.python_code is not None:
        result = convert_python(args.python_code, name=args.name, validate_fe=validate)
    elif args.python_file is not None:
        code = args.python_file.read_text(encoding="utf-8")
        result = convert_python(
            code,
            name=args.name or args.python_file.stem,
            validate_fe=validate,
        )
    else:
        result = convert_auto(args.auto or "", validate_fe=validate)

    return _print_one(result, as_json=bool(args.json))


if __name__ == "__main__":
    raise SystemExit(main())
