#!/usr/bin/env python3
"""Permanently bake sign flip into weekly factor formulas, then FE-retest.

For rows with sign_flipped=True (display |IC| but formula unsigned):
  - wrap DSL / LQTP / FE DSL as -(expr)
  - negate python returns if present
  - wipe lake year-parts so rematerialize uses new formula
  - clear display-only sign_flipped; mark formula_sign_flipped=True

Also patches retest helpers used by this run.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts" / "cogalpha_lqtp")]

from scripts.cogalpha_lqtp.run_production_batch import (  # noqa: E402
    _negate_formula,
    _negate_python_code,
)

WORK = ROOT / "data/cogalpha_lqtp_production"


def _balanced_parens(s: str) -> bool:
    depth = 0
    for ch in s:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def prep_fe_dsl(expr: str) -> str:
    """Strip outer CS rank for FE; keep a leading unary minus if present."""
    s = (expr or "").strip()
    if not s:
        return s
    # -(rank(inner))  /  -( RANK(inner) )
    m = re.fullmatch(r"-\s*\(\s*rank\s*\((.*)\)\s*\)", s, flags=re.I | re.S)
    if m and _balanced_parens(m.group(1)):
        return f"-({m.group(1).strip()})"
    # -rank(inner)
    m = re.fullmatch(r"-\s*rank\s*\((.*)\)", s, flags=re.I | re.S)
    if m and _balanced_parens(m.group(1)):
        return f"-({m.group(1).strip()})"
    # rank(inner)
    m = re.fullmatch(r"rank\s*\((.*)\)", s, flags=re.I | re.S)
    if m and _balanced_parens(m.group(1)):
        return m.group(1).strip()
    return s


def already_outer_negated(dsl: str) -> bool:
    t = (dsl or "").strip()
    if t.startswith("-(") and t.endswith(")") and _balanced_parens(t[1:]):
        return True
    if re.match(r"^-\s*rank\s*\(", t, flags=re.I):
        return True
    if re.match(r"^-\s*1\s*\*\s*rank\s*\(", t, flags=re.I):
        return True
    return False


def wipe_lake(work: Path, name: str) -> None:
    lake = work / "factor_lake" / name
    if not lake.exists():
        return
    parts = lake / "_year_parts"
    if parts.exists():
        shutil.rmtree(parts, ignore_errors=True)
    vp = lake / "values.parquet"
    if vp.exists():
        vp.unlink()
    # leave directory; materialize recreates parts


def permanent_flip_row(row: dict[str, Any]) -> bool:
    """Return True if formula was newly negated."""
    if row.get("formula_sign_flipped") and already_outer_negated(
        str(row.get("lqtp_formula") or row.get("dsl") or "")
    ):
        return False
    if not row.get("sign_flipped") and not row.get("force_formula_flip"):
        # only flip display-reversed rows unless forced
        return False

    show = str(row.get("lqtp_formula") or row.get("dsl") or "").strip()
    fe = str(row.get("fe_dsl") or "").strip()
    if not show or show.startswith("(python)"):
        # python-only stub: still try python_code
        py = str(row.get("python_code") or "")
        if py.strip() and "IC_SIGN_FLIPPED" not in py:
            row["python_code_before_flip"] = py
            row["python_code"] = _negate_python_code(py)
            row["formula_sign_flipped"] = True
            row["sign_flipped"] = False
            row["formula_flipped_at"] = datetime.now().isoformat()
            return True
        return False

    if already_outer_negated(show):
        row["formula_sign_flipped"] = True
        row["sign_flipped"] = False
        return False

    row["dsl_before_flip"] = show
    if fe:
        row["fe_dsl_before_flip"] = fe
    negated = _negate_formula(show)
    row["dsl"] = negated
    row["lqtp_formula"] = negated
    # FE: negate the stripped body (fe_dsl is usually already without outer rank)
    if fe and not already_outer_negated(fe):
        row["fe_dsl"] = _negate_formula(fe) if not fe.startswith("-") else fe
    else:
        row["fe_dsl"] = prep_fe_dsl(negated)

    py = str(row.get("python_code") or "")
    if py.strip() and "IC_SIGN_FLIPPED" not in py:
        row["python_code_before_flip"] = py
        row["python_code"] = _negate_python_code(py)

    row["formula_sign_flipped"] = True
    row["sign_flipped"] = False  # baked into formula; UI「取负显示」= 否
    row["ic_sign_flipped"] = True
    row["formula_flipped_at"] = datetime.now().isoformat()
    return True


def apply_flips(work: Path, *, only: set[str] | None = None) -> list[str]:
    wp = work / "reports/weekly_dug_neutral_rankic.json"
    weekly = json.loads(wp.read_text(encoding="utf-8"))
    flipped: list[str] = []
    for row in weekly.get("selected") or []:
        name = str(row.get("display_name") or row.get("factor_id"))
        if only and name not in only:
            continue
        if permanent_flip_row(row):
            wipe_lake(work, name)
            flipped.append(name)
            print(f"FLIP {name}", flush=True)
            print(f"  -> {str(row.get('lqtp_formula'))[:120]}", flush=True)
        elif row.get("formula_sign_flipped"):
            print(f"ALREADY {name}", flush=True)
    weekly["formula_sign_flip_at"] = datetime.now().isoformat()
    weekly["formula_sign_flip_n"] = len(flipped)
    weekly["formula_sign_flip_names"] = flipped
    wp.write_text(json.dumps(weekly, ensure_ascii=False, indent=2), encoding="utf-8")
    return flipped


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", type=Path, default=WORK)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    only = set(args.only) if args.only else None
    if args.dry_run:
        wp = args.work_dir / "reports/weekly_dug_neutral_rankic.json"
        weekly = json.loads(wp.read_text(encoding="utf-8"))
        n = 0
        for row in weekly.get("selected") or []:
            name = str(row.get("display_name") or "")
            if only and name not in only:
                continue
            if row.get("sign_flipped") and not already_outer_negated(
                str(row.get("lqtp_formula") or row.get("dsl") or "")
            ):
                n += 1
                print(f"WOULD_FLIP {name}", flush=True)
        print(f"dry-run would_flip={n}", flush=True)
        return 0
    flipped = apply_flips(args.work_dir, only=only)
    print(f"DONE flipped={len(flipped)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
