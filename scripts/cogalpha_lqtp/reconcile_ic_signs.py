#!/usr/bin/env python3
"""Reconcile IC sign under current panel returns: flip/unflip values + formulas."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.cogalpha_lqtp.data_access_panel import resolve_factor_values_parquet  # noqa: E402
from scripts.cogalpha_lqtp.run_production_batch import (  # noqa: E402
    _negate_formula,
    _negate_python_code,
    _negate_values_parquet,
)


def _unwrap_formula(dsl: str) -> str:
    text = (dsl or "").strip()
    if text.startswith("-(") and text.endswith(")"):
        inner = text[2:-1]
        # Only unwrap when outer parens are balanced for the whole wrap.
        depth = 0
        for i, ch in enumerate(inner):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth < 0:
                    return text
        if depth == 0:
            return inner
    return text


def _unwrap_python_code(code: str) -> str:
    text = (code or "").rstrip()
    if not text:
        return text
    lines_out: list[str] = []
    for line in text.splitlines():
        if "IC_SIGN_FLIPPED" in line and line.lstrip().startswith("#"):
            continue
        stripped = line.lstrip()
        if stripped.startswith("return -(") and stripped.endswith(")"):
            indent = line[: len(line) - len(stripped)]
            expr = stripped[len("return -(") : -1]
            lines_out.append(f"{indent}return {expr}")
        else:
            lines_out.append(line)
    return "\n".join(lines_out) + "\n"


def _invalidate_index_row(row: dict[str, Any], *, flipped: bool) -> None:
    """Mark row stale after value/formula sign change.

    Do **not** negate RankIC/LS numbers in place: decile curves and reports must be
    recomputed from the flipped lake values, otherwise RankIC and G10−G1 disagree.
    """
    row["ic_sign_flipped"] = flipped
    row["eval_mode"] = "pending_sign_reconcile"
    row.pop("mean_ic", None)
    row.pop("mean_rank_ic", None)
    row.pop("icir", None)
    row.pop("rank_icir", None)
    row.pop("long_short_return", None)
    row.pop("long_short_sharpe", None)
    row.pop("industry_neutral_mean_rank_ic", None)
    row.pop("size_neutral_mean_rank_ic", None)


def main() -> int:
    parser = argparse.ArgumentParser(description="Flip/unflip factors so RankIC >= 0")
    parser.add_argument("--work-dir", type=Path, default=ROOT / "data/cogalpha_lqtp_production")
    parser.add_argument("--progress-file", type=Path, default=None)
    parser.add_argument("--catalog", type=Path, default=None)
    parser.add_argument("--parsed-json", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    work = args.work_dir
    progress_path = args.progress_file or (work / "screening_reeval_progress.json")
    catalog_path = args.catalog or (work / "screening_reeval_catalog.json")
    parsed_path = args.parsed_json or (work / "screening_reeval_parsed_factors.json")

    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    cat_by = {e["function_name"]: e for e in catalog}
    python_map = {
        r["function_name"]: r.get("python_code", "")
        for r in (json.loads(parsed_path.read_text(encoding="utf-8")) if parsed_path.exists() else [])
    }
    flip_state: dict[str, Any] = progress.setdefault("flip_state", {})
    rows = {r["factor_name"]: r for r in progress.get("index_rows", []) if r.get("factor_name")}

    to_flip: list[str] = []
    to_unflip: list[str] = []
    for name, row in rows.items():
        try:
            ic = float(row.get("mean_rank_ic", row.get("mean_ic")))
        except (TypeError, ValueError):
            continue
        if ic != ic:
            continue
        flipped = bool(row.get("ic_sign_flipped") or flip_state.get(name, {}).get("ic_sign_flipped"))
        if ic < 0 and flipped:
            to_unflip.append(name)
        elif ic < 0 and not flipped:
            to_flip.append(name)

    print(f"return_kind={progress.get('return_kind')} flip={len(to_flip)} unflip={len(to_unflip)}")
    if args.dry_run:
        print("flip sample", to_flip[:10])
        print("unflip sample", to_unflip[:10])
        return 0

    changed: list[str] = []
    for name in to_unflip:
        pq = resolve_factor_values_parquet(work, name)
        if pq is None:
            print(f"skip unflip missing values {name}")
            continue
        _negate_values_parquet(pq)
        state = dict(flip_state.get(name, {}))
        entry = cat_by.get(name, {})
        dsl = state.get("dsl") or entry.get("dsl") or ""
        py = state.get("python_code") or python_map.get(name) or entry.get("python_code") or ""
        if dsl and dsl != "(python only)":
            state["dsl"] = _unwrap_formula(dsl)
            entry["dsl"] = state["dsl"]
            entry["lqtp_formula"] = state["dsl"]
        if py.strip():
            state["python_code"] = _unwrap_python_code(py)
            python_map[name] = state["python_code"]
        state["ic_sign_flipped"] = False
        state["values_negated"] = True
        state["sign_reconciled"] = "unflip_for_positive_vwap_ic"
        flip_state[name] = state
        entry["ic_sign_flipped"] = False
        if name in rows:
            _invalidate_index_row(rows[name], flipped=False)
            rows[name]["return_kind"] = progress.get("return_kind") or rows[name].get("return_kind")
        changed.append(name)
        print(f"unflip {name}")

    for name in to_flip:
        pq = resolve_factor_values_parquet(work, name)
        if pq is None:
            print(f"skip flip missing values {name}")
            continue
        _negate_values_parquet(pq)
        state = dict(flip_state.get(name, {}))
        entry = cat_by.get(name, {})
        dsl = state.get("dsl") or entry.get("dsl") or ""
        py = state.get("python_code") or python_map.get(name) or entry.get("python_code") or ""
        if dsl and dsl != "(python only)":
            state["dsl"] = _negate_formula(dsl)
            entry["dsl"] = state["dsl"]
            entry["lqtp_formula"] = state["dsl"]
        if py.strip():
            state["python_code"] = _negate_python_code(py)
            python_map[name] = state["python_code"]
        state["ic_sign_flipped"] = True
        state["values_negated"] = True
        state["sign_reconciled"] = "flip_for_positive_vwap_ic"
        flip_state[name] = state
        entry["ic_sign_flipped"] = True
        if name in rows:
            _invalidate_index_row(rows[name], flipped=True)
        changed.append(name)
        print(f"flip {name}")

    # Ensure already-positive flipped factors keep a leading minus in stored formula.
    ensured = 0
    for name, row in rows.items():
        if name in changed:
            continue
        try:
            ic = float(row.get("mean_rank_ic", 0.0))
        except (TypeError, ValueError):
            continue
        if ic < 0:
            continue
        state = flip_state.get(name) or {}
        if not (row.get("ic_sign_flipped") or state.get("ic_sign_flipped")):
            continue
        entry = cat_by.get(name, {})
        dsl = (state.get("dsl") or entry.get("dsl") or "").strip()
        if dsl and dsl != "(python only)" and not dsl.startswith("-"):
            state = dict(state)
            state["dsl"] = _negate_formula(dsl)
            state["ic_sign_flipped"] = True
            flip_state[name] = state
            entry["dsl"] = state["dsl"]
            entry["lqtp_formula"] = state["dsl"]
            ensured += 1

    progress["flip_state"] = flip_state
    progress["index_rows"] = list(rows.values())
    progress["sign_reconcile_at"] = __import__("datetime").datetime.now().isoformat()
    progress["sign_reconcile_changed"] = changed

    # Drop completed so eval can refresh reports for changed factors if requested later.
    completed = [n for n in progress.get("completed", []) if n not in set(changed)]
    progress["completed"] = completed

    catalog_path.write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")
    if parsed_path.exists():
        parsed = json.loads(parsed_path.read_text(encoding="utf-8"))
        for rec in parsed:
            name = rec.get("function_name")
            if name in python_map:
                rec["python_code"] = python_map[name]
        parsed_path.write_text(json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8")
    progress_path.write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8")

    # Also write changed list for re-eval
    out_list = work / "sign_reconcile_changed.txt"
    out_list.write_text("\n".join(changed) + ("\n" if changed else ""), encoding="utf-8")
    print(f"changed={len(changed)} ensured_minus={ensured} → {progress_path}")
    print(f"reeval list → {out_list}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
