#!/usr/bin/env python3
"""Fix per-symbol full-history .rank() look-ahead in Python factor code."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DEFAULT_WINDOW = 252
DEFAULT_MIN_PERIODS = 20


def is_safe_before_rank(before: str) -> bool:
    tail = before[-300:].replace("\n", " ")
    return bool(
        re.search(r"\.rolling\s*\([^)]*\)\s*$", tail)
        or re.search(r"\.expanding\s*\([^)]*\)\s*$", tail)
    )


def has_unsafe_rank(code: str) -> bool:
    for m in re.finditer(r"\.rank\s*\(", code):
        if not is_safe_before_rank(code[: m.start()]):
            return True
    return False


def fix_rank_lookahead(
    code: str,
    *,
    window: int = DEFAULT_WINDOW,
    min_periods: int = DEFAULT_MIN_PERIODS,
) -> str:
    if not has_unsafe_rank(code):
        return code
    out: list[str] = []
    i = 0
    for m in re.finditer(r"\.rank\s*\(", code):
        if is_safe_before_rank(code[: m.start()]):
            continue
        out.append(code[i : m.start()])
        out.append(f".rolling({window}, min_periods={min_periods})")
        i = m.start()
    out.append(code[i:])
    return "".join(out)


def fix_parsed_factors(path: Path) -> list[str]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    changed: list[str] = []
    for row in rows:
        code = row.get("python_code", "")
        if not code or not has_unsafe_rank(code):
            continue
        fixed = fix_rank_lookahead(code)
        if fixed != code:
            row["python_code"] = fixed
            changed.append(row["function_name"])
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return sorted(set(changed))


def fix_factors_md(path: Path) -> int:
    text = path.read_text(encoding="utf-8")
    n = 0

    def repl(block: re.Match[str]) -> str:
        nonlocal n
        code = block.group(1)
        if not has_unsafe_rank(code):
            return block.group(0)
        fixed = fix_rank_lookahead(code)
        if fixed == code:
            return block.group(0)
        n += 1
        return f"```python\n{fixed}```"

    new_text = re.sub(r"```python\n(.*?)```", repl, text, flags=re.DOTALL)
    if n:
        path.write_text(new_text, encoding="utf-8")
    return n


def invalidate_lake(work_dir: Path, names: list[str]) -> None:
    lake = work_dir / "factor_lake"
    progress_path = work_dir / "production_progress.json"
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        completed = [x for x in progress.get("completed", []) if x not in names]
        progress["completed"] = completed
        progress["index_rows"] = [
            r for r in progress.get("index_rows", []) if r.get("factor_name") not in names
        ]
        for name in names:
            progress.get("failed", {}).pop(name, None)
        progress_path.write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8")
    for name in names:
        parquet = lake / name / "values.parquet"
        if parquet.exists():
            parquet.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description="Fix full-history .rank() look-ahead in factor Python code")
    parser.add_argument("--factors-md", type=Path, default=ROOT / "factors(1).md")
    parser.add_argument("--parsed", type=Path, default=ROOT / "data/cogalpha_lqtp_production/parsed_factors.json")
    parser.add_argument("--work-dir", type=Path, default=ROOT / "data/cogalpha_lqtp_production")
    parser.add_argument("--reparse", action="store_true", default=True)
    parser.add_argument("--invalidate-lake", action="store_true", default=True)
    args = parser.parse_args()

    md_blocks = fix_factors_md(args.factors_md)
    changed = fix_parsed_factors(args.parsed)
    print(f"fixed factors: {len(changed)} (md blocks touched: {md_blocks})")
    for name in changed:
        print(f"  - {name}")

    if args.reparse and changed:
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/cogalpha_lqtp/parse_factors_md.py"),
                str(args.factors_md),
                "--out",
                str(args.parsed),
            ],
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/cogalpha_lqtp/python_to_dsl.py"),
                str(args.parsed),
                "--out",
                str(args.work_dir / "dsl_catalog.json"),
            ],
            check=True,
        )
        print("re-parsed catalog + dsl_catalog.json")

    if args.invalidate_lake and changed:
        invalidate_lake(args.work_dir, changed)
        print(f"invalidated factor_lake parquet for {len(changed)} factors (removed from completed)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
