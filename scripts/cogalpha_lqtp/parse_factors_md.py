#!/usr/bin/env python3
"""Parse CogAlpha factor catalog markdown into structured records."""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class FactorRecord:
    factor_id: str
    title: str
    formula_text: str
    rationale: str
    tools: str
    python_code: str
    function_name: str


_SECTION_RE = re.compile(r"^## Factor (\d+)\s*$", re.MULTILINE)
_FORMULA_RE = re.compile(r"\*\*Formula:\*\*\s*(.+)")
_RATIONALE_RE = re.compile(r"\*\*Rationale:\*\*\s*(.+)")
_TOOLS_RE = re.compile(r"\*\*Tools:\*\*\s*(.+)")
_CODE_RE = re.compile(r"```python\n(.*?)```", re.DOTALL)
_DEF_RE = re.compile(r"def\s+(\w+)\s*\(")


def parse_factors_md(text: str) -> list[FactorRecord]:
    parts = _SECTION_RE.split(text)
    if len(parts) < 3:
        return []

    records: list[FactorRecord] = []
    for idx in range(1, len(parts), 2):
        factor_num = parts[idx].strip()
        body = parts[idx + 1]
        code_match = _CODE_RE.search(body)
        if not code_match:
            continue
        code = code_match.group(1).strip()
        def_match = _DEF_RE.search(code)
        function_name = def_match.group(1) if def_match else f"factor_{factor_num}"

        formula_match = _FORMULA_RE.search(body)
        rationale_match = _RATIONALE_RE.search(body)
        tools_match = _TOOLS_RE.search(body)

        records.append(
            FactorRecord(
                factor_id=f"factor_{factor_num}",
                title=function_name,
                formula_text=formula_match.group(1).strip() if formula_match else "",
                rationale=rationale_match.group(1).strip() if rationale_match else "",
                tools=tools_match.group(1).strip() if tools_match else "",
                python_code=code,
                function_name=function_name,
            )
        )
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description="Parse CogAlpha factors markdown")
    parser.add_argument("markdown", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    text = args.markdown.read_text(encoding="utf-8")
    records = parse_factors_md(text)
    payload = [asdict(r) for r in records]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"parsed {len(records)} factors -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
