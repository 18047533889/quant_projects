#!/usr/bin/env python3
"""生成 docs/operator_market_capabilities.json + docs/operator_market_matrix.md。

对 OperatorRegistry 中全部 canonical 自动生成 A/US 市场能力结论（spec §118）：
generic math/TS/CS → both；市场机制算子 → 契约判定；全部 canonical 必须覆盖，
``UNKNOWN / NOT_REVIEWED`` 计数必须为 0（CI 门槛，spec §53）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]  # factor_engine
OUT_JSON = ROOT / "docs" / "operator_market_capabilities.json"
OUT_MD = ROOT / "docs" / "operator_market_matrix.md"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def build_operator_manifest() -> dict[str, Any]:
    from cleaned_operators import OperatorRegistry, load_all

    load_all()
    from market.capability_resolver import build_market_operator_manifest

    return build_market_operator_manifest(OperatorRegistry.list_canonical())


def _md_cell(status: str) -> str:
    return {"CERTIFIED_NATIVE": "✓", "CERTIFIED_DERIVED": "✓*"}.get(status, status)


def render_matrix(doc: dict[str, Any]) -> str:
    rows = doc["operators"]
    lines = [
        "# Operator × Market 能力矩阵",
        "",
        f"覆盖 canonical：**{doc['counts']['total']}** ｜ UNKNOWN：**{doc['counts']['unknown']}** ｜ "
        "NOT_REVIEWED：**" + str(doc["counts"]["not_reviewed"]) + "**",
        "",
        "| canonical | A | US | 说明 |",
        "| --- | --- | --- | --- |",
    ]
    for name in sorted(rows):
        row = rows[name]
        a = row["ashare"]["status"]
        u = row["us"]["status"]
        note = ""
        if "UNSUPPORTED" in (a, u) or "PROVIDER_REQUIRED" in (a, u):
            if u == "UNSUPPORTED_MARKET_MECHANISM":
                note = "market mechanism (A-share price limits)"
            elif u == "PROVIDER_REQUIRED":
                note = "US provider required: " + ", ".join(row["required_capabilities"])
            elif a == "UNSUPPORTED_MARKET_MECHANISM":
                note = "US-only mechanism"
            elif a == "PROVIDER_REQUIRED":
                note = "A-share provider required: " + ", ".join(row["required_capabilities"])
        lines.append(
            f"| {name} | {_md_cell(a)} | {_md_cell(u)} | {note} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    doc = build_operator_manifest()
    OUT_JSON.write_text(
        json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    OUT_MD.write_text(render_matrix(doc), encoding="utf-8")
    counts = doc["counts"]
    print(
        f"Wrote {OUT_JSON} and {OUT_MD} "
        f"({counts['total']} canonicals, unknown={counts['unknown']}, "
        f"not_reviewed={counts['not_reviewed']})"
    )
    if counts["unknown"] or counts["not_reviewed"]:
        print("CI FAIL: unknown/not_reviewed must be 0", file=sys.stderr)
        return 1
    print("CI OK: all canonicals have explicit A/US market status")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
