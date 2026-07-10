#!/usr/bin/env python3
"""从 OperatorRegistry + OperatorPolicy 自动生成 operators_catalog.md。"""
from __future__ import annotations

from datetime import date
from pathlib import Path

FE_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = FE_ROOT / "cleaned_operators" / "docs" / "operators_catalog.md"


def _load():
    """加载算子注册表、policy 推断器并注册 SQL backend。"""
    import sys

    root = str(FE_ROOT.parent)
    if root not in sys.path:
        sys.path.insert(0, root)
    if str(FE_ROOT) not in sys.path:
        sys.path.insert(0, str(FE_ROOT))
    from cleaned_operators import load_all
    from cleaned_operators.operator_policy import infer_operator_policy
    from cleaned_operators.registry import OperatorRegistry
    from backend.sql_pushdown.sql_registry import register_sql_backends

    load_all()
    register_sql_backends()
    return OperatorRegistry, infer_operator_policy


def _render(registry, infer_policy) -> str:
    """将注册表与 policy 渲染为 operators_catalog.md 正文。"""
    catalog = registry._catalog
    lines = [
        "# Operators Catalog（自动生成）",
        "",
        f"> 生成日期：{date.today().isoformat()}",
        "> 由 `scripts/generate_operators_catalog.py` 从注册表 + OperatorPolicy 生成。",
        "",
        "## 摘要",
        "",
        f"- canonical 总数：{len(catalog)}",
        "",
        "## Implemented 算子",
        "",
        "| canonical | backends | pit_safe | scope | lookback | min_periods | lag |",
        "|-----------|----------|----------|-------|----------|-------------|-----|",
    ]

    implemented = sorted(
        name for name, meta in catalog.items() if meta.get("status", "implemented") != "stub"
    )
    for canon in implemented:
        meta = catalog[canon]
        backends = ", ".join(meta.get("backends") or [])
        op = registry.get(canon)
        policy = infer_policy(op, canonical=canon) if op else None
        if policy:
            lines.append(
                f"| {canon} | {backends} | {policy.pit_safe} | {policy.scope} | "
                f"{policy.lookback_window} | {policy.min_periods} | {policy.lag} |"
            )
        else:
            lines.append(f"| {canon} | {backends} | — | — | — | — | — |")

    lines.extend(["", "## Stub / catalog-only", ""])
    stubs = sorted(name for name, meta in catalog.items() if meta.get("status") == "stub")
    for canon in stubs:
        lines.append(f"- `{canon}`")

    return "\n".join(lines) + "\n"


def main() -> None:
    """从 OperatorRegistry + OperatorPolicy 生成 operators_catalog.md。"""
    registry, infer_policy = _load()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(_render(registry, infer_policy), encoding="utf-8")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
