#!/usr/bin/env python3
"""审计 GTJA-191 DSL：算子白名单、括号平衡、EMA 参数 arity。"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from lib.dsl_validate import ALLOW, DSLParseError, parse_expr  # noqa: E402

GTJA_ROOT = PACKAGE_ROOT
PV_FIELDS = {"open", "high", "low", "close", "volume", "amount", "ret", "pre_close", "vwap"}
EXTRA_FIELDS = {"index_close", "index_open"}
KNOWN_NAMES = PV_FIELDS | EXTRA_FIELDS | ALLOW


def _is_int_const(node: ast.AST) -> int | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value
    return None


def _audit_formula(name: str, formula: str) -> list[str]:
    issues: list[str] = []
    if formula.count("(") != formula.count(")"):
        issues.append("paren_mismatch")

    try:
        parse_expr(formula)
    except DSLParseError as exc:
        issues.append(f"parse_fail: {exc}")

    try:
        tree = ast.parse(formula, mode="eval")
    except SyntaxError as exc:
        issues.append(f"ast_fail: {exc}")
        return issues

    class V(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call) -> None:
            if isinstance(node.func, ast.Name):
                op = node.func.id
                if op not in ALLOW:
                    issues.append(f"unknown_op: {op}")
                if op in ("EMA", "SMA", "WMA") and len(node.args) != 2:
                    issues.append(f"bad_arity: {op} expects 2 args, got {len(node.args)}")
                if op in ("max", "min") and len(node.args) == 2:
                    n = _is_int_const(node.args[1])
                    a0 = node.args[0]
                    if (
                        n is not None
                        and n >= 2
                        and not (isinstance(a0, ast.Constant) and a0.value == 0)
                    ):
                        issues.append(f"rolling_{op}_not_ts: use ts_{op}(..., {n})")
            self.generic_visit(node)

        def visit_Name(self, node: ast.Name) -> None:
            if node.id not in KNOWN_NAMES:
                issues.append(f"unknown_field_or_name: {node.id}")
            elif node.id in EXTRA_FIELDS:
                issues.append("needs_benchmark_field")
            self.generic_visit(node)

    V().visit(tree)
    return issues


def main() -> int:
    catalog_path = GTJA_ROOT / "dsl" / "gtja191_dsl_catalog.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    report: dict[str, list[str]] = {}
    for name, item in sorted(catalog.items()):
        issues = _audit_formula(name, item["dsl_formula"])
        if issues:
            report[name] = issues

    out = GTJA_ROOT / "dsl" / "audit_report.json"
    blocking = {
        k: [x for x in v if x != "needs_benchmark_field"]
        for k, v in report.items()
    }
    blocking = {k: v for k, v in blocking.items() if v}
    summary = {
        "total": len(catalog),
        "clean": len(catalog) - len(blocking),
        "blocking_issues": blocking,
        "benchmark_pending": [k for k, v in report.items() if "needs_benchmark_field" in v],
        "note": "独立包内审计；算子白名单见 dsl/fe_dsl_allowlist.json",
    }
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "total": summary["total"],
                "clean": summary["clean"],
                "blocking": len(blocking),
                "benchmark_pending": len(summary["benchmark_pending"]),
            },
            ensure_ascii=False,
        )
    )
    if blocking:
        for k, v in list(blocking.items())[:10]:
            print(k, v)
    return 0 if not blocking else 1


if __name__ == "__main__":
    raise SystemExit(main())
