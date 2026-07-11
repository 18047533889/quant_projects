"""独立包内 DSL 校验：优先 factor_engine.parse_expr，否则 AST + 本地白名单。"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

from lib.paths import PACKAGE_ROOT, resolve_factor_engine_root


class DSLParseError(ValueError):
    pass


def load_allowlist() -> set[str]:
    snap = PACKAGE_ROOT / "dsl" / "fe_dsl_allowlist.json"
    if snap.exists():
        raw = json.loads(snap.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            return set(raw)
        if isinstance(raw, dict):
            ops = raw.get("operators", [])
            return set(ops)
    raw = PACKAGE_ROOT / "dsl" / "dsl_allowlist.json"
    data = json.loads(raw.read_text(encoding="utf-8"))
    return {op.lower() for op in data.get("operators", [])}


ALLOW = load_allowlist()
FIELD_NAMES = {
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "ret",
    "pre_close",
    "preclose",
    "vwap",
    "index_close",
    "index_open",
}


def _parse_with_factor_engine(formula: str) -> None:
    fe_root = resolve_factor_engine_root()
    if fe_root is None:
        raise DSLParseError("factor_engine not available")
    if str(fe_root) not in sys.path:
        sys.path.insert(0, str(fe_root))
    from api.dsl_parser import DSLParseError as FEError  # noqa: WPS433
    from api.dsl_parser import parse_expr  # noqa: WPS433

    try:
        parse_expr(formula)
    except FEError as exc:
        raise DSLParseError(str(exc)) from exc


class _Validator(ast.NodeVisitor):
    def __init__(self) -> None:
        self.errors: list[str] = []

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name):
            op = node.func.id
            if op not in ALLOW:
                self.errors.append(f"unknown_op: {op}")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id not in ALLOW and node.id not in FIELD_NAMES:
            self.errors.append(f"unknown_name: {node.id}")
        self.generic_visit(node)


def parse_expr(formula: str) -> None:
    """校验公式；成功无返回，失败抛 DSLParseError。"""
    if formula.count("(") != formula.count(")"):
        raise DSLParseError("parenthesis mismatch")

    try:
        _parse_with_factor_engine(formula)
        return
    except DSLParseError:
        pass

    try:
        tree = ast.parse(formula, mode="eval")
    except SyntaxError as exc:
        raise DSLParseError(f"Invalid expression syntax: {formula}") from exc

    v = _Validator()
    v.visit(tree)
    if v.errors:
        raise DSLParseError("; ".join(v.errors[:3]))


def validate_formula(formula: str) -> tuple[bool, str]:
    try:
        parse_expr(formula)
        return True, "OK"
    except DSLParseError as exc:
        return False, str(exc)
