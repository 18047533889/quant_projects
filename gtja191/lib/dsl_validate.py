"""GTJA DSL 校验：FactorEngine 可用时必须以真实 parser 为准。"""
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
            return set(raw.get("operators", []))
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


def _parse_with_factor_engine(formula: str) -> bool:
    """真实 FactorEngine 可用时解析；不可用时返回 False。

    关键点：真实 parser 一旦加载成功，公式解析失败必须直接失败，不能再退回较宽松的
    本地 AST 检查，否则“投递校验通过、运行时失败”。
    """
    fe_root = resolve_factor_engine_root()
    if fe_root is None:
        return False
    if str(fe_root) not in sys.path:
        sys.path.insert(0, str(fe_root))
    try:
        from api.dsl_parser import DSLParseError as FEError  # noqa: WPS433
        from api.dsl_parser import parse_expr as fe_parse_expr  # noqa: WPS433
    except (ImportError, ModuleNotFoundError):
        return False

    try:
        fe_parse_expr(formula)
    except FEError as exc:
        raise DSLParseError(str(exc)) from exc
    except Exception as exc:
        raise DSLParseError(f"FactorEngine parser failed: {type(exc).__name__}: {exc}") from exc
    return True


class _Validator(ast.NodeVisitor):
    def __init__(self) -> None:
        self.errors: list[str] = []

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name):
            op = node.func.id
            if op not in ALLOW:
                self.errors.append(f"unknown_op: {op}")
        else:
            self.errors.append("only direct function calls are allowed")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        self.errors.append("attribute access is forbidden")

    def visit_Name(self, node: ast.Name) -> None:
        if node.id not in ALLOW and node.id not in FIELD_NAMES:
            self.errors.append(f"unknown_name: {node.id}")
        self.generic_visit(node)


def parse_expr(formula: str) -> None:
    """校验公式；成功无返回，失败抛 DSLParseError。"""
    if formula.count("(") != formula.count(")"):
        raise DSLParseError("parenthesis mismatch")

    if _parse_with_factor_engine(formula):
        return

    # 独立投递包没有 FactorEngine 时才使用本地白名单快照。
    try:
        tree = ast.parse(formula, mode="eval")
    except SyntaxError as exc:
        raise DSLParseError(f"Invalid expression syntax: {formula}") from exc

    validator = _Validator()
    validator.visit(tree)
    if validator.errors:
        raise DSLParseError("; ".join(validator.errors[:3]))


def validate_formula(formula: str) -> tuple[bool, str]:
    try:
        parse_expr(formula)
        return True, "OK"
    except DSLParseError as exc:
        return False, str(exc)
