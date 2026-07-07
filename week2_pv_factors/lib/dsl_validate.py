"""Week2 因子包 DSL 校验：优先接 factor_engine.parse_expr。"""
from __future__ import annotations

import sys

from lib.paths import resolve_factor_engine_root


class DSLParseError(ValueError):
    pass


def parse_expr(formula: str) -> None:
    fe_root = resolve_factor_engine_root()
    if fe_root is None:
        raise DSLParseError("factor_engine not available; set FACTOR_ENGINE_ROOT")
    if str(fe_root) not in sys.path:
        sys.path.insert(0, str(fe_root))
    from api.dsl_parser import DSLParseError as FEError  # noqa: WPS433
    from api.dsl_parser import parse_expr as fe_parse  # noqa: WPS433

    try:
        fe_parse(formula)
    except FEError as exc:
        raise DSLParseError(str(exc)) from exc


def validate_formula(formula: str) -> tuple[bool, str]:
    try:
        parse_expr(formula)
        return True, ""
    except DSLParseError as exc:
        return False, str(exc)
