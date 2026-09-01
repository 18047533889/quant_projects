"""factor_engine DSL 校验与算子名提取。"""

from __future__ import annotations

import re

_DSL_CALL_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(")


def extract_dsl_operator_names(dsl: str) -> tuple[str, ...]:
    from cold_start_library.runtime.paths import ensure_factor_engine_importable
    from api.operator_registry import build_dsl_allowlist

    ensure_factor_engine_importable()
    allow = set(build_dsl_allowlist().keys())
    found: list[str] = []
    for match in _DSL_CALL_RE.finditer(str(dsl)):
        name = match.group(1)
        if name in allow and name not in found:
            found.append(name)
    return tuple(found)


def validate_factor_engine_dsl(dsl: str) -> tuple[bool, str]:
    from cold_start_library.runtime.paths import ensure_factor_engine_importable

    ensure_factor_engine_importable()
    from api.dsl_parser import DSLParseError, parse_expr

    text = str(dsl).strip()
    if not text:
        return False, "empty DSL"
    try:
        parse_expr(text, surface="compat")
        return True, "OK"
    except DSLParseError as exc:
        return False, str(exc)
    except Exception as exc:  # pragma: no cover
        return False, f"{type(exc).__name__}: {exc}"
