"""Opt-in migrations for reviewed R12e formulas using obsolete parameters.

The recipes are deliberately exact and fail closed.  Tail-probability rewrites
and estimator-resolution rewrites are marked as semantic redesigns and choose
an existing, documented contract value.
"""
from __future__ import annotations

import ast
import re

from factor_engine.tools.catalog_recipe_migration import RecipeMigration

_MAX_FORMULA_BYTES = 65_536


def _offset(lines: list[str], lineno: int, byte_col: int) -> int:
    prefix = lines[lineno - 1].encode("utf-8")[:byte_col]
    return sum(len(line) for line in lines[: lineno - 1]) + len(prefix.decode("utf-8"))


def _span(lines: list[str], node: ast.AST) -> tuple[int, int]:
    if node.end_lineno is None or node.end_col_offset is None:
        raise ValueError("AST node has no complete source location")
    return (
        _offset(lines, node.lineno, node.col_offset),
        _offset(lines, node.end_lineno, node.end_col_offset),
    )


def _literal(node: ast.AST, value: object) -> bool:
    return isinstance(node, ast.Constant) and type(node.value) is type(value) and node.value == value


def _keywords(node: ast.Call) -> dict[str, ast.keyword] | None:
    if any(keyword.arg is None for keyword in node.keywords):
        return None
    result = {keyword.arg: keyword for keyword in node.keywords}
    return result if len(result) == len(node.keywords) else None


def _keyword_literal(keywords: dict[str, ast.keyword], name: str, value: object) -> bool:
    keyword = keywords.get(name)
    return keyword is not None and _literal(keyword.value, value)


def _has_logic(logic: str, *needles: str) -> bool:
    normalized = logic.casefold()
    return any(needle.casefold() in normalized for needle in needles)


def _explicit_gaussian_method(logic: str) -> str | None:
    """Return the one canonical method explicitly named by the review text."""
    normalized = logic.casefold()
    methods = {
        method
        for method in ("blom", "van_der_waerden")
        if re.search(rf"(?<![a-z0-9_]){re.escape(method)}(?![a-z0-9_])", normalized)
    }
    if len(methods) != 1:
        return None
    if re.search(
        r"(?:do\s+not\s+use|not|不是|不要使用).{0,32}"
        r"(?:blom|van_der_waerden)(?![a-z0-9_])",
        normalized,
    ):
        return None
    declared = re.search(
        r"canonical[\s_-]+method\s*(?:is|=|:)\s*['\"]?"
        r"(blom|van_der_waerden)(?![a-z0-9_])",
        normalized,
    )
    return declared.group(1) if declared is not None else None


def _replace_keyword(
    edits: list[tuple[int, int, str]], lines: list[str], keywords: dict[str, ast.keyword],
    name: str, old: object, replacement: str,
) -> bool:
    keyword = keywords.get(name)
    if keyword is None or not _literal(keyword.value, old):
        return False
    start, end = _span(lines, keyword.value)
    edits.append((start, end, replacement))
    return True


def migrate_catalog_r13_parameter_formula(
    formula: str, *, logic: str = "", enabled: bool = False
) -> RecipeMigration:
    """Migrate only exact R12e legacy parameter shapes after explicit opt-in."""
    if not isinstance(formula, str):
        raise TypeError("formula must be a string")
    if not isinstance(logic, str):
        raise TypeError("logic must be a string")
    if len(formula.encode("utf-8")) > _MAX_FORMULA_BYTES:
        raise ValueError("formula exceeds migration input budget")
    if not enabled or not logic.strip():
        return RecipeMigration(formula, ())

    tree = ast.parse(formula, mode="eval")
    lines = formula.splitlines(keepends=True) or [""]
    edits: list[tuple[int, int, str]] = []
    changes: list[str] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        name = node.func.id
        keywords = _keywords(node)
        if keywords is None:
            continue

        gaussian_method = _explicit_gaussian_method(logic)
        if (
            name == "cs_rank_gaussian" and len(node.args) == 2 and not node.keywords
            and _literal(node.args[1], 3.0)
            and gaussian_method is not None
        ):
            start, end = _span(lines, node.args[1])
            edits.append((start, end, repr(gaussian_method)))
            changes.append(
                "SEMANTIC_REDESIGN cs_rank_gaussian: numeric legacy method=3.0 -> "
                f"explicitly reviewed canonical method={gaussian_method!r}; "
                "3.0 was never a supported method"
            )
            continue

        if name in {"cs_rank_copula_entropy", "cs_rank_copula_mi"}:
            if _has_logic(logic, "copula") and _replace_keyword(edits, lines, keywords, "grid", 10, "8"):
                changes.append(
                    f"SEMANTIC_REDESIGN {name}: legacy grid=10 -> canonical default grid=8; "
                    "grid=10 is not a verified estimator resolution"
                )
            continue

        if name == "ts_conditional_transfer_entropy":
            exact = (
                _keyword_literal(keywords, "window", 120)
                and _keyword_literal(keywords, "lag", 1)
                and _has_logic(logic, "transfer entropy", "条件传递熵", "方向性信息流")
            )
            if exact and _replace_keyword(edits, lines, keywords, "bins", 5, "2"):
                changes.append(
                    "SEMANTIC_REDESIGN ts_conditional_transfer_entropy: bins=5 -> bins=2; "
                    "the canonical default is the only verified grid feasible at window=120"
                )
            continue

        resolution_rules = {
            "ts_markov_entropy_production": (6, 5, 120),
            "ts_km_diffusion_gradient": (6, 5, 80),
        }
        if name in resolution_rules:
            old, new, window = resolution_rules[name]
            logic_matches = (
                _has_logic(logic, "entropy production", "熵产生")
                if name == "ts_markov_entropy_production"
                else _has_logic(logic, "diffusion gradient", "扩散梯度")
            )
            exact = (
                _keyword_literal(keywords, "window", window)
                and _keyword_literal(keywords, "lag", 1)
                and logic_matches
            )
            if exact and _replace_keyword(edits, lines, keywords, "bins", old, str(new)):
                changes.append(
                    f"SEMANTIC_REDESIGN {name}: bins={old} -> nearest lower verified "
                    f"estimator resolution bins={new}"
                )
            continue

        if name == "ts_active_information_storage":
            exact = (
                _keyword_literal(keywords, "window", 120)
                and _keyword_literal(keywords, "history_length", 2)
                and _has_logic(logic, "active information storage", "活性信息存储", "自身可预测信息")
            )
            if exact and _replace_keyword(edits, lines, keywords, "bins", 6, "2"):
                changes.append(
                    "SEMANTIC_REDESIGN ts_active_information_storage: bins=6 -> bins=2; "
                    "bins=2 is the only verified grid feasible for window=120/history_length=2"
                )
            continue

        if name == "ts_conditional_mutual_information":
            if (
                _keyword_literal(keywords, "window", 60)
                and _has_logic(logic, "conditional mutual information", "条件互信息")
                and _replace_keyword(edits, lines, keywords, "bins", 4, "3")
            ):
                changes.append(
                    "SEMANTIC_REDESIGN ts_conditional_mutual_information: bins=4 -> "
                    "nearest lower verified estimator resolution bins=3"
                )
            continue

        single_tail = {
            "ts_quantile_crossing_spectral_concentration",
            "ts_extremogram",
            "ts_extremal_dependence_decay",
        }
        if name in single_tail:
            if (
                keywords.get("side") is not None
                and _literal(keywords["side"].value, "upper")
                and _has_logic(logic, "upper", "上尾", "极值", "分位")
                and _replace_keyword(edits, lines, keywords, "quantile", 0.9, "0.1")
            ):
                changes.append(
                    f"SEMANTIC_REDESIGN {name}: assumed upper quantile threshold 0.9 -> "
                    "canonical upper-tail probability 0.1 (threshold remains Q_0.9)"
                )
            continue

        if name in {"ts_cross_extremogram", "ts_cross_quantilogram"}:
            for q_name, side_name in (("target_q", "target_side"), ("source_q", "source_side")):
                if (
                    keywords.get(side_name) is not None
                    and _literal(keywords[side_name].value, "upper")
                    and _has_logic(logic, "upper", "上尾", "极值", "分位")
                    and _replace_keyword(edits, lines, keywords, q_name, 0.9, "0.1")
                ):
                    changes.append(
                        f"SEMANTIC_REDESIGN {name}.{q_name}: assumed upper quantile threshold "
                        "0.9 -> canonical upper-tail probability 0.1 (threshold remains Q_0.9)"
                    )

    ordered = sorted(edits)
    if any(start < previous_end for (_, previous_end, _), (start, _, _) in zip(ordered, ordered[1:])):
        return RecipeMigration(formula, ())
    migrated = formula
    for start, end, replacement in reversed(ordered):
        migrated = migrated[:start] + replacement + migrated[end:]
    return RecipeMigration(migrated, tuple(changes))
