from __future__ import annotations

import ast
from pathlib import Path


MODULE = (
    Path(__file__).resolve().parents[1]
    / "cleaned_operators"
    / "polars_native"
    / "ts_advanced_batch5.py"
)


def test_explicit_batch5_placeholders_are_research_only() -> None:
    source = MODULE.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(MODULE))
    violations: list[str] = []

    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        class_source = ast.get_source_segment(source, node) or ""
        if "Placeholder:" not in class_source and "TODO:" not in class_source:
            continue

        registration = next(
            (
                decorator
                for decorator in node.decorator_list
                if isinstance(decorator, ast.Call)
                and isinstance(decorator.func, ast.Name)
                and decorator.func.id == "register_operator"
            ),
            None,
        )
        status = None
        if registration is not None:
            status_keyword = next(
                (keyword for keyword in registration.keywords if keyword.arg == "status"),
                None,
            )
            if status_keyword is not None and isinstance(status_keyword.value, ast.Constant):
                status = status_keyword.value.value

        if status != "research_only":
            violations.append(f"{node.name}:{node.lineno}")

    assert not violations, f"production-visible placeholders: {violations}"
