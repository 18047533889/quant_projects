# -*- coding: utf-8 -*-
"""Rewrite known logical derived-field identifiers without guessing definitions."""
from __future__ import annotations

import ast

KNOWN_DERIVED_FIELDS = frozenset({"enterprise_value", "ebitda_approx"})


class _DerivedNameRewriter(ast.NodeTransformer):
    def visit_Name(self, node: ast.Name):
        if node.id not in KNOWN_DERIVED_FIELDS:
            return node
        return ast.copy_location(
            ast.Call(
                func=ast.Name(id="source_col", ctx=ast.Load()),
                args=[ast.Constant("DerivedField"), ast.Constant(node.id)],
                keywords=[],
            ),
            node,
        )


def normalize_lqtp_derived_fields(text: str) -> str:
    tree = ast.parse(str(text), mode="eval")
    tree = _DerivedNameRewriter().visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)
