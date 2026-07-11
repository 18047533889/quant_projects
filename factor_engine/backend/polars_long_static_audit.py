# -*- coding: utf-8
"""PolarsLong native 实现静态审计：禁止 rolling_map / map_groups / pandas callback 误入 NATIVE。"""
from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Iterable

from backend.polars_long_policy import POLARS_LONG_NATIVE, POLARS_LONG_PYTHON_ROLLING

_EMITTER = Path(__file__).resolve().parent / "polars_expr_emitter.py"

_FORBIDDEN_TOKENS: frozenset[str] = frozenset(
    {
        "rolling_map",
        "map_groups",
        "map_batches",
        "map_elements",
        "to_pandas",
        "pd.Series",
        "pd.DataFrame",
    }
)

_META_OPS: frozenset[str] = frozenset(
    {"column", "literal", "materialized_series", "plan_ref"}
)


def _emitter_source() -> str:
    return _EMITTER.read_text(encoding="utf-8")


def _native_op_blocks(source: str) -> dict[str, str]:
    """从 ``_compile_polars`` 提取 ``if op == "canon":`` 代码块（近似）。"""
    blocks: dict[str, str] = {}
    pattern = re.compile(
        r'^\s+if op (?:== "([^"]+)"|in \{[^}]+\}):',
        re.MULTILINE,
    )
    matches = list(pattern.finditer(source))
    for i, m in enumerate(matches):
        canon = m.group(1)
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(source)
        block = source[start:end]
        if canon:
            blocks[canon] = block
        elif 'if op in' in source[start : start + 40]:
            # ``if op in {"ewm_corr", ...}`` — 不写入单 canon 块
            pass
    return blocks


def find_forbidden_tokens_in_native_blocks(
    *,
    native_ops: Iterable[str] | None = None,
) -> dict[str, list[str]]:
    """返回 native 算子代码块中命中的 forbidden token 列表。"""
    source = _emitter_source()
    blocks = _native_op_blocks(source)
    targets = frozenset(native_ops or (POLARS_LONG_NATIVE - _META_OPS))
    violations: dict[str, list[str]] = {}
    for canon in sorted(targets):
        block = blocks.get(canon)
        if block is None:
            continue
        hits = sorted(t for t in _FORBIDDEN_TOKENS if t in block)
        if hits:
            violations[canon] = hits
    return violations


def native_tier_consistency_errors() -> list[str]:
    """tier 集合互斥与 python_rolling 不得标为 native。"""
    errors: list[str] = []
    overlap = POLARS_LONG_NATIVE & POLARS_LONG_PYTHON_ROLLING
    if overlap:
        errors.append(f"POLARS_LONG_NATIVE ∩ PYTHON_ROLLING 非空: {sorted(overlap)}")
    return errors


def audit_polars_long_native(*, native_ops: Iterable[str] | None = None) -> list[str]:
    """运行全部 native 静态审计，返回错误消息列表（空 = 通过）。"""
    errors = native_tier_consistency_errors()
    violations = find_forbidden_tokens_in_native_blocks(native_ops=native_ops)
    for canon, hits in sorted(violations.items()):
        errors.append(f"{canon} native 块含 forbidden: {', '.join(hits)}")
    return errors


def assert_polars_long_native_clean() -> None:
    errs = audit_polars_long_native()
    if errs:
        raise AssertionError("PolarsLong native static audit failed:\n" + "\n".join(errs))
