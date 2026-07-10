# -*- coding: utf-8
"""第一阶段 production 范围冻结（~90 primitive + 16 composite）。"""
from __future__ import annotations

import json
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any

_SCOPE_YAML = Path(__file__).resolve().parents[1] / "evidence" / "phase1_production_scope.yaml"


class Phase1ScopeError(ValueError):
    """违反第一阶段冻结范围或证据规则。"""


@lru_cache(maxsize=1)
def _load_scope() -> dict[str, Any]:
    path = _SCOPE_YAML
    if not path.is_file():
        raise FileNotFoundError(
            f"缺少第一阶段冻结清单: {path}；运行 scripts/export_phase1_scope.py 生成"
        )
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("phase1 scope 需要 PyYAML：pip install pyyaml") from exc
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def scope_frozen_commit_sha() -> str:
    return str(_load_scope().get("frozen_commit_sha") or "")


def phase1_primitives() -> frozenset[str]:
    ops = _load_scope().get("primitives") or {}
    return frozenset(str(k) for k in ops.keys())


def phase1_composites() -> frozenset[str]:
    return frozenset(str(x) for x in (_load_scope().get("composites") or []))


def phase1_all_operators() -> frozenset[str]:
    return phase1_primitives() | phase1_composites()


def get_phase1_entry(canon: str) -> dict[str, Any] | None:
    return (_load_scope().get("primitives") or {}).get(canon)


def phase1_status(canon: str) -> str:
    entry = get_phase1_entry(canon)
    if entry:
        return str(entry.get("phase1_status") or "structural_candidate")
    if canon in phase1_composites():
        return str((_load_scope().get("composite_defaults") or {}).get("phase1_status") or "composite_pending")
    return "out_of_scope"


def is_phase1_in_scope(canon: str) -> bool:
    return canon in phase1_all_operators()


def phase1_production_certified(canon: str) -> bool:
    """是否已通过 dual-backend / composite 完整证据认证（可 production fastpath）。"""
    from backend.composite_evidence import composite_production_safe
    from backend.primitive_evidence import primitive_dual_backend_production_safe

    if canon in phase1_composites():
        return composite_production_safe(canon)
    return primitive_dual_backend_production_safe(canon)


def assert_phase1_freeze(canon: str) -> None:
    """新增 production 候选必须在第一阶段冻结清单内。"""
    if not is_phase1_in_scope(canon):
        raise Phase1ScopeError(
            f"{canon}: 不在第一阶段冻结范围；"
            f"须先更新 evidence/phase1_production_scope.yaml 并走 certify 流程"
        )


def assert_production_requires_certification(canon: str) -> None:
    """production 只允许 evidence 已认证算子（冻结后不得绕过）。"""
    assert_phase1_freeze(canon)
    if not phase1_production_certified(canon):
        raise Phase1ScopeError(
            f"{canon}: 第一阶段已冻结但未完成证据认证；"
            f"运行 python scripts/certify_operator.py {canon}"
        )


def phase1_summary() -> dict[str, Any]:
    from backend.composite_evidence import COMPOSITE_FULL_PARITY_VERIFIED
    from backend.primitive_evidence import PRIMITIVE_DUAL_BACKEND_PRODUCTION_SAFE

    prim = phase1_primitives()
    comp = phase1_composites()
    return {
        "frozen_commit_sha": scope_frozen_commit_sha(),
        "primitive_count": len(prim),
        "composite_count": len(comp),
        "total_scope": len(prim) + len(comp),
        "primitive_certified_dual": len(PRIMITIVE_DUAL_BACKEND_PRODUCTION_SAFE & prim),
        "composite_certified": len(COMPOSITE_FULL_PARITY_VERIFIED & comp),
        "certified_total": len(
            (PRIMITIVE_DUAL_BACKEND_PRODUCTION_SAFE & prim) | (COMPOSITE_FULL_PARITY_VERIFIED & comp)
        ),
    }


def current_git_sha() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=str(_SCOPE_YAML.parent.parent),
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return ""
