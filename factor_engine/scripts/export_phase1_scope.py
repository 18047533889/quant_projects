#!/usr/bin/env python3
"""导出 / 校验第一阶段 production 范围冻结清单（~90 primitive + 16 composite）。"""
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

FE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = FE_ROOT / "evidence" / "phase1_production_scope.yaml"


def _bootstrap() -> None:
    root = str(FE_ROOT.parent)
    fe = str(FE_ROOT)
    for p in (root, fe):
        if p not in sys.path:
            sys.path.insert(0, p)
    from cleaned_operators import load_all
    from backend.sql_pushdown.sql_registry import register_sql_backends

    load_all()
    register_sql_backends()


def _git_sha() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=str(FE_ROOT),
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return ""


def _default_semantics(canon: str, policy_row: dict) -> dict:
    from backend.numeric_semantics import rank_tie_method, semantics_for

    scope = str(policy_row.get("scope") or "elementwise")
    sem = semantics_for(canon)
    return {
        "scope": scope,
        "null_policy": "propagate",
        "nan_policy": "propagate",
        "inf_policy": "to_nan" if sem.output_inf_to_nan else "propagate",
        "min_periods": policy_row.get("min_periods", 1 if scope == "ts" else None),
        "ddof": 1 if sem.std_ddof == "sample" and canon in {"ts_std", "ts_var", "group_std"} else None,
        "tie_policy": rank_tie_method(canon) if "rank" in canon or canon in {"ts_rank", "ts_argmax", "ts_argmin"} else None,
        "zero_denominator": sem.div_zero if "div" in canon or canon.endswith("_beta") else None,
        "return_type": "float64",
    }


def build_scope_document() -> dict:
    from backend.composite_evidence import composite_production_safe
    from backend.fastpath_coverage import build_fastpath_coverage_row
    from backend.operator_capability import polars_long_tier
    from backend.phase1_scope import current_git_sha
    from backend.polars_long_production import polars_long_production_tier
    from backend.primitive_evidence import primitive_dual_backend_production_safe
    from backend.production_fastpath_tiers import (
        FASTPATH_DEFERRED_CANONICALS,
        FORBIDDEN_PRODUCTION_FASTPATH,
        P0_PRODUCTION_FASTPATH_CANONICALS,
        P1_POLARS_CORE_PRODUCTION_SAFE,
        dual_backend_structural_candidates,
    )
    from backend.sql_tiers import effective_sql_production_safe, is_sql_implemented
    from cleaned_operators.operator_policy import infer_operator_policy
    from cleaned_operators.operator_spec import (
        PRODUCTION_ALLOWED_DEFERRED_CANONICALS,
        PRODUCTION_DUAL_BACKEND_CORE_CANONICALS,
        infer_production_policy,
    )
    from planner.composite_lowering import list_composite_lowerings

    primitives: dict = {}
    batch2 = frozenset({"ts_argmax", "ts_argmin"})
    core = sorted(
        PRODUCTION_DUAL_BACKEND_CORE_CANONICALS
        | PRODUCTION_ALLOWED_DEFERRED_CANONICALS
        | batch2
    )
    structural = dual_backend_structural_candidates()

    for canon in core:
        if canon in FORBIDDEN_PRODUCTION_FASTPATH:
            status = "forbidden"
        elif canon in FASTPATH_DEFERRED_CANONICALS or canon in PRODUCTION_ALLOWED_DEFERRED_CANONICALS:
            status = "deferred"
        elif primitive_dual_backend_production_safe(canon):
            status = "certified_dual"
        elif canon in batch2:
            status = "python_rolling_pending"
        elif canon in structural:
            status = "structural_candidate"
        else:
            status = "implemented_single_backend"

        tier = "p0" if canon in P0_PRODUCTION_FASTPATH_CANONICALS else "p1"
        if canon in P1_POLARS_CORE_PRODUCTION_SAFE and canon not in P0_PRODUCTION_FASTPATH_CANONICALS:
            tier = "p1_core"

        policy = infer_operator_policy(canon)
        policy_row = {
            "scope": getattr(policy, "scope", "elementwise"),
            "pit_safe": getattr(policy, "pit_safe", True),
            "min_periods": getattr(policy, "min_periods", None),
        }
        row = build_fastpath_coverage_row(canon)
        primitives[canon] = {
            "kind": "primitive",
            "tier": tier,
            "phase1_status": status,
            "production_policy": infer_production_policy(canon),
            "polars_long": {
                "tier": polars_long_tier(canon),
                "production_tier": polars_long_production_tier(canon),
            },
            "duckdb_sql": {
                "implemented": is_sql_implemented(canon),
                "production_safe": effective_sql_production_safe(canon),
            },
            "certification": {
                "dual_backend_production_safe": primitive_dual_backend_production_safe(canon),
                "effective_dual_backend_fastpath": row.effective_dual_backend_fastpath,
            },
            "semantics": _default_semantics(canon, policy_row),
        }

    composites = sorted(list_composite_lowerings())
    return {
        "schema_version": 1,
        "freeze_id": "phase1-dual-backend-primitives-v1",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "frozen_commit_sha": _git_sha() or current_git_sha(),
        "description": (
            "第一阶段 production 范围冻结：约 90 primitive + 16 composite。"
            "未经 scripts/certify_operator.py 完整证据链不得新增 production 算子。"
        ),
        "invariants": [
            "phase1_status=certified_dual 须 primitive_verified.json 三证交集",
            "composite 须 composite_verified.json 五证齐全才可 production_policy=allowed",
            "冻结后新增算子须先更新本文件并走 certify_operator",
        ],
        "composite_defaults": {
            "phase1_status": "composite_pending",
            "production_policy": "pending",
            "lowering_only": True,
        },
        "composites": composites,
        "primitives": primitives,
        "counts": {
            "primitives": len(primitives),
            "composites": len(composites),
            "certified_dual": sum(
                1 for v in primitives.values() if v.get("phase1_status") == "certified_dual"
            ),
            "composite_certified": sum(1 for c in composites if composite_production_safe(c)),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--check", action="store_true", help="校验 --out 与当前 build 一致")
    args = parser.parse_args()
    _bootstrap()

    try:
        import yaml
    except ImportError:
        print("需要 PyYAML: pip install pyyaml", file=sys.stderr)
        return 1

    doc = build_scope_document()
    if args.check:
        if not args.out.is_file():
            print(f"缺少 {args.out}", file=sys.stderr)
            return 1
        ref = yaml.safe_load(args.out.read_text(encoding="utf-8"))
        if ref.get("primitives") != doc.get("primitives") or ref.get("composites") != doc.get(
            "composites"
        ):
            print("phase1_production_scope.yaml 过期", file=sys.stderr)
            return 1
        print(f"phase1 scope fresh: {args.out}")
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        yaml.safe_dump(doc, allow_unicode=True, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )
    print(f"wrote {args.out} ({doc['counts']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
