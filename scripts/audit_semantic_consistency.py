#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""四方语义一致性审计：SemanticFieldCatalog × datasets.yaml × FactorEngine
FIELD_REGISTRY × COS contract。

用途
    防止「SemanticFieldCatalog 单一事实源」与其它来源漂移：catalog 声明的
    dataset / physical 列 / knowledge_time / period_time / revision_order /
    scale 必须能在 datasets.yaml 里对得上；factor_engine 的 FIELD_REGISTRY
    同名字段的单位换算必须一致。CI 集成时任一不一致 → 退出码 1。

用法
    python3 scripts/audit_semantic_consistency.py [--strict]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dataaccess():
    sys.path.insert(0, str(ROOT / "dataaccess"))
    import data_access  # noqa: F401
    return sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true",
                        help="production：所有不一致都 fail-closed")
    args = parser.parse_args()
    problems: list[str] = []
    warnings: list[str] = []

    _load_dataaccess()
    from data_access.core.engine import DuckDBEngine  # noqa: F401
    from data_access.read.semantic_catalog import get_semantic_catalog
    from data_access.registry import load_registry

    # ---- 1. 加载 sources ----
    catalog = get_semantic_catalog()
    cfg_path = ROOT / "dataaccess" / "config" / "datasets.yaml"
    if not cfg_path.exists():
        # 允许环境变量覆盖（部署/镜像机）
        import os
        cfg_path = Path(os.environ.get("DATA_ACCESS_CONFIG", ""))
    registry = load_registry(str(cfg_path)) if cfg_path.exists() else None

    # factor_engine FIELD_REGISTRY（可选）
    field_registry = None
    try:
        sys.path.insert(0, str(ROOT / "factor_engine"))
        from fields import FIELD_REGISTRY
        field_registry = FIELD_REGISTRY
    except Exception as exc:  # pragma: no cover
        warnings.append(f"factor_engine FIELD_REGISTRY 不可用（跳过）：{exc}")

    # COS contract（可选）
    cos_contract = None
    try:
        from data_access.cos_contract import get_cos_contract
        cos_contract = get_cos_contract
    except Exception as exc:  # pragma: no cover
        warnings.append(f"COS contract 不可用（跳过）：{exc}")

    # ---- 2. catalog × datasets.yaml ----
    for name in catalog.names():
        f = catalog.get(name)
        if registry is not None:
            try:
                ds = registry.get(f.dataset)
            except Exception as exc:
                problems.append(f"[catalog↔registry] '{name}': 数据集 '{f.dataset}' 未注册：{exc}")
                continue
            schema = getattr(ds, "schema", None) or {}
            if f.physical_name not in schema:
                problems.append(
                    f"[catalog↔registry] '{name}': 物理列 '{f.dataset}.{f.physical_name}' "
                    f"不在 datasets.yaml schema 中"
                )
            for role, col in (
                ("knowledge_time", f.knowledge_time),
                ("effective_time", f.effective_time),
                ("period_time", f.period_time),
            ):
                if col and col not in schema:
                    problems.append(
                        f"[catalog↔registry] '{name}': {role} 列 '{col}' 不在 '{f.dataset}' schema"
                    )
            for col in f.revision_order:
                if col and col not in schema:
                    problems.append(
                        f"[catalog↔registry] '{name}': revision_order 列 '{col}' "
                        f"不在 '{f.dataset}' schema"
                    )

        # ---- 3. catalog × FIELD_REGISTRY（单位换算一致性） ----
        if field_registry is not None and f.scale is not None:
            spec = field_registry.get(name)
            # catalog 逻辑名 + 别名都试一遍（catalog 用 return_bp，FE 用 ret）
            if spec is None:
                for alias in f.aliases:
                    spec = field_registry.get(alias)
                    if spec is not None:
                        break
            if spec is not None and getattr(spec, "scale_to_canonical", None):
                fe_scale = float(spec.scale_to_canonical)
                if abs(fe_scale - float(f.scale)) > 1e-9:
                    problems.append(
                        f"[catalog↔FIELD_REGISTRY] '{name}': scale 不一致 "
                        f"catalog={f.scale} vs FE={fe_scale}"
                    )
            else:
                warnings.append(f"[catalog↔FIELD_REGISTRY] '{name}': FE 无同名字段，无法核对 scale")

        # ---- 4. catalog × COS contract（事件表可见性） ----
        if cos_contract is not None and f.temporal_model == "financial_event":
            try:
                c = cos_contract(f.dataset)
                if c is not None and getattr(c, "knowledge_time_column", None):
                    if f.knowledge_time and c.knowledge_time_column != f.knowledge_time:
                        problems.append(
                            f"[catalog↔COS] '{name}': knowledge_time 不一致 "
                            f"catalog={f.knowledge_time} vs COS={c.knowledge_time_column}"
                        )
            except Exception as exc:  # pragma: no cover
                warnings.append(f"[catalog↔COS] '{name}' 无法核对：{exc}")

    # ---- 5. 汇总 ----
    print("Semantic consistency audit")
    print("=" * 40)
    print(f"catalog 字段数: {len(catalog.names())}")
    print(f"问题: {len(problems)}  告警: {len(warnings)}")
    for p in sorted(problems):
        print(f"  PROBLEM  {p}")
    for w in sorted(warnings):
        print(f"  warning  {w}")

    if problems:
        print("RESULT: FAIL")
        return 1
    print("RESULT: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
