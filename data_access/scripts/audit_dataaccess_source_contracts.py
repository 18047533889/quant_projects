#!/usr/bin/env python3
"""R30-P1-027 / R30-§67 —— Dataset Registry source contract 静态审计。

遍历 ``data_access.registry`` 的全部数据集，检查 contract 字段一致性：

    1. 必填字段存在（name / access_mode / layout / format）；
    2. ``mutation_owner`` ∈ {dataaccess, external_versioned, external_mutable,
       immutable}；
    3. ``generation_required`` 与 ``mutation_owner`` 匹配：
       - external_mutable + generation_required=true 自相矛盾（外部可变源无法
         保证不可变代）；
       - generation_required=true 但未声明 generation_pointer=true（读路径
         ``resolve_paths`` 无法解析代 → fail-closed）；
    4. 有 ``schema_migrations`` 时每项 ``approved`` 必须为 bool（非 bool 说明
       审批状态不可判定，fail-closed）；且 ``from_fingerprint / to_fingerprint /
       kind`` 非空。

只做静态检查，不连接数据源。退出码：0 = 0 违规；1 = 有违规（打印清单）。

用法：
    python3 scripts/audit_dataaccess_source_contracts.py [--json]
    python3 scripts/audit_dataaccess_source_contracts.py --config <datasets.yaml>

注：registry loader 本身在 parse 时已做 strict 校验（mutation_owner 枚举、
approved bool 等），本脚本是对**加载后的对象模型**再做一次独立一致性核对，
防止对象被绕路构造（测试/mock/手工构造）时 contract 漂移。
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Mapping

# 合法的 mutation ownership 枚举（R29-P0 #206，与 registry/loader.py 一致）。
VALID_MUTATION_OWNERS = frozenset(
    {"dataaccess", "external_versioned", "external_mutable", "immutable"}
)

# 加载后对象必填的关键字段。
REQUIRED_DATASET_FIELDS = ("name", "access_mode", "layout", "format")

# schema_migrations 每项必填字段。
REQUIRED_MIGRATION_FIELDS = ("from_fingerprint", "to_fingerprint", "kind")


def _attr(obj: Any, name: str, default: Any = None) -> Any:
    """防御性 getattr：对象可能缺字段（StaticDataset 无 generation_required 等）。"""
    return getattr(obj, name, default)


def _iter_datasets(reg: Any) -> list[Any]:
    """防御性遍历 registry：优先 __iter__（yields Dataset），退化为 names()+get()。"""
    datasets: list[Any] = []
    try:
        for ds in reg:
            datasets.append(ds)
    except Exception:
        names = getattr(reg, "names", None)
        getter = getattr(reg, "get", None)
        if callable(names) and callable(getter):
            datasets = [getter(n) for n in names()]
    return datasets


def audit_registry(reg: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """遍历 registry，返回 (violations, counts)。"""
    datasets = _iter_datasets(reg)
    violations: list[dict[str, Any]] = []
    counts: dict[str, Any] = {
        "datasets_total": len(datasets),
        "static": 0,
        "parametric": 0,
        "generation_pointer": 0,
        "generation_required": 0,
        "schema_migrations_entries": 0,
        "mutation_owner_dist": {},
        "datasets_with_schema_migrations": 0,
    }

    for ds in datasets:
        name = str(_attr(ds, "name", "<unnamed>"))
        kind = str(_attr(ds, "kind", "")).lower()
        if kind in ("static", "parametric"):
            counts[kind] += 1
        else:
            # 兜底：按类型名推断
            type_name = type(ds).__name__.lower()
            counts["static" if "static" in type_name else "parametric"] += 1
            kind = "static" if "static" in type_name else "parametric"

        # 1) 必填字段存在
        for field in REQUIRED_DATASET_FIELDS:
            value = _attr(ds, field, None)
            if value is None or (isinstance(value, str) and not value.strip()):
                violations.append(
                    {
                        "dataset": name,
                        "check": "REQUIRED_FIELD_MISSING",
                        "detail": f"必填字段 {field} 缺失/为空",
                    }
                )

        # 2) mutation_owner 枚举
        mo_raw = _attr(ds, "mutation_owner", "dataaccess")
        mo = str(mo_raw or "dataaccess").strip().lower()
        counts["mutation_owner_dist"][mo] = (
            counts["mutation_owner_dist"].get(mo, 0) + 1
        )
        if mo not in VALID_MUTATION_OWNERS:
            violations.append(
                {
                    "dataset": name,
                    "check": "MUTATION_OWNER_ENUM",
                    "detail": (
                        f"mutation_owner={mo!r} 不在合法集合 "
                        f"{sorted(VALID_MUTATION_OWNERS)} 内"
                    ),
                }
            )

        # 3) generation_pointer / generation_required（StaticDataset 可能无此字段）
        gen_pointer = bool(_attr(ds, "generation_pointer", False) or False)
        gen_required = bool(_attr(ds, "generation_required", False) or False)
        if gen_pointer:
            counts["generation_pointer"] += 1
        if gen_required:
            counts["generation_required"] += 1

        if gen_required and mo == "external_mutable":
            violations.append(
                {
                    "dataset": name,
                    "check": "GENERATION_REQUIRED_MUTATION_MISMATCH",
                    "detail": (
                        "mutation_owner=external_mutable 却声明 generation_required="
                        "true：外部可变源无法保证不可变代（manifest 新鲜度语义矛盾）"
                    ),
                }
            )
        if gen_required and not gen_pointer:
            violations.append(
                {
                    "dataset": name,
                    "check": "GENERATION_REQUIRED_WITHOUT_POINTER",
                    "detail": (
                        "generation_required=true 但未声明 generation_pointer=true："
                        "读路径 resolve_paths 无法解析当前代（fail-closed 会拒绝读取）"
                    ),
                }
            )

        # 4) schema_migrations：approved 必须 bool，且 fingerprint/kind 非空
        migrations = _attr(ds, "schema_migrations", ()) or ()
        if migrations:
            counts["datasets_with_schema_migrations"] += 1
        for i, mig in enumerate(migrations):
            counts["schema_migrations_entries"] += 1
            if not isinstance(mig, Mapping):
                violations.append(
                    {
                        "dataset": name,
                        "check": "MIGRATION_NOT_MAPPING",
                        "detail": f"schema_migrations[{i}] 不是 mapping",
                    }
                )
                continue
            approved = mig.get("approved")
            if not isinstance(approved, bool):
                violations.append(
                    {
                        "dataset": name,
                        "check": "MIGRATION_APPROVED_NOT_BOOL",
                        "detail": (
                            f"schema_migrations[{i}].approved={approved!r} 非 bool"
                            "（未显式批准不构成放行依据，fail-closed）"
                        ),
                    }
                )
            for field in REQUIRED_MIGRATION_FIELDS:
                value = mig.get(field)
                if value is None or (isinstance(value, str) and not value.strip()):
                    violations.append(
                        {
                            "dataset": name,
                            "check": "MIGRATION_FIELD_MISSING",
                            "detail": f"schema_migrations[{i}].{field} 缺失/为空",
                        }
                    )

    return violations, counts


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Dataset Registry source contract 静态审计"
    )
    parser.add_argument(
        "--config",
        default=None,
        help="datasets.yaml 路径（缺省用 registry.load_registry 默认路径）",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="以 JSON 输出结果（含 counts + violations）",
    )
    args = parser.parse_args()

    try:
        from data_access.registry.loader import load_registry
    except Exception as exc:  # pragma: no cover - 环境缺依赖
        print(f"[FATAL] 无法导入 data_access.registry.loader: {exc}", file=sys.stderr)
        return 2

    try:
        reg = load_registry(args.config)
    except Exception as exc:
        print(f"[FATAL] 加载 registry 失败: {exc}", file=sys.stderr)
        return 2

    violations, counts = audit_registry(reg)

    if args.json:
        print(
            json.dumps(
                {
                    "ok": len(violations) == 0,
                    "counts": counts,
                    "violations": violations,
                    "violation_count": len(violations),
                },
                ensure_ascii=False,
                indent=2,
                default=str,
            )
        )
        return 0 if not violations else 1

    dist = ", ".join(
        f"{k}={v}" for k, v in sorted(counts["mutation_owner_dist"].items())
    )
    print(
        f"Dataset source contract 审计：{counts['datasets_total']} 个数据集 "
        f"(static={counts['static']}, parametric={counts['parametric']})"
    )
    print(
        f"  generation_pointer={counts['generation_pointer']}, "
        f"generation_required={counts['generation_required']}, "
        f"schema_migrations_entries={counts['schema_migrations_entries']} "
        f"({counts['datasets_with_schema_migrations']} 数据集)"
    )
    print(f"  mutation_owner 分布: {dist}")

    if not violations:
        print("结果: 0 违规")
        return 0

    print(f"结果: {len(violations)} 违规")
    for v in violations:
        print(f"  - [{v['check']}] {v['dataset']}: {v['detail']}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
