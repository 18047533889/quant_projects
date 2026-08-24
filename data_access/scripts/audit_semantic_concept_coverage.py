#!/usr/bin/env python3
"""R30-P1-027 / R30-§67 —— SemanticFieldCatalog concept coverage 静态审计。

遍历 ``data_access.read.semantic_catalog`` 的全部 SemanticField，检查：

    1. ``logical_name`` 非空；
    2. ``dimension / frequency / grain`` 为**非空字符串**（``None`` 视为合法的
       「未声明」，空字符串视为违规——配置里写了但为空说明写错了）；
    3. 概念一致性：``dimension == "money"`` 但 ``currency is None`` → 违规
       （money 维度的字段必须有币种语义）；
    4. ``cross_market_comparable=true`` 与 ``requires_fx=true`` 同时成立 → 违规
       （自相矛盾：宣称跨市场可比又要求显式 FX）；
    5. 找出 **missing concept mapping** 的字段：dimension / source_unit /
       canonical_unit / currency 全为 None（没有任何单位/币种语义声明）。
       这类字段**列出并计数**，但不计为硬违规（无量纲字段（count/shares/
       identifier/boolean）合法缺省）。

只做静态检查，不连接数据源。退出码：0 = 0 硬违规；1 = 有硬违规（打印清单）。

用法：
    python3 scripts/audit_semantic_concept_coverage.py [--json]
    python3 scripts/audit_semantic_concept_coverage.py --catalog <semantic_fields.yaml>
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any


def _attr(obj: Any, name: str, default: Any = None) -> Any:
    return getattr(obj, name, default)


def _field_items(catalog: Any) -> list[tuple[str, Any]]:
    """防御性遍历：优先内部 ``_fields``（dict：YAML key → SemanticField），
    退化为 ``names()`` + ``resolve_one()``。"""
    fields = getattr(catalog, "_fields", None)
    if isinstance(fields, dict):
        return [(name, f) for name, f in fields.items()]
    names = getattr(catalog, "names", None)
    if callable(names):
        resolve = getattr(catalog, "resolve_one", None)
        if callable(resolve):
            out: list[tuple[str, Any]] = []
            for n in names():
                try:
                    out.append((n, resolve(n)))
                except Exception:
                    out.append((n, None))
            return out
    return []


def audit_catalog(catalog: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    items = _field_items(catalog)
    violations: list[dict[str, Any]] = []
    missing_concept: list[str] = []
    counts: dict[str, Any] = {
        "fields_total": len(items),
        "with_dimension": 0,
        "with_currency": 0,
        "derived": 0,
        "market_dist": {},
        "missing_concept_mapping": 0,
    }

    for key, f in items:
        logical = str(_attr(f, "logical_name", "") or "")
        display = logical or key or "<unnamed>"
        market = str(_attr(f, "market", "any") or "any")
        counts["market_dist"][market] = counts["market_dist"].get(market, 0) + 1
        if _attr(f, "derived_expression", None):
            counts["derived"] += 1

        # 1) logical_name 非空
        if not logical.strip():
            violations.append(
                {
                    "field": display,
                    "check": "MISSING_LOGICAL_NAME",
                    "detail": "logical_name 缺失/为空",
                }
            )

        # 2) dimension / frequency / grain 非空字符串（None 合法）
        for field in ("dimension", "frequency", "grain"):
            value = _attr(f, field, None)
            if isinstance(value, str) and not value.strip():
                violations.append(
                    {
                        "field": display,
                        "check": "EMPTY_DECLARATION",
                        "detail": f"{field}={value!r} 是空字符串（应省略或给非空值）",
                    }
                )
        if _attr(f, "dimension", None):
            counts["with_dimension"] += 1
        if _attr(f, "currency", None):
            counts["with_currency"] += 1

        # 3) money 维度必须有币种
        if _attr(f, "dimension", None) == "money" and _attr(f, "currency", None) is None:
            violations.append(
                {
                    "field": display,
                    "check": "MONEY_WITHOUT_CURRENCY",
                    "detail": "dimension=money 但未声明 currency（币种语义缺失）",
                }
            )

        # 4) cross_market_comparable 与 requires_fx 矛盾
        cm = bool(_attr(f, "cross_market_comparable", True))
        fx = bool(_attr(f, "requires_fx", False))
        if cm and fx:
            violations.append(
                {
                    "field": display,
                    "check": "CM_CONTRADICTS_FX",
                    "detail": (
                        "cross_market_comparable=true 且 requires_fx=true 自相矛盾"
                        "（宣称可直接跨市场比较又要求显式 FX）"
                    ),
                }
            )

        # 5) missing concept mapping：dimension/unit/currency 全无
        if (
            _attr(f, "dimension", None) is None
            and _attr(f, "source_unit", None) is None
            and _attr(f, "canonical_unit", None) is None
            and _attr(f, "currency", None) is None
        ):
            missing_concept.append(display)

    counts["missing_concept_mapping"] = len(missing_concept)
    return violations, counts, missing_concept


def main() -> int:
    parser = argparse.ArgumentParser(
        description="SemanticFieldCatalog concept coverage 静态审计"
    )
    parser.add_argument(
        "--catalog",
        default=None,
        help="semantic_fields.yaml 路径（缺省用 get_semantic_catalog 默认路径）",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="以 JSON 输出结果（含 counts + violations + missing_concept）",
    )
    args = parser.parse_args()

    try:
        from data_access.read.semantic_catalog import get_semantic_catalog
    except Exception as exc:  # pragma: no cover
        print(f"[FATAL] 无法导入 data_access.read.semantic_catalog: {exc}", file=sys.stderr)
        return 2

    try:
        catalog = get_semantic_catalog(args.catalog)
    except Exception as exc:
        print(f"[FATAL] 加载 semantic catalog 失败: {exc}", file=sys.stderr)
        return 2

    violations, counts, missing_concept = audit_catalog(catalog)

    if args.json:
        print(
            json.dumps(
                {
                    "ok": len(violations) == 0,
                    "counts": counts,
                    "violations": violations,
                    "violation_count": len(violations),
                    "missing_concept_mapping_fields": sorted(missing_concept),
                },
                ensure_ascii=False,
                indent=2,
                default=str,
            )
        )
        return 0 if not violations else 1

    markets = ", ".join(
        f"{k}={v}" for k, v in sorted(counts["market_dist"].items())
    )
    print(
        f"Semantic concept coverage 审计：{counts['fields_total']} 个字段 "
        f"(with_dimension={counts['with_dimension']}, with_currency="
        f"{counts['with_currency']}, derived={counts['derived']})"
    )
    print(f"  market 分布: {markets}")

    if missing_concept:
        print(
            f"  missing concept mapping（dimension/unit/currency 全无，合法无量纲）: "
            f"{counts['missing_concept_mapping']} 个"
        )
        print("    " + ", ".join(sorted(missing_concept)))

    if not violations:
        print("结果: 0 硬违规")
        return 0

    print(f"结果: {len(violations)} 硬违规")
    for v in violations:
        print(f"  - [{v['check']}] {v['field']}: {v['detail']}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
