#!/usr/bin/env python3
"""算子契约验收：metadata / policy / alias / stub / cost 门禁。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _bootstrap() -> None:
    root = Path(__file__).resolve().parents[1]
    project = root.parent
    for p in (str(root), str(project)):
        if p not in sys.path:
            sys.path.insert(0, p)
    from cleaned_operators import load_all
    from runtime.env_bootstrap import bootstrap_runtime_env

    bootstrap_runtime_env()
    load_all()


def check_operator_contracts(*, strict_tier1_cost: bool = True) -> list[str]:
    from backend.operator_cost import tier1_has_explicit_cost
    from cleaned_operators.operator_policy import (
        TIER1_CANONICALS,
        _EXPLICIT_POLICIES,
        infer_operator_policy,
        resolve_tier1_canonical,
        tier1_policy_keys,
    )
    from cleaned_operators.operator_spec import check_production_pit_declarations, iter_operator_specs
    from cleaned_operators.operator_spec import (
        check_capm_param_contracts,
        check_fundamental_param_contracts,
        check_intraday_param_contracts,
        check_microstructure_param_contracts,
    )
    from cleaned_operators.registry import OperatorRegistry

    errors: list[str] = []

    for canon, backends in OperatorRegistry._operators.items():
        for backend, op in backends.items():
            meta = getattr(op, "metadata", None)
            if meta is None:
                errors.append(f"{canon}/{backend}: 缺少 metadata")
                continue
            if not getattr(meta, "name", ""):
                errors.append(f"{canon}/{backend}: metadata.name 为空")

    for alias, canon in OperatorRegistry._aliases.items():
        if canon not in OperatorRegistry._operators and canon not in OperatorRegistry._catalog:
            errors.append(f"alias {alias!r} → 不存在的 canonical {canon!r}")

    for key in tier1_policy_keys():
        if key not in _EXPLICIT_POLICIES:
            errors.append(f"Tier-1 {key!r} 缺少显式 OperatorPolicy")

    for name in sorted(TIER1_CANONICALS):
        canon = resolve_tier1_canonical(name)
        if strict_tier1_cost and not tier1_has_explicit_cost(canon):
            errors.append(f"Tier-1 {canon!r} 缺少显式 OperatorCost")

    for canon, entry in OperatorRegistry._catalog.items():
        if str(entry.get("status", "")).endswith("stub"):
            if canon in OperatorRegistry._operators:
                errors.append(f"stub 算子 {canon!r} 不应有 runtime 注册")

    for canon in OperatorRegistry._operators:
        op = OperatorRegistry.get(canon)
        if op is None:
            continue
        policy = infer_operator_policy(op, canonical=canon)
        if canon in _EXPLICIT_POLICIES and not policy.pit_safe:
            if canon not in {"Lead", "next", "bfill", "fillna_interpolate", "shuffle"}:
                errors.append(f"显式 policy 算子 {canon!r} pit_safe=False 异常")

    for spec in iter_operator_specs():
        if spec.status in ("stub", "doc_only") and spec.backends:
            errors.append(f"stub/doc_only 算子 {spec.canonical!r} 不应有 runtime backend")

    errors.extend(check_production_pit_declarations())
    errors.extend(check_microstructure_param_contracts())
    errors.extend(check_fundamental_param_contracts())
    errors.extend(check_capm_param_contracts())
    errors.extend(check_intraday_param_contracts())

    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="算子契约验收")
    parser.add_argument(
        "--skip-tier1-cost",
        action="store_true",
        help="不检查 Tier-1 OperatorCost 显式登记",
    )
    args = parser.parse_args(argv)
    _bootstrap()
    errors = check_operator_contracts(strict_tier1_cost=not args.skip_tier1_cost)
    if errors:
        for err in errors:
            print(f"[FAIL] {err}", file=sys.stderr)
        print(f"共 {len(errors)} 项失败", file=sys.stderr)
        return 1
    print("算子契约验收通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
