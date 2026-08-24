#!/usr/bin/env python3
"""#12 Contract IR 一致性审计：registry + COS 契约 + 语义字段 三方对齐。

用法：
    python3 scripts/audit_contract_ir.py [--external us_fact_news]

输出：
    退出码 0 = 一致；非 0 = 有问题（打印详情）。
"""
from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Contract IR 一致性审计")
    parser.add_argument(
        "--external",
        nargs="*",
        default=(),
        help="契约声明但不在 registry 的数据集白名单（如 COS remote 才可见）",
    )
    args = parser.parse_args()

    from data_access import get_store
    from data_access.read.contract_ir import build_contract_ir

    store = get_store()
    ir = build_contract_ir(
        store.registry,
        catalog=__import__(
            "data_access.read.semantic_catalog", fromlist=["get_semantic_catalog"]
        ).get_semantic_catalog(),
        external_contract_datasets=args.external,
    )
    problems = ir.audit()
    if not problems:
        print(
            f"Contract IR 一致：{len(ir.names())} 个数据集，"
            f"fingerprint={ir.fingerprint()}"
        )
        return 0
    print(f"Contract IR 不一致（{len(problems)} 个问题）：")
    for p in problems:
        print(f"  - {p}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
