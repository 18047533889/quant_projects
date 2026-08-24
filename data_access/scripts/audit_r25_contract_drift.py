#!/usr/bin/env python3
"""R25 §109 Audit 1 —— MirrorSpec/PhysicalSpec vs ContractIR 漂移静态审计。

用法：
    python -m data_access.scripts.audit_r25_contract_drift

检查：
    1. COS contract storage_layout（period_files/event_files/prefixed）与
       RuntimeDatasetContract 编译出的 PhysicalPartitionSpec 一致；
    2. US finance 不是 daily_parquet（P0-001）；
    3. StockCapital split/shares file_selector 一致（P0-002）。

退出码：0 = 无 blocking；1 = 有漂移（CI failure）。
"""
from __future__ import annotations

import sys


def main() -> int:
    from data_access.contract.runtime_contract import compile_runtime_contract
    from data_access.cos_contract import get_cos_contract
    from data_access.read.contract_ir import audit_runtime_contract_drift
    from data_access.registry import load_registry

    registry = load_registry()
    problems = audit_runtime_contract_drift(registry)

    # 额外：US finance 必须编译成 PERIOD_END_FILE（P0-001 硬卡）。
    from data_access.contract.physical_partition import PhysicalLayout

    for name in ("us_stock_balance", "us_stock_income", "us_stock_cashflow"):
        rc = compile_runtime_contract(name, registry)
        if rc is None or rc.physical_partition.layout != PhysicalLayout.PERIOD_END_FILE:
            problems.append(
                f"US finance {name!r} 未编译成 PERIOD_END_FILE（P0-001 fail）"
            )
    # StockCapital file_selector（P0-002 硬卡）。
    for name, sel in (
        ("us_stock_capital_split", None),
        ("us_stock_capital_shares", "shares_"),
    ):
        from data_access.cos.mirror import mirror_spec_for_dataset

        spec = mirror_spec_for_dataset(name)
        if spec is None:
            problems.append(f"StockCapital {name!r} 无 mirror spec（P0-002 fail）")
            continue
        if spec.file_selector != sel:
            problems.append(
                f"StockCapital {name!r} file_selector={spec.file_selector!r} != {sel!r}"
            )

    if problems:
        print("R25 ContractIR drift audit FAILED:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("R25 ContractIR drift audit OK (0 blocking problems)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
