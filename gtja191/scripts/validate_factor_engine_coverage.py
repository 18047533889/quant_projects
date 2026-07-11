#!/usr/bin/env python3
"""对 185 条 GTJA-191 因子执行真实 FactorEngine 编译与合成数据计算门禁。"""
from __future__ import annotations

import argparse
import ast
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent
FACTOR_ENGINE_ROOT = REPO_ROOT / "factor_engine"
for path in (PACKAGE_ROOT, FACTOR_ENGINE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from lib.catalog import DELIVERABLE_COUNT, deliverable_catalog  # noqa: E402
from lib.synthetic_source import SyntheticGTJADataSource  # noqa: E402

_ALLOWED_FIELDS = {
    "open",
    "high",
    "low",
    "close",
    "volume",
    "vwap",
    "amount",
    "ret",
    "preclose",
}


class _FormulaInventory(ast.NodeVisitor):
    def __init__(self) -> None:
        self.calls: Counter[str] = Counter()
        self.names: set[str] = set()
        self.explicit_columns: set[str] = set()

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name):
            self.calls[node.func.id] += 1
            if node.func.id == "col" and node.args:
                arg = node.args[0]
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    self.explicit_columns.add(arg.value)
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        self.names.add(node.id)
        self.generic_visit(node)


def _inventory(formula: str) -> _FormulaInventory:
    inv = _FormulaInventory()
    inv.visit(ast.parse(formula, mode="eval"))
    return inv


def validate_catalog(*, execute: bool = True, periods: int = 420, symbols: int = 8) -> dict[str, Any]:
    from api.dsl_parser import parse_factor
    from backend.factory import build_backend
    from runtime.engine import FactorEngine

    catalog = deliverable_catalog()
    if len(catalog) != DELIVERABLE_COUNT:
        raise AssertionError(f"expected {DELIVERABLE_COUNT} factors, got {len(catalog)}")

    source = SyntheticGTJADataSource(periods=periods, symbols=symbols)
    engine = FactorEngine(
        backend=build_backend("pandas"),
        data_source=source,
        run_mode="research",
    )

    started = time.perf_counter()
    errors: list[dict[str, Any]] = []
    compiled = 0
    executed = 0
    finite_counts: dict[str, int] = {}
    operator_counts: Counter[str] = Counter()
    referenced_fields: set[str] = set()

    for name, item in sorted(catalog.items()):
        formula = str(item["dsl_formula"])
        inv = _inventory(formula)
        operator_counts.update(inv.calls)
        referenced_fields.update(inv.explicit_columns)
        bare_fields = {n for n in inv.names if n in _ALLOWED_FIELDS}
        referenced_fields.update(bare_fields)

        source_formula = str(item.get("source_formula", "")).upper()
        if "VWAP" in source_formula and "col('vwap')" not in formula and 'col("vwap")' not in formula:
            errors.append(
                {
                    "factor": name,
                    "stage": "semantic",
                    "error": "source formula uses VWAP but normalized DSL does not use col('vwap')",
                    "formula": formula,
                }
            )
            continue

        unknown_fields = sorted((inv.names - set(inv.calls)) - _ALLOWED_FIELDS)
        # 字符串常量、函数名不会进入 unknown_fields；剩余 bare name 必须是已知字段。
        if unknown_fields:
            errors.append(
                {
                    "factor": name,
                    "stage": "fields",
                    "error": f"unknown bare fields: {unknown_fields}",
                    "formula": formula,
                }
            )
            continue

        try:
            factor = parse_factor(
                formula,
                name=name,
                freq="1d",
                universe="ASHARE_ALL",
                description=f"GTJA-191 {name}",
            )
            plan, analysis = engine.compile(factor)
            compiled += 1
            missing = sorted(set(analysis.referenced_columns) - _ALLOWED_FIELDS)
            if missing:
                raise AssertionError(f"unsupported input columns: {missing}")
            if not execute:
                continue
            out = engine.run(factor, plan=plan, analysis=analysis)
            result = out["result"]
            if len(result) != periods * symbols:
                raise AssertionError(
                    f"unexpected result length {len(result)} != {periods * symbols}"
                )
            values = np.asarray(result, dtype=float)
            finite = int(np.isfinite(values).sum())
            finite_counts[name] = finite
            if finite <= 0:
                raise AssertionError("factor produced no finite values on synthetic panel")
            executed += 1
        except Exception as exc:  # aggregate all 185 failures in one report
            errors.append(
                {
                    "factor": name,
                    "stage": "execute" if execute else "compile",
                    "error": f"{type(exc).__name__}: {exc}",
                    "formula": formula,
                }
            )

    elapsed = time.perf_counter() - started
    return {
        "ok": not errors,
        "deliverable_count": len(catalog),
        "compiled": compiled,
        "executed": executed,
        "compile_only": not execute,
        "periods": periods,
        "symbols": symbols,
        "elapsed_seconds": round(elapsed, 3),
        "operators": dict(sorted(operator_counts.items())),
        "referenced_fields": sorted(referenced_fields),
        "min_finite_values": min(finite_counts.values()) if finite_counts else 0,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--periods", type=int, default=420)
    parser.add_argument("--symbols", type=int, default=8)
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    report = validate_catalog(
        execute=not args.compile_only,
        periods=args.periods,
        symbols=args.symbols,
    )
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.report:
        Path(args.report).write_text(text + "\n", encoding="utf-8")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
