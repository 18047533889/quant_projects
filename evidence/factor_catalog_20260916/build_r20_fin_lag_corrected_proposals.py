"""Build and verify the final delta for reversed financial announcement lags."""
from __future__ import annotations

import ast
import csv
import gzip
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SOURCE = HERE / "factor_catalog_review_r19_final.csv.gz"
MERGED = HERE / "r20-merged-final.jsonl.gz"
OUTPUT = HERE / "r20_fin_lag_corrected_proposals.jsonl"
EVIDENCE = HERE / "r20_fin_lag_corrected_evidence.json"

# A single-statement factor uses that statement's own disclosure calendar.  A
# cross-statement factor uses StockIndicator as the canonical joined filing
# calendar, avoiding a PubDate/period_end pair drawn from different vintages.
TABLE_BY_ID = {
    "R65X_00448": "StockIndicator",
    "R65X_00449": "StockIndicator",
    "R65X_00450": "StockIndicator",
    "R65X_00451": "StockBalance",
    "R65X_00452": "StockIndicator",
    "R65X_00453": "StockBalance",
    "R65X_00454": "StockIncome",
    "R66_01319": "StockIncome",
    "R66_01321": "StockIncome",
    "R66_01323": "StockIncome",
    "R66_01325": "StockCashFlow",
    "R66_01327": "StockBalance",
    "R66_01329": "StockBalance",
    "R66_01331": "StockBalance",
    "R66_01333": "StockBalance",
    "R66_01335": "StockBalance",
    "R66_01337": "StockIncome",
    "R67_00373": "StockBalance",
    "R67_00378": "StockBalance",
    "R67_00383": "StockBalance",
    "R67_00388": "StockIndicator",
}


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def lag_calls(formula: str) -> list[ast.Call]:
    tree = ast.parse(formula, mode="eval")
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "fin_announcement_lag"
    ]


def corrected_formula(formula: str, table: str) -> str:
    calls = lag_calls(formula)
    if len(calls) != 1:
        raise ValueError(f"expected one fin_announcement_lag call, got {len(calls)}")
    call = calls[0]
    if len(call.args) != 2:
        raise ValueError("fin_announcement_lag must have two positional arguments")
    replacement = (
        "fin_announcement_lag("
        f"field('report_period_end_date', table='{table}'), "
        f"field('pub_date', table='{table}'))"
    )
    return formula[: call.col_offset] + replacement + formula[call.end_col_offset :]


def main() -> None:
    if OUTPUT.exists() or EVIDENCE.exists():
        raise FileExistsError("refusing to overwrite existing correction evidence")

    merged: dict[str, dict] = {}
    with gzip.open(MERGED, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            merged[row["id"]] = row

    source: dict[str, dict] = {}
    with gzip.open(SOURCE, "rt", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["id"]:
                source[row["id"]] = row
    if len(source) != 113893:
        raise ValueError(f"unexpected source count: {len(source)}")

    missing = sorted(set(TABLE_BY_ID) - set(merged))
    if missing:
        raise ValueError(f"missing merged rows: {missing}")

    # Scan the complete effective R20 surface: R19 plus every merged override.
    effective_lags: dict[str, str] = {}
    for factor_id, row in source.items():
        formula = merged.get(factor_id, {}).get("current_formula", row["current_formula"])
        if lag_calls(formula):
            effective_lags[factor_id] = formula

    root = ROOT
    sys.path.insert(0, str(root / "evidence/factor_catalog_20260915"))
    from compile_catalog import build_runtime
    from smoke_catalog import bind_fields
    from factor_engine.api.factor import Factor

    parser, engine = build_runtime()
    records = []
    for factor_id, table in TABLE_BY_ID.items():
        prior = merged[factor_id]
        current = corrected_formula(prior["current_formula"], table)
        expr = parser.parse(current)
        bindings, failures = bind_fields(expr)
        if failures:
            raise ValueError(f"{factor_id} binding failures: {failures}")
        engine.compile(
            Factor(name=factor_id, expr=expr, source_expr=current, surface="compat_research")
        )
        changes = list(prior.get("changes") or [])
        changes.append(
            "FIN_ANNOUNCEMENT_LAG_CONTRACT_REPAIR: 参数按正式算子合同改为"
            f"(report_period_end_date, pub_date)，两字段同表绑定 {table}，"
            "结果口径为公告日减报告期末的自然日天数"
        )
        if table == "StockIndicator":
            changes.append(
                "FIN_ANNOUNCEMENT_CALENDAR: 多报表联合因子采用 StockIndicator "
                "作为统一披露日历，避免跨表混配不同报告版本"
            )
        records.append(
            {
                "source_row": int(prior["source_row"]),
                "id": factor_id,
                "before_formula": prior["before_formula"],
                "current_formula": current,
                "changes": changes,
                "compile_status": "COMPILED",
                "error": "",
                "bindings": bindings,
                "binding_failures": [],
            }
        )

    records.sort(key=lambda row: (row["source_row"], row["id"]))
    with OUTPUT.open("x", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Re-scan with the delta applied and assert every remaining call is in the
    # documented operator order.  Field calls make the semantic check exact.
    corrected = {row["id"]: row["current_formula"] for row in records}
    effective_lags.update(corrected)
    violations = []
    for factor_id, formula in sorted(effective_lags.items()):
        for call in lag_calls(formula):
            first = ast.unparse(call.args[0]).lower()
            second = ast.unparse(call.args[1]).lower()
            if "period_end" not in first or "pub_date" not in second:
                violations.append(
                    {"id": factor_id, "first_arg": first, "second_arg": second}
                )
    evidence = {
        "scope": "complete effective R20 surface (113893 R19 rows plus merged overrides)",
        "source_sha256": digest(SOURCE),
        "merged_sha256": digest(MERGED),
        "proposal_count": len(records),
        "effective_factor_count": len(source),
        "effective_fin_announcement_lag_factor_count": len(effective_lags),
        "remaining_contract_violations": violations,
        "compile_counts": {"COMPILED": len(records)},
        "output_sha256": digest(OUTPUT),
    }
    if violations:
        raise ValueError(f"remaining lag contract violations: {violations}")
    with EVIDENCE.open("x", encoding="utf-8") as handle:
        json.dump(evidence, handle, ensure_ascii=False, indent=2)
    print(json.dumps(evidence, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
