#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R17 §7.5: PIT / period-policy audit over financial/event/dividend/news/holder
/capital sources.

For every production-readable TableSpec:

- knowledge_time is declared for financial_pit / relation_pit tables;
- effective_time is declared for effective_only tables;
- strict PIT tables are NOT effective-only;
- future ex-date is not a knowledge time (dividends);
- required filters (timeframe) are declared where the table needs them.

Run:  python3 scripts/audit_pit_and_period_policy.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _load() -> None:
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO.parent))


def main() -> int:
    _load()
    from fields.catalog import ASHARE_TABLE_SPECS
    from fields.catalog_us import US_TABLE_SPECS

    problems: list[str] = []
    for table in list(ASHARE_TABLE_SPECS) + list(US_TABLE_SPECS):
        name = table.name
        join = str(table.join_policy or "")
        knowledge = table.knowledge_time_column
        effective = table.effective_time_column
        pit = table.strict_pit_allowed
        if join in {"financial_pit", "relation_pit"} and not knowledge:
            problems.append(f"{name}: {join} requires knowledge_time_column")
        if join == "effective_only" and not effective:
            problems.append(f"{name}: effective_only requires effective_time_column")
        if join == "effective_only" and pit is True:
            problems.append(f"{name}: effective_only table must NOT claim PIT-safe")
        # Dividend: declaration_date knowledge, ex-date effective (R17-020).
        if name == "StockDividend" and table.dataset.startswith("us_"):
            if table.knowledge_time_column != "declaration_date":
                problems.append("US StockDividend knowledge_time != declaration_date (R17-020)")
            if table.effective_time_column != "ex_dividend_date":
                problems.append("US StockDividend effective_time != ex_dividend_date (R17-020)")
        if table.dataset.startswith("us_") and join == "financial_pit":
            # R17-034: US FINANCIAL STATEMENTS require a timeframe filter.  A
            # shares-PIT event source (USTickerSharesPITEvent) is not a statement
            # and has no timeframe dimension.
            is_statement = table.name in {"StockBalance", "StockIncome", "StockCashFlow"}
            if is_statement and "timeframe" not in (table.required_parameters or ()):
                problems.append(f"{name}: US financial statement requires timeframe filter (R17-034)")
        if not table.strict_pit_allowed and table.strict_pit_allowed is not False:
            problems.append(
                f"{name}: strict_pit_allowed is UNKNOWN (None) — every "
                "production-readable table must declare it (R17-013)"
            )

    print("R17 §7.5 PIT / period-policy audit")
    if problems:
        print(f"FAIL: {len(problems)}")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("OK: all production tables declare coherent PIT / period / filter contracts")
    return 0


if __name__ == "__main__":
    sys.exit(main())
