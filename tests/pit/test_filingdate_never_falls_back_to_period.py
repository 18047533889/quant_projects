# -*- coding: utf-8 -*-
"""R24-116..118 + R24-244: FilingDate.resolve() must NEVER fall back to the
report PERIOD end date — period_time is not knowledge_time.  A missing filing
timestamp fails (UNKNOWN), never resolves to 2024-03-31 for a Q1 report."""
from __future__ import annotations

import pandas as pd
import pytest

from factor_engine.ir.types import FilingDate


def test_filing_date_resolves_from_filing_column() -> None:
    row = pd.Series({
        "filing_date": pd.Timestamp("2024-04-30"),
        "report_period_end_date": pd.Timestamp("2024-03-31"),
    })
    out = FilingDate().resolve(row)
    assert out == pd.Timestamp("2024-04-30")


def test_filing_date_rejects_period_end_fallback() -> None:
    # R24-244 golden: ReportPeriodEndDate=2024-03-31, filing_date missing → must
    # fail, never resolve to 2024-03-31.
    row = pd.Series({
        "report_period_end_date": pd.Timestamp("2024-03-31"),
    })
    with pytest.raises((ValueError, KeyError, NotImplementedError)):
        FilingDate().resolve(row)


def test_report_date_column_is_no_longer_a_filing_fallback() -> None:
    # R24-116: report_date / ReportPeriodEndDate were removed as filing
    # knowledge-time fallbacks.
    row = pd.Series({
        "report_date": pd.Timestamp("2024-03-31"),
        "ReportPeriodEndDate": pd.Timestamp("2024-03-31"),
    })
    with pytest.raises((ValueError, KeyError, NotImplementedError)):
        FilingDate().resolve(row)


def test_filing_date_is_knowledge_kind() -> None:
    from factor_engine.ir.types import KnowledgeAvailabilityExpr

    assert isinstance(FilingDate(), KnowledgeAvailabilityExpr)
