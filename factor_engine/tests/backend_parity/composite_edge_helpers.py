# -*- coding: utf-8
"""Composite edge case 定义（与 evidence/composite_verified.json 同步）。"""
from __future__ import annotations

from tests.backend_parity.composite_reference_helpers import CompositeReferenceCase

COMPOSITE_EDGE_CASES: tuple[CompositeReferenceCase, ...] = (
    CompositeReferenceCase("MOM", ("close",), window=3),
    CompositeReferenceCase("ROC", ("close",), window=3),
    CompositeReferenceCase("BollingerBands", ("close",), window=3),
    CompositeReferenceCase("BollingerUpper", ("close",), window=3, calc_kwargs={"std_dev": 2.0}),
    CompositeReferenceCase("BollingerLower", ("close",), window=3, calc_kwargs={"std_dev": 2.0}),
    CompositeReferenceCase("DPO", ("close",), window=4),
    CompositeReferenceCase("WilliamsR", ("high", "low", "close"), window=3),
    CompositeReferenceCase("StochasticK", ("high", "low", "close"), window=3),
    CompositeReferenceCase("StochasticD", ("high", "low", "close"), window=3),
    CompositeReferenceCase("OBV", ("close", "volume")),
    CompositeReferenceCase("operating_margin", ("operating_income", "revenue")),
    CompositeReferenceCase("current_ratio", ("current_assets", "current_liabilities")),
    CompositeReferenceCase("quick_ratio", ("current_assets", "inventory", "current_liabilities")),
    CompositeReferenceCase("debt_to_equity", ("total_debt", "total_equity")),
    CompositeReferenceCase("real_turnover_rate", ("volume", "float_shares")),
    CompositeReferenceCase("micro_spread", ("high", "low", "close")),
)
