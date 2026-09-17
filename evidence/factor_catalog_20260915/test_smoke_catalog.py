from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pandas as pd

from smoke_catalog import bind_fields, collect_field_refs, summarize_result


@dataclass
class _Spec:
    dataset: str
    table: str
    source_name: str
    unit: str = "ratio"
    source_unit: str = "ratio"
    scale_to_canonical: float = 1.0


def test_collect_field_refs_uses_ast_not_projection_string():
    from factor_engine.api.dsl_parser import parse_expr

    expr = parse_expr("add(ts_mean(ret, 5), close)", surface="compat_research")
    assert [(x.name, getattr(x, "table", None)) for x in collect_field_refs(expr)] == [
        ("close", "StockDailyBarAdj"), ("ret", "StockDailyBarAdj")
    ]


def test_bind_fields_preserves_cross_table_source_identity():
    from factor_engine.api.dsl_parser import parse_expr

    expr = parse_expr("add(ret, turnover_ratio)", surface="compat_research")

    def resolver(leaf, market, strict):
        assert market == "ashare" and strict is True
        if leaf.name == "ret":
            spec = _Spec("ashare_stock_daily_adj", "StockDailyBarAdj", "Return")
        else:
            spec = _Spec("ashare_stock_valuation_daily", "StockValuationDaily", "TurnoverRatio")
        return SimpleNamespace(spec=spec)

    bindings, failures = bind_fields(expr, resolver=resolver)
    assert failures == []
    assert len(bindings) == 2
    price = next(x for x in bindings if x["dataset"] == "ashare_stock_daily_adj")
    assert price["input"] == "ret"
    assert price["column"] == "Return"
    cross = next(x for x in bindings if x["dataset"] == "ashare_stock_valuation_daily")
    assert cross["input"].startswith("__fe_source_ref_v1__")
    assert cross["table"] == "StockValuationDaily"
    assert cross["column"] == "TurnoverRatio"


def test_summarize_result_is_bounded_and_counts_finite_values():
    index = pd.MultiIndex.from_product(
        [pd.to_datetime(["2026-01-05", "2026-01-06"]), ["000001.SZ"]],
        names=["timestamp", "instrument"],
    )
    out = summarize_result({"result": pd.Series([1.0, float("nan")], index=index)})
    assert out["value_count"] == 2
    assert out["finite_count"] == 1
    assert len(out["result_hash"]) == 64
