from __future__ import annotations

import json
from pathlib import Path

from api.operator_registry import build_dsl_allowlist
from backend.polars_long_policy import classify_plan_op
from cleaned_operators import load_all
from cleaned_operators.edge_requirements import missing_edge_dimensions
from cleaned_operators.operator_surface import (
    classify_canonical,
    unclassified_canonicals,
)
from cleaned_operators.registry import OperatorRegistry


def test_surface_is_fail_closed() -> None:
    assert classify_canonical("brand_new_unreviewed_operator") == "unclassified"
    assert "brand_new_unreviewed_operator" not in build_dsl_allowlist()


def test_runtime_registry_has_no_unreviewed_canonicals() -> None:
    load_all()
    assert unclassified_canonicals(OperatorRegistry.list_canonical()) == ()


def test_problematic_names_are_not_daily_dsl() -> None:
    allowed = build_dsl_allowlist()
    for name in (
        "causal_bfill", "ACF", "Mode", "autocorr", "pacf", "max_drawdown",
        "sharpe_ratio", "sem", "lasso", "ridge", "regress", "residual",
        "r_squared", "constant",
    ):
        assert name not in allowed
    assert classify_canonical("causal_bfill") == "unsafe"


def test_recursive_operators_are_not_polars_native() -> None:
    for name in (
        "ema", "RSI_WILDER", "ATR_WILDER", "MACD", "KAMA", "TRIX", "ADX", "ADXR"
    ):
        assert classify_plan_op(name) == "stateful"


def test_primitive_evidence_uses_final_canonical_names() -> None:
    old = {"clip", "ts_delay", "safe_div_null", "ts_ema", "ts_regression", "ts_decay_linear"}
    paths = sorted((Path(__file__).resolve().parents[1]).rglob("primitive_verified.json"))
    assert paths
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for value in payload.values():
            if isinstance(value, list) and all(isinstance(x, str) for x in value):
                assert old.isdisjoint(value)
        assert old.isdisjoint((payload.get("operators") or {}).keys())


def test_ieee_edge_policy_is_fail_closed_for_statistics() -> None:
    evidence = {"duckdb_nan_edge_verified": [], "duckdb_inf_edge_verified": []}
    assert missing_edge_dimensions("ts_std", evidence) == {"nan", "inf"}
    assert missing_edge_dimensions("abs", evidence) == set()
