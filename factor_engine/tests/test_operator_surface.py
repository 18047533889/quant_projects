from __future__ import annotations

import json
from pathlib import Path

import pytest

from api.operator_registry import build_dsl_allowlist
from backend.cleaned_bridge import build_cleaned_dsl_allowlist
from backend.polars_long_policy import classify_plan_op
from cleaned_operators import load_all
from cleaned_operators.edge_requirements import missing_edge_dimensions
from cleaned_operators.operator_surface import classify_canonical, unclassified_canonicals
from cleaned_operators.registry import OperatorRegistry
from research_operators import build_research_dsl_allowlist, build_unsafe_dsl_allowlist
from research_tools.registry import ResearchToolRegistry


def test_daily_surface_keeps_factor_primitives() -> None:
    public = build_dsl_allowlist()
    for name in ("col", "abs", "rank", "zscore", "ts_mean", "ts_sum", "ts_corr", "where", "power"):
        assert name in public


@pytest.mark.parametrize(
    "name",
    [
        "Lead", "next", "bfill", "FillBackward", "fillna_interpolate",
        "shuffle", "sample", "rand_normal", "rand_uniform",
        "jarque_bera_test", "ttest_one_sample", "pdf_normal", "pca",
        "mat_inverse", "fft", "wavelet", "filter_lowpass",
        "inv", "reciprocal", "fmax", "fmin", "sqr", "cube",
        "cumulative_max", "cumulative_mean", "cumulative_min",
        "causal_bfill", "ACF", "Mode", "autocorr", "pacf",
        "max_drawdown", "sharpe_ratio", "sem", "lasso", "ridge",
        "regress", "residual", "r_squared", "constant", "vwap",
        "ttm", "quarter", "yoy", "avg2", "MOM", "ROC", "KAMA",
    ],
)
def test_removed_names_are_not_in_public_daily_dsl(name: str) -> None:
    assert name not in build_dsl_allowlist()


def test_research_tools_are_explicit_and_unsafe_is_separate() -> None:
    public = build_dsl_allowlist()
    research = build_research_dsl_allowlist()
    unsafe = build_unsafe_dsl_allowlist()
    for name in (
        "jarque_bera_test", "pca", "fft", "ACF", "max_drawdown",
        "causal_linear_extrapolate", "dropna", "fillna", "expanding_mean",
    ):
        assert name in research
        assert name in ResearchToolRegistry.list_canonical()
        assert name not in public
    for deleted in ("ttm", "quarter", "yoy", "avg2"):
        assert deleted not in research
        assert deleted not in public
    for removed in ("next", "bfill", "causal_bfill", "rand_normal"):
        assert removed not in unsafe
        assert removed not in research


def test_production_all_runtime_excludes_research_tools() -> None:
    all_runtime = build_cleaned_dsl_allowlist(surface="all")
    assert "cube" in all_runtime
    for research_name in ("fft", "pca", "fillna", "expanding_mean"):
        assert research_name not in all_runtime
    for removed in ("Lead", "next", "bfill", "causal_bfill", "shuffle", "rand_normal"):
        assert removed not in all_runtime


def test_duplicate_canonicals_are_merged_to_one_runtime() -> None:
    load_all()
    expected = {
        "inv": "inverse",
        "reciprocal": "inverse",
        "fmax": "maximum",
        "fmin": "minimum",
        "sqr": "square",
        "WMA": "ts_decay_linear",
        "log_returns": "ts_log_return",
    }
    canonicals = set(OperatorRegistry.list_canonical())
    for alias, canonical in expected.items():
        assert OperatorRegistry._aliases.get(alias) == canonical
        assert alias not in canonicals
        assert OperatorRegistry.get(canonical) is not None
    for expanding in ("expanding_max", "expanding_mean", "expanding_min"):
        assert expanding not in canonicals
        assert expanding in ResearchToolRegistry.list_canonical()


def test_api_dynamic_import_cannot_bypass_public_surface() -> None:
    import api

    for name in ("fft", "next", "causal_bfill", "constant", "fillna", "vwap"):
        with pytest.raises(AttributeError):
            getattr(api, name)
    assert callable(getattr(api, "ts_mean"))


def test_surface_is_fail_closed() -> None:
    assert classify_canonical("brand_new_unreviewed_operator") == "unclassified"
    assert "brand_new_unreviewed_operator" not in build_dsl_allowlist()


def test_runtime_registry_has_no_unreviewed_canonicals() -> None:
    load_all()
    assert unclassified_canonicals(OperatorRegistry.list_canonical()) == ()


def test_retained_recursive_operators_are_stateful() -> None:
    for name in ("ema", "RSI_WILDER", "ATR_WILDER", "MACD_line", "ADX"):
        assert classify_plan_op(name) == "stateful"
    for removed in ("KAMA", "TRIX", "ADXR"):
        assert removed not in OperatorRegistry.list_canonical()


def test_primitive_evidence_uses_final_canonical_names() -> None:
    old = {"cap", "delay", "safe_div", "ema", "ts_regression", "decay_linear"}
    paths = sorted((Path(__file__).resolve().parents[1]).rglob("primitive_verified.json"))
    assert paths
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for value in payload.values():
            if isinstance(value, list) and all(isinstance(x, str) for x in value):
                assert old.isdisjoint(value)
        assert old.isdisjoint((payload.get("operators") or {}).keys())


def test_ieee_edge_policy_is_fail_closed_for_statistics() -> None:
    load_all()
    evidence = {"duckdb_nan_edge_verified": [], "duckdb_inf_edge_verified": []}
    assert missing_edge_dimensions("ts_std", evidence) == {"nan", "inf"}
    assert missing_edge_dimensions("cs_mean", evidence) == {"nan", "inf"}
    assert missing_edge_dimensions("c_mean", evidence) == {"nan", "inf"}  # alias
    assert missing_edge_dimensions("abs", evidence) == set()
