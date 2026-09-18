from __future__ import annotations

import json
from pathlib import Path

import pytest

from factor_engine.api.operator_registry import build_dsl_allowlist
from factor_engine.backend.cleaned_bridge import build_cleaned_dsl_allowlist
from factor_engine.backend.polars_long_policy import classify_plan_op
from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.edge_requirements import missing_edge_dimensions
from factor_engine.cleaned_operators.operator_surface import classify_canonical, unclassified_canonicals
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.research_operators import build_research_dsl_allowlist, build_unsafe_dsl_allowlist
from factor_engine.research_tools.registry import ResearchToolRegistry


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
        "ttm", "quarter", "yoy", "MOM", "ROC",
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
        "causal_linear_extrapolate", "dropna", "expanding_mean",
    ):
        assert name in research
        assert name in ResearchToolRegistry.list_canonical()
        assert name not in public
    for deleted in ("ttm", "quarter", "yoy"):
        assert deleted not in research
        assert deleted not in public
    for promoted in ("avg2", "fillna"):
        assert promoted in public
        assert promoted not in research
    for removed in ("next", "bfill", "causal_bfill", "rand_normal"):
        assert removed not in unsafe
        assert removed not in research


def test_production_all_runtime_excludes_research_tools() -> None:
    all_runtime = build_cleaned_dsl_allowlist(surface="all")
    assert "cube" in all_runtime
    for research_name in ("fft", "pca", "expanding_mean"):
        assert research_name not in all_runtime
    assert "fillna" in all_runtime
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
    assert "WMA" in canonicals
    assert OperatorRegistry._aliases.get("WMA") is None
    assert OperatorRegistry._aliases.get("ts_wma") == "WMA"
    assert OperatorRegistry._aliases.get("wma") == "WMA"


def test_api_dynamic_import_cannot_bypass_public_surface() -> None:
    import factor_engine.api as api

    for name in ("fft", "next", "causal_bfill", "constant", "vwap"):
        with pytest.raises(AttributeError):
            getattr(api, name)
    assert callable(getattr(api, "ts_mean"))
    assert callable(getattr(api, "fillna"))


def test_surface_is_fail_closed() -> None:
    assert classify_canonical("brand_new_unreviewed_operator") == "unclassified"
    assert "brand_new_unreviewed_operator" not in build_dsl_allowlist()


def test_runtime_registry_has_no_unreviewed_canonicals() -> None:
    load_all()
    actual = unclassified_canonicals(OperatorRegistry.list_canonical())
    assert actual == (), f"unreviewed runtime canonicals: {actual!r}"
    polars_only_legacy = {
        "ts_deviation_from_mean", "ts_jump_bipower", "ts_lag1_autocorr",
        "ts_returns",
    }
    assert all(classify_canonical(name) == "legacy" for name in polars_only_legacy)
    assert polars_only_legacy.isdisjoint(build_dsl_allowlist())


def test_retained_recursive_operators_are_stateful() -> None:
    for name in (
        "ema",
        "RSI_WILDER",
        "ATR_WILDER",
        "MACD_line",
        "ADX",
        "KAMA",
        "Supertrend",
        "SupertrendDirection",
        "PSAR",
    ):
        assert classify_plan_op(name) == "stateful"
    for removed in ("TRIX", "ADXR"):
        assert removed not in OperatorRegistry.list_canonical()


def test_primitive_evidence_uses_final_canonical_names() -> None:
    old = {"cap", "delay", "safe_div", "ema", "ts_regression", "decay_linear"}
    evidence_path = Path(__file__).resolve().parents[2] / "evidence" / "primitive_verified.json"
    assert evidence_path.is_file()
    paths = [evidence_path]
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
