from __future__ import annotations

import json
from pathlib import Path

import pytest

from api.operator_registry import build_dsl_allowlist
from backend.cleaned_bridge import build_cleaned_dsl_allowlist
from backend.polars_long_policy import classify_plan_op
from cleaned_operators import load_all
from cleaned_operators.edge_requirements import missing_edge_dimensions
from cleaned_operators.operator_surface import (
    classify_canonical,
    unclassified_canonicals,
)
from cleaned_operators.registry import OperatorRegistry
from research_operators import (
    build_research_dsl_allowlist,
    build_unsafe_dsl_allowlist,
)


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
        "regress", "residual", "r_squared", "constant",
    ],
)
def test_removed_names_are_not_in_public_daily_dsl(name: str) -> None:
    assert name not in build_dsl_allowlist()


def test_research_tools_are_explicit_and_unsafe_is_separate() -> None:
    public = build_dsl_allowlist()
    research = build_research_dsl_allowlist()
    unsafe = build_unsafe_dsl_allowlist()
    assert "jarque_bera_test" in research
    assert "pca" in research
    assert "fft" in research
    assert "ACF" in research
    assert "max_drawdown" in research
    assert "causal_linear_extrapolate" in research
    assert "dropna" in research
    assert "ttm" in research
    assert "jarque_bera_test" not in public
    for removed in ("next", "bfill", "causal_bfill", "rand_normal"):
        assert removed not in unsafe
        assert removed not in research


def test_removed_unsafe_runtime_is_not_available_for_compatibility() -> None:
    all_runtime = build_cleaned_dsl_allowlist(surface="all")
    assert "fft" in all_runtime
    assert "cube" in all_runtime
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
        "cumulative_max": "expanding_max",
        "cumulative_mean": "expanding_mean",
        "cumulative_min": "expanding_min",
    }
    canonicals = set(OperatorRegistry.list_canonical())
    for alias, canonical in expected.items():
        assert OperatorRegistry._aliases.get(alias) == canonical
        assert alias not in canonicals
        assert OperatorRegistry.get(canonical) is not None


def test_api_dynamic_import_cannot_bypass_public_surface() -> None:
    import api

    with pytest.raises(AttributeError):
        getattr(api, "fft")
    with pytest.raises(AttributeError):
        getattr(api, "next")
    with pytest.raises(AttributeError):
        getattr(api, "causal_bfill")
    with pytest.raises(AttributeError):
        getattr(api, "constant")
    assert callable(getattr(api, "ts_mean"))


def test_surface_is_fail_closed() -> None:
    assert classify_canonical("brand_new_unreviewed_operator") == "unclassified"
    assert "brand_new_unreviewed_operator" not in build_dsl_allowlist()


def test_runtime_registry_has_no_unreviewed_canonicals() -> None:
    load_all()
    assert unclassified_canonicals(OperatorRegistry.list_canonical()) == ()


def test_recursive_operators_are_not_polars_native() -> None:
    for name in (
        "ema", "RSI_WILDER", "ATR_WILDER", "MACD", "KAMA", "TRIX", "ADX", "ADXR"
    ):
        assert classify_plan_op(name) == "stateful"


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
    evidence = {"duckdb_nan_edge_verified": [], "duckdb_inf_edge_verified": []}
    assert missing_edge_dimensions("ts_std", evidence) == {"nan", "inf"}
    assert missing_edge_dimensions("abs", evidence) == set()
