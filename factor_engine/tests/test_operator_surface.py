from __future__ import annotations

import pytest

from api.operator_registry import build_dsl_allowlist
from backend.cleaned_bridge import build_cleaned_dsl_allowlist
from cleaned_operators import load_all
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
    assert "jarque_bera_test" not in public
    assert "next" in unsafe
    assert "bfill" in unsafe
    assert "rand_normal" in unsafe
    assert "next" not in research


def test_all_runtime_surface_remains_available_for_compatibility() -> None:
    all_runtime = build_cleaned_dsl_allowlist(surface="all")
    assert "fft" in all_runtime
    assert "next" in all_runtime
    assert "cube" in all_runtime


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
    assert callable(getattr(api, "ts_mean"))
