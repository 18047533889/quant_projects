# -*- coding: utf-8 -*-
"""Native Polars daily backends survive overhaul cleanup and match pandas."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_engine.backend.cleaned_bridge import ensure_cleaned_loaded


@pytest.fixture(scope="module", autouse=True)
def _load():
    ensure_cleaned_loaded()


_DAILY_NATIVE = (
    "signed_sqrt",
    "ts_log_return",
    "ts_rank",
    "ts_sharpe",
    "cs_std",
    "cs_mad",
    "cs_mad_zscore",
    "normalize",
    "winsorize",
    "group_neutralize",
    "group_normalize",
    "group_winsorize",
)


@pytest.mark.parametrize("name", _DAILY_NATIVE)
def test_daily_native_polars_backend_registered(name: str) -> None:
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    backends = OperatorRegistry.backends_for(name)
    assert "polars" in backends, (name, backends)
    assert "sql" in backends, (name, backends)
    meta = (OperatorRegistry.catalog().get(name) or {}).get("backend_meta") or {}
    expected_kind = (
        "polars_eager_native"
        if name in {
            "cs_std", "cs_mad", "cs_mad_zscore", "normalize", "winsorize",
            "group_neutralize", "group_normalize", "group_winsorize",
        }
        else "expression_native"
    )
    assert meta.get("polars", {}).get("execution_kind") == expected_kind
    assert bool(meta.get("polars", {}).get("materializes_full_panel")) == (
        expected_kind == "polars_eager_native"
    )


def test_signed_sqrt_polars_matches_pandas() -> None:
    pl = pytest.importorskip("polars")
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    pdf = pd.DataFrame({"a": [1.0, 4.0, 9.0], "b": [-1.0, -4.0, 0.0]})
    expected = OperatorRegistry.get("signed_sqrt").calculate(pdf)
    actual = (
        OperatorRegistry.get("signed_sqrt", "polars")
        .calculate(pl.from_pandas(pdf))
        .to_pandas()
    )
    np.testing.assert_allclose(actual.to_numpy(), expected.to_numpy(), equal_nan=True)


def test_normalize_polars_matches_pandas() -> None:
    pl = pytest.importorskip("polars")
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    pdf = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0], "c": [5.0, 6.0]})
    expected = OperatorRegistry.get("normalize").calculate(pdf)
    actual = (
        OperatorRegistry.get("normalize", "polars")
        .calculate(pl.from_pandas(pdf))
        .to_pandas()
    )
    np.testing.assert_allclose(actual.to_numpy(), expected.to_numpy(), equal_nan=True, rtol=1e-6)


def test_daily_dsl_does_not_leak_research_ops() -> None:
    from factor_engine.backend.cleaned_bridge import build_cleaned_dsl_allowlist
    from factor_engine.cleaned_operators.operator_surface import RESEARCH_ONLY_CANONICALS

    daily = set(build_cleaned_dsl_allowlist(set(), surface="daily"))
    assert not (daily & RESEARCH_ONLY_CANONICALS)
