# -*- coding: utf-8 -*-
"""Focused closure tests for MI identities and spectral canonical authority."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import cleaned_operators.nonlinear_dependence  # noqa: F401
import cleaned_operators.spectral_ext  # noqa: F401
from backend.operator_errors import OperatorParameterError
from cleaned_operators.operator_surface import extended_only_canonicals
from cleaned_operators.registry import OperatorRegistry


def _op(name: str):
    op = OperatorRegistry.get(name, "pandas_numpy")
    assert op is not None
    return op


def _frames(n: int = 80) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(23)
    index = pd.bdate_range("2024-01-02", periods=n)
    x = pd.DataFrame(rng.normal(size=(n, 2)), index=index, columns=["A", "B"])
    y = pd.DataFrame(x.to_numpy() + 0.2 * rng.normal(size=(n, 2)), index=index, columns=x.columns)
    return x, y


def test_mi_unit_split_and_compat_alias():
    x, y = _frames()
    nats = _op("ts_mutual_information_nats")
    norm = _op("ts_normalized_mutual_information")
    assert nats.metadata.output_unit == "nats"
    assert norm.metadata.output_unit is None
    assert "unit:ratio" in norm.metadata.tags
    assert "normalized" not in nats.metadata.param_names
    assert "normalized" not in norm.metadata.param_names
    assert OperatorRegistry.resolve_canonical("ts_mutual_information") == "ts_normalized_mutual_information"
    assert "ts_mutual_information" not in extended_only_canonicals()

    raw = nats.calculate(x, y, window=40)
    scaled = norm.calculate(x, y, window=40)
    assert np.isfinite(raw.to_numpy(dtype=float)).any()
    finite = scaled.to_numpy(dtype=float)
    assert np.nanmin(finite) >= 0.0
    assert np.nanmax(finite) <= 1.0


@pytest.mark.parametrize(
    "canonical",
    [
        "ts_distance_corr",
        "ts_distance_cov",
        "ts_mutual_information_nats",
        "ts_normalized_mutual_information",
        "ts_lagged_mutual_information",
    ],
)
def test_dependence_window_must_cover_min_periods(canonical: str):
    x, y = _frames()
    with pytest.raises((OperatorParameterError, ValueError), match="window"):
        _op(canonical).calculate(x, y, window=10, min_periods=11)


def test_spectral_legacy_name_is_alias_not_duplicate_canonical():
    canonical = _op("ts_return_spectral_entropy")
    legacy = _op("ts_spectral_entropy")
    assert legacy is canonical
    assert OperatorRegistry.resolve_canonical("ts_spectral_entropy") == "ts_return_spectral_entropy"
    assert "ts_spectral_entropy" not in extended_only_canonicals()
    assert canonical.metadata.input_units == {"x": "return_decimal"}


def test_spectral_runtime_does_not_guess_typed_source_from_values():
    index = pd.bdate_range("2024-01-02", periods=80)
    price_like = pd.DataFrame(np.exp(np.linspace(4.0, 5.0, 80)), index=index, columns=["A"])
    return_like = pd.DataFrame(np.sin(np.arange(80)), index=index, columns=["A"])

    # Typed IR / metadata is authoritative. Numeric shape heuristics must not
    # reject values after the source concept has already been checked upstream.
    ret_out = _op("ts_return_spectral_entropy").calculate(price_like, window=60)
    level_out = _op("ts_detrended_level_spectral_entropy").calculate(return_like, window=60)
    assert ret_out.shape == price_like.shape
    assert level_out.shape == return_like.shape
