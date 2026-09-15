# -*- coding: utf-8 -*-
"""Focused closure tests for MI identities and spectral canonical authority."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import factor_engine.cleaned_operators.nonlinear_dependence  # noqa: F401
import factor_engine.cleaned_operators.spectral_ext  # noqa: F401
from factor_engine.backend.operator_errors import OperatorParameterError
from factor_engine.cleaned_operators.operator_surface import extended_only_canonicals
from factor_engine.cleaned_operators.registry import OperatorRegistry


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
    mask = np.isfinite(raw.to_numpy(dtype=float)) & np.isfinite(finite)
    assert mask.any()
    assert not np.allclose(raw.to_numpy(dtype=float)[mask], finite[mask])
    with pytest.raises(OperatorParameterError, match="normalized"):
        # Compatibility name defaults to the normalized identity.  The former
        # unit-changing switch must fail loudly and direct callers to the nats
        # canonical rather than silently returning a normalized result.
        _op("ts_mutual_information").calculate(x, y, window=40, normalized=False)


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


def test_spectral_legacy_name_is_live_typed_canonical_with_same_semantics():
    canonical = _op("ts_return_spectral_entropy")
    legacy = _op("ts_spectral_entropy")
    # The literal legacy spelling remains a live canonical because the typed-IR
    # spectral gate is keyed by resolved canonical name.  Both registrations
    # deliberately share the return-spectrum kernel and input contract.
    assert legacy is not canonical
    assert OperatorRegistry.resolve_canonical("ts_spectral_entropy") == "ts_spectral_entropy"
    assert "ts_spectral_entropy" in extended_only_canonicals()
    assert canonical.metadata.input_units == {"x": "return_decimal"}
    assert legacy.metadata.input_units == canonical.metadata.input_units


def test_spectral_runtime_does_not_guess_typed_source_from_values():
    index = pd.bdate_range("2024-01-02", periods=80)
    price_like = pd.DataFrame(np.exp(np.linspace(4.0, 5.0, 80)), index=index, columns=["A"])
    return_like = pd.DataFrame(np.sin(np.arange(80)), index=index, columns=["A"])

    # Direct callers must state the already-established semantic kind. Numeric
    # shape is never guessed, and values are not reclassified heuristically.
    with pytest.raises(ValueError, match="input_kind"):
        _op("ts_return_spectral_entropy").calculate(price_like, window=60)
    with pytest.raises(ValueError, match="input_kind"):
        _op("ts_detrended_level_spectral_entropy").calculate(return_like, window=60)
    ret_out = _op("ts_return_spectral_entropy").calculate(
        price_like, window=60, input_kind="ReturnDecimal")
    level_out = _op("ts_detrended_level_spectral_entropy").calculate(
        return_like, window=60, input_kind="PriceContinuous")
    assert ret_out.shape == price_like.shape
    assert level_out.shape == return_like.shape
