"""Formal eight-panel contract for Polars holder class entropy."""
from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

import factor_engine.cleaned_operators.shareholder.churn_network  # noqa: F401
from factor_engine.cleaned_operators.common.polars_holder import HolderClassEntropyNative
from factor_engine.cleaned_operators.registry import OperatorRegistry


def _panels():
    timestamps = pd.date_range("2024-01-02", periods=4, freq="D")
    rows = (
        [1.0, 3.0, 1.0, -1.0],
        [1.0, 1.0, np.nan, 2.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    )
    return tuple(pl.DataFrame({"timestamp": timestamps, "A": row}) for row in rows)


def test_holder_class_entropy_has_exact_public_eight_panel_contract():
    operator = OperatorRegistry.get("holder_class_entropy", backend="polars")
    assert operator is not None
    assert operator.metadata.param_names == [f"s{i}" for i in range(1, 9)]

    panels = _panels()
    result = operator.calculate(*panels)

    q = -(0.75 * np.log(0.75) + 0.25 * np.log(0.25)) / np.log(2.0)
    np.testing.assert_allclose(result["A"][:2].to_numpy(), [1.0, q])
    assert np.isnan(result["A"][2])
    assert np.isnan(result["A"][3])
    assert result["timestamp"].to_list() == panels[0]["timestamp"].to_list()


def test_holder_class_entropy_rejects_misaligned_panel_axis():
    panels = list(_panels())
    panels[-1] = panels[-1].reverse()
    with pytest.raises(ValueError, match="different PanelIdentity"):
        HolderClassEntropyNative()._calculate_series(*panels)
