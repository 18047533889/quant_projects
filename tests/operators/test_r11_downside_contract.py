# -*- coding: utf-8 -*-
"""R11 downside/upside deviation type-contract regression tests.

The natural domain of ``ts_downside_deviation`` / ``ts_upside_deviation`` is a
RETURN (signed) series measured against a same-unit MAR target return — not a
raw price with target=0 (which yields ~0 for positive prices and is a dead
factor).  These tests pin:

- the metadata contract: ``x`` accepts return-or-same-unit numeric (NOT
  price-only), ``target`` is a same-unit-as-x MAR threshold, and the output
  unit is ``same_as:x``;
- the kernel semantics on synthetic returns (downside = sqrt(mean(min(r-t,0)^2)),
  upside = sqrt(mean(max(r-t,0)^2)));
- a non-zero MAR target shifts the value as expected (higher target -> higher
  downside deviation for the same negative returns);
- a raw positive price series still computes (contract is metadata-level; the
  math runs) while the description reflects the return/MAR interpretation.

The operator classes are imported directly from their module (the module
registers its operators on import), so this file is not blocked by other
workstreams' in-flight edits to unrelated canonical families.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_engine.cleaned_operators.downside_risk import TsDownsideDeviation, TsUpsideDeviation


def _frame(values: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"A": np.asarray(values, dtype=float)})


# ---------------------------------------------------------------------------
# metadata contract: return-or-same-unit input + same-unit MAR target
# ---------------------------------------------------------------------------
def test_downside_upside_metadata_contract_is_return_mar():
    for cls in (TsDownsideDeviation, TsUpsideDeviation):
        meta = cls.metadata
        # x is return-or-same-unit numeric, NOT price-only.
        assert "return" in meta.input_units["x"], f"{cls.__name__} x -> {meta.input_units['x']!r}"
        assert "price" not in meta.input_units["x"], f"{cls.__name__} x must not be price-only"
        assert "return" in meta.compatible_units["x"], (
            f"{cls.__name__} compatible x must allow return, got "
            f"{meta.compatible_units['x']!r}"
        )
        assert "price" not in meta.compatible_units["x"], (
            f"{cls.__name__} compatible x must NOT be price-only, got "
            f"{meta.compatible_units['x']!r}"
        )
        # target is a MAR threshold in the same unit as x.
        assert meta.input_units["target"] == "same_unit_as:x", (
            f"{cls.__name__} target -> {meta.input_units['target']!r}"
        )
        assert meta.compatible_units["target"] == ("same_unit_as:x",), (
            f"{cls.__name__} compatible target -> {meta.compatible_units['target']!r}"
        )
        # output inherits x's unit.
        assert meta.output_unit == "same_as:x", (
            f"{cls.__name__} output_unit -> {meta.output_unit!r}"
        )
        # the description reflects the return/MAR interpretation.
        assert "MAR" in meta.description or "minimal acceptable return" in meta.description, (
            f"{cls.__name__} description must mention MAR, got {meta.description!r}"
        )


# ---------------------------------------------------------------------------
# semantics on a synthetic return series vs a zero MAR target
# ---------------------------------------------------------------------------
def test_downside_deviation_semantics_returns_vs_zero_target():
    op = TsDownsideDeviation()
    r = np.array([-0.02, 0.01, -0.03, 0.00, 0.02])
    out = op.calculate(_frame(r), window=5, target=0.0)
    below = np.minimum(r - 0.0, 0.0)
    expected = float(np.sqrt(np.mean(below * below)))
    assert out["A"].iloc[-1] == pytest.approx(expected)
    # min_periods=2: a single finite observation must stay NaN.
    assert np.isnan(out["A"].iloc[0])
    # the earlier 2-obs window agrees with the hand-computed value too.
    below2 = np.minimum(r[:2] - 0.0, 0.0)
    expected2 = float(np.sqrt(np.mean(below2 * below2)))
    assert out["A"].iloc[1] == pytest.approx(expected2)


def test_upside_deviation_semantics_returns_vs_zero_target():
    op = TsUpsideDeviation()
    r = np.array([-0.02, 0.01, -0.03, 0.00, 0.02])
    out = op.calculate(_frame(r), window=5, target=0.0)
    above = np.maximum(r - 0.0, 0.0)
    expected = float(np.sqrt(np.mean(above * above)))
    assert out["A"].iloc[-1] == pytest.approx(expected)
    assert np.isnan(out["A"].iloc[0])


# ---------------------------------------------------------------------------
# a non-zero MAR target shifts the deviation as expected
# ---------------------------------------------------------------------------
def test_downside_deviation_higher_mar_raises_downside():
    op = TsDownsideDeviation()
    r = np.array([-0.02, 0.01, -0.03, 0.00, 0.02])
    d0 = op.calculate(_frame(r), window=5, target=0.0)["A"].iloc[-1]
    dmar = op.calculate(_frame(r), window=5, target=0.005)["A"].iloc[-1]
    # a higher MAR makes the same negative returns sit further below target.
    assert dmar > d0
    # exact value against numpy: sqrt(mean(min(r - 0.005, 0)^2)).
    below = np.minimum(r - 0.005, 0.0)
    expected = float(np.sqrt(np.mean(below * below)))
    assert dmar == pytest.approx(expected)


def test_upside_deviation_higher_mar_raises_upside():
    op = TsUpsideDeviation()
    r = np.array([-0.02, 0.01, -0.03, 0.00, 0.02])
    u0 = op.calculate(_frame(r), window=5, target=0.0)["A"].iloc[-1]
    umar = op.calculate(_frame(r), window=5, target=-0.005)["A"].iloc[-1]
    # a LOWER target bar lets more of the series count as upside.
    assert umar > u0
    above = np.maximum(r - (-0.005), 0.0)
    expected = float(np.sqrt(np.mean(above * above)))
    assert umar == pytest.approx(expected)


# ---------------------------------------------------------------------------
# contract is metadata-level: a raw positive price still computes, but the
# description reflects the return/MAR interpretation
# ---------------------------------------------------------------------------
def test_raw_positive_price_still_computes_but_interpretation_is_return_mar():
    op = TsDownsideDeviation()
    price = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
    out = op.calculate(_frame(price), window=5, target=0.0)
    # the math runs; min(price - 0, 0) == 0 for all-positive prices -> ~0.
    assert np.isfinite(out["A"].iloc[-1])
    assert out["A"].iloc[-1] == pytest.approx(0.0)
    # the docstring/description carries the return/MAR reading.
    for cls in (TsDownsideDeviation, TsUpsideDeviation):
        assert "MAR" in cls.metadata.description or "minimal acceptable return" in cls.metadata.description
