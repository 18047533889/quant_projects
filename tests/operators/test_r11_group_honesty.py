# -*- coding: utf-8 -*-
"""R11 honesty fixes for the group/regression operators.

* ISSUE 1 (P0): ``hierarchical_group_neutralize`` (polars backend) must keep
  unknown/missing group or subgroup labels NaN instead of manufacturing them
  into a perfectly-neutralized 0.
* ISSUE 2 (P2): ``cs_robust_resid`` is genuinely trimmed-OLS (trim extreme x,
  then ordinary least squares) — not a robust regression — so the canonical
  name is now ``cs_trimmed_ols_resid`` and ``cs_robust_resid`` is a deprecated
  alias that must keep resolving and return identical output.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pl = pytest.importorskip("polars")

from cleaned_operators import load_all
from cleaned_operators.registry import OperatorRegistry


@pytest.fixture(scope="module", autouse=True)
def _load_registry():
    load_all()


def _polars(frame: pd.DataFrame) -> pl.DataFrame:
    return pl.DataFrame({column: frame[column].to_numpy() for column in frame.columns})


# ---------------------------------------------------------------------------
# ISSUE 1 — unknown group/subgroup labels must stay NaN (not 0)
# ---------------------------------------------------------------------------
def test_unknown_float_group_label_stays_nan_polars() -> None:
    cols = list("ABCDEF")
    x = pd.DataFrame([[10.0, 0.0, 4.0, 8.0, 5.0, 9.0]], index=[0], columns=cols)
    # E and F carry a NaN group label -> their unknown key is made of the cell
    # alone, which the old implementation demeaned to 0 (mean == self).
    group = pd.DataFrame([[1.0, 1.0, 2.0, 2.0, np.nan, np.nan]], index=[0], columns=cols)
    subgroup = pd.DataFrame([["S", "S", "T", "T", "X", "Y"]], index=[0], columns=cols)
    op = OperatorRegistry.get("hierarchical_group_neutralize", backend="polars")
    assert op is not None, "polars backend missing"
    out = op.calculate(_polars(x), _polars(group), _polars(subgroup)).to_numpy()[0]
    # Valid composite keys are demeaned normally.
    np.testing.assert_allclose(out[:4], [5.0, -5.0, -2.0, 2.0], rtol=1e-9, atol=1e-9)
    # Unknown group cells keep their residual NaN — never a manufactured 0.
    assert np.isnan(out[4]) and np.isnan(out[5])


def test_unknown_string_group_label_stays_nan_polars() -> None:
    cols = list("ABCDEF")
    x = pd.DataFrame([[10.0, 0.0, 4.0, 8.0, 5.0, 9.0]], index=[0], columns=cols)
    # None and "" are NOT real membership labels; E (None) is alone in its
    # (invalid) key and must stay NaN, not become 0.
    group = pd.DataFrame([["G1", "G1", "G2", "G2", None, ""]], index=[0], columns=cols)
    subgroup = pd.DataFrame([["S", "S", "T", "T", "X", "Y"]], index=[0], columns=cols)
    op = OperatorRegistry.get("hierarchical_group_neutralize", backend="polars")
    assert op is not None, "polars backend missing"
    out = op.calculate(_polars(x), _polars(group), _polars(subgroup)).to_numpy()[0]
    np.testing.assert_allclose(out[:4], [5.0, -5.0, -2.0, 2.0], rtol=1e-9, atol=1e-9)
    assert np.isnan(out[4]) and np.isnan(out[5])


def test_unknown_subgroup_label_stays_nan_polars() -> None:
    cols = list("ABCD")
    x = pd.DataFrame([[10.0, 0.0, 4.0, 8.0]], index=[0], columns=cols)
    group = pd.DataFrame([["G1", "G1", "G2", "G2"]], index=[0], columns=cols)
    # C and D have a valid group but a missing subgroup -> residual must be NaN.
    subgroup = pd.DataFrame([["S", "S", None, None]], index=[0], columns=cols)
    op = OperatorRegistry.get("hierarchical_group_neutralize", backend="polars")
    assert op is not None, "polars backend missing"
    out = op.calculate(_polars(x), _polars(group), _polars(subgroup)).to_numpy()[0]
    np.testing.assert_allclose(out[:2], [5.0, -5.0], rtol=1e-9, atol=1e-9)
    assert np.isnan(out[2]) and np.isnan(out[3])


# ---------------------------------------------------------------------------
# ISSUE 2 — honest canonical name cs_trimmed_ols_resid, cs_robust_resid alias
# ---------------------------------------------------------------------------
def test_cs_trimmed_ols_resid_is_the_canonical() -> None:
    assert OperatorRegistry.resolve_canonical("cs_trimmed_ols_resid") == "cs_trimmed_ols_resid"
    assert "cs_trimmed_ols_resid" in OperatorRegistry._operators
    assert "cs_robust_resid" not in OperatorRegistry._operators  # now an alias
    op = OperatorRegistry.get("cs_trimmed_ols_resid", backend="pandas_numpy")
    assert op is not None
    assert op.metadata.name == "cs_trimmed_ols_resid"
    # both backends resolve under the honest canonical.
    assert {"pandas_numpy", "polars"} <= set(OperatorRegistry.backends_for("cs_trimmed_ols_resid"))


def test_cs_robust_resid_is_deprecated_alias() -> None:
    # the historical name still resolves — through the honest canonical.
    assert OperatorRegistry.resolve_canonical("cs_robust_resid") == "cs_trimmed_ols_resid"
    for backend in ("pandas_numpy", "polars"):
        op = OperatorRegistry.get("cs_robust_resid", backend=backend)
        assert op is not None, backend
        assert op.metadata.name == "cs_trimmed_ols_resid"
    # surface classification still holds for the deprecated spelling (the
    # pre-rename contract: daily-promoted, never an unclassified stray).
    from cleaned_operators.operator_surface import classify_canonical

    assert classify_canonical("cs_robust_resid") in {"daily", "extended"}
    # the new canonical is surface-registered too (so layer_governance's exact
    # classification check passes at load time).
    assert classify_canonical("cs_trimmed_ols_resid") in {"daily", "extended"}


def test_cs_robust_alias_and_canonical_output_identical() -> None:
    rng = np.random.default_rng(7)
    index = pd.date_range("2024-01-01", periods=5, freq="D")
    cols = [f"C{i}" for i in range(12)]  # >= min_breadth (10) for the breadth gate
    x = pd.DataFrame(rng.normal(0.5, 2.0, (5, 12)), index=index, columns=cols)
    y = x * 0.7 + rng.normal(0.0, 0.5, (5, 12))
    kwargs = {"trim_ratio": 0.1, "add_intercept": True}
    for backend in ("pandas_numpy", "polars"):
        alias = OperatorRegistry.get("cs_robust_resid", backend=backend)
        canon = OperatorRegistry.get("cs_trimmed_ols_resid", backend=backend)
        assert alias is not None and canon is not None, backend
        args = (y, x) if backend == "pandas_numpy" else (_polars(y), _polars(x))
        out_alias = alias.calculate(*args, **kwargs)
        out_canon = canon.calculate(*args, **kwargs)
        np.testing.assert_allclose(
            out_alias.to_numpy(),
            out_canon.to_numpy(),
            rtol=1e-8,
            atol=1e-8,
            equal_nan=True,
        )


def test_cs_trimmed_ols_resid_metadata_is_honest() -> None:
    for backend in ("pandas_numpy", "polars"):
        op = OperatorRegistry.get("cs_trimmed_ols_resid", backend=backend)
        assert op is not None, backend
        desc = (op.metadata.description or "").lower()
        # honest: names itself trimmed OLS, not a true robust regression.
        assert "trimmed" in desc and "ols" in desc, backend
        # and documents the deprecated alias.
        assert "cs_robust_resid" in (op.metadata.description or ""), backend


def test_hierarchical_neutralize_pandas_twin_unknown_group_stays_nan() -> None:
    """P0-9 pandas twin: a cell whose group/subgroup label is unknown keeps its
    residual NaN — it must never be demeaned alone into a perfectly-neutralized 0."""
    from cleaned_operators.group_ext import HierarchicalGroupNeutralize

    values = np.array([10.0, 12.0, 11.0, 5.0, 7.0])
    groups = np.array(["A", "A", "A", np.nan, "B"], dtype=object)
    subs = np.array(["x", "x", "y", "z", "x"], dtype=object)
    out = HierarchicalGroupNeutralize._demean_composite_row(values, groups, subs)
    assert np.isnan(out[3]), "unknown-group cell manufactured into a 0 residual"
    assert abs(out[0] + 1.0) < 1e-12 and abs(out[1] - 1.0) < 1e-12  # (A,x) demeaned
    assert abs(out[2]) < 1e-12 and abs(out[4]) < 1e-12  # single-member valid keys -> 0
