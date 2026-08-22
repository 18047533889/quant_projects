# -*- coding: utf-8 -*-
"""R11 canonical-honesty regression tests for the cs_multi_ridge_resid rename.

``cs_multi_robust_resid`` was never a robust regression — the implementation is
a STANDARDIZED RIDGE least-squares residual with a fixed, versioned ridge=1e-3
(R11 #139/#140).  The honest canonical is now ``cs_multi_ridge_resid`` and the
old name is a DEPRECATED alias to the same implementation (same kernel, same
ridge math).

This module imports ``cleaned_operators.weighted_moment_ext`` directly and
shadows the operators-conftest autouse ``load_all`` guard (same pattern as
``test_r11_contract_round_2026_08``) so it stays green while the full operator
registry is being rebuilt by concurrent sessions.  The registry-level lookups
still go through the real ``OperatorRegistry``, exactly like the rest of the
suite.

Verifies:
* (a) cs_multi_ridge_resid is the canonical (both backends registered).
* (b) cs_multi_robust_resid still resolves and maps to the SAME operator.
* (c) both names produce IDENTICAL output on a synthetic panel (pandas
       ``calculate`` and polars ``_calculate_series`` — same ridge math).
* (d) the canonical's description is honest: mentions ridge / 1e-3 and marks
       cs_multi_robust_resid as a deprecated alias; ridge is NOT searchable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pl = pytest.importorskip("polars")

# Importing the module registers this module's operators (and the deprecated
# alias) directly into the shared OperatorRegistry — fast, and independent of
# the heavy session-wide load_all.
import cleaned_operators.weighted_moment_ext as _wm_ext  # noqa: E402,F401
from cleaned_operators.registry import OperatorRegistry  # noqa: E402
from cleaned_operators.weighted_moment_ext import _RIDGE  # noqa: E402

_NEW = "cs_multi_ridge_resid"
_OLD = "cs_multi_robust_resid"


@pytest.fixture(scope="session", autouse=True)
def strict_fiscal_parameter_domain_certification_guard() -> None:
    """Shadow the operators-conftest autouse load_all guard (see module doc)."""
    yield


def _panel(seed: int = 7, n_stocks: int = 25):
    rng = np.random.default_rng(seed)
    cols = [f"C{i}" for i in range(n_stocks)]
    y = pd.DataFrame(rng.normal(0, 1, (1, n_stocks)), index=[0], columns=cols)
    x1 = pd.DataFrame(rng.normal(0, 1, (1, n_stocks)), index=[0], columns=cols)
    x2 = pd.DataFrame(rng.normal(0, 1, (1, n_stocks)), index=[0], columns=cols)
    x3 = pd.DataFrame(rng.normal(0, 1, (1, n_stocks)), index=[0], columns=cols)
    y = 0.5 * x1 - 0.3 * x2 + 0.2 * x3 + 2.0
    return y, x1, x2, x3


# ---------------------------------------------------------------------------
# (a) cs_multi_ridge_resid is the honest canonical
# ---------------------------------------------------------------------------
def test_ridge_canonical_registered() -> None:
    assert OperatorRegistry.resolve_canonical(_NEW) == _NEW
    op = OperatorRegistry.get(_NEW, "pandas_numpy")
    assert op is not None
    assert getattr(op.metadata, "name", None) == _NEW
    backends = OperatorRegistry.backends_for(_NEW)
    assert "pandas_numpy" in backends
    assert "polars" in backends


def test_ridge_versioned_implementation_constant() -> None:
    # R11 #140: the fixed ridge is a versioned implementation constant.
    assert _RIDGE == 1e-3


# ---------------------------------------------------------------------------
# (b) cs_multi_robust_resid is a deprecated alias to the SAME operator
# ---------------------------------------------------------------------------
def test_robust_name_is_deprecated_alias_same_operator() -> None:
    assert OperatorRegistry.resolve_canonical(_OLD) == _NEW
    old_pd = OperatorRegistry.get(_OLD, "pandas_numpy")
    new_pd = OperatorRegistry.get(_NEW, "pandas_numpy")
    assert old_pd is not None
    assert old_pd is new_pd  # alias resolves to the same registered impl
    old_pl = OperatorRegistry.get(_OLD, "polars")
    new_pl = OperatorRegistry.get(_NEW, "polars")
    assert old_pl is new_pl


# ---------------------------------------------------------------------------
# (c) identical output under both names (same ridge math)
# ---------------------------------------------------------------------------
def test_identical_output_pandas_calculate() -> None:
    y, x1, x2, x3 = _panel()
    out_new = OperatorRegistry.get(_NEW, "pandas_numpy").calculate(
        y, x1, x2, x3, add_intercept=True
    )
    out_old = OperatorRegistry.get(_OLD, "pandas_numpy").calculate(
        y, x1, x2, x3, add_intercept=True
    )
    np.testing.assert_allclose(
        out_new.to_numpy(), out_old.to_numpy(), rtol=0.0, atol=0.0, equal_nan=True
    )
    # the ridge regression actually produced finite residuals (25 stocks pass
    # the R11 #141 DOF-margin gate with k = 3 exposures + intercept).
    assert np.isfinite(out_new.to_numpy()).any()


def test_identical_output_calculate_series_polars() -> None:
    y, x1, x2, x3 = _panel()
    py = pl.DataFrame(y)
    px1 = pl.DataFrame(x1)
    px2 = pl.DataFrame(x2)
    px3 = pl.DataFrame(x3)
    op_new = OperatorRegistry.get(_NEW, "polars")
    op_old = OperatorRegistry.get(_OLD, "polars")
    assert op_new is not None and op_old is not None
    res_new = op_new._calculate_series(py, px1, px2, px3, add_intercept=True)
    res_old = op_old._calculate_series(py, px1, px2, px3, add_intercept=True)
    assert res_new is res_old or np.allclose(
        res_new.to_numpy(), res_old.to_numpy(), rtol=0.0, atol=0.0, equal_nan=True
    )


# ---------------------------------------------------------------------------
# (d) the canonical description is honest
# ---------------------------------------------------------------------------
def test_ridge_description_honest() -> None:
    meta = getattr(OperatorRegistry.get(_NEW, "pandas_numpy"), "metadata", None)
    assert meta is not None
    desc = (meta.description or "").lower()
    assert "ridge" in desc
    assert "1e-3" in desc
    # old name documented as a deprecated alias, not the canonical
    assert _OLD in (meta.description or "")
    # the ridge must NOT be a searchable parameter (R11 #140)
    assert "ridge" not in (meta.param_names or [])
    assert "ridge" not in (meta.param_specs or {})
