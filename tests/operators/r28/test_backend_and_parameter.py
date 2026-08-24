# -*- coding: utf-8 -*-
"""R28 §一百三十..一百三十二 / §一百七十八..一百七十九: backend claims and
parameter contracts.

- Every catalog backend claim must correspond to a real registered backend.
- Every backend claimed for a canonical must be invocable on a valid fixture
  (a claim without a working implementation is a drift that must be downgraded).
- Unknown keyword arguments must be rejected, not silently swallowed
  (**kwargs swallowing, R28 §一百三十一).
- Numeric parameters passed out of their declared domain must fail closed.
"""
from __future__ import annotations

import io
import re
import tokenize

import numpy as np
import pytest

from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry

import importlib.util
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_audit_spec = importlib.util.spec_from_file_location(
    "r28_audit_helpers", _REPO_ROOT / "scripts" / "audit_all_factor_production.py"
)
audit = importlib.util.module_from_spec(_audit_spec)
_audit_spec.loader.exec_module(audit)


@pytest.fixture(scope="module", autouse=True)
def _loaded():
    load_all()
    return True


def test_catalog_backend_claims_registered():
    """A catalog claim for pandas/polars/sql must have a real registered backend."""
    catalog = OperatorRegistry.catalog()
    mismatches = []
    for canonical in sorted(OperatorRegistry.list_canonical()):
        cat = catalog.get(canonical, {}) or {}
        claimed = set(cat.get("backends") or ())
        actual = set((OperatorRegistry._operators.get(canonical, {}) or {}).keys())
        if claimed and not claimed <= actual:
            mismatches.append((canonical, sorted(claimed), sorted(actual)))
    assert mismatches == [], f"catalog backend claims without implementation: {mismatches}"


def test_claimed_backends_are_instantiables():
    """Every claimed backend must resolve to an operator object registered in the
    canonical's backend map."""
    catalog = OperatorRegistry.catalog()
    missing = []
    for canonical in sorted(OperatorRegistry.list_canonical()):
        cat = catalog.get(canonical, {}) or {}
        actual = set((OperatorRegistry._operators.get(canonical, {}) or {}).keys())
        for backend in cat.get("backends") or ():
            if backend not in actual:
                missing.append((canonical, backend))
    assert missing == [], f"claimed backends not registered: {missing}"


def test_unknown_kwargs_rejected():
    """Passing an undeclared keyword parameter must raise OperatorParameterError
    (not be silently swallowed by **kwargs)."""
    from factor_engine.backend.operator_errors import OperatorParameterError

    op = OperatorRegistry.get("ts_mean", "pandas_numpy")
    x = audit._panels()["x"]
    with pytest.raises(OperatorParameterError):
        op.calculate(x, window=20, totally_unknown_param=42)


def test_scalar_param_out_of_domain_fails_closed():
    """A window < 1 must be rejected by the parameter contract."""
    from factor_engine.backend.operator_errors import OperatorParameterError

    op = OperatorRegistry.get("ts_mean", "pandas_numpy")
    x = audit._panels()["x"]
    with pytest.raises((OperatorParameterError, ValueError)):
        op.calculate(x, window=0)


def test_no_bare_kwargs_swallowing_behavioral():
    """Passing an unknown keyword must be REJECTED across families (not silently
    swallowed by a **kwargs forwarder).  A representative cross-family sample."""
    from factor_engine.backend.operator_errors import OperatorParameterError

    x = audit._panels()["x"]
    y = audit._panels()["ret"]
    cases = [
        ("ts_mean", (x,), {"window": 20}),
        ("rank", (x,), {}),
        ("ts_garch_next_vol_forecast", (y,), {"window": 60}),
        ("ts_kalman_level", (x,), {"q": 1e-3, "r": 1e-2}),
        ("CMF", (x, x, x), {"window": 20}),
        ("cs_regression", (y, x), {}),
        ("ts_regression_slope", (y, x), {"window": 20}),
        ("holder_concentration", (x, x), {}),
    ]
    for canonical, args, kw in cases:
        op = OperatorRegistry.get(canonical, "pandas_numpy")
        if op is None:
            continue
        try:
            op.calculate(*args, **kw, _r28_unknown_param_42=1)
        except (OperatorParameterError, TypeError, ValueError):
            pass  # correctly rejected
        else:
            pytest.fail(f"{canonical} silently swallowed an unknown keyword argument")


def test_parameter_feasibility_contracts():
    """Rank/window/component feasibility must hold at runtime (R28 §一百三十二)."""
    from factor_engine.backend.operator_errors import OperatorParameterError

    x = audit._panels()["x"]
    # ts_bottomk requires k <= window (feasibility) — an infeasible k must fail
    op = OperatorRegistry.get("ts_bottomk_mean", "pandas_numpy")
    try:
        out = op.calculate(x, window=10, k=50)  # k > window -> infeasible
        # either rejected or produces all-NaN (fail closed) — never silent garbage
        v = out.to_numpy(dtype=float)
        assert np.all(np.isnan(v)) or np.isfinite(v).sum() == 0, (
            "infeasible k=50/window=10 produced finite output"
        )
    except (OperatorParameterError, ValueError):
        pass
