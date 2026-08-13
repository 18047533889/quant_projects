# -*- coding: utf-8 -*-
"""R13 final-runtime-state contract: the overhaul layer must not silently
replace a fixed canonical implementation with old semantics.

The audit book's central complaint: ``common.daily_panel.py`` was fixed, a unit
test read the source and believed the operator was fixed, but ``load_all()`` ran
``operator_overhaul`` afterwards and the REAL runtime resolved to a stale
implementation.  These tests assert the FINAL post-``load_all`` registry state:
* the audited fixed operators resolve to the fixed kernel modules,
* their implementation hash equals the fixed kernel's hash,
* the conditional family is strict ``ConditionBool`` ({0,1,NaN}),
* ``days_since`` / ``true_streak`` censor unknown (NaN) conditions,
* overhaul replacements inherit the canonical logical contract,
* backend registrations never self-declare ``pit_safe`` / ``audited`` tags.
"""
from __future__ import annotations

import hashlib
import importlib

import numpy as np
import pandas as pd
import pytest

from cleaned_operators import load_all
from cleaned_operators.registry import OperatorRegistry
from cleaned_operators.registry import _fn_payload, _impl_source_hash


@pytest.fixture(scope="module", autouse=True)
def _loaded() -> None:
    load_all()


# --- NEW-P0-01: final-resolve audit of the fixed canonical implementations ---

# canonical -> per-backend (expected kernel module, expected kernel name)
AUDITED_FIXED: dict[str, dict[str, tuple[str, str]]] = {
    "ts_count_if": {
        "pandas_numpy": ("cleaned_operators.overhaul.daily", "pd_count_if"),
        "polars": ("cleaned_operators.overhaul.daily", "pl_count_if"),
    },
    "ts_sum_if": {
        "pandas_numpy": ("cleaned_operators.overhaul.daily", "pd_sum_if"),
        "polars": ("cleaned_operators.overhaul.daily", "pl_sum_if"),
    },
    "ts_mean_if": {
        "pandas_numpy": ("cleaned_operators.overhaul.daily", "pd_mean_if"),
        "polars": ("cleaned_operators.overhaul.daily", "pl_mean_if"),
    },
    "ts_std_if": {
        "pandas_numpy": ("cleaned_operators.overhaul.daily", "pd_std_if"),
        "polars": ("cleaned_operators.overhaul.daily", "pl_std_if"),
    },
    "ts_last_if": {
        "pandas_numpy": ("cleaned_operators.overhaul.daily", "pd_last_if"),
        "polars": ("cleaned_operators.overhaul.daily", "pl_last_if"),
    },
    # ``ts_days_since`` is deliberately the ``layer_composite_fixes`` final layer
    # (inclusive max_lookback + NaN censor/reset semantics).
    "ts_days_since": {
        "pandas_numpy": ("cleaned_operators.layer_composite_fixes", "pd_days_since_inclusive"),
        "polars": ("cleaned_operators.layer_composite_fixes", "pl_days_since_inclusive"),
    },
    "ts_true_streak": {
        "pandas_numpy": ("cleaned_operators.overhaul.daily", "pd_true_streak"),
        "polars": ("cleaned_operators.overhaul.daily", "pl_true_streak"),
    },
    # Return-decomposition family: fixed price_basis / PositivePrice gates live
    # in ``cleaned_operators.return_decomp`` and must resolve there untouched.
    "overnight_return": {
        "pandas_numpy": ("cleaned_operators.return_decomp", "OvernightReturn._calculate_series"),
    },
    "open_close_return": {
        "pandas_numpy": ("cleaned_operators.return_decomp", "OpenCloseReturn._calculate_series"),
    },
    "open_to_vwap_return": {
        "pandas_numpy": ("cleaned_operators.return_decomp", "OpenToVwapReturn._calculate_series"),
    },
    "vwap_to_close_return": {
        "pandas_numpy": ("cleaned_operators.return_decomp", "VwapToCloseReturn._calculate_series"),
    },
}


def _kernel_fn(operator: object):
    """Return the operator's actual kernel callable (``_fn`` or a class method)."""
    fn = getattr(operator, "_fn", None)
    if fn is not None:
        return fn
    return getattr(operator, "_calculate_series", None) or getattr(operator, "calculate", None)


def _kernel_module(operator: object) -> str:
    fn = getattr(operator, "_fn", None)
    if fn is not None:
        return getattr(fn, "__module__", "") or ""
    return getattr(type(operator), "__module__", "") or ""


def _fn_hash(fn: object) -> str:
    payload = _fn_payload(getattr(fn, "__func__", fn), None)
    assert payload, "kernel has no stable payload hash"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _resolve_expected(module: str, name: str):
    if "." in name and "._calculate_series" in name:
        class_name, method = name.split("._calculate_series")
        cls = getattr(importlib.import_module(module), class_name)
        return cls._calculate_series
    return getattr(importlib.import_module(module), name)


@pytest.mark.parametrize("canonical,backends", [
    (canonical, list(specs))
    for canonical, specs in AUDITED_FIXED.items()
])
def test_audited_fixed_ops_resolve_to_fixed_kernels(canonical, backends) -> None:
    for backend in backends:
        expected_module, expected_name = AUDITED_FIXED[canonical][backend]
        op = OperatorRegistry.get(canonical, backend=backend)
        assert op is not None, f"{canonical}/{backend} not registered"
        # 1. the resolved kernel module must be the fixed module.
        assert _kernel_module(op) == expected_module, (
            f"{canonical}/{backend} resolved module {_kernel_module(op)} != "
            f"expected fixed module {expected_module}"
        )
        # 2. the implementation hash must equal the fixed kernel's hash.
        expected_kernel = _resolve_expected(expected_module, expected_name)
        assert _fn_hash(_kernel_fn(op)) == _fn_hash(expected_kernel), (
            f"{canonical}/{backend} implementation hash differs from the fixed "
            f"kernel {expected_module}.{expected_name}"
        )


@pytest.mark.parametrize("canonical", [
    "ts_count_if", "ts_sum_if", "ts_mean_if", "ts_std_if", "ts_last_if",
    "ts_days_since", "ts_true_streak",
    "overnight_return", "open_close_return", "open_to_vwap_return",
    "vwap_to_close_return",
])
def test_audited_fixed_ops_have_an_implementation_hash(canonical) -> None:
    """The resolved operators carry a real implementation hash (not a fallback)."""
    for backend in OperatorRegistry.backends_for(canonical):
        op = OperatorRegistry.get(canonical, backend=backend)
        assert op is not None
        h = _impl_source_hash(op)
        assert h and len(h) == 16, f"{canonical}/{backend} impl hash missing: {h!r}"


# --- NEW-P0-02: strict ConditionBool for the ts_*_if family ---

@pytest.mark.parametrize("bad_value", [2.0, -1.0, 0.5])
@pytest.mark.parametrize("name", [
    "ts_count_if", "ts_sum_if", "ts_mean_if", "ts_std_if", "ts_last_if",
])
def test_conditional_family_rejects_non_conditionbool(name, bad_value) -> None:
    x = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0]})
    condition = pd.DataFrame({"A": [1.0, bad_value, 0.0, 1.0]})
    op = OperatorRegistry.get(name)
    if name == "ts_count_if":
        with pytest.raises(ValueError, match="ConditionBool"):
            op.calculate(condition, 3, 1)
    else:
        with pytest.raises(ValueError, match="ConditionBool"):
            op.calculate(x, condition, 3)


@pytest.mark.parametrize("name", [
    "ts_count_if", "ts_sum_if", "ts_mean_if", "ts_std_if", "ts_last_if",
])
def test_conditional_family_nan_condition_is_missing_not_false(name) -> None:
    """A NaN condition is UNKNOWN — it must never count as False/0."""
    idx = pd.RangeIndex(5)
    x = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 5.0]}, index=idx)
    cond = pd.DataFrame({"A": [1.0, 1.0, np.nan, 1.0, 1.0]}, index=idx)
    op = OperatorRegistry.get(name)
    if name == "ts_count_if":
        result = op.calculate(cond, 5, 1)
        assert result["A"].iloc[-1] == 4.0, result["A"].tolist()  # NaN not counted
    elif name == "ts_sum_if":
        result = op.calculate(x, cond, 5, 1)
        # NaN condition at row 2 excludes x=3 from selection (never "False"),
        # so the trailing window sum is 1+2+4+5 = 12, not 15 and not 9.
        assert result["A"].tolist() == [1.0, 3.0, 3.0, 7.0, 12.0], result["A"].tolist()
    elif name == "ts_mean_if":
        result = op.calculate(x, cond, 5, 1)
        assert result["A"].iloc[-1] == 3.0, result["A"].tolist()  # (1+2+4+5)/4
    elif name == "ts_std_if":
        result = op.calculate(x, cond, 5, 1)
        assert result["A"].iloc[-1] == pytest.approx(np.std([1.0, 2.0, 4.0, 5.0], ddof=1))
    else:  # ts_last_if
        result = op.calculate(x, cond, 5)
        assert result["A"].iloc[-1] == 5.0  # NaN-condition row does not select


@pytest.mark.parametrize("name", [
    "ts_count_if", "ts_sum_if", "ts_mean_if", "ts_std_if", "ts_last_if",
])
def test_conditional_family_polars_rejects_non_conditionbool_and_nan_is_missing(name) -> None:
    """The polars backend enforces the same strict ConditionBool contract."""
    pl = pytest.importorskip("polars")
    if "polars" not in OperatorRegistry.backends_for(name):
        pytest.skip(f"{name} has no polars backend")
    op = OperatorRegistry.get(name, backend="polars")

    def to_pl(frame: pd.DataFrame) -> Any:
        return pl.DataFrame({c: frame[c].to_numpy() for c in frame.columns})

    x_pl = to_pl(pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 5.0]}))
    for bad in (2.0, -1.0, 0.5):
        bad_cond = to_pl(pd.DataFrame({"A": [1.0, bad, 0.0, 1.0, 1.0]}))
        if name == "ts_count_if":
            with pytest.raises(ValueError, match="ConditionBool"):
                op.calculate(bad_cond, 5, 1)
        else:
            with pytest.raises(ValueError, match="ConditionBool"):
                op.calculate(x_pl, bad_cond, 5)

    # NaN condition is missing, never False, on the polars backend too.
    nan_cond = to_pl(pd.DataFrame({"A": [1.0, 1.0, np.nan, 1.0, 1.0]}))
    if name == "ts_count_if":
        res = op.calculate(nan_cond, 5, 1)
        assert res["A"].to_list()[-1] == 4.0
    elif name == "ts_sum_if":
        res = op.calculate(x_pl, nan_cond, 5, 1)
        assert res["A"].to_list()[-1] == 12.0  # 1+2+4+5, x=3 excluded
    elif name == "ts_mean_if":
        res = op.calculate(x_pl, nan_cond, 5, 1)
        assert res["A"].to_list()[-1] == 3.0
    elif name == "ts_std_if":
        res = op.calculate(x_pl, nan_cond, 5, 1)
        assert res["A"].to_list()[-1] == pytest.approx(np.std([1.0, 2.0, 4.0, 5.0], ddof=1))
    else:  # ts_last_if
        res = op.calculate(x_pl, nan_cond, 5)
        assert res["A"].to_list()[-1] == 5.0


# --- NEW-P0-03: days_since / true_streak NaN censoring ---

def test_days_since_nan_censors_and_resets() -> None:
    cond = pd.DataFrame({"A": [1.0, 0.0, np.nan, 1.0, 0.0, 1.0]})
    result = OperatorRegistry.get("ts_days_since").calculate(cond, 10)
    # Row 2 is an unknown event state: output censored to NaN and the previously
    # known "last true" is reset (row 3 restarts at 0).
    np.testing.assert_allclose(
        result["A"].to_numpy(dtype=float),
        np.array([0.0, 1.0, np.nan, 0.0, 1.0, 0.0]),
        equal_nan=True,
    )


def test_days_since_nan_resets_even_with_prior_true() -> None:
    cond = pd.DataFrame({"A": [1.0, 0.0, 0.0, np.nan, 0.0]})
    result = OperatorRegistry.get("ts_days_since").calculate(cond, 10)
    # After the NaN at row 3 the distance is unknown -> NaN; the run from the
    # row-0 true must not "carry" across the uncertainty gap.
    assert np.isnan(result["A"].iloc[3])
    assert np.isnan(result["A"].iloc[4])


def test_true_streak_nan_censors_and_resets() -> None:
    cond = pd.DataFrame({"A": [1.0, 1.0, np.nan, 1.0, 0.0]})
    result = OperatorRegistry.get("ts_true_streak").calculate(cond)
    np.testing.assert_allclose(
        result["A"].to_numpy(dtype=float),
        np.array([1.0, 2.0, np.nan, 1.0, 0.0]),
        equal_nan=True,
    )


# --- NEW-P0-04: overhaul replacements inherit the canonical logical contract ---

def test_overhaul_replacements_inherit_canonical_logical_contract() -> None:
    from cleaned_operators.overhaul.base import _CONTRACT_FIELDS

    for canonical in OperatorRegistry.list_canonical():
        backend_meta = (OperatorRegistry._catalog.get(canonical) or {}).get("backend_meta") or {}
        catalog = OperatorRegistry._catalog.get(canonical) or {}
        for backend in ("pandas_numpy", "polars"):
            source = str((backend_meta.get(backend) or {}).get("source", ""))
            if not source.startswith("operator_overhaul_"):
                continue
            op = OperatorRegistry._operators.get(canonical, {}).get(backend)
            if op is None:
                continue
            om = getattr(op, "metadata", None)
            for field, _empty in _CONTRACT_FIELDS:
                canonical_value = catalog.get(field)
                if canonical_value in (None, "", (), [], {}):
                    continue
                current = getattr(om, field, None) if om is not None else None
                if current in (None, "", (), [], {}):
                    continue  # polars metadata lacks the field; catalog carries it
                # The operator's declared logical contract must match the
                # canonical catalog contract.
                if not _contract_equal(current, canonical_value):
                    pytest.fail(
                        f"{canonical}/{backend} logical-contract field "
                        f"{field!r} diverges from the canonical contract: "
                        f"{current!r} vs {canonical_value!r} (NEW-P0-04)"
                    )


def _contract_equal(left, right) -> bool:
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left.keys() != set(right.keys()):
            return False
        return all(_contract_equal(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            return False
        return all(_contract_equal(a, b) for a, b in zip(left, right))
    return left == right


# --- NEW-P0-05: no self-declared pit_safe / audited tags on backend impls ---

def test_overhaul_registrations_do_not_self_declare_certification_tags() -> None:
    """Backend registrations from the overhaul layer must not self-declare
    ``pit_safe`` / ``audited`` — those are owned by the canonical certification
    record, never by a backend registration.  Only backends whose OWN source is
    an ``operator_overhaul_*`` layer are checked (a later layer that replaced a
    backend may legitimately carry its own provenance)."""
    for canonical in OperatorRegistry.list_canonical():
        backend_meta = (OperatorRegistry._catalog.get(canonical) or {}).get("backend_meta") or {}
        for backend in ("pandas_numpy", "polars"):
            source = str((backend_meta.get(backend) or {}).get("source", ""))
            if not source.startswith("operator_overhaul_"):
                continue
            op = OperatorRegistry._operators.get(canonical, {}).get(backend)
            if op is None:
                continue
            tags = list(getattr(getattr(op, "metadata", None), "tags", None) or [])
            assert "pit_safe" not in tags, (canonical, backend, source, tags)
            assert "audited" not in tags, (canonical, backend, source, tags)
