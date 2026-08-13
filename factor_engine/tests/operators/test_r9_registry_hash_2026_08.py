# -*- coding: utf-8 -*-
"""R9-P1-041/042/043/044: registry semantic-hash correctness sweep.

* R9-P1-041 — ``_freeze_const`` must not fall back to ``repr()`` (frozensets /
  dicts iterate hash-seed-dependently).  It must delegate to ``_freeze_value``.
* R9-P1-042 — ``_freeze_value`` must hash frozen dataclasses by their fields,
  any Mapping canonically, objects with an explicit ``semantic_identity()`` by
  that identity, and must FAIL for unsupported mutable objects instead of
  hashing by type name alone.
* R9-P1-043 — DataFrame freeze must include the index (values), index name(s),
  timezone, dtypes, NaN mask and column order.
* R9-P1-044 — ``_contract_hash`` must cover the full logical contract,
  including relational_specs and the cost contract.

These are pure-helper tests on directly-constructed objects (no registry
registration), mirroring ``test_r7_registry_governance.py``.
"""
from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from cleaned_operators.registry import (
    _contract_hash,
    _freeze_const,
    _freeze_value,
)


# ---------------------------------------------------------------------------
# R9-P1-042: frozen dataclasses hash by fields, never by type name alone
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class _FrozenCfg:
    window: int
    name: str


def test_frozen_dataclass_different_fields_different_hash():
    assert _freeze_value(_FrozenCfg(window=20, name="a")) != _freeze_value(
        _FrozenCfg(window=21, name="a")
    )
    assert _freeze_value(_FrozenCfg(window=20, name="a")) != _freeze_value(
        _FrozenCfg(window=20, name="b")
    )


def test_frozen_dataclass_same_fields_same_hash():
    assert _freeze_value(_FrozenCfg(window=20, name="a")) == _freeze_value(
        _FrozenCfg(window=20, name="a")
    )


def test_unsupported_mutable_object_raises():
    class _OpaqueMutable:
        pass

    with pytest.raises(TypeError, match="R9-P1-042"):
        _freeze_value(_OpaqueMutable())


def test_mapping_other_than_dict_is_canonical():
    from types import MappingProxyType

    plain = {"b": 2, "a": 1}
    proxy = MappingProxyType(plain)
    assert _freeze_value(proxy) == _freeze_value(plain)


# ---------------------------------------------------------------------------
# R9-P1-041: _freeze_const delegates to _freeze_value for frozenset/dict
# ---------------------------------------------------------------------------

def test_freeze_const_frozenset_matches_freeze_value():
    fs = frozenset({1, 2, 3})
    assert _freeze_const(fs) == _freeze_value(fs)
    # sanity: the canonical payload is hash-seed independent (sorted).
    assert _freeze_const(fs) == "set{1,2,3}"


def test_freeze_const_dict_matches_freeze_value():
    d = {"b": 2, "a": 1}
    assert _freeze_const(d) == _freeze_value(d)
    # sorted-key canonical payload — repr() would be hash-seed dependent.
    canonical = _freeze_const(d)
    assert canonical.index("'a'") < canonical.index("'b'")


# ---------------------------------------------------------------------------
# R9-P1-043: DataFrame freeze covers index / dtypes / NaN mask / column order
# ---------------------------------------------------------------------------

def _frame(data=None, **kwargs) -> pd.DataFrame:
    return pd.DataFrame(data, **kwargs)


def test_dataframe_index_values_participate():
    a = _frame({"x": [1.0, 2.0]}, index=pd.Index([10, 20], name="idx"))
    b = _frame({"x": [1.0, 2.0]}, index=pd.Index([11, 20], name="idx"))
    assert _freeze_value(a) != _freeze_value(b)


def test_dataframe_index_name_participates():
    a = _frame({"x": [1.0, 2.0]}, index=pd.Index([10, 20], name="first"))
    b = _frame({"x": [1.0, 2.0]}, index=pd.Index([10, 20], name="second"))
    assert _freeze_value(a) != _freeze_value(b)


def test_dataframe_timezone_participates():
    base = pd.to_datetime(["2024-01-01", "2024-01-02"])
    a = _frame({"x": [1.0, 2.0]}, index=pd.DatetimeIndex(base, tz="UTC"))
    b = _frame({"x": [1.0, 2.0]}, index=pd.DatetimeIndex(base, tz="Asia/Shanghai"))
    assert _freeze_value(a) != _freeze_value(b)


def test_dataframe_dtypes_participate():
    a = _frame({"x": [1, 2]})  # int64
    b = _frame({"x": [1.0, 2.0]})  # float64
    assert _freeze_value(a) != _freeze_value(b)


def test_dataframe_nan_mask_participates():
    a = _frame({"x": [1.0, np.nan]})
    b = _frame({"x": [1.0, 0.0]})
    assert _freeze_value(a) != _freeze_value(b)
    # NaN at a DIFFERENT position also differs.
    c = _frame({"x": [np.nan, 1.0]})
    assert _freeze_value(a) != _freeze_value(c)
    # identical NaN frames are stable (mask canonicalizes NaN bit patterns).
    assert _freeze_value(a) == _freeze_value(_frame({"x": [1.0, float("nan")]}))


def test_dataframe_column_order_participates():
    a = _frame({"x": [1.0], "y": [2.0]})
    b = _frame({"y": [2.0], "x": [1.0]})
    assert _freeze_value(a) != _freeze_value(b)


def test_dataframe_date_index_vs_other_index_participates():
    # identical values but 2024 vs 2025 date indices must hash apart.
    a = _frame({"x": [1.0, 2.0]}, index=pd.DatetimeIndex(pd.to_datetime(["2024-01-01", "2024-01-02"])))
    b = _frame({"x": [1.0, 2.0]}, index=pd.DatetimeIndex(pd.to_datetime(["2025-01-01", "2025-01-02"])))
    assert _freeze_value(a) != _freeze_value(b)


# ---------------------------------------------------------------------------
# R9-P1-044: _contract_hash covers relational_specs and cost contract
# ---------------------------------------------------------------------------

def _fake_operator(name: str, relational_specs=None):
    return SimpleNamespace(
        metadata=SimpleNamespace(
            name=name,
            category="test",
            param_names=["x"],
            param_specs={},
            panel_params=(),
            scalar_params=(),
            param_aliases={},
            input_units={},
            output_unit=None,
            input_grain=None,
            output_grain=None,
            available_at=None,
            input_fields=[],
            relational_specs=relational_specs or [],
        )
    )


def test_contract_hash_changes_with_relational_spec():
    from cleaned_operators.base import RelationalParamSpec

    base = _fake_operator("test_op")
    with_rel = _fake_operator(
        "test_op",
        relational_specs=[RelationalParamSpec(expression="window >= 4")],
    )
    assert _contract_hash(base) != _contract_hash(with_rel)
    # two DIFFERENT relations also hash apart.
    with_other = _fake_operator(
        "test_op",
        relational_specs=[RelationalParamSpec(expression="window <= 64")],
    )
    assert _contract_hash(with_rel) != _contract_hash(with_other)


def test_contract_hash_changes_with_cost_contract():
    # ts_mean is the default O(W) band; ts_hvg_motif_entropy is the O(W^3) band.
    cheap = _fake_operator("ts_mean")
    expensive = _fake_operator("ts_hvg_motif_entropy")
    assert _contract_hash(cheap) != _contract_hash(expensive)


def test_contract_hash_deterministic_same_contract():
    assert _contract_hash(_fake_operator("test_op")) == _contract_hash(
        _fake_operator("test_op")
    )
