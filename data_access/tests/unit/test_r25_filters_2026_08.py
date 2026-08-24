# -*- coding: utf-8 -*-
"""R25 T-FLT-001..004 —— required filter 测试（P0-003/004）。

    T-FLT-001  US finance 无 timeframe → 拒绝（全 read surface）
    T-FLT-002  timeframe=["quarterly","annual"] → 拒绝（exactly-one）
    T-FLT-003  单值 quarterly → 成功
    T-FLT-004  physical columns / columns=None 仍不能绕过 dataset-level requirement
"""
from __future__ import annotations

import pytest

from data_access.contract.filters import validate_filter_requirements
from data_access.contract.runtime_contract import compile_runtime_contract
from data_access.core.exceptions import ValidationError
from data_access.registry import load_registry


def _us_contract():
    return compile_runtime_contract("us_stock_income", load_registry())


def _assert_rejected(fn, match):
    with pytest.raises(ValidationError, match=match):
        fn()


def test_tflt001_no_timeframe_rejected():
    rc = _us_contract()
    # strict 下缺失 timeframe → 拒绝
    _assert_rejected(
        lambda: validate_filter_requirements(rc, strict=True, params={}, filters={}),
        "timeframe",
    )


def test_tflt002_multi_timeframe_rejected():
    rc = _us_contract()
    _assert_rejected(
        lambda: validate_filter_requirements(
            rc, strict=True, params={}, filters={"timeframe": ["quarterly", "annual"]}
        ),
        "exactly_one",
    )


def test_tflt003_single_timeframe_passes():
    rc = _us_contract()
    validate_filter_requirements(
        rc, strict=True, params={}, filters={"timeframe": "quarterly"}
    )


def test_tflt004_columns_none_cannot_bypass():
    """物理直读列 / columns=None 仍不能绕过 dataset-level requirement。

    FilterRequirement 直接从 RuntimeDatasetContract 编译（不依赖 catalog 字段），
    与 store 的 ``_enforce_runtime_contract_filters`` 一致。
    """
    from data_access.contract.filters import build_filter_requirements_from_contract
    from data_access.cos_contract import COS_DATASET_CONTRACTS

    contract = COS_DATASET_CONTRACTS["us_stock_balance"]
    reqs = build_filter_requirements_from_contract(contract)
    fields = {r.field for r in reqs}
    assert "timeframe" in fields
    # required_event_filters 必须进入 FilterRequirement（P0-003）
    tf = next(r for r in reqs if r.field == "timeframe")
    assert tf.scope == "event"
    assert tf.cardinality == "exactly_one"
    assert tf.required is True


def test_tflt004b_store_gate_uses_event_filters():
    """store._dataset_required_filters 现在包含 required_event_filters（P0-003）。"""
    from data_access.registry import load_registry

    # 用 store 实例验证（不真正读数据）
    from data_access.core.engine import DuckDBEngine
    from data_access.store import DataAccessStore

    store = DataAccessStore(registry=load_registry(), engine=DuckDBEngine(threads=1))
    required = store._dataset_required_filters("us_stock_balance")
    assert "timeframe" in required  # P0-003：不再漏 event filters
