# -*- coding: utf-8 -*-
"""R37-P0-006/007/008/009：参数域 store + exact-call membership + production fail-closed。

禁止 "bool(passed)" 式过度认证（P0-007）：operator_has_any_certified_region
≠ exact_call_is_certified。production 准入只看后者。
"""
from __future__ import annotations

import os

os.environ.setdefault("FACTOR_ENGINE_CPU_BUDGET", "4")


def _load_store():
    from runtime.parameter_domain_store import ParameterDomainCertificationStore

    store = ParameterDomainCertificationStore()
    n = store.load_json(os.path.join("docs", "evidence", "r37", "R37_PARAMETER_DOMAIN_STORE.json"))
    assert n > 0, "参数域 store 必须已生成"
    return store


def test_exact_call_is_certified_for_certified_point():
    store = _load_store()
    assert store.exact_call_is_certified("ts_mean", {"window": 20}) is True
    assert store.exact_call_is_certified("ts_mean", {"window": 252}) is True
    assert store.exact_call_is_certified("rank", {}) is True


def test_exact_call_is_certified_false_for_uncertified():
    store = _load_store()
    # 未认证的 window（例如 window=13 从未测过）=> False（fail closed）
    assert store.exact_call_is_certified("ts_mean", {"window": 13}) is False
    # 未认证的算子 => False
    assert store.exact_call_is_certified("ts_ema", {"window": 20}) is False


def test_any_region_vs_exact_call_separation():
    """P0-007：any_certified_region 弱查询必须不能替代 exact_call。"""
    store = _load_store()
    # ts_mean 有 certified 区域
    assert store.operator_has_any_certified_region("ts_mean") is True
    # 但 window=13 这个精确调用不 certified
    assert store.exact_call_is_certified("ts_mean", {"window": 13}) is False


def test_invalid_rejection_all_zero():
    """P0-008：invalid 参数（0/负/小数/NaN/Inf）必须全部被拒绝。"""
    import json

    d = json.load(open(os.path.join("docs", "evidence", "r37",
                                     "R37_PARAMETER_DOMAIN_CERTIFICATION.json"), encoding="utf-8"))
    assert d["invalid_rejection_failed"] == 0, d["invalid_rejection_failures"]
    assert d["invalid_rejection_passed"] > 0


def test_production_assert_parameter_point_fail_closed():
    """P0-009：production + uncertified point => fail closed（ParameterDomainError）。"""
    import os

    from runtime.exceptions import ParameterDomainError
    from runtime.parameter_domain_store import (
        ParameterDomainCertificationStore,
        assert_parameter_point_certified,
        reset_parameter_domain_store,
    )

    reset_parameter_domain_store()
    store = ParameterDomainCertificationStore()
    store.load_json(os.path.join("docs", "evidence", "r37", "R37_PARAMETER_DOMAIN_STORE.json"))

    _prev = os.environ.get("QUANT_PRODUCTION_MODE")
    os.environ["QUANT_PRODUCTION_MODE"] = "1"
    try:
        # uncertified -> raise
        try:
            assert_parameter_point_certified("ts_mean", {"window": 13}, store=store)
            raised = False
        except ParameterDomainError:
            raised = True
        assert raised, "production 下 uncertified 参数点必须 fail closed"
        # certified -> 不抛
        assert_parameter_point_certified("ts_mean", {"window": 20}, store=store)
    finally:
        if _prev:
            os.environ["QUANT_PRODUCTION_MODE"] = _prev
        else:
            os.environ.pop("QUANT_PRODUCTION_MODE", None)


def test_research_uncertified_allow_telemetry():
    """P0-009：research + uncertified => allow（不抛）。"""
    import logging

    from runtime.parameter_domain_store import (
        ParameterDomainCertificationStore,
        assert_parameter_point_certified,
    )

    store = ParameterDomainCertificationStore()
    store.load_json(os.path.join("docs", "evidence", "r37", "R37_PARAMETER_DOMAIN_STORE.json"))
    # research 模式（默认）：不抛
    assert_parameter_point_certified("ts_mean", {"window": 13}, store=store)


def test_certification_key_distinguishes_points():
    """P0-006：认证 key 必须区分精确参数点。"""
    from runtime.parameter_domain_store import CertificationKey

    k1 = CertificationKey.from_kwargs("ts_mean", {"window": 20})
    k2 = CertificationKey.from_kwargs("ts_mean", {"window": 60})
    k3 = CertificationKey.from_kwargs("ts_mean", {"window": 20}, backend="polars")
    assert k1.to_key() != k2.to_key()
    assert k1.to_key() != k3.to_key()
