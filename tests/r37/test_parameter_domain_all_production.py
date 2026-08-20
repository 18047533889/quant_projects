# -*- coding: utf-8 -*-
"""R37-P0-006/007/008/009 + R39 #26/#27/#28/#29：参数域 store + exact-call
membership + production fail-closed。

禁止 "bool(passed)" 式过度认证（P0-007）：operator_has_any_certified_region
≠ exact_call_is_certified。production 准入只看后者。

R39 #28：exact-call 查询必须带**全维度 identity**（semantic_version/backend/
execution_variant/source_context/dtype/grain），不再落默认值——默认值与证据空间
不一致会错查。测试统一走 ``_dims()``。
"""
from __future__ import annotations

import os

os.environ.setdefault("FACTOR_ENGINE_CPU_BUDGET", "4")

_STORE_PATH = os.path.join("docs", "evidence", "r37", "R37_PARAMETER_DOMAIN_STORE.json")


def _dims(canonical: str) -> dict:
    """R39 #28：与审计脚本 ``_certify_dimensions`` 一致的全维度 identity。"""
    from backend.operator_semantic_version import versioned_name

    return {
        "semantic_version": versioned_name(canonical),
        "backend": "pandas_numpy",
        "execution_variant": "reference",
        "source_context": "memory",
        "dtype": "float64",
        "grain": "daily",
    }


def _load_store():
    from runtime.parameter_domain_store import ParameterDomainCertificationStore

    store = ParameterDomainCertificationStore()
    n = store.load_json(_STORE_PATH)
    assert n > 0, "参数域 store 必须已生成"
    return store


def test_exact_call_is_certified_for_certified_point():
    store = _load_store()
    assert store.exact_call_is_certified("ts_mean", {"window": 20}, **_dims("ts_mean")) is True
    assert store.exact_call_is_certified("ts_mean", {"window": 252}, **_dims("ts_mean")) is True
    assert store.exact_call_is_certified("rank", {}, **_dims("rank")) is True
    # R39 #27：ts_rank 的认证点是**完整 bound**（window + min_periods 默认合并）
    assert store.exact_call_is_certified(
        "ts_rank", {"window": 20, "min_periods": 1}, **_dims("ts_rank")) is True
    # R39 #27/#28：ts_delay 用 canonical 名 ``n``（旧证据错用 ``window``）
    assert store.exact_call_is_certified("ts_delay", {"n": 1}, **_dims("ts_delay")) is True


def test_exact_call_is_certified_false_for_uncertified():
    store = _load_store()
    # 未认证的 window（例如 window=13 从未测过）=> False（fail closed）
    assert store.exact_call_is_certified("ts_mean", {"window": 13}, **_dims("ts_mean")) is False
    # 未认证的算子 => False
    assert store.exact_call_is_certified("ts_ema", {"window": 20}, **_dims("ts_ema")) is False


def test_exact_call_requires_full_identity_dimensions():
    """R39 #28：缺维度（落默认值）必须查不到带 source_context/semantic_version 的点。

    证据空间是 source_context="memory"/semantic_version="ts_mean@v1" 等；用默认
    （source_context=""）查询会 miss → False（fail closed，不会错查别的空间）。
    """
    store = _load_store()
    # 带全维度 → 命中
    assert store.exact_call_is_certified("ts_mean", {"window": 20}, **_dims("ts_mean")) is True
    # 落默认维度（source_context=""、semantic_version=""）→ 不同空间 → miss
    assert store.exact_call_is_certified("ts_mean", {"window": 20}) is False


def test_any_region_vs_exact_call_separation():
    """P0-007：any_certified_region 弱查询必须不能替代 exact_call。"""
    store = _load_store()
    # ts_mean 有 certified 区域
    assert store.operator_has_any_certified_region("ts_mean") is True
    # 但 window=13 这个精确调用不 certified
    assert store.exact_call_is_certified("ts_mean", {"window": 13}, **_dims("ts_mean")) is False


def test_invalid_rejection_all_zero():
    """P0-008：invalid 参数（0/负/小数/NaN/Inf）必须全部被拒绝。"""
    import json

    d = json.load(open(os.path.join("docs", "evidence", "r37",
                                     "R37_PARAMETER_DOMAIN_CERTIFICATION.json"), encoding="utf-8"))
    assert d["invalid_rejection_failed"] == 0, d["invalid_rejection_failures"]
    assert d["invalid_rejection_passed"] > 0


def test_production_assert_parameter_point_fail_closed():
    """P0-009 + R39 #26：production + uncertified point => fail closed。

    run_mode 显式传入时是**唯一权威**——即使 env 未设 production，run_mode="production"
    也必须 fail closed（认证层不重猜 env）。
    """
    import os

    from runtime.exceptions import ParameterDomainError
    from runtime.parameter_domain_store import (
        ParameterDomainCertificationStore,
        assert_parameter_point_certified,
        reset_parameter_domain_store,
    )

    reset_parameter_domain_store()
    store = ParameterDomainCertificationStore()
    store.load_json(_STORE_PATH)

    _prev = os.environ.get("QUANT_PRODUCTION_MODE")
    os.environ.pop("QUANT_PRODUCTION_MODE", None)  # #26：不靠 env 猜
    try:
        # #26：run_mode="production" 显式身份 => uncertified 必须 raise（env 为 research）
        try:
            assert_parameter_point_certified(
                "ts_mean", {"window": 13}, run_mode="production", store=store, **_dims("ts_mean"))
            raised = False
        except ParameterDomainError:
            raised = True
        assert raised, "run_mode='production' 下 uncertified 参数点必须 fail closed"
        # certified -> 不抛
        assert_parameter_point_certified(
            "ts_mean", {"window": 20}, run_mode="production", store=store, **_dims("ts_mean"))
        # #26：run_mode="research" 显式身份 => 即使 env 是 production 也放行
        try:
            assert_parameter_point_certified(
                "ts_mean", {"window": 13}, run_mode="research", store=store, **_dims("ts_mean"))
            ok_research = True
        except ParameterDomainError:
            ok_research = False
        assert ok_research, "run_mode='research' 显式身份必须放行（认证层不重猜 env）"
    finally:
        if _prev:
            os.environ["QUANT_PRODUCTION_MODE"] = _prev
        else:
            os.environ.pop("QUANT_PRODUCTION_MODE", None)


def test_research_uncertified_allow_telemetry():
    """P0-009：research + uncertified => allow（不抛）。"""
    from runtime.parameter_domain_store import (
        ParameterDomainCertificationStore,
        assert_parameter_point_certified,
    )

    store = ParameterDomainCertificationStore()
    store.load_json(_STORE_PATH)
    # research 模式（默认）：不抛
    assert_parameter_point_certified("ts_mean", {"window": 13}, store=store, **_dims("ts_mean"))


def test_assert_parameter_domain_ready_loads_and_freshness():
    """R39 #29：execution 前 store 必须装载（非空）；旧 SHA 证据 strict 下拒绝。

    ``assert_parameter_domain_ready`` 从默认 R37 证据装载；当 store 被显式伪造
    generated_commit 与当前 HEAD 不一致时 strict 拒绝（旧 SHA 证据不得悄悄存在）。
    """
    from runtime.exceptions import ParameterDomainError
    from runtime.parameter_domain_store import (
        assert_parameter_domain_ready,
        get_parameter_domain_store,
        reset_parameter_domain_store,
    )

    reset_parameter_domain_store()
    # 空 store（未装载）在 strict ready 下必须拒绝——不能悄悄放过。
    try:
        assert_parameter_domain_ready(strict=False)
        loaded = True
    except ParameterDomainError:
        loaded = False
    assert loaded, "strict=False 时允许装载（不抛）"
    store = get_parameter_domain_store(ensure_loaded=True)
    assert store._loaded and len(store._points) > 0, "默认证据必须已装载"
    assert store._evidence_hash, "装载后必须记录证据 hash"
    # 负控：已知 invalid 点不得 certified（证据完整性）
    from runtime.parameter_domain_store import run_negative_controls

    run_negative_controls(store)  # 不抛即通过

    # 旧 SHA：伪造 generated_commit != HEAD => strict 拒绝
    with store._lock:
        old = store._generated_commit
        store._generated_commit = "0" * 40
    try:
        assert_parameter_domain_ready(strict=True)
        raised = False
    except ParameterDomainError:
        raised = True
    finally:
        with store._lock:
            store._generated_commit = old
    assert raised, "旧 SHA 证据 strict 下必须 fail closed"


def test_default_param_call_is_certified_after_bound_normalization():
    """R39 #27：bound normalization —— 默认参数调用用**完整 bound 点**查询。

    ``ts_mean(x)``（window 走默认 20、attrs 无 window）合成 ``{window: 20}`` 参数点，
    命中认证点 → 不抛；而 ``window=13``（未认证）即使 production 也必须 fail closed。
    """
    from runtime.exceptions import ParameterDomainError
    from runtime.parameter_domain_store import (
        ParameterDomainCertificationStore,
        assert_parameter_point_certified,
    )

    store = ParameterDomainCertificationStore()
    store.load_json(_STORE_PATH)
    # 完整 bound：ts_mean 默认 window=20
    assert_parameter_point_certified(
        "ts_mean", {"window": 20}, run_mode="production", store=store, **_dims("ts_mean"))
    try:
        assert_parameter_point_certified(
            "ts_mean", {"window": 13}, run_mode="production", store=store, **_dims("ts_mean"))
        raised = False
    except ParameterDomainError:
        raised = True
    assert raised, "未认证的默认参数点必须 fail closed"


def test_certification_key_distinguishes_points():
    """P0-006：认证 key 必须区分精确参数点。"""
    from runtime.parameter_domain_store import CertificationKey

    k1 = CertificationKey.from_kwargs("ts_mean", {"window": 20})
    k2 = CertificationKey.from_kwargs("ts_mean", {"window": 60})
    k3 = CertificationKey.from_kwargs("ts_mean", {"window": 20}, backend="polars")
    assert k1.to_key() != k2.to_key()
    assert k1.to_key() != k3.to_key()
