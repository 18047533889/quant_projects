# -*- coding: utf-8 -*-
"""R21-DEF5：registry contract 的 plain-JSON 序列化（manifest serialize crash）。

缺陷：``_sql_emitter_ok`` → ``_safe_payload_hash(_sql_contract(canon))`` 对
registry catalog entry 的 RAW dict 做 ``json.dumps``，而 20 个 canonical 的
``param_specs`` 里是 live ``cleaned_operators.base.ParamSpec`` dataclass →
TypeError → CapabilityInfrastructureError → export_operator_manifest exit 1。

修复：``_sql_contract`` 现在返回 typed-serialized plain-JSON 视图
（ParamSpec 按声明字段、``dtype``→``__name__``、``ParamRole``→``.value``、
MISSING sentinel→``"__MISSING__"``），未知对象类型仍然 fail-closed。
"""
from __future__ import annotations

import json

import pytest

from factor_engine.backend.operator_capability import (
    CapabilityInfrastructureError,
    _json_contract_value,
    _safe_payload_hash,
    _sql_contract,
)
from factor_engine.cleaned_operators.base import ParamSpec
from factor_engine.cleaned_operators.registry import OperatorRegistry


# 20 offending canonicals（coordinator-verified exact list，R21-DEF5 ticket）
R21_DEF5_CANONICALS = (
    "group_neutralize",
    "ewm_corr",
    "ewm_cov",
    "clip",
    "group_rank_weighted_value",
    "group_decay_linear",
    "group_mean",
    "group_sum",
    "group_min",
    "group_max",
    "group_count",
    "group_normalize",
    "group_rank",
    "group_std",
    "group_zscore",
    "ts_argmax",
    "ts_argmin",
    "ts_quantile",
    "rank_corr",
    "ts_topk_sum",
)


def test_offending_canonicals_still_carry_live_paramspecs_in_catalog() -> None:
    """前提守卫：这 20 个 canonical 在 RAW catalog 里确实是 live ParamSpec。

    如果未来 registry 侧改成自序列化（plain dict），本守卫会失败提醒复查
    ``_sql_contract`` 的 typed serialization 是否还需要。
    """
    for canon in R21_DEF5_CANONICALS:
        entry = OperatorRegistry._catalog.get(canon)
        assert entry is not None, f"{canon} missing from OperatorRegistry catalog"
        specs = entry.get("param_specs") or {}
        live = [n for n, s in specs.items() if isinstance(s, ParamSpec)]
        assert live, f"{canon}: expected live ParamSpec in catalog param_specs"


def test_offending_canonicals_hash_stable_and_idempotent() -> None:
    """每个 offending canonical：stable 64-hex SHA-256，两次计算相等。"""
    for canon in R21_DEF5_CANONICALS:
        contract = _sql_contract(canon)
        h1 = _safe_payload_hash(contract)
        h2 = _safe_payload_hash(_sql_contract(canon))  # 重新读 registry 再算
        assert isinstance(h1, str), canon
        assert len(h1) == 64, f"{canon}: expected 64-hex sha256, got {len(h1)}"
        int(h1, 16)  # hex guard
        assert h1 == h2, f"{canon}: hash not idempotent/stable"


def test_all_catalog_contracts_json_serializable() -> None:
    """负控/结构守卫：catalog 全部 canonical 的 contract 无 live 对象。

    json.dumps 直接成功 = 没有 ParamSpec/type/enum 之类 live 对象漏进 payload。
    """
    serialized = 0
    for canon in sorted(OperatorRegistry._catalog):
        payload = _sql_contract(canon)
        json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        serialized += 1
    assert serialized == len(OperatorRegistry._catalog)


def test_contract_paramspec_fields_typed_serialized() -> None:
    """ParamSpec 序列化为声明字段的 plain dict，无 dataclass/enum/type 残留。"""
    contract = _sql_contract("group_neutralize")
    specs = contract.get("param_specs")
    assert isinstance(specs, dict) and specs
    expected_fields = {f.name for f in __import__("dataclasses").fields(ParamSpec)}
    for name, spec in specs.items():
        assert isinstance(spec, dict), f"{name}: ParamSpec not serialized to dict"
        assert set(spec) == expected_fields, f"{name}: field set mismatch"
        import enum as _enum

        role = spec.get("param_role")
        assert role is None or (
            isinstance(role, str) and not isinstance(role, _enum.Enum)
        ), f"{name}: param_role not plain str: {role!r}"
        dtype = spec.get("dtype")
        assert dtype is None or isinstance(dtype, str), f"{name}: dtype not str/None"
        # text round-trip proves no live object anywhere in the spec dict
        json.dumps(spec, sort_keys=True)


def test_missing_default_sentinel_serialized_deterministically() -> None:
    """MISSING sentinel（无 default 声明）→ 稳定字符串标记，不是 repr。"""
    found_missing = 0
    for canon, entry in OperatorRegistry._catalog.items():
        for pname, spec in (entry.get("param_specs") or {}).items():
            if isinstance(spec, ParamSpec) and spec.default is not None and not isinstance(
                spec.default, (str, int, float, bool)
            ):
                from factor_engine.cleaned_operators.base import MISSING

                if spec.default is MISSING:
                    contract = _sql_contract(canon)
                    got = contract["param_specs"][pname]["default"]
                    assert got == "__MISSING__", f"{canon}.{pname}: {got!r}"
                    found_missing += 1
    assert found_missing > 0, "guard degraded: catalog no longer has MISSING defaults"


def test_unknown_live_object_still_fail_closed() -> None:
    """未知 live 对象 → CapabilityInfrastructureError（不允许 repr/伪 identity）。"""

    class _Foreign:
        pass

    with pytest.raises(CapabilityInfrastructureError):
        _json_contract_value(_Foreign())
    with pytest.raises(CapabilityInfrastructureError):
        _json_contract_value({"nested": {"bad": _Foreign()}})


def test_json_contract_value_plain_passthrough() -> None:
    """纯 JSON 原生值（含 tuple→list、嵌套 dict）原样通过、确定性排序可 dumps。"""
    payload = {
        "a": 1,
        "b": [1, 2.5, "x", None, True],
        "t": (1, 2, (3,)),
        "d": {"k": {"deep": "v"}},
    }
    out = _json_contract_value(payload)
    assert out == {"a": 1, "b": [1, 2.5, "x", None, True], "t": [1, 2, [3]],
                   "d": {"k": {"deep": "v"}}}
    h1 = _safe_payload_hash(_json_contract_value(payload))
    h2 = _safe_payload_hash(_json_contract_value(payload))
    assert h1 == h2 and len(h1) == 64
