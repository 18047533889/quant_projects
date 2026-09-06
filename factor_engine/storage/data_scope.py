# -*- coding: utf-8 -*-
"""数据源窗口指纹：plan/column 缓存键的稳定作用域。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, is_dataclass
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class DataExecutionScope:
    """执行语义作用域：freq/calendar/timezone 等，随 anchor 一起进入 data_scope。

    Round-8 audit #325/#326：``compute_data_scope`` 只绑定 anchor source 属性，
    不包含二级 SourceRef 依赖与执行语义（frequency、calendar、timezone 等）。
    同一份底层数据以不同执行语义读取时应得到不同缓存键；这里把执行语义显式
    纳入指纹，避免缓存跨 freq/日历/时区污染。
    """

    frequency: str = "1d"
    bar_freq: str = ""
    session_calendar: str = ""
    timezone: str = ""
    timestamp_convention: str = ""
    full_history_start: str = ""
    factor_lake_version: str = ""
    adjustment_basis: str = ""
    decision_time_policy: str = ""


@dataclass(frozen=True)
class ExecutionCacheNamespace:
    """The full cache namespace for ONE factor execution (R10-P0-001).

    The old ``_build_cache`` derived the scope from the anchor data source at
    ENGINE-CONSTRUCTION time — before the factor was compiled, so the secondary
    SourceRef dependencies, the execution scope and the operator evidence
    version were NOT in the key.  Two factors sharing an anchor source but
    differing in secondary deps / execution semantics / evidence could
    therefore land in the same namespace and reuse each other's cached subtrees.

    This namespace is built AFTER compile (that is when the source-dependency
    manifest is known) and re-scopes the plan cache for the run.  ``to_scope_key``
    folds it into the same 24-hex prefix ``compute_data_scope`` uses.
    """

    anchor_snapshot_hash: str = ""
    secondary_dependency_hash: str = ""
    execution_scope_hash: str = ""
    operator_evidence_hash: str = ""
    field_contract_hash: str = ""
    universe_hash: str = ""
    # R20-062..066: 实际 resolved universe membership 进入 cache namespace。
    universe_membership_hash: str = ""

    def to_scope_key(self) -> str:
        payload = json.dumps(
            asdict(self),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def compute_execution_cache_scope(
    data_source: Any,
    *,
    execution: DataExecutionScope | None = None,
    source_dependencies: tuple[str, ...] | None = None,
    evidence_version: str = "",
    field_catalog_hash: str = "",
    universe_hash: str = "",
    # R20-062..066: 实际 resolved universe membership 哈希进入 cache namespace。
    universe_membership_hash: str = "",
) -> str:
    """Build the execution cache namespace for one compiled factor (R10-P0-001).

    ``source_dependencies`` is the plan's source-dependency manifest hash
    (``planner.source_dependencies.source_dependency_hash``) — only known after
    compile, which is why the construction-time ``_build_cache`` scope is no
    longer sufficient.

    R20-062..066: ``universe_membership_hash`` 绑定实际 resolved membership
    （按 as-of date 解析），非空 ``instrument_filter`` 与命名 universe 的
    相同标签不再能共享同一 cache namespace。
    """
    ns = ExecutionCacheNamespace(
        anchor_snapshot_hash=compute_data_scope(
            data_source,
            execution=execution,
            source_dependencies=source_dependencies,
        ),
        secondary_dependency_hash=json.dumps(
            sorted(source_dependencies or ()),
            sort_keys=True,
            separators=(",", ":"),
        ),
        execution_scope_hash=(
            json.dumps(asdict(execution), sort_keys=True, separators=(",", ":"))
            if execution is not None
            else ""
        ),
        operator_evidence_hash=str(evidence_version or ""),
        field_contract_hash=str(field_catalog_hash or ""),
        universe_hash=str(universe_hash or ""),
        universe_membership_hash=str(universe_membership_hash or ""),
    )
    return ns.to_scope_key()


def _jsonable(value: Any) -> Any:
    """将常见配置值转为跨进程稳定的 JSON 结构。

    禁止优先使用任意对象 ``repr``：很多对象的 repr 含内存地址，会让同一配置
    在不同 worker 上得到不同 cache key。覆盖 dataclass/Enum/date/datetime/
    Path/list/tuple/set/frozenset/Mapping 后，任何仍无法结构化的对象直接
    ``TypeError`` 失败关闭（Round-8 audit #327）——不再退化为 ``str()``，防止
    内存地址等不稳定表示混入缓存键。
    """
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Enum):
        return _jsonable(value.value)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value.expanduser())
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_jsonable(v) for v in value), key=lambda v: json.dumps(v, sort_keys=True))
    if isinstance(value, Mapping):
        if not all(isinstance(k,str) for k in value):
            raise TypeError("data_scope mapping keys must be strings")
        return {
            k: _jsonable(v)
            for k, v in sorted(value.items())
        }
    raise TypeError(
        f"unsupported data_scope value type: {type(value).__module__}.{type(value).__qualname__}"
    )


def compute_data_scope(
    data_source: Any,
    *,
    execution: DataExecutionScope | None = None,
    source_dependencies: tuple[str, ...] | None = None,
) -> str:
    """从数据源提取稳定作用域指纹，防止跨参数/快照/读模式缓存污染。

    Round-8 audit #325/#326：在 anchor 指纹之外，可选地把执行语义
    （``DataExecutionScope``）与二级 SourceRef 依赖清单（``source_dependencies``）
    纳入指纹。两者都不传时行为与旧版完全一致（向后兼容）。
    """
    payload: dict[str, Any] = {"source_type":f"{type(data_source).__module__}.{type(data_source).__qualname__}"}
    has_stable_identity=False
    for attr in (
        "dataset",
        "start_date",
        "end_date",
        "root",
        "timestamp_column",
        "instrument_column",
        "timestamp_unit",
        "normalize_timestamp",
        "read_auto",
        "lazy_scan",
        "data_snapshot_id",
        # R10-P0-002: these DataAccess source options change the ACTUAL read
        # result or PIT semantics, so they must be part of the scope — two
        # sources identical except ``semantic_filters`` / ``read_mode`` /
        # ``snapshot_now_only`` must NOT share a cache namespace.
        "semantic_filters",
        "read_mode",
        "snapshot_now_only",
        "mining_coverage_threshold",
        "enforce_mining_gate",
        "strict_unknown_fields",
        # #收官轮 P0（Integration）：pit_enforce 改变实际读语义（production 下四层
        # PIT fail-closed vs research 放行），两个仅差 pit_enforce 的 source 不得
        # 共享缓存 namespace。
        "pit_enforce",
        "max_files",
        "recursive",
        "snapshot_id",
        "generation_id",
        "source_hash",
        "content_hash",
        "source_identity",
    ):
        val = getattr(data_source, attr, None)
        if val is not None and (not isinstance(val, str) or val.strip()):
            payload[attr] = _jsonable(val)
            if attr in {"dataset","root","data_snapshot_id","snapshot_id","generation_id","source_hash","content_hash","source_identity"}:
                has_stable_identity=True

    if execution is not None:
        payload["execution"] = _jsonable(asdict(execution))

    if source_dependencies:
        payload["source_dependencies"] = sorted(source_dependencies)

    params = getattr(data_source, "params", None)
    if isinstance(params, Mapping) and params:
        payload["params"] = _jsonable(params)

    fields = getattr(data_source, "fields", None)
    if isinstance(fields, Mapping) and fields:
        payload["fields"] = _jsonable(fields)

    instrument_filter = getattr(data_source, "instrument_filter", None)
    # #收官轮 P0：``None``（全市场）与 ``[]``（空股票池，0 行）是**不同执行语义**，
    # 绝不能共用缓存作用域——否则「先缓存全市场结果 → 后执行空 universe → cache
    # hit → 空 universe 拿到全市场结果」。显式编码 ALL / EMPTY / LIST(...) 三种
    # 身份，保证三者的 cache namespace 互不相同。
    if instrument_filter is None:
        payload["instrument_filter_kind"] = "ALL"
    elif len(instrument_filter) == 0:
        payload["instrument_filter_kind"] = "EMPTY"
    else:
        payload["instrument_filter_kind"] = "LIST"
        payload["instrument_filter"] = _jsonable(sorted(instrument_filter))

    if not has_stable_identity:
        # 没有任何可描述字段的数据源无法跨实例安全复用缓存，只在当前对象生命周期内稳定。
        return f"ephemeral:{id(data_source)}"

    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


# ---------------------------------------------------------------------------
# R20-047..051: 唯一 canonical 执行配置序列化函数
# ---------------------------------------------------------------------------


class ExecutionConfigTypeError(TypeError):
    """执行配置含无法 canonical 序列化的对象（R20-047..051）。

    与 ``_jsonable`` 同语义：unsupported object 直接抛错，绝不 ``str()`` 兜底
    （字符串化含内存地址 / 无法重建）。
    """


def canonicalize_execution_config(value: Any) -> Any:
    """唯一 canonical 执行配置归一化函数（R20-047..051）。

    CSE scope / cache / lineage / identity / snapshot / checkpoint 复用同一逻辑，
    基于 ``_jsonable`` 的 fail-closed typed schema：

    - 覆盖 str/int/float/bool/None/Enum/date/datetime/Path/dataclass/list/
      tuple/set/frozenset/Mapping；
    - float 非有限（NaN/Inf）直接抛 ``NonFiniteLiteralError``-style
      ``ValueError``（禁止进入稳定 key）；
    - 任何其它对象抛 ``ExecutionConfigTypeError``，绝不 ``default=str``
      字符串化。

    配置 **dict 输入**在 hash 前必须先经本函数归一，确保跨进程稳定。
    """
    try:
        return _jsonable(value)
    except TypeError as exc:
        raise ExecutionConfigTypeError(str(exc)) from exc


def canonicalize_execution_config_strict(value: Any) -> Any:
    """``canonicalize_execution_config`` 的别名（显式 strict 语义）。"""
    try:
        return _jsonable(value)
    except TypeError as exc:
        raise ExecutionConfigTypeError(str(exc)) from exc


def execution_scope_key(
    scope: Any,
    *,
    source_dependency_hash: str = "",
    universe_membership_hash: str = "",
) -> str:
    """CSE 执行作用域键：绑定真实二级 SourceRef 依赖 + resolved universe membership。

    R20-058..061: 仅 ``FactorExecutionScope.scope_key()``（factor hint + anchor
    source config）不足以隔离 CSE —— 两个 factor 共享 anchor source 但二级
    SourceRef 依赖不同时不得跨 dependency 共享 CSE。R20-062..066: 命名
    universe 的实际 membership（按 as-of 解析）也必须进入作用域键。

    参数：
        scope: ``planner.dag.FactorExecutionScope`` 或任何带 ``scope_key()``
            的对象；也可传 ``dict``（字段为 scope_key 的 JSON 载荷）。
        source_dependency_hash: ``source_dependency_hash(plan)``
        universe_membership_hash: ``universe_membership_hash(...)``
    """
    if isinstance(scope, dict):
        base = json.dumps(scope, sort_keys=True, separators=(",", ":"))
    else:
        base = scope.scope_key() if hasattr(scope, "scope_key") else json.dumps(
            asdict(scope), sort_keys=True, separators=(",", ":")
        )
    payload = {
        "scope": base,
        "source_dependency_hash": str(source_dependency_hash or ""),
        "universe_membership_hash": str(universe_membership_hash or ""),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
