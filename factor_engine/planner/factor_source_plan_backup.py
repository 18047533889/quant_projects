# -*- coding: utf-8 -*-
"""R30-P0-002: FactorSourcePlan —— 单个因子的 source 依赖 IR（FE 侧）。

R30 分工：**FE 负责 dependency extraction（FactorSourcePlan）；DA 不反向解析
factor expression**。本模块只构造/封装依赖 IR 与分组统计，不读任何数据，也不
引入 DuckDB 相关依赖。

``FactorSourcePlan`` 把一个因子的数据依赖折叠成一张确定性快照：

- ``leaf_concepts``：因子表达式直接引用的逻辑字段（concept）；
- ``source_datasets``：这些 concept 所在的物理 dataset；
- ``required_frequency`` / ``required_grain``：读取频率与经济粒度；
- ``pit_requirements``：PIT / decision-policy 要求；
- ``price_basis``：价格口径（raw / backward_adjusted / …）；
- ``aggregations`` / ``joins``：跨源聚合与 join 描述；
- ``coverage_requirements``：最小覆盖率等证据要求。

``identity()`` 是稳定 digest：同一 factor 编译多次 → 同一 digest；
source / market / PIT / price_basis 变化 → identity 变。**绝不依赖 Date.now。**

``extract(...)`` 从既有 source-dependency manifest（``build_source_dependency_
manifest`` 的产物）抽取 leaf_concepts / source_datasets；expression_plan 缺失时
用 overrides 填充。依赖提取仍属 FE——本模块只是把它封装成 R30 命名 IR。
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, Mapping

__all__ = ["FactorSourcePlan", "stable_digest"]


def _json_scalar(value: Any) -> Any:
    """把不可 JSON 序列化的值折叠成可序列化标量/列表。"""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_json_scalar(v) for v in value]
    if isinstance(value, Mapping):
        return {str(k): _json_scalar(v) for k, v in value.items()}
    return str(value)


def _normalize_dict(d: Mapping[str, Any]) -> dict[str, Any]:
    """dict 折叠为排序键、JSON 安全的值视图（identity/digest 用）。"""
    return {
        str(k): _json_scalar(v)
        for k, v in sorted((d or {}).items(), key=lambda kv: str(kv[0]))
    }


def _normalize_grain(grain: Any) -> Any:
    """grain 归一化为可哈希/可序列化形态（tuple 排序列化，字符串原样保留）。

    ``required_grain`` 允许是 ``("instrument", "time")`` 这类经济粒度 tuple，
    也允许是 ``"quarterly"`` / ``"ttm"`` 这类 timeframe 标签字符串（调用方在
    batch-plan 分组时把它解释为 timeframe 语义）。
    """
    if grain is None:
        return None
    if isinstance(grain, (tuple, list, set, frozenset)):
        return tuple(str(g) for g in grain)
    return str(grain)


def stable_digest(payload: Any) -> str:
    """确定性 sha256 前缀 digest（24 hex）。不依赖对象 id / 时间。"""
    raw = json.dumps(
        _json_scalar(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _build_manifest(expression_plan: Any) -> tuple[str, ...] | None:
    """防御性调用 ``planner.source_dependencies.build_source_dependency_manifest``。

    并发 session 可能改签名——任何失败都返回 None，绝不抛。
    """
    try:
        from planner.source_dependencies import build_source_dependency_manifest
    except Exception:  # pragma: no cover - import 环境缺失时降级
        return None
    try:
        manifest = build_source_dependency_manifest(expression_plan)
    except Exception:  # pragma: no cover - 底层签名/语义变化时降级
        return None
    return tuple(manifest or ())


def _parse_manifest(
    manifest: Iterable[Any],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """从 source-dependency manifest 抽取 (leaf_concepts, source_datasets)。

    manifest 项可能是 ``build_source_dependency_manifest`` 的规范 JSON 字符串，
    也可能是 dict（``{"table": ..., "field": ...}``）；两者都只读 table/field，
    解析失败的单条直接跳过（不吞整个 manifest）。
    """
    concepts: set[str] = set()
    datasets: set[str] = set()
    for item in manifest or ():
        payload: Any = None
        if isinstance(item, str):
            try:
                payload = json.loads(item)
            except Exception:  # noqa: BLE001
                continue
        elif isinstance(item, Mapping):
            payload = item
        else:
            payload = getattr(item, "to_dict", lambda: None)()
            if payload is None and item is not None:
                payload = {"table": getattr(item, "table", None),
                           "field": getattr(item, "field", None)}
        if not isinstance(payload, Mapping):
            continue
        table = payload.get("table") or payload.get("dataset")
        field = payload.get("field")
        if table:
            datasets.add(str(table))
        if field:
            concepts.add(str(field))
    return tuple(sorted(concepts)), tuple(sorted(datasets))


class FactorSourcePlan:
    """单个因子的 source 依赖 IR（R30-P0-002）。

    字段全部可 JSON 序列化；``identity()`` 基于 ``to_dict()`` 的稳定 digest。

    额外 ``**extra`` 参数（如 ``timeframe`` / ``universe`` / ``security_scope``）
    会保留到 ``_extra``，供 :mod:`planner.factor_batch_plan` 的分组语义读取——
    FactorSourcePlan 构造签名之外的维度不丢失，也不污染主字段。
    """

    def __init__(
        self,
        factor_id: str,
        market: str,
        leaf_concepts: Iterable[str] = (),
        source_datasets: Iterable[str] = (),
        required_frequency: str | None = None,
        required_grain: Any = None,
        pit_requirements: Mapping[str, Any] | None = None,
        price_basis: str | None = None,
        aggregations: Iterable[Any] = (),
        joins: Iterable[Any] = (),
        coverage_requirements: Mapping[str, Any] | None = None,
        **extra: Any,
    ) -> None:
        self.factor_id = str(factor_id)
        self.market = str(market or "")
        self.leaf_concepts = tuple(sorted(str(c) for c in (leaf_concepts or ())))
        self.source_datasets = tuple(sorted(str(d) for d in (source_datasets or ())))
        self.required_frequency = str(required_frequency) if required_frequency else None
        self.required_grain = _normalize_grain(required_grain)
        self.pit_requirements = dict(pit_requirements or {})
        self.price_basis = str(price_basis) if price_basis else None
        self.aggregations = tuple(aggregations or ())
        self.joins = tuple(joins or ())
        self.coverage_requirements = dict(coverage_requirements or {})
        self._extra = dict(extra or {})

    # -- identity -----------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "factor_id": self.factor_id,
            "market": self.market,
            "leaf_concepts": list(self.leaf_concepts),
            "source_datasets": list(self.source_datasets),
            "required_frequency": self.required_frequency,
            "required_grain": _json_scalar(self.required_grain),
            "pit_requirements": _normalize_dict(self.pit_requirements),
            "price_basis": self.price_basis,
            "aggregations": _json_scalar(list(self.aggregations)),
            "joins": _json_scalar(list(self.joins)),
            "coverage_requirements": _normalize_dict(self.coverage_requirements),
            "extra": _normalize_dict(self._extra),
        }

    def identity(self) -> str:
        """稳定 digest：同一 factor 编译多次 → 同一 digest；source / market /
        PIT / price_basis 变化 → identity 变。"""
        return stable_digest(self.to_dict())

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return (
            f"FactorSourcePlan(factor_id={self.factor_id!r}, market={self.market!r}, "
            f"price_basis={self.price_basis!r}, identity={self.identity()[:8]}…)"
        )

    # -- extraction ---------------------------------------------------------
    @classmethod
    def extract(
        cls,
        factor_id: str,
        market: str,
        expression_plan: Any = None,
        source_manifest: Iterable[Any] | None = None,
        **overrides: Any,
    ) -> "FactorSourcePlan":
        """从既有 source-dependency manifest 抽取依赖并构造 :class:`FactorSourcePlan`。

        - ``source_manifest`` 显式给出 → 直接解析它（``build_source_dependency_
          manifest`` 的产物，元组/可迭代）；
        - ``source_manifest`` 缺省但 ``expression_plan`` 给出 → 先调用
          ``build_source_dependency_manifest`` 再解析；
        - 两者都缺省 → ``overrides`` 里的 ``leaf_concepts`` / ``source_datasets``
          直接作为依赖。

        其余维度（``required_frequency`` / ``price_basis`` / ``pit_requirements``
        / ``aggregations`` / ``joins`` / ``coverage_requirements`` / 任意 ``extra``）
        全部透传 ``overrides``。``expression_plan`` 的依赖提取仍属 FE——本方法只
        是把既有 manifest 封装成 R30 命名 IR。
        """
        leaf_concepts = tuple(str(c) for c in (overrides.pop("leaf_concepts", ()) or ()))
        source_datasets = tuple(str(d) for d in (overrides.pop("source_datasets", ()) or ()))
        if source_manifest is None and expression_plan is not None:
            source_manifest = _build_manifest(expression_plan)
        if source_manifest is not None:
            concepts, datasets = _parse_manifest(source_manifest)
            if not leaf_concepts:
                leaf_concepts = concepts
            if not source_datasets:
                source_datasets = datasets
        return cls(
            factor_id=factor_id,
            market=market,
            leaf_concepts=leaf_concepts,
            source_datasets=source_datasets,
            **overrides,
        )
