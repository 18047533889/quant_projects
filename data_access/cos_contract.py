# -*- coding: utf-8 -*-
"""Executable semantic contracts for COS-backed datasets."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Mapping

import numpy as np
import pyarrow as pa

from data_access.core.exceptions import ValidationError

_MODELS = {"D1", "E1", "E2", "S1", "X0", "STATIC", "EMPTY", "MINUTE"}
_PANEL = {"dense", "state_ready", "minute", "event_only", "sparse", "dimension", "forbidden"}
_PIT = {"strict", "effective_time_only", "unsupported", "not_applicable"}


@dataclass(frozen=True)
class COSDatasetContract:
    name: str
    market: str
    temporal_model: str
    join_policy: str
    instrument_column: str
    panel_policy: str
    pit_policy: str = "not_applicable"
    availability_column: str | None = None
    period_column: str | None = None
    event_column: str | None = None
    revision_columns: tuple[str, ...] = ()
    event_id_columns: tuple[str, ...] = ()
    required_panel_filters: tuple[str, ...] = ()
    required_event_filters: tuple[str, ...] = ()
    allowed_filter_values: tuple[tuple[str, tuple[Any, ...]], ...] = ()
    return_column: str | None = None
    return_scale: float = 1.0
    adjustment_column: str | None = None
    adjustment_convention: str | None = None
    availability_must_follow_period: bool = False
    storage_layout: str | None = None
    note: str = ""

    def __post_init__(self) -> None:
        if self.temporal_model not in _MODELS or self.panel_policy not in _PANEL or self.pit_policy not in _PIT:
            raise ValueError(f"invalid COS contract: {self.name}")
        if self.pit_policy == "strict" and not self.availability_column:
            raise ValueError(f"strict PIT requires availability_column: {self.name}")
        if self.pit_policy == "effective_time_only" and not self.event_column:
            raise ValueError(f"effective-time PIT requires event_column: {self.name}")

    @property
    def is_event(self) -> bool:
        return self.temporal_model in {"E1", "E2"}

    @property
    def is_sparse(self) -> bool:
        return self.temporal_model == "X0"

    @property
    def is_empty(self) -> bool:
        return self.temporal_model == "EMPTY"

    @property
    def is_static(self) -> bool:
        return self.temporal_model == "STATIC"

    @property
    def factor_panel_allowed(self) -> bool:
        return self.panel_policy in {"dense", "state_ready", "minute"}

    @property
    def allowed_filters(self) -> dict[str, tuple[Any, ...]]:
        return dict(self.allowed_filter_values)


def _c(name: str, market: str, model: str, join: str, instrument: str, *, panel: str | None = None, pit: str | None = None, **kwargs: Any) -> COSDatasetContract:
    default_panel = {"D1": "dense", "S1": "state_ready", "MINUTE": "minute", "E1": "event_only", "E2": "event_only", "X0": "sparse", "STATIC": "dimension", "EMPTY": "forbidden"}[model]
    default_pit = "strict" if model in {"E1", "E2"} and kwargs.get("availability_column") else "not_applicable"
    return COSDatasetContract(name, market, model, join, instrument, panel or default_panel, pit or default_pit, **kwargs)


from .cos_contract_ashare import ASHARE_COS_CONTRACTS  # noqa: E402
from .cos_contract_us import US_COS_CONTRACTS  # noqa: E402

COS_DATASET_CONTRACTS = {**ASHARE_COS_CONTRACTS, **US_COS_CONTRACTS}


def get_cos_contract(dataset: str) -> COSDatasetContract | None:
    return COS_DATASET_CONTRACTS.get(str(dataset))


def require_cos_contract(dataset: str) -> COSDatasetContract:
    contract = get_cos_contract(dataset)
    if contract is None:
        raise ValidationError(f"数据集 {dataset!r} 没有 COS 语义契约；不得推断 panel/PIT 语义")
    return contract


def _validate_filters(contract: COSDatasetContract, filters: Mapping[str, Any] | None, required: tuple[str, ...], context: str) -> dict[str, Any]:
    supplied = dict(filters or {})
    missing = [name for name in required if name not in supplied or supplied[name] in (None, "")]
    if missing:
        raise ValidationError(f"数据集 {contract.name!r} 的{context}必须显式提供过滤条件 {missing}")
    for name, value in supplied.items():
        choices = contract.allowed_filters.get(name)
        if choices is None:
            continue
        values = value if isinstance(value, (list, tuple, set, frozenset)) else (value,)
        bad = [item for item in values if item not in choices]
        if bad:
            raise ValidationError(f"数据集 {contract.name!r} 的过滤条件 {name}={bad!r} 不在允许值 {choices!r} 中")
    return supplied


def validate_panel_request(dataset: str, *, semantic_filters: Mapping[str, Any] | None = None, allow_sparse: bool = False) -> COSDatasetContract:
    contract = require_cos_contract(dataset)
    if contract.is_empty:
        raise ValidationError(f"数据集 {dataset!r} 是 EMPTY 占位表，禁止作为数据源")
    if contract.is_event:
        raise ValidationError(f"数据集 {dataset!r} 是 {contract.temporal_model} 事件表；必须使用事件/PIT API")
    if contract.is_sparse and not allow_sparse:
        raise ValidationError(f"数据集 {dataset!r} 是 X0 稀疏表，不是完整日频面板")
    if contract.is_static:
        raise ValidationError(f"数据集 {dataset!r} 是 STATIC 维表，不是时间×标的面板")
    if not contract.factor_panel_allowed and not (contract.is_sparse and allow_sparse):
        raise ValidationError(f"数据集 {dataset!r} panel_policy={contract.panel_policy!r}，禁止面板化")
    _validate_filters(contract, semantic_filters, contract.required_panel_filters, " panel 读取")
    return contract


def resolve_event_clock(dataset: str, *, allow_effective_time: bool = False) -> tuple[COSDatasetContract, str]:
    contract = require_cos_contract(dataset)
    if not contract.is_event:
        raise ValidationError(f"数据集 {dataset!r} 不是 E1/E2 事件表")
    if contract.pit_policy == "strict" and contract.availability_column:
        return contract, contract.availability_column
    if contract.pit_policy == "effective_time_only" and contract.event_column:
        if not allow_effective_time:
            raise ValidationError(f"数据集 {dataset!r} 没有可靠公告/可知时间；仅可显式按生效日保守使用")
        return contract, contract.event_column
    raise ValidationError(f"数据集 {dataset!r} pit_policy={contract.pit_policy!r}，不支持 PIT")


def validate_event_filters(contract: COSDatasetContract, event_filters: Mapping[str, Any] | None) -> dict[str, Any]:
    return _validate_filters(contract, event_filters, contract.required_event_filters, "事件/PIT 读取")


def normalize_return_values(values: Any, dataset: str) -> Any:
    contract = require_cos_contract(dataset)
    if contract.return_column is None:
        raise ValidationError(f"数据集 {dataset!r} 未声明收益字段")
    scale = float(contract.return_scale)
    if isinstance(values, (pa.Array, pa.ChunkedArray)):
        import pyarrow.compute as pc
        return pc.multiply(pc.cast(values, pa.float64(), safe=False), pa.scalar(scale, type=pa.float64()))
    return values.astype(float) * scale if hasattr(values, "astype") else np.asarray(values, dtype=float) * scale


def semantic_contract_fingerprint() -> str:
    payload = {name: asdict(contract) for name, contract in sorted(COS_DATASET_CONTRACTS.items())}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()[:16]


__all__ = ["COSDatasetContract", "COS_DATASET_CONTRACTS", "get_cos_contract", "require_cos_contract", "validate_panel_request", "resolve_event_clock", "validate_event_filters", "normalize_return_values", "semantic_contract_fingerprint"]
