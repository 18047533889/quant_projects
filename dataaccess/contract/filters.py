"""R25 P0-003/004 —— 结构化过滤契约（FilterRequirement）。

废弃 Store 手工拼 ``_dataset_required_filters`` 的 panel/dimension 二元 tuple，
改成 ``FilterRequirement`` 覆盖 panel / event / dimension / all 四种 scope，
并由 ``validate_filter_requirements`` 在**所有公共读面**统一强制执行。

关键：
    - ``required_event_filters``（US finance timeframe）不再被 Store gate 漏掉；
    - ``cardinality="exactly_one"``（timeframe quarterly/annual/ttm）进入
      CompiledDataRequest digest / identity，季度因子与 TTM 因子不共享 cache/version。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

from data_access.core.exceptions import ValidationError

FilterScope = Literal["panel", "event", "dimension", "all"]
FilterCardinality = Literal["any", "exactly_one", "at_most_one"]


@dataclass(frozen=True)
class FilterRequirement:
    """单条结构化过滤契约（R25 §6）。

    - ``field``       ：过滤字段（如 timeframe / IndustrySource / IndexSymbol / currency）
    - ``scope``       ：适用范围（panel / event / dimension / all）
    - ``required``    ：必须提供（True）或可空（False）
    - ``cardinality`` ：any / exactly_one / at_most_one
    - ``allowed_values``：允许枚举（空 = 不限制）
    """

    field: str
    scope: FilterScope = "all"
    required: bool = True
    cardinality: FilterCardinality = "any"
    allowed_values: tuple[Any, ...] = ()

    def __post_init__(self) -> None:
        if self.scope not in {"panel", "event", "dimension", "all"}:
            raise ValidationError(
                f"FilterRequirement.scope 必须是 panel/event/dimension/all，收到 {self.scope!r}"
            )
        if self.cardinality not in {"any", "exactly_one", "at_most_one"}:
            raise ValidationError(
                f"FilterRequirement.cardinality 必须是 any/exactly_one/at_most_one，"
                f"收到 {self.cardinality!r}"
            )

    @property
    def key(self) -> str:
        return self.field


def _coerce_values(value: Any) -> tuple[Any, ...]:
    """过滤值 → 值元组（标量 → 单元素）。"""
    if value is None:
        return ()
    if isinstance(value, (list, tuple, set, frozenset)):
        return tuple(value)
    return (value,)


def _values_from_mapping(mapping: Mapping[str, Any] | None, field: str) -> tuple[Any, ...]:
    """从 params / filters mapping 提取某字段的值（大小写宽松匹配）。"""
    for key, val in (mapping or {}).items():
        if str(key).lower() == str(field).lower() and val not in (None, ""):
            return _coerce_values(val)
    return ()


def validate_filter_requirements(
    contract: Any,
    *,
    read_mode: str = "auto",
    params: Mapping[str, Any] | None = None,
    filters: Mapping[str, Any] | None = None,
    filters_by_dataset: Mapping[str, Any] | None = None,
    strict: bool | None = None,
) -> None:
    """统一过滤契约校验（P0-003/004）。

    覆盖：
        - params（路径参数）
        - filters（全局过滤）
        - filters_by_dataset（per-dataset 过滤，read_joined）

    对每条 FilterRequirement：
        1. 缺失 required 字段 → fail-closed（production/strict）；
        2. exactly_one：字段值必须恰好 1 个（missing/multiple 拒绝）；
        3. allowed_values：值必须在枚举内（越界拒绝）。

    与旧的 ``_validate_filters``（cos_contract）对齐：strict（production）直接抛
    ValidationError；research 允许 warning 降级（调用方决定是否放行）。返回 None，
    通过异常传达拒绝。
    """
    from data_access.read.query_budget import is_strict_semantics

    effective_strict = strict if strict is not None else is_strict_semantics()
    requirements: tuple[FilterRequirement, ...] = tuple(
        getattr(contract, "filter_requirements", ()) or ()
    )
    if not requirements:
        return

    problems: list[str] = []
    # per-dataset filters 合并：read_joined 的 filters_by_dataset 优先于全局。
    merged_filters = dict(filters or {})
    for ds_filters in (filters_by_dataset or {}).values():
        merged_filters.update(dict(ds_filters or {}))

    for req in requirements:
        pv = _values_from_mapping(params, req.field)
        fv = _values_from_mapping(merged_filters, req.field)
        values: tuple[Any, ...] = ()
        if pv:
            values = pv
        elif fv:
            values = fv
        if req.required and not values:
            problems.append(f"{req.field!r} 缺失（required={req.scope}）")
            continue
        if not values:
            continue  # 非 required 且未提供 → 跳过
        if req.cardinality == "exactly_one" and len(values) != 1:
            problems.append(
                f"{req.field!r} 必须恰好一个值（cardinality=exactly_one），收到 {list(values)!r}"
            )
            continue
        if req.cardinality == "at_most_one" and len(values) > 1:
            problems.append(
                f"{req.field!r} 最多一个值（cardinality=at_most_one），收到 {list(values)!r}"
            )
            continue
        if req.allowed_values:
            bad = [v for v in values if v not in req.allowed_values]
            if bad:
                problems.append(
                    f"{req.field!r}={bad!r} 不在允许值 {list(req.allowed_values)!r}"
                )

    if not problems:
        return
    detail = "; ".join(problems)
    if effective_strict:
        raise ValidationError(
            f"数据集 {getattr(contract, 'dataset', getattr(contract, 'name', '?'))!r} "
            f"过滤契约未满足（production fail-closed）：{detail}"
        )
    import logging

    logging.getLogger("data_access.contract_filters").warning(
        "过滤契约未满足（research 放行）：%s", detail
    )


def build_filter_requirements_from_contract(
    contract: Any,
) -> tuple[FilterRequirement, ...]:
    """从 COSDatasetContract 编译 FilterRequirement（P0-003/004）。

    - required_panel_filters / required_dimension_filters / required_event_filters
      全部进入（**含 event filters**，不再手工漏）。
    - filter_cardinalities（如 ("timeframe","exactly_one")）合并为 cardinality。
    - allowed_filter_values 合并为 allowed_values。
    """
    if contract is None:
        return ()

    reqs: list[FilterRequirement] = []
    by_field: dict[str, dict[str, Any]] = {}

    scope_map = {
        "required_panel_filters": "panel",
        "required_event_filters": "event",
        "required_dimension_filters": "dimension",
    }
    for attr, scope in scope_map.items():
        for field in tuple(getattr(contract, attr, ()) or ()):
            if not field:
                continue
            by_field.setdefault(
                str(field),
                {"scope": scope, "required": True, "cardinality": "any"},
            )

    for field, card in tuple(getattr(contract, "filter_cardinalities", ()) or ()):
        entry = by_field.setdefault(
            str(field),
            {"scope": "all", "required": True, "cardinality": "any"},
        )
        entry["cardinality"] = str(card)

    for field, allowed in dict(getattr(contract, "allowed_filter_values", ()) or {}).items():
        entry = by_field.setdefault(
            str(field),
            {"scope": "all", "required": False, "cardinality": "any"},
        )
        entry["allowed_values"] = tuple(allowed)

    for field, meta in by_field.items():
        reqs.append(
            FilterRequirement(
                field=field,
                scope=meta["scope"],  # type: ignore[arg-type]
                required=bool(meta.get("required", True)),
                cardinality=meta["cardinality"],  # type: ignore[arg-type]
                allowed_values=tuple(meta.get("allowed_values", ()) or ()),
            )
        )
    return tuple(reqs)
