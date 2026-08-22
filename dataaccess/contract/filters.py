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


def _scope_applies(scope: str, read_mode: str) -> bool:
    """FilterRequirement.scope 是否在当前 read_mode 生效（R26-P0-012）。"""
    if scope == "all":
        return True
    if scope == "panel":
        return read_mode in {"panel", "auto"}
    if scope == "event":
        return read_mode in {"event", "pit", "auto"}
    if scope == "dimension":
        return read_mode in {"dimension", "auto"}
    return True


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
    dataset: str | None = None,
    read_mode: str = "auto",
    params: Mapping[str, Any] | None = None,
    filters: Mapping[str, Any] | None = None,
    filters_by_dataset: Mapping[str, Any] | None = None,
    strict: bool | None = None,
) -> None:
    """统一过滤契约校验（P0-003/004 + R26-P0-012 **per-dataset**）。

    覆盖：
        - params（路径参数）
        - filters（全局过滤）
        - filters_by_dataset（per-dataset 过滤，read_joined）——**只取当前
          dataset 自己的过滤器**，绝不把 B dataset 的 filter merge 进 A 去满足
          A 的 requirement（R26-P0-012 跨 dataset 污染）。

    对每条 FilterRequirement：
        1. 缺失 required 字段 → fail-closed（production/strict）；
        2. exactly_one：字段值必须恰好 1 个（missing/multiple 拒绝）；
        3. allowed_values：值必须在枚举内（越界拒绝）。

    ``read_mode``：FilterRequirement.scope（panel/event/dimension/all）生效判定。
        - panel requirement  只在 read_mode ∈ {panel, auto} 生效
        - event requirement  在 read_mode ∈ {event, pit} 生效
        - all               始终生效

    与旧的 ``_validate_filters``（cos_contract）对齐：strict（production）直接抛
    ValidationError；research 允许 warning 降级（调用方决定是否放行）。返回 None，
    通过异常传达拒绝。
    """
    from data_access.read.query_budget import is_strict_semantics

    effective_strict = strict if strict is not None else is_strict_semantics()
    # RuntimeDatasetContract.filters 或裸 contract.filter_requirements 都能用。
    requirements: tuple[FilterRequirement, ...] = tuple(
        getattr(contract, "filters", None)
        or getattr(contract, "filter_requirements", ())
        or ()
    )
    if not requirements:
        return
    if dataset is None:
        dataset = getattr(contract, "dataset", None) or getattr(contract, "name", None)

    problems: list[str] = []
    # R26-P0-012：per-dataset——只用 filters_by_dataset[dataset]（若提供），
    # 不做跨 dataset 合并。
    merged_filters = dict(filters or {})
    if dataset is not None and filters_by_dataset is not None:
        ds_filters = filters_by_dataset.get(dataset)
        if ds_filters:
            merged_filters.update(dict(ds_filters))
    elif filters_by_dataset:
        import logging

        logging.getLogger("data_access.contract_filters").warning(
            "validate_filter_requirements 未指定 dataset，filters_by_dataset 被忽略"
            "（R26-P0-012：禁止跨 dataset merge）"
        )

    mode = str(read_mode or "auto").strip().lower()
    for req in requirements:
        if not _scope_applies(req.scope, mode):
            continue
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
        # R26-P0-008：required 只接受真实 bool（internal 构造，但禁止 "false"→True）。
        required = meta.get("required", True)
        if not isinstance(required, bool):
            raise ValidationError(
                f"FilterRequirement[{field}].required 必须是布尔值，收到 {required!r}"
            )
        reqs.append(
            FilterRequirement(
                field=field,
                scope=meta["scope"],  # type: ignore[arg-type]
                required=required,
                cardinality=meta["cardinality"],  # type: ignore[arg-type]
                allowed_values=tuple(meta.get("allowed_values", ()) or ()),
            )
        )
    return tuple(reqs)
