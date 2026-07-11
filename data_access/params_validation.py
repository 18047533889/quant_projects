# -*- coding: utf-8
"""参数化数据集 params 校验与 canonicalize。"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping

from .exceptions import ValidationError
from .read_contract import canonicalize_params

_UNSAFE_PATH_CHARS = re.compile(r"[/\\:\0]")
_PATH_SEGMENT_DEFAULT = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")


@dataclass(frozen=True)
class ParamSpec:
    """单个 params_schema 字段规格。"""

    name: str
    type: str = "str"
    pattern: str | None = None
    path_segment: bool = False
    enum_values: tuple[str, ...] = ()
    min_int: int | None = None
    max_int: int | None = None
    max_length: int = 256

    @classmethod
    def from_yaml_value(cls, name: str, raw: Any) -> "ParamSpec":
        if isinstance(raw, str):
            return cls(name=name, type=raw.lower())
        if not isinstance(raw, dict):
            raise ValidationError(
                f"params_schema['{name}'] 必须是字符串或 mapping，收到 {type(raw).__name__}"
            )
        ptype = str(raw.get("type", "str")).lower()
        pattern = raw.get("pattern")
        if pattern is not None:
            pattern = str(pattern)
        enum_raw = raw.get("values") or raw.get("enum")
        enum_values: tuple[str, ...] = ()
        if enum_raw is not None:
            if not isinstance(enum_raw, (list, tuple)):
                raise ValidationError(f"params_schema['{name}'].values 必须是列表")
            enum_values = tuple(str(v) for v in enum_raw)
        min_int = raw.get("min")
        max_int = raw.get("max")
        return cls(
            name=name,
            type=ptype,
            pattern=pattern,
            path_segment=bool(raw.get("path_segment", False)),
            enum_values=enum_values,
            min_int=int(min_int) if min_int is not None else None,
            max_int=int(max_int) if max_int is not None else None,
            max_length=int(raw.get("max_length", 256)),
        )


def parse_params_schema(raw: Mapping[str, Any]) -> dict[str, ParamSpec]:
    if not raw:
        raise ValidationError("parametric 数据集必须声明非空 params_schema")
    out: dict[str, ParamSpec] = {}
    for key, val in raw.items():
        out[str(key)] = ParamSpec.from_yaml_value(str(key), val)
    return out


def validate_params(
    dataset_name: str,
    specs: Mapping[str, ParamSpec],
    params: Mapping[str, Any],
) -> dict[str, Any]:
    """校验并规范化 params；通过后才允许进入路径模板 .format()。"""
    missing = [k for k in specs if k not in params]
    if missing:
        raise ValidationError(
            f"参数化数据集 '{dataset_name}' 缺参数：{missing}；必填：{list(specs)}"
        )
    extra = set(params) - set(specs)
    if extra:
        raise ValidationError(
            f"数据集 '{dataset_name}' 收到未登记参数：{sorted(extra)}"
        )

    normalized: dict[str, Any] = {}
    for name, spec in specs.items():
        raw = params[name]
        normalized[name] = _validate_one(dataset_name, spec, raw)
    return normalized


def _validate_one(dataset_name: str, spec: ParamSpec, raw: Any) -> Any:
    ctx = f"数据集 '{dataset_name}' 参数 '{spec.name}'"

    if spec.type == "int":
        if isinstance(raw, bool) or not isinstance(raw, int):
            try:
                val = int(raw)
            except (TypeError, ValueError) as exc:
                raise ValidationError(f"{ctx} 必须是整数，收到 {raw!r}") from exc
        else:
            val = raw
        if spec.min_int is not None and val < spec.min_int:
            raise ValidationError(f"{ctx} 小于最小值 {spec.min_int}")
        if spec.max_int is not None and val > spec.max_int:
            raise ValidationError(f"{ctx} 大于最大值 {spec.max_int}")
        return val

    text = str(raw)
    if not text:
        raise ValidationError(f"{ctx} 不能为空字符串")
    if len(text) > spec.max_length:
        raise ValidationError(f"{ctx} 长度超过 {spec.max_length}")
    if _UNSAFE_PATH_CHARS.search(text):
        raise ValidationError(f"{ctx} 含非法路径字符（/ \\ : NUL）")

    if spec.enum_values and text not in spec.enum_values:
        raise ValidationError(
            f"{ctx} 必须是 {list(spec.enum_values)} 之一，收到 {text!r}"
        )

    pattern = spec.pattern
    if spec.path_segment and pattern is None:
        pattern = _PATH_SEGMENT_DEFAULT.pattern
    if pattern is not None and not re.fullmatch(pattern, text):
        raise ValidationError(f"{ctx} 不匹配 pattern {pattern!r}，收到 {text!r}")

    if spec.path_segment and (".." in text or text in {".", ".."}):
        raise ValidationError(f"{ctx} 不能是路径遍历片段")

    return text


def params_fingerprint(params: Mapping[str, Any] | None) -> str:
    import hashlib
    import json

    canon = canonicalize_params(params)
    text = json.dumps(canon, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
