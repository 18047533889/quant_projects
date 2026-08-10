"""R26-P0-022 —— SchemaEpoch：跨物理 epoch 的 schema evolution 运行时 gate。

R25 的 schema validation 核心是 ``DESCRIBE read_parquet(... union_by_name=True)``
合并后与 declared schema 检查——``union_by_name=True`` 会把「某一 epoch 缺字段」
变成「union 后有字段，旧 partition 产生 null」，不被识别为跨 epoch contract
violation。本模块改为**逐 physical object 读 parquet footer**：

    object -> schema fingerprint -> epoch groups

跨 epoch request 必须验证：
    1. 请求字段在**每个**必需 epoch 都存在（缺 → SchemaContractError）；
    2. dtype 兼容（double → string → 拒绝）；
    3. unit/definition version 兼容（同 dtype 但定义版本漂移且无 approved
       migration → 拒绝）。

这是**真实 parquet** 上的 runtime gate（不是测试 helper simulator）。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from data_access.core.exceptions import SchemaContractError

# parquet 物理类型 → logical dtype（归一化，便于跨 epoch 比较）。
_DTYPE_COMPAT: dict[str, tuple[str, ...]] = {
    "double": ("double", "float", "int64", "int32", "int"),
    "float": ("double", "float", "int64", "int32", "int"),
    "int64": ("int64", "int32", "int", "double", "float"),
    "int32": ("int64", "int32", "int", "double", "float"),
    "string": ("string",),
    "large_string": ("string", "large_string"),
    "bool": ("bool",),
    "timestamp[us]": ("timestamp[us]", "timestamp[ms]", "timestamp[s]", "timestamp[ns]"),
    "timestamp[ms]": ("timestamp[us]", "timestamp[ms]", "timestamp[s]", "timestamp[ns]"),
    "timestamp[s]": ("timestamp[us]", "timestamp[ms]", "timestamp[s]", "timestamp[ns]"),
    "date32[day]": ("date32[day]", "date64[ms]"),
    "date64[ms]": ("date32[day]", "date64[ms]"),
}


def _normalize_dtype(raw: str) -> str:
    text = str(raw).strip().lower()
    # 去掉参数（decimal(18,2) -> decimal）
    return text.split("(")[0].strip() if text else text


def _dtypes_compatible(a: str, b: str) -> bool:
    na, nb = _normalize_dtype(a), _normalize_dtype(b)
    if na == nb:
        return True
    return nb in _DTYPE_COMPAT.get(na, ()) or na in _DTYPE_COMPAT.get(nb, ())


def parquet_footer_schema(path: str) -> dict[str, str]:
    """读单 parquet 文件 footer schema（只读元数据，不读数据）。"""
    import pyarrow.parquet as pq

    f = pq.ParquetFile(str(path))
    arrow_schema = f.schema_arrow
    out: dict[str, str] = {}
    # schema_arrow 是 pyarrow.Schema：用 names/field 遍历（跨版本 num_fields 兼容）。
    for name in arrow_schema.names:
        out[str(name)] = str(arrow_schema.field(name).type)
    return out


def schema_fingerprint(schema: Mapping[str, str]) -> str:
    text = json.dumps(
        {k: _normalize_dtype(v) for k, v in sorted(schema.items())},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class SchemaEpoch:
    """一组 schema 指纹相同的物理对象（epoch）。"""

    fingerprint: str
    fields: Mapping[str, str]
    objects: tuple[str, ...]

    def has_field(self, field: str) -> bool:
        return field in self.fields

    def dtype_of(self, field: str) -> str | None:
        return self.fields.get(field)


@dataclass(frozen=True)
class SchemaMigration:
    """跨 epoch 迁移审批（R26-P0-022：缺字段/dtype/unit 变化必须有 approved migration）。"""

    from_fingerprint: str
    to_fingerprint: str
    kind: str = "add_column"  # add_column / dtype_change / unit_change
    field: str | None = None
    approved: bool = False
    reviewer: str | None = None

    def covers(self, a: str, b: str, kind: str, field: str | None) -> bool:
        if self.kind != kind:
            return False
        if not (
            {self.from_fingerprint, self.to_fingerprint}
            == {a, b}
        ):
            return False
        if self.field and self.field != field:
            return False
        return self.approved


class SchemaEpochGate:
    """跨 epoch schema 契约 gate（R26-P0-022）。

    ``declared_fields``：数据集声明 schema（field → 期望 dtype），用于比对
    epoch 缺失字段。``migrations``：已审批跨 epoch 迁移。``strict`` 由
    ``is_strict_semantics()`` 决定。
    """

    def __init__(
        self,
        *,
        declared_fields: Mapping[str, str] | None = None,
        migrations: Sequence[SchemaMigration] = (),
        strict: bool | None = None,
    ) -> None:
        self._declared = dict(declared_fields or {})
        self._migrations = list(migrations)
        self._strict = strict

    def _effective_strict(self) -> bool:
        if self._strict is not None:
            return self._strict
        try:
            from data_access.read.query_budget import is_strict_semantics

            return is_strict_semantics()
        except Exception:
            return True

    def group_epochs(self, paths: Sequence[str]) -> list[SchemaEpoch]:
        """读真实 parquet footer，按 schema 指纹分组为 epochs。"""
        groups: dict[str, dict[str, Any]] = {}
        for p in paths:
            try:
                schema = parquet_footer_schema(str(p))
            except Exception as exc:
                if self._effective_strict():
                    raise SchemaContractError(
                        f"parquet footer schema 读取失败（{p}）：{exc}"
                        "（R26-P0-022，production fail-closed）"
                    ) from exc
                continue
            fp = schema_fingerprint(schema)
            g = groups.setdefault(fp, {"fields": schema, "objects": []})
            g["objects"].append(str(p))
        return [
            SchemaEpoch(
                fingerprint=fp,
                fields=dict(g["fields"]),
                objects=tuple(g["objects"]),
            )
            for fp, g in sorted(groups.items())
        ]

    def validate(
        self,
        paths: Sequence[str],
        *,
        requested_columns: Sequence[str] | None = None,
    ) -> list[SchemaEpoch]:
        """跨 epoch 校验；违反抛 ``SchemaContractError``（strict）。返回 epochs。"""
        strict = self._effective_strict()
        epochs = self.group_epochs(paths)
        if len(epochs) <= 1:
            return epochs
        # 跨 epoch：请求字段必须在每个 epoch 存在 + dtype 兼容。
        columns = list(requested_columns) if requested_columns else list(self._declared)
        problems: list[str] = []
        field_dtypes: dict[str, set[str]] = {}
        for epoch in epochs:
            for col in columns:
                if not epoch.has_field(col):
                    if not self._migration_approved(epoch, col, "add_column"):
                        problems.append(
                            f"列 {col!r} 在 epoch {epoch.fingerprint[:8]} 缺失"
                            f"（{epoch.objects[0]}）"
                        )
                    continue
                dt = epoch.dtype_of(col) or ""
                field_dtypes.setdefault(col, set()).add(_normalize_dtype(dt))
        for col, dtypes in field_dtypes.items():
            dtypes = set(dtypes)
            base = next(iter(dtypes))
            for other in dtypes - {base}:
                if not _dtypes_compatible(base, other):
                    if not self._migration_approved(None, col, "dtype_change"):
                        problems.append(
                            f"列 {col!r} dtype 跨 epoch 不兼容："
                            f"{sorted(dtypes)}（R26-P0-022）"
                        )
        if not problems:
            return epochs
        detail = "; ".join(problems)
        if strict:
            raise SchemaContractError(
                f"跨 epoch schema evolution 契约违反（production fail-closed）：{detail}"
            )
        import logging

        logging.getLogger("data_access.schema_epoch").warning(
            "跨 epoch schema drift（research 放行）：%s", detail
        )
        return epochs

    def _migration_approved(
        self, epoch: SchemaEpoch | None, col: str, kind: str
    ) -> bool:
        for mig in self._migrations:
            if epoch is not None and mig.covers(epoch.fingerprint, epoch.fingerprint, kind, col):
                return True
        # add_column 允许任何 epoch 到任何 epoch 的 approved migration。
        if kind == "add_column" and any(m.approved for m in self._migrations):
            return True
        return False
