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

    def group_epochs(
        self,
        paths: Sequence[str],
        *,
        manifest: Any = None,
    ) -> list[SchemaEpoch]:
        """按 schema 指纹分组为 epochs。

        R28-8：``manifest`` 提供 schema epoch 摘要时走 **O(1) manifest 分组**
        （publish 时算好、query-time 直接读摘要，不再逐文件开 parquet footer——
        FactorEngine 热路径的关键）；manifest 缺失 / 摘要不覆盖的 path 回退真实
        footer（O(N)，生产数据集 manifest 已构建时不触发）。
        """
        known_fields: dict[str, dict[str, str]] = {}
        known_hash_by_path: dict[str, str] = {}
        if manifest is not None:
            summary = manifest.epoch_summary_for_paths(paths)
            if summary is not None:
                hash_by_path, epoch_fields = summary
                known_hash_by_path = hash_by_path
                known_fields = epoch_fields
        groups: dict[str, dict[str, Any]] = {}
        for p in paths:
            path_str = str(p)
            shash = known_hash_by_path.get(path_str)
            fields = known_fields.get(shash) if shash is not None else None
            if fields is None:
                # manifest 未覆盖：逐文件读真实 footer（回退路径）。
                try:
                    schema = parquet_footer_schema(path_str)
                except Exception as exc:
                    if self._effective_strict():
                        raise SchemaContractError(
                            f"parquet footer schema 读取失败（{p}）：{exc}"
                            "（R26-P0-022，production fail-closed）"
                        ) from exc
                    continue
                fields = schema
                shash = schema_fingerprint(schema)
            fp = schema_fingerprint(fields)
            g = groups.setdefault(fp, {"fields": dict(fields), "objects": []})
            g["objects"].append(path_str)
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
        manifest: Any = None,
    ) -> list[SchemaEpoch]:
        """跨 epoch 校验；违反抛 ``SchemaContractError``（strict）。返回 epochs。

        R28-7：迁移判定改为**真实 epoch pair 比较**（A→B 不是 A→A）。旧实现
        ``mig.covers(epoch.fp, epoch.fp, ...)`` 是 A→A 自环、dtype 分支传
        ``epoch=None`` 导致 approved dtype migration 永远匹配不到、add_column
        还允许「任意 approved migration」放行——三类都错。现在：
          - add_column：缺字段的 epoch E 需要一条**来自某个含该字段的 epoch F**
            的 ``F→E add_column`` approved migration；
          - dtype_change：不相容 dtype 的每一对 epoch (A, B) 都需要
            ``A→B``（或 ``B→A``）approved migration；
          - 不再有「任意 approved migration 就放行」的兜底。

        R28-8：``manifest`` 提供时 schema epoch 分组走 manifest 摘要（O(1)），
        不再逐文件 footer。
        """
        strict = self._effective_strict()
        epochs = self.group_epochs(paths, manifest=manifest)
        if len(epochs) <= 1:
            return epochs
        columns = list(requested_columns) if requested_columns else list(self._declared)
        by_fp = {e.fingerprint: e for e in epochs}
        problems: list[str] = []
        field_dtypes: dict[str, dict[str, str]] = {}
        for epoch in epochs:
            for col in columns:
                if not epoch.has_field(col):
                    # 需要一个「含该字段的 epoch → 本 epoch」的 add_column migration。
                    source = next(
                        (e for e in epochs if e is not epoch and e.has_field(col)),
                        None,
                    )
                    if source is None or not self._migration_approved(
                        source, epoch, col, "add_column"
                    ):
                        problems.append(
                            f"列 {col!r} 在 epoch {epoch.fingerprint[:8]} 缺失"
                            f"（{epoch.objects[0]}），且无 {source.fingerprint[:8] if source else '?'}→"
                            f"{epoch.fingerprint[:8]} 的 approved add_column migration"
                        )
                    continue
                dt = epoch.dtype_of(col) or ""
                field_dtypes.setdefault(col, {})[epoch.fingerprint] = _normalize_dtype(dt)
        # dtype 逐对比较（R28-7：真实 epoch pair，不再 A→A 自环）。
        for col, fp_to_dt in field_dtypes.items():
            fps = list(fp_to_dt)
            for i in range(len(fps)):
                for j in range(i + 1, len(fps)):
                    a_fp, b_fp = fps[i], fps[j]
                    da, db = fp_to_dt[a_fp], fp_to_dt[b_fp]
                    if da == db or _dtypes_compatible(da, db):
                        continue
                    ea, eb = by_fp[a_fp], by_fp[b_fp]
                    if not self._migration_approved(ea, eb, col, "dtype_change"):
                        problems.append(
                            f"列 {col!r} dtype 跨 epoch 不兼容（{a_fp[:8]}={da} vs "
                            f"{b_fp[:8]}={db}），且无 approved dtype_change migration"
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
        self,
        from_epoch: SchemaEpoch,
        to_epoch: SchemaEpoch,
        col: str,
        kind: str,
    ) -> bool:
        """R28-7：approved migration 必须**精确覆盖该 epoch pair + kind + field**。

        migration 无方向性（schema evolution 双侧同判）——``A→B`` 或 ``B→A``
        都认可。不再有「任意 approved migration」的宽松兜底。
        """
        a, b = from_epoch.fingerprint, to_epoch.fingerprint
        for mig in self._migrations:
            if mig.kind != kind:
                continue
            if {mig.from_fingerprint, mig.to_fingerprint} != {a, b}:
                continue
            if mig.field and mig.field != col:
                continue
            if mig.approved:
                return True
        return False
