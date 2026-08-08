"""
data_access.read.factors —— 因子目录（FactorCatalog）与批量因子读

职责
    1. ``FactorMeta``：单因子的元数据（factor_id/version/frequency/universe/
       dtype/start/end/instruments/storage_uri/layout/schema_hash/formula_hash/
       operator_hash/data_snapshot/created/status）
    2. ``FactorCatalog``：因子湖内的目录清单（``_factor_catalog.json``），
       聚合各因子目录下的 ``_factor_meta.json``
    3. ``build_factor_union_sql`` / 宽表 pivot：一次查询批量读多因子，
       避免 ``for fid in ids: store.read(factor_id=fid)`` 逐因子循环

设计要点
    1. 长表读：UNION ALL，每分支注入字面量 ``factor_id`` 虚拟列——单次 DuckDB
       查询读 N 个因子，一次执行。
    2. 宽表读：DuckDB ``PIVOT`` 把 (datetime, asset, factor_id, value) 转成
       (datetime, asset, f1, f2, ...) 矩阵，等价 factor_matrix 物化层的读法。
    3. 物理上仍是 factor_id 一棵树；目录层负责「一次问多个因子」的聚合，
       不改变因子湖的存储布局。

非职责
    不计算因子；不写因子（写走 write_arrow / publish）；不推断 schema。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from data_access.core.exceptions import DataError, ValidationError

CATALOG_FILENAME = "_factor_catalog.json"
FACTOR_META_FILENAME = "_factor_meta.json"


@dataclass(frozen=True)
class FactorMeta:
    """单因子元数据。"""

    factor_id: str
    factor_version: str | None = None
    frequency: str | None = None
    universe: str | None = None
    dtype: str | None = None
    start_time: str | None = None
    end_time: str | None = None
    instruments: int | None = None
    storage_uri: str | None = None
    storage_layout: str | None = None
    schema_hash: str | None = None
    formula_hash: str | None = None
    operator_hash: str | None = None
    data_snapshot: str | None = None
    created_at: str | None = None
    status: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "factor_id": self.factor_id,
            "factor_version": self.factor_version,
            "frequency": self.frequency,
            "universe": self.universe,
            "dtype": self.dtype,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "instruments": self.instruments,
            "storage_uri": self.storage_uri,
            "storage_layout": self.storage_layout,
            "schema_hash": self.schema_hash,
            "formula_hash": self.formula_hash,
            "operator_hash": self.operator_hash,
            "data_snapshot": self.data_snapshot,
            "created_at": self.created_at,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any], *, strict: bool = False) -> "FactorMeta":
        """从 JSON dict 构建 FactorMeta。

        #P0-final closure 9：``strict``（production）下非法值（``instruments``
        不是整数、缺 ``factor_id`` 等）fail-closed 抛错——损坏的元数据不能
        静默变成 ``None`` 进权威 catalog；research/recovery 才允许宽容。
        """
        fid = str(payload.get("factor_id", "")).strip()
        if not fid:
            raise ValidationError("FactorMeta 缺少 factor_id")
        return cls(
            factor_id=fid,
            factor_version=_opt_str(payload.get("factor_version")),
            frequency=_opt_str(payload.get("frequency")),
            universe=_opt_str(payload.get("universe")),
            dtype=_opt_str(payload.get("dtype")),
            start_time=_opt_str(payload.get("start_time")),
            end_time=_opt_str(payload.get("end_time")),
            instruments=_opt_int(payload.get("instruments"), strict=strict, field="instruments"),
            storage_uri=_opt_str(payload.get("storage_uri")),
            storage_layout=_opt_str(payload.get("storage_layout")),
            schema_hash=_opt_str(payload.get("schema_hash")),
            formula_hash=_opt_str(payload.get("formula_hash")),
            operator_hash=_opt_str(payload.get("operator_hash")),
            data_snapshot=_opt_str(payload.get("data_snapshot")),
            created_at=_opt_str(payload.get("created_at")),
            status=_opt_str(payload.get("status")),
        )


def _opt_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _opt_int(value: Any, *, strict: bool = False, field: str = "int") -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        if strict:
            raise ValidationError(f"FactorMeta.{field} 必须是整数，收到 bool {value!r}")
        return None
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        if strict:
            raise ValidationError(f"FactorMeta.{field} 必须是整数，收到 {value!r}")
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        if strict:
            raise ValidationError(f"FactorMeta.{field} 必须是整数，收到 {value!r}")
        return None


def _strict() -> bool:
    from data_access.read.query_budget import is_strict_semantics

    return is_strict_semantics()


@dataclass
class FactorCatalog:
    """因子湖目录：factor_id → FactorMeta。"""

    root: Path
    records: dict[str, FactorMeta] = field(default_factory=dict)

    def get(self, factor_id: str) -> FactorMeta | None:
        return self.records.get(factor_id)

    def __contains__(self, factor_id: str) -> bool:
        return factor_id in self.records

    def ids(self) -> list[str]:
        return sorted(self.records)

    def __len__(self) -> int:
        return len(self.records)

    def save(self, root: Path | None = None) -> Path:
        root = Path(root) if root is not None else self.root
        root.mkdir(parents=True, exist_ok=True)
        out = root / CATALOG_FILENAME
        payload = {
            "version": 1,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            # #P0-final closure 9：按 factor_id 排序输出，落盘确定性（repeated
            # build 产出字节级一致的目录文件）。
            "factors": [r.to_dict() for _, r in sorted(self.records.items())],
        }
        tmp = root / f".{CATALOG_FILENAME}.tmp"
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(out)
        return out

    @classmethod
    def load(cls, root: Path, *, strict: bool | None = None) -> "FactorCatalog":
        """加载 ``_factor_catalog.json``。

        #P0-final closure 9：strict（production）下损坏 JSON / 非法 FactorMeta
        ⇒ ``DataError`` fail-closed——损坏的权威目录不能被静默跳过成空目录。
        research/recovery 才允许宽容跳过。
        """
        if strict is None:
            strict = _strict()
        path = Path(root) / CATALOG_FILENAME
        if not path.exists():
            return cls(root=Path(root), records={})
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            if strict:
                raise DataError(
                    f"因子目录 {path} 损坏/不可读（production fail-closed）：{exc}"
                ) from exc
            return cls(root=Path(root), records={})
        if not isinstance(payload, dict):
            if strict:
                raise DataError(
                    f"因子目录 {path} 顶层不是 mapping（production fail-closed）"
                )
            return cls(root=Path(root), records={})
        records: dict[str, FactorMeta] = {}
        for item in (payload or {}).get("factors", []):
            try:
                meta = FactorMeta.from_dict(item, strict=strict)
                records[meta.factor_id] = meta
            except ValidationError:
                if strict:
                    raise
                continue
        return cls(root=Path(root), records=records)

    @classmethod
    def discover(
        cls,
        lake_root: Path,
        *,
        factor_ids: Sequence[str] | None = None,
        strict: bool | None = None,
    ) -> "FactorCatalog":
        """扫描因子湖各因子目录下的 ``_factor_meta.json`` 聚合目录。

        #P0-final closure 9：strict 下单个 ``_factor_meta.json`` 损坏 ⇒ 整个
        discover 失败（production 权威 catalog 不许静默缺因子）；research 跳过。
        """
        if strict is None:
            strict = _strict()
        records: dict[str, FactorMeta] = {}
        root = Path(lake_root)
        factors_dir = root / "factors" if (root / "factors").exists() else root
        if factors_dir.is_dir():
            for child in factors_dir.iterdir():
                if not child.is_dir():
                    continue
                meta_path = child / FACTOR_META_FILENAME
                if factor_ids is not None and child.name not in set(factor_ids):
                    continue
                if not meta_path.exists():
                    continue
                try:
                    payload = json.loads(meta_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    if strict:
                        raise DataError(
                            f"因子元数据 {meta_path} 损坏/不可读"
                            f"（production fail-closed）：{exc}"
                        ) from exc
                    continue
                try:
                    meta = FactorMeta.from_dict(
                        {**payload, "factor_id": child.name}, strict=strict
                    )
                    records[meta.factor_id] = meta
                except ValidationError:
                    if strict:
                        raise
                    continue
        return cls(root=root, records=records)


# ---------------------------------------------------------------------------
# 批量读 SQL 构建
# ---------------------------------------------------------------------------

_FACTOR_KEYS = ("datetime", "asset", "value")


def _quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def build_factor_union_sql(
    branches: Sequence[tuple[str, list[str]]],
    *,
    columns: Sequence[str] | None = None,
    time_range: tuple[Any, Any] | None = None,
    time_column: str = "datetime",
    factor_id_column: str = "factor_id",
    limit: int | None = None,
    hive_partitioning: bool = False,
    union_by_name: bool = True,
) -> tuple[str, list[Any]]:
    """多因子长表 UNION ALL：每分支 ``SELECT <fid> AS factor_id, ...``。

    返回 (sql, params)。路径用 ? 绑定，factor_id 也是绑定参数。
    """
    if not branches:
        raise ValidationError("read_factors: factor_ids 为空")
    cols = list(columns) if columns else list(_FACTOR_KEYS)
    # 确保 key 列在投影里
    final_cols: list[str] = []
    for c in [*_FACTOR_KEYS, *cols]:
        if c not in final_cols:
            final_cols.append(c)

    opts: list[str] = []
    if hive_partitioning:
        opts.append("hive_partitioning=true")
    if union_by_name:
        opts.append("union_by_name=true")
    opts_suffix = (", " + ", ".join(opts)) if opts else ""

    sel_cols = ", ".join(_quote_ident(c) for c in final_cols)
    parts: list[str] = []
    params: list[Any] = []
    for fid, paths in branches:
        path_param = paths if len(paths) > 1 else paths[0]
        parts.append(
            f"SELECT ? AS {_quote_ident(factor_id_column)}, {sel_cols} "
            f"FROM read_parquet(?{opts_suffix})"
        )
        params.extend([fid, path_param])

    sql = " UNION ALL ".join(parts)
    if time_range is not None:
        start, end = time_range
        conds: list[str] = []
        t_col = _quote_ident(time_column)
        if start is not None:
            conds.append(f"{t_col} >= ?")
            params.append(start)
        if end is not None:
            conds.append(f"{t_col} <= ?")
            params.append(end)
        if conds:
            sql = f"SELECT * FROM ({sql}) AS __factors WHERE {' AND '.join(conds)}"
    if limit is not None:
        sql = f"SELECT * FROM ({sql}) AS __b LIMIT {int(limit)}"
    return sql, params


def build_factor_pivot_sql(
    union_sql: str,
    union_params: list[Any],
    *,
    factor_ids: Sequence[str],
    value_column: str = "value",
    factor_id_column: str = "factor_id",
    limit: int | None = None,
    duplicate_check_sql: str | None = None,
) -> tuple[str, list[Any]]:
    """长表 union → 宽表 PIVOT：(datetime, asset, factor_id, value) → (datetime, asset, f1, f2...)。

    #P0-final closure 9：``USING first(value)`` 在 ``(datetime, asset, factor_id)``
    重复时静默取第一条——调用方（store.read_factors）在 strict 下必须先跑
    ``build_factor_duplicate_check_sql`` 的唯一性门，命中重复即 fail-closed。
    此处 ``duplicate_check_sql`` 保留为显式契约钩子（如需要可拼接进子查询）。
    """
    quoted_ids = ", ".join(_quote_ident(str(f)) for f in factor_ids)
    pivot = (
        f"PIVOT (\n"
        f"  SELECT * FROM ({union_sql}) AS __f\n"
        f") ON {_quote_ident(factor_id_column)} IN ({quoted_ids}) "
        f"USING first({_quote_ident(value_column)})\n"
        f"GROUP BY datetime, asset"
    )
    sql = f"SELECT * FROM ({pivot}) AS __p"
    if limit is not None:
        sql = f"{sql} LIMIT {int(limit)}"
    return sql, list(union_params)


def build_factor_duplicate_check_sql(
    union_sql: str,
    union_params: list[Any],
    *,
    time_column: str = "datetime",
    asset_column: str = "asset",
    factor_id_column: str = "factor_id",
) -> tuple[str, list[Any]]:
    """宽表 pivot 前唯一性门：#P0-final closure 9。

    返回「找任意一个 ``(datetime, asset, factor_id)`` 重复」的探测 SQL
    （``HAVING COUNT(*)>1 LIMIT 1``）。命中 → 调用方 fail-closed，绝不静默
    ``USING first`` 挑一条掩盖数据完整性问题。
    """
    sql = (
        f"SELECT {_quote_ident(factor_id_column)} AS _fid, "
        f"{_quote_ident(time_column)} AS _dt, {_quote_ident(asset_column)} AS _a, "
        f"COUNT(*) AS _n FROM ({union_sql}) AS __d "
        f"GROUP BY 1, 2, 3 HAVING COUNT(*) > 1 LIMIT 1"
    )
    return sql, list(union_params)


def factor_catalog_root(store: Any, dataset: str = "factor_lake") -> Path | None:
    """从 registry 的 factor_lake 解析因子湖根目录。"""
    try:
        ds = store._registry.get(dataset)
        from data_access.registry.loader import ParametricDataset

        if isinstance(ds, ParametricDataset):
            # static_root 是 factors/ 前缀；再上一层是 lake 根
            from data_access.registry.paths import canonicalize, resolve_namespace_path

            static = Path(resolve_namespace_path(str(ds.static_root)))
            if static and static.name == "factors":
                return static.parent
            return static
        return Path(resolve_namespace_path(str(ds.root)))
    except Exception:
        return None
