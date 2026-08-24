# -*- coding: utf-8 -*-
"""Whole-matrix Arrow 状态块存储（R44 节点级增量）。

与 ``stateful_checkpoint_store`` 的**按 instrument 一个 JSON 侧车文件**的布局
不同，本模块把一个 ``StateNodeIdentity`` 在某一个 ``as_of`` 的**整块**（全部
instrument 的 state 矩阵）编码成单个 Arrow/IPC 文件（``state.arrow``）——5000
个 instrument 只写一次文件而不是几十万个小 JSON。它对外保持与现有 JSON 路径
兼容：``StateBlock.from_checkpoints`` / ``to_checkpoints`` 在整块与逐
instrument 的 :class:`StateCheckpoint` 之间互转，``StatefulCheckpointStoreAdapter``
把 ``StateBlockStore`` 伪装成现有的 ``StatefulCheckpointStore``（``load_latest`` +
``commit_batch``），使既有 segmented 路径可以直接消费块状状态。

失败闭合（fail-closed）语义与 JSON 路径一致：缺失 / stale / schema 不匹配 /
损坏的块一律返回 ``None``（回退 full-history replay），绝不返回错误值。

写路径原子性：
* 单块：先写 ``*.tmp`` 再 ``os.replace``（读者永远看不到半块）；
* 一代（多分片）：先写**所有**分片块的临时文件，最后才 ``os.replace`` 发布
  ``manifest.json``——读者要么看到完整的旧代，要么看到完整的新代；
* 垃圾回收按代计数（refcount 风格），绝不删除当前代。
"""
from __future__ import annotations

import io
import json
import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pyarrow as pa
import pyarrow.ipc

# ``state_node_key`` 允许跨多个 factor / canonical 复用同一状态节点身份；新
# 代码直接传一个稳定字符串即可，``runtime.stateful_incremental`` 若引入
# ``StateNodeIdentity`` 也以同样的字符串键进入本模块（不在此强依赖）。
try:  # pragma: no cover - 仅当 runtime 侧已有类型定义时启用
    from factor_engine.runtime.stateful_incremental import StateNodeIdentity  # type: ignore

    _HAS_STATE_NODE_IDENTITY = True
except Exception:  # pragma: no cover - 没有类型定义时退回纯字符串键
    _HAS_STATE_NODE_IDENTITY = False


def _safe_key(value: Any) -> str:
    """把任意 key 压成安全目录名（与 JSON 存储同一套转义）。"""
    return str(value).replace("/", "_").replace("\\", "_").replace(":", "_")


#: 主块文件内 schema 元数据键（``pa.Table`` schema 级 metadata）。
_META_STATE_NODE_KEY = "state_node_key"
_META_AS_OF = "as_of"
_META_SCHEMA_VERSION = "schema_version"
_META_PARAMS_IDENTITY = "params_identity"
_META_GENERATION = "generation"
_META_FORMAT = "format_version"

#: 本模块的持久化格式版本；读取时严格比对，不匹配即 fail-closed。
_FORMAT_VERSION = "r44-state-block.v1"

#: 与 ``stateful_contract`` 保留给 audit 的 state 字段名。
_LAST_TIMESTAMP_FIELD = "last_timestamp"

#: 主块必需的非状态列。
_REQUIRED_COLUMNS = frozenset({"instrument"})
#: 元数据（列方式）携带的字段，不属于 per-instrument state。
_META_COLUMNS = frozenset(
    {"as_of", "schema_version", "state_node_key", "params_identity", "generation",
     "semantic_version", "input_fingerprints"}
)

#: 生成号（generation tag）的正则：``gen-<单调计数>`` 或 ``gen-<uuid-hex>``。
_GENERATION_RE = re.compile(r"^gen-[0-9]+$|^gen-[0-9a-f]{32}$")


def _generation_tag(counter: int) -> str:
    return f"gen-{counter:06d}"


def _is_generation_tag(value: str) -> bool:
    return bool(_GENERATION_RE.match(str(value)))


def _as_utc_iso(value: Any) -> str:
    import pandas as pd

    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.isoformat()


def _schema_for_value(name: str, value: Any) -> pa.Field:
    """为单个 per-instrument state 值推导 Arrow schema。

    ``None`` / NaN 缺失标记 → ``float64``（可空）；布尔优先于整型；映射（如
    ``EwmState`` dict）→ 递归的 ``struct``（按 dict 排序后的键）。数值一律
    ``float64`` 以容纳 NaN 缺失标记。
    """
    if isinstance(value, Mapping):
        keys = sorted(value.keys())
        return pa.field(
            str(name),
            pa.struct(
                [_schema_for_value(str(k), value[k]).with_nullable(True) for k in keys]
            ),
            nullable=True,
        )
    if value is None or (isinstance(value, float) and not _isfinite(value)):
        return pa.field(str(name), pa.float64(), nullable=True)
    if isinstance(value, bool):
        return pa.field(str(name), pa.bool_(), nullable=True)
    if isinstance(value, int):
        return pa.field(str(name), pa.int64(), nullable=True)
    if isinstance(value, float):
        return pa.field(str(name), pa.float64(), nullable=True)
    if isinstance(value, str):
        return pa.field(str(name), pa.string(), nullable=True)
    return pa.field(str(name), pa.string(), nullable=True)


def _isfinite(value: Any) -> bool:
    import math

    return isinstance(value, float) and math.isfinite(value)


def _build_struct_array(values: list[Any], keys: list[str], schema: pa.StructType) -> pa.Array:
    """按 ``schema`` 的字段顺序把每个 instrument 的 dict 压成 StructArray。

    缺键 → null（None / NaN 缺失标记），与 JSON 路径的"缺失字段"语义一致。
    """
    arrays: list[pa.Array] = []
    for field_ in schema:
        key = field_.name
        column: list[Any] = []
        for value in values:
            if isinstance(value, Mapping) and key in value:
                column.append(value[key])
            else:
                column.append(None)
        if pa.types.is_struct(field_.type):
            arrays.append(
                _build_struct_array(column, [f2.name for f2 in field_.type], field_.type)
            )
        else:
            arrays.append(pa.array(column, type=field_.type))
    return pa.StructArray.from_arrays(arrays, fields=schema)


def _value_from_array(array: pa.ChunkedArray, index: int) -> Any:
    """把 StructArray 的某一行的 pylist 转回与 JSON 等价的值（保持 None）。"""
    item = array[index].as_py()
    if item is None:
        return None
    if isinstance(item, Mapping):
        return dict(item)
    return item


@dataclass(frozen=True)
class StateBlock:
    """一个状态节点在某个 ``as_of`` 的全量 instrument 状态矩阵（Arrow 块）。

    字段:
        state_node_key: 状态节点稳定键（可跨 factor/canonical 复用）。
        as_of: ISO 时间戳（本块覆盖到的最新已提交 bar）。
        schema_version: 状态 schema 版本（如 ``ema_state.v2``）。
        instruments: 本块的 instrument 列表（与每列数组等长）。
        state_arrays: 状态字段名 → 逐 instrument 的向量/数组（如 ``ema`` →
            ``np.ndarray``/dict 列表）；缺失字段用 ``None``/NaN。
        params_identity: 参数指纹（空串表示未绑定参数）。
        generation: 本块所属的 generation 标签（``gen-*``，可空）。
    """

    state_node_key: str
    as_of: str
    schema_version: str
    instruments: tuple[str, ...]
    state_arrays: dict[str, Any]
    params_identity: str
    generation: str | None = None
    #: 逐 instrument 的 input 指纹（与 JSON 存储一致，供 resume 时
    #: ``require_for_segment`` 校验）；空元组表示未绑定。
    input_fingerprints: tuple[str, ...] = ()
    #: operator 的 semantic_version（整块共享，供 registry 校验）。
    semantic_version: str = ""

    # ------------------------------------------------------------------ #
    # Arrow IPC 编码 / 解码
    # ------------------------------------------------------------------ #
    def to_arrow(self) -> pa.Table:
        """编码成单张 Arrow/Table（IPC 文件格式）。

        布局：首列 ``instrument``（字符串），随后是 ``as_of`` /
        ``schema_version`` / ``state_node_key`` / ``params_identity`` /
        ``generation`` 元数据列，再按序排每个状态字段列（映射字段 → struct
        列，标量字段 → 对应标量列，类型为各 instrument 值的并集）。schema
        级 metadata 冗余保存关键身份字段，供读取时快速 fail-closed 校验。
        """
        fields: list[pa.Field] = []
        columns: list[pa.Array] = []

        fields.append(pa.field("instrument", pa.string(), nullable=False))
        columns.append(pa.array(list(self.instruments), type=pa.string()))

        meta_pairs = [
            ("as_of", self.as_of, pa.string()),
            ("schema_version", self.schema_version, pa.string()),
            ("state_node_key", self.state_node_key, pa.string()),
            ("params_identity", self.params_identity, pa.string()),
            ("generation", self.generation or "", pa.string()),
            ("semantic_version", self.semantic_version, pa.string()),
        ]
        n_rows = len(self.instruments)
        for name, value, dtype in meta_pairs:
            fields.append(pa.field(name, dtype, nullable=True))
            columns.append(pa.array([value] * n_rows, type=dtype))
        # input_fingerprints：逐 instrument 的指纹列（与 instruments 等长）。
        fingerprints = list(self.input_fingerprints)
        if len(fingerprints) != n_rows:
            fingerprints = [""] * n_rows
        fields.append(pa.field("input_fingerprints", pa.string(), nullable=True))
        columns.append(pa.array(fingerprints, type=pa.string()))

        for field_name in self.state_arrays:
            values = list(self.state_arrays[field_name])
            if values and isinstance(values[0], Mapping):
                keys: list[str] = []
                for v in values:
                    if isinstance(v, Mapping):
                        for k in v.keys():
                            if k not in keys:
                                keys.append(k)
                schema = pa.struct(
                    [_schema_for_value(k, v.get(k)) for k in keys]
                    if keys
                    else []
                )
                if schema:
                    fields.append(pa.field(field_name, schema, nullable=True))
                    columns.append(
                        _build_struct_array(values, keys, schema)
                    )
                else:
                    fields.append(pa.field(field_name, pa.null(), nullable=True))
                    columns.append(pa.nulls(len(values), type=pa.null()))
                continue
            # 标量字段：union 各值类型，取第一个非空类型（None 先行保留）。
            dtype = pa.float64()
            for v in values:
                if isinstance(v, bool):
                    dtype = pa.bool_()
                    break
                if isinstance(v, int) and not isinstance(v, bool):
                    dtype = pa.int64()
                    break
                if isinstance(v, float):
                    dtype = pa.float64()
                    break
                if isinstance(v, str):
                    dtype = pa.string()
                    break
            fields.append(pa.field(field_name, dtype, nullable=True))
            columns.append(pa.array(values, type=dtype))

        schema = pa.schema(fields)
        schema = schema.with_metadata(
            {
                _META_FORMAT: _FORMAT_VERSION,
                _META_STATE_NODE_KEY: self.state_node_key,
                _META_AS_OF: self.as_of,
                _META_SCHEMA_VERSION: self.schema_version,
                _META_PARAMS_IDENTITY: self.params_identity,
                _META_GENERATION: self.generation or "",
            }
        )
        return pa.Table.from_arrays(columns, schema=schema)

    @classmethod
    def from_arrow(cls, table: pa.Table) -> "StateBlock":
        """从 Arrow/Table 解码回 ``StateBlock``（严格校验，失败闭合）。

        任何缺失必需列 / schema metadata 不匹配 / 行数不一致 / 格式版本不
        匹配都抛 ``ValueError``，由调用方吞掉返回 ``None``。
        """
        meta = dict(table.schema.metadata or {})
        fmt = meta.get(_META_FORMAT.encode())
        if fmt is None or fmt.decode() != _FORMAT_VERSION:
            raise ValueError(
                f"state block format mismatch: {fmt!r} != {_FORMAT_VERSION!r}"
            )
        names = set(table.column_names)
        if not _REQUIRED_COLUMNS.issubset(names):
            raise ValueError(f"state block missing required columns: {_REQUIRED_COLUMNS - names}")
        if "as_of" not in names:
            raise ValueError("state block missing as_of column")
        n = table.num_rows
        instruments = tuple(table.column("instrument").to_pylist())
        if len(instruments) != n:
            raise ValueError("instrument column length mismatch")

        def _col(name: str) -> str:
            array = table.column(name)
            if array.num_chunks == 0:
                return ""
            value = array[0].as_py()
            return "" if value is None else str(value)

        state_node_key = _col("state_node_key")
        as_of = _col("as_of")
        schema_version = _col("schema_version")
        params_identity = _col("params_identity")
        generation = _col("generation") or None
        semantic_version = _col("semantic_version")
        input_fingerprints = tuple(
            ("" if v is None else str(v))
            for v in table.column("input_fingerprints").to_pylist()
        ) if "input_fingerprints" in names else ()

        # 元数据列 + instrument + 状态字段列 的划分。
        state_arrays: dict[str, Any] = {}
        for name in names:
            if name in _REQUIRED_COLUMNS or name in _META_COLUMNS:
                continue
            array = table.column(name)
            if pa.types.is_struct(array.type):
                state_arrays[name] = [
                    _value_from_array(array, i) for i in range(n)
                ]
            else:
                state_arrays[name] = [
                    None if array[i].as_py() is None else array[i].as_py() for i in range(n)
                ]
        return cls(
            state_node_key=state_node_key,
            as_of=as_of,
            schema_version=schema_version,
            instruments=instruments,
            state_arrays=state_arrays,
            params_identity=params_identity,
            generation=generation,
            input_fingerprints=input_fingerprints,
            semantic_version=semantic_version,
        )

    # ------------------------------------------------------------------ #
    # 与逐 instrument 的 ``StateCheckpoint`` 互转
    # ------------------------------------------------------------------ #
    @classmethod
    def from_checkpoints(
        cls,
        state_node_key: str,
        checkpoints: list[StateCheckpoint],
        *,
        params_identity: str = "",
        generation: str | None = None,
    ) -> "StateBlock":
        """把逐 instrument 的 ``StateCheckpoint`` 列表压成整块。

        每个 checkpoint 必须是同一 operator / schema 版本 / as_of；以第一个
        checkpoint 的身份字段为准（其余不一致时抛 ``ValueError``，失败闭合）。
        ``as_of`` 取第一个 checkpoint 的 ``as_of``（生成时即 state 自身的
        ``last_timestamp``）。per-instrument state 字段按各 checkpoint 的并集
        取值，缺失字段 → ``None``。
        """
        if not checkpoints:
            raise ValueError("cannot build a StateBlock from an empty checkpoint list")
        operator = checkpoints[0].operator
        as_of = checkpoints[0].as_of
        schema_version = checkpoints[0].state_schema_version
        semantic_version = checkpoints[0].semantic_version
        for checkpoint in checkpoints[1:]:
            if checkpoint.operator != operator:
                raise ValueError(
                    "cannot mix operators in one StateBlock: "
                    f"{operator} != {checkpoint.operator}"
                )
            if checkpoint.state_schema_version != schema_version:
                raise ValueError("cannot mix schema versions in one StateBlock")
        instruments: list[str] = []
        field_names: list[str] = []
        rows: list[dict[str, Any]] = []
        fingerprints: list[str] = []
        for checkpoint in checkpoints:
            instruments.append(str(checkpoint.instrument))
            state = dict(checkpoint.state or {})
            for key in state.keys():
                if key not in field_names:
                    field_names.append(key)
            rows.append(state)
            fingerprints.append(str(checkpoint.input_fingerprint))
        state_arrays: dict[str, list[Any]] = {}
        for key in field_names:
            state_arrays[key] = [row.get(key) for row in rows]
        return cls(
            state_node_key=str(state_node_key),
            as_of=str(as_of),
            schema_version=str(schema_version),
            instruments=tuple(instruments),
            state_arrays=state_arrays,
            params_identity=str(params_identity),
            generation=generation,
            input_fingerprints=tuple(fingerprints),
            semantic_version=str(semantic_version),
        )

    def to_checkpoints(self) -> list[StateCheckpoint]:
        """把整块解回逐 instrument 的 ``StateCheckpoint`` 列表。

        状态字段并集缺失的 instrument → 该字段 ``None``；``state`` 内带
        ``last_timestamp``（audit 用），与 JSON 路径的 checkpoint 布局一致。
        单 instrument 的 identity 字段（input_fingerprint / semantic_version）
        在块内未保存，回读为占位空值——适配器使用时以 operator 的 registry
        校验为准（见 ``StatefulCheckpointStoreAdapter``）。
        """
        from factor_engine.stateful_contract import StateCheckpoint

        operator = self.schema_version.rsplit("_state.", 1)[0] if "_state." in self.schema_version else ""
        fingerprints = list(self.input_fingerprints)
        if len(fingerprints) != len(self.instruments):
            fingerprints = [""] * len(self.instruments)
        checkpoints: list[StateCheckpoint] = []
        for j, instrument in enumerate(self.instruments):
            state: dict[str, Any] = {}
            for field_name, values in self.state_arrays.items():
                state[field_name] = values[j]
            if _LAST_TIMESTAMP_FIELD not in state:
                state[_LAST_TIMESTAMP_FIELD] = self.as_of
            checkpoints.append(
                StateCheckpoint(
                    operator=operator,
                    instrument=str(instrument),
                    as_of=str(self.as_of),
                    state_schema_version=str(self.schema_version),
                    semantic_version=str(self.semantic_version),
                    input_fingerprint=str(fingerprints[j]),
                    state=state,
                )
            )
        return checkpoints


class StateBlockStore:
    """文件布局的 Arrow 块存储。

    布局::

        <root>/state_lake/<state_node_key>/<schema_version>/as_of=<as_of>/generation=<gen>/state.arrow
        <root>/state_lake/<state_node_key>/<schema_version>/as_of=<as_of>/manifest.json

    单块写：``state.arrow`` 先写 ``*.tmp`` 再 ``os.replace``；一代写：
    所有分片块的临时文件写完后，最后才 ``os.replace`` 发布 ``manifest.json``，
    读者要么看到完整旧代要么看到完整新代。
    """

    def __init__(self, root: str | Path | None = None) -> None:
        if root is None:
            try:
                from factor_engine.util.workspace_paths import default_factor_lake_root
            except Exception:  # pragma: no cover - 无 workspace shim 时
                default_factor_lake_root = lambda: Path(".")  # type: ignore
            root = Path(default_factor_lake_root()) / "state_lake"
        self.root = Path(root)

    # ------------------------------------------------------------------ #
    # 路径解析
    # ------------------------------------------------------------------ #
    def _key_dir(self, state_node_key: str) -> Path:
        return self.root / _safe_key(state_node_key)

    def _schema_dirs(self, state_node_key: str) -> list[Path]:
        key_dir = self._key_dir(state_node_key)
        if not key_dir.is_dir():
            return []
        return sorted(
            (p for p in key_dir.iterdir() if p.is_dir()),
            key=lambda p: p.name,
        )

    def _as_of_dirs(self, state_node_key: str, schema_version: str) -> list[Path]:
        base = self._key_dir(state_node_key) / _safe_key(schema_version)
        if not base.is_dir():
            return []
        out: list[Path] = []
        for p in base.iterdir():
            if not p.is_dir() or not p.name.startswith("as_of="):
                continue
            out.append(p)
        return sorted(out, key=lambda p: p.name)

    def _generation_dirs(self, state_node_key: str, schema_version: str, as_of: str) -> list[Path]:
        base = (
            self._key_dir(state_node_key)
            / _safe_key(schema_version)
            / f"as_of={_safe_key(as_of)}"
        )
        if not base.is_dir():
            return []
        out: list[Path] = []
        for p in base.iterdir():
            if not p.is_dir() or not p.name.startswith("generation="):
                continue
            out.append(p)
        return sorted(out, key=lambda p: p.name)

    def _generation_dir(
        self, state_node_key: str, schema_version: str, as_of: str, generation: str
    ) -> Path:
        return (
            self._key_dir(state_node_key)
            / _safe_key(schema_version)
            / f"as_of={_safe_key(as_of)}"
            / f"generation={_safe_key(generation)}"
        )

    # ------------------------------------------------------------------ #
    # 单块读写
    # ------------------------------------------------------------------ #
    def write_block(self, block: StateBlock) -> None:
        """原子写单个分片块：先写 ``*.tmp`` 再 ``os.replace``。"""
        gen = block.generation or self.current_generation(block.state_node_key) or "gen-000001"
        if not _is_generation_tag(gen):
            gen = f"gen-{gen}"
        target_dir = self._generation_dir(
            block.state_node_key, block.schema_version, block.as_of, gen
        )
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / "state.arrow"
        tmp = target_dir / f".{uuid.uuid4().hex}.tmp"
        try:
            with pa.ipc.new_file(tmp, block.to_arrow().schema) as writer:
                writer.write_table(block.to_arrow())
            os.replace(tmp, target)
        except Exception:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            raise

    def _load_block_from_dir(self, gen_dir: Path) -> StateBlock:
        target = gen_dir / "state.arrow"
        if not target.is_file():
            # 一代可能以 ``shard_<name>.arrow`` 分片发布（``write_generation``）；
            # 无 ``state.arrow`` 时回退读 manifest 的第一个分片。
            manifest_path = gen_dir / "manifest.json"
            if manifest_path.is_file():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                parts = manifest.get("parts", [])
                if parts:
                    target = gen_dir / parts[0]["file"]
            if not target.is_file():
                raise ValueError(f"state block missing: {gen_dir}")
        with pa.ipc.open_file(target) as reader:
            table = reader.read_all()
        return StateBlock.from_arrow(table)

    def load_block(
        self,
        state_node_key: str,
        *,
        as_of: str | None = None,
        schema_version: str | None = None,
        generation: str | None = None,
    ) -> StateBlock | None:
        """按 key（可选 as_of / schema_version / generation）读最新可用块。

        缺失 / 损坏 / schema 不匹配 → ``None``（fail-closed）。
        """
        if generation is not None:
            gen = generation if _is_generation_tag(generation) else f"gen-{generation}"
            gen_dir = self._generation_dir(
                state_node_key,
                schema_version or "",
                as_of or "",
                gen,
            )
            try:
                return self._load_block_from_dir(gen_dir)
            except Exception:
                return None
        schema_dirs = self._schema_dirs(state_node_key)
        if schema_version is not None:
            schema_dirs = [
                d for d in schema_dirs
                if d.name == _safe_key(schema_version) or d.name == schema_version
            ]
        for schema_dir in reversed(schema_dirs):
            as_of_dirs = self._as_of_dirs(state_node_key, schema_dir.name)
            if as_of is not None:
                candidates = [d for d in as_of_dirs if d.name.endswith(_safe_key(as_of))]
                as_of_dirs = candidates or as_of_dirs
            for as_of_dir in reversed(as_of_dirs):
                gen_dirs = sorted(
                    (p for p in as_of_dir.iterdir() if p.name.startswith("generation=")),
                    key=lambda p: p.name,
                )
                for gen_dir in reversed(gen_dirs):
                    try:
                        return self._load_block_from_dir(gen_dir)
                    except Exception:
                        continue
        return None

    def list_blocks(self, state_node_key: str) -> list[dict[str, Any]]:
        """列出该 key 下的所有块：``[{as_of, schema_version, generation, path}]``。"""
        out: list[dict[str, Any]] = []
        for schema_dir in self._schema_dirs(state_node_key):
            for as_of_dir in self._as_of_dirs(state_node_key, schema_dir.name):
                as_of = as_of_dir.name.removeprefix("as_of=")
                for gen_dir in self._generation_dirs(
                    state_node_key, schema_dir.name, as_of
                ):
                    generation = gen_dir.name.removeprefix("generation=")
                    path = gen_dir / "state.arrow"
                    out.append(
                        {
                            "as_of": as_of,
                            "schema_version": schema_dir.name,
                            "generation": generation,
                            "path": str(path),
                        }
                    )
        return out

    def latest_as_of(
        self,
        state_node_key: str,
        *,
        schema_version: str | None = None,
    ) -> str | None:
        """该 key 最新一代的 ``as_of``（无可用块 → None）。"""
        block = self.load_block(state_node_key, schema_version=schema_version)
        return None if block is None else block.as_of

    # ------------------------------------------------------------------ #
    # 世代（多分片原子批量）发布
    # ------------------------------------------------------------------ #
    def write_generation(
        self,
        state_node_key: str,
        as_of: str,
        blocks_by_shard: dict[str, StateBlock],
    ) -> str:
        """原子发布一代：先写所有分片块临时文件，最后 ``os.replace`` manifest。

        返回新 generation 标签（``gen-<计数器>``，按现有 as_of 目录内已有
        代数的最大计数 +1；无则从 1 开始）。任一分片写失败 → 清理临时文件
        并抛异常，读者仍看到完整的旧代。
        """
        if not blocks_by_shard:
            raise ValueError("write_generation requires at least one shard block")
        schema_version = None
        for block in blocks_by_shard.values():
            if schema_version is None:
                schema_version = block.schema_version
            elif block.schema_version != schema_version:
                raise ValueError("cannot mix schema versions in one generation")
        if schema_version is None:
            raise ValueError("no shard blocks provided")

        existing = self._generation_dirs(state_node_key, schema_version, as_of)
        counters = [0]
        for d in existing:
            tag = d.name.removeprefix("generation=")
            if tag.startswith("gen-"):
                try:
                    counters.append(int(tag[4:]))
                except ValueError:
                    continue
        generation = _generation_tag(max(counters) + 1)

        gen_dir = self._generation_dir(state_node_key, schema_version, as_of, generation)
        gen_dir.mkdir(parents=True, exist_ok=True)
        staged: list[tuple[Path, Path]] = []  # (tmp, target)
        try:
            for shard, block in blocks_by_shard.items():
                if block.generation is None:
                    block = StateBlock(
                        state_node_key=block.state_node_key,
                        as_of=block.as_of,
                        schema_version=block.schema_version,
                        instruments=block.instruments,
                        state_arrays=block.state_arrays,
                        params_identity=block.params_identity,
                        generation=generation,
                        input_fingerprints=block.input_fingerprints,
                        semantic_version=block.semantic_version,
                    )
                target = gen_dir / f"shard_{_safe_key(shard)}.arrow"
                tmp = gen_dir / f".{uuid.uuid4().hex}.tmp"
                with pa.ipc.new_file(tmp, block.to_arrow().schema) as writer:
                    writer.write_table(block.to_arrow())
                staged.append((tmp, target))
            # 所有分片块已落到临时文件 —— 逐个原子改名。
            for tmp, target in staged:
                os.replace(tmp, target)
            # 最后发布 manifest（当前代的唯一"提交点"）。
            manifest = {
                "format": _FORMAT_VERSION,
                "state_node_key": state_node_key,
                "schema_version": schema_version,
                "as_of": as_of,
                "generation": generation,
                "shards": sorted(blocks_by_shard.keys()),
                "parts": [
                    {
                        "shard": shard,
                        "file": f"shard_{_safe_key(shard)}.arrow",
                        "row_count": len(block.instruments),
                    }
                    for shard, block in blocks_by_shard.items()
                ],
                "total_rows": sum(len(b.instruments) for b in blocks_by_shard.values()),
            }
            manifest_tmp = gen_dir / f".{uuid.uuid4().hex}.manifest.tmp"
            manifest_tmp.write_text(
                json.dumps(manifest, ensure_ascii=False, sort_keys=True), encoding="utf-8"
            )
            os.replace(manifest_tmp, gen_dir / "manifest.json")
            return generation
        except Exception:
            for tmp, _ in staged:
                try:
                    tmp.unlink(missing_ok=True)
                except OSError:
                    pass
            raise

    def load_generation(
        self, state_node_key: str, generation: str
    ) -> dict[str, StateBlock]:
        """按 generation 读完整一代（manifest 内列出的每个分片都必须可读）。

        manifest 缺失 / 损坏 / 任一 shard 块缺失 → 抛 ``ValueError``（fail
        closed：不返回残缺代）。
        """
        if not _is_generation_tag(generation):
            generation = f"gen-{generation}"
        for schema_dir in self._schema_dirs(state_node_key):
            for as_of_dir in self._as_of_dirs(state_node_key, schema_dir.name):
                gen_dir = as_of_dir / f"generation={_safe_key(generation)}"
                if not gen_dir.is_dir():
                    continue
                manifest_path = gen_dir / "manifest.json"
                if not manifest_path.is_file():
                    raise ValueError(f"generation manifest missing: {manifest_path}")
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest.get("generation") != generation:
                    raise ValueError("generation tag mismatch in manifest")
                blocks: dict[str, StateBlock] = {}
                for part in manifest.get("parts", []):
                    shard = part["shard"]
                    target = gen_dir / part["file"]
                    if not target.is_file():
                        raise ValueError(f"generation part missing: {target}")
                    with pa.ipc.open_file(target) as reader:
                        block = StateBlock.from_arrow(reader.read_all())
                    blocks[str(shard)] = block
                if not blocks:
                    raise ValueError("generation manifest lists no parts")
                return blocks
        raise ValueError(f"generation not found: {state_node_key}/{generation}")

    def current_generation(self, state_node_key: str) -> str | None:
        """该 key 最新一代的标签（按 manifest 提交点 / 代目录排序取最新）。"""
        best: tuple[Any, str] | None = None
        for schema_dir in self._schema_dirs(state_node_key):
            for as_of_dir in self._as_of_dirs(state_node_key, schema_dir.name):
                for gen_dir in sorted(
                    (p for p in as_of_dir.iterdir() if p.name.startswith("generation=")),
                    key=lambda p: p.name,
                ):
                    if not (gen_dir / "manifest.json").is_file():
                        continue
                    tag = gen_dir.name.removeprefix("generation=")
                    key: tuple[Any, ...]
                    if tag.startswith("gen-") and tag[4:].isdigit():
                        key = (0, int(tag[4:]), as_of_dir.name)
                    else:
                        key = (1, tag, as_of_dir.name)
                    if best is None or key > best[0]:
                        best = (key, tag)
        return None if best is None else best[1]

    # ------------------------------------------------------------------ #
    # 垃圾回收（refcount 风格：保留最新 ``keep`` 代）
    # ------------------------------------------------------------------ #
    def garbage_collect(self, state_node_key: str, keep: int = 1) -> int:
        """删除该 key 下除最新 ``keep`` 代之外的所有旧代，返回删除的目录数。

        绝不删除当前代（manifest 已发布的代）。无 manifest 的孤儿代目录也会
        被清理（视为未提交成功）。
        """
        if keep < 1:
            raise ValueError("keep must be >= 1")
        removed = 0
        for schema_dir in self._schema_dirs(state_node_key):
            for as_of_dir in self._as_of_dirs(state_node_key, schema_dir.name):
                gen_dirs = sorted(
                    (p for p in as_of_dir.iterdir() if p.name.startswith("generation=")),
                    key=lambda p: p.name,
                )
                published: list[Path] = []
                orphan: list[Path] = []
                for d in gen_dirs:
                    if (d / "manifest.json").is_file():
                        published.append(d)
                    else:
                        orphan.append(d)
                # 保留最新 keep 个已发布代；孤儿代全部删除。
                keep_dirs = set(published[-keep:] if published else [])
                for d in orphan + [p for p in published if p not in keep_dirs]:
                    import shutil

                    shutil.rmtree(d, ignore_errors=True)
                    removed += 1
        return removed


# ---------------------------------------------------------------------------
# 现有 JSON 路径的 drop-in 适配器
# ---------------------------------------------------------------------------
class StatefulCheckpointStoreAdapter:
    """把 ``StateBlockStore`` 伪装成现有的 ``StatefulCheckpointStore``。

    供既有 segmented 路径（``try_stateful_segmented_incremental``）消费块状
    状态：``load_latest(factor_id, canonical, instrument, before=...)`` 读最新
    一代块，解回逐 instrument checkpoint，做与 JSON 存储**相同**的 fail-closed
    校验（operator 匹配、as_of 严格早于 ``before``、registry 校验、
    ``state["last_timestamp"] == as_of``），通过才返回该 instrument 的
    checkpoint；``commit_batch`` 把逐 instrument checkpoint 累积成
    ``StateBlock`` 并 ``write_generation`` 原子发布。

    说明：
    * 块内未保存 input_fingerprint / semantic_version（整块布局只保留
      operator/schema/as_of/state），因此适配器无法做指纹级校验——这是与
      JSON 存储的已知差异，仅在通过块路径读写时生效；JSON 存储本身不变。
    * 一个 generation 只对应一个 ``(schema_version, as_of)``；若累积的
      checkpoint 跨 operator / schema 版本，提交时抛 ``ValueError``（失败闭合）。
    """

    def __init__(
        self,
        block_store: StateBlockStore | None = None,
        *,
        root: str | Path | None = None,
        state_node_key: str | None = None,
    ) -> None:
        self._block_store = block_store or StateBlockStore(root=root)
        self._key_override = state_node_key

    @property
    def block_store(self) -> StateBlockStore:
        return self._block_store

    def _key(self, factor_id: str, canonical: str) -> str:
        if self._key_override is not None:
            return str(self._key_override)
        return f"{_safe_key(factor_id)}::{_safe_key(canonical)}"

    def load_latest(
        self,
        factor_id: str,
        canonical: str,
        instrument: str,
        *,
        before: Any,
    ) -> Any:
        """读最新一代块，返回指定 instrument 且严格早于 ``before`` 的有效 checkpoint。

        任一校验失败 → ``None``（fail-closed）。
        """
        from factor_engine.stateful_contract import (
            StatefulCheckpointRegistry,
            StatefulContractError,
        )

        spec = StatefulCheckpointRegistry.get(canonical)
        if spec is None:
            return None
        key = self._key(factor_id, canonical)
        block = self._block_store.load_block(key)
        if block is None:
            return None
        if block.schema_version != spec.state_schema_version:
            return None
        try:
            checkpoints = block.to_checkpoints()
        except Exception:
            return None
        # 块内未保存 operator / semantic_version（整块布局只保留 schema 版本），
        # 由适配器按请求的 canonical 及其 registry spec 显式回填，使
        # ``StatefulCheckpointRegistry.validate`` 能通过（与 JSON 存储一致）。
        import dataclasses

        checkpoints = [
            dataclasses.replace(
                ck,
                operator=canonical,
                semantic_version=spec.semantic_version,
            )
            for ck in checkpoints
        ]
        # 与 JSON 存储相同的 as_of 严格早于 before 判定。
        import pandas as pd

        try:
            cutoff = pd.Timestamp(before)
            if cutoff.tzinfo is None:
                cutoff = cutoff.tz_localize("UTC")
            else:
                cutoff = cutoff.tz_convert("UTC")
        except (ValueError, TypeError):
            return None
        target: Any = None
        for checkpoint in checkpoints:
            if str(checkpoint.instrument) != str(instrument):
                continue
            if checkpoint.operator != canonical:
                continue
            try:
                as_of = pd.Timestamp(checkpoint.as_of)
                if as_of.tzinfo is None:
                    as_of = as_of.tz_localize("UTC")
                else:
                    as_of = as_of.tz_convert("UTC")
            except (ValueError, TypeError):
                return None
            if not as_of < cutoff:
                continue
            state_last = (checkpoint.state or {}).get("last_timestamp")
            if state_last is not None:
                try:
                    last = pd.Timestamp(state_last)
                    if last.tzinfo is None:
                        last = last.tz_localize("UTC")
                    else:
                        last = last.tz_convert("UTC")
                    if last != as_of:
                        continue
                except (ValueError, TypeError):
                    continue
            target = checkpoint
            break
        if target is None:
            return None
        try:
            StatefulCheckpointRegistry.validate(target)
        except StatefulContractError:
            return None
        return target

    def save(self, factor_id: str, checkpoint: Any) -> None:
        """单 checkpoint 保存：累积成一代并原子发布（多一个临时文件的成本可忽略）。"""
        from factor_engine.stateful_contract import StatefulCheckpointRegistry

        StatefulCheckpointRegistry.validate(checkpoint)
        key = self._key(factor_id, checkpoint.operator)
        self._block_store.write_generation(
            key,
            checkpoint.as_of,
            {str(checkpoint.instrument): StateBlock.from_checkpoints(key, [checkpoint])},
        )

    def commit_batch(
        self,
        factor_id: str,
        checkpoints: list[Any],
        *,
        execution_id: str | None = None,
    ) -> None:
        """原子提交一批 checkpoint：全部累积成一个 ``StateBlock`` 一代发布。

        任一 checkpoint 校验失败 → 抛异常，什么都不写（fail-closed）。
        """
        from factor_engine.stateful_contract import StatefulCheckpointRegistry

        if not checkpoints:
            return
        for checkpoint in checkpoints:
            StatefulCheckpointRegistry.validate(checkpoint)
        canonical = checkpoints[0].operator
        key = self._key(factor_id, canonical)
        block = StateBlock.from_checkpoints(key, list(checkpoints))
        self._block_store.write_generation(key, block.as_of, {"all": block})

    def latest_as_of(self, factor_id: str, canonical: str, instrument: str) -> Any:
        """块存储无逐 instrument as_of，返回最新一代的块级 ``as_of``。"""
        return self._block_store.latest_as_of(self._key(factor_id, canonical))


__all__ = [
    "StateBlock",
    "StateBlockStore",
    "StatefulCheckpointStoreAdapter",
]
