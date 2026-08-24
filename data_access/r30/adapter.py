"""data_access.r30.adapter —— SourceAdapter Protocol / SourceCapabilities / 后端能力矩阵。

R30-P1-001：把数据读取从具体引擎/格式中抽离成统一的 ``SourceAdapter`` 协议层：

    1. ``SourceCapabilities`` —— 数据源能力声明（projection/filter 下推、分区
       裁剪、流式、远程、快照、写、事务、asof）；``to_dict()`` / ``subset()``。
    2. ``SourceAdapter``（typing.Protocol）—— 统一数据源接口：capabilities /
       resolve / estimate / scan / snapshot / healthcheck。
    3. ``FormatSourceAdapter`` —— 把既有 read/formats.py 的 ``FormatAdapter``
       （DuckDB FROM 子句生成器）包装成 backend-agnostic 的 SourceAdapter；
       能力声明按格式给出（parquet 支持 projection/filter/partition_pruning；
       csv/arrow 支持 streaming 但无 asof；delta/iceberg 有 transaction）。
    4. ``BackendAdapter`` + ``BACKEND_CAPABILITY_MATRIX`` —— 查询引擎能力矩阵
       （duckdb/polars/pyarrow/local/cos/clickhouse）与 ``choose_backend`` 路由。

本层完全 additive：**不修改** 既有 read/formats.py、core/storage.py、
core/duckdb_capabilities.py——既有 ``BackendCapability`` 是存储后端能力
（authorization/duckdb_httpfs/snapshot_metadata/read_implemented），本层
``SourceCapabilities`` 是查询引擎能力矩阵，二者互补、各自独立。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, runtime_checkable

__all__ = [
    "BACKEND_CAPABILITY_MATRIX",
    "BackendAdapter",
    "FormatSourceAdapter",
    "SourceAdapter",
    "SourceCapabilities",
    "choose_backend",
    "source_capabilities_from_matrix",
]

_CAPABILITY_FIELDS: tuple[str, ...] = (
    "projection_pushdown",
    "filter_pushdown",
    "partition_pruning",
    "streaming",
    "remote",
    "snapshot",
    "write",
    "transaction",
    "asof",
)


@dataclass(frozen=True)
class SourceCapabilities:
    """一次数据源/后端的能力声明。

    全部为布尔能力位；缺失即不具备。``subset`` 用于判断「本能力需求 ⊆ 候选
    能力集」，供 ``choose_backend`` 做满足性筛选。
    """

    projection_pushdown: bool = False
    filter_pushdown: bool = False
    partition_pruning: bool = False
    streaming: bool = False
    remote: bool = False
    snapshot: bool = False
    write: bool = False
    transaction: bool = False
    asof: bool = False

    def to_dict(self) -> dict[str, bool]:
        return {name: bool(getattr(self, name)) for name in _CAPABILITY_FIELDS}

    def subset(self, other: "SourceCapabilities") -> bool:
        """self 宣称的每一项能力，other 都具备（self ⊆ other）。"""
        for name in _CAPABILITY_FIELDS:
            if getattr(self, name) and not getattr(other, name):
                return False
        return True


@runtime_checkable
class SourceAdapter(Protocol):
    """统一数据源协议：capabilities / resolve / estimate / scan / snapshot / healthcheck。

    这是一个 structural Protocol：任何实现了这些方法的对象都能充当 SourceAdapter，
    不必继承本类。
    """

    def capabilities(self) -> SourceCapabilities:  # pragma: no cover - protocol
        ...

    def resolve(self, request: Any, contract: Any = None) -> Any:
        """把请求解析成已解析计划（plan）。"""

    def estimate(self, resolved: Any) -> dict[str, Any]:
        """对已解析计划做成本估算，返回 dict。"""

    def scan(self, prepared: Any) -> Any:
        """把已准备的执行计划变成可执行的扫描描述。"""

    def snapshot(self, resolved: Any) -> Any:
        """返回数据源当前快照元数据（无则 None）。"""

    def healthcheck(self) -> bool:
        """数据源是否可用的健康检查。"""


# ---------------------------------------------------------------------------
# 按物理格式的能力声明（FormatSourceAdapter 用）。remote 不在此矩阵——它取决于
# 所在存储后端，由 FormatSourceAdapter 构造时以 ``storage_remote`` 注入。
# ---------------------------------------------------------------------------

_FORMAT_CAPABILITY_MATRIX: dict[str, dict[str, bool]] = {
    "parquet": {
        "projection_pushdown": True,
        "filter_pushdown": True,
        "partition_pruning": True,
        "streaming": False,
        "snapshot": False,
        "write": False,
        "transaction": False,
        "asof": False,
    },
    "csv": {
        "projection_pushdown": True,
        "filter_pushdown": True,
        "partition_pruning": False,
        "streaming": True,
        "snapshot": False,
        "write": False,
        "transaction": False,
        "asof": False,
    },
    "tsv": {
        "projection_pushdown": True,
        "filter_pushdown": True,
        "partition_pruning": False,
        "streaming": True,
        "snapshot": False,
        "write": False,
        "transaction": False,
        "asof": False,
    },
    "jsonl": {
        "projection_pushdown": True,
        "filter_pushdown": True,
        "partition_pruning": False,
        "streaming": True,
        "snapshot": False,
        "write": False,
        "transaction": False,
        "asof": False,
    },
    "arrow": {
        "projection_pushdown": True,
        "filter_pushdown": True,
        "partition_pruning": False,
        "streaming": True,
        "snapshot": False,
        "write": False,
        "transaction": False,
        "asof": False,
    },
    "feather": {
        "projection_pushdown": True,
        "filter_pushdown": True,
        "partition_pruning": False,
        "streaming": True,
        "snapshot": False,
        "write": False,
        "transaction": False,
        "asof": False,
    },
    "delta": {
        "projection_pushdown": True,
        "filter_pushdown": True,
        "partition_pruning": True,
        "streaming": True,
        "snapshot": True,
        "write": False,
        "transaction": True,
        "asof": False,
    },
    "iceberg": {
        "projection_pushdown": True,
        "filter_pushdown": True,
        "partition_pruning": True,
        "streaming": True,
        "snapshot": True,
        "write": False,
        "transaction": True,
        "asof": False,
    },
}


class FormatSourceAdapter:
    """包装既有 read/formats.py 的 FormatAdapter → backend-agnostic SourceAdapter。

    FormatAdapter 负责生成 DuckDB FROM 子句（ParquetAdapter/CSVAdapter/
    ArrowIPCAdapter…）；本类把它包成统一 SourceAdapter 协议面，并把格式能力
    （projection/filter/partition_pruning/streaming/asof…）翻译成
    ``SourceCapabilities``。所有字段访问用 getattr 防御性获取，绝不假设
    底层 adapter 一定有什么属性。
    """

    def __init__(
        self,
        format_adapter: Any,
        *,
        storage_remote: bool = False,
        name: str | None = None,
    ) -> None:
        self._format_adapter = format_adapter
        self._storage_remote = bool(storage_remote)
        self.name = name
        if self.name is None:
            fmt = getattr(format_adapter, "data_format", None)
            if fmt is not None:
                self.name = getattr(fmt, "value", None) or str(fmt)

    @property
    def format_adapter(self) -> Any:
        """被包装的既有 FormatAdapter 实例。"""
        return self._format_adapter

    # ---- SourceAdapter 协议实现 ----

    def capabilities(self) -> SourceCapabilities:
        fmt = self._format_name()
        base = _FORMAT_CAPABILITY_MATRIX.get(fmt, _FORMAT_CAPABILITY_MATRIX["parquet"])
        return SourceCapabilities(
            projection_pushdown=bool(base["projection_pushdown"]),
            filter_pushdown=bool(base["filter_pushdown"]),
            partition_pruning=bool(base["partition_pruning"]),
            streaming=bool(base["streaming"]),
            remote=self._storage_remote,
            snapshot=bool(base["snapshot"]),
            write=bool(base["write"]),
            transaction=bool(base["transaction"]),
            asof=bool(base["asof"]),
        )

    def resolve(self, request: Any, contract: Any = None) -> dict[str, Any]:
        spec = getattr(self._format_adapter, "spec", None)
        return {
            "source": self.name,
            "format": self._format_name(),
            "uses_duckdb": bool(getattr(self._format_adapter, "uses_duckdb", True)),
            "spec": spec.to_dict() if hasattr(spec, "to_dict") else None,
            "request": request,
            "contract": contract,
        }

    def estimate(self, resolved: Any) -> dict[str, Any]:
        resolved = resolved or {}
        rows, byte_size = _request_row_count_byte_size(resolved.get("request"))
        uses_duckdb = bool(resolved.get("uses_duckdb", True))
        return {
            "estimated_rows": rows,
            "estimated_bytes": byte_size,
            "engine": "duckdb" if uses_duckdb else "pyarrow",
        }

    def scan(self, prepared: Any) -> dict[str, Any]:
        prepared = prepared or {}
        if not bool(getattr(self._format_adapter, "uses_duckdb", True)):
            # arrow/feather：DuckDB 不能按文件路径读 → PyArrow 引擎直读
            return {
                "kind": "pyarrow",
                "source": self.name,
                "format": self._format_name(),
                "uses_duckdb": False,
            }
        path = prepared.get("path") if isinstance(prepared, Mapping) else None
        build = getattr(self._format_adapter, "build_from_clause", None)
        from_clause = None
        if callable(build) and path is not None:
            try:
                from_clause = build(path, hive_partitioning=False, union_by_name=False)
            except Exception:
                from_clause = None
        return {
            "kind": "format",
            "source": self.name,
            "format": self._format_name(),
            "uses_duckdb": True,
            "from_clause": from_clause,
        }

    def snapshot(self, resolved: Any) -> None:
        # 格式层不持有对象版本元数据（snapshot 元数据属存储后端 BackendCapability）
        return None

    def healthcheck(self) -> bool:
        try:
            if self._format_name() is None:
                return bool(getattr(self._format_adapter, "build_from_clause", None))
            return True
        except Exception:
            return False

    # ---- 内部 ----

    def _format_name(self) -> str:
        fmt = getattr(self._format_adapter, "data_format", None)
        if fmt is not None:
            fmt = getattr(fmt, "value", None) or str(fmt)
        if fmt is None:
            spec = getattr(self._format_adapter, "spec", None)
            if spec is not None:
                fmt = getattr(spec, "type", None)
        return str(fmt or "parquet").strip().lower()


# ---------------------------------------------------------------------------
# 查询后端能力矩阵（duckdb/polars/pyarrow/local/cos/clickhouse）。
# 键：SourceCapabilities 字段名 + asof_join/window/groupby 扩展能力位。
# ---------------------------------------------------------------------------

BACKEND_CAPABILITY_MATRIX: dict[str, dict[str, bool]] = {
    "duckdb": {
        "projection_pushdown": True,
        "filter_pushdown": True,
        "partition_pruning": True,
        "asof_join": True,
        "streaming": True,
        "window": True,
        "groupby": True,
        "write": True,
        "remote": True,
        "snapshot": True,
        "transaction": True,
    },
    "polars": {
        "projection_pushdown": True,
        "filter_pushdown": True,
        "partition_pruning": False,
        "asof_join": True,
        "streaming": True,
        "window": True,
        "groupby": True,
        "write": False,
        "remote": False,
        "snapshot": False,
        "transaction": False,
    },
    "pyarrow": {
        "projection_pushdown": True,
        "filter_pushdown": True,
        "partition_pruning": False,
        "asof_join": False,
        "streaming": True,
        "window": False,
        "groupby": False,
        "write": True,
        "remote": False,
        "snapshot": False,
        "transaction": False,
    },
    "local": {
        "projection_pushdown": True,
        "filter_pushdown": True,
        "partition_pruning": True,
        "asof_join": False,
        "streaming": True,
        "window": True,
        "groupby": True,
        "write": True,
        "remote": False,
        "snapshot": False,
        "transaction": False,
    },
    "cos": {
        "projection_pushdown": True,
        "filter_pushdown": True,
        "partition_pruning": True,
        "asof_join": False,
        "streaming": True,
        "window": True,
        "groupby": True,
        "write": True,
        "remote": True,
        "snapshot": True,
        "transaction": False,
    },
    "clickhouse": {
        "projection_pushdown": True,
        "filter_pushdown": True,
        "partition_pruning": True,
        "asof_join": False,
        "streaming": True,
        "window": True,
        "groupby": True,
        "write": True,
        "remote": True,
        "snapshot": False,
        "transaction": True,
    },
}


def source_capabilities_from_matrix(matrix: Mapping[str, bool]) -> SourceCapabilities:
    """把后端能力矩阵一行翻译成 SourceCapabilities（asof ← asof_join）。"""
    return SourceCapabilities(
        projection_pushdown=bool(matrix.get("projection_pushdown", False)),
        filter_pushdown=bool(matrix.get("filter_pushdown", False)),
        partition_pruning=bool(matrix.get("partition_pruning", False)),
        streaming=bool(matrix.get("streaming", False)),
        remote=bool(matrix.get("remote", False)),
        snapshot=bool(matrix.get("snapshot", False)),
        write=bool(matrix.get("write", False)),
        transaction=bool(matrix.get("transaction", False)),
        asof=bool(matrix.get("asof_join", matrix.get("asof", False))),
    )


def choose_backend(required: SourceCapabilities, costs: Mapping[str, float]) -> str | None:
    """从能力矩阵选满足 ``required`` 且 cost 最小的后端；全部不满足返回 None。"""
    best: str | None = None
    best_cost = float("inf")
    for backend, matrix in BACKEND_CAPABILITY_MATRIX.items():
        caps = source_capabilities_from_matrix(matrix)
        if not required.subset(caps):
            continue
        cost = costs.get(backend)
        if cost is None:
            continue
        if cost < best_cost:
            best_cost = cost
            best = backend
    return best


class BackendAdapter:
    """查询后端（duckdb/polars/pyarrow/local/cos/clickhouse）的 SourceAdapter。"""

    def __init__(self, backend: str) -> None:
        norm = str(backend or "").strip().lower()
        if norm not in BACKEND_CAPABILITY_MATRIX:
            raise ValueError(
                f"未知查询后端 {backend!r}；支持 {sorted(BACKEND_CAPABILITY_MATRIX)}"
            )
        self.backend = norm

    # ---- SourceAdapter 协议实现 ----

    def capabilities(self) -> SourceCapabilities:
        return source_capabilities_from_matrix(BACKEND_CAPABILITY_MATRIX[self.backend])

    def resolve(self, request: Any, contract: Any = None) -> dict[str, Any]:
        return {"backend": self.backend, "request": request, "contract": contract}

    def estimate(self, resolved: Any) -> dict[str, Any]:
        resolved = resolved or {}
        rows, byte_size = _request_row_count_byte_size(resolved.get("request"))
        return {
            "estimated_rows": rows,
            "estimated_bytes": byte_size,
            "backend": self.backend,
        }

    def scan(self, prepared: Any) -> dict[str, Any]:
        prepared = prepared or {}
        scan_type = prepared.get("scan_type") if isinstance(prepared, Mapping) else None
        return {"backend": self.backend, "scan_type": scan_type}

    def snapshot(self, resolved: Any) -> None:
        return None

    def healthcheck(self) -> bool:
        return self.backend in BACKEND_CAPABILITY_MATRIX


def _request_row_count_byte_size(request: Any) -> tuple[int, int]:
    """从 request（dict 或对象）防御性取 row_count / byte_size。"""
    rows = 0
    byte_size = 0
    if isinstance(request, Mapping):
        rows = int(request.get("row_count", 0) or 0)
        byte_size = int(request.get("byte_size", 0) or 0)
    elif request is not None:
        rows = int(getattr(request, "row_count", 0) or 0)
        byte_size = int(getattr(request, "byte_size", 0) or 0)
    return rows, byte_size
