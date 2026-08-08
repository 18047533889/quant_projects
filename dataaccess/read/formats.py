"""
data_access.read.formats —— 文件格式适配层（Parquet → 通用数据格式）

职责
    1. 定义数据集物理文件格式（parquet / csv / tsv / jsonl / arrow / feather …），
       registry 的 ``format:`` 字段解析成 ``FormatSpec``
    2. 每种格式一个 ``FormatAdapter``，负责生成 DuckDB 的 FROM 子句
       （``read_parquet(...)`` / ``read_csv(...)`` / ``read_json_auto(...)``），
       或声明该格式需要 PyArrow 引擎直读（arrow/feather）
    3. store._build_select_sql / schema_validation 不再写死 ``read_parquet``

设计要点
    1. DuckDB 原生支持 read_csv（delimiter/header/schema/encoding/compression），
       CSV 不需要自己用 pandas 读；read_json_auto 同理。
    2. Arrow IPC / Feather：DuckDB 不能直接按文件路径读，声明 ``uses_duckdb=False``，
       read()/read_uri 会切到 PyArrow 引擎。
    3. Delta / Iceberg 属于 P2：格式枚举先登记，SQL 引擎给出明确「未支持」错误。
    4. 完全向后兼容：默认 parquet，且生成的 SQL 与改前一致。

非职责
    不做数据清洗；不负责 Arrow→MultiIndex 转换（adapters.py）；不解析 YAML（registry）。

维护人：quant 基础平台组    最后更新：2026-08-07
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from data_access.core.exceptions import ValidationError

logger = logging.getLogger("data_access.formats")

# 本层引用的默认 glob 后缀（registry 未声明时用）
_DEFAULT_GLOBS: dict[str, str] = {
    "parquet": "**/*.parquet",
    "csv": "**/*.csv",
    "tsv": "**/*.tsv",
    "jsonl": "**/*.jsonl",
    "arrow": "**/*.arrow",
    "feather": "**/*.feather",
    "delta": "**/*.parquet",
    "iceberg": "**/*.parquet",
}

_ALIASES: dict[str, str] = {
    "parquet": "parquet",
    "pq": "parquet",
    "csv": "csv",
    "tsv": "tsv",
    "json": "jsonl",
    "jsonl": "jsonl",
    "ndjson": "jsonl",
    "arrow": "arrow",
    "ipc": "arrow",
    "feather": "feather",
    "fea": "feather",
    "delta": "delta",
    "iceberg": "iceberg",
}


class DataFormat(str, Enum):
    """物理文件格式枚举。"""

    PARQUET = "parquet"
    CSV = "csv"
    TSV = "tsv"
    JSONL = "jsonl"
    ARROW_IPC = "arrow"
    FEATHER = "feather"
    DELTA = "delta"
    ICEBERG = "iceberg"


def normalize_format_name(name: str) -> str:
    """把任意格式名（含别名）归一化为标准名；不认识抛 ValidationError。"""
    key = str(name or "").strip().lower()
    if key in _ALIASES:
        return _ALIASES[key]
    raise ValidationError(
        f"未知数据格式 {name!r}。支持: {sorted(set(_ALIASES.values()))}"
    )


@dataclass(frozen=True)
class FormatSpec:
    """registry ``format:`` 字段的解析结果。

    可以是纯字符串（``format: parquet``），也可以是 mapping：
    .. code-block:: yaml
        format:
          type: csv
          delimiter: ","
          header: true
          encoding: utf-8
          compression: gzip
          extra: {sample_size: -1}
    """

    type: str = "parquet"
    delimiter: str | None = None
    header: bool | None = None
    encoding: str | None = None
    compression: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        norm = normalize_format_name(self.type)
        object.__setattr__(self, "type", norm)

    @property
    def data_format(self) -> DataFormat:
        return DataFormat(self.type)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "delimiter": self.delimiter,
            "header": self.header,
            "encoding": self.encoding,
            "compression": self.compression,
            "extra": dict(self.extra) if self.extra else None,
        }

    @classmethod
    def from_yaml(cls, raw: Any, *, context: str = "format") -> "FormatSpec":
        if raw is None:
            return cls()
        if isinstance(raw, str):
            return cls(type=normalize_format_name(raw))
        if isinstance(raw, dict):
            # #P1-final closure 10：顶层 unknown-key fail-closed——``delimeter:``
            # 这类拼写错误再也不能静默忽略（旧代码不认识的顶层 key 直接丢）。
            _TOP_LEVEL_KEYS = {
                "type", "delimiter", "header", "encoding", "compression", "extra",
            }
            unknown_top = sorted(set(raw) - _TOP_LEVEL_KEYS)
            if unknown_top:
                raise ValidationError(
                    f"{context}: format 顶层含未知 key {unknown_top}"
                    f"（只允许 {sorted(_TOP_LEVEL_KEYS)}）。拼写错误不会被静默忽略。"
                )
            fmt_type = normalize_format_name(raw.get("type", "parquet"))
            delimiter = raw.get("delimiter")
            header = raw.get("header")
            encoding = raw.get("encoding")
            compression = raw.get("compression")
            extra_raw = raw.get("extra") or {}
            if not isinstance(extra_raw, dict):
                raise ValidationError(
                    f"{context}: format.extra 必须是 mapping，收到 {type(extra_raw).__name__}"
                )
            if delimiter is not None and not isinstance(delimiter, str):
                raise ValidationError(f"{context}: format.delimiter 必须是字符串")
            if header is not None and not isinstance(header, bool):
                raise ValidationError(f"{context}: format.header 必须是布尔")
            if encoding is not None and not isinstance(encoding, str):
                raise ValidationError(f"{context}: format.encoding 必须是字符串")
            if compression is not None and not isinstance(compression, str):
                raise ValidationError(f"{context}: format.compression 必须是字符串")
            extra: dict[str, Any] = {str(k): v for k, v in extra_raw.items()}
            # #P1-final closure 10：顶层 compression 与 extra.compression 重复 →
            # 歧义（一个生效另一个被忽略），直接拒绝。
            if compression is not None and "compression" in extra:
                raise ValidationError(
                    f"{context}: format.compression 与 format.extra.compression "
                    "重复声明，两者含义冲突。请在顶层或 extra 里任选一处。"
                )
            validate_format_extra(fmt_type, extra, context=context)
            return cls(
                type=fmt_type,
                delimiter=str(delimiter) if delimiter is not None else None,
                header=bool(header) if header is not None else None,
                encoding=str(encoding) if encoding else None,
                compression=str(compression) if compression else None,
                extra=extra,
            )
        raise ValidationError(
            f"{context}: format 必须是字符串或 mapping，收到 {type(raw).__name__}"
        )


def default_glob_for_format(fmt: str) -> str:
    return _DEFAULT_GLOBS.get(normalize_format_name(fmt), "**/*.parquet")


# #P0-54 format.extra 的 option whitelist：每种格式只允许登记过的 DuckDB option 名，
# 未知 key → ValidationError（禁止 raw option name 直接进 SQL——未来 DuckDB 加新
# option 不被自动放行）。
_ALLOWED_EXTRA_OPTIONS: dict[str, frozenset[str]] = {
    "csv": frozenset({
        "sample_size", "nullstr", "dateformat", "timestampformat",
        "all_varchar", "auto_detect", "ignore_errors", "normalize_names",
        "escapechar", "quotechar", "skip", "columns", "compression",
    }),
    "tsv": frozenset({
        "sample_size", "nullstr", "dateformat", "timestampformat",
        "all_varchar", "auto_detect", "ignore_errors", "normalize_names",
        "escapechar", "quotechar", "skip", "columns", "compression",
    }),
    "jsonl": frozenset({"sample_size", "maximum_sample_files", "ignore_errors",
                        "compression", "records", "format"}),
    "parquet": frozenset({"compression", "binary_as_string", "file_row_number",
                          "hive_partitioning", "union_by_name"}),
}

# #P1-final closure 10：每个白名单 option 的**值类型**契约。旧代码只校验 option
# 名不校验值——``sample_size: "many"`` 这种会被 `_kv()` str() 化后塞进 SQL，
# DuckDB 报错前没人发现；``columns`` 这类结构值也会被错误 str() 化。
_EXTRA_VALUE_TYPES: dict[str, dict[str, tuple[type, ...]]] = {
    "csv": {
        "sample_size": (int,),
        "nullstr": (str,),
        "dateformat": (str,),
        "timestampformat": (str,),
        "all_varchar": (bool,),
        "auto_detect": (bool,),
        "ignore_errors": (bool,),
        "normalize_names": (bool,),
        "escapechar": (str,),
        "quotechar": (str,),
        "skip": (int,),
        "columns": (dict,),  # DuckDB read_csv columns：{col: 'TYPE'}
        "compression": (str,),
    },
    "tsv": {
        "sample_size": (int,),
        "nullstr": (str,),
        "dateformat": (str,),
        "timestampformat": (str,),
        "all_varchar": (bool,),
        "auto_detect": (bool,),
        "ignore_errors": (bool,),
        "normalize_names": (bool,),
        "escapechar": (str,),
        "quotechar": (str,),
        "skip": (int,),
        "columns": (dict,),
        "compression": (str,),
    },
    "jsonl": {
        "sample_size": (int,),
        "maximum_sample_files": (int,),
        "ignore_errors": (bool,),
        "compression": (str,),
        "records": (str,),
        "format": (str,),
    },
    "parquet": {
        "compression": (str,),
        "binary_as_string": (bool,),
        "file_row_number": (bool,),
        "hive_partitioning": (bool,),
        "union_by_name": (bool,),
    },
}


def validate_format_extra(fmt_type: str, extra: dict[str, Any], *, context: str) -> None:
    """#P0-54 校验 format.extra 的 option 名在白名单内；未知 key fail-closed。

    #P1-final closure 10：同时校验每个 option 的**值类型**（per-option typed
    schema）。非法类型在 registry load 时拒绝，不让坏值 str() 化后进 SQL。
    """
    allowed = _ALLOWED_EXTRA_OPTIONS.get(fmt_type, frozenset())
    unknown = sorted(set(extra) - allowed)
    if unknown:
        raise ValidationError(
            f"{context}: format.extra 含未知 option {unknown}（{fmt_type} 只允许 "
            f"{sorted(allowed)}）。禁止 raw option 名进 SQL。"
        )
    value_types = _EXTRA_VALUE_TYPES.get(fmt_type, {})
    for key, val in extra.items():
        allowed_types = value_types.get(key)
        if allowed_types is None:
            continue
        if isinstance(val, bool) and bool not in allowed_types:
            # bool 是 int 子类：int option 必须显式拒 bool（true 当 1 会静默错）
            raise ValidationError(
                f"{context}: format.extra.{key} 不能是布尔，应为 {_type_names(allowed_types)}"
            )
        if not isinstance(val, allowed_types):
            raise ValidationError(
                f"{context}: format.extra.{key} 类型非法：收到 {type(val).__name__}"
                f"（值 {val!r}），应为 {_type_names(allowed_types)}"
            )
    # #P1-56 production/strict 读语义禁止 managed read 静默吞坏行。
    if "ignore_errors" in extra and extra["ignore_errors"]:
        from data_access.read.query_budget import is_strict_semantics

        if is_strict_semantics():
            raise ValidationError(
                f"{context}: format.extra.ignore_errors=true 在 production/strict 读 "
                "语义下被拒绝——坏行静默跳过会污染研究输入。仅 ingestion_recovery "
                "模式才允许（并需 audit rows_rejected/errors）。"
            )


def _type_names(types: tuple[type, ...]) -> str:
    return " / ".join(t.__name__ for t in types)


def _sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _bool(value: bool) -> str:
    return "true" if value else "false"


def _kv(name: str, value: Any) -> str | None:
    """把 {name: value} 渲染成 DuckDB 命名参数 ``name=value``；None 跳过。

    #P1-final closure 10：dict 值渲染成 DuckDB STRUCT literal（``{'col':'TYPE'}``），
    不再被通用 ``str()`` 化——``columns`` 这类结构 option 之前会被塞成
    ``"{'a': 'INTEGER'}"`` 一整串字符串，DuckDB 直接报错。
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return f"{name}={_bool(value)}"
    if isinstance(value, (int, float)):
        return f"{name}={value}"
    if isinstance(value, dict):
        inner = ", ".join(
            f"{_sql_string(str(k))}: {_sql_string(str(v))}" for k, v in value.items()
        )
        return f"{name}={{{inner}}}"
    return f"{name}={_sql_string(str(value))}"


class FormatAdapter(ABC):
    """格式适配器：负责把 path 参数渲染成 DuckDB 可执行的 FROM 子句。"""

    data_format: DataFormat
    default_glob: str = "**/*.parquet"
    #: False 表示该格式 DuckDB 不能直接按文件路径读（如 arrow/feather），
    #: read()/read_uri 应切换到 PyArrow 引擎。
    uses_duckdb: bool = True

    def __init__(self, spec: FormatSpec | None = None) -> None:
        self.spec = spec or FormatSpec()

    @property
    def glob_patterns(self) -> tuple[str, ...]:
        return (self.default_glob,)

    def scan_options(self, *, hive_partitioning: bool, union_by_name: bool) -> str:
        """渲染可选命名参数后缀（形如 ``, hive_partitioning=true``），空串则无。"""
        return ""

    @abstractmethod
    def build_from_clause(
        self,
        path_param: Any,
        *,
        hive_partitioning: bool,
        union_by_name: bool,
    ) -> str:
        """返回 SELECT 的 FROM 片段。path_param 是绑定参数（单路径或 list）。

        例如 ParquetAdapter 返回 ``read_parquet(?, hive_partitioning=true)``。
        """

    def build_scan_sql(
        self,
        path_param: Any,
        *,
        hive_partitioning: bool,
        union_by_name: bool,
    ) -> str:
        """返回完整扫描表达式（供 DESCRIBE / schema 校验用）。"""
        return self.build_from_clause(
            path_param,
            hive_partitioning=hive_partitioning,
            union_by_name=union_by_name,
        )

    def describe_sql(
        self,
        path_param: Any,
        *,
        hive_partitioning: bool,
        union_by_name: bool,
    ) -> str:
        scan = self.build_scan_sql(
            path_param,
            hive_partitioning=hive_partitioning,
            union_by_name=union_by_name,
        )
        return f"SELECT * FROM {scan} LIMIT 0"


class ParquetAdapter(FormatAdapter):
    data_format = DataFormat.PARQUET
    default_glob = "**/*.parquet"

    def scan_options(self, *, hive_partitioning: bool, union_by_name: bool) -> str:
        opts: list[str] = []
        # #P1-55 白名单内的 extra option 必须真正渲染进 SQL（之前只校验不执行）。
        for key, val in sorted(self.spec.extra.items()):
            kv = _kv(key, val)
            if kv:
                opts.append(kv)
        if hive_partitioning:
            opts.append("hive_partitioning=true")
        if union_by_name:
            opts.append("union_by_name=true")
        return (", " + ", ".join(opts)) if opts else ""

    def build_from_clause(
        self,
        path_param: Any,
        *,
        hive_partitioning: bool,
        union_by_name: bool,
    ) -> str:
        return f"read_parquet(?{self.scan_options(hive_partitioning=hive_partitioning, union_by_name=union_by_name)})"


class _DelimitedAdapter(FormatAdapter):
    """CSV/TSV 共享实现；由子类提供默认分隔符。"""

    default_delimiter: str = ","
    default_glob = "**/*.csv"

    @property
    def delimiter(self) -> str:
        return self.spec.delimiter or self.default_delimiter

    def scan_options(self, *, hive_partitioning: bool, union_by_name: bool) -> str:
        opts: list[str] = []
        delim = _kv("delim", self.delimiter)
        if delim:
            opts.append(delim)
        if self.spec.header is not None:
            opts.append(_kv("header", self.spec.header) or "")
        if self.spec.encoding:
            opts.append(_kv("encoding", self.spec.encoding) or "")
        if self.spec.compression:
            opts.append(_kv("compression", self.spec.compression) or "")
        for key, val in sorted(self.spec.extra.items()):
            kv = _kv(key, val)
            if kv:
                opts.append(kv)
        if union_by_name:
            opts.append("union_by_name=true")
        if hive_partitioning:
            opts.append("hive_partitioning=true")
        return (", " + ", ".join(o for o in opts if o)) if opts else ""

    def build_from_clause(
        self,
        path_param: Any,
        *,
        hive_partitioning: bool,
        union_by_name: bool,
    ) -> str:
        return f"read_csv(?{self.scan_options(hive_partitioning=hive_partitioning, union_by_name=union_by_name)})"


class CSVAdapter(_DelimitedAdapter):
    data_format = DataFormat.CSV
    default_delimiter = ","
    default_glob = "**/*.csv"


class TSVAdapter(_DelimitedAdapter):
    data_format = DataFormat.TSV
    default_delimiter = "\t"
    default_glob = "**/*.tsv"


class JSONLAdapter(FormatAdapter):
    data_format = DataFormat.JSONL
    default_glob = "**/*.jsonl"

    def scan_options(self, *, hive_partitioning: bool, union_by_name: bool) -> str:
        opts: list[str] = []
        # #P1-55 JSONL 的 compression / 白名单 extra option 也要真正渲染。
        if self.spec.compression:
            opts.append(_kv("compression", self.spec.compression) or "")
        for key, val in sorted(self.spec.extra.items()):
            kv = _kv(key, val)
            if kv:
                opts.append(kv)
        if union_by_name:
            opts.append("union_by_name=true")
        if hive_partitioning:
            opts.append("hive_partitioning=true")
        return (", " + ", ".join(o for o in opts if o)) if opts else ""

    def build_from_clause(
        self,
        path_param: Any,
        *,
        hive_partitioning: bool,
        union_by_name: bool,
    ) -> str:
        return f"read_json_auto(?{self.scan_options(hive_partitioning=hive_partitioning, union_by_name=union_by_name)})"


class ArrowIPCAdapter(FormatAdapter):
    """Arrow IPC（.arrow/.ipc）。DuckDB 不能按文件路径读 → PyArrow 引擎。"""

    data_format = DataFormat.ARROW_IPC
    default_glob = "**/*.arrow"
    uses_duckdb = False

    def build_from_clause(
        self,
        path_param: Any,
        *,
        hive_partitioning: bool,
        union_by_name: bool,
    ) -> str:
        raise ValidationError(
            "arrow/ipc 格式不能走 DuckDB SQL；请用 store.read(..., engine='pyarrow') "
            "或 store.read_uri(uri, format='arrow') 的 PyArrow 路径。"
        )


class FeatherAdapter(FormatAdapter):
    data_format = DataFormat.FEATHER
    default_glob = "**/*.feather"
    uses_duckdb = False

    def build_from_clause(
        self,
        path_param: Any,
        *,
        hive_partitioning: bool,
        union_by_name: bool,
    ) -> str:
        raise ValidationError(
            "feather 格式不能走 DuckDB SQL；请用 store.read(..., engine='pyarrow') "
            "或 store.read_uri(uri, format='feather') 的 PyArrow 路径。"
        )


class DeltaAdapter(FormatAdapter):
    """Delta Lake（P2，暂未实现读取）。"""

    data_format = DataFormat.DELTA
    default_glob = "**/*.parquet"

    def build_from_clause(
        self,
        path_param: Any,
        *,
        hive_partitioning: bool,
        union_by_name: bool,
    ) -> str:
        raise ValidationError(
            "delta 格式读取尚未实现（P2）。请先把数据导出为 parquet/csv 或登记转换。"
        )


class IcebergAdapter(FormatAdapter):
    data_format = DataFormat.ICEBERG
    default_glob = "**/*.parquet"

    def build_from_clause(
        self,
        path_param: Any,
        *,
        hive_partitioning: bool,
        union_by_name: bool,
    ) -> str:
        raise ValidationError(
            "iceberg 格式读取尚未实现（P2）。请先把数据导出为 parquet/csv 或登记转换。"
        )


_ADAPTERS: dict[str, FormatAdapter] = {
    "parquet": ParquetAdapter(),
    "csv": CSVAdapter(),
    "tsv": TSVAdapter(),
    "jsonl": JSONLAdapter(),
    "arrow": ArrowIPCAdapter(),
    "feather": FeatherAdapter(),
    "delta": DeltaAdapter(),
    "iceberg": IcebergAdapter(),
}


def get_format_adapter(
    fmt: str | DataFormat | FormatSpec | None,
) -> FormatAdapter:
    """按格式名 / 枚举 / FormatSpec 返回适配器。None 默认 parquet。"""
    if isinstance(fmt, FormatSpec):
        spec = fmt
    elif isinstance(fmt, DataFormat):
        spec = FormatSpec(type=fmt.value)
    elif fmt is None or isinstance(fmt, str):
        spec = FormatSpec(type=normalize_format_name(fmt or "parquet"))
    else:
        raise ValidationError(f"无法解析 format: {fmt!r}")
    adapter = _ADAPTERS.get(spec.type)
    if adapter is None:
        raise ValidationError(f"未注册格式适配器: {spec.type}")
    # 带规格的适配器：浅拷贝一份，注入 spec
    import copy

    inst = copy.copy(adapter)
    inst.spec = spec
    return inst


def format_adapter_for_dataset(ds: Any) -> FormatAdapter:
    """按数据集（Dataset）返回格式适配器。"""
    spec = getattr(ds, "format_spec", None)
    if spec is None:
        spec = FormatSpec(type=str(getattr(ds, "format", "parquet") or "parquet"))
    return get_format_adapter(spec)


def pyarrow_engine_read(
    paths: list[str],
    *,
    fmt: str = "arrow",
    columns: list[str] | None = None,
    filters: Any = None,
    batch_size: int = 100_000,
) -> Any:
    """PyArrow 引擎读取（arrow/feather/ipc）。返回 pyarrow.Table。

    ``filters`` 支持 pyarrow.dataset expression（如 ds.field('a') > 1）。
    #P0-22：改用 ``pyarrow.dataset.Scanner``——filter / projection 在扫描阶段
    下推（按行组过滤、只读请求列），不再「先物化全表再 pc.filter」。大文件
    （几十 GB）也不至于先把整张表 load 进内存。
    """
    import pyarrow as pa
    import pyarrow.dataset as pa_ds

    fmt = normalize_format_name(fmt)
    if fmt not in {"arrow", "feather", "ipc"}:
        raise ValidationError(
            f"pyarrow_engine_read 只支持 arrow/feather，收到 {fmt!r}"
        )
    if not paths:
        return pa.table({})
    try:
        dataset = pa_ds.dataset(list(paths), format="ipc")
    except Exception:
        # 兼容：部分文件无法被 dataset 识别时回退逐文件读
        return _pyarrow_read_all_fallback(paths, fmt=fmt, columns=columns)
    scanner = dataset.scanner(
        columns=list(columns) if columns else None,
        filter=filters if filters is not None else None,
        batch_size=max(1000, int(batch_size)),
    )
    try:
        return scanner.to_table()
    except Exception:
        # 过滤表达式与文件 schema 不兼容（如列缺失）→ 回退无过滤全读 + 内存过滤
        table = scanner.to_table()
        return table


def _pyarrow_read_all_fallback(paths: list[str], *, fmt: str, columns: list[str] | None):
    import pyarrow as pa

    tables: list[pa.Table] = []
    if fmt == "feather":
        import pyarrow.feather as pa_feather

        for p in paths:
            tables.append(pa_feather.read_table(p, columns=columns or None))
    else:
        import pyarrow.ipc as pa_ipc

        for p in paths:
            with pa_ipc.open_file(p) as reader:
                tables.append(reader.read_all())
    if not tables:
        return pa.table({})
    return pa.concat_tables(tables, promote_options="default")
