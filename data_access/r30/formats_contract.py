"""data_access.r30.formats_contract —— CSV 生产契约（R30-P1-002）。

目标：把 CSV / Arrow IPC 正式纳入统一读取路径（包装既有 read/formats.py，不编辑），
并给 CSV 提供**显式 schema 的生产契约**——不依赖自动 dtype inference：

    1. ``CSVProductionContract(schema, delimiter, encoding, header, ...)``：
       ``schema`` 为 {列名: pyarrow 类型} 的显式类型声明；
    2. ``validate(path) -> list[str]``：用纯 pyarrow 按显式 schema 读文件头/类型，
       校验与契约一致，返回问题列表（空 = 通过）；
    3. ``build_csv_contract_from_table(table, delimiter=',')``：从 Arrow Table 生成契约；
    4. ``validate_no_pandas(path)``：保证**只走 pyarrow** 校验，绝不触发 pandas。

CSV 默认类型映射（未显式声明时，built_from_table 直接取 Arrow 类型，天然显式）：
本契约把 ``read/formats.py`` 的 CSVAdapter（DuckDB read_csv）能力声明为
streaming=True / asof=False——见 r30/adapter.py。
"""
from __future__ import annotations

from typing import Any, Mapping

__all__ = [
    "CSVProductionContract",
    "build_csv_contract_from_table",
]


def _to_pa_type(t: Any) -> Any:
    """把 pyarrow 类型 / pyarrow DataType 子类 / 类型字符串统一成 pyarrow.DataType。"""
    import pyarrow as pa

    if isinstance(t, pa.DataType):
        return t
    if isinstance(t, type) and issubclass(t, pa.DataType):
        return t()
    text = str(t).strip()
    try:
        return pa.type_for_alias(text)
    except Exception:
        # 兜底常见别名
        aliases = {
            "str": pa.string(),
            "varchar": pa.string(),
            "text": pa.string(),
            "float": pa.float64(),
            "double": pa.float64(),
            "int": pa.int64(),
            "timestamp": pa.timestamp("us"),
            "datetime": pa.timestamp("us"),
            "bool": pa.bool_(),
            "boolean": pa.bool_(),
        }
        low = text.lower()
        if low in aliases:
            return aliases[low]
        raise


def _read_csv_table(
    path: str,
    *,
    schema: Any,
    delimiter: str,
    encoding: str,
    header: bool,
    null_representation: str | None,
):
    """按显式 schema 读 CSV（纯 pyarrow，无自动 dtype inference）。"""
    import pyarrow.csv as pa_csv

    parse_options = pa_csv.ParseOptions(delimiter=delimiter)
    read_options = pa_csv.ReadOptions(
        encoding=encoding,
        column_names=list(schema.names) if not header else None,
    )
    convert_kwargs: dict[str, Any] = {
        "column_types": schema,
        "include_columns": list(schema.names),
        "include_missing_columns": False,
    }
    if null_representation is not None:
        convert_kwargs["null_values"] = [null_representation]
    convert_options = pa_csv.ConvertOptions(**convert_kwargs)
    return pa_csv.read_csv(
        path,
        read_options=read_options,
        parse_options=parse_options,
        convert_options=convert_options,
    )


class CSVProductionContract:
    """CSV 生产契约：显式 schema + 文件形态参数，逐文件校验。"""

    def __init__(
        self,
        schema: Mapping[str, Any],
        delimiter: str = ",",
        encoding: str = "utf-8",
        header: bool = True,
        null_representation: str | None = "",
        timestamp_parse: str | None = None,
        float_format: str | None = None,
    ) -> None:
        # 显式 schema：{列名: pa.DataType | 类型字符串}
        self.schema: dict[str, Any] = dict(schema)
        self.delimiter = delimiter
        self.encoding = encoding
        self.header = bool(header)
        self.null_representation = null_representation
        self.timestamp_parse = timestamp_parse
        self.float_format = float_format

    # ---- 序列化 ----

    def to_dict(self) -> dict[str, Any]:
        schema_out: dict[str, str] = {}
        for name, t in self.schema.items():
            try:
                schema_out[str(name)] = str(_to_pa_type(t))
            except Exception:
                schema_out[str(name)] = str(t)
        return {
            "schema": schema_out,
            "delimiter": self.delimiter,
            "encoding": self.encoding,
            "header": self.header,
            "null_representation": self.null_representation,
            "timestamp_parse": self.timestamp_parse,
            "float_format": self.float_format,
        }

    # ---- 校验 ----

    def validate(self, path: str) -> list[str]:
        """校验文件与契约一致；返回问题列表（空 = 通过）。

        不依赖自动 dtype inference：按显式 schema 读文件头/类型。
        """
        import pyarrow as pa

        problems: list[str] = []

        # 1) 契约自身 schema 必须能编译成 pyarrow schema
        try:
            schema = pa.schema(
                [(str(name), _to_pa_type(t)) for name, t in self.schema.items()]
            )
        except Exception as exc:
            return [f"contract schema invalid: {type(exc).__name__}: {exc}"]

        # 2) 列名核对（header=True 时读文件头，纯 pyarrow，不做 dtype 推断）
        if self.header:
            file_names = _csv_header_names(
                path, delimiter=self.delimiter, encoding=self.encoding
            )
            if file_names is not None and list(file_names) != list(schema.names):
                problems.append(
                    f"columns mismatch: contract={list(schema.names)} file={file_names}"
                )

        # 3) 按显式 schema 读数据（列缺失/类型冲突 → 抛 → 记问题）
        try:
            table = _read_csv_table(
                path,
                schema=schema,
                delimiter=self.delimiter,
                encoding=self.encoding,
                header=self.header,
                null_representation=self.null_representation,
            )
        except Exception as exc:
            problems.append(
                f"csv does not match contract: {type(exc).__name__}: {exc}"
            )
            return problems

        # 4) 防御性类型核对（column_types 会 cast，正常情况下不会走到）
        for name in schema.names:
            if name not in table.column_names:
                problems.append(f"column {name} missing after read")
                continue
            actual_type = table.schema.field(name).type
            expected_type = schema.field(name).type
            if not actual_type.equals(expected_type):
                problems.append(
                    f"column {name} type mismatch: contract={expected_type} "
                    f"file={actual_type}"
                )
        return problems

    def validate_no_pandas(self, path: str) -> list[str]:
        """纯 pyarrow 校验：保证不触发 pandas.read_csv。

        与 ``validate`` 同一条路径，但显式声明只允许 pyarrow——本契约的实现
        从未 import pandas。
        """
        return self.validate(path)


def build_csv_contract_from_table(
    table: Any,
    delimiter: str = ",",
    *,
    encoding: str = "utf-8",
    header: bool = True,
    null_representation: str | None = "",
) -> CSVProductionContract:
    """从 Arrow Table 生成 CSV 生产契约（schema 取显式 Arrow 类型）。"""
    schema = {
        name: table.schema.field(i).type
        for i, name in enumerate(table.column_names)
    }
    return CSVProductionContract(
        schema=schema,
        delimiter=delimiter,
        encoding=encoding,
        header=header,
        null_representation=null_representation,
    )


def _csv_header_names(
    path: str, *, delimiter: str, encoding: str
) -> list[str] | None:
    """读文件头列名（纯 pyarrow，仅取列名，不做 dtype 推断）。"""
    try:
        import pyarrow.csv as pa_csv

        t = pa_csv.read_csv(
            path,
            read_options=pa_csv.ReadOptions(encoding=encoding),
            parse_options=pa_csv.ParseOptions(delimiter=delimiter),
            convert_options=pa_csv.ConvertOptions(include_columns=[]),
        )
        return list(t.column_names)
    except Exception:
        return None
