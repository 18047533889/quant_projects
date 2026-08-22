# -*- coding: utf-8 -*-
"""HTTP 读数服务请求/响应模型。"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictRequestModel(BaseModel):
    """R25 §59：所有 request 都 ``extra="forbid"``——未知字段直接 422，不静默忽略。

    未知 key 静默忽略会掩盖 typo（``instument_filter`` 悄悄变成 None，安全预算 /
    授权路径被绕过）。Pydantic v2 用 ``ConfigDict(extra="forbid")``。
    """

    model_config = ConfigDict(extra="forbid")


# R25 §59：请求字段基数/长度上限（防资源放大）。
MAX_DATASET_NAME_LEN = 200
MAX_COLUMN_LEN = 128
MAX_COLUMN_COUNT = 200
MAX_INSTRUMENT_FILTER_CARDINALITY = 5000
MAX_FACTOR_IDS_CARDINALITY = 500
MAX_URI_LEN = 1024
MAX_TIME_RANGE_YEARS = 50


class ReadRequest(StrictRequestModel):
    """POST /v1/read 请求体。"""

    dataset: str
    columns: list[str] | None = None
    time_range: tuple[str | None, str | None] | None = None
    instrument_filter: list[str] | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    format: Literal["parquet", "arrow_ipc", "arrow_ipc_stream", "json"] = "parquet"
    max_rows: int | None = Field(
        default=None,
        ge=1,
        description="json 格式时的行数上限；未指定时由服务端默认预算决定",
    )

    @field_validator("time_range", mode="before")
    @classmethod
    def _normalize_time_range(cls, value: Any) -> tuple[str | None, str | None] | None:
        if value is None:
            return None
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError("time_range 必须是 [start, end] 二元组")
        start, end = value
        return (None if start is None else str(start), None if end is None else str(end))

    @field_validator("columns")
    @classmethod
    def _validate_columns(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        if not value or any(not isinstance(col, str) or not col.strip() for col in value):
            raise ValueError("columns 必须是非空字符串列表")
        if len(set(value)) != len(value):
            raise ValueError("columns 不允许重复")
        # R25 §59：列数上限（防宽表 + 防 JSON 内存放大）。
        if len(value) > MAX_COLUMN_COUNT:
            raise ValueError(f"columns 数量 {len(value)} 超过上限 {MAX_COLUMN_COUNT}")
        if any(len(col) > MAX_COLUMN_LEN for col in value):
            raise ValueError(f"列名长度超过上限 {MAX_COLUMN_LEN}")
        return value

    @field_validator("dataset")
    @classmethod
    def _validate_dataset(cls, value: str) -> str:
        # R25 §59：dataset 名长度上限。
        if len(str(value)) > MAX_DATASET_NAME_LEN:
            raise ValueError(f"dataset 名长度超过上限 {MAX_DATASET_NAME_LEN}")
        return value

    @field_validator("instrument_filter")
    @classmethod
    def _validate_instrument_filter(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        # R25 §59：instrument_filter 基数上限。
        if len(value) > MAX_INSTRUMENT_FILTER_CARDINALITY:
            raise ValueError(
                f"instrument_filter 数量 {len(value)} 超过上限 "
                f"{MAX_INSTRUMENT_FILTER_CARDINALITY}"
            )
        return value

    @field_validator("max_rows")
    @classmethod
    def _validate_max_rows(cls, value: int | None) -> int | None:
        if value is not None and value <= 0:
            raise ValueError("max_rows 必须是正整数")
        return value

    @model_validator(mode="after")
    def _validate_time_range(self) -> "ReadRequest":
        if self.time_range is not None:
            start, end = self.time_range
            if start is None and end is None:
                raise ValueError("time_range 不能同时为 [null, null]")
        return self


class DatasetInfo(StrictRequestModel):
    name: str
    kind: str
    access_mode: str
    # R26-P1-022：真实 registry 里 static 维表/特殊 dataset 可能没有 time/instrument
    # 列——nullable，否则 FastAPI response validation 对 None 报错。
    time_column: str | None = None
    instrument_column: str | None = None
    params_schema: dict[str, str] = Field(default_factory=dict)


class ReadResponseMeta(BaseModel):
    dataset: str
    snapshot_id: str
    rows: int
    bytes: int
    elapsed_ms: float
    format: str


class ReadURIRequest(StrictRequestModel):
    """POST /v1/read_uri 请求体（不必先登记数据集）。"""

    uri: str
    columns: list[str] | None = None
    time_range: tuple[str | None, str | None] | None = None
    instrument_filter: list[str] | None = None
    filters: dict[str, Any] | None = None
    format: str = "auto"
    time_column: str | None = None
    instrument_column: str | None = None
    limit: int | None = None
    format_out: Literal["parquet", "arrow_ipc", "json"] = "parquet"

    @field_validator("uri")
    @classmethod
    def _validate_uri(cls, value: str) -> str:
        # R25 §59：URI 长度上限。
        if len(str(value)) > MAX_URI_LEN:
            raise ValueError(f"URI 长度超过上限 {MAX_URI_LEN}")
        return value


class FactorReadRequest(StrictRequestModel):
    """POST /v1/factors/read 请求体（一次读多因子）。"""

    factor_ids: list[str]
    time_range: tuple[str | None, str | None] | None = None
    layout: Literal["long", "wide"] = "long"
    columns: list[str] | None = None
    universe: str | None = None
    frequency: str | None = None
    limit: int | None = None
    format: Literal["parquet", "arrow_ipc", "json"] = "parquet"

    @field_validator("factor_ids")
    @classmethod
    def _validate_factor_ids(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("factor_ids 不能为空")
        if len(value) > MAX_FACTOR_IDS_CARDINALITY:
            raise ValueError(
                f"factor_ids 数量 {len(value)} 超过上限 {MAX_FACTOR_IDS_CARDINALITY}"
            )
        return value
