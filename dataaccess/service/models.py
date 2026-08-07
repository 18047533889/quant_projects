# -*- coding: utf-8 -*-
"""HTTP 读数服务请求/响应模型。"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class ReadRequest(BaseModel):
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


class DatasetInfo(BaseModel):
    name: str
    kind: str
    access_mode: str
    time_column: str
    instrument_column: str
    params_schema: dict[str, str] = Field(default_factory=dict)


class ReadResponseMeta(BaseModel):
    dataset: str
    snapshot_id: str
    rows: int
    bytes: int
    elapsed_ms: float
    format: str


class ReadURIRequest(BaseModel):
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


class FactorReadRequest(BaseModel):
    """POST /v1/factors/read 请求体（一次读多因子）。"""

    factor_ids: list[str]
    time_range: tuple[str | None, str | None] | None = None
    layout: Literal["long", "wide"] = "long"
    columns: list[str] | None = None
    universe: str | None = None
    frequency: str | None = None
    limit: int | None = None
    format: Literal["parquet", "arrow_ipc", "json"] = "parquet"
