# -*- coding: utf-8 -*-
"""HTTP 读数服务请求/响应模型。"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class ReadRequest(BaseModel):
    """POST /v1/read 请求体。"""

    dataset: str
    columns: list[str] | None = None
    time_range: tuple[str | None, str | None] | None = None
    instrument_filter: list[str] | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    format: Literal["parquet", "arrow_ipc", "json"] = "parquet"
    max_rows: int | None = Field(
        default=None,
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
