# -*- coding: utf-8 -*-
"""轻量 HTTP 客户端：团队机器无需 clone 全仓库，只需本文件 + httpx/pyarrow/pandas。"""
from __future__ import annotations

import io
from typing import Any, Literal, Sequence

import httpx
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc

from .models import DatasetInfo, ReadResponseMeta


class DataAccessClient:
    """连接远程 data_access 读数服务。"""

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        timeout: float = 300.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        return headers

    def health(self) -> dict[str, str]:
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.get(f"{self.base_url}/health")
            resp.raise_for_status()
            return resp.json()

    def list_datasets(self) -> list[DatasetInfo]:
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.get(f"{self.base_url}/v1/datasets", headers=self._headers())
            resp.raise_for_status()
            return [DatasetInfo.model_validate(item) for item in resp.json()]

    def get_dataset(self, name: str) -> DatasetInfo:
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.get(
                f"{self.base_url}/v1/datasets/{name}",
                headers=self._headers(),
            )
            resp.raise_for_status()
            return DatasetInfo.model_validate(resp.json())

    def read_arrow(
        self,
        dataset: str,
        *,
        columns: Sequence[str] | None = None,
        time_range: tuple[Any, Any] | None = None,
        instrument_filter: Sequence[str] | None = None,
        params: dict[str, Any] | None = None,
        format: Literal["parquet", "arrow_ipc", "json"] = "arrow_ipc",
        max_rows: int | None = None,
    ) -> tuple[pa.Table, ReadResponseMeta]:
        body = self._build_body(
            dataset=dataset,
            columns=columns,
            time_range=time_range,
            instrument_filter=instrument_filter,
            params=params,
            format=format,
            max_rows=max_rows,
        )
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(
                f"{self.base_url}/v1/read",
                json=body,
                headers=self._headers(),
            )
            resp.raise_for_status()
            meta = self._meta_from_headers(resp, dataset=dataset, format=format)
            if format == "json":
                payload = resp.json()
                meta = ReadResponseMeta.model_validate(payload["meta"])
                records = payload["data"]
                return pa.Table.from_pylist(records), meta
            if format == "arrow_ipc":
                reader = ipc.open_stream(resp.content)
                return reader.read_all(), meta
            return pa.parquet.read_table(io.BytesIO(resp.content)), meta

    def read_frame(
        self,
        dataset: str,
        *,
        columns: Sequence[str] | None = None,
        time_range: tuple[Any, Any] | None = None,
        instrument_filter: Sequence[str] | None = None,
        params: dict[str, Any] | None = None,
        format: Literal["parquet", "arrow_ipc", "json"] = "parquet",
        max_rows: int | None = None,
    ) -> pd.DataFrame:
        table, _meta = self.read_arrow(
            dataset,
            columns=columns,
            time_range=time_range,
            instrument_filter=instrument_filter,
            params=params,
            format=format,
            max_rows=max_rows,
        )
        return table.to_pandas(self_destruct=True)

    @staticmethod
    def _build_body(
        *,
        dataset: str,
        columns: Sequence[str] | None,
        time_range: tuple[Any, Any] | None,
        instrument_filter: Sequence[str] | None,
        params: dict[str, Any] | None,
        format: str,
        max_rows: int | None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "dataset": dataset,
            "params": params or {},
            "format": format,
        }
        if columns is not None:
            body["columns"] = list(columns)
        if time_range is not None:
            body["time_range"] = [time_range[0], time_range[1]]
        if instrument_filter is not None:
            body["instrument_filter"] = list(instrument_filter)
        if max_rows is not None:
            body["max_rows"] = max_rows
        return body

    @staticmethod
    def _meta_from_headers(
        resp: httpx.Response,
        *,
        dataset: str,
        format: str,
    ) -> ReadResponseMeta:
        return ReadResponseMeta(
            dataset=dataset,
            snapshot_id=resp.headers.get("X-Data-Snapshot-Id", ""),
            rows=int(resp.headers.get("X-Rows", "0")),
            bytes=int(resp.headers.get("X-Bytes", "0")),
            elapsed_ms=float(resp.headers.get("X-Elapsed-Ms", "0")),
            format=format,
        )
