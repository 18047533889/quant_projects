# -*- coding: utf-8 -*-
"""Lightweight HTTP client for the remote read service."""
from __future__ import annotations

import io
from typing import Any, Literal, Sequence

import httpx
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as parquet

from .models import DatasetInfo, ReadResponseMeta


class DataAccessClient:
    """Connection-pooled client for the data_access read service."""

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        timeout: float = 300.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self._client = client or httpx.Client(timeout=timeout)
        self._owns_client = client is None

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "DataAccessClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _headers(self) -> dict[str, str]:
        return {"X-API-Key": self.api_key} if self.api_key else {}

    def health(self) -> dict[str, str]:
        resp = self._client.get(f"{self.base_url}/health")
        resp.raise_for_status()
        return resp.json()

    def list_datasets(self) -> list[DatasetInfo]:
        resp = self._client.get(f"{self.base_url}/v1/datasets", headers=self._headers())
        resp.raise_for_status()
        return [DatasetInfo.model_validate(item) for item in resp.json()]

    def get_dataset(self, name: str) -> DatasetInfo:
        resp = self._client.get(
            f"{self.base_url}/v1/datasets/{name}",
            headers=self._headers(),
        )
        resp.raise_for_status()
        return DatasetInfo.model_validate(resp.json())

    def iter_arrow_batches(
        self,
        dataset: str,
        *,
        columns: Sequence[str] | None = None,
        time_range: tuple[Any, Any] | None = None,
        instrument_filter: Sequence[str] | None = None,
        params: dict[str, Any] | None = None,
        max_rows: int | None = None,
    ):
        body = self._build_body(
            dataset=dataset,
            columns=columns,
            time_range=time_range,
            instrument_filter=instrument_filter,
            params=params,
            format="arrow_ipc_stream",
            max_rows=max_rows,
        )

        def batches():
            with self._client.stream(
                "POST",
                f"{self.base_url}/v1/read/arrow-stream",
                json=body,
                headers=self._headers(),
            ) as resp:
                resp.raise_for_status()
                reader = ipc.open_stream(resp.iter_bytes())
                yield from reader

        return batches()

    read_arrow_stream = iter_arrow_batches

    def read_arrow(
        self,
        dataset: str,
        *,
        columns: Sequence[str] | None = None,
        time_range: tuple[Any, Any] | None = None,
        instrument_filter: Sequence[str] | None = None,
        params: dict[str, Any] | None = None,
        format: Literal["parquet", "arrow_ipc", "arrow_ipc_stream", "json"] = "arrow_ipc",
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
        resp = self._client.post(
            f"{self.base_url}/v1/read",
            json=body,
            headers=self._headers(),
        )
        resp.raise_for_status()
        meta = self._meta_from_headers(resp, dataset=dataset, format=format)
        if format == "json":
            payload = resp.json()
            meta = ReadResponseMeta.model_validate(payload["meta"])
            return pa.Table.from_pylist(payload["data"]), meta
        if format == "arrow_ipc":
            return ipc.open_stream(resp.content).read_all(), meta
        return parquet.read_table(io.BytesIO(resp.content)), meta

    def read_frame(self, dataset: str, **kwargs: Any) -> pd.DataFrame:
        table, _meta = self.read_arrow(dataset, **kwargs)
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
        body: dict[str, Any] = {"dataset": dataset, "params": params or {}, "format": format}
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
