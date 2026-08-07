# -*- coding: utf-8 -*-
"""data_access read-only HTTP service."""
from __future__ import annotations

import hmac
import io
import json
import os
import threading
import time
import uuid
from typing import Any

import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq
from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

import data_access
from data_access import QueryBudget, get_store
from data_access.core.exceptions import DataAccessError, ValidationError
from data_access.read.query_budget import resolve_query_budget

from .config import ServiceSettings
from .models import (
    DatasetInfo,
    FactorReadRequest,
    ReadRequest,
    ReadResponseMeta,
    ReadURIRequest,
)

API_KEY = ""


def _api_budget(request: ReadRequest, settings: ServiceSettings) -> QueryBudget:
    base = resolve_query_budget()
    requested_rows = request.max_rows if request.max_rows is not None else settings.max_rows
    max_rows = min(base.max_rows, requested_rows) if base.max_rows is not None else requested_rows
    max_bytes = min(base.max_result_bytes, settings.max_bytes) if base.max_result_bytes is not None else settings.max_bytes
    return QueryBudget(
        max_rows=max_rows,
        max_result_bytes=max_bytes,
        max_elapsed_ms=base.max_elapsed_ms,
        max_scan_files=base.max_scan_files,
        require_columns=True,
        require_time_range=base.require_time_range,
    )


def create_app(settings: ServiceSettings | None = None) -> FastAPI:
    settings = settings or ServiceSettings.from_env()
    app = FastAPI(
        title="data_access read service",
        version=data_access.__version__,
        description="只读量化数据访问服务。",
    )
    query_slots = threading.BoundedSemaphore(settings.max_concurrency)

    def require_api_key(
        x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    ) -> None:
        key = API_KEY or settings.api_key
        production = settings.production_mode or os.environ.get("QUANT_PRODUCTION_MODE", "").lower() in {"1", "true", "yes"}
        allow_open = settings.allow_open or os.environ.get("DATA_ACCESS_API_ALLOW_OPEN", "").lower() in {"1", "true", "yes"}
        if key:
            if not x_api_key or not hmac.compare_digest(x_api_key, key):
                raise HTTPException(status_code=401, detail="无效或缺失 X-API-Key")
        elif production and not allow_open:
            raise HTTPException(status_code=503, detail="DATA_ACCESS_API_KEY 未配置")

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request_id = request.headers.get(settings.request_id_header) or uuid.uuid4().hex
        response = await call_next(request)
        response.headers[settings.request_id_header] = request_id
        response.headers["X-Data-Access-Version"] = data_access.__version__
        return response

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": data_access.__version__}

    @app.get("/ready")
    def ready() -> dict[str, str]:
        try:
            count = len(get_store().registry.names())
        except Exception as exc:
            raise HTTPException(status_code=503, detail="数据访问服务未就绪") from exc
        return {"status": "ready", "datasets": str(count)}

    @app.get("/version")
    def version() -> dict[str, str]:
        return {"package": "data-access", "version": data_access.__version__, "api": "v1"}

    @app.get("/v1/metrics", dependencies=[Depends(require_api_key)])
    def metrics() -> dict[str, Any]:
        from data_access.read.telemetry import get_counters_snapshot

        return {
            "operators": {
                name: {
                    "total_queries": counter.total_queries,
                    "slow_queries": counter.slow_queries,
                    "total_elapsed_ms": counter.total_elapsed_ms,
                }
                for name, counter in get_counters_snapshot().items()
            }
        }

    def list_datasets() -> list[DatasetInfo]:
        store = get_store()
        result = []
        for name in store.registry.names():
            ds = store.registry.get(name)
            result.append(DatasetInfo(
                name=name,
                kind=getattr(ds, "kind", "static"),
                access_mode=ds.access_mode,
                time_column=ds.time_column,
                instrument_column=ds.instrument_column,
                params_schema=dict(getattr(ds, "params_schema", None) or {}),
            ))
        return result

    @app.get("/v1/datasets", dependencies=[Depends(require_api_key)])
    def datasets_endpoint() -> list[DatasetInfo]:
        return list_datasets()


    @app.get("/v1/datasets/{dataset_name}", dependencies=[Depends(require_api_key)])
    def get_dataset(dataset_name: str) -> DatasetInfo:
        try:
            ds = get_store().registry.get(dataset_name)
        except ValidationError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return DatasetInfo(
            name=dataset_name,
            kind=getattr(ds, "kind", "static"),
            access_mode=ds.access_mode,
            time_column=ds.time_column,
            instrument_column=ds.instrument_column,
            params_schema=dict(getattr(ds, "params_schema", None) or {}),
        )

    @app.post("/v1/read/arrow-stream", dependencies=[Depends(require_api_key)])
    def read_arrow_stream(request: ReadRequest) -> Response:
        if not query_slots.acquire(blocking=False):
            raise HTTPException(status_code=503, detail="查询并发已达上限，请稍后重试")
        try:
            store = get_store()
            budget = _api_budget(request, settings)
            try:
                batches = store.read_arrow_stream(
                    request.dataset,
                    columns=request.columns,
                    time_range=request.time_range,
                    instrument_filter=request.instrument_filter,
                    query_budget=budget,
                    **request.params,
                )
            except ValidationError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except DataAccessError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            try:
                first = next(batches)
            except StopIteration:
                query_slots.release()
                return Response(status_code=204)

            def body():
                try:
                    sink = io.BytesIO()
                    with ipc.new_stream(sink, first.schema) as writer:
                        writer.write_batch(first)
                        yield sink.getvalue()
                        sink.seek(0)
                        sink.truncate(0)
                        for batch in batches:
                            writer.write_batch(batch)
                            yield sink.getvalue()
                            sink.seek(0)
                            sink.truncate(0)
                finally:
                    close = getattr(batches, "close", None)
                    if callable(close):
                        close()
                    query_slots.release()

            return StreamingResponse(
                body(),
                media_type="application/vnd.apache.arrow.stream",
                headers={"X-Query-Mode": "stream", "X-Arrow-Stream-Version": "1"},
            )
        except Exception:
            query_slots.release()
            raise

    @app.post("/v1/read", dependencies=[Depends(require_api_key)])
    def read_dataset(request: ReadRequest) -> Response:
        if not query_slots.acquire(blocking=False):
            raise HTTPException(status_code=503, detail="查询并发已达上限，请稍后重试")
        try:
            return _read_dataset(request, settings)
        finally:
            query_slots.release()

    def _read_dataset(request: ReadRequest, settings: ServiceSettings) -> Response:
        budget = _api_budget(request, settings)
        try:
            result = get_store().read_result(
                request.dataset,
                columns=request.columns,
                time_range=request.time_range,
                instrument_filter=request.instrument_filter,
                query_budget=budget,
                **request.params,
            )
        except ValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except DataAccessError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        table = result.table
        meta = ReadResponseMeta(
            dataset=request.dataset,
            snapshot_id=result.snapshot.snapshot_id,
            rows=result.stats.rows,
            bytes=result.stats.bytes,
            elapsed_ms=result.stats.elapsed_ms,
            format=request.format,
        )
        headers = {
            "X-Data-Snapshot-Id": result.snapshot.snapshot_id,
            "X-Rows": str(result.stats.rows),
            "X-Bytes": str(result.stats.bytes),
            "X-Elapsed-Ms": f"{result.stats.elapsed_ms:.2f}",
        }
        if request.format == "json":
            if table.num_rows > (budget.max_rows or settings.max_rows):
                raise HTTPException(status_code=413, detail="结果超过服务端行数上限")
            columns = list(table.column_names)
            df = table.to_pandas(self_destruct=False)
            payload: dict[str, Any] = {
                "meta": meta.model_dump(),
                "columns": columns,
                "data": json.loads(df.to_json(orient="records", date_format="iso", default_handler=str)),
            }
            return JSONResponse(content=payload, headers=headers)
        if request.format == "arrow_ipc":
            buf = io.BytesIO()
            with ipc.new_stream(buf, table.schema) as writer:
                writer.write_table(table)
            buf.seek(0)
            return StreamingResponse(buf, media_type="application/vnd.apache.arrow.stream", headers=headers)
        buf = io.BytesIO()
        pq.write_table(table, buf)
        buf.seek(0)
        return StreamingResponse(buf, media_type="application/vnd.apache.parquet", headers=headers)

    def _serialize(table: pa.Table, meta: ReadResponseMeta, fmt: str) -> Response:
        headers = {
            "X-Data-Snapshot-Id": meta.snapshot_id,
            "X-Rows": str(meta.rows),
            "X-Bytes": str(meta.bytes),
            "X-Elapsed-Ms": f"{meta.elapsed_ms:.2f}",
        }
        if fmt == "json":
            payload: dict[str, Any] = {
                "meta": meta.model_dump(),
                "columns": list(table.column_names),
                "data": json.loads(
                    table.to_pandas(self_destruct=False).to_json(
                        orient="records", date_format="iso", default_handler=str
                    )
                ),
            }
            return JSONResponse(content=payload, headers=headers)
        if fmt == "arrow_ipc":
            buf = io.BytesIO()
            with ipc.new_stream(buf, table.schema) as writer:
                writer.write_table(table)
            buf.seek(0)
            return StreamingResponse(
                buf, media_type="application/vnd.apache.arrow.stream", headers=headers
            )
        buf = io.BytesIO()
        pq.write_table(table, buf)
        buf.seek(0)
        return StreamingResponse(
            buf, media_type="application/vnd.apache.parquet", headers=headers
        )

    @app.post("/v1/read_uri", dependencies=[Depends(require_api_key)])
    def read_uri(request: ReadURIRequest) -> Response:
        """读取任意 URI（dev 白名单下），不必先登记数据集。"""
        if not query_slots.acquire(blocking=False):
            raise HTTPException(status_code=503, detail="查询并发已达上限，请稍后重试")
        try:
            budget = _api_budget(ReadRequest(dataset="uri"), settings)
            start = time.perf_counter()
            try:
                handle = get_store().read_uri(
                    request.uri,
                    columns=request.columns,
                    time_range=request.time_range,
                    instrument_filter=request.instrument_filter,
                    filters=request.filters,
                    limit=request.limit,
                    format=request.format,
                    time_column=request.time_column,
                    instrument_column=request.instrument_column,
                    query_budget=budget,
                )
            except ValidationError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except DataAccessError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            table = handle.to_arrow()
            elapsed_ms = (time.perf_counter() - start) * 1000
            meta = ReadResponseMeta(
                dataset=f"<uri:{request.uri[:80]}>",
                snapshot_id=getattr(handle.snapshot, "snapshot_id", "-"),
                rows=table.num_rows,
                bytes=table.nbytes,
                elapsed_ms=elapsed_ms,
                format=request.format_out,
            )
            return _serialize(table, meta, request.format_out)
        finally:
            query_slots.release()

    @app.get("/v1/factors", dependencies=[Depends(require_api_key)])
    def factors_catalog() -> dict[str, Any]:
        """因子目录清单（FactorCatalog）。"""
        catalog = get_store().get_factor_catalog()
        return {
            "root": str(catalog.root),
            "count": len(catalog),
            "factors": [m.to_dict() for m in catalog.records.values()],
        }

    @app.post("/v1/factors/read", dependencies=[Depends(require_api_key)])
    def factors_read(request: FactorReadRequest) -> Response:
        """一次读多个因子（单查询 UNION ALL / 宽表 PIVOT）。"""
        if not query_slots.acquire(blocking=False):
            raise HTTPException(status_code=503, detail="查询并发已达上限，请稍后重试")
        try:
            start = time.perf_counter()
            try:
                handle = get_store().read_factors(
                    request.factor_ids,
                    time_range=request.time_range,
                    universe=request.universe,
                    frequency=request.frequency,
                    layout=request.layout,
                    columns=request.columns,
                    limit=request.limit,
                )
            except ValidationError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except DataAccessError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            table = handle.to_arrow()
            elapsed_ms = (time.perf_counter() - start) * 1000
            meta = ReadResponseMeta(
                dataset="factors:" + ",".join(request.factor_ids),
                snapshot_id=getattr(handle.snapshot, "snapshot_id", "-"),
                rows=table.num_rows,
                bytes=table.nbytes,
                elapsed_ms=elapsed_ms,
                format=request.format,
            )
            return _serialize(table, meta, request.format)
        finally:
            query_slots.release()

    return app


app = create_app()
