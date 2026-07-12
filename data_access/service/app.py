# -*- coding: utf-8 -*-
"""data_access 只读 HTTP 服务（FastAPI）。"""
from __future__ import annotations

import io
import json
import os
from typing import Any

import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq
from fastapi import Depends, FastAPI, Header, HTTPException, Response
from fastapi.responses import JSONResponse, StreamingResponse

from data_access import QueryBudget, get_store
from data_access.exceptions import DataAccessError, ValidationError
from data_access.query_budget import resolve_query_budget

from .models import DatasetInfo, ReadRequest, ReadResponseMeta

DEFAULT_API_MAX_ROWS = int(os.environ.get("DATA_ACCESS_API_MAX_ROWS", "500_000").replace("_", ""))
DEFAULT_API_MAX_BYTES = int(
    os.environ.get("DATA_ACCESS_API_MAX_BYTES", "512_000_000").replace("_", "")
)
API_KEY = os.environ.get("DATA_ACCESS_API_KEY", "").strip()


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes"}


def _require_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
    if not API_KEY:
        # 生产默认 fail-closed：避免无密钥误部署暴露全量读接口
        if _env_flag("QUANT_PRODUCTION_MODE") and not _env_flag("DATA_ACCESS_API_ALLOW_OPEN"):
            raise HTTPException(
                status_code=503,
                detail=(
                    "生产模式必须设置 DATA_ACCESS_API_KEY；"
                    "本地调试可设 DATA_ACCESS_API_ALLOW_OPEN=1"
                ),
            )
        return
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="无效或缺失 X-API-Key")


def _api_budget(request: ReadRequest) -> QueryBudget:
    base = resolve_query_budget()
    max_rows = request.max_rows if request.max_rows is not None else DEFAULT_API_MAX_ROWS
    tighter_rows = (
        min(base.max_rows, max_rows)
        if base.max_rows is not None
        else max_rows
    )
    tighter_bytes = (
        min(base.max_result_bytes, DEFAULT_API_MAX_BYTES)
        if base.max_result_bytes is not None
        else DEFAULT_API_MAX_BYTES
    )
    return QueryBudget(
        max_rows=tighter_rows,
        max_result_bytes=tighter_bytes,
        max_elapsed_ms=base.max_elapsed_ms,
        max_scan_files=base.max_scan_files,
        require_columns=True,
        require_time_range=base.require_time_range,
    )


def create_app() -> FastAPI:
    app = FastAPI(
        title="data_access read service",
        version="0.1.0",
        description="团队远程读数接口：服务端部署在有 COS/数据盘的服务器，客户端无需 clone 全仓库。",
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/datasets", dependencies=[Depends(_require_api_key)])
    def list_datasets() -> list[DatasetInfo]:
        store = get_store()
        out: list[DatasetInfo] = []
        for name in store.registry.names():
            ds = store.registry.get(name)
            params_schema = getattr(ds, "params_schema", None) or {}
            out.append(
                DatasetInfo(
                    name=name,
                    kind=getattr(ds, "kind", "static"),
                    access_mode=ds.access_mode,
                    time_column=ds.time_column,
                    instrument_column=ds.instrument_column,
                    params_schema=dict(params_schema),
                )
            )
        return out

    @app.get("/v1/datasets/{dataset_name}", dependencies=[Depends(_require_api_key)])
    def get_dataset(dataset_name: str) -> DatasetInfo:
        store = get_store()
        try:
            ds = store.registry.get(dataset_name)
        except ValidationError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        params_schema = getattr(ds, "params_schema", None) or {}
        return DatasetInfo(
            name=dataset_name,
            kind=getattr(ds, "kind", "static"),
            access_mode=ds.access_mode,
            time_column=ds.time_column,
            instrument_column=ds.instrument_column,
            params_schema=dict(params_schema),
        )

    @app.post("/v1/read", dependencies=[Depends(_require_api_key)])
    def read_dataset(request: ReadRequest) -> Response:
        store = get_store()
        budget = _api_budget(request)
        try:
            result = store.read_result(
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
            if table.num_rows > (budget.max_rows or DEFAULT_API_MAX_ROWS):
                raise HTTPException(
                    status_code=413,
                    detail=f"结果行数 {table.num_rows} 超过 API 上限 {budget.max_rows}",
                )
            df = table.to_pandas(self_destruct=True)
            payload: dict[str, Any] = {
                "meta": meta.model_dump(),
                "columns": table.column_names,
                "data": json.loads(
                    df.to_json(orient="records", date_format="iso", default_handler=str)
                ),
            }
            return JSONResponse(content=payload, headers=headers)

        if request.format == "arrow_ipc":
            buf = io.BytesIO()
            with ipc.new_file(buf, table.schema) as writer:
                writer.write_table(table)
            buf.seek(0)
            headers["Content-Type"] = "application/vnd.apache.arrow.stream"
            return StreamingResponse(buf, media_type=headers["Content-Type"], headers=headers)

        buf = io.BytesIO()
        pq.write_table(table, buf)
        buf.seek(0)
        return StreamingResponse(
            buf,
            media_type="application/vnd.apache.parquet",
            headers=headers,
        )

    return app


app = create_app()
