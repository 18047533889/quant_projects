# -*- coding: utf-8 -*-
"""data_access read-only HTTP service（R24 P0-S5 §7 / P1-S6 §8 / P1-S8 §24）。

安全模型：
    - API key → principal：``X-API-Key`` hash → ``principal_id`` → roles →
      dataset scopes → factor scopes（§7），不再「一把 API Key = 整台服务器权限」。
    - production 恒要求 auth（T-S12）：``DATA_ACCESS_API_ALLOW_OPEN=1`` 在
      production 下**不绕过**认证（§7 production 不允许 open mode）。
    - ``/v1/datasets``：只返回 principal 可见数据集（T-S09），restricted dataset
      名本身也是 metadata，不全部暴露。
    - ``/v1/read``：先 authorization 再调用 backend（T-S10）。
    - ``/v1/read_uri``：production 默认 DISABLED；需要显式 ``uri:read``
      permission（§7 / §24），不能仅因 URI 落在全局 registered prefix 就允许。
    - ``/v1/factors``：普通 ``factor:list`` 只返回脱敏摘要；敏感字段（expression /
      source config / local root / lineage / snapshot / internal metadata）另需
      ``factor:metadata_sensitive``。
    - 外部错误统一脱敏（P1-S6 §8）：不泄露 allowed prefixes / 完整 local path /
      COS prefix；内部安全日志写 request_id / principal_id / dataset_id /
      policy_decision。
"""
from __future__ import annotations

import hmac
import io
import json
import os
import threading
import time
import uuid
from dataclasses import dataclass
from contextvars import ContextVar
from typing import Any

# R26-P1-011：request_id 单一来源（middleware 写入，dependency 复用）。
_request_id_var: ContextVar[str] = ContextVar("data_access_http_request_id", default="")

import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq
from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

import data_access
from data_access import QueryBudget, get_store
from data_access.core.exceptions import (
    AccessDeniedError,
    AuthorizationError,
    DataAccessError,
    ValidationError,
)
from data_access.read.query_budget import resolve_query_budget
from data_access.security.api_principals import (
    get_api_principal_registry,
    hash_api_key,
)
from data_access.security.policy import DefaultAuthorizer
from data_access.security.principal import (
    ACTION_URI_READ,
    DEFAULT_LOCAL_PRINCIPAL,
    AccessPolicy,
    DataPrincipal,
)

from .config import ServiceSettings
from .models import (
    DatasetInfo,
    FactorReadRequest,
    ReadRequest,
    ReadResponseMeta,
    ReadURIRequest,
)

API_KEY = ""

# dev/research open 模式的默认动作集：除 uri:read / factor:metadata_sensitive 外全部。
_DEFAULT_API_ACTIONS = frozenset(
    a
    for a in (
        "dataset:list",
        "dataset:read",
        "factor:list",
        "factor:read",
        "metadata:read",
    )
)


@dataclass(frozen=True)
class _ApiCallContext:
    """一次 HTTP 调用的身份 + 授权器。"""

    principal: DataPrincipal
    authorizer: DefaultAuthorizer
    production: bool
    request_id: str

    def authorize(self, dataset: str, action: str = "dataset:read") -> None:
        self.authorizer.authorize(self.principal, dataset, action=action)


def _execution_context_for(ctx: _ApiCallContext):
    """R26-P0-005：把 HTTP 请求身份放进 request-scoped 执行上下文。

    Store 的所有嵌套读（calendar / universe / factor catalog / dependent
    datasets / credential / cache scope）通过 ContextVar 自动继承请求 principal，
    绝不改全局 authorizer。并发 API key A/B 不会身份串扰。
    """
    from data_access.security.execution_context import (
        DataAccessExecutionContext,
    )
    from data_access.security.run_mode import RunMode

    return DataAccessExecutionContext(
        principal=ctx.principal,
        authorizer=ctx.authorizer,
        access_policy=getattr(ctx.authorizer, "policy", None),
        request_id=ctx.request_id,
        run_mode=RunMode("production") if ctx.production else RunMode("interactive_research"),
    )


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
        x_request_id: str | None = Header(default=None, alias=settings.request_id_header),
    ) -> _ApiCallContext:
        key = API_KEY or settings.api_key
        production = settings.production_mode or os.environ.get("QUANT_PRODUCTION_MODE", "").lower() in {"1", "true", "yes"}
        # R26-P1-011：优先复用 middleware 生成的 request_id ContextVar（单一来源）。
        request_id = x_request_id or _request_id_var.get() or uuid.uuid4().hex

        # R24 P0-S5 §7 / T-S12：production 恒要求认证。``DATA_ACCESS_API_ALLOW_OPEN=1``
        # 在 production 下**不绕过**（open mode 只允许 dev/research）。
        if production:
            if not x_api_key:
                if not settings.api_key and not os.environ.get("DATA_ACCESS_API_PRINCIPALS"):
                    # 未配置任何 key/principal 映射 = 服务未就绪（部署配置缺失）。
                    raise HTTPException(status_code=503, detail="DATA_ACCESS_API_KEY 未配置")
                raise HTTPException(status_code=401, detail="无效或缺失 X-API-Key")
            registry = get_api_principal_registry()
            principal, policy = registry.resolve(x_api_key)
            if principal is None or policy is None:
                raise HTTPException(
                    status_code=403, detail="api key 未映射到任何 principal"
                )
            return _ApiCallContext(
                principal=principal,
                authorizer=DefaultAuthorizer(policy=policy, principal=principal),
                production=True,
                request_id=request_id,
            )

        # dev / research：可配置 api key 或 open mode。
        if key:
            if not x_api_key:
                raise HTTPException(status_code=401, detail="无效或缺失 X-API-Key")
            registry = get_api_principal_registry()
            principal, policy = registry.resolve(x_api_key)
            if principal is None or policy is None:
                # 未命中 principal registry：必须是共享 api_key（默认 principal）。
                if not hmac.compare_digest(x_api_key, key):
                    raise HTTPException(status_code=401, detail="无效或缺失 X-API-Key")
                principal, policy = DEFAULT_LOCAL_PRINCIPAL, AccessPolicy(
                    allowed_actions=frozenset(_DEFAULT_API_ACTIONS)
                )
            return _ApiCallContext(
                principal=principal,
                authorizer=DefaultAuthorizer(policy=policy, principal=principal),
                production=False,
                request_id=request_id,
            )
        # open mode（仅 dev/research）：默认 principal，uri:read 仍拒绝。
        policy = AccessPolicy(allowed_actions=frozenset(_DEFAULT_API_ACTIONS))
        return _ApiCallContext(
            principal=DEFAULT_LOCAL_PRINCIPAL,
            authorizer=DefaultAuthorizer(policy=policy, principal=DEFAULT_LOCAL_PRINCIPAL),
            production=False,
            request_id=request_id,
        )

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        # R26-P1-011：request_id 唯一生成点——middleware 生成并写入 ContextVar，
        # require_api_key 直接复用，不再二次生成（否则 response id 与 security
        # context id 不一致）。
        request_id = request.headers.get(settings.request_id_header) or uuid.uuid4().hex
        token = _request_id_var.set(request_id)
        try:
            response = await call_next(request)
        finally:
            _request_id_var.reset(token)
        response.headers[settings.request_id_header] = request_id
        response.headers["X-Data-Access-Version"] = data_access.__version__
        return response

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": data_access.__version__}

    @app.get("/ready")
    def ready() -> dict[str, str]:
        """R25 §62 + R26-P0-021：真正 readiness（复用 startup gate health state）。

        R26-P0-021 修正：
            - engine 探测用真实 public API（``execute_arrow``，不是不存在的
              ``engine.execute()``）；
            - credential resolve 失败 / legacy root in_use（strict）**必须** 503，
              不能仍返回 status=ready；
            - 直接复用 ``run_startup_gate`` 的 health state，不自己另写一套逻辑。
        """
        from data_access.runtime.startup_gate import run_startup_gate

        store = get_store()
        try:
            result = run_startup_gate(store)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        if not result.passed:
            raise HTTPException(
                status_code=503, detail="数据访问服务未就绪：" + "; ".join(result.problems)
            )
        # engine 真实 public API 探测（SELECT 1）。
        try:
            store._engine.execute_arrow("SELECT 1", [], deadline_ms=5_000)
        except Exception as exc:
            raise HTTPException(status_code=503, detail="engine 探测失败") from exc
        return {
            "status": "ready",
            "datasets": str(len(store.registry.names())),
            "startup_gate": "ok",
            "engine": "ok",
        }

    @app.get("/version")
    def version() -> dict[str, str]:
        # R29-P0 #207：build SHA 进 /version——两台机器版本号相同但代码不同时，
        # build_sha 一眼可辨（可复现）。
        build = getattr(data_access, "__build_sha__", None)
        return {
            "package": "data-access",
            "version": data_access.__version__,
            "build_sha": build,
            "api": "v1",
        }

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

    def list_datasets(ctx: _ApiCallContext) -> list[DatasetInfo]:
        store = get_store()
        result = []
        # R24 P0-S5 §7 / T-S09：restricted dataset 名本身也是 metadata，不全部暴露。
        # 用 dataset:read 做可见性过滤——principal 只能列出自己可读的数据集。
        for name in store.registry.names():
            try:
                ctx.authorize(name, action="dataset:read")
            except AuthorizationError:
                continue
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
    def datasets_endpoint(ctx: _ApiCallContext = Depends(require_api_key)) -> list[DatasetInfo]:
        return list_datasets(ctx)

    @app.get("/v1/datasets/{dataset_name}", dependencies=[Depends(require_api_key)])
    def get_dataset(dataset_name: str, ctx: _ApiCallContext = Depends(require_api_key)) -> DatasetInfo:
        # R24 P0-S5：访问单个数据集也要先授权（metadata:read）。
        ctx.authorize(dataset_name, action="metadata:read")
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
    def read_arrow_stream(request: ReadRequest, ctx: _ApiCallContext = Depends(require_api_key)) -> Response:
        # R32 P0-112：query slot exactly-once release 保证（先预分配，
        # 所有路径——first/exception/body-finally/early exception——都恰好释放一次）。
        slot_held = False
        try:
            ctx.authorize(request.dataset, action="dataset:read")
        except AuthorizationError:
            raise HTTPException(status_code=403, detail="resource is not authorized")
        if not query_slots.acquire(blocking=False):
            raise HTTPException(status_code=503, detail="查询并发已达上限，请稍后重试")
        slot_held = True
        try:
            store = get_store()
            budget = _api_budget(request, settings)
            # R28-16：request-scoped 执行上下文必须覆盖**整个 stream 生命周期**
            # （含真正执行的那部分）。旧实现 ``with execution_scope(...)`` 只包住
            # ``store.read_arrow_stream(...)`` 调用——生成器是惰性的，prepare_read
            # 在调用时执行，但 verify/execute/流式读在 ``next()`` 时才跑，那时
            # scope 早已退出，嵌套读（calendar/universe/credential/cache scope）
            # 全部回退到 process 级 principal → 身份串扰。这里：
            #   - 创建生成器（prepare_read 立即执行）时在 scope 内；
            #   - ``next()`` 与 body() 流式消费都包在 ``execution_scope`` 内。
            exec_ctx = _execution_context_for(ctx)
            from data_access.security.execution_context import execution_scope

            try:
                with execution_scope(exec_ctx):
                    batches = store.read_arrow_stream(
                        request.dataset,
                        columns=request.columns,
                        time_range=request.time_range,
                        instrument_filter=request.instrument_filter,
                        query_budget=budget,
                        **request.params,
                    )
            except AccessDeniedError:
                raise HTTPException(status_code=403, detail="resource is not authorized")
            except ValidationError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except DataAccessError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc

            # R32 P0-115：HTTP buffer limit 防 OOM（StreamingResponse
            # 默认无界缓冲，单 batch > 可用内存则 OOM）。sink 每次 write
            # 清空（sink.truncate(0)），不累积跨 batch。
            _MAX_BUFFER_BYTES = 128 * 1024 * 1024  # 128 MiB

            # R28-16：创建生成器（prepare_read 立即执行）在 request-scoped 执行上下文内。
            with execution_scope(exec_ctx):
                # 不能用包装生成器（会导致 "generator already executing"）——
                # 直接用原始 batches，在 first/body 时消费。
                it = batches
            try:
                first = next(it)
            except StopIteration:
                query_slots.release()
                slot_held = False
                return Response(status_code=204)
            except AccessDeniedError:
                raise HTTPException(status_code=403, detail="resource is not authorized")
            except ValidationError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except DataAccessError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc

            # R32 P0-113：disconnect_cleanup：client 断开时（generator
            # 未耗尽就退出 body()）回调确保 batches.close() +
            # query_slots.release() 只跑一次。
            cleanup_done = False

            def _disconnect_cleanup():
                nonlocal cleanup_done, slot_held
                if cleanup_done:
                    return
                cleanup_done = True
                close = getattr(batches, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception:
                        pass
                if slot_held:
                    query_slots.release()
                    slot_held = False

            def body():
                nonlocal cleanup_done, slot_held
                try:
                    sink = io.BytesIO()
                    with ipc.new_stream(sink, first.schema) as writer:
                        writer.write_batch(first)
                        chunk = sink.getvalue()
                        # R32 P0-115：单 batch 超限 fail-closed（不静默截断）。
                        if len(chunk) > _MAX_BUFFER_BYTES:
                            raise HTTPException(
                                status_code=413,
                                detail=f"单批次超 {_MAX_BUFFER_BYTES} 字节（{len(chunk)} bytes），"
                                "需缩小 batch_size 或增大 buffer limit"
                            )
                        yield chunk
                        sink.seek(0)
                        sink.truncate(0)
                        for batch in it:
                            writer.write_batch(batch)
                            chunk = sink.getvalue()
                            if len(chunk) > _MAX_BUFFER_BYTES:
                                raise HTTPException(
                                    status_code=413,
                                    detail=f"单批次超 {_MAX_BUFFER_BYTES} 字节"
                                )
                            yield chunk
                            sink.seek(0)
                            sink.truncate(0)
                finally:
                    _disconnect_cleanup()

            # R32 P0-114：StreamingResponse background callback 注册
            # disconnect 清理（client 断开时 FastAPI 调 background task）。
            from fastapi import BackgroundTasks
            bg_tasks = BackgroundTasks()
            bg_tasks.add_task(_disconnect_cleanup)

            return StreamingResponse(
                body(),
                media_type="application/vnd.apache.arrow.stream",
                headers={"X-Query-Mode": "stream", "X-Arrow-Stream-Version": "1"},
                background=bg_tasks,
            )
        except Exception:
            if slot_held:
                query_slots.release()
                slot_held = False
            raise

    @app.post("/v1/read", dependencies=[Depends(require_api_key)])
    def read_dataset(request: ReadRequest, ctx: _ApiCallContext = Depends(require_api_key)) -> Response:
        # R24 P0-S5 / T-S10：先 authorization，再调用 backend。
        # R32 P0-112：query slot exactly-once release 保证。
        slot_held = False
        try:
            ctx.authorize(request.dataset, action="dataset:read")
        except AuthorizationError:
            raise HTTPException(status_code=403, detail="resource is not authorized")
        if not query_slots.acquire(blocking=False):
            raise HTTPException(status_code=503, detail="查询并发已达上限，请稍后重试")
        slot_held = True
        try:
            # R26-P0-005：request-scoped 执行上下文（嵌套读继承请求 principal）。
            from data_access.security.execution_context import execution_scope

            with execution_scope(_execution_context_for(ctx)):
                return _read_dataset(request, settings)
        finally:
            if slot_held:
                query_slots.release()
                slot_held = False

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
        except AccessDeniedError:
            raise HTTPException(status_code=403, detail="resource is not authorized")
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
    def read_uri(request: ReadURIRequest, ctx: _ApiCallContext = Depends(require_api_key)) -> Response:
        """读取任意 URI（privileged API）。R24 P1-S8 §24 / §7：
        production 默认 DISABLED；需要显式 ``uri:read`` permission——
        不能仅因 URI 落在全局 registered prefix 就允许。
        """
        # R32 P0-112：query slot exactly-once release 保证。
        slot_held = False
        try:
            ctx.authorize(request.uri, action=ACTION_URI_READ)
        except AuthorizationError:
            raise HTTPException(status_code=403, detail="resource is not authorized")
        if not query_slots.acquire(blocking=False):
            raise HTTPException(status_code=503, detail="查询并发已达上限，请稍后重试")
        slot_held = True
        try:
            budget = _api_budget(ReadRequest(dataset="uri"), settings)
            start = time.perf_counter()
            try:
                # R26-P0-005：request-scoped 执行上下文。
                from data_access.security.execution_context import execution_scope

                with execution_scope(_execution_context_for(ctx)):
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
            except AccessDeniedError:
                raise HTTPException(status_code=403, detail="resource is not authorized")
            except ValidationError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except DataAccessError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            table = handle.to_arrow()
            elapsed_ms = (time.perf_counter() - start) * 1000
            # P1-S6 §8：外部错误/响应不泄露完整 URI（signed URL / 完整路径）。
            meta = ReadResponseMeta(
                dataset="<uri>",
                snapshot_id=getattr(handle.snapshot, "snapshot_id", "-"),
                rows=table.num_rows,
                bytes=table.nbytes,
                elapsed_ms=elapsed_ms,
                format=request.format_out,
            )
            return _serialize(table, meta, request.format_out)
        finally:
            if slot_held:
                query_slots.release()
                slot_held = False

    @app.get("/v1/factors", dependencies=[Depends(require_api_key)])
    def factors_catalog(ctx: _ApiCallContext = Depends(require_api_key)) -> dict[str, Any]:
        """因子目录清单（FactorCatalog）。普通 ``factor:list`` 只返回脱敏摘要；
        敏感字段另需 ``factor:metadata_sensitive``（§7）。

        R26-P0-025：改用 ``VisibleFactorCatalog``——count / summary 全基于当前
        principal 可见 set（premium factor 的名字/存在性不外泄）；不暴露服务器
        本地 ``catalog.root``。
        """
        from data_access.read.visible_catalog import VisibleFactorCatalog

        ctx.authorize("factor_lake", action="factor:list")
        catalog = get_store().get_factor_catalog()
        sensitive = False
        try:
            ctx.authorize("factor_lake", action="factor:metadata_sensitive")
            sensitive = True
        except AuthorizationError:
            sensitive = False
        visible = VisibleFactorCatalog(
            catalog,
            policy=getattr(ctx.authorizer, "policy", None),
            principal=ctx.principal,
        )
        summary = [
            _redact_factor_meta(m, sensitive=sensitive)
            for m in visible.summary()
        ]
        return {
            "count": len(visible),
            "factors": summary,
        }

    @app.post("/v1/factors/read", dependencies=[Depends(require_api_key)])
    def factors_read(request: FactorReadRequest, ctx: _ApiCallContext = Depends(require_api_key)) -> Response:
        """一次读多个因子（单查询 UNION ALL / 宽表 PIVOT）。"""
        # R32 P0-112：query slot exactly-once release 保证。
        slot_held = False
        ctx.authorize("factor_lake", action="factor:read")
        if not query_slots.acquire(blocking=False):
            raise HTTPException(status_code=503, detail="查询并发已达上限，请稍后重试")
        slot_held = True
        try:
            start = time.perf_counter()
            try:
                # R26-P0-005：request-scoped 执行上下文（factor gate 用同一 principal）。
                from data_access.security.execution_context import execution_scope

                with execution_scope(_execution_context_for(ctx)):
                    handle = get_store().read_factors(
                        request.factor_ids,
                        time_range=request.time_range,
                        universe=request.universe,
                        frequency=request.frequency,
                        layout=request.layout,
                        columns=request.columns,
                        limit=request.limit,
                    )
            except AccessDeniedError:
                raise HTTPException(status_code=403, detail="resource is not authorized")
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
            if slot_held:
                query_slots.release()
                slot_held = False

    return app


_SENSITIVE_FACTOR_KEYS = {
    "expression",
    "source_config",
    "local_root",
    "lineage",
    "snapshot",
    "full_definition",
    "source",
    "data_source_config",
    "recipe",
}


def _redact_factor_meta(meta: dict[str, Any], *, sensitive: bool) -> dict[str, Any]:
    """因子元数据脱敏摘要（§7）：普通 factor:list 不下发敏感字段。"""
    if sensitive:
        return meta
    return {
        k: v for k, v in meta.items() if k not in _SENSITIVE_FACTOR_KEYS
    }


app = create_app()
