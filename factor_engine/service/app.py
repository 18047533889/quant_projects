"""R21 production HTTP service adapter around FactorEngine.

This module rewires the service around the R21 contract:

- production endpoints carry a non-downgradable :class:`EndpointExecutionPolicy`
  floor (R21-001..005); a research config / pit_enforce=false / direct-local
  write served by a production endpoint is rejected, never downgraded.
- authentication is endpoint-policy-aware (production routes are
  unconditionally authenticated) and principals come from trusted sources only
  (R21-011..018); job/artifact reads are owner-or-admin (R21-017).
- requests are typed Pydantic models with ``extra="forbid"`` (R21-032..036).
- jobs go through a bounded queue with deadline / cancel / heartbeat / real
  phases (R21-048..060) and durable, checksummed manifests (R21-070..075).
- errors cross the boundary as stable code + sanitized message + error_id
  (R21-087..090); full tracebacks stay internal.

Backward-compatible surface (existing tests import these): ``create_app``,
``STORE``, ``EXECUTOR``, ``JobStore``, ``JobRecord``, ``JobStatus``,
``JobType``, ``validate_spec``, ``submit_job``, ``_require_service_api_key``,
``_authorized_config_path``.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

try:  # module-level so FastAPI can resolve Request annotations (closure scope breaks detection)
    from fastapi import FastAPI, HTTPException, Request  # noqa: E402
except ImportError:  # pragma: no cover
    FastAPI = None  # type: ignore[assignment]
    HTTPException = None  # type: ignore[assignment]
    Request = None  # type: ignore[assignment]

from factor_engine.runtime.endpoint_policy import (
    EndpointExecutionPolicy,
    ProductionPolicyConflictError,
)

# R21 service modules
from factor_engine.service.errors import (
    ServiceError,
    classify_exception,
    redact_config,
    sanitize_message,
)
from factor_engine.service.jobstore import (
    JobPhase,
    JobRecord,
    JobStatus,
    JobStore,
    check_single_process_workers,
)
from factor_engine.service.models import (
    ComputeRequest,
    MaterializeRequest,
    ValidatedFactorRequest,
    estimate_factor_cost,
    size_budget,
)
from factor_engine.service.observability import (
    METRICS,
    bind_log_context,
    info,
    new_execution_id,
    trace_span,
    warning,
)
from factor_engine.service.policies import RuntimeFeaturePolicy, resolve_ambient_run_mode
from factor_engine.service.queue import (
    BoundedJobQueue,
    JobCancelledError,
    JobDeadlineExceeded,
    check_job_alive,
    set_phase,
)
from factor_engine.service.security import (
    ANONYMOUS_PRINCIPAL,
    ApprovedSourcePolicy,
    Principal,
    authorized_config_path as _security_authorized_config_path,
    require_privilege,
    resolve_principal,
)

JobType = Literal["compute", "materialize"]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Global service state (R40 #94: 惰性运行时 —— import service.app 无副作用)
#
# STORE/EXECUTOR/QUEUE/FEATURE_POLICY/SOURCE_POLICY 都是 _LazyRuntimeProxy：
# import 时不建目录、不启线程、不读 policy env。真正的构造 + QUEUE.start() 在
# create_app() 的 FastAPI lifespan 启动 handler 里调 _ensure_runtime() 完成。
# 模块级直接访问这些名字（如既有测试 ``from service.app import STORE``）会经代理
# 惰性构造 —— 向后兼容且 import 干净。
# ---------------------------------------------------------------------------

_EXECUTOR_MAX_WORKERS = max(1, int(os.environ.get("FACTOR_ENGINE_SERVICE_MAX_WORKERS", "4") or 4))
_JOB_TIMEOUT_DEFAULT = float(os.environ.get("FACTOR_ENGINE_SERVICE_JOB_TIMEOUT", "300") or "300")

_RUNTIME_LOCK = threading.RLock()
_RUNTIME_CONSTRUCTED = False
#: R40 #95: policy 版本 —— reload_policies() 刷新时 bump；在途 job 保留旧快照。
_POLICY_VERSION = 0


class _LazyRuntimeProxy:
    """延迟构造的运行时对象代理：首次属性/调用访问时触发 _ensure_runtime()。"""

    __slots__ = ("_attr",)

    def __init__(self, attr: str) -> None:
        object.__setattr__(self, "_attr", attr)

    def __getattr__(self, name: str) -> Any:
        _ensure_runtime(start=True)
        return getattr(globals()[self._attr], name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        _ensure_runtime(start=True)
        return globals()[self._attr](*args, **kwargs)


def _ensure_runtime(*, start: bool = True) -> None:
    """构造（并可选启动）service 运行时：STORE/EXECUTOR/QUEUE/policies。

    - ``import service.app`` 不调用本函数（无副作用）；
    - create_app() 的 lifespan 启动 handler 调用 ``_ensure_runtime(start=True)``；
    - 测试 monkeypatch ``service_app.STORE = JobStore(tmp_path)`` 后调用本函数会
      保留被替换的 STORE（非代理），QUEUE 用它作为持久化后端启动。
    """
    global STORE, EXECUTOR, QUEUE, FEATURE_POLICY, SOURCE_POLICY
    global _POLICY_VERSION, _RUNTIME_CONSTRUCTED
    with _RUNTIME_LOCK:
        if _RUNTIME_CONSTRUCTED:
            return
        _RUNTIME_CONSTRUCTED = True
        if isinstance(STORE, _LazyRuntimeProxy):
            STORE = JobStore()
        if isinstance(EXECUTOR, _LazyRuntimeProxy):
            EXECUTOR = ThreadPoolExecutor(
                max_workers=_EXECUTOR_MAX_WORKERS,
                thread_name_prefix="factor-engine-job",
            )
        if isinstance(QUEUE, _LazyRuntimeProxy):
            QUEUE = BoundedJobQueue(
                max_running=_EXECUTOR_MAX_WORKERS,
                timeout_default=_JOB_TIMEOUT_DEFAULT,
            )
        if start and isinstance(QUEUE, BoundedJobQueue) and not getattr(QUEUE, "_started", False):
            QUEUE.start(STORE)
        if isinstance(FEATURE_POLICY, _LazyRuntimeProxy):
            FEATURE_POLICY = RuntimeFeaturePolicy.from_env()
        if isinstance(SOURCE_POLICY, _LazyRuntimeProxy):
            SOURCE_POLICY = ApprovedSourcePolicy.from_env()


def reload_policies() -> dict[str, Any]:
    """R40 #95: 锁下刷新 FEATURE/SOURCE policy 并 bump version。

    在途 job 保留提交时快照的 policy_id/version/digest（JobRecord 字段），
    不受 reload 影响 —— 受控重载，不撕裂正在执行的 job 的不可变绑定。
    """
    global FEATURE_POLICY, SOURCE_POLICY, _POLICY_VERSION
    with _RUNTIME_LOCK:
        _ensure_runtime(start=False)
        new_feature = RuntimeFeaturePolicy.from_env()
        new_source = ApprovedSourcePolicy.from_env()
        changed = (
            new_feature.digest() != FEATURE_POLICY.digest()
            or new_source.digest() != SOURCE_POLICY.digest()
        )
        if changed:
            _POLICY_VERSION += 1
        FEATURE_POLICY = new_feature
        SOURCE_POLICY = new_source
        return {
            "policy_version": _POLICY_VERSION,
            "feature_digest": FEATURE_POLICY.digest(),
            "source_policy_digest": SOURCE_POLICY.digest(),
            "changed": changed,
        }


STORE = _LazyRuntimeProxy("STORE")
EXECUTOR = _LazyRuntimeProxy("EXECUTOR")
QUEUE = _LazyRuntimeProxy("QUEUE")
FEATURE_POLICY = _LazyRuntimeProxy("FEATURE_POLICY")
SOURCE_POLICY = _LazyRuntimeProxy("SOURCE_POLICY")

#: R40 #96: run_id -> request-scoped CancellationToken（job 线程可见，非持久化）。
_JOB_TOKEN_LOCK = threading.Lock()
_JOB_TOKENS: dict[str, Any] = {}


def _set_job_cancellation_token(run_id: str, token: Any) -> None:
    with _JOB_TOKEN_LOCK:
        _JOB_TOKENS[run_id] = token


def _get_job_cancellation_token(run_id: str) -> Any | None:
    with _JOB_TOKEN_LOCK:
        return _JOB_TOKENS.get(run_id)


def _drop_job_cancellation_token(run_id: str) -> None:
    with _JOB_TOKEN_LOCK:
        _JOB_TOKENS.pop(run_id, None)


def _resolve_version() -> str:
    """R21-127..129: single version authority — package metadata first, then
    the pyproject.toml source (never a hardcoded duplicate)."""
    try:
        import importlib.metadata

        return importlib.metadata.version("factor-engine")
    except Exception:
        pass
    try:
        from pathlib import Path

        pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
        if pyproject.is_file():
            import tomllib  # Python 3.11+

            with pyproject.open("rb") as fh:
                data = tomllib.load(fh)
            return str(data.get("project", {}).get("version") or "0.0.0")
    except (ImportError, OSError, ValueError):
        pass
    try:
        import yaml

        with open(Path(__file__).resolve().parents[1] / "pyproject.toml", encoding="utf-8") as fh:
            text = fh.read()
        # tomllib may be unavailable on 3.10; fall back to a small regex.
        import re

        m = re.search(r"^version\s*=\s*[\"']([^\"']+)[\"']", text, re.MULTILINE)
        if m:
            return m.group(1)
    except OSError:
        pass
    return "0.0.0"


_VERSION = os.environ.get(
    "FACTOR_ENGINE_SERVICE_VERSION",
    _resolve_version(),
)


# ---------------------------------------------------------------------------
# Backward-compat helpers
# ---------------------------------------------------------------------------


def _require_service_api_key(request) -> Dict[str, Any]:
    """Legacy auth helper — kept for the ``/health`` endpoint contract.  The
    real auth path is :func:`_authenticate` (endpoint-policy aware).

    Matches the legacy contract: when a service API key is configured, a valid
    key is required unless ``FACTOR_ENGINE_SERVICE_ALLOW_OPEN`` is set.
    """
    from factor_engine.service.security import get_principal_registry

    service_key = os.environ.get("FACTOR_ENGINE_SERVICE_API_KEY", "").strip() or None
    allow_open = os.environ.get("FACTOR_ENGINE_SERVICE_ALLOW_OPEN", "").lower() in {
        "1", "true", "yes", "on",
    }
    has_keys = bool(service_key) or get_principal_registry().has_keys()
    required = bool(has_keys and not allow_open)
    principal = resolve_principal(request, service_key=service_key, required=required)
    return {
        "identity": principal.identity,
        "authenticated": principal.authenticated(),
        "request_id": str(request.headers.get("X-Request-ID") or ""),
        "roles": list(principal.roles),
        "source": principal.source,
    }


def _authorized_config_path(raw: Any) -> Path:
    """Backward-compatible config path hardening wrapper (R21-028..031)."""
    return _security_authorized_config_path(raw)


def _authenticate(
    request,
    *,
    endpoint_policy: EndpointExecutionPolicy,
    required_role: str = "READ",
) -> Principal:
    """Endpoint-policy-aware authentication + authorization (R21-011..018).

    Production routes are *always* authenticated regardless of
    ``QUANT_PRODUCTION_MODE``/``FACTOR_ENGINE_SERVICE_ALLOW_OPEN``.
    """
    from factor_engine.service.security import get_principal_registry

    service_key = os.environ.get("FACTOR_ENGINE_SERVICE_API_KEY", "").strip() or None
    allow_open = os.environ.get("FACTOR_ENGINE_SERVICE_ALLOW_OPEN", "").lower() in {
        "1", "true", "yes", "on",
    }
    has_keys = bool(service_key) or get_principal_registry().has_keys()
    if endpoint_policy.is_production and not has_keys:
        raise ServiceError(
            "AUTH_REQUIRED",
            "production endpoints require a configured FACTOR_ENGINE_SERVICE_API_KEY",
            status=500,
        )
    # R21-011..013: production routes are unconditionally authenticated;
    # research routes require the key when one is configured and the route is
    # not explicitly opened by FACTOR_ENGINE_SERVICE_ALLOW_OPEN.
    required = endpoint_policy.is_production or (has_keys and not allow_open)
    principal = resolve_principal(request, service_key=service_key, required=required)
    bind_log_context(principal=principal.identity)
    require_privilege(principal, required_role)
    if not principal.authenticated():
        METRICS.incr("auth_failures")
    return principal


def _strict_sync(payload: dict[str, Any]) -> bool:
    """R21-033: only a real boolean ``True`` enables sync execution — the string
    ``"false"``/``"true"`` must never be Python-truthiness-coerced."""
    return payload.get("sync") is True


async def _read_json_body(request) -> dict[str, Any]:
    """Parse a JSON body with a hard size cap (R21-043)."""
    body_bytes = await request.body()
    body_bytes = bytes(body_bytes or b"")
    max_bytes = int(os.environ.get("FACTOR_ENGINE_SERVICE_MAX_BODY_BYTES", "1048576"))
    if len(body_bytes) > max_bytes:
        raise ServiceError(
            "MALFORMED_REQUEST",
            f"request body {len(body_bytes)} bytes exceeds {max_bytes}",
            status=413,
        )
    if not body_bytes:
        raise ServiceError("MALFORMED_REQUEST", "empty request body", status=400)
    try:
        payload = json.loads(body_bytes.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        raise ServiceError("MALFORMED_REQUEST", f"invalid JSON: {sanitize_message(exc)}", status=400) from exc
    if not isinstance(payload, dict):
        raise ServiceError("MALFORMED_REQUEST", "JSON object required", status=400)
    return payload


def _to_service_error(exc: BaseException, *, run_mode: str) -> ServiceError:
    code, family = classify_exception(exc, run_mode=run_mode)
    if isinstance(exc, ProductionPolicyConflictError):
        code, family = "PRODUCTION_ENDPOINT_CONFIG_POLICY_CONFLICT", "CONFLICT"
    status = 422 if family in {"VALIDATION", "PIT", "SOURCE", "SCHEMA", "DQ", "MATERIALIZE"} else None
    return ServiceError(code, str(exc), status=status)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_spec(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Validate DSL string and/or factor-engine spec (no execution).

    Backward-compatible return shape: ``{"ok","errors","warnings","checked"}``.
    R21-006..010: validation is surface/dialect/dialect_version aware and, for
    production, applies pre-parse size + cost budgets.
    """
    from factor_engine.api.dsl_parser import DSLParseError, parse_expr
    from factor_engine.api.mining_integration import validate_factor_engine_dsl, validate_production_dsl

    if not isinstance(payload, dict):
        return {"ok": False, "errors": ["payload must be an object"], "warnings": [], "checked": {}}
    errors: List[str] = []
    warnings: List[str] = []
    checked: Dict[str, Any] = {}
    run_mode = str(payload.get("run_mode") or "research").lower()
    if run_mode not in {"research", "production"}:
        errors.append("run_mode must be 'research' or 'production'")

    budget = size_budget()
    formula = (
        payload.get("formula")
        or payload.get("dsl")
        or ((payload.get("factor") or {}) if isinstance(payload.get("factor"), dict) else {}).get("expr")
        or ""
    )
    formula = str(formula or "").strip()
    surface = str(payload.get("surface") or "daily")
    if formula:
        # R21-037..039: pre-parse size budget.
        if len(formula.encode("utf-8")) > budget["max_formula_bytes"]:
            errors.append(f"formula exceeds max_formula_bytes={budget['max_formula_bytes']}")
        elif len(formula) > budget["max_formula_chars"]:
            errors.append(f"formula exceeds max_formula_chars={budget['max_formula_chars']}")
        else:
            validator = validate_production_dsl if run_mode == "production" else validate_factor_engine_dsl
            if run_mode == "production":
                # R40 #85: production 校验必须带真实 market —— US 公式绝不能用
                # 默认 A-share registry 校验（R24-159）。
                resolved_market = str(payload.get("market") or "ashare").strip().lower() or "ashare"
                ok, msg = validator(formula, market=resolved_market)
            else:
                ok, msg = validator(formula, surface=surface)
            checked["dsl"] = {"ok": bool(ok), "message": msg, "formula": formula}
            if not ok:
                errors.append(f"dsl: {msg}")
            else:
                try:
                    parse_expr(
                        formula,
                        surface=surface,
                        dialect=str(payload.get("dialect") or "native"),
                        dialect_version=payload.get("dialect_version"),
                    )
                except DSLParseError as exc:
                    errors.append(f"parse_expr: {exc}")
                    checked["dsl"]["ok"] = False
                    checked["dsl"]["message"] = str(exc)
    else:
        warnings.append("no formula/dsl/factor.expr provided")

    for key in ("data_source", "backend", "engine"):
        if key in payload and not isinstance(payload[key], dict):
            errors.append(f"{key} must be an object")
        elif key in payload:
            checked[key] = {"ok": True, "keys": sorted(payload[key].keys())}

    if "factor" in payload and not isinstance(payload["factor"], dict):
        errors.append("factor must be an object")

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "checked": checked,
    }


def _stable_hex(*parts: Any) -> str:
    return hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()[:16]


def _first(*values: Any) -> str | None:
    for v in values:
        if v is not None and str(v).strip():
            return str(v)
    return None


def _catalog_generations(model: ComputeRequest, source_binding: dict[str, str | None]) -> tuple[str, ...]:
    """R40 #149: catalog generations 多元组 —— 绑定全部 registry 维度。

    ``op_gen / field_gen / market_gen / calendar_gen / source_profile_gen /
    backend_evidence_gen / compiler_build_gen``。任一维度变化都会改变
    ``ValidatedFactorRequest.digest()``，从而失效 idempotency / cache / checkpoint。
    """
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    op_gen = str(OperatorRegistry.version())
    field_gen = "unavailable"
    try:
        from factor_engine.fields.market_registry import MULTI_MARKET_FIELD_REGISTRY

        registry = MULTI_MARKET_FIELD_REGISTRY.registry_for(str(model.market or "ashare"))
        field_gen = str(registry.catalog_hash())
    except Exception:
        pass
    market_gen = _stable_hex("market", str(model.market or "ashare"))
    calendar_gen = _stable_hex("calendar", str(model.calendar or ""))
    source_profile_gen = str(source_binding["source_profile_version"] or "none")
    backend_evidence_gen = _stable_hex("backend_evidence", str(model.backend or "auto"))
    compiler_build_gen = _stable_hex("build", _VERSION)
    return (
        op_gen, field_gen, market_gen, calendar_gen,
        source_profile_gen, backend_evidence_gen, compiler_build_gen,
    )


def _source_profile_binding(model: ComputeRequest, source_cfg: Any) -> dict[str, str | None]:
    """R40 #91: 从 source config / approved profile 提取 profile 版本与契约绑定。

    同 ``approved_source_profile_id`` 不同版本的 profile 内容 → 不同
    ``source_profile_version`` → 不同 digest（profile_id 只是引用，不是内容指纹）。
    """
    if not isinstance(source_cfg, dict):
        return {
            "source_profile_version": None,
            "source_contract_hash": None,
            "dataset_contract": None,
            "snapshot_policy": None,
            "provider_identity": None,
        }
    try:
        cfg_json = json.dumps(source_cfg, sort_keys=True, default=str)
    except (TypeError, ValueError):
        cfg_json = str(source_cfg)
    binding: dict[str, str | None] = {
        "source_contract_hash": _stable_hex("source", cfg_json),
        "dataset_contract": _first(source_cfg.get("dataset")),
        "snapshot_policy": _first(source_cfg.get("snapshot_policy")),
        "provider_identity": _first(source_cfg.get("type"), "data_access"),
        "source_profile_version": None,
    }
    profile_id = model.approved_source_profile_id
    if profile_id:
        profile = SOURCE_POLICY.approved_profiles.get(profile_id)
        if profile:
            try:
                binding["source_profile_version"] = _stable_hex(
                    "profile", json.dumps(profile, sort_keys=True, default=str)
                )
            except (TypeError, ValueError):
                binding["source_profile_version"] = _stable_hex("profile", profile_id)
        else:
            binding["source_profile_version"] = _stable_hex("profile", profile_id)
    return binding


def _build_execution_semantic_identity(
    model: ComputeRequest,
    *,
    formula: str,
    universe: tuple[str, ...],
    generation: str,
    source_binding: dict[str, str | None],
    run_mode: str,
) -> Any:
    """R40 #89: HTTP 路径构造不可变 ExecutionSemanticIdentityV2。

    市场/日历/decision_time/run_mode/PIT/source_scope 全字段 —— 与 FE 内部
    ``ExecutionSemanticIdentityV2`` 同一类型，供下游投影链消费（不可变 frozen）。
    """
    from factor_engine.runtime.factor_identity import ExecutionSemanticIdentityV2

    return ExecutionSemanticIdentityV2(
        ir_hash=_stable_hex("ir", formula),
        operator_contract_hash=_stable_hex("op", generation),
        field_contract_hash=_stable_hex("field", str(model.market or "ashare")),
        source_contract_hash=str(source_binding["source_contract_hash"] or ""),
        source_dependency_hash=_stable_hex("dep", str(model.calendar or "")),
        universe_membership_hash=_stable_hex("universe", *sorted(universe)),
        market=model.market,
        calendar=model.calendar,
        universe=str(sorted(universe)) or None,
        pit_policy="enforce" if run_mode == "production" else None,
        decision_time_policy=model.decision_time_policy,
        dialect=model.dialect.value,
        dialect_version=model.dialect_version,
        frequency=model.freq,
        source_scope_hash=str(source_binding["source_contract_hash"] or ""),
    )


def _validate_and_build_request(
    payload: dict[str, Any],
    *,
    endpoint_policy: EndpointExecutionPolicy,
    principal: Principal,
) -> tuple[dict[str, Any], str]:
    """Build the executable request + validated digest (R21-008..010).

    Returns ``(execution_payload, validated_digest)``.  The digest binds the
    canonical formula + execution context + catalog generations + complexity
    budget so execution cannot use a different context than validation.
    """
    try:
        model = ComputeRequest.model_validate(payload)
    except Exception as exc:
        raise ServiceError("MALFORMED_REQUEST", f"request validation failed: {sanitize_message(exc)}", status=422) from exc

    formula = model.formula_text()
    budget = size_budget()
    # R21-037..039
    if len(formula.encode("utf-8")) > budget["max_formula_bytes"]:
        raise ServiceError("MALFORMED_REQUEST", f"formula exceeds max_formula_bytes={budget['max_formula_bytes']}", status=422)
    if len(formula) > budget["max_formula_chars"]:
        raise ServiceError("MALFORMED_REQUEST", f"formula exceeds max_formula_chars={budget['max_formula_chars']}", status=422)
    if len(formula) > 0 and model.name and len(model.name) > budget["max_name_length"]:
        raise ServiceError("MALFORMED_REQUEST", "factor name too long", status=422)

    # Canonical formula: surface/dialect aware parse must already have succeeded
    # in validate_spec for the production route; here we compute the digest.
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    generation = OperatorRegistry.version()
    universe = tuple(model.universe or [])
    run_mode = endpoint_policy.value if endpoint_policy.is_production else model.run_mode.value

    # R40 #91: source profile 版本/契约绑定 —— 同 profile_id 不同版本 digest 不同。
    if endpoint_policy.is_production and not model.config_path:
        source_binding_cfg = _production_source_config(model)
    else:
        source_binding_cfg = model.data_source
    source_binding = _source_profile_binding(model, source_binding_cfg)

    # R40 #149: catalog generations 多元组（op/field/market/calendar/source-profile/
    # backend-evidence/compiler-build）。
    catalog_generations = _catalog_generations(model, source_binding)

    vr = ValidatedFactorRequest(
        canonical_formula=formula,
        surface=model.surface.value,
        dialect=model.dialect.value,
        dialect_version=model.dialect_version,
        frequency=model.freq,
        market=model.market,
        universe=universe,
        calendar=model.calendar,
        decision_time_policy=model.decision_time_policy,
        production_policy=endpoint_policy.value,
        catalog_generations=catalog_generations,
        complexity_budget=str(budget["max_ast_nodes"]) if "max_ast_nodes" in budget else "default",
        source_profile=model.approved_source_profile_id,
        backend=model.backend or "auto",
        resolved_backend_policy=model.backend or "auto",
        source_profile_version=source_binding["source_profile_version"],
        source_contract_hash=source_binding["source_contract_hash"],
        dataset_contract=source_binding["dataset_contract"],
        snapshot_policy=source_binding["snapshot_policy"],
        provider_identity=source_binding["provider_identity"],
    )

    # R21-044..047: cost gate before queue admission.
    cost = estimate_factor_cost(
        formula,
        universe_size=max(1, len(universe)),
        surface=model.surface.value,
    )
    max_cost = model.max_cost_per_job or int(
        os.environ.get("FACTOR_ENGINE_SERVICE_MAX_COST_PER_JOB", "2000")
    )
    max_cells = int(os.environ.get("FACTOR_ENGINE_SERVICE_MAX_CELLS", "500000000"))
    max_mem = int(os.environ.get("FACTOR_ENGINE_SERVICE_MAX_EXPECTED_MEMORY_MB", "8192"))
    over, reason = cost.exceeds(max_cost=max_cost, max_cells=max_cells, max_expected_memory=max_mem)
    if over:
        raise ServiceError(
            "RESOURCE_BUDGET_EXCEEDED",
            f"cost estimate rejected before admission: {reason}",
            status=422,
        )

    execution = {
        "validated": vr.to_payload(),
        "run_mode": run_mode,
        "backend": model.backend or "auto",
        "config_path": model.config_path,
        "formula_schema_version": model.formula_schema_version,
        # R40 #92: execution dict 必须带 factor name —— 否则 _execute_inline 里
        # ``execution.get("name")`` 恒 None，所有 HTTP 因子都叫 "inline_factor"。
        "name": model.name,
        # R40 #93: timeout 经 Pydantic 校验后由这里流入（不再 raw payload.get）。
        "timeout_seconds": model.timeout_seconds,
    }
    if endpoint_policy.is_production:
        # R21-020: production HTTP may only reference an approved source profile.
        if model.config_path:
            execution["config_path"] = model.config_path
        else:
            source = _production_source_config(model)
            execution["data_source"] = source
    else:
        execution["data_source"] = model.data_source
        execution["approved_source_profile_id"] = model.approved_source_profile_id
    execution["cost_estimate"] = cost.to_dict()
    # R40 #88: DataSourceBuildContext（run_mode/market/calendar/PIT/snapshot_policy）
    # 以 dict 形式流入 execution（JSON 可序列化），_execute_inline 再还原并传给
    # build_data_source(..., build_context=ctx)。
    from factor_engine.storage.factory import DataSourceBuildContext

    build_context = DataSourceBuildContext(
        run_mode=run_mode,
        market=model.market,
        calendar_id=model.calendar,
        pit_enforce=endpoint_policy.is_production,
        snapshot_policy=source_binding["snapshot_policy"],
        coverage_policy=None,
    )
    execution["build_context"] = {
        "run_mode": build_context.run_mode,
        "market": build_context.market,
        "calendar_id": build_context.calendar_id,
        "timezone": build_context.timezone,
        "pit_enforce": build_context.pit_enforce,
        "enforce_mining_gate": build_context.enforce_mining_gate,
        "snapshot_policy": build_context.snapshot_policy,
        "coverage_policy": build_context.coverage_policy,
    }
    # R40 #89: HTTP 路径构造不可变 ExecutionSemanticIdentityV2（frozen dataclass，
    # 全字段），to_dict 流入 execution dict；下游 _execute_inline 还原同一对象。
    execution["identity"] = _build_execution_semantic_identity(
        model,
        formula=formula,
        universe=universe,
        generation=str(generation),
        source_binding=source_binding,
        run_mode=run_mode,
    ).to_dict()
    return execution, vr.digest()


def _production_source_config(model: ComputeRequest) -> dict[str, Any]:
    """R21-020..026: resolve a production inline source from an approved profile."""
    from factor_engine.service.security import resolve_source_profile

    return resolve_source_profile({"approved_source_profile_id": model.approved_source_profile_id}, policy=SOURCE_POLICY)


def _validate_config_path_sources(config_path: str, *, production: bool) -> None:
    """R21-027 + R40 #84: config-file sources pass the same approved-source policy.

    import 失败绝不复现旧 fail-open（``except Exception: return`` 会让 production
    config-path source 绕过 approved-source 校验）—— 显式 INTERNAL_ERROR。
    """
    if not production:
        return
    try:
        from factor_engine.runtime.config_runtime import load_config
    except ImportError as exc:
        raise ServiceError(
            "INTERNAL_ERROR",
            f"cannot validate config path sources: {exc}",
            status=500,
        ) from exc
    config = load_config(config_path)
    SOURCE_POLICY.validate_source(config.data_source, production=True)


# ---------------------------------------------------------------------------
# Job execution workers
# ---------------------------------------------------------------------------


def _execute_config_path(job: JobRecord, config_path: str) -> dict[str, Any]:
    from factor_engine.runtime.engine import FactorEngine

    # R40 #96: engine 入口边界检查 request-scoped cancellation token。
    token = _get_job_cancellation_token(job.run_id)
    if token is not None:
        token.raise_if_cancelled()
    authorized = _authorized_config_path(config_path)
    _validate_config_path_sources(str(authorized), production=job.endpoint_policy == "production")
    return FactorEngine.run_from_config(
        str(authorized),
        execution_policy=job.endpoint_policy or None,
    )


def _execute_inline(job: JobRecord, execution: dict[str, Any]) -> dict[str, Any]:
    from factor_engine.api.dsl_parser import parse_factor
    from factor_engine.backend.factory import build_backend
    from factor_engine.runtime.engine import FactorEngine
    from factor_engine.storage.factory import build_data_source

    validated = execution["validated"]
    vr = ValidatedFactorRequest(
        canonical_formula=str(validated["canonical_formula"]),
        surface=str(validated["surface"]),
        dialect=str(validated["dialect"]),
        dialect_version=validated.get("dialect_version"),
        frequency=validated.get("frequency"),
        market=validated.get("market"),
        universe=tuple(validated.get("universe") or ()),
        calendar=validated.get("calendar"),
        decision_time_policy=validated.get("decision_time_policy"),
        production_policy=str(validated["production_policy"]),
        catalog_generations=tuple(validated.get("catalog_generations") or ()),
        complexity_budget=str(validated.get("complexity_budget") or "default"),
        source_profile=validated.get("source_profile"),
        # R40 #90/#91: 与 _validate_and_build_request 构造的 digest 全字段对齐。
        backend=validated.get("backend"),
        resolved_backend_policy=validated.get("resolved_backend_policy"),
        source_profile_version=validated.get("source_profile_version"),
        source_contract_hash=validated.get("source_contract_hash"),
        dataset_contract=validated.get("dataset_contract"),
        snapshot_policy=validated.get("snapshot_policy"),
        provider_identity=validated.get("provider_identity"),
    )
    # R21-010: execution must use exactly the validated request.
    if not vr.execution_digest_matches(job.request_digest or ""):
        raise ServiceError(
            "INTERNAL_CONTRACT_VIOLATION",
            "validated_request_digest != execution_request_digest",
            status=500,
        )

    production = job.endpoint_policy == "production"
    source_cfg = execution.get("data_source")
    if not isinstance(source_cfg, dict):
        raise ValueError("compute job requires config_path or data_source")
    # R40 #96: engine 入口边界检查 request-scoped cancellation token。
    token = _get_job_cancellation_token(job.run_id)
    if token is not None:
        token.raise_if_cancelled()
    # R40 #88: 把验证期构造的 DataSourceBuildContext 传给 build_data_source
    #（run_mode/market/calendar/timezone/PIT/snapshot_policy/coverage_policy 流入源）。
    from factor_engine.storage.factory import DataSourceBuildContext

    build_ctx_raw = execution.get("build_context")
    build_context = (
        DataSourceBuildContext(**build_ctx_raw)
        if isinstance(build_ctx_raw, dict)
        else None
    )
    source = build_data_source(source_cfg, build_context=build_context)
    backend = build_backend(str(execution.get("backend") or "auto"))
    engine = FactorEngine(
        backend=backend,
        data_source=source,
        run_mode=execution.get("run_mode") or ("production" if production else "research"),
    )
    factor = parse_factor(
        vr.canonical_formula,
        name=str(execution.get("name") or "inline_factor"),
        freq=vr.frequency or "1d",
        universe=vr.universe or None,
        surface=vr.surface,
        dialect=vr.dialect,
        dialect_version=vr.dialect_version,
    )
    out = engine.run(
        factor,
        input_dq_check=production,
        auto_warmup=production,
        pit_enforce=production,
    )
    return out


def _run_compute(job: JobRecord) -> None:
    _job_wrapper(job, phase_target=JobPhase.EXECUTING)


def _run_materialize(job: JobRecord) -> None:
    _job_wrapper(job, phase_target=JobPhase.MATERIALIZING)


def _job_wrapper(job: JobRecord, *, phase_target: str) -> None:
    """Run a job worker with full lifecycle: phases, heartbeat, deadline,
    cancel, sanitized errors (R21-055..060, 087..090).

    R40 #96: 阶段边界额外查 request-scoped CancellationToken —— cancel/deadline
    在 engine 入口与阶段边界都能立即停止，而不是等 worker 自然结束。
    """
    from factor_engine.service.errors import json_dumps_redacted
    from factor_engine.runtime.exceptions import (
        Cancellation,
        DeadlineExceeded,
        reset_active_cancellation_token,
        set_active_cancellation_token,
    )

    _token_ctx = None
    try:
        token = _get_job_cancellation_token(job.run_id)
        if token is not None:
            # R40 #96: 把 token 放进 engine/scheduler 可见的 ContextVar —— engine
            # 阶段边界查询。finally 里 reset。
            _token_ctx = set_active_cancellation_token(token)
        try:
            if token is not None:
                token.raise_if_cancelled()
            if job.status == JobStatus.CANCELLED or job.cancel_requested_at:
                job.status = JobStatus.CANCELLED
                job.finished_at = _utc_now()
                STORE.update(job)
                return
            job.status = JobStatus.RUNNING
            job.started_at = _utc_now()
            job.worker_id = os.environ.get("FACTOR_ENGINE_SERVICE_NODE", "local")
            job.touch_heartbeat()
            STORE.update(job)
            execution = job.request.get("execution") if isinstance(job.request, dict) else None
            info("job.start", run_id=job.run_id, job_type=job.job_type, endpoint_policy=job.endpoint_policy)
            set_phase(job, STORE, JobPhase.VALIDATING)
            if token is not None:
                token.raise_if_cancelled()
            check_job_alive(job)
            set_phase(job, STORE, phase_target)
            out = _dispatch_execution(job, execution)
            set_phase(job, STORE, JobPhase.FINALIZING)
            summary = _summarize_result(out, job)
            job.result_summary = summary
            job.status = JobStatus.SUCCEEDED
            job.finished_at = _utc_now()
            METRICS.incr("job_succeeded", labels={"job_type": job.job_type})
            info("job.end", run_id=job.run_id, status="succeeded")
        except (JobCancelledError, Cancellation):
            job.status = JobStatus.CANCELLED
            job.error_code = "JOB_CANCELLED"
            job.finished_at = _utc_now()
            METRICS.incr("job_cancel", labels={"job_type": job.job_type})
            info("job.cancelled", run_id=job.run_id)
        except (JobDeadlineExceeded, DeadlineExceeded):
            job.status = JobStatus.TIMED_OUT
            job.error_code = "JOB_DEADLINE_EXCEEDED"
            job.finished_at = _utc_now()
            METRICS.incr("job_timeout", labels={"job_type": job.job_type})
            info("job.timeout", run_id=job.run_id)
        except Exception as exc:  # noqa: BLE001
            se = _to_service_error(exc, run_mode=job.endpoint_policy)
            job.status = JobStatus.FAILED
            job.error_code = se.code
            job.error_id = se.error_id
            job.error = sanitize_message(se.message)
            job.finished_at = _utc_now()
            job.artifacts["traceback"] = traceback.format_exc(limit=15)
            METRICS.incr("error_total", labels={"error_code": se.code, "job_type": job.job_type})
            info("job.failed", run_id=job.run_id, error_code=se.code, error_id=se.error_id)
        STORE.update(job)
    finally:
        _drop_job_cancellation_token(job.run_id)
        if _token_ctx is not None:
            try:
                reset_active_cancellation_token(_token_ctx)
            except Exception:  # noqa: BLE001
                pass


def _dispatch_execution(job: JobRecord, execution: dict[str, Any] | None) -> dict[str, Any]:
    if job.job_type == "materialize":
        return _execute_materialize(job)
    config_path = execution.get("config_path") if execution else None
    if config_path:
        with trace_span("run_from_config", run_id=job.run_id, config_path=config_path):
            out = _execute_config_path(job, str(config_path))
            return {"out": out, "mode": "run_from_config"}
    with trace_span("inline_dsl", run_id=job.run_id):
        out = _execute_inline(job, execution or {})
        return {"out": out, "mode": "inline_dsl"}


def _execute_materialize(job: JobRecord) -> dict[str, Any]:
    from factor_engine.runtime.engine import FactorEngine

    execution = job.request.get("execution") if isinstance(job.request, dict) else {}
    config_path = execution.get("config_path") or job.request.get("config_path")
    if not config_path:
        raise ValueError("materialize job requires config_path")
    config_path = _authorized_config_path(str(config_path))
    _validate_config_path_sources(str(config_path), production=job.endpoint_policy == "production")
    kwargs: dict[str, Any] = {}
    for k in ("factor_id", "author", "frequency", "description", "expression"):
        if execution.get(k) is not None:
            kwargs[k] = execution[k]
    write_target = execution.get("write_target")
    with trace_span("materialize_from_config", run_id=job.run_id, config_path=str(config_path)):
        out = FactorEngine.materialize_from_config(
            str(config_path),
            execution_policy=job.endpoint_policy or None,
            write_target=write_target,
            **kwargs,
        )
    return {"out": out, "mode": "materialize_from_config"}


#: R40 #150: 角色 → 可读数据敏感度上限（与 security/access 的 tag 分级对齐：
#: public=10, market.basic=20, fundamental=30, alt.premium=40, internal.restricted=50）。
_ROLE_ACCESS_LEVEL = {
    "READ": 10,
    "COMPUTE": 20,
    "MATERIALIZE": 30,
    "PUBLISH": 40,
    "ADMIN": 50,
}


def _principal_access_level(roles: Any) -> int:
    if not roles:
        return 10  # 未知/anonymous → 只能看 public
    return max((_ROLE_ACCESS_LEVEL.get(str(r), 10) for r in roles), default=10)


def _collect_source_access_tags(source_cfg: Any) -> list[str]:
    """递归收集 source config 的 access_tags / classification（含 composite 子源）。"""
    tags: list[str] = []
    if isinstance(source_cfg, dict):
        for key, value in source_cfg.items():
            if key in ("access_tags", "classification"):
                if isinstance(value, str):
                    tags.append(value)
                elif isinstance(value, (list, tuple)):
                    tags.extend(str(v) for v in value if v)
                elif isinstance(value, dict):
                    tags.extend(str(v) for v in value.values() if v)
            elif key in ("sources", "source", "inner") and isinstance(value, dict):
                tags.extend(_collect_source_access_tags(value))
            elif key == "sources" and isinstance(value, list):
                for sub in value:
                    tags.extend(_collect_source_access_tags(sub))
    return tags


def _redact_preview_for_access(source_cfg: Any, job: JobRecord) -> str | None:
    """R40 #150: 结果预览继承 source 的 derived access classification。

    max_sensitivity(source tags) 超过调用方（job owner）principal.access_level 时
    拒绝写入原始预览 —— 返回 redacted 占位串；否则返回 None（原样写入）。
    """
    if not isinstance(source_cfg, dict):
        return None
    try:
        from factor_engine.security.access import derive_derived_access_tags, max_sensitivity

        tags = derive_derived_access_tags(_collect_source_access_tags(source_cfg))
        sensitivity = max_sensitivity(tags)
        if sensitivity <= 0:
            return None
        roles: list[Any] = []
        meta = job.request_metadata if isinstance(job.request_metadata, dict) else {}
        roles = list(meta.get("roles") or [])
        if _principal_access_level(roles) >= sensitivity:
            return None
        return (
            f"<preview redacted: source access classification max_sensitivity="
            f"{sensitivity}>"
        )
    except Exception:  # noqa: BLE001 - redaction 失败不阻断结果汇总
        return None


def _summarize_result(payload: dict[str, Any], job: JobRecord) -> dict[str, Any]:
    mode = payload.get("mode")
    out = payload.get("out") or {}
    execution = job.request.get("execution") if isinstance(job.request, dict) else {}
    source_cfg = execution.get("data_source") if isinstance(execution, dict) else None

    def _store_preview(result: Any) -> None:
        preview = str(result.head(5).to_string())
        redacted = _redact_preview_for_access(source_cfg, job)
        job.artifacts["result_preview"] = redacted if redacted is not None else preview

    if mode == "run_from_config":
        summary: dict[str, Any] = {"mode": mode}
        if isinstance(out, dict):
            result = out.get("result")
            if result is not None:
                try:
                    summary["rows"] = int(getattr(result, "shape", [0])[0])
                    _store_preview(result)
                except Exception:
                    summary["result_type"] = type(result).__name__
            summary["factor"] = getattr(out.get("factor"), "name", None)
        return summary
    if mode == "materialize_from_config":
        summary = {"mode": mode}
        if isinstance(out, dict):
            for key in ("lake_root", "factor_id", "catalog_path", "output_path", "manifest_path"):
                if out.get(key) is not None:
                    job.artifacts[key] = str(out[key])
            summary["keys"] = sorted(out.keys())
        return summary
    # inline
    summary = {"mode": "inline_dsl"}
    out = payload.get("out") or {}
    if isinstance(out, dict):
        result = out.get("result")
        if result is not None:
            try:
                summary["rows"] = int(getattr(result, "shape", [0])[0])
                _store_preview(result)
            except Exception:
                pass
    return summary


# ---------------------------------------------------------------------------
# Submission
# ---------------------------------------------------------------------------


def _scoped_idempotency_key(principal: Principal, job_type: str, key: str) -> str:
    return f"{principal.scope_key()}|{job_type}|{key}"


def _submit_job(
    job_type: JobType,
    payload: dict[str, Any],
    *,
    principal: Principal,
    endpoint_policy: EndpointExecutionPolicy,
    sync: bool = False,
) -> dict[str, Any]:
    budget = size_budget()
    idem_key = str(payload.get("idempotency_key") or "").strip() or None
    if idem_key and len(idem_key) > budget["max_idempotency_key_length"]:
        raise ServiceError("MALFORMED_REQUEST", "idempotency_key too long", status=422)
    scoped_key = _scoped_idempotency_key(principal, job_type, idem_key) if idem_key else None

    # R21-052..054: production compute/materialize are always async.
    if sync and endpoint_policy.is_production:
        raise ServiceError(
            "POLICY",
            "sync execution is not allowed on production endpoints (jobs are async)",
            status=422,
        )

    # Validate + build the immutable request + digest (R21-008..010).
    if job_type == "materialize":
        execution, digest = _validate_materialize_request(payload, endpoint_policy=endpoint_policy)
    else:
        execution, digest = _validate_and_build_request(payload, endpoint_policy=endpoint_policy, principal=principal)

    # Idempotency must be request-bound (R21-065..068).
    if scoped_key:
        existing = STORE.get_by_idempotency_key(scoped_key)
        if existing is not None:
            if existing.request_digest and existing.request_digest != digest:
                raise ServiceError(
                    "IDEMPOTENCY_KEY_CONFLICT",
                    f"idempotency_key {idem_key!r} already used with a different request",
                    status=409,
                )
            return {
                "run_id": existing.run_id,
                "status": existing.status,
                "submitted_at": existing.submitted_at,
                "job_type": existing.job_type,
                "idempotent": True,
            }

    run_id = uuid.uuid4().hex
    # R40 #93: timeout 走 Pydantic 校验后的值（execution dict），不再 raw
    # ``payload.get("timeout_seconds")`` —— 非法值在模型校验阶段已被拒。
    timeout = float(execution.get("timeout_seconds") or QUEUE.timeout_default)
    # R40 #95: 提交时快照当前 policy 的 id/version/digest —— 之后 reload_policies()
    # 刷新全局 policy 不影响在途 job 的不可变绑定。
    _ensure_runtime(start=True)
    policy_id = "runtime_feature_policy"
    policy_version = _POLICY_VERSION
    policy_digest = FEATURE_POLICY.digest()
    job = JobRecord(
        run_id=run_id,
        requested_by=principal.identity,
        owner_principal=principal.identity,
        tenant=principal.tenant,
        project=principal.project,
        job_type=job_type,
        endpoint_policy=endpoint_policy.value,
        idempotency_key=scoped_key,
        request_metadata=dict(payload.get("request_metadata") or {}),
        request={"execution": execution},
        request_digest=digest,
        execution_policy_digest=policy_digest,
        policy_id=policy_id,
        policy_version=policy_version,
        policy_digest=policy_digest,
        timeout_seconds=timeout,
        attempt=1,
        phase=JobPhase.VALIDATING,
        cost_estimate=dict(execution.get("cost_estimate") or {}),
    )
    import datetime as _dt
    import time as _time

    job.deadline_at = (_dt.datetime.now(_dt.timezone.utc) + _dt.timedelta(seconds=timeout)).isoformat()
    job.deadline_monotonic = _time.monotonic() + timeout  # R21-277
    # R40 #96: request-scoped CancellationToken（cancel event + monotonic deadline），
    # _job_wrapper / _execute_inline / _execute_config_path 阶段边界查询。
    from factor_engine.runtime.exceptions import CancellationToken

    _set_job_cancellation_token(run_id, CancellationToken(deadline_monotonic=_time.monotonic() + timeout))
    job = STORE.create(job)
    target = _run_compute if job_type == "compute" else _run_materialize
    if sync:
        target(job)
    else:
        QUEUE.submit(job, run_fn=target)
    return {
        "run_id": job.run_id,
        "status": job.status,
        "submitted_at": job.submitted_at,
        "job_type": job.job_type,
    }


def _validate_materialize_request(
    payload: dict[str, Any],
    *,
    endpoint_policy: EndpointExecutionPolicy,
) -> tuple[dict[str, Any], str]:
    try:
        model = MaterializeRequest.model_validate(payload)
    except Exception as exc:
        raise ServiceError("MALFORMED_REQUEST", f"materialize request validation failed: {sanitize_message(exc)}", status=422) from exc
    run_mode = endpoint_policy.value if endpoint_policy.is_production else model.run_mode.value
    execution = {
        "config_path": model.config_path,
        "factor_id": model.factor_id,
        "author": model.author,
        "frequency": model.frequency,
        "description": model.description,
        "expression": model.expression,
        "write_target": model.write_target.value if not endpoint_policy.is_production else "production",
        "run_mode": run_mode,
        # R40 #93: timeout 经 Pydantic 校验后由这里流入。
        "timeout_seconds": model.timeout_seconds,
    }
    digest = hashlib.sha256(
        json.dumps({"config_path": model.config_path, "write_target": execution["write_target"], "policy": run_mode}, sort_keys=True).encode()
    ).hexdigest()
    return execution, digest


def submit_job(job_type: JobType, payload: dict[str, Any], *, sync: bool = False) -> dict[str, Any]:
    """Backward-compatible submission entry (kept for tests/external callers)."""
    principal = ANONYMOUS_PRINCIPAL
    if isinstance(payload.get("request_metadata"), dict) and payload["request_metadata"].get("identity"):
        identity = str(payload["request_metadata"].get("identity"))
        if identity and identity != "anonymous":
            principal = Principal(identity=identity, roles=("COMPUTE", "MATERIALIZE"), source="api_key_mapping")
    run_mode = str(payload.get("run_mode") or "research").lower()
    policy = EndpointExecutionPolicy.PRODUCTION if run_mode == "production" else EndpointExecutionPolicy.RESEARCH
    return _submit_job(job_type, payload, principal=principal, endpoint_policy=policy, sync=sync)


# ---------------------------------------------------------------------------
# Readiness / liveness
# ---------------------------------------------------------------------------


def _readiness_check() -> dict[str, Any]:
    from factor_engine.service.release_blockers import evaluate_blockers

    blockers = evaluate_blockers()
    fail = sum(1 for b in blockers.values() if b["status"] == "FAIL")
    unknown = sum(1 for b in blockers.values() if b["status"] == "UNKNOWN")
    disk_free_mb = 0
    try:
        import shutil

        disk_free_mb = int(shutil.disk_usage(STORE.root).free / (1024 * 1024))
    except OSError:
        disk_free_mb = -1
    return {
        "ready": fail == 0 and unknown == 0 and disk_free_mb >= 0,
        "critical_blockers": fail,
        "unknown_contracts": unknown,
        "disk_free_mb": disk_free_mb,
        "corrupt_manifests": STORE.corruption_count,
        "queue": QUEUE.snapshot().__dict__,
        "version": _VERSION,
        "feature_digest": FEATURE_POLICY.digest(),
        "source_policy_digest": SOURCE_POLICY.digest(),
    }


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------


def create_app():
    from contextlib import asynccontextmanager

    if FastAPI is None:  # pragma: no cover
        raise ImportError("HTTP service requires fastapi. Install with: pip install 'factor-engine[service]'")
    check_single_process_workers()

    from factor_engine.service.preflight import production_preflight

    PREFLIGHT: dict[str, Any] = {"ok": True, "checks": {}}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # R40 #94: 运行时（STORE/EXECUTOR/QUEUE/policies）在 lifespan 启动时构造并
        # start —— ``import service.app`` 无副作用（无线程/无目录）。
        _ensure_runtime(start=True)
        # R21-264..266: one-shot production preflight before ready.
        PREFLIGHT.update(production_preflight())
        try:
            ambient = resolve_ambient_run_mode()
        except ValueError:
            ambient = "invalid"
        is_production_deploy = ambient == "production" or os.environ.get(
            "FACTOR_ENGINE_PREFLIGHT_HARD_FAIL", "0"
        ).lower() in {"1", "true", "yes"}
        if not PREFLIGHT["ok"] and is_production_deploy:
            info("factor_engine.service.preflight.blocked")
            raise RuntimeError("production preflight failed; refusing to serve")
        yield
        # R21-080..082: graceful shutdown — drain queued, cancel running, close.
        info("factor_engine.service.shutdown.start")
        QUEUE.drain(timeout=float(os.environ.get("FACTOR_ENGINE_SERVICE_DRAIN_TIMEOUT", "30")))
        EXECUTOR.shutdown(wait=True)
        STORE.close()
        info("factor_engine.service.shutdown.done")

    app = FastAPI(
        title="Factor Engine Service",
        version=_VERSION,
        description="R21 production service adapter over FactorEngine",
        lifespan=lifespan,
    )

    def _auth_http(request: Request, *, policy: EndpointExecutionPolicy, role: str = "READ") -> Principal:
        try:
            return _authenticate(request, endpoint_policy=policy, required_role=role)
        except ServiceError as exc:
            METRICS.incr("error_total", labels={"error_code": exc.code})
            raise HTTPException(status_code=exc.status, detail=exc.to_public()) from exc

    def _owner_check(request: Request, job: JobRecord, principal: Principal) -> None:
        if job.owner_principal != principal.identity and not principal.has_role("ADMIN"):
            raise HTTPException(
                status_code=403,
                detail=ServiceError("OWNER_ONLY", "job access is owner-or-admin only", status=403).to_public(),
            )

    @app.get("/health")
    def health(request: Request) -> dict[str, str]:
        try:
            identity = _require_service_api_key(request)
        except ServiceError as exc:
            raise HTTPException(status_code=exc.status, detail=exc.to_public()) from exc
        return {"status": "ok", "service": "factor_engine", "identity": identity["identity"]}

    @app.get("/livez")
    def livez() -> dict[str, str]:
        # R21-083: liveness only proves the process is up.
        return {"status": "alive"}

    @app.get("/readyz")
    def readyz() -> dict[str, Any]:
        # R21-084..086: readiness reflects blockers, not just process up.
        check = _readiness_check()
        if not check["ready"]:
            raise HTTPException(status_code=503, detail=check)
        return check

    @app.get("/metrics")
    def metrics(request: Request) -> dict[str, Any]:
        _auth_http(request, policy=EndpointExecutionPolicy.RESEARCH, role="READ")
        return {"metrics": METRICS.snapshot(), "queue": QUEUE.snapshot().__dict__}

    @app.get("/factor-engine/operators")
    def operators(request: Request) -> dict[str, Any]:
        _auth_http(request, policy=EndpointExecutionPolicy.RESEARCH, role="READ")
        return list_operators()

    @app.post("/factor-engine/validate-spec")
    async def validate(request: Request) -> dict[str, Any]:
        _auth_http(request, policy=EndpointExecutionPolicy.RESEARCH, role="COMPUTE")
        payload = await _read_json_body(request)
        return validate_spec(payload)

    @app.post("/factor-engine/research/compute")
    async def research_compute(request: Request) -> dict[str, Any]:
        principal = _auth_http(request, policy=EndpointExecutionPolicy.RESEARCH, role="COMPUTE")
        payload = await _read_json_body(request)
        payload = dict(payload)
        payload["run_mode"] = "research"
        payload["request_metadata"] = principal.to_public()
        try:
            return _submit_job(
                "compute", payload, principal=principal,
                endpoint_policy=EndpointExecutionPolicy.RESEARCH,
                sync=_strict_sync(payload),
            )
        except ServiceError as exc:
            raise HTTPException(status_code=exc.status, detail=exc.to_public()) from exc

    @app.post("/factor-engine/production/compute")
    async def production_compute(request: Request) -> dict[str, Any]:
        principal = _auth_http(request, policy=EndpointExecutionPolicy.PRODUCTION, role="COMPUTE")
        payload = await _read_json_body(request)
        validation = validate_spec(payload)
        if not validation["ok"]:
            raise HTTPException(status_code=422, detail=validation["errors"])
        payload = dict(payload)
        payload["request_metadata"] = principal.to_public()
        try:
            # R21-052..054: production is always async into the job system; a
            # caller sending sync=true gets a POLICY rejection, never a silent
            # heavy synchronous run.
            return _submit_job(
                "compute", payload, principal=principal,
                endpoint_policy=EndpointExecutionPolicy.PRODUCTION,
                sync=_strict_sync(payload),
            )
        except ServiceError as exc:
            raise HTTPException(status_code=exc.status, detail=exc.to_public()) from exc

    @app.post("/factor-engine/production/materialize")
    async def production_materialize(request: Request) -> dict[str, Any]:
        # R21-227..229: materialize/publish is a write privilege above compute.
        principal = _auth_http(request, policy=EndpointExecutionPolicy.PRODUCTION, role="MATERIALIZE")
        payload = await _read_json_body(request)
        payload = dict(payload)
        payload["request_metadata"] = principal.to_public()
        try:
            return _submit_job(
                "materialize", payload, principal=principal,
                endpoint_policy=EndpointExecutionPolicy.PRODUCTION,
                sync=_strict_sync(payload),
            )
        except ServiceError as exc:
            raise HTTPException(status_code=exc.status, detail=exc.to_public()) from exc

    @app.post("/factor-engine/jobs/compute")
    async def jobs_compute(request: Request) -> dict[str, Any]:
        # R21-224..226: generic /jobs/compute is NOT a backdoor — it routes
        # through the same validation/admission as the dedicated routes.
        payload = await _read_json_body(request)
        run_mode = str(payload.get("run_mode") or "research").lower()
        policy = EndpointExecutionPolicy.PRODUCTION if run_mode == "production" else EndpointExecutionPolicy.RESEARCH
        principal = _auth_http(request, policy=policy, role="COMPUTE")
        payload = dict(payload)
        payload["request_metadata"] = principal.to_public()
        try:
            return _submit_job(
                "compute", payload, principal=principal, endpoint_policy=policy,
                sync=_strict_sync(payload),
            )
        except ServiceError as exc:
            raise HTTPException(status_code=exc.status, detail=exc.to_public()) from exc

    @app.post("/factor-engine/jobs/materialize")
    async def jobs_materialize(request: Request) -> dict[str, Any]:
        payload = await _read_json_body(request)
        run_mode = str(payload.get("run_mode") or "production").lower()
        policy = EndpointExecutionPolicy.PRODUCTION if run_mode == "production" else EndpointExecutionPolicy.RESEARCH
        principal = _auth_http(request, policy=policy, role="MATERIALIZE")
        payload = dict(payload)
        payload["request_metadata"] = principal.to_public()
        try:
            return _submit_job(
                "materialize", payload, principal=principal, endpoint_policy=policy,
                sync=_strict_sync(payload),
            )
        except ServiceError as exc:
            raise HTTPException(status_code=exc.status, detail=exc.to_public()) from exc

    @app.get("/factor-engine/jobs/{run_id}")
    def job_status(run_id: str, request: Request) -> dict[str, Any]:
        policy = EndpointExecutionPolicy.RESEARCH
        principal = _auth_http(request, policy=policy, role="READ")
        job = STORE.get(run_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"unknown run_id: {run_id}")
        _owner_check(request, job, principal)
        return job.to_public()

    @app.get("/factor-engine/jobs/{run_id}/artifacts")
    def job_artifacts(run_id: str, request: Request) -> dict[str, Any]:
        principal = _auth_http(request, policy=EndpointExecutionPolicy.RESEARCH, role="READ")
        job = STORE.get(run_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"unknown run_id: {run_id}")
        _owner_check(request, job, principal)
        return {"run_id": run_id, "status": job.status, "artifacts": job.to_public()["artifacts"]}

    @app.post("/factor-engine/jobs/{run_id}/cancel")
    def job_cancel(run_id: str, request: Request) -> dict[str, Any]:
        # R21-057: cancel endpoint — propagates via the queue's cancel flag which
        # the worker checks between phases (R21-058).
        principal = _auth_http(request, policy=EndpointExecutionPolicy.RESEARCH, role="COMPUTE")
        job = STORE.get(run_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"unknown run_id: {run_id}")
        _owner_check(request, job, principal)
        if QUEUE.cancel(run_id):
            return {"run_id": run_id, "status": "cancelling"}
        return {"run_id": run_id, "status": job.status, "message": "job already terminal"}

    @app.post("/factor-engine/jobs/{run_id}/retry")
    def job_retry(run_id: str, request: Request) -> dict[str, Any]:
        # R21-167..170: only non-validation/terminal failures are retried; each
        # retry carries a new attempt id (new run_id) and never re-runs a
        # materialization side effect blindly.
        principal = _auth_http(request, policy=EndpointExecutionPolicy.RESEARCH, role="COMPUTE")
        job = STORE.get(run_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"unknown run_id: {run_id}")
        _owner_check(request, job, principal)
        if job.status in {"succeeded", "cancelled", "timed_out", "interrupted"}:
            raise HTTPException(status_code=409, detail="job not retryable from terminal state")
        if job.error_code and job.error_code in {"VALIDATION_FAILED", "PIT_VIOLATION", "OUTPUT_DOMAIN_VIOLATION", "SOURCE_EMPTY", "POLICY"}:
            raise HTTPException(status_code=422, detail="validation/PIT/DQ/policy errors are never retried")
        return _submit_job(
            job.job_type,
            {"request_metadata": principal.to_public(), **job.request.get("execution", {})},
            principal=principal,
            endpoint_policy=EndpointExecutionPolicy(job.endpoint_policy or "research"),
            sync=False,
        )

    return app


def list_operators() -> Dict[str, Any]:
    from factor_engine.api.operator_registry import build_dsl_allowlist

    names = sorted(build_dsl_allowlist().keys())
    return {"count": len(names), "operators": names}


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run Factor Engine HTTP service")
    parser.add_argument("--host", default=os.environ.get("FACTOR_ENGINE_SERVICE_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("FACTOR_ENGINE_SERVICE_PORT", "8088")))
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args(argv)
    if args.reload and os.environ.get("QUANT_PRODUCTION_MODE", "").lower() in {"1", "true", "yes"}:
        raise SystemExit("--reload is forbidden in production (R21-082)")
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "uvicorn missing. Install with: pip install 'factor-engine[service]'"
        ) from exc
    uvicorn.run("service.app:create_app", factory=True, host=args.host, port=args.port, reload=args.reload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
