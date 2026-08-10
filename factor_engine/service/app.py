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

from runtime.endpoint_policy import (
    EndpointExecutionPolicy,
    ProductionPolicyConflictError,
)

# R21 service modules
from service.errors import (
    ServiceError,
    classify_exception,
    redact_config,
    sanitize_message,
)
from service.jobstore import (
    JobPhase,
    JobRecord,
    JobStatus,
    JobStore,
    check_single_process_workers,
)
from service.models import (
    ComputeRequest,
    MaterializeRequest,
    ValidatedFactorRequest,
    estimate_factor_cost,
    size_budget,
)
from service.observability import (
    METRICS,
    bind_log_context,
    info,
    new_execution_id,
    trace_span,
    warning,
)
from service.policies import RuntimeFeaturePolicy, resolve_ambient_run_mode
from service.queue import (
    BoundedJobQueue,
    JobCancelledError,
    JobDeadlineExceeded,
    check_job_alive,
    set_phase,
)
from service.security import (
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
# Global service state
# ---------------------------------------------------------------------------

STORE = JobStore()
_EXECUTOR_MAX_WORKERS = max(1, int(os.environ.get("FACTOR_ENGINE_SERVICE_MAX_WORKERS", "4") or 4))
EXECUTOR = ThreadPoolExecutor(
    max_workers=_EXECUTOR_MAX_WORKERS,
    thread_name_prefix="factor-engine-job",
)
QUEUE = BoundedJobQueue(
    max_running=_EXECUTOR_MAX_WORKERS,
    timeout_default=float(os.environ.get("FACTOR_ENGINE_SERVICE_JOB_TIMEOUT", "300") or "300"),
)
QUEUE.start(STORE)

FEATURE_POLICY = RuntimeFeaturePolicy.from_env()
SOURCE_POLICY = ApprovedSourcePolicy.from_env()


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
    from service.security import get_principal_registry

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
    from service.security import get_principal_registry

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
    from api.dsl_parser import DSLParseError, parse_expr
    from api.mining_integration import validate_factor_engine_dsl, validate_production_dsl

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
                ok, msg = validator(formula)
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
    from cleaned_operators.registry import OperatorRegistry

    generation = OperatorRegistry.version()
    universe = tuple(model.universe or [])
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
        catalog_generations=(generation,),
        complexity_budget=str(budget["max_ast_nodes"]) if "max_ast_nodes" in budget else "default",
        source_profile=model.approved_source_profile_id,
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

    run_mode = endpoint_policy.value if endpoint_policy.is_production else model.run_mode.value
    execution = {
        "validated": vr.to_payload(),
        "run_mode": run_mode,
        "backend": model.backend or "auto",
        "config_path": model.config_path,
        "formula_schema_version": model.formula_schema_version,
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
    return execution, vr.digest()


def _production_source_config(model: ComputeRequest) -> dict[str, Any]:
    """R21-020..026: resolve a production inline source from an approved profile."""
    from service.security import resolve_source_profile

    return resolve_source_profile({"approved_source_profile_id": model.approved_source_profile_id}, policy=SOURCE_POLICY)


def _validate_config_path_sources(config_path: str, *, production: bool) -> None:
    """R21-027: config-file sources pass the same approved-source policy."""
    if not production:
        return
    try:
        from runtime.config_runtime import load_config
    except Exception:
        return
    config = load_config(config_path)
    SOURCE_POLICY.validate_source(config.data_source, production=True)


# ---------------------------------------------------------------------------
# Job execution workers
# ---------------------------------------------------------------------------


def _execute_config_path(job: JobRecord, config_path: str) -> dict[str, Any]:
    from runtime.engine import FactorEngine

    authorized = _authorized_config_path(config_path)
    _validate_config_path_sources(str(authorized), production=job.endpoint_policy == "production")
    return FactorEngine.run_from_config(
        str(authorized),
        execution_policy=job.endpoint_policy or None,
    )


def _execute_inline(job: JobRecord, execution: dict[str, Any]) -> dict[str, Any]:
    from api.dsl_parser import parse_factor
    from backend.factory import build_backend
    from runtime.engine import FactorEngine
    from storage.factory import build_data_source

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
    source = build_data_source(source_cfg)
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
    cancel, sanitized errors (R21-055..060, 087..090)."""
    from service.errors import json_dumps_redacted

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
    try:
        set_phase(job, STORE, JobPhase.VALIDATING)
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
    except JobCancelledError:
        job.status = JobStatus.CANCELLED
        job.error_code = "JOB_CANCELLED"
        job.finished_at = _utc_now()
        METRICS.incr("job_cancel", labels={"job_type": job.job_type})
        info("job.cancelled", run_id=job.run_id)
    except JobDeadlineExceeded:
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
    from runtime.engine import FactorEngine

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


def _summarize_result(payload: dict[str, Any], job: JobRecord) -> dict[str, Any]:
    mode = payload.get("mode")
    out = payload.get("out") or {}
    if mode == "run_from_config":
        summary: dict[str, Any] = {"mode": mode}
        if isinstance(out, dict):
            result = out.get("result")
            if result is not None:
                try:
                    summary["rows"] = int(getattr(result, "shape", [0])[0])
                    job.artifacts["result_preview"] = str(result.head(5).to_string())
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
                job.artifacts["result_preview"] = str(result.head(5).to_string())
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
    timeout = float(payload.get("timeout_seconds") or QUEUE.timeout_default)
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
        execution_policy_digest=FEATURE_POLICY.digest(),
        timeout_seconds=timeout,
        attempt=1,
        phase=JobPhase.VALIDATING,
        cost_estimate=dict(execution.get("cost_estimate") or {}),
    )
    import datetime as _dt
    import time as _time

    job.deadline_at = (_dt.datetime.now(_dt.timezone.utc) + _dt.timedelta(seconds=timeout)).isoformat()
    job.deadline_monotonic = _time.monotonic() + timeout  # R21-277
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
    from service.release_blockers import evaluate_blockers

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

    from service.preflight import production_preflight

    PREFLIGHT: dict[str, Any] = {"ok": True, "checks": {}}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
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
            info("service.preflight.blocked")
            raise RuntimeError("production preflight failed; refusing to serve")
        yield
        # R21-080..082: graceful shutdown — drain queued, cancel running, close.
        info("service.shutdown.start")
        QUEUE.drain(timeout=float(os.environ.get("FACTOR_ENGINE_SERVICE_DRAIN_TIMEOUT", "30")))
        EXECUTOR.shutdown(wait=True)
        STORE.close()
        info("service.shutdown.done")

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
    from api.operator_registry import build_dsl_allowlist

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
