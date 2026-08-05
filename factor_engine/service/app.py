"""Minimal HTTP service adapter around FactorEngine (library core unchanged).

Endpoints follow ``IT_HANDOFF.md`` §9:
- GET  /health
- GET  /factor-engine/operators
- POST /factor-engine/validate-spec
- POST /factor-engine/jobs/compute
- POST /factor-engine/jobs/materialize
- GET  /factor-engine/jobs/{run_id}
- GET  /factor-engine/jobs/{run_id}/artifacts
"""

import hmac
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

JobStatus = Literal["submitted", "running", "succeeded", "failed"]
JobType = Literal["compute", "materialize"]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class JobRecord:
    run_id: str
    service: str = "factor_engine"
    requested_by: str = ""
    job_type: JobType = "compute"
    idempotency_key: Optional[str] = None
    request_metadata: Dict[str, Any] = field(default_factory=dict)
    status: JobStatus = "submitted"
    submitted_at: str = field(default_factory=_utc_now)
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    error: Optional[str] = None
    request: Dict[str, Any] = field(default_factory=dict)
    result_summary: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, Any] = field(default_factory=dict)

    def to_public(self) -> Dict[str, Any]:
        public_artifacts = {
            key: value for key, value in self.artifacts.items()
            if key not in {"traceback", "manifest_path", "lake_root", "config_path"}
        }
        return {
            "run_id": self.run_id,
            "service": self.service,
            "requested_by": self.requested_by,
            "job_type": self.job_type,
            "idempotency_key": self.idempotency_key,
            "request_metadata": self.request_metadata,
            "status": self.status,
            "submitted_at": self.submitted_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "result_summary": self.result_summary,
            "artifacts": public_artifacts,
        }


class JobStore:
    """Process-local job registry with optional JSON manifests on disk."""

    def __init__(self, root: Optional[Path] = None) -> None:
        self.root = Path(
            root
            or os.environ.get("FACTOR_ENGINE_SERVICE_ROOT")
            or (Path.cwd() / ".factor_engine_service")
        )
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._jobs: Dict[str, JobRecord] = {}
        self._idempotency_index: Dict[str, str] = {}
        self._restore_manifests()

    def _restore_manifests(self) -> None:
        manifest_root = self.root / "manifests"
        if not manifest_root.exists():
            return
        for path in sorted(manifest_root.glob("*.json")):
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                job = JobRecord(
                    run_id=str(raw["run_id"]),
                    service=str(raw.get("service") or "factor_engine"),
                    requested_by=str(raw.get("requested_by") or ""),
                    job_type=str(raw.get("job_type") or "compute"),
                    idempotency_key=raw.get("idempotency_key"),
                    request_metadata=dict(raw.get("request_metadata") or {}),
                    status=str(raw.get("status") or "failed"),
                    submitted_at=str(raw.get("submitted_at") or _utc_now()),
                    started_at=raw.get("started_at"),
                    finished_at=raw.get("finished_at"),
                    error=raw.get("error"),
                    result_summary=dict(raw.get("result_summary") or {}),
                    artifacts=dict(raw.get("artifacts") or {}),
                )
                self._jobs[job.run_id] = job
                if job.idempotency_key:
                    self._idempotency_index[job.idempotency_key] = job.run_id
            except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
                continue

    def create(self, job: JobRecord) -> JobRecord:
        with self._lock:
            if job.idempotency_key:
                existing = self._idempotency_index.get(job.idempotency_key)
                if existing and existing in self._jobs:
                    return self._jobs[existing]
                self._idempotency_index[job.idempotency_key] = job.run_id
            self._jobs[job.run_id] = job
            self._write_manifest(job)
        return job

    def get_by_idempotency_key(self, key: str) -> Optional[JobRecord]:
        with self._lock:
            run_id = self._idempotency_index.get(str(key))
            return self._jobs.get(run_id) if run_id else None

    def get(self, run_id: str) -> Optional[JobRecord]:
        with self._lock:
            return self._jobs.get(run_id)

    def update(self, job: JobRecord) -> None:
        with self._lock:
            self._jobs[job.run_id] = job
            self._write_manifest(job)

    def _write_manifest(self, job: JobRecord) -> None:
        path = self.root / "manifests" / f"{job.run_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(job.to_public(), ensure_ascii=False, indent=2), encoding="utf-8")
        job.artifacts["manifest_path"] = str(path)


STORE = JobStore()
EXECUTOR = ThreadPoolExecutor(
    max_workers=max(1, int(os.environ.get("FACTOR_ENGINE_SERVICE_MAX_WORKERS", "4") or 4)),
    thread_name_prefix="factor-engine-job",
)


def _require_service_api_key(request) -> Dict[str, Any]:
    key = os.environ.get("FACTOR_ENGINE_SERVICE_API_KEY", "").strip()
    allow_open = os.environ.get("FACTOR_ENGINE_SERVICE_ALLOW_OPEN", "").lower() in {
        "1", "true", "yes", "on",
    }
    production = os.environ.get("QUANT_PRODUCTION_MODE", "").lower() in {
        "1", "true", "yes", "on",
    }
    if not key:
        if production and not allow_open:
            raise PermissionError("FACTOR_ENGINE_SERVICE_API_KEY is required in production")
        return {
            "identity": "anonymous",
            "authenticated": False,
            "request_id": str(request.headers.get("X-Request-ID") or ""),
        }
    supplied = str(request.headers.get("X-API-Key") or "")
    if not hmac.compare_digest(supplied, key):
        raise PermissionError("invalid API key")
    return {
        "identity": str(request.headers.get("X-Request-Identity") or "authenticated"),
        "authenticated": True,
        "request_id": str(request.headers.get("X-Request-ID") or ""),
        "user_agent": str(request.headers.get("User-Agent") or ""),
    }


def _authorized_config_path(raw: Any) -> Path:
    root = Path(os.environ.get("FACTOR_ENGINE_CONFIG_ROOT") or (STORE.root / "configs")).resolve()
    path = Path(str(raw))
    candidate = (root / path).resolve() if not path.is_absolute() else path.resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError("config_path is outside FACTOR_ENGINE_CONFIG_ROOT")
    if candidate.is_symlink():
        raise ValueError("config_path symlinks are forbidden")
    return candidate


def list_operators() -> Dict[str, Any]:
    from api.operator_registry import build_dsl_allowlist

    names = sorted(build_dsl_allowlist().keys())
    return {"count": len(names), "operators": names}


def validate_spec(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Validate DSL string and/or factor-engine YAML-like JSON spec (no execution)."""
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

    formula = (
        payload.get("formula")
        or payload.get("dsl")
        or ((payload.get("factor") or {}) if isinstance(payload.get("factor"), dict) else {}).get("expr")
        or ""
    )
    formula = str(formula or "").strip()
    surface = str(payload.get("surface") or "daily")
    if formula:
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


def _run_compute(job: JobRecord) -> None:
    from runtime.engine import FactorEngine

    req = job.request
    job.status = "running"
    job.started_at = _utc_now()
    STORE.update(job)
    try:
        config_path = req.get("config_path")
        if config_path:
            config_path = _authorized_config_path(config_path)
            out = FactorEngine.run_from_config(str(config_path))
            summary = {
                "mode": "run_from_config",
                "factor": getattr(out.get("factor"), "name", None) if isinstance(out, dict) else None,
                "keys": sorted(out.keys()) if isinstance(out, dict) else [],
            }
            artifacts: dict[str, Any] = {}
            if isinstance(out, dict) and out.get("result") is not None:
                preview = out["result"]
                try:
                    artifacts["result_preview"] = preview.head(5).to_string()
                    artifacts["result_rows"] = int(getattr(preview, "shape", [0])[0])
                except Exception:
                    artifacts["result_type"] = type(preview).__name__
            job.result_summary = summary
            job.artifacts.update(artifacts)
        else:
            # Inline DSL compute requires data_source in request
            from api.dsl_parser import parse_factor
            from backend.factory import build_backend
            from storage.factory import build_data_source

            formula = str(req.get("formula") or req.get("dsl") or (req.get("factor") or {}).get("expr") or "")
            name = str((req.get("factor") or {}).get("name") or req.get("name") or "inline_factor")
            if not formula:
                raise ValueError("compute job requires config_path or formula/dsl")
            if not isinstance(req.get("data_source"), dict):
                raise ValueError("inline compute requires data_source object")
            source = build_data_source(req["data_source"])
            backend_cfg = req.get("backend") if isinstance(req.get("backend"), dict) else {}
            backend_type = str(
                (backend_cfg or {}).get("type")
                or os.environ.get("FACTOR_ENGINE_OPERATOR_BACKEND")
                or "auto"
            )
            backend = build_backend(backend_type)
            engine = FactorEngine(
                backend=backend,
                data_source=source,
                run_mode=str(req.get("run_mode") or "research"),
            )
            factor = parse_factor(formula, name=name)
            production = str(req.get("run_mode") or "research") == "production"
            out = engine.run(
                factor,
                input_dq_check=production,
                auto_warmup=production,
                pit_enforce=production,
            )
            series = out["result"]
            job.result_summary = {
                "mode": "inline_dsl",
                "factor": name,
                "rows": int(getattr(series, "shape", [0])[0]) if series is not None else 0,
            }
            try:
                job.artifacts["result_preview"] = series.head(5).to_string()
            except Exception:
                pass
        job.status = "succeeded"
        job.finished_at = _utc_now()
    except Exception as exc:  # noqa: BLE001
        job.status = "failed"
        job.finished_at = _utc_now()
        job.error = f"{type(exc).__name__}: {exc}"
        job.artifacts["traceback"] = traceback.format_exc(limit=20)
    STORE.update(job)


def _run_materialize(job: JobRecord) -> None:
    from runtime.engine import FactorEngine

    req = job.request
    job.status = "running"
    job.started_at = _utc_now()
    STORE.update(job)
    try:
        config_path = req.get("config_path")
        if not config_path:
            raise ValueError("materialize job requires config_path")
        config_path = _authorized_config_path(config_path)
        kwargs = {}
        for k in ("factor_id", "author", "frequency", "description", "expression"):
            if req.get(k) is not None:
                kwargs[k] = req[k]
        out = FactorEngine.materialize_from_config(str(config_path), **kwargs)
        job.result_summary = {
            "mode": "materialize_from_config",
            "keys": sorted(out.keys()) if isinstance(out, dict) else [],
        }
        if isinstance(out, dict):
            for key in ("lake_root", "factor_id", "catalog_path", "output_path", "manifest_path"):
                if out.get(key) is not None:
                    job.artifacts[key] = str(out[key])
        job.status = "succeeded"
        job.finished_at = _utc_now()
    except Exception as exc:  # noqa: BLE001
        job.status = "failed"
        job.finished_at = _utc_now()
        job.error = f"{type(exc).__name__}: {exc}"
        job.artifacts["traceback"] = traceback.format_exc(limit=20)
    STORE.update(job)


def submit_job(job_type: JobType, payload: dict[str, Any], *, sync: bool = False) -> dict[str, Any]:
    idempotency_key = str(payload.get("idempotency_key") or "").strip() or None
    if idempotency_key:
        existing = STORE.get_by_idempotency_key(idempotency_key)
        if existing is not None:
            return {
                "run_id": existing.run_id,
                "status": existing.status,
                "submitted_at": existing.submitted_at,
                "job_type": existing.job_type,
                "idempotent": True,
            }
    run_id = uuid.uuid4().hex
    metadata = dict(payload.get("request_metadata") or {})
    job = JobRecord(
        run_id=run_id,
        requested_by=str(metadata.get("identity") or "anonymous"),
        job_type=job_type,
        idempotency_key=idempotency_key,
        request_metadata=metadata,
        request=payload,
    )
    job = STORE.create(job)
    target = _run_compute if job_type == "compute" else _run_materialize
    if sync or bool(payload.get("sync")) or os.environ.get("FACTOR_ENGINE_SERVICE_SYNC", "").lower() in {
        "1",
        "true",
        "yes",
    }:
        target(job)
    else:
        EXECUTOR.submit(target, job)
    return {
        "run_id": job.run_id,
        "status": job.status,
        "submitted_at": job.submitted_at,
        "job_type": job.job_type,
    }


def create_app():
    try:
        from fastapi import FastAPI, HTTPException, Request
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "HTTP service requires fastapi. Install with: pip install 'factor-engine[service]'"
        ) from exc

    app = FastAPI(
        title="Factor Engine Service",
        version="0.3.1",
        description="Thin HTTP adapter over HKUST FactorEngine Python API",
    )

    @app.get("/health")
    def health(request: Request) -> dict[str, str]:
        try:
            identity = _require_service_api_key(request)
        except PermissionError as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc
        return {"status": "ok", "service": "factor_engine", "identity": identity["identity"]}

    def _auth(request: Request) -> Dict[str, Any]:
        try:
            return _require_service_api_key(request)
        except PermissionError as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc

    def _submit_authenticated(request: Request, payload: dict[str, Any], job_type: JobType) -> dict[str, Any]:
        identity = _auth(request)
        payload = dict(payload)
        payload["request_metadata"] = identity
        payload["requested_by"] = identity["identity"]
        return submit_job(job_type, payload, sync=bool(payload.get("sync")))

    @app.get("/factor-engine/operators")
    def operators(request: Request) -> dict[str, Any]:
        _auth(request)
        return list_operators()

    @app.post("/factor-engine/validate-spec")
    async def validate(request: Request) -> dict[str, Any]:
        _auth(request)
        payload = await request.json()
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="JSON object required")
        return validate_spec(payload)

    @app.post("/factor-engine/research/compute")
    async def research_compute(request: Request) -> dict[str, Any]:
        payload = await request.json()
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="JSON object required")
        payload = dict(payload)
        payload["run_mode"] = "research"
        return _submit_authenticated(request, payload, "compute")

    @app.post("/factor-engine/production/compute")
    async def production_compute(request: Request) -> dict[str, Any]:
        payload = await request.json()
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="JSON object required")
        payload = dict(payload)
        payload["run_mode"] = "production"
        validation = validate_spec(payload)
        if not validation["ok"]:
            raise HTTPException(status_code=422, detail=validation["errors"])
        return _submit_authenticated(request, payload, "compute")

    @app.post("/factor-engine/production/materialize")
    async def production_materialize(request: Request) -> dict[str, Any]:
        payload = await request.json()
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="JSON object required")
        payload = dict(payload)
        payload["run_mode"] = "production"
        return _submit_authenticated(request, payload, "materialize")

    @app.post("/factor-engine/jobs/compute")
    async def jobs_compute(request: Request) -> dict[str, Any]:
        _auth(request)
        payload = await request.json()
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="JSON object required")
        return _submit_authenticated(request, payload, "compute")

    @app.post("/factor-engine/jobs/materialize")
    async def jobs_materialize(request: Request) -> dict[str, Any]:
        _auth(request)
        payload = await request.json()
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="JSON object required")
        return _submit_authenticated(request, payload, "materialize")

    @app.get("/factor-engine/jobs/{run_id}")
    def job_status(run_id: str, request: Request) -> dict[str, Any]:
        _auth(request)
        job = STORE.get(run_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"unknown run_id: {run_id}")
        return job.to_public()

    @app.get("/factor-engine/jobs/{run_id}/artifacts")
    def job_artifacts(run_id: str, request: Request) -> dict[str, Any]:
        _auth(request)
        job = STORE.get(run_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"unknown run_id: {run_id}")
        return {"run_id": run_id, "status": job.status, "artifacts": job.to_public()["artifacts"]}

    return app


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run Factor Engine HTTP service")
    parser.add_argument("--host", default=os.environ.get("FACTOR_ENGINE_SERVICE_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("FACTOR_ENGINE_SERVICE_PORT", "8088")))
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args(argv)
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
