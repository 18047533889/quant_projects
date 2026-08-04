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

import json
import os
import threading
import traceback
import uuid
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
    status: JobStatus = "submitted"
    submitted_at: str = field(default_factory=_utc_now)
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    error: Optional[str] = None
    request: Dict[str, Any] = field(default_factory=dict)
    result_summary: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, Any] = field(default_factory=dict)

    def to_public(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "service": self.service,
            "requested_by": self.requested_by,
            "job_type": self.job_type,
            "status": self.status,
            "submitted_at": self.submitted_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "result_summary": self.result_summary,
            "artifacts": self.artifacts,
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

    def create(self, job: JobRecord) -> JobRecord:
        with self._lock:
            self._jobs[job.run_id] = job
            self._write_manifest(job)
        return job

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


def list_operators() -> Dict[str, Any]:
    from api.operator_registry import build_dsl_allowlist

    names = sorted(build_dsl_allowlist().keys())
    return {"count": len(names), "operators": names}


def validate_spec(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Validate DSL string and/or factor-engine YAML-like JSON spec (no execution)."""
    from api.dsl_parser import DSLParseError, parse_expr
    from api.mining_integration import validate_factor_engine_dsl

    errors: List[str] = []
    warnings: List[str] = []
    checked: Dict[str, Any] = {}

    formula = (
        payload.get("formula")
        or payload.get("dsl")
        or ((payload.get("factor") or {}) if isinstance(payload.get("factor"), dict) else {}).get("expr")
        or ""
    )
    formula = str(formula or "").strip()
    if formula:
        ok, msg = validate_factor_engine_dsl(formula, surface=str(payload.get("surface") or "compat"))
        checked["dsl"] = {"ok": bool(ok), "message": msg, "formula": formula}
        if not ok:
            errors.append(f"dsl: {msg}")
        else:
            try:
                parse_expr(formula)
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
            engine = FactorEngine(backend=backend, data_source=source)
            factor = parse_factor(formula, name=name)
            out = engine.run(factor)
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
        kwargs = {}
        for k in ("lake_root", "factor_id", "author", "frequency", "description", "expression"):
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
    run_id = uuid.uuid4().hex
    job = JobRecord(
        run_id=run_id,
        requested_by=str(payload.get("requested_by") or payload.get("user") or "anonymous"),
        job_type=job_type,
        request=payload,
    )
    STORE.create(job)
    target = _run_compute if job_type == "compute" else _run_materialize
    if sync or bool(payload.get("sync")) or os.environ.get("FACTOR_ENGINE_SERVICE_SYNC", "").lower() in {
        "1",
        "true",
        "yes",
    }:
        target(job)
    else:
        threading.Thread(target=target, args=(job,), daemon=True).start()
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
    def health() -> dict[str, str]:
        return {"status": "ok", "service": "factor_engine"}

    @app.get("/factor-engine/operators")
    def operators() -> dict[str, Any]:
        return list_operators()

    @app.post("/factor-engine/validate-spec")
    async def validate(request: Request) -> dict[str, Any]:
        payload = await request.json()
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="JSON object required")
        return validate_spec(payload)

    @app.post("/factor-engine/jobs/compute")
    async def jobs_compute(request: Request) -> dict[str, Any]:
        payload = await request.json()
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="JSON object required")
        return submit_job("compute", payload, sync=bool(payload.get("sync")))

    @app.post("/factor-engine/jobs/materialize")
    async def jobs_materialize(request: Request) -> dict[str, Any]:
        payload = await request.json()
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="JSON object required")
        return submit_job("materialize", payload, sync=bool(payload.get("sync")))

    @app.get("/factor-engine/jobs/{run_id}")
    def job_status(run_id: str) -> dict[str, Any]:
        job = STORE.get(run_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"unknown run_id: {run_id}")
        return job.to_public()

    @app.get("/factor-engine/jobs/{run_id}/artifacts")
    def job_artifacts(run_id: str) -> dict[str, Any]:
        job = STORE.get(run_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"unknown run_id: {run_id}")
        return {"run_id": run_id, "status": job.status, "artifacts": job.artifacts}

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
