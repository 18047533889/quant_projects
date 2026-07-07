from __future__ import annotations

import json
import re
import shutil
import traceback
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from factor_layer.factor_admission.admission import admit_evaluation_run
from factor_layer.factor_admission.config import FactorAdmissionConfig, load_config
from workspace_paths import default_factor_evaluation_root, workspace_data_root


def _safe_name(name: str, max_len: int = 100) -> str:
	cleaned = re.sub(r"[^0-9A-Za-z_\u4e00-\u9fff\-]+", "_", name).strip("_")
	return cleaned[:max_len] if len(cleaned) > max_len else cleaned


def _default_output_root(name: str) -> Path:
	stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
	return workspace_data_root() / "factor_admission" / "pipeline_runs" / f"{stamp}_{_safe_name(name)}"


def _prepare_output_root(output_root: str | Path | None, fallback_name: str) -> tuple[Path, Path, Path]:
	root = Path(output_root).resolve() if output_root is not None else _default_output_root(fallback_name)
	results_root = root / "results"
	snapshots_root = root / "config_snapshots"

	root.mkdir(parents=True, exist_ok=True)
	if results_root.exists():
		shutil.rmtree(results_root)
	if snapshots_root.exists():
		shutil.rmtree(snapshots_root)
	results_root.mkdir(parents=True, exist_ok=True)
	snapshots_root.mkdir(parents=True, exist_ok=True)
	return root, results_root, snapshots_root


def _json_safe(value: Any) -> Any:
	if value is None or isinstance(value, (str, int, float, bool)):
		return value
	if isinstance(value, dict):
		return {str(key): _json_safe(item) for key, item in value.items()}
	if isinstance(value, (list, tuple, set)):
		return [_json_safe(item) for item in value]
	return str(value)


def _resolve_run_dir(config: FactorAdmissionConfig) -> Path:
	evaluation_root = Path(
		config.source.evaluation_root or default_factor_evaluation_root(factor_lake_root=config.source.factor_lake_root)
	)
	return evaluation_root / config.meta.factor_id / config.meta.run_id


def _write_yaml_snapshot(config: FactorAdmissionConfig, target_path: Path) -> str:
	target_path.parent.mkdir(parents=True, exist_ok=True)
	target_path.write_text(yaml.safe_dump(asdict(config), allow_unicode=True, sort_keys=False), encoding="utf-8")
	return str(target_path)


def _copy_snapshot(config_path: Path, target_path: Path) -> str:
	target_path.parent.mkdir(parents=True, exist_ok=True)
	target_path.write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
	return str(target_path)


def _failed_result(
	*,
	config_name: str,
	factor_id: str | None,
	run_id: str | None,
	exc: Exception,
) -> dict[str, Any]:
	return {
		"config_name": config_name,
		"factor_id": factor_id,
		"run_id": run_id,
		"timestamp": datetime.now().isoformat(timespec="seconds"),
		"status": "failed",
		"operation": "admit",
		"decision": None,
		"approved": None,
		"decision_id": None,
		"primary_horizon": None,
		"reason": None,
		"diagnostics": [],
		"factor_status": None,
		"catalog_db_path": None,
		"evaluation_run_dir": None,
		"summary_path": None,
		"decision_file": None,
		"error": {
			"type": type(exc).__name__,
			"message": str(exc),
		},
		"traceback": traceback.format_exc(),
	}


def _execute_config(config: FactorAdmissionConfig, *, config_name: str) -> dict[str, Any]:
	decision_payload = admit_evaluation_run(config)
	run_dir = _resolve_run_dir(config)
	decision_file = run_dir / "admission_decision.json"

	return {
		"config_name": config_name,
		"factor_id": config.meta.factor_id,
		"run_id": config.meta.run_id,
		"timestamp": datetime.now().isoformat(timespec="seconds"),
		"status": "success",
		"operation": "admit",
		"decision": decision_payload.get("decision"),
		"approved": bool(decision_payload.get("approved")),
		"decision_id": decision_payload.get("decision_id"),
		"primary_horizon": decision_payload.get("primary_horizon"),
		"reason": decision_payload.get("reason"),
		"diagnostics": _json_safe(decision_payload.get("diagnostics", [])),
		"factor_status": _json_safe(decision_payload.get("status")),
		"catalog_db_path": str(Path(config.source.factor_lake_root) / "_catalog.sqlite"),
		"evaluation_run_dir": str(run_dir),
		"summary_path": str(run_dir / "summary.json"),
		"decision_file": str(decision_file) if decision_file.exists() else None,
		"error": None,
		"traceback": None,
	}


def _write_result_json(results_root: Path, name: str, item: dict[str, Any]) -> str:
	filename = f"{_safe_name(name)}.json"
	relative_path = Path("results") / filename
	item["result_json_path"] = str(relative_path)
	target_path = results_root / filename
	target_path.write_text(json.dumps(item, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
	return str(relative_path)


def _build_summary(*, output_root: Path, results: list[dict[str, Any]], config_source: str | None = None) -> dict[str, Any]:
	return {
		"timestamp": datetime.now().isoformat(timespec="seconds"),
		"output_root": str(output_root),
		"config_source": config_source,
		"configs_total": len(results),
		"configs_success": sum(1 for item in results if item.get("status") == "success"),
		"configs_failed": sum(1 for item in results if item.get("status") == "failed"),
		"decisions_approved": sum(1 for item in results if item.get("status") == "success" and item.get("approved") is True),
		"decisions_rejected": sum(1 for item in results if item.get("status") == "success" and item.get("approved") is False),
		"factor_ids": [item.get("factor_id") for item in results if item.get("factor_id")],
		"run_ids": [item.get("run_id") for item in results if item.get("run_id")],
		"result_json_files": [item.get("result_json_path") for item in results],
	}


def run_pipeline(
	config: FactorAdmissionConfig,
	*,
	config_path: str | Path | None = None,
	output_root: str | Path | None = None,
) -> dict[str, Any]:
	fallback_name = f"{config.meta.factor_id}_{config.meta.run_id}"
	root, results_root, _ = _prepare_output_root(output_root, fallback_name)
	config_name = Path(config_path).stem if config_path is not None else fallback_name

	try:
		item = _execute_config(config, config_name=config_name)
	except Exception as exc:
		item = _failed_result(
			config_name=config_name,
			factor_id=config.meta.factor_id,
			run_id=config.meta.run_id,
			exc=exc,
		)

	_write_result_json(results_root, config_name, item)

	snapshot_path = root / "config_snapshot.yaml"
	if config_path is not None:
		config_snapshot = _copy_snapshot(Path(config_path), snapshot_path)
	else:
		config_snapshot = _write_yaml_snapshot(config, snapshot_path)

	summary = _build_summary(output_root=root, results=[item], config_source=None if config_path is None else str(config_path))
	run_summary_path = root / "run_summary.json"
	run_summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

	return {
		"summary": summary,
		"results": [item],
		"output_root": str(root),
		"run_summary_path": str(run_summary_path),
		"config_snapshot": config_snapshot,
	}


def run(
	config: FactorAdmissionConfig,
	*,
	config_path: str | Path | None = None,
	output_root: str | Path | None = None,
) -> dict[str, Any]:
	return run_pipeline(
		config,
		config_path=config_path,
		output_root=output_root,
	)


def run_from_config(
	config_path: str | Path,
	*,
	output_root: str | Path | None = None,
) -> dict[str, Any]:
	config_path = Path(config_path)
	root, results_root, _ = _prepare_output_root(output_root, config_path.stem)

	try:
		config = load_config(config_path)
	except Exception as exc:
		item = _failed_result(config_name=config_path.stem, factor_id=None, run_id=None, exc=exc)
		_write_result_json(results_root, config_path.stem, item)
		config_snapshot = _copy_snapshot(config_path, root / "config_snapshot.yaml")
		summary = _build_summary(output_root=root, results=[item], config_source=str(config_path))
		run_summary_path = root / "run_summary.json"
		run_summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
		return {
			"summary": summary,
			"results": [item],
			"output_root": str(root),
			"run_summary_path": str(run_summary_path),
			"config_snapshot": config_snapshot,
		}

	return run_pipeline(
		config,
		config_path=config_path,
		output_root=root,
	)


def run_config_directory(
	config_dir: str | Path,
	*,
	pattern: str = "*.yaml",
	output_root: str | Path | None = None,
	stop_on_error: bool = False,
) -> dict[str, Any]:
	config_dir = Path(config_dir)
	config_paths = sorted(path for path in config_dir.glob(pattern) if path.is_file())
	if not config_paths:
		raise FileNotFoundError(f"No config files matched under {config_dir} with pattern {pattern!r}")

	root, results_root, snapshots_root = _prepare_output_root(output_root, config_dir.name)
	results: list[dict[str, Any]] = []
	config_snapshots: list[str] = []

	for config_path in config_paths:
		config_name = config_path.stem
		config_snapshots.append(_copy_snapshot(config_path, snapshots_root / f"{_safe_name(config_name)}.yaml"))
		try:
			config = load_config(config_path)
			item = _execute_config(config, config_name=config_name)
		except Exception as exc:
			item = _failed_result(config_name=config_name, factor_id=None, run_id=None, exc=exc)

		_write_result_json(results_root, config_name, item)
		results.append(item)
		if stop_on_error and item.get("status") == "failed":
			break

	summary = _build_summary(output_root=root, results=results, config_source=str(config_dir))
	summary["pattern"] = pattern
	run_summary_path = root / "run_summary.json"
	run_summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

	return {
		"summary": summary,
		"results": results,
		"output_root": str(root),
		"run_summary_path": str(run_summary_path),
		"config_snapshots": config_snapshots,
	}


__all__ = ["run", "run_pipeline", "run_from_config", "run_config_directory"]
