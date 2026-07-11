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

from runtime.config import FactorEngineConfig, load_config
from runtime.engine import FactorEngine


def _safe_name(name: str, max_len: int = 100) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z_\u4e00-\u9fff\-]+", "_", name).strip("_")
    return cleaned[:max_len] if len(cleaned) > max_len else cleaned


def _prepare_output_root(output_root: str | Path, fallback_name: str) -> tuple[Path, Path, Path]:
    """准备输出目录结构。``output_root`` 必须传入，不提供默认值。"""
    root = Path(output_root).resolve()
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


def _count_plan_nodes(plan: Any) -> int:
    inputs = getattr(plan, "inputs", []) or []
    return 1 + sum(_count_plan_nodes(item) for item in inputs)


def _build_analysis_summary(analysis: Any) -> dict[str, Any]:
    return {
        "lookback": int(getattr(analysis, "lookback", 0)),
        "has_ts_op": bool(getattr(analysis, "has_ts_op", False)),
        "has_cs_op": bool(getattr(analysis, "has_cs_op", False)),
        "referenced_columns": sorted(str(item) for item in getattr(analysis, "referenced_columns", set())),
        "ir_op": getattr(getattr(analysis, "ir", None), "op", None),
    }


def _build_plan_summary(plan: Any) -> dict[str, Any]:
    return {
        "root_op": getattr(plan, "op", None),
        "node_count": _count_plan_nodes(plan),
    }


def _build_result_summary(result: Any, preview_rows: int) -> dict[str, Any]:
    row_count = int(len(result)) if hasattr(result, "__len__") else None
    non_null_count = None
    if hasattr(result, "notna"):
        non_null_count = int(result.notna().sum())

    preview: list[dict[str, Any]] = []
    if hasattr(result, "head") and hasattr(result, "reset_index"):
        preview_frame = result.head(preview_rows).reset_index()
        columns = list(preview_frame.columns)
        if columns:
            preview_frame = preview_frame.rename(columns={columns[-1]: "value"})
        preview = [_json_safe(item) for item in preview_frame.to_dict(orient="records")]

    index_names = None
    if hasattr(result, "index") and hasattr(result.index, "names"):
        index_names = [name if name is None else str(name) for name in result.index.names]

    dtype = getattr(result, "dtype", None)
    return {
        "type": type(result).__name__,
        "row_count": row_count,
        "non_null_count": non_null_count,
        "dtype": None if dtype is None else str(dtype),
        "index_names": index_names,
        "preview": preview,
    }


def _write_yaml_snapshot(config: FactorEngineConfig, target_path: Path) -> str:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(yaml.safe_dump(asdict(config), allow_unicode=True, sort_keys=False), encoding="utf-8")
    return str(target_path)


def _copy_snapshot(config_path: Path, target_path: Path) -> str:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    return str(target_path)


def _failed_result(*, config_name: str, factor_name: str | None, mode: str, exc: Exception) -> dict[str, Any]:
    return {
        "config_name": config_name,
        "factor_name": factor_name,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "status": "failed",
        "mode": mode,
        "analysis": None,
        "plan": None,
        "result": None,
        "materialization": None,
        "errors": [str(exc)],
        "traceback": traceback.format_exc(),
    }


def _execute_config(
    config: FactorEngineConfig,
    *,
    config_name: str,
    materialize: bool | None,
    preview_rows: int,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    engine, factor = FactorEngine.from_loaded_config(config)
    should_materialize = config.materialization is not None if materialize is None else materialize
    mode = "materialize" if should_materialize else "run"

    if should_materialize:
        if output_path is None:
            raise ValueError(
                "output_path is required when materialize=True. "
                "Provide an explicit output_path to specify where factor values are written."
            )
        materialization = config.materialization
        output = engine.materialize(
            factor,
            output_path=output_path,
            factor_id=materialization.factor_id if materialization is not None else None,
            author=materialization.author if materialization is not None else None,
            frequency=materialization.frequency if materialization is not None else None,
            description=materialization.description if materialization is not None else None,
            expression=(
                materialization.expression if materialization is not None else config.factor.expr
            ),
        )
    else:
        output = engine.run(factor)

    item: dict[str, Any] = {
        "config_name": config_name,
        "factor_name": factor.name,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "status": "success",
        "mode": mode,
        "analysis": _build_analysis_summary(output["analysis"]),
        "plan": _build_plan_summary(output["plan"]),
        "result": _build_result_summary(output["result"], preview_rows=preview_rows),
        "materialization": None,
        "errors": [],
    }
    if "materialization" in output:
        item["materialization"] = _json_safe(output["materialization"])
    return item


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
        "materialized_runs": sum(
            1 for item in results if item.get("status") == "success" and item.get("mode") == "materialize"
        ),
        "result_json_files": [item.get("result_json_path") for item in results],
    }


def run_pipeline(
    config: FactorEngineConfig,
    *,
    config_path: str | Path | None = None,
    output_root: str | Path,
    materialize: bool | None = None,
    preview_rows: int = 5,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """运行单个因子配置。

    Args:
        config: 因子引擎配置对象。
        config_path: 配置文件路径（用于快照）。
        output_root: **必填**。pipeline 输出根目录（日志、摘要 JSON）。
        materialize: 是否落盘。None=跟随配置。
        preview_rows: 结果预览行数。
        output_path: 因子值落盘根路径。若提供，因子值写入
            ``{output_path}/{factor_id}/data/{YYYY-MM-DD}.parquet``。
    """
    root, results_root, _ = _prepare_output_root(output_root, config.factor.name)
    config_name = Path(config_path).stem if config_path is not None else config.factor.name

    try:
        item = _execute_config(
            config,
            config_name=config_name,
            materialize=materialize,
            preview_rows=preview_rows,
            output_path=output_path,
        )
    except Exception as exc:
        mode = "materialize" if (materialize or (materialize is None and config.materialization is not None)) else "run"
        item = _failed_result(config_name=config_name, factor_name=config.factor.name, mode=mode, exc=exc)

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
    config: FactorEngineConfig,
    *,
    config_path: str | Path | None = None,
    output_root: str | Path,
    materialize: bool | None = None,
    preview_rows: int = 5,
) -> dict[str, Any]:
    return run_pipeline(
        config,
        config_path=config_path,
        output_root=output_root,
        materialize=materialize,
        preview_rows=preview_rows,
    )


def run_from_config(
    config_path: str | Path,
    *,
    output_root: str | Path,
    materialize: bool | None = None,
    preview_rows: int = 5,
) -> dict[str, Any]:
    config_path = Path(config_path)
    root, results_root, _ = _prepare_output_root(output_root, config_path.stem)

    try:
        config = load_config(config_path)
    except Exception as exc:
        item = _failed_result(config_name=config_path.stem, factor_name=None, mode="run", exc=exc)
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
        materialize=materialize,
        preview_rows=preview_rows,
    )


def run_config_directory(
    config_dir: str | Path,
    *,
    pattern: str = "*.yaml",
    output_root: str | Path,
    materialize: bool | None = None,
    preview_rows: int = 5,
    stop_on_error: bool = False,
) -> dict[str, Any]:
    config_dir = Path(config_dir)
    config_paths = sorted(path for path in config_dir.glob(pattern) if path.is_file())
    if not config_paths:
        raise FileNotFoundError(f"No config files matched under {config_dir} with pattern {pattern!r}")

    root, results_root, snapshots_root = _prepare_output_root(output_root, config_dir.name)
    results: list[dict[str, Any]] = []

    for config_path in config_paths:
        config_name = config_path.stem
        snapshot_target = snapshots_root / f"{_safe_name(config_name)}.yaml"
        _copy_snapshot(config_path, snapshot_target)

        try:
            config = load_config(config_path)
            item = _execute_config(
                config,
                config_name=config_name,
                materialize=materialize,
                preview_rows=preview_rows,
            )
        except Exception as exc:
            mode = "materialize" if materialize else "run"
            item = _failed_result(config_name=config_name, factor_name=None, mode=mode, exc=exc)
            if stop_on_error:
                _write_result_json(results_root, config_name, item)
                results.append(item)
                break

        _write_result_json(results_root, config_name, item)
        results.append(item)

    summary = _build_summary(output_root=root, results=results, config_source=str(config_dir))
    summary["pattern"] = pattern
    run_summary_path = root / "run_summary.json"
    run_summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "summary": summary,
        "results": results,
        "output_root": str(root),
        "run_summary_path": str(run_summary_path),
        "config_snapshot_root": str(snapshots_root),
    }


__all__ = ["run", "run_pipeline", "run_from_config", "run_config_directory"]