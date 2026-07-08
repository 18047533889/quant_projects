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
from runtime.config_runtime import resolve_materialize_kwargs, resolve_run_kwargs
from runtime.engine import FactorEngine
from runtime.metrics_export import (
    enrich_pipeline_summary,
    push_otlp_http,
    write_otlp_metrics_file,
    write_pipeline_metrics_file,
    write_prometheus_metrics_file,
)
from runtime.task_queue import shard_config_paths
from logging_utils import get_logger
from workspace_paths import workspace_data_root


def _safe_name(name: str, max_len: int = 100) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z_\u4e00-\u9fff\-]+", "_", name).strip("_")
    return cleaned[:max_len] if len(cleaned) > max_len else cleaned


def _default_output_root(name: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return workspace_data_root() / "factor_engine" / "pipeline_runs" / f"{stamp}_{_safe_name(name)}"


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


logger = get_logger("pipeline")


def _execute_config_with_retries(
    config: FactorEngineConfig,
    *,
    config_name: str,
    materialize: bool | None,
    preview_rows: int,
    incremental: bool = False,
    dq_check: bool = False,
    dq_strict: bool = True,
    input_dq_check: bool = False,
    input_dq_strict: bool = True,
    max_retries: int = 0,
    resume_materialize: bool = False,
    write_target: str | None = None,
    preserve_invalid_rows: bool | None = None,
    since: str | None = None,
    end_date: str | None = None,
    lookback_extra: int | None = None,
    recompute_tail_bars: int | None = None,
) -> dict[str, Any]:
    attempts = max(0, int(max_retries)) + 1
    last_exc: Exception | None = None
    for attempt in range(attempts):
        try:
            return _execute_config(
                config,
                config_name=config_name,
                materialize=materialize,
                preview_rows=preview_rows,
                incremental=incremental,
                dq_check=dq_check,
                dq_strict=dq_strict,
                input_dq_check=input_dq_check,
                input_dq_strict=input_dq_strict,
                resume_materialize=resume_materialize,
                write_target=write_target,
                preserve_invalid_rows=preserve_invalid_rows,
                since=since,
                end_date=end_date,
                lookback_extra=lookback_extra,
                recompute_tail_bars=recompute_tail_bars,
            )
        except Exception as exc:
            last_exc = exc
            if attempt + 1 >= attempts:
                break
            logger.warning(
                "config '%s' 第 %d/%d 次失败，重试: %s",
                config_name,
                attempt + 1,
                attempts,
                exc,
            )
    assert last_exc is not None
    raise last_exc


def _execute_config(
    config: FactorEngineConfig,
    *,
    config_name: str,
    materialize: bool | None,
    preview_rows: int,
    incremental: bool = False,
    dq_check: bool = False,
    dq_strict: bool = True,
    input_dq_check: bool = False,
    input_dq_strict: bool = True,
    resume_materialize: bool = False,
    write_target: str | None = None,
    preserve_invalid_rows: bool | None = None,
    since: str | None = None,
    end_date: str | None = None,
    lookback_extra: int | None = None,
    recompute_tail_bars: int | None = None,
) -> dict[str, Any]:
    engine, factor = FactorEngine.from_loaded_config(config)
    should_materialize = config.materialization is not None if materialize is None else materialize
    if incremental and not should_materialize:
        logger.warning(
            "config '%s' 请求 incremental 但未启用 materialize，已忽略 incremental",
            config_name,
        )
        incremental = False
    if incremental and should_materialize:
        mode = "materialize_incremental"
    elif should_materialize:
        mode = "materialize"
    else:
        mode = "run"

    run_opts = resolve_run_kwargs(
        config,
        cli_input_dq_check=True if input_dq_check else None,
        cli_input_dq_strict=input_dq_strict if input_dq_check else None,
    )

    if should_materialize:
        mat_opts = resolve_materialize_kwargs(
            config,
            cli_dq_check=True if dq_check else None,
            cli_dq_strict=dq_strict if dq_check else None,
            cli_input_dq_check=True if input_dq_check else None,
            cli_input_dq_strict=input_dq_strict if input_dq_check else None,
            write_target_override=write_target,
            preserve_invalid_rows_override=preserve_invalid_rows,
            since_override=since,
            end_date_override=end_date,
            lookback_extra_override=lookback_extra,
            recompute_tail_bars_override=recompute_tail_bars,
            resume_materialize_override=resume_materialize,
        )
        common_kwargs = dict(
            factor_id=mat_opts.factor_id,
            author=mat_opts.author,
            frequency=mat_opts.frequency,
            description=mat_opts.description,
            expression=mat_opts.expression,
            auto_warmup=mat_opts.auto_warmup,
            trim_warmup=mat_opts.trim_warmup,
            market=mat_opts.market,
            dq_check=mat_opts.dq_check,
            dq_strict=mat_opts.dq_strict,
            dq_thresholds=mat_opts.dq_thresholds,
            input_dq_check=mat_opts.input_dq_check,
            input_dq_strict=mat_opts.input_dq_strict,
            input_dq_thresholds=mat_opts.input_dq_thresholds,
            preserve_invalid_rows=mat_opts.preserve_invalid_rows,
            value_dtype=mat_opts.value_dtype,
            data_source_config=mat_opts.data_source_config,
            pit_enforce=mat_opts.pit_enforce,
            pit_forbid_forward_fill=mat_opts.pit_forbid_forward_fill,
        )
        target = str(mat_opts.write_target).lower()
        if incremental:
            mat_kwargs = {
                **common_kwargs,
                "lake_root": mat_opts.lake_root,
                "write_target": mat_opts.write_target,
                "resume_materialize": mat_opts.resume_materialize,
                "isolate_partition_failures": mat_opts.isolate_partition_failures,
                "since": mat_opts.since,
                "end_date": mat_opts.end_date,
                "lookback_extra": mat_opts.lookback_extra,
                "recompute_tail_bars": mat_opts.recompute_tail_bars,
                "ch_ensure_table": mat_opts.ch_ensure_table,
            }
            if target in ("clickhouse", "staging_clickhouse"):
                mat_kwargs.update(
                    {
                        "clickhouse_table": mat_opts.clickhouse_table,
                        "ch_host": mat_opts.clickhouse_host,
                        "ch_port": mat_opts.clickhouse_port,
                        "ch_database": mat_opts.clickhouse_database,
                        "ch_username": mat_opts.clickhouse_username,
                        "ch_password": mat_opts.clickhouse_password,
                        "ch_secure": mat_opts.clickhouse_secure,
                    }
                )
            output = engine.materialize_incremental(factor, **mat_kwargs)
        elif target == "clickhouse":
            output = engine.materialize(
                factor,
                **common_kwargs,
                lake_root=mat_opts.lake_root,
                write_target="clickhouse",
                isolate_partition_failures=mat_opts.isolate_partition_failures,
                clickhouse_table=mat_opts.clickhouse_table,
                ch_host=mat_opts.clickhouse_host,
                ch_port=mat_opts.clickhouse_port,
                ch_database=mat_opts.clickhouse_database,
                ch_username=mat_opts.clickhouse_username,
                ch_password=mat_opts.clickhouse_password,
                ch_secure=mat_opts.clickhouse_secure,
            )
        elif target == "staging_clickhouse":
            output = engine.materialize(
                factor,
                **common_kwargs,
                lake_root=mat_opts.lake_root,
                write_target="staging_clickhouse",
                isolate_partition_failures=mat_opts.isolate_partition_failures,
                clickhouse_table=mat_opts.clickhouse_table,
                ch_host=mat_opts.clickhouse_host,
                ch_port=mat_opts.clickhouse_port,
                ch_database=mat_opts.clickhouse_database,
                ch_username=mat_opts.clickhouse_username,
                ch_password=mat_opts.clickhouse_password,
                ch_secure=mat_opts.clickhouse_secure,
            )
        else:
            output = engine.materialize(
                factor,
                **common_kwargs,
                lake_root=mat_opts.lake_root,
                write_target=mat_opts.write_target,
                resume_materialize=mat_opts.resume_materialize,
                isolate_partition_failures=mat_opts.isolate_partition_failures,
            )
    else:
        output = engine.run(
            factor,
            auto_warmup=run_opts.auto_warmup,
            trim_warmup=run_opts.trim_warmup,
            market=run_opts.market,
            input_dq_check=run_opts.input_dq_check,
            input_dq_strict=run_opts.input_dq_strict,
            input_dq_thresholds=run_opts.input_dq_thresholds,
            pit_enforce=run_opts.pit_enforce,
            pit_forbid_forward_fill=run_opts.pit_forbid_forward_fill,
        )

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
        "incremental": _json_safe(output.get("incremental")),
        "input_dq": _json_safe(output.get("input_dq")),
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


def _finalize_run_summary(
    *,
    output_root: Path,
    results: list[dict[str, Any]],
    config_source: str | None,
    extra: dict[str, Any] | None = None,
    otlp_endpoint: str | None = None,
) -> dict[str, Any]:
    summary = _build_summary(
        output_root=output_root,
        results=results,
        config_source=config_source,
    )
    if extra:
        summary.update(extra)
    summary = enrich_pipeline_summary(summary, results)
    run_summary_path = output_root / "run_summary.json"
    run_summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    write_pipeline_metrics_file(summary, output_root / "metrics.json")
    write_prometheus_metrics_file(summary, output_root / "metrics.prom")
    write_otlp_metrics_file(summary, output_root / "metrics.otlp.json")
    if otlp_endpoint:
        summary["otlp_push"] = push_otlp_http(summary, otlp_endpoint)
    return summary


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
    output_root: str | Path | None = None,
    materialize: bool | None = None,
    preview_rows: int = 5,
    incremental: bool = False,
    dq_check: bool = False,
    dq_strict: bool = True,
    input_dq_check: bool = False,
    input_dq_strict: bool = True,
    max_retries: int = 0,
    resume_materialize: bool = False,
    write_target: str | None = None,
    preserve_invalid_rows: bool | None = None,
    since: str | None = None,
    end_date: str | None = None,
    lookback_extra: int | None = None,
    recompute_tail_bars: int | None = None,
) -> dict[str, Any]:
    root, results_root, _ = _prepare_output_root(output_root, config.factor.name)
    config_name = Path(config_path).stem if config_path is not None else config.factor.name

    try:
        item = _execute_config_with_retries(
            config,
            config_name=config_name,
            materialize=materialize,
            preview_rows=preview_rows,
            incremental=incremental,
            dq_check=dq_check,
            dq_strict=dq_strict,
            input_dq_check=input_dq_check,
            input_dq_strict=input_dq_strict,
            max_retries=max_retries,
            resume_materialize=resume_materialize,
            write_target=write_target,
            preserve_invalid_rows=preserve_invalid_rows,
            since=since,
            end_date=end_date,
            lookback_extra=lookback_extra,
            recompute_tail_bars=recompute_tail_bars,
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

    summary = _finalize_run_summary(
        output_root=root,
        results=[item],
        config_source=None if config_path is None else str(config_path),
    )
    run_summary_path = root / "run_summary.json"

    return {
        "summary": summary,
        "results": [item],
        "output_root": str(root),
        "run_summary_path": str(run_summary_path),
        "metrics_path": str(root / "metrics.json"),
        "config_snapshot": config_snapshot,
    }


def run(
    config: FactorEngineConfig,
    *,
    config_path: str | Path | None = None,
    output_root: str | Path | None = None,
    materialize: bool | None = None,
    preview_rows: int = 5,
    incremental: bool = False,
    dq_check: bool = False,
    dq_strict: bool = True,
    input_dq_check: bool = False,
    input_dq_strict: bool = True,
    max_retries: int = 0,
    resume_materialize: bool = False,
    write_target: str | None = None,
    preserve_invalid_rows: bool | None = None,
    since: str | None = None,
    end_date: str | None = None,
    lookback_extra: int | None = None,
    recompute_tail_bars: int | None = None,
) -> dict[str, Any]:
    return run_pipeline(
        config,
        config_path=config_path,
        output_root=output_root,
        materialize=materialize,
        preview_rows=preview_rows,
        incremental=incremental,
        dq_check=dq_check,
        dq_strict=dq_strict,
        input_dq_check=input_dq_check,
        input_dq_strict=input_dq_strict,
        max_retries=max_retries,
        resume_materialize=resume_materialize,
        write_target=write_target,
        preserve_invalid_rows=preserve_invalid_rows,
        since=since,
        end_date=end_date,
        lookback_extra=lookback_extra,
        recompute_tail_bars=recompute_tail_bars,
    )


def run_from_config(
    config_path: str | Path,
    *,
    output_root: str | Path | None = None,
    materialize: bool | None = None,
    preview_rows: int = 5,
    incremental: bool = False,
    dq_check: bool = False,
    dq_strict: bool = True,
    input_dq_check: bool = False,
    input_dq_strict: bool = True,
    max_retries: int = 0,
    resume_materialize: bool = False,
    profile: str | None = None,
    write_target: str | None = None,
    preserve_invalid_rows: bool | None = None,
    since: str | None = None,
    end_date: str | None = None,
    lookback_extra: int | None = None,
    recompute_tail_bars: int | None = None,
) -> dict[str, Any]:
    config_path = Path(config_path)
    root, results_root, _ = _prepare_output_root(output_root, config_path.stem)

    try:
        config = load_config(config_path, profile=profile)
    except Exception as exc:
        item = _failed_result(config_name=config_path.stem, factor_name=None, mode="run", exc=exc)
        _write_result_json(results_root, config_path.stem, item)
        config_snapshot = _copy_snapshot(config_path, root / "config_snapshot.yaml")
        summary = _finalize_run_summary(
            output_root=root,
            results=[item],
            config_source=str(config_path),
        )
        run_summary_path = root / "run_summary.json"
        return {
            "summary": summary,
            "results": [item],
            "output_root": str(root),
            "run_summary_path": str(run_summary_path),
            "metrics_path": str(root / "metrics.json"),
            "config_snapshot": config_snapshot,
        }

    return run_pipeline(
        config,
        config_path=config_path,
        output_root=root,
        materialize=materialize,
        preview_rows=preview_rows,
        incremental=incremental,
        dq_check=dq_check,
        dq_strict=dq_strict,
        input_dq_check=input_dq_check,
        input_dq_strict=input_dq_strict,
        max_retries=max_retries,
        resume_materialize=resume_materialize,
        write_target=write_target,
        preserve_invalid_rows=preserve_invalid_rows,
        since=since,
        end_date=end_date,
        lookback_extra=lookback_extra,
        recompute_tail_bars=recompute_tail_bars,
    )


def _run_single_config_file(
    config_path: Path,
    *,
    profile: str | None,
    materialize: bool | None,
    preview_rows: int,
    incremental: bool,
    dq_check: bool,
    dq_strict: bool,
    input_dq_check: bool,
    input_dq_strict: bool,
    max_retries: int,
    resume_materialize: bool,
    write_target: str | None = None,
    preserve_invalid_rows: bool | None = None,
    since: str | None = None,
    end_date: str | None = None,
    lookback_extra: int | None = None,
    recompute_tail_bars: int | None = None,
) -> tuple[str, dict[str, Any]]:
    config_name = config_path.stem
    try:
        config = load_config(config_path, profile=profile)
        item = _execute_config_with_retries(
            config,
            config_name=config_name,
            materialize=materialize,
            preview_rows=preview_rows,
            incremental=incremental,
            dq_check=dq_check,
            dq_strict=dq_strict,
            input_dq_check=input_dq_check,
            input_dq_strict=input_dq_strict,
            max_retries=max_retries,
            resume_materialize=resume_materialize,
            write_target=write_target,
            preserve_invalid_rows=preserve_invalid_rows,
            since=since,
            end_date=end_date,
            lookback_extra=lookback_extra,
            recompute_tail_bars=recompute_tail_bars,
        )
    except Exception as exc:
        mode = "materialize" if materialize else "run"
        item = _failed_result(config_name=config_name, factor_name=None, mode=mode, exc=exc)
    return config_name, item


def run_config_directory(
    config_dir: str | Path,
    *,
    pattern: str = "*.yaml",
    output_root: str | Path | None = None,
    materialize: bool | None = None,
    preview_rows: int = 5,
    stop_on_error: bool = False,
    incremental: bool = False,
    dq_check: bool = False,
    dq_strict: bool = True,
    input_dq_check: bool = False,
    input_dq_strict: bool = True,
    max_retries: int = 0,
    resume_materialize: bool = False,
    profile: str | None = None,
    n_jobs: int = 1,
    shard_index: int | None = None,
    shard_count: int = 1,
    otlp_endpoint: str | None = None,
    write_target: str | None = None,
    preserve_invalid_rows: bool | None = None,
    since: str | None = None,
    end_date: str | None = None,
    lookback_extra: int | None = None,
    recompute_tail_bars: int | None = None,
) -> dict[str, Any]:
    config_dir = Path(config_dir)
    all_config_paths = sorted(path for path in config_dir.glob(pattern) if path.is_file())
    if not all_config_paths:
        raise FileNotFoundError(f"No config files matched under {config_dir} with pattern {pattern!r}")

    if shard_index is not None:
        config_paths = shard_config_paths(
            all_config_paths,
            shard_index=int(shard_index),
            shard_count=int(shard_count),
        )
    else:
        config_paths = all_config_paths

    root, results_root, snapshots_root = _prepare_output_root(output_root, config_dir.name)

    for config_path in config_paths:
        snapshot_target = snapshots_root / f"{_safe_name(config_path.stem)}.yaml"
        _copy_snapshot(config_path, snapshot_target)

    workers = max(1, int(n_jobs))
    run_kwargs = dict(
        profile=profile,
        materialize=materialize,
        preview_rows=preview_rows,
        incremental=incremental,
        dq_check=dq_check,
        dq_strict=dq_strict,
        input_dq_check=input_dq_check,
        input_dq_strict=input_dq_strict,
        max_retries=max_retries,
        resume_materialize=resume_materialize,
        write_target=write_target,
        preserve_invalid_rows=preserve_invalid_rows,
        since=since,
        end_date=end_date,
        lookback_extra=lookback_extra,
        recompute_tail_bars=recompute_tail_bars,
    )

    if workers > 1:
        try:
            from joblib import Parallel, delayed

            pairs = Parallel(n_jobs=workers, backend="threading")(
                delayed(_run_single_config_file)(config_path, **run_kwargs)
                for config_path in config_paths
            )
        except ImportError:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            pairs = []
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(_run_single_config_file, config_path, **run_kwargs): config_path
                    for config_path in config_paths
                }
                for future in as_completed(futures):
                    pairs.append(future.result())
            pairs.sort(key=lambda item: item[0])
    else:
        pairs = [_run_single_config_file(config_path, **run_kwargs) for config_path in config_paths]

    results: list[dict[str, Any]] = []
    for config_name, item in pairs:
        if stop_on_error and item.get("status") == "failed":
            _write_result_json(results_root, config_name, item)
            results.append(item)
            break
        _write_result_json(results_root, config_name, item)
        results.append(item)

    summary = _finalize_run_summary(
        output_root=root,
        results=results,
        config_source=str(config_dir),
        extra={
            "pattern": pattern,
            "shard_index": shard_index,
            "shard_count": shard_count,
            "configs_selected": len(config_paths),
            "configs_total_in_dir": len(all_config_paths),
        },
        otlp_endpoint=otlp_endpoint,
    )
    run_summary_path = root / "run_summary.json"

    return {
        "summary": summary,
        "results": results,
        "output_root": str(root),
        "run_summary_path": str(run_summary_path),
        "metrics_path": str(root / "metrics.json"),
        "prometheus_metrics_path": str(root / "metrics.prom"),
        "otlp_metrics_path": str(root / "metrics.otlp.json"),
        "config_snapshot_root": str(snapshots_root),
    }


__all__ = ["run", "run_pipeline", "run_from_config", "run_config_directory"]