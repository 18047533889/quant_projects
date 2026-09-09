"""Config-driven Riskfolio optimizer tuning with canonical VectorBT evaluation.

The pipeline keeps the optimizer and backtester on their supported public
interfaces:

1. ``riskfolio_qs.cli.run_from_config`` produces one audited target-position
   directory per optimizer grid point.
2. VectorBT's frozen ``standard_accurate_benchmark_v1`` profile evaluates the
   positions with one canonical accurate backtest configuration.
3. The public ``python -m vectorbt_qs accurate-benchmark`` command generates
   complete delivery reports for the best trials.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from itertools import product
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

import pandas as pd
import yaml


BACKTEST_WEIGHT_ZERO_TOLERANCE = 1e-6


class PipelineConfigError(ValueError):
    """Raised when the tuning pipeline YAML is invalid."""


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise PipelineConfigError(f"{name} must be a mapping")
    return dict(value)


def _reject_unknown(
    mapping: Mapping[str, Any],
    allowed: set[str],
    name: str,
) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise PipelineConfigError(f"{name} contains unknown keys: {unknown}")


def _resolve_path(value: Any, base_dir: Path, name: str) -> Path:
    if value is None or not str(value).strip():
        raise PipelineConfigError(f"{name} is required")
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _load_yaml(path: Path, name: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{name} does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return _require_mapping(payload, name)


def _write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            _yaml_safe(dict(payload)),
            handle,
            sort_keys=False,
            allow_unicode=True,
        )


def _yaml_safe(value: Any) -> Any:
    """Convert pandas/numpy/path scalars into SafeDumper-compatible values."""

    if isinstance(value, Mapping):
        return {str(key): _yaml_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_yaml_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, pd.Timedelta):
        return str(value)
    item = getattr(value, "item", None)
    if callable(item):
        try:
            converted = item()
            if converted is value or type(converted) is type(value):
                return str(converted)
            return _yaml_safe(converted)
        except (TypeError, ValueError):
            pass
    return value


def _expand_grid(grid: Mapping[str, Any]) -> list[dict[str, Any]]:
    if not grid:
        raise PipelineConfigError("riskfolio.grid must not be empty")
    keys: list[str] = []
    values: list[list[Any]] = []
    for raw_key, raw_candidates in grid.items():
        key = str(raw_key).strip()
        if not key:
            raise PipelineConfigError("riskfolio.grid contains an empty key")
        if isinstance(raw_candidates, (str, bytes)) or not isinstance(
            raw_candidates, Sequence
        ):
            raise PipelineConfigError(f"riskfolio.grid.{key} must be a list")
        candidates = list(raw_candidates)
        if not candidates:
            raise PipelineConfigError(
                f"riskfolio.grid.{key} must contain at least one value"
            )
        keys.append(key)
        values.append(candidates)
    return [
        dict(zip(keys, combination))
        for combination in product(*values)
    ]


def _normalize_base_riskfolio_paths(
    config: dict[str, Any],
    base_dir: Path,
) -> dict[str, Any]:
    """Make paths independent of the generated trial-config location."""

    normalized = deepcopy(config)
    adapter = _require_mapping(normalized.get("adapter", {}), "base.adapter")
    if adapter.get("risk_root"):
        adapter["risk_root"] = str(
            _resolve_path(adapter["risk_root"], base_dir, "base.adapter.risk_root")
        )
    normalized["adapter"] = adapter

    inputs = _require_mapping(normalized.get("inputs", {}), "base.inputs")
    alpha = _require_mapping(inputs.get("alpha", {}), "base.inputs.alpha")
    alpha["path"] = str(
        _resolve_path(alpha.get("path"), base_dir, "base.inputs.alpha.path")
    )
    inputs["alpha"] = alpha
    normalized["inputs"] = inputs

    run = _require_mapping(normalized.get("run", {}), "base.run")
    for key in ("mapping_path", "parameter_path"):
        if run.get(key):
            run[key] = str(
                _resolve_path(run[key], base_dir, f"base.run.{key}")
            )
    normalized["run"] = run
    return normalized


def _validate_base_riskfolio_config(config: Mapping[str, Any]) -> None:
    run = _require_mapping(config.get("run", {}), "base.run")
    optimizer = str(run.get("optimizer_name", "")).strip()
    if not optimizer:
        raise PipelineConfigError(
            "base.run.optimizer_name is required; scenario routing is not "
            "supported by the generic tuning pipeline because every trial "
            "must resolve one explicit parameter set"
        )
    if run.get("scenario") is not None:
        raise PipelineConfigError(
            "base.run.scenario cannot be combined with explicit optimizer tuning"
        )
    _require_mapping(config.get("adapter", {}), "base.adapter")


def _validate_optimizer_overrides(
    base_config: Mapping[str, Any],
    fixed_overrides: Mapping[str, Any],
    trials: Sequence[Mapping[str, Any]],
) -> None:
    """Fail before a long run if any generated runtime override is invalid."""

    from riskfolio_qs.optimizers.parameter_store import ParameterStore

    run = _require_mapping(base_config.get("run", {}), "base.run")
    existing = _require_mapping(
        run.get("runtime_overrides", {}),
        "base.run.runtime_overrides",
    )
    unsupported_cost_overrides = {
        "default_linear_cost_bps",
        "default_impact_cost",
        "impact_cost_penalty",
        "enable_impact_cost",
    }
    supplied = set(existing) | set(fixed_overrides)
    for trial in trials:
        supplied.update(trial)
    unsupported = sorted(supplied.intersection(unsupported_cost_overrides))
    if unsupported:
        raise PipelineConfigError(
            "riskfolio production CLI does not accept cost-data overrides: "
            f"{unsupported}"
        )

    parameter_path = run.get("parameter_path")
    store = ParameterStore(
        config_path=None if not parameter_path else Path(str(parameter_path))
    )
    optimizer = str(run["optimizer_name"])
    baseline, _ = store.resolve(optimizer, existing)
    requested_keys = set(fixed_overrides)
    for trial in trials:
        requested_keys.update(trial)
    undefined = sorted(requested_keys - set(baseline))
    if undefined:
        raise PipelineConfigError(
            f"optimizer {optimizer!r} does not define parameters {undefined}; "
            "remove no-op grid dimensions or provide a compatible parameter_path"
        )
    for number, trial in enumerate(trials, start=1):
        overrides = deepcopy(existing)
        overrides.update(deepcopy(dict(fixed_overrides)))
        overrides.update(deepcopy(dict(trial)))
        try:
            store.resolve(optimizer, overrides)
        except (KeyError, TypeError, ValueError) as exc:
            raise PipelineConfigError(
                f"trial_{number:04d} has invalid optimizer overrides: {exc}"
            ) from exc


def load_pipeline_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path).expanduser().resolve()
    base_dir = path.parent
    config = _load_yaml(path, "pipeline config")
    _reject_unknown(
        config,
        {
            "config_version",
            "riskfolio",
            "backtest",
            "selection",
            "execution",
            "reports",
        },
        "pipeline config",
    )
    if int(config.get("config_version", 1)) != 1:
        raise PipelineConfigError("config_version must be 1")

    riskfolio = _require_mapping(config.get("riskfolio"), "riskfolio")
    _reject_unknown(
        riskfolio,
        {"base_config", "fixed_overrides", "grid"},
        "riskfolio",
    )
    base_config_path = _resolve_path(
        riskfolio.get("base_config"),
        base_dir,
        "riskfolio.base_config",
    )
    base_config = _normalize_base_riskfolio_paths(
        _load_yaml(base_config_path, "riskfolio.base_config"),
        base_config_path.parent,
    )
    _validate_base_riskfolio_config(base_config)
    fixed_overrides = _require_mapping(
        riskfolio.get("fixed_overrides", {}),
        "riskfolio.fixed_overrides",
    )
    grid = _require_mapping(riskfolio.get("grid"), "riskfolio.grid")
    collisions = sorted(set(fixed_overrides).intersection(grid))
    if collisions:
        raise PipelineConfigError(
            "parameters cannot appear in both fixed_overrides and grid: "
            f"{collisions}"
        )
    trials = _expand_grid(grid)
    _validate_optimizer_overrides(base_config, fixed_overrides, trials)

    backtest = _require_mapping(config.get("backtest"), "backtest")
    _reject_unknown(
        backtest,
        {"profile", "data_root", "barra_root", "start", "end"},
        "backtest",
    )
    profile = str(
        backtest.get("profile", "standard_accurate_benchmark_v1")
    )
    if profile != "standard_accurate_benchmark_v1":
        raise PipelineConfigError(
            "backtest.profile must be standard_accurate_benchmark_v1"
        )
    data_root = _resolve_path(
        backtest.get("data_root"),
        base_dir,
        "backtest.data_root",
    )
    barra_root = (
        None
        if backtest.get("barra_root") is None
        else _resolve_path(
            backtest.get("barra_root"),
            base_dir,
            "backtest.barra_root",
        )
    )
    if not data_root.is_dir():
        raise FileNotFoundError(f"backtest.data_root is not a directory: {data_root}")
    if barra_root is not None and not barra_root.is_dir():
        raise FileNotFoundError(
            f"backtest.barra_root is not a directory: {barra_root}"
        )

    selection = _require_mapping(config.get("selection", {}), "selection")
    _reject_unknown(selection, {"metric", "ascending"}, "selection")
    metric = str(selection.get("metric", "Information Ratio")).strip()
    if not metric:
        raise PipelineConfigError("selection.metric must not be empty")

    execution = _require_mapping(config.get("execution"), "execution")
    _reject_unknown(
        execution,
        {
            "output_directory",
            "max_trials",
            "workers",
            "resume",
            "overwrite",
            "on_trial_error",
        },
        "execution",
    )
    output_directory = _resolve_path(
        execution.get("output_directory"),
        base_dir,
        "execution.output_directory",
    )
    max_trials = int(execution.get("max_trials", 128))
    if max_trials <= 0:
        raise PipelineConfigError("execution.max_trials must be positive")
    if len(trials) > max_trials:
        raise PipelineConfigError(
            f"grid expands to {len(trials)} trials, exceeding "
            f"execution.max_trials={max_trials}"
        )
    workers = execution.get("workers", 1)
    if isinstance(workers, bool) or not isinstance(workers, int) or workers <= 0:
        raise PipelineConfigError("execution.workers must be a positive integer")
    on_trial_error = str(execution.get("on_trial_error", "raise")).lower()
    if on_trial_error not in {"raise", "continue"}:
        raise PipelineConfigError(
            "execution.on_trial_error must be raise or continue"
        )

    reports = _require_mapping(config.get("reports", {}), "reports")
    _reject_unknown(reports, {"enabled", "top_n", "on_error"}, "reports")
    report_top_n = int(reports.get("top_n", 3))
    if report_top_n < 0:
        raise PipelineConfigError("reports.top_n must be non-negative")
    report_on_error = str(reports.get("on_error", "raise")).lower()
    if report_on_error not in {"raise", "continue"}:
        raise PipelineConfigError("reports.on_error must be raise or continue")
    reports_enabled = bool(reports.get("enabled", True))
    if reports_enabled and report_top_n > 0 and barra_root is None:
        raise PipelineConfigError(
            "backtest.barra_root is required when complete reports are enabled"
        )

    return {
        "config_path": path,
        "base_config_path": base_config_path,
        "base_config": base_config,
        "fixed_overrides": fixed_overrides,
        "trials": trials,
        "backtest": {
            "profile": profile,
            "data_root": data_root,
            "barra_root": barra_root,
            "start": (
                None
                if backtest.get("start") is None
                else str(backtest["start"])
            ),
            "end": (
                None
                if backtest.get("end") is None
                else str(backtest["end"])
            ),
        },
        "selection": {
            "metric": metric,
            "ascending": bool(selection.get("ascending", False)),
        },
        "execution": {
            "output_directory": output_directory,
            "max_trials": max_trials,
            "workers": workers,
            "resume": bool(execution.get("resume", True)),
            "overwrite": bool(execution.get("overwrite", False)),
            "on_trial_error": on_trial_error,
        },
        "reports": {
            "enabled": reports_enabled,
            "top_n": report_top_n,
            "on_error": report_on_error,
        },
    }


def _build_trial_config(
    base_config: Mapping[str, Any],
    fixed_overrides: Mapping[str, Any],
    trial_overrides: Mapping[str, Any],
    riskfolio_output: Path,
) -> dict[str, Any]:
    config = deepcopy(dict(base_config))
    run = _require_mapping(config.get("run", {}), "base.run")
    existing = _require_mapping(
        run.get("runtime_overrides", {}),
        "base.run.runtime_overrides",
    )
    existing.update(deepcopy(dict(fixed_overrides)))
    existing.update(deepcopy(dict(trial_overrides)))
    run["runtime_overrides"] = existing
    config["run"] = run
    config["output"] = {
        "directory": str(riskfolio_output),
        "overwrite": False,
    }
    return config


def _optimization_diagnostics(
    weights: pd.DataFrame,
    summary: pd.DataFrame,
) -> dict[str, Any]:
    values = weights.fillna(0.0)
    diagnostics: dict[str, Any] = {
        "optimizer_date_count": int(len(values)),
        "optimizer_asset_count": int(len(values.columns)),
        "optimizer_mean_holding_count": float(
            values.abs().gt(BACKTEST_WEIGHT_ZERO_TOLERANCE).sum(axis=1).mean()
        ),
        "optimizer_max_single_weight": float(values.max(axis=1).max()),
    }
    for column, output_name, aggregation in (
        ("turnover", "optimizer_mean_turnover", "mean"),
        ("turnover", "optimizer_max_turnover", "max"),
        (
            "predicted_tracking_error_annual",
            "optimizer_mean_predicted_te_annual",
            "mean",
        ),
        (
            "max_constraint_violation",
            "optimizer_max_constraint_violation",
            "max",
        ),
    ):
        if column not in summary:
            continue
        numeric = pd.to_numeric(summary[column], errors="coerce")
        value = numeric.mean() if aggregation == "mean" else numeric.max()
        diagnostics[output_name] = float(value)
    return diagnostics


def _prepare_backtest_weights(weights: pd.DataFrame) -> pd.DataFrame:
    """Remove economically zero solver dust without altering audit artifacts."""
    prepared = weights.astype(float).copy()
    prepared = prepared.mask(
        prepared.abs() <= BACKTEST_WEIGHT_ZERO_TOLERANCE,
        0.0,
    )
    return prepared


def _run_backtest(
    weights: pd.DataFrame,
    *,
    data_root: Path,
    start: str | None,
    end: str | None,
) -> dict[str, Any]:
    from vectorbt_qs.mvp.engine import (
        run_backtest_batch,
        standard_accurate_benchmark_v1_base_config,
        standard_accurate_benchmark_v1_grid,
    )

    result = run_backtest_batch(
        "ashare",
        weights,
        standard_accurate_benchmark_v1_grid(),
        base_config=standard_accurate_benchmark_v1_base_config(),
        start=start,
        end=end,
        max_runs=1,
        on_error="raise",
        workers=1,
        data_root=str(data_root),
    )
    performance = result.performance_table()
    if len(performance) != 1:
        raise RuntimeError(
            "standard_accurate_benchmark_v1 must return exactly one run"
        )
    row = performance.iloc[0].to_dict()
    if "status" in row:
        row["backtest_status"] = row.pop("status")
    if "error" in row:
        row["backtest_error"] = row.pop("error")
    return row


def _run_trial(
    spec: Mapping[str, Any],
    trial_number: int,
    trial_overrides: Mapping[str, Any],
) -> dict[str, Any]:
    from riskfolio_qs.cli import run_from_config

    trial_id = f"trial_{trial_number:04d}"
    output_root = Path(spec["execution"]["output_directory"])
    config_path = output_root / "trial_configs" / f"{trial_id}.yaml"
    riskfolio_output = output_root / "riskfolio_runs" / trial_id
    stats_path = output_root / "trial_stats" / f"{trial_id}.csv"
    status_path = output_root / "trial_status" / f"{trial_id}.yaml"
    resume = bool(spec["execution"]["resume"])
    overwrite = bool(spec["execution"]["overwrite"])

    if resume and stats_path.is_file() and not overwrite:
        cached = pd.read_csv(stats_path).iloc[0].to_dict()
        if str(cached.get("pipeline_status", "")) == "success":
            print(f"[{trial_id}] resume: loading {stats_path}")
            return cached
        print(f"[{trial_id}] resume: retrying previous failed trial")

    trial_config = _build_trial_config(
        spec["base_config"],
        spec["fixed_overrides"],
        trial_overrides,
        riskfolio_output,
    )
    _write_yaml(config_path, trial_config)
    target_path = riskfolio_output / "target_positions.parquet"
    summary_path = riskfolio_output / "summary.parquet"

    if resume and target_path.is_file() and summary_path.is_file() and not overwrite:
        print(f"[{trial_id}] resume: reusing optimizer artifacts")
        weights = pd.read_parquet(target_path)
        summary = pd.read_parquet(summary_path)
    else:
        print(f"[{trial_id}] optimizing with {dict(trial_overrides)}")
        # A failed/interrupted CLI run can leave resolved configuration files
        # without the target/summary artifacts required for a valid resume.
        # Rebuild only this trial directory; successful sibling trials remain
        # untouched and continue to be loaded from their cached stats.
        retry_incomplete = (
            resume
            and riskfolio_output.is_dir()
            and not (target_path.is_file() and summary_path.is_file())
        )
        completed = run_from_config(
            config_path,
            output_dir=riskfolio_output,
            overwrite=overwrite or retry_incomplete,
        )
        weights = completed.output.target_positions
        summary = completed.output.summary

    backtest_weights = _prepare_backtest_weights(weights)
    backtest_path = riskfolio_output / "backtest_positions.parquet"
    backtest_weights.to_parquet(backtest_path)
    print(f"[{trial_id}] running standard_accurate_benchmark_v1")
    performance = _run_backtest(
        backtest_weights,
        data_root=spec["backtest"]["data_root"],
        start=spec["backtest"]["start"],
        end=spec["backtest"]["end"],
    )
    row = {
        "trial_id": trial_id,
        **dict(trial_overrides),
        "pipeline_status": "success",
        "pipeline_error": "",
        **_optimization_diagnostics(weights, summary),
        **performance,
    }
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(
        stats_path,
        index=False,
        encoding="utf-8-sig",
    )
    _write_yaml(
        status_path,
        {
            "trial_id": trial_id,
            "status": "success",
            "parameters": dict(trial_overrides),
            "riskfolio_output": str(riskfolio_output),
            "backtest_positions": str(backtest_path),
            "stats": str(stats_path),
        },
    )
    return row


def _failed_trial_row(
    trial_number: int,
    trial_overrides: Mapping[str, Any],
    exc: Exception,
) -> dict[str, Any]:
    return {
        "trial_id": f"trial_{trial_number:04d}",
        **dict(trial_overrides),
        "pipeline_status": "failed",
        "pipeline_error": f"{type(exc).__name__}: {exc}",
    }


def _write_failed_trial(
    output_root: Path,
    row: Mapping[str, Any],
) -> None:
    trial_id = str(row["trial_id"])
    stats_path = output_root / "trial_stats" / f"{trial_id}.csv"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([dict(row)]).to_csv(
        stats_path,
        index=False,
        encoding="utf-8-sig",
    )
    _write_yaml(
        output_root / "trial_status" / f"{trial_id}.yaml",
        {
            "trial_id": trial_id,
            "status": "failed",
            "error": row["pipeline_error"],
            "parameters": {
                key: value
                for key, value in row.items()
                if key
                not in {
                    "trial_id",
                    "pipeline_status",
                    "pipeline_error",
                }
            },
        },
    )


def _rank_results(
    results: pd.DataFrame,
    *,
    metric: str,
    ascending: bool,
) -> pd.DataFrame:
    successful = results.loc[
        results["pipeline_status"].eq("success")
    ].copy()
    if successful.empty:
        raise RuntimeError("no successful trials are available for ranking")
    if metric not in successful:
        raise PipelineConfigError(
            f"selection metric {metric!r} is not in backtest output; "
            f"available columns: {sorted(map(str, successful.columns))}"
        )
    successful[metric] = pd.to_numeric(successful[metric], errors="coerce")
    successful = successful.loc[successful[metric].notna()]
    if successful.empty:
        raise RuntimeError(
            f"all successful trials have NaN selection metric {metric!r}"
        )
    ranked = successful.sort_values(
        metric,
        ascending=ascending,
        kind="stable",
    ).reset_index(drop=True)
    ranked.insert(0, "rank", range(1, len(ranked) + 1))
    return ranked


def _run_final_reports(
    spec: Mapping[str, Any],
    ranked: pd.DataFrame,
) -> None:
    reports = spec["reports"]
    if not reports["enabled"] or reports["top_n"] == 0:
        return
    if spec["backtest"]["barra_root"] is None:
        raise PipelineConfigError(
            "backtest.barra_root is required for accurate-benchmark reports"
        )
    output_root = Path(spec["execution"]["output_directory"])
    report_root = output_root / "final_reports"
    report_root.mkdir(parents=True, exist_ok=True)
    top = ranked.head(int(reports["top_n"]))
    env = os.environ.copy()
    data_root = str(spec["backtest"]["data_root"])
    env["VECTORBT_QS_ASHARE_DATA_ROOT"] = data_root
    env["ASHARE_PARQUET_ROOT"] = data_root

    for _, row in top.iterrows():
        trial_id = str(row["trial_id"])
        positions = (
            output_root
            / "riskfolio_runs"
            / trial_id
            / "backtest_positions.parquet"
        )
        command = [
            sys.executable,
            "-m",
            "vectorbt_qs",
            "accurate-benchmark",
            "--positions",
            str(positions),
            "--barra-root",
            str(spec["backtest"]["barra_root"]),
            "--output-root",
            str(report_root),
        ]
        if spec["backtest"]["start"]:
            command.extend(["--start", str(spec["backtest"]["start"])])
        if spec["backtest"]["end"]:
            command.extend(["--end", str(spec["backtest"]["end"])])
        print(f"[report] {trial_id}: {' '.join(command)}")
        try:
            subprocess.run(command, check=True, env=env)
        except subprocess.CalledProcessError:
            if reports["on_error"] == "raise":
                raise
            print(f"[report] failed but continuing: {trial_id}", file=sys.stderr)


def run_pipeline(spec: Mapping[str, Any]) -> pd.DataFrame:
    output_root = Path(spec["execution"]["output_directory"])
    output_root.mkdir(parents=True, exist_ok=True)
    local_data_root = str(spec["backtest"]["data_root"])
    os.environ["ASHARE_PARQUET_ROOT"] = local_data_root
    os.environ["VECTORBT_QS_ASHARE_DATA_ROOT"] = local_data_root
    # This pipeline is explicitly configured with a complete local market-data
    # root.  Keep workers offline instead of invoking an optional COS CLI.
    os.environ.setdefault("DATA_ACCESS_SKIP_COS_MIRROR", "1")
    _write_yaml(
        output_root / "resolved_pipeline_config.yaml",
        {
            "config_version": 1,
            "source_config": str(spec["config_path"]),
            "riskfolio": {
                "base_config": str(spec["base_config_path"]),
                "fixed_overrides": spec["fixed_overrides"],
                "grid": spec["trials"],
            },
            "backtest": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in spec["backtest"].items()
            },
            "selection": spec["selection"],
            "execution": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in spec["execution"].items()
            },
            "reports": spec["reports"],
        },
    )

    combined_path = output_root / "search_results.csv"
    jobs = list(enumerate(spec["trials"], start=1))
    rows_by_number: dict[int, dict[str, Any]] = {}
    workers = int(spec["execution"]["workers"])

    def record_result(
        trial_number: int,
        trial_overrides: Mapping[str, Any],
        *,
        row: dict[str, Any] | None = None,
        exc: Exception | None = None,
    ) -> None:
        if row is None:
            assert exc is not None
            row = _failed_trial_row(trial_number, trial_overrides, exc)
            _write_failed_trial(output_root, row)
            print(
                f"[{row['trial_id']}] failed: {row['pipeline_error']}",
                file=sys.stderr,
            )
        rows_by_number[trial_number] = row
        ordered_rows = [
            rows_by_number[number]
            for number in sorted(rows_by_number)
        ]
        pd.DataFrame(ordered_rows).to_csv(
            combined_path,
            index=False,
            encoding="utf-8-sig",
        )

    if workers == 1:
        for trial_number, trial_overrides in jobs:
            try:
                row = _run_trial(spec, trial_number, trial_overrides)
            except Exception as exc:
                record_result(
                    trial_number,
                    trial_overrides,
                    exc=exc,
                )
                if spec["execution"]["on_trial_error"] == "raise":
                    raise
            else:
                record_result(
                    trial_number,
                    trial_overrides,
                    row=row,
                )
    else:
        # Windows uses spawn, so every worker receives an independent config
        # and loads its own pandas/risk/market data. Keep numerical libraries
        # single-threaded by default to avoid process x BLAS oversubscription.
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
        print(
            f"[pipeline] running {len(jobs)} trials with {workers} processes"
        )
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_jobs = {
                executor.submit(
                    _run_trial,
                    spec,
                    trial_number,
                    trial_overrides,
                ): (trial_number, trial_overrides)
                for trial_number, trial_overrides in jobs
            }
            for future in as_completed(future_jobs):
                trial_number, trial_overrides = future_jobs[future]
                try:
                    row = future.result()
                except Exception as exc:
                    record_result(
                        trial_number,
                        trial_overrides,
                        exc=exc,
                    )
                    if spec["execution"]["on_trial_error"] == "raise":
                        for pending in future_jobs:
                            pending.cancel()
                        raise
                else:
                    record_result(
                        trial_number,
                        trial_overrides,
                        row=row,
                    )

    rows = [
        rows_by_number[number]
        for number in sorted(rows_by_number)
    ]
    results = pd.DataFrame(rows)
    ranked = _rank_results(
        results,
        metric=spec["selection"]["metric"],
        ascending=spec["selection"]["ascending"],
    )
    ranked.to_csv(
        output_root / "ranked_results.csv",
        index=False,
        encoding="utf-8-sig",
    )
    _write_yaml(
        output_root / "best_trials.yaml",
        {
            "selection_metric": spec["selection"]["metric"],
            "ascending": spec["selection"]["ascending"],
            "trials": ranked.head(int(spec["reports"]["top_n"])).to_dict(
                orient="records"
            ),
        },
    )
    _run_final_reports(spec, ranked)
    return ranked


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Tune an explicit riskfolio_qs optimizer and evaluate every trial "
            "with VectorBT's frozen single accurate benchmark."
        )
    )
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        help="pipeline YAML path",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate configuration and print the expanded grid only",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        spec = load_pipeline_config(args.config)
        if args.dry_run:
            print(f"config={spec['config_path']}")
            print(f"base_config={spec['base_config_path']}")
            print(f"output={spec['execution']['output_directory']}")
            print(f"profile={spec['backtest']['profile']}")
            print(f"trials={len(spec['trials'])}")
            print(f"workers={spec['execution']['workers']}")
            for number, trial in enumerate(spec["trials"], start=1):
                print(f"trial_{number:04d}: {trial}")
            return 0
        ranked = run_pipeline(spec)
    except Exception as exc:
        print(f"pipeline failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    best = ranked.iloc[0]
    metric = spec["selection"]["metric"]
    print(
        f"pipeline completed: best={best['trial_id']}, "
        f"{metric}={best[metric]}, "
        f"output={spec['execution']['output_directory']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
