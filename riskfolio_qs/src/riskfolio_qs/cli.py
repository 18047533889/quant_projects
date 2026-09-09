"""Configuration-driven command line interface for riskfolio_qs."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import pandas as pd
import yaml

from . import __version__
from .adapters import (
    BarraPrecomputedAdapter,
    load_historical_market_returns,
    load_portfolio_inputs,
)
from .adapters.data_access_inputs import DEFAULT_LINEAR_COST_BPS
from .core.contracts import InputBundle, OptimizationContext, OutputBundle
from .runners import OptimizationPipeline
from .smoothers.signal_smoother import SignalSmoother


class CliConfigError(ValueError):
    """Raised when a CLI configuration is incomplete or inconsistent."""


@dataclass(frozen=True)
class CliRun:
    """Completed CLI run and the directory containing its artifacts."""

    output: OutputBundle
    output_dir: Path


_ARTIFACT_NAMES = {
    "target_positions.parquet",
    "trades.parquet",
    "summary.parquet",
    "metadata.json",
    "run_manifest.yaml",
    "resolved_cli_config.yaml",
    "resolved_mapping.yaml",
    "resolved_params.yaml",
}

_LEGACY_BARRA_ALIASES = {
    "meanvar_enhance_index": "meanvar_enhance_barra_precomputed",
    "minvar_enhance_index": "minvar_enhance_barra_precomputed",
}


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CliConfigError(f"{name} must be a mapping")
    return dict(value)


def _reject_unknown(mapping: Mapping[str, Any], allowed: set[str], name: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise CliConfigError(f"{name} contains unknown keys: {unknown}")


def _expand_path(value: Any, base_dir: Path, name: str) -> Path:
    if not isinstance(value, (str, os.PathLike)) or not str(value).strip():
        raise CliConfigError(f"{name} must be a non-empty path")
    expanded = Path(os.path.expandvars(str(value))).expanduser()
    if not expanded.is_absolute():
        expanded = base_dir / expanded
    return expanded.resolve()


def _normalize_datetime_index(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    result = frame.copy()
    try:
        index = pd.DatetimeIndex(pd.to_datetime(result.index))
    except Exception as exc:
        raise CliConfigError(f"{name} index cannot be parsed as dates") from exc
    if index.tz is not None:
        index = index.tz_localize(None)
    result.index = index.normalize()
    result.index.name = "date"
    if result.index.has_duplicates:
        duplicates = result.index[result.index.duplicated()].unique()[:5]
        raise CliConfigError(f"{name} has duplicate dates: {list(duplicates)}")
    result.columns = pd.Index([str(column) for column in result.columns], name="asset")
    if result.columns.has_duplicates:
        duplicates = result.columns[result.columns.duplicated()].unique()[:5]
        raise CliConfigError(f"{name} has duplicate assets: {list(duplicates)}")
    return result.sort_index()


def _read_table(
    path: Path,
    columns: list[str] | None,
    name: str,
    filters: list[tuple[str, str, Any]] | None = None,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{name} file does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        try:
            return pd.read_parquet(path, columns=columns, filters=filters)
        except Exception:
            # Parquet producers do not always agree on timestamp physical
            # types. A full column-pruned scan remains a correct fallback.
            if filters:
                return pd.read_parquet(path, columns=columns)
            raise
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path, usecols=columns)
    raise CliConfigError(
        f"{name} uses unsupported file type {suffix!r}; use parquet or csv"
    )


def load_matrix(
    raw_spec: Any,
    *,
    base_dir: Path,
    name: str,
    path_override: Path | None = None,
) -> tuple[pd.DataFrame, Path]:
    """Load one date-by-asset matrix from a long or wide parquet/CSV file."""

    if isinstance(raw_spec, (str, os.PathLike)):
        spec: dict[str, Any] = {"path": str(raw_spec)}
    else:
        spec = _require_mapping(raw_spec, f"inputs.{name}")
    _reject_unknown(
        spec,
        {
            "path",
            "layout",
            "date_column",
            "asset_column",
            "value_column",
            "start_date",
            "end_date",
        },
        f"inputs.{name}",
    )
    if path_override is None:
        path = _expand_path(spec.get("path"), base_dir, f"inputs.{name}.path")
    else:
        path = path_override.resolve()

    layout = str(spec.get("layout", "wide")).lower()
    if layout not in {"long", "wide"}:
        raise CliConfigError(f"inputs.{name}.layout must be long or wide")

    date_column = spec.get("date_column")
    start = spec.get("start_date")
    end = spec.get("end_date")
    if layout == "long":
        date_column = str(date_column or "date")
        asset_column = str(spec.get("asset_column") or "asset")
        value_column = spec.get("value_column")
        if not value_column:
            raise CliConfigError(
                f"inputs.{name}.value_column is required for long layout"
            )
        value_column = str(value_column)
        columns = [date_column, asset_column, value_column]
        filters: list[tuple[str, str, Any]] = []
        if start is not None:
            filters.append((date_column, ">=", pd.Timestamp(start)))
        if end is not None:
            filters.append((date_column, "<=", pd.Timestamp(end)))
        table = _read_table(path, columns, name, filters or None)
        missing = sorted(set(columns) - set(table.columns))
        if missing:
            raise CliConfigError(f"{name} is missing columns: {missing}")
        table[date_column] = pd.to_datetime(table[date_column]).dt.normalize()
        if table.duplicated([date_column, asset_column]).any():
            raise CliConfigError(f"{name} has duplicate date-asset rows")
        frame = table.pivot(
            index=date_column,
            columns=asset_column,
            values=value_column,
        )
    else:
        table = _read_table(path, None, name)
        if date_column:
            date_column = str(date_column)
            if date_column not in table.columns:
                raise CliConfigError(
                    f"{name} date_column {date_column!r} does not exist"
                )
            frame = table.set_index(date_column)
        elif isinstance(table.index, pd.DatetimeIndex):
            frame = table
        else:
            raise CliConfigError(
                f"inputs.{name}.date_column is required when the wide file "
                "does not preserve a DatetimeIndex"
            )

    frame = _normalize_datetime_index(frame, name)
    if start is not None:
        frame = frame.loc[frame.index >= pd.Timestamp(start).normalize()]
    if end is not None:
        frame = frame.loc[frame.index <= pd.Timestamp(end).normalize()]
    if frame.empty or len(frame.columns) == 0:
        raise CliConfigError(f"{name} is empty after loading and date filtering")
    return frame, path


def _load_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"configuration file does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    config = _require_mapping(payload, "configuration")
    _reject_unknown(
        config,
        {
            "config_version",
            "adapter",
            "inputs",
            "smoother",
            "run",
            "output",
        },
        "configuration",
    )
    if config.get("config_version") != 1:
        raise CliConfigError("config_version must be 1")
    return config


def _build_smoother(raw: Any) -> SignalSmoother:
    config = {} if raw is None else _require_mapping(raw, "smoother")
    allowed = {
        "mode",
        "topn_n",
        "turnover_threshold",
        "turnover_window",
        "ema_span",
    }
    _reject_unknown(config, allowed, "smoother")
    try:
        smoother = SignalSmoother(**config)
    except TypeError as exc:
        raise CliConfigError(f"invalid smoother configuration: {exc}") from exc
    if smoother.mode not in {"auto", "always", "never"}:
        raise CliConfigError("smoother.mode must be auto, always or never")
    return smoother


def _resolve_optimizer_for_inputs(
    pipeline: OptimizationPipeline,
    *,
    optimizer_name: Any,
    scenario: Any,
) -> tuple[str, str, Any]:
    """Resolve the optimizer before loading any optional input family."""

    if optimizer_name:
        requested = str(optimizer_name)
        effective = _LEGACY_BARRA_ALIASES.get(requested, requested)
        return requested, effective, pipeline.router.resolve(effective)
    resolved_scenario = str(scenario or "index_enhancement")
    spec = pipeline.router.resolve_default(resolved_scenario)
    return f"scenario:{resolved_scenario}", spec.name, spec


def _build_non_barra_bundle(
    *,
    alpha: pd.DataFrame,
    market: pd.DataFrame | None,
    benchmark: pd.DataFrame | None,
    tradable: pd.DataFrame | None,
    benchmark_index: str,
    alpha_input_type: str,
    alpha_horizon_days: int,
    calendar_name: str,
    risk_data_lag_periods: int,
    exposure_data_lag_periods: int,
    annualization_factor: float,
    absolute_return: bool,
) -> InputBundle:
    """Build the lightweight Alpha-only or data_access historical bundle."""

    assets = alpha.columns
    if benchmark is not None:
        previous = benchmark.iloc[[0]].copy()
    elif market is not None:
        previous = pd.DataFrame(
            0.0,
            index=pd.DatetimeIndex([alpha.index[0]], name="date"),
            columns=assets,
        )
    else:
        previous = None
    linear_cost = (
        None
        if market is None
        else pd.DataFrame(
            DEFAULT_LINEAR_COST_BPS,
            index=alpha.index,
            columns=assets,
            dtype=float,
        )
    )
    return InputBundle(
        alpha=alpha,
        market=market,
        benchmark=benchmark,
        prev_positions=previous,
        tradable=tradable,
        linear_cost_bps=linear_cost,
        metadata=OptimizationContext(
            alpha_is_absolute_return=absolute_return,
            benchmark_name=benchmark_index if benchmark is not None else "",
            calendar_name=calendar_name,
            decision_time="close",
            execution_time="next_open",
            market_data_lag_periods=risk_data_lag_periods,
            exposure_data_lag_periods=exposure_data_lag_periods,
            alpha_input_type=alpha_input_type,
            alpha_horizon_days=alpha_horizon_days,
            market_input_type="return" if market is not None else "price",
            risk_covariance_units="daily_variance",
            risk_horizon_days=1,
            annualization_factor=annualization_factor,
            extras={
                "input_assembly": (
                    "data_access_historical" if market is not None else "alpha_only"
                )
            },
        ),
    )


def _prepare_output_dir(output_dir: Path, *, overwrite: bool) -> None:
    if output_dir.exists() and not output_dir.is_dir():
        raise CliConfigError(f"output path is not a directory: {output_dir}")
    existing = sorted(
        name for name in _ARTIFACT_NAMES if (output_dir / name).exists()
    )
    if existing and not overwrite:
        raise CliConfigError(
            f"output directory already contains CLI artifacts: {existing}; "
            "pass --overwrite or set output.overwrite=true"
        )
    output_dir.mkdir(parents=True, exist_ok=True)


def _write_outputs(
    run_output: OutputBundle,
    output_dir: Path,
    *,
    config_path: Path,
    resolved_config: dict[str, Any],
    input_provenance: dict[str, Any],
    status: str,
) -> None:
    run_output.target_positions.to_parquet(output_dir / "target_positions.parquet")
    run_output.trades.to_parquet(output_dir / "trades.parquet")
    run_output.summary.to_parquet(output_dir / "summary.parquet")
    run_output.metadata.reset_index().to_json(
        output_dir / "metadata.json",
        orient="records",
        date_format="iso",
        force_ascii=False,
        indent=2,
        default_handler=str,
    )
    with (output_dir / "resolved_cli_config.yaml").open(
        "w", encoding="utf-8"
    ) as handle:
        yaml.safe_dump(resolved_config, handle, sort_keys=False, allow_unicode=True)

    metadata = run_output.metadata.iloc[0]
    manifest = {
        "status": status,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "riskfolio_qs_version": __version__,
        "config_path": str(config_path),
        "optimizer_name": str(metadata["optimizer_name"]),
        "fallback_used": bool(metadata["fallback_used"]),
        "solve_status": sorted(
            set(run_output.summary["solve_status"].astype(str))
            if "solve_status" in run_output.summary
            else {str(metadata["solve_status"])}
        ),
        "date_count": int(len(run_output.target_positions)),
        "asset_count": int(len(run_output.target_positions.columns)),
        "inputs": input_provenance,
        "artifacts": sorted(_ARTIFACT_NAMES - {"run_manifest.yaml"}),
    }
    with (output_dir / "run_manifest.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(manifest, handle, sort_keys=False, allow_unicode=True)


def run_from_config(
    config_path: str | Path,
    *,
    alpha_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    overwrite: bool | None = None,
    data_store: Any | None = None,
) -> CliRun:
    """Run one optimization from a YAML configuration."""

    path = Path(config_path).expanduser().resolve()
    base_dir = path.parent
    config = _load_config(path)
    resolved_config = deepcopy(config)

    adapter_config = _require_mapping(config.get("adapter", {}), "adapter")
    _reject_unknown(
        adapter_config,
        {
            "type",
            "risk_root",
            "alpha_input_type",
            "alpha_horizon_days",
            "calendar_name",
            "risk_data_lag_periods",
            "exposure_data_lag_periods",
            "annualization_factor",
            "load_factor_returns",
            "minimum_annual_specific_volatility",
            "benchmark_index",
            "minimum_benchmark_weight_coverage",
        },
        "adapter",
    )
    adapter_type = str(adapter_config.get("type", "auto"))
    if adapter_type not in {"auto", "barra_precomputed", "data_access"}:
        raise CliConfigError(
            "adapter.type must be auto, barra_precomputed or data_access"
        )
    benchmark_index = str(adapter_config.get("benchmark_index", "")).strip()
    minimum_benchmark_weight_coverage = adapter_config.get(
        "minimum_benchmark_weight_coverage", 0.95
    )
    alpha_input_type = str(adapter_config.get("alpha_input_type", "score"))
    alpha_horizon_days = int(adapter_config.get("alpha_horizon_days", 1))
    calendar_name = str(adapter_config.get("calendar_name", "XSHG"))
    risk_data_lag_periods = int(
        adapter_config.get("risk_data_lag_periods", 0)
    )
    exposure_data_lag_periods = int(
        adapter_config.get("exposure_data_lag_periods", 0)
    )
    annualization_factor = float(
        adapter_config.get("annualization_factor", 252.0)
    )

    inputs = _require_mapping(config.get("inputs"), "inputs")
    _reject_unknown(
        inputs,
        {"alpha"},
        "inputs",
    )
    required_inputs = {"alpha"}
    missing_inputs = sorted(required_inputs - set(inputs))
    if missing_inputs:
        raise CliConfigError(f"inputs is missing required entries: {missing_inputs}")

    alpha_override = (
        None
        if alpha_path is None
        else _expand_path(alpha_path, Path.cwd(), "--alpha")
    )
    alpha, resolved_alpha_path = load_matrix(
        inputs["alpha"],
        base_dir=base_dir,
        name="alpha",
        path_override=alpha_override,
    )

    run_config = _require_mapping(config.get("run", {}), "run")
    _reject_unknown(
        run_config,
        {
            "optimizer_name",
            "scenario",
            "runtime_overrides",
            "data_version_hash",
            "allow_fallback",
            "mapping_path",
            "parameter_path",
        },
        "run",
    )
    optimizer_name = run_config.get("optimizer_name")
    scenario = run_config.get("scenario")
    if optimizer_name and scenario:
        raise CliConfigError("run.optimizer_name and run.scenario are mutually exclusive")
    if not optimizer_name and not scenario:
        scenario = "index_enhancement"
    runtime_overrides = run_config.get("runtime_overrides")
    if runtime_overrides is not None:
        runtime_overrides = _require_mapping(
            runtime_overrides, "run.runtime_overrides"
        )
        unsupported_cost_overrides = sorted(
            set(runtime_overrides)
            & {
                "default_linear_cost_bps",
                "default_impact_cost",
                "impact_cost_penalty",
                "enable_impact_cost",
            }
        )
        if unsupported_cost_overrides:
            raise CliConfigError(
                "the production CLI owns its cost inputs and does not accept "
                f"cost-data overrides: {unsupported_cost_overrides}"
            )

    pipeline_kwargs: dict[str, Any] = {
        "smoother": _build_smoother(config.get("smoother"))
    }
    for key in ("mapping_path", "parameter_path"):
        if run_config.get(key):
            pipeline_kwargs[key] = _expand_path(
                run_config[key], base_dir, f"run.{key}"
            )
    pipeline = OptimizationPipeline(**pipeline_kwargs)
    try:
        requested_optimizer, effective_optimizer, optimizer_spec = (
            _resolve_optimizer_for_inputs(
                pipeline,
                optimizer_name=optimizer_name,
                scenario=scenario,
            )
        )
    except KeyError as exc:
        raise CliConfigError(str(exc)) from exc

    input_provenance: dict[str, Any] = {
        "alpha": {"source": "file", "path": str(resolved_alpha_path)},
        "routing": {
            "requested": requested_optimizer,
            "resolved_optimizer": effective_optimizer,
            "risk_mode": optimizer_spec.risk_mode,
        },
    }
    strong_inputs = set(optimizer_spec.required_inputs.get("strong", []))
    needs_benchmark = "benchmark" in strong_inputs
    if needs_benchmark and not benchmark_index:
        raise CliConfigError(
            f"adapter.benchmark_index is required for {effective_optimizer}"
        )

    if optimizer_spec.risk_mode == "none":
        bundle = _build_non_barra_bundle(
            alpha=alpha,
            market=None,
            benchmark=None,
            tradable=None,
            benchmark_index="",
            alpha_input_type=alpha_input_type,
            alpha_horizon_days=alpha_horizon_days,
            calendar_name=calendar_name,
            risk_data_lag_periods=risk_data_lag_periods,
            exposure_data_lag_periods=exposure_data_lag_periods,
            annualization_factor=annualization_factor,
            absolute_return=False,
        )
    elif optimizer_spec.risk_mode == "historical_cov":
        if adapter_type == "barra_precomputed":
            raise CliConfigError(
                f"{effective_optimizer} requires data_access historical returns; "
                "use adapter.type=auto or data_access"
            )
        try:
            production_inputs = load_portfolio_inputs(
                alpha,
                benchmark_index=benchmark_index or None,
                minimum_benchmark_weight_coverage=float(
                    minimum_benchmark_weight_coverage
                ),
                include_benchmark=needs_benchmark,
                store=data_store,
            )
            resolved_params, _ = pipeline.parameter_store.resolve(
                effective_optimizer, runtime_overrides
            )
            historical = load_historical_market_returns(
                alpha,
                lookback_days=int(resolved_params["hist_cov_lookback_days"]),
                market_data_lag_periods=risk_data_lag_periods,
                store=data_store,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise CliConfigError(
                f"invalid data_access historical inputs: {exc}"
            ) from exc
        bundle = _build_non_barra_bundle(
            alpha=alpha,
            market=historical.returns,
            benchmark=production_inputs.benchmark,
            tradable=production_inputs.tradable,
            benchmark_index=benchmark_index,
            alpha_input_type=alpha_input_type,
            alpha_horizon_days=alpha_horizon_days,
            calendar_name=calendar_name,
            risk_data_lag_periods=risk_data_lag_periods,
            exposure_data_lag_periods=exposure_data_lag_periods,
            annualization_factor=annualization_factor,
            absolute_return=optimizer_spec.objective_id == "OBJ_MEANVAR_ABS",
        )
        input_provenance.update(production_inputs.provenance)
        input_provenance.update(historical.provenance)
        input_provenance["initial_positions"] = (
            {
                "source": "data_access_benchmark",
                "assumption": "backtest_starts_from_benchmark",
                "date": alpha.index[0].date().isoformat(),
                "index_symbol": benchmark_index,
            }
            if production_inputs.benchmark is not None
            else {
                "source": "riskfolio_qs_backtest_default",
                "assumption": "backtest_starts_from_cash",
                "date": alpha.index[0].date().isoformat(),
            }
        )
    elif optimizer_spec.risk_mode == "barra_precomputed":
        if adapter_type == "data_access":
            raise CliConfigError(
                f"{effective_optimizer} requires a precomputed Barra package; "
                "adapter.type=data_access explicitly disables it"
            )
        risk_root_raw = adapter_config.get("risk_root")
        risk_root = _expand_path(
            risk_root_raw, base_dir, "adapter.risk_root"
        )
        try:
            production_inputs = load_portfolio_inputs(
                alpha,
                benchmark_index=benchmark_index,
                minimum_benchmark_weight_coverage=float(
                    minimum_benchmark_weight_coverage
                ),
                include_benchmark=True,
                store=data_store,
            )
        except (TypeError, ValueError) as exc:
            raise CliConfigError(
                f"invalid data_access portfolio inputs: {exc}"
            ) from exc
        if production_inputs.benchmark is None:
            raise CliConfigError("precomputed Barra optimization requires benchmark")
        try:
            adapter = BarraPrecomputedAdapter(
                risk_root=risk_root,
                alpha_df=alpha,
                benchmark_df=production_inputs.benchmark,
                prev_positions_df=production_inputs.benchmark.iloc[[0]].copy(),
                tradable_df=production_inputs.tradable,
                linear_cost_bps_df=None,
                impact_cost_df=None,
                alpha_input_type=alpha_input_type,
                alpha_horizon_days=alpha_horizon_days,
                calendar_name=calendar_name,
                risk_data_lag_periods=risk_data_lag_periods,
                exposure_data_lag_periods=exposure_data_lag_periods,
                annualization_factor=annualization_factor,
                load_factor_returns=bool(
                    adapter_config.get("load_factor_returns", False)
                ),
                minimum_annual_specific_volatility=float(
                    adapter_config.get(
                        "minimum_annual_specific_volatility", 0.05
                    )
                ),
                default_linear_cost_bps=DEFAULT_LINEAR_COST_BPS,
            )
            bundle = adapter.build_bundle()
        except TypeError as exc:
            raise CliConfigError(f"invalid adapter configuration: {exc}") from exc
        input_provenance.update(
            {
                "initial_positions": {
                    "source": "data_access_benchmark",
                    "assumption": "backtest_starts_from_benchmark",
                    "date": alpha.index[0].date().isoformat(),
                    "index_symbol": benchmark_index,
                },
                "barra_risk": {
                    "source": "precomputed_package",
                    "path": str(risk_root),
                },
                **production_inputs.provenance,
            }
        )
    else:
        raise CliConfigError(
            f"CLI input assembly does not support risk_mode="
            f"{optimizer_spec.risk_mode!r} for {effective_optimizer}"
        )

    output_config = _require_mapping(config.get("output", {}), "output")
    _reject_unknown(output_config, {"directory", "overwrite"}, "output")
    if output_dir is None:
        final_output_dir = _expand_path(
            output_config.get("directory", "riskfolio_output"),
            base_dir,
            "output.directory",
        )
    else:
        final_output_dir = _expand_path(output_dir, Path.cwd(), "--output-dir")
    final_overwrite = (
        bool(output_config.get("overwrite", False))
        if overwrite is None
        else overwrite
    )
    _prepare_output_dir(final_output_dir, overwrite=final_overwrite)

    if alpha_override is not None:
        alpha_spec = resolved_config.setdefault("inputs", {}).setdefault("alpha", {})
        if not isinstance(alpha_spec, dict):
            alpha_spec = {"path": str(resolved_alpha_path)}
            resolved_config["inputs"]["alpha"] = alpha_spec
        else:
            alpha_spec["path"] = str(resolved_alpha_path)
    resolved_config.setdefault("output", {})["directory"] = str(final_output_dir)
    resolved_config["output"]["overwrite"] = final_overwrite
    resolved_adapter = resolved_config.setdefault("adapter", {})
    resolved_adapter["type"] = adapter_type
    if optimizer_spec.risk_mode == "barra_precomputed":
        resolved_adapter["risk_root"] = str(risk_root)
    if needs_benchmark:
        resolved_adapter["benchmark_index"] = benchmark_index
        resolved_adapter[
            "minimum_benchmark_weight_coverage"
        ] = float(minimum_benchmark_weight_coverage)
    result = pipeline.run(
        bundle,
        optimizer_name=effective_optimizer,
        scenario=None,
        runtime_overrides=runtime_overrides,
        data_version_hash=str(run_config.get("data_version_hash", "")),
        output_dir=str(final_output_dir),
    )

    is_feasible = bool(result.summary["feasible_flag"].fillna(False).all())
    fallback_used = bool(result.metadata.iloc[0]["fallback_used"])
    fallback_allowed = bool(run_config.get("allow_fallback", False))
    health_status = (
        "passed"
        if is_feasible and (not fallback_used or fallback_allowed)
        else "failed"
    )
    _write_outputs(
        result,
        final_output_dir,
        config_path=path,
        resolved_config=resolved_config,
        input_provenance=input_provenance,
        status=health_status,
    )

    if not is_feasible:
        raise RuntimeError(
            "optimization output failed feasibility checks; diagnostics were "
            f"written to {final_output_dir}"
        )
    if fallback_used and not fallback_allowed:
        raise RuntimeError(
            "optimizer used a fallback while run.allow_fallback=false; "
            f"diagnostics were written to {final_output_dir}"
        )
    return CliRun(output=result, output_dir=final_output_dir)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="riskfolio-qs",
        description="Configuration-driven portfolio optimization",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    optimize = subparsers.add_parser(
        "optimize", help="run an optimization from a YAML configuration"
    )
    optimize.add_argument(
        "-c", "--config", required=True, help="YAML configuration path"
    )
    optimize.add_argument(
        "--alpha",
        help="override inputs.alpha.path; relative paths use the current directory",
    )
    optimize.add_argument(
        "-o",
        "--output-dir",
        help="override output.directory; relative paths use the current directory",
    )
    optimize.add_argument(
        "--overwrite",
        action="store_true",
        default=None,
        help="overwrite known artifacts in an existing output directory",
    )
    optimize.add_argument(
        "--debug", action="store_true", help="show a traceback when the run fails"
    )
    analyze = subparsers.add_parser(
        "analyze-positions",
        help="analyze a riskfolio_qs optimization artifact directory",
    )
    analyze.add_argument(
        "-c", "--config", required=True, help="position analysis YAML path"
    )
    analyze.add_argument(
        "--input-dir",
        help=(
            "override input.optimization_dir; relative paths use the current "
            "directory"
        ),
    )
    analyze.add_argument(
        "-o",
        "--output-dir",
        help=(
            "override output.directory; relative paths use the current directory"
        ),
    )
    analyze.add_argument(
        "--overwrite",
        action="store_true",
        default=None,
        help="overwrite known artifacts in an existing analysis directory",
    )
    analyze.add_argument(
        "--debug", action="store_true", help="show a traceback when analysis fails"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""

    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "analyze-positions":
        try:
            from .analysis import analyze_position_output

            result = analyze_position_output(
                config=args.config,
                input_dir=args.input_dir,
                output_dir=args.output_dir,
                overwrite=args.overwrite,
            )
        except Exception as exc:
            if args.debug:
                raise
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        checks = result.quality_checks
        errors = (
            int(
                (
                    checks["severity"].eq("error")
                    & checks["status"].eq("failed")
                ).sum()
            )
            if not checks.empty
            else 0
        )
        warnings = (
            int(
                (
                    checks["severity"].eq("warning")
                    & checks["status"].isin(["failed", "unavailable"])
                ).sum()
            )
            if not checks.empty
            else 0
        )
        provenance = result.metadata.get("provenance", {})
        snapshot_values = [
            details.get("snapshot_match")
            for details in provenance.values()
            if isinstance(details, Mapping) and "snapshot_match" in details
        ]
        snapshot_status = (
            "not_applicable"
            if not snapshot_values
            else "true"
            if all(value is True for value in snapshot_values)
            else "false"
            if any(value is False for value in snapshot_values)
            else "unknown"
        )
        print(f"status={result.status}")
        print(f"dates={len(result.position_summary)}")
        print(f"assets={result.metadata['asset_count']}")
        print(f"quality_errors={errors}")
        print(f"quality_warnings={warnings}")
        print(f"snapshot_match={snapshot_status}")
        print(f"output_dir={result.output_dir}")
        if result.status == "passed":
            return 0
        if result.status == "partial":
            return (
                2
                if result.metadata.get("allow_partial_enrichment", False)
                else 1
            )
        return 1

    try:
        completed = run_from_config(
            args.config,
            alpha_path=args.alpha,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
        )
    except Exception as exc:
        if args.debug:
            raise
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    metadata = completed.output.metadata.iloc[0]
    summary = completed.output.summary
    max_violation = (
        float(summary["max_constraint_violation"].max())
        if "max_constraint_violation" in summary
        else 0.0
    )
    print("status=passed")
    print(f"optimizer={metadata['optimizer_name']}")
    print(f"dates={len(completed.output.target_positions)}")
    print(f"assets={len(completed.output.target_positions.columns)}")
    print(f"max_constraint_violation={max_violation:.3e}")
    print(f"output_dir={completed.output_dir}")
    return 0


__all__ = ["CliConfigError", "CliRun", "load_matrix", "main", "run_from_config"]
