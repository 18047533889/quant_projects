"""Strict YAML configuration for position analysis."""

from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
from typing import Any, Mapping

import yaml


class AnalysisConfigError(ValueError):
    """Raised when an analysis configuration violates its schema."""


DEFAULT_CONFIG: dict[str, Any] = {
    "analysis_version": 1,
    "input": {"optimization_dir": None},
    "validation": {
        "reconstruction_tolerance": 1e-10,
        "comparison_tolerance": 1e-8,
        "binding_tolerance": 1e-6,
        "provenance_policy": "strict",
        "allow_partial_enrichment": False,
    },
    "position": {
        "weight_epsilon": 1e-8,
        "topk": [5, 10, 20],
        "include_full_universe": True,
        "concentration": True,
        "turnover": True,
        "holding_persistence": True,
    },
    "benchmark": {"mode": "auto", "source": "manifest"},
    "barra": {
        "mode": "auto",
        "source": "manifest",
        "analyze_absolute_risk": True,
        "analyze_active_risk": True,
    },
    "liquidity": {
        "mode": "off",
        "portfolio_notional": None,
        "adv_window_days": 20,
        "maximum_adv_participation": 0.10,
    },
    "report": {"html": True, "top_holdings": 20, "top_trades": 20},
    "output": {"directory": None, "parquet": True, "overwrite": False},
}

_ALLOWED = {
    "configuration": set(DEFAULT_CONFIG),
    "input": {"optimization_dir"},
    "validation": set(DEFAULT_CONFIG["validation"]),
    "position": set(DEFAULT_CONFIG["position"]),
    "benchmark": set(DEFAULT_CONFIG["benchmark"]),
    "barra": set(DEFAULT_CONFIG["barra"]),
    "liquidity": set(DEFAULT_CONFIG["liquidity"]),
    "report": set(DEFAULT_CONFIG["report"]),
    "output": set(DEFAULT_CONFIG["output"]),
}


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise AnalysisConfigError(f"{name} must be a mapping")
    result = dict(value)
    unknown = sorted(set(result) - _ALLOWED[name])
    if unknown:
        raise AnalysisConfigError(f"{name} contains unknown keys: {unknown}")
    return result


def _resolve_path(value: Any, base_dir: Path, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, (str, os.PathLike)) or not str(value).strip():
        raise AnalysisConfigError(f"{name} must be a non-empty path")
    path = Path(os.path.expandvars(str(value))).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return str(path.resolve())


def load_analysis_config(
    config: str | Path | Mapping[str, Any],
) -> tuple[dict[str, Any], Path | None]:
    """Parse, validate and fill defaults for an analysis config."""

    config_path: Path | None = None
    if isinstance(config, (str, os.PathLike)):
        config_path = Path(config).expanduser().resolve()
        if not config_path.exists():
            raise FileNotFoundError(
                f"analysis configuration does not exist: {config_path}"
            )
        with config_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        base_dir = config_path.parent
    elif isinstance(config, Mapping):
        raw = dict(config)
        base_dir = Path.cwd()
    else:
        raise AnalysisConfigError("analysis configuration must be a path or mapping")

    raw = _mapping(raw, "configuration")
    if raw.get("analysis_version") != 1:
        raise AnalysisConfigError("analysis_version must be 1")
    resolved = deepcopy(DEFAULT_CONFIG)
    resolved["analysis_version"] = 1
    for section in (
        "input",
        "validation",
        "position",
        "benchmark",
        "barra",
        "liquidity",
        "report",
        "output",
    ):
        supplied = _mapping(raw.get(section, {}), section)
        resolved[section].update(supplied)

    resolved["input"]["optimization_dir"] = _resolve_path(
        resolved["input"]["optimization_dir"], base_dir, "input.optimization_dir"
    )
    resolved["output"]["directory"] = _resolve_path(
        resolved["output"]["directory"], base_dir, "output.directory"
    )
    for section in ("benchmark", "barra", "liquidity"):
        raw_mode = resolved[section]["mode"]
        mode = "off" if raw_mode is False else str(raw_mode).lower()
        if mode not in {"auto", "required", "off"}:
            raise AnalysisConfigError(
                f"{section}.mode must be auto, required or off"
            )
        resolved[section]["mode"] = mode
    for section in ("benchmark", "barra"):
        if resolved[section]["source"] != "manifest":
            raise AnalysisConfigError(f"{section}.source must be manifest")
    policy = str(resolved["validation"]["provenance_policy"])
    if policy not in {"strict", "warn"}:
        raise AnalysisConfigError(
            "validation.provenance_policy must be strict or warn"
        )

    numeric_positive = [
        (
            "validation.reconstruction_tolerance",
            resolved["validation"]["reconstruction_tolerance"],
        ),
        (
            "validation.comparison_tolerance",
            resolved["validation"]["comparison_tolerance"],
        ),
        (
            "validation.binding_tolerance",
            resolved["validation"]["binding_tolerance"],
        ),
        ("position.weight_epsilon", resolved["position"]["weight_epsilon"]),
        ("liquidity.adv_window_days", resolved["liquidity"]["adv_window_days"]),
        (
            "liquidity.maximum_adv_participation",
            resolved["liquidity"]["maximum_adv_participation"],
        ),
    ]
    for name, value in numeric_positive:
        if float(value) <= 0:
            raise AnalysisConfigError(f"{name} must be positive")
    topk = resolved["position"]["topk"]
    if (
        not isinstance(topk, list)
        or not topk
        or any(int(value) <= 0 for value in topk)
    ):
        raise AnalysisConfigError("position.topk must be a non-empty positive list")
    resolved["position"]["topk"] = sorted(set(int(value) for value in topk))
    notional = resolved["liquidity"]["portfolio_notional"]
    if notional is not None and float(notional) <= 0:
        raise AnalysisConfigError("liquidity.portfolio_notional must be positive")
    if (
        resolved["liquidity"]["mode"] == "required"
        and notional is None
    ):
        raise AnalysisConfigError(
            "liquidity.portfolio_notional is required when liquidity.mode=required"
        )
    return resolved, config_path


__all__ = ["AnalysisConfigError", "DEFAULT_CONFIG", "load_analysis_config"]
