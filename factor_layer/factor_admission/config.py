from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import yaml

from workspace_paths import default_factor_evaluation_root, default_factor_lake_root


@dataclass(frozen=True)
class MetaConfig:
    factor_id: str
    run_id: str


@dataclass(frozen=True)
class SourceConfig:
    factor_lake_root: str
    evaluation_root: str | None = None


@dataclass(frozen=True)
class ThresholdConfig:
    min_ic_mean: float | None = None
    min_rank_ic_mean: float | None = None
    min_ic_win_rate: float | None = None
    min_rank_ic_win_rate: float | None = None
    min_top_minus_bottom_mean: float | None = None
    min_monotonicity_score: float | None = None
    min_long_short_total_return: float | None = None
    min_long_short_ann_return: float | None = None
    min_long_short_sharpe: float | None = None
    min_long_short_max_drawdown: float | None = None


@dataclass(frozen=True)
class DecisionConfig:
    mode: str = "rule_based"
    approve: bool | None = None
    decided_by: str = "system"
    reason: str | None = None
    policy_name: str = "default_daily_v1"
    primary_horizon: int | None = None
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)


@dataclass(frozen=True)
class OutputConfig:
    write_decision_file: bool = True


@dataclass(frozen=True)
class FactorAdmissionConfig:
    meta: MetaConfig
    source: SourceConfig
    decision: DecisionConfig = field(default_factory=DecisionConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


def _as_mapping(value: Any, *, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} 必须是 object")
    return dict(value)


def _normalize_scalar_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat(sep=" ")
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


def _require_non_empty_text(value: Any, *, field_name: str) -> str:
    text = _normalize_scalar_text(value)
    if text is None or not text.strip():
        raise ValueError(f"{field_name} 不能为空")
    return text.strip()


def _resolve_path_text(
    value: Any,
    *,
    field_name: str,
    base_dir: Path,
) -> str | None:
    if value is None:
        return None
    if not isinstance(value, (str, Path)):
        raise ValueError(f"{field_name} 必须是路径字符串")
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field_name} 不能为空字符串")
    expanded = Path(os.path.expandvars(os.path.expanduser(text)))
    if not expanded.is_absolute():
        expanded = (base_dir / expanded).resolve()
    return str(expanded)


def _to_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def load_config(path: str | Path) -> FactorAdmissionConfig:
    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("配置文件顶层必须是 object")

    base_dir = config_path.resolve().parent
    meta_payload = _as_mapping(payload.get("meta"), field_name="meta")
    source_payload = _as_mapping(payload.get("source"), field_name="source")
    decision_payload = _as_mapping(payload.get("decision"), field_name="decision")
    output_payload = _as_mapping(payload.get("output"), field_name="output")
    thresholds_payload = _as_mapping(decision_payload.get("thresholds"), field_name="decision.thresholds")

    meta = MetaConfig(
        factor_id=_require_non_empty_text(meta_payload.get("factor_id"), field_name="meta.factor_id"),
        run_id=_require_non_empty_text(meta_payload.get("run_id"), field_name="meta.run_id"),
    )
    factor_lake_root = _resolve_path_text(
        source_payload.get("factor_lake_root"),
        field_name="source.factor_lake_root",
        base_dir=base_dir,
    ) or str(default_factor_lake_root())
    source = SourceConfig(
        factor_lake_root=factor_lake_root,
        evaluation_root=_resolve_path_text(
            source_payload.get("evaluation_root") or str(default_factor_evaluation_root(factor_lake_root=factor_lake_root)),
            field_name="source.evaluation_root",
            base_dir=base_dir,
        ),
    )
    decision = DecisionConfig(
        mode=str(decision_payload.get("mode") or "rule_based"),
        approve=decision_payload.get("approve"),
        decided_by=str(decision_payload.get("decided_by") or "system"),
        reason=_normalize_scalar_text(decision_payload.get("reason")),
        policy_name=str(decision_payload.get("policy_name") or "default_daily_v1"),
        primary_horizon=(int(decision_payload["primary_horizon"]) if decision_payload.get("primary_horizon") is not None else None),
        thresholds=ThresholdConfig(
            min_ic_mean=_to_optional_float(thresholds_payload.get("min_ic_mean")),
            min_rank_ic_mean=_to_optional_float(thresholds_payload.get("min_rank_ic_mean")),
            min_ic_win_rate=_to_optional_float(thresholds_payload.get("min_ic_win_rate")),
            min_rank_ic_win_rate=_to_optional_float(thresholds_payload.get("min_rank_ic_win_rate")),
            min_top_minus_bottom_mean=_to_optional_float(thresholds_payload.get("min_top_minus_bottom_mean")),
            min_monotonicity_score=_to_optional_float(thresholds_payload.get("min_monotonicity_score")),
            min_long_short_total_return=_to_optional_float(thresholds_payload.get("min_long_short_total_return")),
            min_long_short_ann_return=_to_optional_float(thresholds_payload.get("min_long_short_ann_return")),
            min_long_short_sharpe=_to_optional_float(thresholds_payload.get("min_long_short_sharpe")),
            min_long_short_max_drawdown=_to_optional_float(thresholds_payload.get("min_long_short_max_drawdown")),
        ),
    )
    if decision.mode not in {"rule_based", "manual"}:
        raise ValueError("decision.mode 仅支持 rule_based 或 manual")
    if decision.mode == "manual" and decision.approve is None:
        raise ValueError("manual 模式下必须显式提供 decision.approve")
    output = OutputConfig(write_decision_file=bool(output_payload.get("write_decision_file", True)))
    return FactorAdmissionConfig(meta=meta, source=source, decision=decision, output=output)