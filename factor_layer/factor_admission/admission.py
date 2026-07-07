from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from factor_layer.factor_admission.catalog import AdmissionCatalog
from factor_layer.factor_admission.config import FactorAdmissionConfig, ThresholdConfig


def _load_summary_payload(run_dir: Path) -> dict[str, object]:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"评估结果不存在: {summary_path}")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _resolve_run_dir(config: FactorAdmissionConfig) -> Path:
    evaluation_root = Path(config.source.evaluation_root or (Path(config.source.factor_lake_root) / "evaluations"))
    run_dir = evaluation_root / config.meta.factor_id / config.meta.run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"评估运行目录不存在: {run_dir}")
    return run_dir


def _check_thresholds(summary_row: dict[str, object], thresholds: ThresholdConfig) -> list[str]:
    diagnostics: list[str] = []
    checks = {
        "min_ic_mean": "ic_mean",
        "min_rank_ic_mean": "rank_ic_mean",
        "min_ic_win_rate": "ic_win_rate",
        "min_rank_ic_win_rate": "rank_ic_win_rate",
        "min_top_minus_bottom_mean": "top_minus_bottom_mean",
        "min_monotonicity_score": "monotonicity_score",
        "min_long_short_total_return": "long_short_total_return",
        "min_long_short_ann_return": "long_short_ann_return",
        "min_long_short_sharpe": "long_short_sharpe",
        "min_long_short_max_drawdown": "long_short_max_drawdown",
    }
    for threshold_field, metric_field in checks.items():
        threshold = getattr(thresholds, threshold_field)
        if threshold is None:
            continue
        actual = summary_row.get(metric_field)
        if actual is None:
            diagnostics.append(f"{metric_field} 缺失")
            continue
        actual_value = float(actual)
        if actual_value < threshold:
            diagnostics.append(f"{metric_field}={actual_value:.6f} < {threshold:.6f}")
    return diagnostics


def admit_evaluation_run(config: FactorAdmissionConfig) -> dict[str, object]:
    run_dir = _resolve_run_dir(config)
    summary_payload = _load_summary_payload(run_dir)
    if summary_payload["factor_id"] != config.meta.factor_id:
        raise ValueError("meta.factor_id 与评估产物不一致")
    if summary_payload["run_id"] != config.meta.run_id:
        raise ValueError("meta.run_id 与评估产物不一致")

    summary_rows = list(summary_payload.get("summary", []))
    if not summary_rows:
        raise ValueError("summary.json 中缺少 summary 结果")
    primary_horizon = config.decision.primary_horizon or int(summary_payload["primary_horizon"])
    target_row = next((row for row in summary_rows if int(row["horizon"]) == primary_horizon), None)
    if target_row is None:
        raise ValueError(f"找不到 primary_horizon={primary_horizon} 的评估摘要")

    if config.decision.mode == "manual":
        approved = bool(config.decision.approve)
        diagnostics = [] if approved else [config.decision.reason or "manual rejection"]
    else:
        diagnostics = _check_thresholds(target_row, config.decision.thresholds)
        approved = not diagnostics

    decision = "approved" if approved else "rejected"
    reason = config.decision.reason or ("; ".join(diagnostics) if diagnostics else "rule_based approval")
    catalog_path = Path(config.source.factor_lake_root) / "_catalog.sqlite"

    with AdmissionCatalog(catalog_path) as catalog:
        catalog.upsert_evaluation_run(
            {
                "run_id": summary_payload["run_id"],
                "factor_id": summary_payload["factor_id"],
                "run_dir": str(run_dir),
                "summary_path": str(run_dir / "summary.json"),
                "sample_start": summary_payload.get("sample_start"),
                "sample_end": summary_payload.get("sample_end"),
                "universe_id": summary_payload.get("universe_id"),
                "primary_horizon": primary_horizon,
                "config_hash": summary_payload["config_hash"],
                "created_at": summary_payload["created_at"],
            }
        )
        catalog.replace_evaluation_summary(summary_payload["run_id"], summary_rows)
        decision_id = catalog.record_decision(
            factor_id=config.meta.factor_id,
            run_id=config.meta.run_id,
            decision=decision,
            decided_by=config.decision.decided_by,
            reason=reason,
            policy_name=config.decision.policy_name,
            policy_snapshot=asdict(config.decision),
        )
        catalog.update_factor_status(
            factor_id=config.meta.factor_id,
            latest_run_id=config.meta.run_id,
            approved=approved,
        )
        status = catalog.get_factor_status(config.meta.factor_id)

    decision_payload = {
        "factor_id": config.meta.factor_id,
        "run_id": config.meta.run_id,
        "decision": decision,
        "approved": approved,
        "decision_id": decision_id,
        "primary_horizon": primary_horizon,
        "reason": reason,
        "diagnostics": diagnostics,
        "status": status,
    }
    if config.output.write_decision_file:
        (run_dir / "admission_decision.json").write_text(
            json.dumps(decision_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return decision_payload