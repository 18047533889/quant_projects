"""disk.v1 candidate_pool 导出（config.json + manifest.json）。"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from alphaprobe.delivery.config import DeliverySettings, ExperimentConfig


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _candidate_hash(formula: str, universe_id: str, frequency_bucket: str) -> str:
    normalized = " ".join(str(formula).split())
    payload = f"{normalized}|{universe_id}|{frequency_bucket}"
    return hashlib.sha256(payload.encode()).hexdigest()[:8]


def _candidate_id(formula: str, settings: DeliverySettings) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    h8 = _candidate_hash(formula, settings.universe_id, settings.frequency_bucket)
    return f"{settings.generator_name}_{stamp}_{h8}"


class DiskV1DeliveryExporter:
    """将 AlphaPROBE 因子池导出到 candidate_pool/{campaign_id}/。"""

    def __init__(self, experiment: ExperimentConfig) -> None:
        self.experiment = experiment
        self.settings = experiment.delivery
        self.started_at = datetime.now(timezone.utc)
        self.candidates_generated = 0
        self.candidates_submitted = 0
        self._exported_hashes: set[str] = set()
        self.campaign_dir = self.settings.output_campaign_dir()

    def write_config(self, *, finalize: bool = False, mining_run_stats: dict[str, Any] | None = None) -> Path:
        cfg = self.settings
        payload: dict[str, Any] = {
            "schema_version": "disk.v1",
            "campaign_id": cfg.resolved_campaign_id(),
            "generator_name": cfg.generator_name,
            "generator_version": cfg.generator_version,
            "market": cfg.market,
            "universe_id": cfg.universe_id,
            "signal_structure": cfg.signal_structure,
            "asset_class": cfg.asset_class,
            "frequency_bucket": cfg.frequency_bucket,
            "domain_root": cfg.domain_root,
            "domain": cfg.domain,
            "mining_scope": cfg.mining_scope,
            "operator_policy": cfg.operator_policy,
            "data_source": cfg.data_source,
            "mined_by": cfg.mined_by,
            "created_at": cfg.campaign_created_at(),
            "mining_config": {
                "train_period": list(cfg.mining_config.get("train_period", [])),
                "valid_period": list(cfg.mining_config.get("valid_period", [])),
                "test_period": list(cfg.mining_config.get("test_period", [])),
            },
        }
        if finalize:
            finished = datetime.now(timezone.utc)
            stats = dict(mining_run_stats or {})
            stats.setdefault("started_at", self.started_at.strftime("%Y-%m-%dT%H:%M:%SZ"))
            stats.setdefault("finished_at", finished.strftime("%Y-%m-%dT%H:%M:%SZ"))
            if "duration_seconds" not in stats:
                stats["duration_seconds"] = int((finished - self.started_at).total_seconds())
            stats.setdefault("candidates_generated", self.candidates_generated)
            stats.setdefault("candidates_submitted", self.candidates_submitted)
            if cfg.llm_model:
                stats.setdefault("llm_model", cfg.llm_model)
            if cfg.llm_provider:
                stats.setdefault("llm_provider", cfg.llm_provider)
            if cfg.llm_base_url:
                stats.setdefault("llm_base_url", cfg.llm_base_url)
            payload["mining_run_stats"] = stats

        out = self.campaign_dir / "config.json"
        _write_json(out, payload)
        return out

    def export_candidate(
        self,
        expr: Any,
        *,
        description: str,
        metrics: dict[str, dict[str, float]],
        generation_stats: dict[str, Any] | None = None,
    ) -> Path | None:
        from alphaprobe.fe_bridge.dsl_convert import expression_to_dsl

        self.candidates_generated += 1
        formula = expression_to_dsl(expr)
        h8 = _candidate_hash(formula, self.settings.universe_id, self.settings.frequency_bucket)
        if h8 in self._exported_hashes:
            return None
        self._exported_hashes.add(h8)

        train_ic = abs(metrics.get("train", {}).get("ic", 0.0))
        if train_ic < self.settings.ic_export_threshold:
            return None

        candidate_id = _candidate_id(formula, self.settings)
        born = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        manifest: dict[str, Any] = {
            "schema_version": "disk.v1",
            "candidate_id": candidate_id,
            "campaign_id": self.settings.resolved_campaign_id(),
            "formula": formula,
            "expression_type": self.settings.expression_type,
            "generator_name": self.settings.generator_name,
            "generator_version": self.settings.generator_version,
            "mined_by": self.settings.mined_by,
            "market": self.settings.market,
            "universe_id": self.settings.universe_id,
            "domain_root": self.settings.domain_root,
            "domain": self.settings.domain,
            "frequency_bucket": self.settings.frequency_bucket,
            "born_timestamp": born,
            "metrics": metrics,
            "description": (description or "").strip() or f"AlphaPROBE factor: {formula[:120]}",
        }
        if self.settings.filter_note:
            manifest["filter_note"] = self.settings.filter_note
        if generation_stats:
            manifest["generation_stats"] = generation_stats

        out = self.campaign_dir / candidate_id / "manifest.json"
        _write_json(out, manifest)
        self.candidates_submitted += 1
        return out

    def export_from_pool(
        self,
        pool: Any,
        target: Any,
        experiment: ExperimentConfig,
        device: Any,
    ) -> dict[str, Any]:
        from alphaprobe.fe_bridge.metrics import evaluate_all_period_metrics

        """从 AlphaKnowledgePool 导出全部合格因子。"""
        self.write_config(finalize=False)
        exported: list[str] = []
        decisions_log: list[dict[str, Any]] = []

        for i in range(pool.size):
            expr = pool.exprs[i]
            if expr is None:
                continue
            description = ""
            topic = ""
            if hasattr(pool, "topics") and pool.topics[i]:
                topic = str(pool.topics[i])
            if hasattr(pool, "descriptions") and pool.descriptions[i]:
                description = str(pool.descriptions[i])
            if topic and description:
                full_desc = f"[{topic}] {description}"
            elif topic:
                full_desc = topic
            else:
                full_desc = description

            metrics = evaluate_all_period_metrics(expr, target, experiment, device)
            # §58：ExportDecision 审计日志（不改变 manifest 输出结构，delivery 链兼容优先）
            decision = self._build_export_decision_if_available(expr, metrics, i)
            if decision is not None:
                decisions_log.append(decision)
            path = self.export_candidate(expr, description=full_desc, metrics=metrics)
            if path is not None:
                exported.append(str(path))

        config_path = self.write_config(
            finalize=True,
            mining_run_stats={
                "llm_model": self.settings.llm_model or None,
                "llm_provider": self.settings.llm_provider or None,
                "llm_base_url": self.settings.llm_base_url or None,
                "token_usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
                "export_decisions": decisions_log,
            },
        )
        return {
            "campaign_dir": str(self.campaign_dir),
            "config_path": str(config_path),
            "exported_manifests": exported,
            "candidates_submitted": self.candidates_submitted,
        }

    def _build_export_decision_if_available(
        self,
        expr: Any,
        metrics: dict[str, dict[str, float]],
        pool_index: int,
    ) -> dict[str, Any] | None:
        """§58：若 export/gate.py 存在则调 build_export_decision 打日志；否则 None。

        不改变现有 manifest 输出结构：decision 只作为 mining_run_stats 附注。
        """
        try:
            from alphaprobe.export.gate import build_export_decision
        except Exception:  # noqa: BLE001 - gate 不可用时不阻塞 delivery 链
            return None
        try:
            from alphaprobe.contracts import EvaluationRecord
            from datetime import datetime, timezone

            from alphaprobe.fe_bridge.dsl_convert import expression_to_dsl

            formula = expression_to_dsl(expr)
            record = EvaluationRecord(
                factor_id=f"ap_pool_{pool_index}",
                segment="train",
                fidelity="L2_full_train",
                metric_bundle=dict(metrics.get("train") or {}),
                artifact_refs={},
                evaluator_version="alphaprobe.disk.v1",
                data_snapshot_id="auto",
                universe_snapshot_id="auto",
                label_spec_hash="vwap_to_vwap_h20",
                created_at=datetime.now(timezone.utc),
            )
            candidate = {"factor_id": f"ap_pool_{pool_index}", "canonical_formula": formula}
            d = build_export_decision(candidate, record, None, refinement_complete=True)
            return d.to_dict()
        except Exception:  # noqa: BLE001
            return None