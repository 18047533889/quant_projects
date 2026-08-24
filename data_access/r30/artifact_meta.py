# -*- coding: utf-8 -*-
"""R30-P1-024 (FACTOR_ARTIFACT_METADATA)：因子产物的完整身份元数据。

R30-P1-024 验收：**不能只保存 factor expression / value**——产物的 metadata 必须
携带完整身份：factor_definition、operator 语义版本、source plan digest、
experiment/universe/calendar snapshot、coverage policy、security
classification、build_sha、computed_at。任何下游复用（审计、复现、交叉验证、
生产准入）都从这份 metadata 出发，而不是从一串裸数值出发。
"""
from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class FactorArtifactMetadata:
    """一个因子产物的完整身份元数据。

    - ``factor_definition``：因子 DSL 定义（expression），可复现的来源。
    - ``operator_semantic_versions``：所用各算子的语义版本（``{op_name: ver}``）。
    - ``factor_source_plan_digest``：因子 source plan 的稳定摘要（血缘根）。
    - ``experiment_snapshot_id`` / ``universe_snapshot_id`` / ``calendar_snapshot_id``：
      计算时实验 / 股票池 / 日历快照，跨时点复现的唯一锚。
    - ``coverage_policy``：覆盖/空值策略（如 ``strict`` / ``allow_missing``）。
    - ``security_classification``：安全分级（如 ``public`` / ``internal`` /
      ``confidential``）。
    - ``build_sha``：计算该产物的代码构建 SHA。
    - ``computed_at``：计算时间（ISO）。
    """

    factor_id: str
    factor_definition: str
    operator_semantic_versions: dict = field(default_factory=dict)
    factor_source_plan_digest: str = ""
    experiment_snapshot_id: str = ""
    universe_snapshot_id: str = ""
    calendar_snapshot_id: str = ""
    coverage_policy: str = ""
    security_classification: str = ""
    build_sha: str = ""
    computed_at: str | None = None

    def __post_init__(self) -> None:
        if self.computed_at is None:
            object.__setattr__(
                self,
                "computed_at",
                _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            )
        if not isinstance(self.operator_semantic_versions, Mapping):
            object.__setattr__(
                self, "operator_semantic_versions", dict(self.operator_semantic_versions or {})
            )
        else:
            object.__setattr__(
                self, "operator_semantic_versions", dict(self.operator_semantic_versions)
            )

    def to_dict(self) -> dict[str, Any]:
        """完整身份字段（不含 artifact_path——那是绑定信息，见 sidecar）。"""
        return {
            "factor_id": self.factor_id,
            "factor_definition": self.factor_definition,
            "operator_semantic_versions": dict(self.operator_semantic_versions),
            "factor_source_plan_digest": self.factor_source_plan_digest,
            "experiment_snapshot_id": self.experiment_snapshot_id,
            "universe_snapshot_id": self.universe_snapshot_id,
            "calendar_snapshot_id": self.calendar_snapshot_id,
            "coverage_policy": self.coverage_policy,
            "security_classification": self.security_classification,
            "build_sha": self.build_sha,
            "computed_at": self.computed_at,
        }

    def bind_metadata(self, artifact_path: Any = None, **extra: Any) -> dict[str, Any]:
        """便捷：返回可随产物绑定的完整 metadata dict（全字段 + 可选路径/附加字段）。"""
        payload = self.to_dict()
        if artifact_path is not None:
            payload["artifact_path"] = str(artifact_path)
        payload.update(extra)
        return payload

    def metadata_sidecar(self, artifact_path: Any) -> dict[str, Any]:
        """生成可随 artifact 保存的 JSON sidecar dict（含路径 + sidecar 版本）。"""
        payload = self.to_dict()
        payload["artifact_path"] = str(artifact_path)
        payload["sidecar_schema"] = "FactorArtifactMetadata"
        payload["sidecar_version"] = "1"
        return payload


__all__ = ["FactorArtifactMetadata"]
