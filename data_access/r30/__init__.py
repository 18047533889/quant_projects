"""data_access.r30 —— R30 全维度成熟度层（additive，不改既有文件）。

R30 在既有 R24-R29 闭环之上新增的运行时对象全部放这里，**不修改** store.py /
read/ / runtime/ 等既有文件（并发 FE/DA session 活跃，本层完全自包含）：

    concepts / specs         —— ConceptId、UnitType、FrequencySpec、GrainSpec、
                                MarketProfile、PriceBasisSpec、InstrumentIdentity、
                                RevisionFidelity、SemanticColumnVersion
    calendar_snapshot        —— 全量 CalendarSnapshot digest（R30-P0-008）
    universe_snapshot        —— UniverseSnapshot（R30-P1-009）
    experiment_snapshot      —— ExperimentDataSnapshot（R30-P0-009）
    coverage_service         —— CoverageService.describe（R30-P0-007）
    partition_index          —— PartitionMetadataIndex 立面 + 验收（R30-P0-006）
    query_trace / metrics    —— QueryTrace + OTel/Prometheus 风格指标（R30-P0-010）
    session                  —— R30ReadSession：prepared/source-block 复用 + 成本模型
    data_change              —— DataChangeSet + 变化探测（R30-P0-011）
    change_impact            —— DataChangeSet → FE ChangeImpact 桥（最小重算）
    join_cost                —— JoinCostPlanner 启发式（R30-P1-011）
    adapter                  —— SourceAdapter + SourceCapabilities + BackendCapabilities
    cache_hierarchy          —— L0-L4 缓存模型（R30-P1-013）
    execution_lease          —— ExecutionLease（R30-P1-014）
    lineage                  —— LineageStore（R30-P1-019）
    policy                   —— PolicyManifest（R30-P1-017）
    data_quality             —— DataQualityService 立面（R30-P1-025）
    mining_profile           —— MiningFieldProfile + FieldCapabilityCatalog
    artifact_meta            —— FactorArtifactMetadata（R30-P1-024）
    training / multi_asset / distributed / api_surface —— R30-P2 接口
"""
from __future__ import annotations

from typing import Any

# REM-043: Version authority unified - r30 is a submodule, not a separate package
# Version comes from parent data_access package only
__all__ = [
    "query_trace",
    "metrics",
    "versioning",
    "concepts",
    "specs",
    "calendar_snapshot",
    "universe_snapshot",
    "experiment_snapshot",
    "coverage_service",
    "partition_index",
    "session",
    "data_change",
    "change_impact",
    "join_cost",
    "adapter",
    "cache_hierarchy",
    "execution_lease",
    "lineage",
    "policy",
    "data_quality",
    "formats_contract",
    "mining_profile",
    "artifact_meta",
    "training",
    "multi_asset",
    "distributed",
    "api_surface",
]


def __getattr__(name: str) -> Any:
    """惰性 import：模块按需加载，未实现的子模块不阻塞包导入。"""
    if name in __all__:
        import importlib

        return importlib.import_module(f"data_access.r30.{name}")
    raise AttributeError(f"data_access.r30 has no attribute {name!r}")
