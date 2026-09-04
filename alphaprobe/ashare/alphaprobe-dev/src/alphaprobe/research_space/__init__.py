"""research_space —— 五层研究空间的包入口（plan Part B / Task 6 / Task 12）。

五层：Market Logic → Schema → Hypothesis → Implementation Plan → Factor AST。
本包已实现（按收口顺序）：

- **Hypothesis Alignment（Task 6）**：``alignment.py`` 的确定性对齐
  （FE static-analysis 事实 × DA taxonomy 期望域），在评估前拦
  hypothesis-domain 矛盾（Non-negotiable #18）。
- **Schema Space（Task 12）**：``schema.py`` + ``registry.py`` 的九维
  SchemaPlan / 稳定版本化 schema_id / schema 级统计注册表（独立于公式层、
  带置信收缩，Part G #29）。
- **Hypothesis Record + 九阶段多阶段状态机（Task 16）**：``hypothesis.py``
  的 HypothesisRegistry / run_stage_pipeline（stage hook 挂载 FE/QE/对齐
  能力，多阶段整体可开关，ablation #30）。
- **Market Logic Library（Task 13）**：``logic.py`` + ``logic_miner.py``
  的 LogicNode / LogicLibrary（稳定机器 logic_id 与 LLM 名字分离、成员
  概率多标签映射、低 support 统计向全局先验收缩、版本化 + 当前 sealed
  survival 不进当前版本 logic reward）。

Part E 边界：本包只**消费**平台能力（FE artifact、DA taxonomy descriptor /
词表），不复制解析 / taxonomy / static-analysis。生产绑定真实 provider 由
调用方注入；测试注入 fake。

模块顶层不 import factor_engine / data_access / factor_assets / torch ——
纯单测与 OFFLINE_TEST 可直接 import。
"""

from __future__ import annotations

from alphaprobe.research_space.contracts import (
    AlignmentResult,
    HypothesisSpec,
    SCHEMA_DIMENSIONS,
    SchemaIdentity,
    SchemaPlan,
    SchemaStats,
)
from alphaprobe.research_space.alignment import (
    DeterministicAligner,
    UnknownFieldTaxonomyError,
    bare_field_name,
)
from alphaprobe.research_space.registry import SchemaRegistry
from alphaprobe.research_space.schema import (
    DEFAULT_DOMAIN_VOCABULARY,
    SchemaValidationError,
    validate_schema_plan,
)

__all__ = [
    "AlignmentResult",
    "DeterministicAligner",
    "HypothesisSpec",
    "SCHEMA_DIMENSIONS",
    "SchemaIdentity",
    "SchemaPlan",
    "SchemaRegistry",
    "SchemaStats",
    "SchemaValidationError",
    "UnknownFieldTaxonomyError",
    "bare_field_name",
    "validate_schema_plan",
    "DEFAULT_DOMAIN_VOCABULARY",
]