"""AlphaPROBE algorithm ablation harness（plan.md Task 24 / Part H）。

目标
----
在**不触碰任何生产模块开关**的前提下，把 plan Task 24 列出的 10 个累积配置
（1. legacy reference → 10. + next-version Survival Memory）装配为可执行配置：

    1. legacy reference（原始 AlphaPROBE V3 参考）
    2. + FactorFitness V2.1
    3. + true Bayesian parent retrieval
    4. + contextual action/schema retriever
    5. + Schema Space
    6. + Market Logic Memory
    7. + Paradigm Scheduler
    8. + Trajectory Credit
    9. + EVI promotion
    10. + next-version Survival Memory

每个新算法特性（T8/T10/T12/T13/T14/T15/T17/T18/T11/T2 等）的开关位已由
各特性模块落好（Part G #30）；本模块只做「配置 → 开关装配」的矩阵 + 一次
run 后的**指标聚合**（AblationMetrics），不做任何真实 LLM/FE/QE 调用。

装配语义（cumulative feature tower）
------------------------------------
配置 N 打开的特性 = 配置 N-1 全部特性 + 第 N 项（plan 列表本身即累积排序）。
- Bayesian parent retrieval（#3）指「retrieval 层用 BayesianRetriever 排序」，
  即 ``BayesianRetrieverConfig.enabled=True``；
- Contextual retriever（#4）在 #3 之上把 retriever 换成
  ``ContextualRetrieverConfig.enabled=True``（parent×action×schema 检索，
  内部自含 Bayesian 基座）——故 #4 的 bayesian.enabled 仍保持 True；
- Schema Space / Market Logic Memory 本体是研究空间表示（SchemaRegistry /
  LogicLibrary / 多阶段状态机），其消费侧开关为 orchestrator
  ``structured_generation`` / ``multistage_generation``（hypothesis→schema→
  AST 九阶段状态机，见 Task 16）。本模块把它们显式投射为装配位，便于
  diff 可枚举；本体 registry/library 作为被动消费结构不设 kill-switch。
- Paradigm Scheduler（#7）用 ``ParadigmSelectionConfig.enabled``；其子开关
  ``schema_enabled`` 在 #7 开启后保留默认 True（SCHEMA 范式可用），但把
  Schema Space 消费入口（multistage）关掉的配置仍可对比「纯范式调度」增量。
- Trajectory Credit（#8）消费侧 = Paradigm TRAJECTORY_REPAIR，映射
  ``repair_enabled``（critic 本体由 orchestrator/routing 接线）。
- EVI promotion（#9）映射 ``PromotionSurrogateConfig.enabled/use_model``。
- next-version Survival（#10）映射 SearchOpportunity 的 ``w_survival`` 权重 +
  survival 归因模型注入（``survival_model`` / ``with_interactions`` /
  ``fit_backend``）——2026-like 下版本 survival 只喂下一研究版本，是
  Plan Part G #23 的已有语义，本模块不改变它。

指标口径（plan Task 24 primary metrics 清单）
---------------------------------------------
- EliteYield = globally-novel L3/L4 elites / 1000 attempts（分母是 attempts，
  不是 generated factor count——generated factor count 绝不是 primary
  metric，见 plan 原文）。
- CostAdjustedEliteYield = elite 数 ÷ 折算总成本（LLM cost + FE/QE compute
  归一），按每 run 归一化到 1000 attempts 当量。
- duplicate rate = 重复 attempt / attempts。
- unique schema coverage / unique logic coverage：attempts 里出现的不同
  schema_id / logic_id 数；unique cluster coverage 由 factor 的
  cluster_id 计数。
- LLM cost per elite、FE/QE compute per elite。
- median/best L3 FactorFitness、median Long-Short Sharpe、median
  Stability、next-version survival rate（available 时）——全部**只聚合
  已评估事实，不重算指标**（Non-negotiable #7）。

聚合只读注入的 attempt 行（dict），对空/无数据 run 输出中性空指标
（不抛、不编造 0 之外的数字）。

模块顶层只 import 轻量模块（contracts / research_space / retrieval /
surrogate / search——均无 torch/faiss/sklearn 硬依赖；survival.attribution
的 sklearn 为 try-import 可选）。
"""

from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

__all__ = [
    "ELITE_YIELD_DENOMINATOR",
    "FEATURE_ORDER",
    "ABLATION_CONFIG_NAMES",
    "DEFAULT_ELITE_YIELD_DENOMINATOR",
    "AblationConfig",
    "AblationAssembly",
    "make_feature_switch",
    "build_ablation_config",
    "all_ablation_configs",
    "AblationAttempt",
    "AblationMetrics",
    "AblationReport",
    "AblationRunResult",
    "AblationRunner",
    "compute_metrics",
    "configs_diff",
    "ablation_config_matrix",
]

#: EliteYield 分母（1000 attempts / run；plan Task 24）。
ELITE_YIELD_DENOMINATOR = 1000
DEFAULT_ELITE_YIELD_DENOMINATOR = ELITE_YIELD_DENOMINATOR

#: plan Task 24 的 10 个配置名（列表即累积 feature tower 的展示顺序）。
FEATURE_ORDER: tuple[str, ...] = (
    "legacy_reference",
    "fitness_v21",
    "bayesian_retrieval",
    "contextual_retriever",
    "schema_space",
    "logic_memory",
    "paradigm_scheduler",
    "trajectory_credit",
    "evi_promotion",
    "next_version_survival",
)
ABLATION_CONFIG_NAMES: tuple[str, ...] = (
    "legacy_reference",
    "plus_fitness_v21",
    "plus_bayesian_retrieval",
    "plus_contextual_retriever",
    "plus_schema_space",
    "plus_logic_memory",
    "plus_paradigm_scheduler",
    "plus_trajectory_credit",
    "plus_evi_promotion",
    "plus_next_version_survival",
)


# ---------------------------------------------------------------------------
# 特性开关位（每个特性的可装配 switch 面）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureSwitch:
    """单个特性的开关位描述（校验 + diff + 可复现序列化）。

    ``probe`` 是装配校验用的引用标记：True 表示「该特性开启后，某个
    dataclass 里对应字段必须为指定值」。None 表示无 probe（纯标记位）。
    """

    name: str
    settings: Mapping[str, Any] = field(default_factory=dict)
    probe: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "settings": dict(self.settings), "probe": dict(self.probe)}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "FeatureSwitch":
        return cls(
            name=str(d.get("name", "")),
            settings=dict(d.get("settings") or {}),
            probe=dict(d.get("probe") or {}),
        )


@dataclass(frozen=True)
class AblationAssembly:
    """一次装配的产物：各特性模块的真实 config 实例（可执行、可注入真实链）。

    字段全部可 None：真实链只消费它打开的组件。dry-run 下调用方应校验
    ``validate()`` 非空（OFFLINE_TEST 语义）。
    """

    config_name: str = ""
    switches: tuple[FeatureSwitch, ...] = ()
    # --- 特性模块配置实例（装配的「落点」）---
    bayesian: Any = None  # BayesianRetrieverConfig
    contextual: Any = None  # ContextualRetrieverConfig
    paradigm: Any = None  # ParadigmSelectionConfig
    promotion: Any = None  # PromotionSurrogateConfig
    opportunity: Any = None  # SearchOpportunityConfig
    orchestrator: Mapping[str, Any] = field(default_factory=dict)  # SearchOrchestrator kwargs
    pipeline: Mapping[str, Any] = field(default_factory=dict)  # PipelineConfig 字段覆盖
    fitness: Mapping[str, Any] = field(default_factory=dict)  # factor_fitness_v2_1 kwargs
    survival: Mapping[str, Any] = field(default_factory=dict)  # SurvivalAttributionModel kwargs
    logic_miner: Any = None  # LogicMiningConfig
    probes: Mapping[str, bool] = field(default_factory=dict)

    def validate(self) -> list[str]:
        """装配自检：返回非空 = 装配缺失/自相矛盾。

        - bayesian.retrieval 开 → 必须有 bayesian config；enabled=True。
        - contextual 开 → contextual config 存在且 enabled=True；且 bayesian
          enabled 保持 True（contextual 在 #3 之上，是超集）。
        - 用 ``enabled`` 哨兵标记「该特性打开」的组件必须同时出现在
          ``switches`` 的 settings 里（settings 与实例字段一致，diff 可枚举）。
        """
        problems: list[str] = []
        sw = {s.name: s for s in self.switches}

        def _expect(sw_name: str, obj: Any, field_name: str, want: Any) -> None:
            if getattr(obj, field_name, None) != want:
                problems.append(
                    f"{self.config_name}: feature {sw_name!r} probe failed — "
                    f"{type(obj).__name__}.{field_name}={getattr(obj, field_name, None)!r} "
                    f"!= {want!r}"
                )

        if "bayesian_retrieval" in sw:
            if self.bayesian is None:
                problems.append(f"{self.config_name}: bayesian_retrieval set but bayesian config None")
            else:
                _expect("bayesian_retrieval", self.bayesian, "enabled", True)
                if getattr(self.bayesian, "enabled", None):
                    _expect("bayesian_retrieval", self.bayesian, "stagnation_enabled", True)
        if "contextual_retriever" in sw:
            if self.contextual is None:
                problems.append(f"{self.config_name}: contextual_retriever set but contextual config None")
            else:
                _expect("contextual_retriever", self.contextual, "enabled", True)
        if "paradigm_scheduler" in sw:
            if self.paradigm is None:
                problems.append(f"{self.config_name}: paradigm_scheduler set but paradigm config None")
            else:
                _expect("paradigm_scheduler", self.paradigm, "enabled", True)
        if "evi_promotion" in sw:
            if self.promotion is None:
                problems.append(f"{self.config_name}: evi_promotion set but promotion config None")
            else:
                _expect("evi_promotion", self.promotion, "enabled", True)
        if "trajectory_credit" in sw:
            if self.paradigm is None:
                problems.append(f"{self.config_name}: trajectory_credit set but paradigm config None")
            else:
                _expect("trajectory_credit", self.paradigm, "repair_enabled", True)
        if "next_version_survival" in sw:
            if self.opportunity is None:
                problems.append(f"{self.config_name}: next_version_survival set but opportunity config None")
        # contextual 是 bayesian 超集：contextual 开而 bayesian.enabled=False → 装配矛盾
        if (
            "contextual_retriever" in sw
            and "bayesian_retrieval" in sw
            and self.contextual is not None
            and self.bayesian is not None
            and not getattr(self.bayesian, "enabled", False)
        ):
            problems.append(
                f"{self.config_name}: contextual_retriever (T10) is a superset of "
                "bayesian_retrieval (T2) — bayesian.enabled must stay True"
            )
        return problems

    def switch_names(self) -> list[str]:
        return [s.name for s in self.switches]

    def to_dict(self) -> dict[str, Any]:
        """可复现序列化（config 实例只留 dataclass 字段差分子集：settings/probes）。"""
        return {
            "config_name": self.config_name,
            "switches": [s.to_dict() for s in self.switches],
            "fitness": dict(self.fitness),
            "orchestrator": dict(self.orchestrator),
            "pipeline": dict(self.pipeline),
            "survival": dict(self.survival),
            "probes": dict(self.probes),
        }


# ---------------------------------------------------------------------------
# 特性 → 开关装配（本模块唯一的「开关位映射表」）
# ---------------------------------------------------------------------------


def _fitness_legacy_kwargs() -> dict[str, Any]:
    """legacy reference 的 fitness 装配：V2.1 全部扩展默认关闭（= V2 逐字一致）。"""
    return {
        "apply_confidence_shrinkage": False,
        "apply_soft_floors": False,
        "requirement_policy": None,
        "dimension_requirement": None,
        "dimension_n_eff": None,
        "metric_n_keys": None,
        "soft_floor": None,
        "k": 64.0,
        "use_v21": False,
    }


def _fitness_v21_kwargs() -> dict[str, Any]:
    """V2.1 特性开启时的装配（扩展打开；soft-floor 用保守默认不宣称最优）。"""
    return {
        "apply_confidence_shrinkage": True,
        "apply_soft_floors": True,
        "requirement_policy": "OPTIONAL_MISSING_PENALTY",
        "dimension_requirement": None,
        "dimension_n_eff": None,
        "metric_n_keys": None,
        "soft_floor": None,
        "k": 64.0,
        "use_v21": True,
    }


def make_feature_switch(name: str, enabled: bool) -> FeatureSwitch | None:
    """按特性名生成 FeatureSwitch（enabled=True 才返回；未知名字 raise）。

    settings = 该特性打开后要写入的各 config 实例字段（真实装配的落点）。
    probe = 装配后校验点（validate() 消费；diff 可枚举的可复现面）。
    """
    if not enabled:
        return None
    name = str(name).strip()
    if name == "legacy_reference":
        # legacy reference = 无任何新特性打开的原始参考（空 switch，仅标记）。
        return FeatureSwitch(name=name, settings={}, probe={})
    if name == "fitness_v21":
        return FeatureSwitch(
            name=name,
            settings={**{"use_v21": True, "apply_confidence_shrinkage": True,
                         "apply_soft_floors": True}},
            probe={"fitness.use_v21": True},
        )
    if name == "bayesian_retrieval":
        return FeatureSwitch(
            name=name,
            settings={"bayesian.enabled": True, "bayesian.stagnation_enabled": True},
            probe={"bayesian.enabled": True, "bayesian.stagnation_enabled": True},
        )
    if name == "contextual_retriever":
        return FeatureSwitch(
            name=name,
            settings={"contextual.enabled": True, "contextual.hierarchical": True},
            probe={"contextual.enabled": True, "contextual.hierarchical": True},
        )
    if name == "schema_space":
        return FeatureSwitch(
            name=name,
            settings={
                "orchestrator.structured_generation": True,
                "orchestrator.multistage_generation": True,
            },
            probe={"orchestrator.structured_generation": True,
                   "orchestrator.multistage_generation": True},
        )
    if name == "logic_memory":
        return FeatureSwitch(
            name=name,
            settings={"logic_miner.enable_llm": False,
                      "logic_miner.min_support": 5,
                      "orchestrator.multistage_generation": True},
            probe={"orchestrator.multistage_generation": True},
        )
    if name == "paradigm_scheduler":
        return FeatureSwitch(
            name=name,
            settings={"paradigm.enabled": True},
            probe={"paradigm.enabled": True},
        )
    if name == "trajectory_credit":
        return FeatureSwitch(
            name=name,
            settings={"paradigm.repair_enabled": True},
            probe={"paradigm.repair_enabled": True},
        )
    if name == "evi_promotion":
        return FeatureSwitch(
            name=name,
            settings={"promotion.enabled": True, "promotion.use_model": True},
            probe={"promotion.enabled": True, "promotion.use_model": True},
        )
    if name == "next_version_survival":
        return FeatureSwitch(
            name=name,
            settings={
                "opportunity.w_survival": 0.15,
                "survival.with_interactions": True,
                "survival.fit_backend": "auto",
            },
            probe={"survival.with_interactions": True},
        )
    raise KeyError(f"unknown ablation feature {name!r}; expected one of {FEATURE_ORDER}")


def _feature_to_index(name: str) -> int:
    for i, f in enumerate(FEATURE_ORDER):
        if f == name:
            return i
    raise KeyError(f"unknown ablation feature {name!r}")


# ---------------------------------------------------------------------------
# AblationConfig：10 配置工厂 + 开关映射
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AblationConfig:
    """一条配置：名字 + 覆盖的（feature → on/off）。

    为保持 diff 可枚举，本类不持有真实 config 实例（避免「dataclass 里藏
    config 实例导致 == 恒 False / diff 不可比」）；真实实例在
    ``AblationAssembly`` 里由 ``build_ablation_config(...).assembly()`` 产出。
    支持 JSON roundtrip（``to_dict`` / ``from_dict``）。
    """

    name: str
    features: Mapping[str, bool] = field(default_factory=dict)
    overrides: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        unknown = [k for k in self.features if k not in FEATURE_ORDER]
        if unknown:
            raise ValueError(f"{name}: unknown feature keys {unknown}")
        if name not in ABLATION_CONFIG_NAMES and name != "custom":
            # 允许自定义名字，但内置名必须拼写正确（防 typo）
            pass
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "features", dict(self.features))
        object.__setattr__(self, "overrides", dict(self.overrides))

    # -- 序列化（可复现） --
    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "features": dict(self.features), "overrides": dict(self.overrides)}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "AblationConfig":
        return cls(
            name=str(d.get("name", "custom")),
            features=dict(d.get("features") or {}),
            overrides=dict(d.get("overrides") or {}),
        )

    @classmethod
    def for_index(cls, index: int, *, name: str | None = None) -> "AblationConfig":
        """第 ``index``（1..10）个 plan 累积配置。index=1 → legacy reference。"""
        if not (1 <= index <= len(ABLATION_CONFIG_NAMES)):
            raise ValueError(f"index must be 1..{len(ABLATION_CONFIG_NAMES)}, got {index}")
        cfg_name = name or ABLATION_CONFIG_NAMES[index - 1]
        feats: dict[str, bool] = {}
        # 累积 tower：前 index 个特性全开
        for i in range(index):
            feats[FEATURE_ORDER[i]] = True
        return cls(name=cfg_name, features=feats)

    @classmethod
    def legacy_reference(cls) -> "AblationConfig":
        return cls.for_index(1, name=ABLATION_CONFIG_NAMES[0])

    # -- 装配 --
    def assembly(self) -> AblationAssembly:
        """把本配置装配为真实 config 实例集合（可执行、可注入真实链）。

        dry-run（OFFLINE_TEST）只调用 ``assembly().validate()`` 校验装配，
        不真跑 campaign。真实路径把返回的 assembly 注入 pipeline/runner。
        """
        sw: list[FeatureSwitch] = []
        enabled_names: list[str] = []
        for f in FEATURE_ORDER:
            if self.features.get(f):
                s = make_feature_switch(f, True)
                if s is not None:
                    sw.append(s)
                    enabled_names.append(f)
        # 装配落点：默认全部 off/legacy，然后按 enable 顺序覆盖
        bayesian: Any = _default_bayesian_config(enabled="bayesian_retrieval" in enabled_names)
        contextual: Any = _default_contextual_config(enabled="contextual_retriever" in enabled_names)
        paradigm = _default_paradigm_config(enabled="paradigm_scheduler" in enabled_names)
        promotion = _default_promotion_config(enabled="evi_promotion" in enabled_names)
        opportunity = _default_opportunity_config()
        fitness = _fitness_v21_kwargs() if "fitness_v21" in enabled_names else _fitness_legacy_kwargs()
        orch: dict[str, Any] = {
            "structured_generation": "schema_space" in enabled_names,
            "multistage_generation": ("schema_space" in enabled_names or "logic_memory" in enabled_names),
        }
        pipe: dict[str, Any] = {"structured_generation": orch["structured_generation"]}
        surv: dict[str, Any] = {
            "with_interactions": "next_version_survival" in enabled_names,
            "fit_backend": "auto",
        }
        logic_cfg: Any = _default_logic_miner_config()
        # 特性缺省装配位已在上方显式写死；这里把声明式 FeatureSwitch.settings
        # 统一应用到实例/dict（settings 与实例字段保持一致，diff 可枚举），
        # 随后 assembly().validate() 二次核对。
        if "bayesian_retrieval" in enabled_names:
            bayesian.enabled = True
            bayesian.stagnation_enabled = True
        if "contextual_retriever" in enabled_names:
            contextual.enabled = True
            contextual.hierarchical = True
        if "paradigm_scheduler" in enabled_names:
            paradigm = dataclasses.replace(paradigm, enabled=True)
        if "trajectory_credit" in enabled_names:
            paradigm = dataclasses.replace(paradigm, repair_enabled=True)
        if "evi_promotion" in enabled_names:
            promotion.enabled = True
            promotion.use_model = True
        if "logic_memory" in enabled_names:
            # Market Logic Library：确定性挖掘（enable_llm 保持 False ——
            # 零 LLM 命名，logic_id 由签名派生，见 Task 13 验收①）。
            orch["multistage_generation"] = True
        if "next_version_survival" in enabled_names:
            opportunity.w_survival = 0.15
            surv["with_interactions"] = True
            surv["fit_backend"] = "auto"
        probes: dict[str, bool] = {
            "fitness_v21": "fitness_v21" in enabled_names,
            "bayesian_retrieval": "bayesian_retrieval" in enabled_names,
            "contextual_retriever": "contextual_retriever" in enabled_names,
            "schema_space": "schema_space" in enabled_names,
            "logic_memory": "logic_memory" in enabled_names,
            "paradigm_scheduler": "paradigm_scheduler" in enabled_names,
            "trajectory_credit": "trajectory_credit" in enabled_names,
            "evi_promotion": "evi_promotion" in enabled_names,
            "next_version_survival": "next_version_survival" in enabled_names,
        }
        # 用户 overrides：覆盖上面任何字段（含关闭某项的复写）
        if self.overrides:
            ov = dict(self.overrides)
            # 简单 key 面：bayesian.* / contextual.* / paradigm.* / promotion.* /
            # opportunity.* / survival.* / fitness.* / orchestrator.* / pipeline.*
            for key, val in ov.items():
                _apply_dotted(self, key, val, locals_=dict(bayesian=bayesian, contextual=contextual,
                                                          paradigm=paradigm, promotion=promotion,
                                                          opportunity=opportunity, fitness=fitness,
                                                          orch=orch, pipe=pipe, surv=surv))
        asm = AblationAssembly(
            config_name=self.name,
            switches=tuple(sw),
            bayesian=bayesian,
            contextual=contextual,
            paradigm=paradigm,
            promotion=promotion,
            opportunity=opportunity,
            orchestrator=orch,
            pipeline=pipe,
            fitness=fitness,
            survival=surv,
            logic_miner=logic_cfg,
            probes=probes,
        )
        problems = asm.validate()
        if problems:
            raise ValueError("\n".join(problems))
        return asm

    def _apply_switch_settings(self, s: FeatureSwitch, *, locals_scope: Any) -> None:
        """[deprecated hook] 声明式 FeatureSwitch.settings 的实际应用在
        ``assembly()`` 内联完成（见 assembly 的逐字段 apply）。本方法保留
        仅作 API 兼容，不执行任何操作。
        """
        del s, locals_scope
        return


def _default_bayesian_config(*, enabled: bool) -> Any:
    from alphaprobe.retrieval.bayesian_retriever import BayesianRetrieverConfig

    return BayesianRetrieverConfig(enabled=enabled, stagnation_enabled=enabled)


def _default_contextual_config(*, enabled: bool) -> Any:
    from alphaprobe.retrieval.contextual_retriever import ContextualRetrieverConfig

    return ContextualRetrieverConfig(enabled=enabled, hierarchical=enabled)


def _default_paradigm_config(*, enabled: bool) -> Any:
    from alphaprobe.search.paradigms import ParadigmSelectionConfig

    # ParadigmSelectionConfig 是 frozen dataclass：用 dataclasses.replace 覆盖。
    return dataclasses.replace(ParadigmSelectionConfig(), enabled=enabled)


def _default_promotion_config(*, enabled: bool) -> Any:
    from alphaprobe.surrogate.promotion import PromotionSurrogateConfig

    return PromotionSurrogateConfig(enabled=enabled, use_model=enabled)


def _default_opportunity_config() -> Any:
    from alphaprobe.retrieval.search_opportunity import SearchOpportunityConfig

    return SearchOpportunityConfig(w_survival=0.15)


def _default_logic_miner_config() -> Any:
    from alphaprobe.research_space.logic_miner import LogicMiningConfig

    return LogicMiningConfig(min_support=5, enable_llm=False)


def _apply_dotted(
    owner: Any, key: str, val: Any, *, locals_: Mapping[str, Any]
) -> None:
    """把 ``a.b.c=val`` 应用到 locals_ 里的对象 ``a``（首段）。"""
    head, _, rest = key.partition(".")
    if head not in locals_:
        raise KeyError(f"override target {head!r} not in {sorted(locals_)}")
    obj = locals_[head]
    if not rest:
        if isinstance(obj, dict):
            raise TypeError("dict target requires dotted path")
        raise TypeError(f"{head} is a dataclass/config; use dotted path")
    parts = rest.split(".")
    cur: Any = obj
    for i, p in enumerate(parts):
        if i == len(parts) - 1:
            if isinstance(cur, dict):
                cur[p] = val
            else:
                setattr(cur, p, val)
        else:
            cur = getattr(cur, p)


def build_ablation_config(index: int, *, name: str | None = None) -> AblationConfig:
    """建第 ``index``（1..10）个内置配置。"""
    return AblationConfig.for_index(index, name=name)


def all_ablation_configs() -> list[AblationConfig]:
    """plan Task 24 的 10 配置（legacy → +next-version Survival）。"""
    return [AblationConfig.for_index(i) for i in range(1, len(ABLATION_CONFIG_NAMES) + 1)]


def ablation_config_matrix() -> list[dict[str, Any]]:
    """10 配置摘要矩阵（装配后 switch names + probe 校验全过）。"""
    out: list[dict[str, Any]] = []
    for cfg in all_ablation_configs():
        asm = cfg.assembly()
        out.append(
            {
                "index": len(out) + 1,
                "config_name": cfg.name,
                "features_on": asm.switch_names(),
                "probes_ok": asm.probes,
            }
        )
    return out


def configs_diff(cfg_a: AblationConfig, cfg_b: AblationConfig) -> dict[str, Any]:
    """两个配置的开关差异（ablation 可比性基础）。

    Returns
    -------
    {"added": [...], "removed": [...], "shared": [...], "a_only_overrides": {...},
     "b_only_overrides": {...}}
    """
    fa = dict(cfg_a.features)
    fb = dict(cfg_b.features)
    all_keys = sorted(set(fa) | set(fb))
    added = [k for k in all_keys if not fa.get(k) and fb.get(k)]
    removed = [k for k in all_keys if fa.get(k) and not fb.get(k)]
    shared = [k for k in all_keys if fa.get(k) and fb.get(k)]
    return {
        "added": added,
        "removed": removed,
        "shared": shared,
        "a_only_overrides": {k: v for k, v in cfg_a.overrides.items() if cfg_b.overrides.get(k) != v},
        "b_only_overrides": {k: v for k, v in cfg_b.overrides.items() if cfg_a.overrides.get(k) != v},
    }


# ---------------------------------------------------------------------------
# 指标聚合（只聚合已评估事实；空 run → 中性空指标，不炸）
# ---------------------------------------------------------------------------


@dataclass
class AblationAttempt:
    """一条 attempt 的指标相关事实（聚合输入；字段宽松兼容 ledger/观测）。"""

    attempt_id: str = ""
    factor_id: str = ""
    formula: str = ""
    parent_ids: tuple[str, ...] = ()
    action_type: str = ""
    paradigm: str = ""
    schema_id: str | None = None
    logic_id: str | None = None
    cluster_id: str | None = None
    #: 评估层级（L0..L5）；elite 判定需要 L3/L4。
    fidelity: str = ""
    elite: bool = False
    duplicate: bool = False
    #: 已评估事实（只聚合，不重算）
    fitness: float | None = None
    long_short_sharpe: float | None = None
    stability: float | None = None
    llm_cost: float = 0.0
    fe_seconds: float = 0.0
    qe_seconds: float = 0.0
    #: 是否 globally novel（相对全局 seen 判定；调用方给出）
    globally_novel: bool = False
    #: next-version survival rate（仅 available 时注入）
    survival_rate: float | None = None

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "AblationAttempt":
        known = {f.name for f in dataclasses.fields(cls)}
        kw: dict[str, Any] = {}
        for k, v in dict(d).items():
            if k in known:
                kw[k] = v
        if "elite" not in kw and ("status" in d or "is_elite" in d):
            kw["elite"] = bool(d.get("is_elite", d.get("status") == "elite"))
        return cls(**kw)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _safe_div(n: float, d: float) -> float:
    if d is None or d <= 0:
        return 0.0
    try:
        return float(n) / float(d)
    except (TypeError, ValueError, ZeroDivisionError):
        return 0.0


def _median(values: Sequence[float | None]) -> float | None:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not vals:
        return None
    s = sorted(vals)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2.0


def _fidelity_at_least(fid: str, level: str) -> bool:
    """fidelity 字符串（L3_search_valid 等）是否达到 level（'L3'/'L4'）。"""
    head = str(fid).split("_", 1)[0].strip().upper()
    order = {"L0": 0, "L1": 1, "L2": 2, "L3": 3, "L4": 4, "L5": 5}
    if head not in order:
        return False
    want = {"L3": 3, "L4": 4}.get(level, 3)
    return order[head] >= want


def compute_metrics(
    attempts: Sequence[AblationAttempt | Mapping[str, Any]],
    *,
    denominator: int = ELITE_YIELD_DENOMINATOR,
) -> dict[str, Any]:
    """从一次 run 的 attempt 序列聚合 plan Task 24 指标。

    口径：
    - ``EliteYield`` = globally-novel L3/L4 elite 数 ÷ denominator × 1000
      （plan：分母 1000 attempts）。
    - ``CostAdjustedEliteYield`` = elite 数 ÷ 折算总成本（归一），再 × 1000。
    - duplicate rate = duplicate attempts / 总 attempts。
    - LLM cost per elite / FE+QE compute per elite。
    - unique cluster/schema/logic coverage = distinct id 计数。
    - median/best L3 FactorFitness、median Long-Short Sharpe、median
      Stability：只对 L3+ attempt 的已评估值取中位/最大（None 跳过；
      无数据 → None，不编造）。
    - next-version survival：attempts 里带 ``survival_rate`` 的 L3+ elite
      平均值；无 → None。

    空/无数据 run → 全中性（数值 0，可空指标 None），绝不抛。
    """
    rows: list[AblationAttempt] = []
    for a in attempts:
        if isinstance(a, AblationAttempt):
            rows.append(a)
        else:
            rows.append(AblationAttempt.from_dict(a))
    n_attempt = len(rows)
    dup = sum(1 for a in rows if a.duplicate)
    #: globally-novel L3/L4 elite
    elites = [
        a for a in rows
        if a.elite and a.globally_novel and _fidelity_at_least(a.fidelity, "L3")
    ]
    n_elite = len(elites)
    #: 只按 L3+ 有评估事实的 attempt 算质量指标
    l3_rows = [a for a in rows if _fidelity_at_least(a.fidelity, "L3")]
    l3_elite = [a for a in elites if _fidelity_at_least(a.fidelity, "L3")]

    llm_total = sum(max(0.0, float(a.llm_cost or 0.0)) for a in rows)
    fe_total = sum(max(0.0, float(a.fe_seconds or 0.0)) for a in rows)
    qe_total = sum(max(0.0, float(a.qe_seconds or 0.0)) for a in rows)
    compute_total = fe_total + qe_total

    # 归一化到 1000 attempts 当量（CostAdjustedEliteYield 口径）
    scale = _safe_div(float(denominator), max(1, n_attempt))
    cost_norm = max(1e-9, (llm_total * scale) + (compute_total * scale))
    cost_adj_yield = _safe_div(float(n_elite) * 1000.0, cost_norm) if n_elite else 0.0
    #: EliteYield（plan Task 24）：globally-novel L3/L4 elite ÷ attempts × 1000。
    #: 分母是 **attempts**（500 attempts、10 elites → 20.0），不是 generated 数。
    elite_yield = (
        _safe_div(float(n_elite) * 1000.0, float(n_attempt)) if n_attempt else 0.0
    )

    schema_ids = sorted({a.schema_id for a in l3_rows if a.schema_id})
    logic_ids = sorted({a.logic_id for a in l3_rows if a.logic_id})
    cluster_ids = sorted({a.cluster_id for a in l3_rows if a.cluster_id})

    surv_rates = [float(a.survival_rate) for a in l3_elite
                  if a.survival_rate is not None and math.isfinite(float(a.survival_rate))]

    return {
        "attempts": n_attempt,
        "generated_factors": n_attempt,  # 仅报告信息；NOT a success metric
        "duplicate_count": dup,
        "duplicate_rate": _safe_div(float(dup), float(n_attempt)) if n_attempt else 0.0,
        "elite_count": n_elite,
        "globally_novel_elite_count": n_elite,
        #: EliteYield 的分母是 attempts（plan 原文）→ elite/attempts × 1000
        "elite_yield_per_1000_attempts": elite_yield,
        "cost_adjusted_elite_yield": cost_adj_yield,
        "llm_cost_total": llm_total,
        "fe_seconds_total": fe_total,
        "qe_seconds_total": qe_total,
        "compute_seconds_total": compute_total,
        "llm_cost_per_elite": _safe_div(llm_total, float(n_elite)) if n_elite else 0.0,
        "compute_seconds_per_elite": _safe_div(compute_total, float(n_elite)) if n_elite else 0.0,
        "median_l3_fitness": _median([a.fitness for a in l3_rows]),
        "best_l3_fitness": max((float(a.fitness) for a in l3_rows
                                if a.fitness is not None and math.isfinite(float(a.fitness))),
                               default=None),
        "median_long_short_sharpe": _median([a.long_short_sharpe for a in l3_rows]),
        "median_stability": _median([a.stability for a in l3_rows]),
        "unique_schema_coverage": len(schema_ids),
        "unique_logic_coverage": len(logic_ids),
        "unique_cluster_coverage": len(cluster_ids),
        "schema_ids": schema_ids,
        "logic_ids": logic_ids,
        "cluster_ids": cluster_ids,
        "next_version_survival_rate": (
            sum(surv_rates) / float(len(surv_rates)) if surv_rates else None
        ),
    }


#: 报告信息列（含排序）。generated_factors 仅信息列，不参与 success。
METRIC_KEYS: tuple[str, ...] = (
    "elite_yield_per_1000_attempts",
    "cost_adjusted_elite_yield",
    "duplicate_rate",
    "median_l3_fitness",
    "best_l3_fitness",
    "median_long_short_sharpe",
    "median_stability",
    "unique_schema_coverage",
    "unique_logic_coverage",
    "unique_cluster_coverage",
    "llm_cost_per_elite",
    "compute_seconds_per_elite",
    "next_version_survival_rate",
)


@dataclass
class AblationRunResult:
    """一次配置的真实（或合成）run 结果。"""

    config: AblationConfig
    attempts: list[AblationAttempt] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_attempts(
        cls,
        config: AblationConfig,
        attempts: Sequence[AblationAttempt | Mapping[str, Any]],
        *,
        denominator: int = ELITE_YIELD_DENOMINATOR,
        meta: Mapping[str, Any] | None = None,
    ) -> "AblationRunResult":
        return cls(
            config=config,
            attempts=[a if isinstance(a, AblationAttempt) else AblationAttempt.from_dict(a)
                      for a in attempts],
            metrics=compute_metrics(attempts, denominator=denominator),
            meta=dict(meta or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "metrics": dict(self.metrics),
            "n_attempts": len(self.attempts),
            "meta": dict(self.meta),
        }


@dataclass
class AblationReport:
    """对比表 + diff（plan Task 24 的 AblationReport 语义）。

    rows[i] = 第 i+1 个配置的指标行；diff 提供相邻配置增量。空输入 → 空表。
    """

    rows: list[dict[str, Any]] = field(default_factory=list)
    diffs: list[dict[str, Any]] = field(default_factory=list)
    config_names: list[str] = field(default_factory=list)

    @classmethod
    def from_results(cls, results: Sequence[AblationRunResult]) -> "AblationReport":
        rows = [dict(r.metrics, **{"config": r.config.name}) for r in results]
        diffs: list[dict[str, Any]] = []
        for a, b in zip(results, results[1:]):
            d = configs_diff(a.config, b.config)
            d["from"] = a.config.name
            d["to"] = b.config.name
            # 指标增量（仅 numeric）
            delta: dict[str, Any] = {}
            for k in METRIC_KEYS:
                va = a.metrics.get(k)
                vb = b.metrics.get(k)
                if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                    delta[k] = round(float(vb) - float(va), 6)
            d["metric_delta"] = delta
            diffs.append(d)
        return cls(rows=rows, diffs=diffs, config_names=[r.config.name for r in results])

    def to_dict(self) -> dict[str, Any]:
        return {"configs": self.config_names, "rows": self.rows, "diffs": self.diffs}

    def to_markdown(self) -> str:
        """生成对比表 markdown（≤15 行/配置；供 CLI --report 直接打印）。"""
        lines = ["| config | " + " | ".join(METRIC_KEYS) + " |"]
        lines.append("|" + "---|" * (len(METRIC_KEYS) + 1))
        for row in self.rows:
            cells = [str(row.get("config", ""))]
            for k in METRIC_KEYS:
                v = row.get(k)
                if v is None:
                    cells.append("—")
                elif isinstance(v, float):
                    cells.append(f"{v:.4g}")
                else:
                    cells.append(str(v))
            lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# AblationRunner：配置 → 开关装配 → run 回调注入 → 指标聚合
# ---------------------------------------------------------------------------


class AblationRunner:
    """把配置装配成真实组件并驱动一次 run 的最小编排。

    ``run`` 接受注入的 ``run_fn(config, assembly) -> attempts`` 回调（真实
    campaign 的适配器；本任务真实路径预留），对装配结果先 ``validate()``
    再回调。dry-run 语义（OFFLINE_TEST）：``run_fn`` 省略时用
    ``dry_run_attempts``（默认 []）产出中性指标，**绝不触碰 LLM / FE / QE**。
    """

    def __init__(
        self,
        *,
        denominator: int = ELITE_YIELD_DENOMINATOR,
        run_fn: Callable[[AblationConfig, AblationAssembly], Sequence[Mapping[str, Any]]] | None = None,
    ) -> None:
        self.denominator = int(denominator)
        self.run_fn = run_fn

    def run(self, config: AblationConfig) -> AblationRunResult:
        asm = config.assembly()  # 装配即校验（失败 raise，不静默）
        if self.run_fn is None:
            # dry-run：不真跑 campaign（无 LLM / 无数据 / 无 FE/QE）
            attempts: list[AblationAttempt] = []
        else:
            raw = self.run_fn(config, asm)
            attempts = [a if isinstance(a, AblationAttempt) else AblationAttempt.from_dict(a)
                        for a in (raw or [])]
        return AblationRunResult.from_attempts(
            config, attempts, denominator=self.denominator
        )

    def run_all(self, configs: Sequence[AblationConfig] | None = None) -> AblationReport:
        cfgs = list(configs) if configs is not None else all_ablation_configs()
        results = [self.run(c) for c in cfgs]
        return AblationReport.from_results(results)

    def dry_run_all(self) -> AblationReport:
        """装配 10 配置并产出中性指标表（不真跑任何 campaign）。"""
        return self.run_all(all_ablation_configs())
