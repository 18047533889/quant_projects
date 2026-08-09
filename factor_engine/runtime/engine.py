"""因子引擎运行时入口：编译、执行与物化的一站式 API。

``FactorEngine`` 是 factor_engine 的核心门面类，串联 DSL 解析、IR 分析、
逻辑计划优化、后端执行及因子湖物化全流程。

单因子编译链::

    Factor.expr  →  Analyzer.lower  →  IRNode
                 →  Optimizer       →  IRNode（可选规则）
                 →  Lowerer         →  PlanNode
                 →  Backend.execute →  MultiIndex Series

多因子批跑（``run_many``）::

    多因子 compile → CSE 提取 shared_nodes → 先算共享子树 → 再算各因子根

主要入口方法：

- ``compile`` / ``run``：单因子编译与执行
- ``run_many`` / ``run_many_parallel``：多因子批跑（支持 CSE）
- ``run_from_config`` / ``materialize_from_config``：YAML 一键运行
- ``materialize`` / ``materialize_incremental``：执行并落盘到因子湖

运行模式（``run_mode``）由 ``production_policy.resolve_run_mode`` 解析；
production 模式下强制 input_dq、auto_warmup、PIT 及算子白名单等约束。
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from api.dsl_parser import parse_factor
from api.factor import Factor
from backend.context import ExecutionContext
from backend.factory import build_backend
from ir.analyzer import AnalysisResult, Analyzer
from logging_utils import get_logger
from planner.cse import apply_cse
from planner.dag import (
    DAGPlan,
    FactorExecutionScope,
    FactorPlan,
    assert_unique_factor_names,
)
from planner.logical_plan import PlanNode
from planner.lowerer import Lowerer
from planner.optimizer import Optimizer
from runtime.config import FactorEngineConfig, load_config
from runtime.env_bootstrap import bootstrap_runtime_env
from runtime.perf_config import PerfConfig
from storage.cache import CacheManager, PersistentPlanCache
from storage.data_scope import compute_data_scope
from storage.factory import build_data_source
from storage.materializer import ParquetMaterializer

logger = get_logger("runtime.engine")

#: run_mode 严格白名单（#6）；任何其他值（含 typo）在引擎构造时直接抛
#: ``ValueError``，不允许静默回落为 research。
VALID_RUN_MODES = frozenset({"production", "research", "paper"})


def _validate_run_mode(mode: str) -> None:
    """严格校验 run_mode：非法值启动即失败。"""
    if mode not in VALID_RUN_MODES:
        raise ValueError(
            f"非法 run_mode={mode!r}；合法值 {sorted(VALID_RUN_MODES)}"
        )


def _scope_from_factor(factor: Any) -> FactorExecutionScope:
    """从 ``Factor`` 对象推断执行作用域；属性取不到时使用默认值（#321）。"""
    return FactorExecutionScope(
        frequency=str(getattr(factor, "freq", None) or "1d"),
        universe_id=str(getattr(factor, "universe", None) or "ALL"),
        market=str(getattr(factor, "market", None) or "A"),
        calendar_id=str(
            getattr(factor, "calendar_id", None)
            or getattr(factor, "calendar", None)
            or ""
        ),
        source_scope_hash=str(getattr(factor, "source_scope_hash", None) or ""),
        decision_time_policy=str(getattr(factor, "decision_time_policy", None) or ""),
    )


def _scope_namespace(scope_key: str) -> str:
    """将作用域键哈希为 sid 前缀，隔离跨作用域 CSE 共享。"""
    import hashlib

    digest = hashlib.sha256(scope_key.encode("utf-8")).hexdigest()[:16]
    return f"scope:{digest}:"


def _namespace_plan_refs(node: PlanNode, namespace: str) -> PlanNode:
    """把子树内所有 ``plan_ref`` 的 sid 加上作用域命名空间前缀。"""
    if node.op == "plan_ref":
        sid = (node.attrs or {}).get("sid")
        if sid:
            return PlanNode(
                op="plan_ref",
                attrs={**node.attrs, "sid": f"{namespace}{sid}"},
                inputs=[],
                semantic_attrs=dict(node.semantic_attrs),
            )
        return node
    return PlanNode(
        op=node.op,
        attrs=dict(node.attrs),
        semantic_attrs=dict(node.semantic_attrs),
        inputs=[_namespace_plan_refs(c, namespace) for c in node.inputs],
        node_id=node.node_id,
    )


#: R9-P0-011: 截面（cross-sectional）算子 —— 这些算子在同一时点的全部股票上
#: 求值；若数据源喂入全市场而因子声明了 scoped universe，结果会静默在错误的
#: 股票池上计算（``Factor.universe=CSI300`` 却对全 A rank 的严重截面 bug）。
#: ``cs_`` / ``group_`` 前缀按命名空间覆盖其余算子（含 cs_knn_*）。
_CROSS_SECTIONAL_OPS = frozenset(
    {
        "rank",
        "rank_pct",
        "zscore",
        "scale",
        "normalize",
        "winsorize",
        "quantile",
        "neutralize",
        "size_neutralize",
        "industry_size_neutralize",
        "industry_neutralize",
        "industry_rank",
    }
)


def _plan_has_cross_sectional_ops(plan: Any) -> bool:
    """递归检测逻辑计划是否含截面算子（rank/zscore/neutralize/group/CS/kNN）。

    优先级：前缀（``cs_`` / ``group_``）→ 静态集合 → OperatorRegistry 分类
    （``category`` ∈ {``cross_sectional``, ``group_neutralization``}）。
    Registry 未加载 / 未知算子时静默跳过分类（保守不误报）。
    """
    if plan is None:
        return False
    op = str(getattr(plan, "op", "") or "")
    if op.startswith("cs_") or op.startswith("group_"):
        return True
    if op in _CROSS_SECTIONAL_OPS:
        return True
    if op:
        try:
            from cleaned_operators.registry import OperatorRegistry

            canonical = OperatorRegistry.resolve_canonical(op)
            if (
                canonical in _CROSS_SECTIONAL_OPS
                or canonical.startswith("cs_")
                or canonical.startswith("group_")
            ):
                return True
            impl = OperatorRegistry.get(canonical)
            category = str(getattr(getattr(impl, "metadata", None), "category", "") or "")
            if category in {"cross_sectional", "group_neutralization"}:
                return True
        except Exception:
            pass
    for child in getattr(plan, "inputs", ()) or ():
        if _plan_has_cross_sectional_ops(child):
            return True
    return False


def _data_source_scoped(data_source: Any) -> bool:
    """数据源是否已把截面限定到某个股票子集（而非全市场）。

    数据层支持 universe 过滤时（``instrument_filter`` 非空 / ``universe`` 属性
    非 ALL），数据源即为 universe 权威：截面在这些已过滤的股票上计算是正确的，
    execution-contract gate 直接放行。
    """
    if data_source is None:
        return False
    inst = getattr(data_source, "instrument_filter", None)
    if inst:
        return True
    univ = getattr(data_source, "universe", None)
    if univ and str(univ).strip().upper() not in {"", "ALL"}:
        return True
    inner = getattr(data_source, "inner", None)
    if inner is not None and inner is not data_source:
        return _data_source_scoped(inner)
    return False


def _is_whole_market_universe(scope: FactorExecutionScope) -> bool:
    """universe 标签是否表示「整个市场」（而非 scoped 子集股票池）。

    - ``ALL`` / 空 → 全市场；
    - 与 ``market`` 相同（如 ``universe="A"`` + ``market="A"``）→ 全市场；
    - 以 ``_ALL`` 结尾（``ASHARE_ALL`` / ``US_MASSIVE_ALL``）→ 全市场；
    - 可被 ``infer_market`` 识别为市场（``ASHARE*`` / ``US_*`` 等）→ 全市场。
    """
    univ = str(scope.universe_id or "").strip().upper()
    if univ in {"", "ALL"}:
        return True
    if univ == str(scope.market or "").strip().upper():
        return True
    if univ.endswith("_ALL"):
        return True
    try:
        from storage.trading_calendar import infer_market

        if infer_market(universe=scope.universe_id):
            return True
    except Exception:
        pass
    return False


def assert_execution_scope_contract(
    scope: FactorExecutionScope,
    plan: Any,
    *,
    factor_name: str,
    data_source: Any = None,
) -> None:
    """R9-P0-011 execution-contract gate：scoped-universe 因子不得在全源上算截面。

    只有同时满足以下条件才 fail-closed：

      * ``universe_id`` 不是全市场标签（``ALL`` / 与 ``market`` 相同 / ``*_ALL`` /
        可被 ``infer_market`` 识别为市场）；
      * 计划含截面算子（rank/zscore/neutralize/group/CS-regression/kNN）；
      * 数据源未显式限定股票子集（无 ``instrument_filter`` 等）。

    数据层已支持 universe 过滤时（``instrument_filter``），数据源即为 universe
    权威，直接放行（选项 a）；否则抛 ``ProductionPolicyViolation``（选项 b），
    绝不静默在全源上计算截面。后续接线 DataAccess universe-filter 以应用
    命名 universe mask（documented follow-up）。
    """
    if _is_whole_market_universe(scope):
        return
    if not _plan_has_cross_sectional_ops(plan):
        return
    if _data_source_scoped(data_source):
        return
    from runtime.production_policy import ProductionPolicyViolation

    raise ProductionPolicyViolation(
        f"execution-scope contract violation: factor '{factor_name}' declares "
        f"scoped universe universe_id={scope.universe_id!r} "
        f"(market={scope.market!r}) but the plan computes cross-sectional "
        f"operator(s) over the FULL source. Cross-sectional execution on the full "
        f"source is disallowed for a scoped universe. Configure the data source to "
        f"serve the scoped pool (e.g. instrument_filter) or set universe='ALL'. "
        f"(follow-up: wire the DataAccess universe-filter to apply the named-universe mask)"
    )


class FactorEngine:
    """因子引擎：注入后端与数据源，对 :class:`api.factor.Factor` 做编译与执行。

    构造时自动引导运行时环境（``.env``）并解析 ``run_mode``（research /
    production）。持有 ``Analyzer``、``Lowerer``、``Optimizer`` 编译组件及
    可选列级/计划缓存。

    Attributes:
        backend: 执行后端（Pandas / Polars / Hybrid 等）。
        data_source: 统一数据读取入口。
        cache: 列级或计划缓存；``None`` 表示每次全量计算。
        run_mode: 当前运行模式字符串。
        analyzer: Expr → IR 分析器。
        lowerer: IR → 逻辑计划转换器。
        optimizer: 逻辑计划优化器。
    """

    def __init__(self, backend, data_source, cache=None, *, run_mode: str | None = None) -> None:
        """构造因子引擎实例。

        Args:
            backend: 由 ``build_backend`` 构造的执行后端。
            data_source: 由 ``build_data_source`` 构造的数据源。
            cache: 可选 ``CacheManager`` 或 ``PersistentPlanCache``。
            run_mode: 显式运行模式；缺省从环境变量解析。
        """
        bootstrap_runtime_env()
        from runtime.production_policy import resolve_run_mode

        self.backend = backend  # PandasBackend / PolarsBackend / …，由 build_backend 构造
        self.data_source = data_source  # 从 parquet 等拉 MultiIndex 面板的统一入口
        self.cache = cache  # 列级缓存；无则每次 execute 全量算
        self.run_mode = resolve_run_mode(run_mode)
        # #6：严格 run_mode 校验——非法值（含 typo）启动即失败，不静默回落。
        _validate_run_mode(self.run_mode)
        # R9-P0-001: the Analyzer must be constructed WITH the production policy
        # from THIS engine's run_mode.  Previously ``Analyzer()`` defaulted to
        # production=False, so the production typed-field gate (unknown raw column
        # rejection, WS-C #273) was silently OFF in every real production
        # ``FactorEngine`` compile.  A run_mode != production still gets a
        # research-policy analyzer.
        from runtime.production_policy import is_production_mode

        self.analyzer = Analyzer(production=is_production_mode(self.run_mode))
        self.lowerer = Lowerer()  # IR → 逻辑计划树
        self.optimizer = Optimizer()  # 计划级优化（常折叠等）

    @staticmethod
    def _sync_production_env(run_mode: str | None = None) -> None:
        """Deprecated compatibility hook; engine construction is context-local."""
        return None

    def compile(self, factor: Factor, *, pit_enforce: bool = False, pit_forbid_forward_fill: bool = False):
        """将因子表达式编译为可执行的逻辑计划。

        流程：``Analyzer.lower`` → ``Lowerer.to_logical_plan`` →
        ``Optimizer.optimize``；production 模式下附加算子白名单与 fast path 审计。

        Args:
            factor: 待编译因子。
            pit_enforce: 是否强制 Point-in-Time 安全审计。
            pit_forbid_forward_fill: PIT 审计是否禁止前向填充算子。

        Returns:
            ``(optimized_plan, analysis)`` 元组，分别为逻辑计划根节点与分析结果。

        Raises:
            ProductionPolicyViolation: production 模式下计划不合规。
            PITAuditError: PIT 审计失败（``pit_enforce=True`` 时）。
        """
        from runtime.production_policy import is_production_mode

        pit_enforce = bool(pit_enforce or is_production_mode(self.run_mode))
        started_at = time.perf_counter()
        logger.info("开始编译因子 '%s'", factor.name)
        analysis = self.analyzer.lower(factor.expr)
        if pit_enforce:
            from runtime.pit_audit import assert_pit_safe

            assert_pit_safe(
                analysis.ir,
                enforce=True,
                forbid_forward_fill=pit_forbid_forward_fill,
            )
        logical_plan = self.lowerer.to_logical_plan(analysis.ir)
        optimized_plan = self.optimizer.optimize(
            logical_plan, production=is_production_mode(self.run_mode)
        )
        from runtime.production_policy import (
            assert_no_unapproved_map_groups_in_production,
            assert_production_fastpath_plan,
            assert_production_plan_ops,
        )

        assert_production_plan_ops(
            optimized_plan,
            mode=self.run_mode,
            context=f"compile:{factor.name}",
        )
        assert_no_unapproved_map_groups_in_production(
            optimized_plan,
            mode=self.run_mode,
            context=f"compile:{factor.name}",
        )
        assert_production_fastpath_plan(
            optimized_plan,
            mode=self.run_mode,
            context=f"compile:{factor.name}",
        )
        logger.info(
            "完成编译因子 '%s'，lookback=%s，耗时 %.2fs",
            factor.name,
            getattr(analysis, "lookback", None),
            time.perf_counter() - started_at,
        )
        return optimized_plan, analysis

    def _dag_from_factors(
        self,
        factors: Sequence[Factor],
        *,
        enable_cse: bool | None = None,
        perf: PerfConfig | None = None,
        pit_enforce: bool = False,
        pit_forbid_forward_fill: bool = False,
    ) -> tuple[DAGPlan, dict[str, AnalysisResult]]:
        """编译多因子并可选 CSE；每个因子只 ``compile`` 一次。

        ``pit_enforce=True`` 时在编译阶段对每个因子做 PIT 安全审计（``assert_pit_safe``
        是纯审计、不改计划），因此批跑也能满足 PIT 而无需退化为逐因子 ``run()``。

        CSE / rolling-CSE 按 :class:`FactorExecutionScope` 分组执行：不同
        freq/universe/market/calendar 执行作用域的因子不共享子树（#321），
        且重复 ``factor.name`` 在此 fail-fast（#322）。
        """
        perf = perf or PerfConfig.from_env()
        if enable_cse is None:
            enable_cse = perf.enable_cse
        # #322：重复 factor name 在编译前 fail-fast（analyses/roots 以 name 为键）。
        names = [f.name for f in factors]
        assert_unique_factor_names(names)
        plans: list[PlanNode] = []
        scopes: list[FactorExecutionScope] = []
        analyses: dict[str, AnalysisResult] = {}
        for factor in factors:
            plan, analysis = self.compile(
                factor,
                pit_enforce=pit_enforce,
                pit_forbid_forward_fill=pit_forbid_forward_fill,
            )
            plans.append(plan)
            analyses[factor.name] = analysis
            scope = _scope_from_factor(factor)
            scopes.append(scope)
            # R9-P0-011: execution-contract gate —— scoped-universe 因子不得在
            # 全源上算截面；fail-closed 而非静默在错误股票池上计算。
            assert_execution_scope_contract(
                scope,
                plan,
                factor_name=factor.name,
                data_source=self.data_source,
            )
        if enable_cse and len(plans) > 0:
            new_plans, shared = self._cse_by_scope(plans, scopes)
        else:
            new_plans, shared = plans, {}
        roots = [
            FactorPlan(factor_name=n, root=r, execution_scope=s)
            for n, r, s in zip(names, new_plans, scopes, strict=True)
        ]
        return DAGPlan(roots=roots, shared_nodes=shared), analyses

    @staticmethod
    def _cse_by_scope(
        plans: list[PlanNode],
        scopes: list[FactorExecutionScope],
    ) -> tuple[list[PlanNode], dict[str, PlanNode]]:
        """按执行作用域分组执行结构 CSE + rolling CSE（#321）。

        同一作用域内的因子才共享子树；不同作用域的因子不交叉。多作用域时
        对每组 shared 的 sid 加作用域命名空间前缀，避免同结构子树因 sid 相同
        而跨作用域互相引用。单一作用域时保持既有行为（不做前缀）。

        参数：
            plans: 各因子逻辑计划根列表
            scopes: 与 ``plans`` 对齐的每个因子的执行作用域

        返回：
            ``(改写后的根列表, shared_nodes 字典)``
        """
        from planner.rolling_cse import apply_rolling_cse

        groups: dict[str, list[int]] = {}
        for i, scope in enumerate(scopes):
            groups.setdefault(scope.scope_key(), []).append(i)

        if len(groups) <= 1:
            # 单一执行作用域：等价于旧的全局 CSE，sid 保持纯结构键。
            new_plans, shared = apply_cse(list(plans))
            return apply_rolling_cse(new_plans, existing_shared=shared)

        result: dict[int, PlanNode] = {}
        shared: dict[str, PlanNode] = {}
        for scope_key, idxs in groups.items():
            ns = _scope_namespace(scope_key)
            group_plans = [plans[i] for i in idxs]
            group_new, group_shared = apply_cse(group_plans)
            group_new, group_shared = apply_rolling_cse(
                group_new, existing_shared=group_shared
            )
            # 命名空间化：shared 键与所有 plan_ref 的 sid 统一加前缀，隔离跨作用域。
            group_shared = {
                f"{ns}{sid}": _namespace_plan_refs(node, ns)
                for sid, node in group_shared.items()
            }
            group_new = [_namespace_plan_refs(p, ns) for p in group_new]
            shared.update(group_shared)
            for j, plan in zip(idxs, group_new):
                result[j] = plan
        new_plans = [result[i] for i in range(len(plans))]
        return new_plans, shared

    def compile_many(
        self,
        factors: Sequence[Factor],
        *,
        enable_cse: bool | None = None,
        perf: PerfConfig | None = None,
    ) -> DAGPlan:
        """多因子编译：可选公共子表达式消除（CSE）。

        对多个因子分别 ``compile`` 后，将重复子树提取到 ``DAGPlan.shared_nodes``，
        根计划通过 ``plan_ref`` 引用共享节点，供 ``run_many`` 一次性物化。

        Args:
            factors: 待编译因子序列。
            enable_cse: 是否启用 CSE；``None`` 时取 ``perf.enable_cse``。
            perf: 性能配置；缺省从环境变量加载。

        Returns:
            含 ``roots`` 与 ``shared_nodes`` 的 ``DAGPlan``。
        """
        dag, _ = self._dag_from_factors(
            factors, enable_cse=enable_cse, perf=perf
        )
        return dag

    def analyze_batch(
        self,
        factors: Sequence[Factor],
        *,
        enable_cse: bool | None = None,
        perf: PerfConfig | None = None,
    ) -> dict[str, Any]:
        """编译多因子并返回 DAG 与列依赖批图。

        除 ``compile_many`` 的 DAG 外，额外构建 ``batch_graph`` 供调度器
        确定并行层与依赖顺序。

        Returns:
            含 ``dag``、``analyses``、``batch_graph`` 的字典。
        """
        from planner.dependency_graph import build_factor_batch_graph

        dag, analyses = self._dag_from_factors(
            factors, enable_cse=enable_cse, perf=perf
        )
        graph = build_factor_batch_graph(factors, analyses)
        return {
            "dag": dag,
            "analyses": analyses,
            "batch_graph": graph.to_dict(),
        }

    @classmethod
    def _build_cache(cls, config: FactorEngineConfig, data_source) -> CacheManager | None:
        """根据引擎配置与数据源作用域构造缓存管理器。"""
        if not config.engine.enable_cache and not config.engine.plan_cache_dir:
            return None
        scope = compute_data_scope(data_source)
        if config.engine.plan_cache_dir:
            return PersistentPlanCache(config.engine.plan_cache_dir, data_scope=scope)
        return CacheManager(data_scope=scope)

    @classmethod
    def from_loaded_config(cls, config: FactorEngineConfig):
        """从已解析的 ``FactorEngineConfig`` 构造 engine 与 factor。

        供 pipeline 编排层或测试复用，避免重复读 YAML。

        Returns:
            ``(engine, factor)`` 元组。
        """
        backend = build_backend(config.backend.type)
        data_source = build_data_source(config.data_source)
        cache = cls._build_cache(config, data_source)
        factor = parse_factor(
            config.factor.expr,
            name=config.factor.name,
            freq=config.factor.freq,
            universe=config.factor.universe,
            description=config.factor.description,
            surface=getattr(config.factor, "surface", "daily") or "daily",
        )
        logger.info(
            "配置对象加载完成: factor=%s, backend=%s, data_source=%s, cache=%s",
            factor.name,
            type(backend).__name__,
            type(data_source).__name__,
            "on" if cache is not None else "off",
        )
        return cls(
            backend=backend,
            data_source=data_source,
            cache=cache,
            run_mode=config.run.mode,
        ), factor

    @classmethod
    def from_config(cls, config_path: str | Path, *, profile: str | None = None):
        """从 YAML 配置文件构造 engine、factor 与 config。

        Args:
            config_path: YAML 配置文件路径。
            profile: 可选 profile 名称，覆盖 YAML 内 ``profile`` 键。

        Returns:
            ``(engine, factor, config)`` 三元组。
        """
        logger.info("加载配置文件: %s", config_path)
        config = load_config(config_path, profile=profile)
        engine, factor = cls.from_loaded_config(config)
        return engine, factor, config

    @classmethod
    def run_from_config(cls, config_path: str | Path, *, profile: str | None = None):
        """一键从 YAML 配置文件执行单因子。

        production 模式或 ``pipeline.batched_engine=True`` 时走 ``run_many``
        批路径；否则直接 ``run``。返回字典附带 ``config`` 对象。

        Args:
            config_path: YAML 配置文件路径。
            profile: 可选 profile 名称。

        Returns:
            含 ``factor``、``analysis``、``plan``、``result``、``config`` 的字典。
        """
        logger.info("开始从配置执行因子: %s", config_path)
        engine, factor, config = cls.from_config(config_path, profile=profile)
        from runtime.config_runtime import resolve_run_kwargs

        opts = resolve_run_kwargs(config)
        run_kwargs = opts.to_run_kwargs()
        if config.pipeline.batched_engine or config.run.mode == "production":
            batch = engine.run_many([factor], **run_kwargs)
            fp = next(r for r in batch["dag"].roots if r.factor_name == factor.name)
            result = {
                "factor": factor,
                "analysis": batch["analyses"][factor.name],
                "plan": fp.root,
                "result": batch["results"][factor.name],
                "config": config,
            }
            if batch.get("input_dq") is not None:
                result["input_dq"] = batch["input_dq"]
        else:
            result = engine.run(factor, **run_kwargs)
            result["config"] = config
        logger.info("完成从配置执行因子: %s", factor.name)
        return result

    @classmethod
    def _run_factor_with_resolved_kwargs(
        cls,
        engine: FactorEngine,
        factor: Factor,
        config: FactorEngineConfig,
        path: str | Path,
        *,
        pipeline_overrides: Any | None = None,
    ) -> dict[str, Any]:
        """单配置文件跑一个因子：合并 pipeline overrides 后 ``engine.run``。"""
        from runtime.config_runtime import resolve_run_kwargs_for_pipeline

        opts = resolve_run_kwargs_for_pipeline(config, pipeline_overrides)
        one = engine.run(factor, **opts.to_run_kwargs())
        one["config"] = config
        one["config_path"] = str(path)
        return one

    @classmethod
    def _run_many_batch_from_config_group(
        cls,
        group: list[tuple[FactorEngine, Factor, FactorEngineConfig, str | Path]],
        *,
        enable_cse: bool | None,
        parallel: bool,
        n_jobs: int | None,
        pipeline_overrides: Any | None,
        results: dict[str, Any],
        runs: dict[str, Any],
        configs: dict[str, FactorEngineConfig],
    ) -> None:
        """同 data_scope + run kwargs 的一组配置：单条直接 run，多条 ``run_many``/并行。"""
        from runtime.config_runtime import config_run_batch_key, resolve_run_kwargs_for_pipeline

        batches: dict[
            tuple[Any, ...],
            list[tuple[FactorEngine, Factor, FactorEngineConfig, str | Path]],
        ] = {}
        for item in group:
            batches.setdefault(
                config_run_batch_key(item[2], pipeline=pipeline_overrides),
                [],
            ).append(item)

        for batch in batches.values():
            if len(batch) == 1:
                engine, factor, config, path = batch[0]
                one = cls._run_factor_with_resolved_kwargs(
                    engine,
                    factor,
                    config,
                    path,
                    pipeline_overrides=pipeline_overrides,
                )
                results[factor.name] = one["result"]
                runs[factor.name] = one
                configs[factor.name] = config
                continue

            base_engine, _, base_config, _ = batch[0]
            opts = resolve_run_kwargs_for_pipeline(base_config, pipeline_overrides)
            run_kwargs = {
                "enable_cse": enable_cse,
                **opts.to_run_kwargs(),
            }
            factors = [f for _, f, _, _ in batch]
            if parallel:
                batch_out = base_engine.run_many_parallel(
                    factors,
                    n_jobs=n_jobs,
                    **run_kwargs,
                )
            else:
                batch_out = base_engine.run_many(factors, **run_kwargs)
            results.update(batch_out["results"])
            for engine, factor, config, path in batch:
                configs[factor.name] = config
                runs[factor.name] = {
                    "factor": factor,
                    "analysis": batch_out["analyses"][factor.name],
                    "result": batch_out["results"][factor.name],
                    "config": config,
                    "config_path": str(path),
                }

    @classmethod
    def run_many_from_config(
        cls,
        config_paths: Sequence[str | Path],
        *,
        enable_cse: bool | None = None,
        parallel: bool = False,
        n_jobs: int | None = None,
        profile: str | None = None,
        pipeline_overrides: Any | None = None,
    ) -> dict[str, Any]:
        """从多个 YAML 配置文件批量执行因子。

        按数据源作用域（``config_data_scope_key``）分组，组内再按 run kwargs
        子分组；同组多因子共享 ``run_many`` 以启用 CSE。

        Args:
            config_paths: 配置文件路径序列。
            enable_cse: 是否启用 CSE。
            parallel: 是否对因子根并行求值。
            n_jobs: 并行 worker 数。
            profile: 可选 profile 名称。
            pipeline_overrides: pipeline 层覆盖参数。

        Returns:
            含 ``results``、``runs``、``configs`` 的字典。
        """
        from runtime.config_runtime import config_data_scope_key

        loaded: list[tuple[FactorEngine, Factor, FactorEngineConfig, str | Path]] = []
        for path in config_paths:
            engine, factor, config = cls.from_config(path, profile=profile)
            loaded.append((engine, factor, config, path))
        if not loaded:
            return {"results": {}, "runs": {}, "configs": {}}

        groups: dict[str, list[tuple[FactorEngine, Factor, FactorEngineConfig, str | Path]]] = {}
        for item in loaded:
            key = config_data_scope_key(item[2])
            groups.setdefault(key, []).append(item)

        results: dict[str, Any] = {}
        runs: dict[str, Any] = {}
        configs: dict[str, FactorEngineConfig] = {}

        for group in groups.values():
            cls._run_many_batch_from_config_group(
                group,
                enable_cse=enable_cse,
                parallel=parallel,
                n_jobs=n_jobs,
                pipeline_overrides=pipeline_overrides,
                results=results,
                runs=runs,
                configs=configs,
            )

        return {"results": results, "runs": runs, "configs": configs}

    @classmethod
    def run_many_from_config_parallel(
        cls,
        config_paths: Sequence[str | Path],
        *,
        enable_cse: bool | None = None,
        n_jobs: int | None = None,
        profile: str | None = None,
        pipeline_overrides: Any | None = None,
    ) -> dict[str, Any]:
        """从多个 YAML 配置文件批量执行因子（根节点并行）。

        等价于 ``run_many_from_config(..., parallel=True)``。
        """
        return cls.run_many_from_config(
            config_paths,
            enable_cse=enable_cse,
            parallel=True,
            n_jobs=n_jobs,
            profile=profile,
            pipeline_overrides=pipeline_overrides,
        )

    @classmethod
    def _materialize_one_from_config_item(
        cls,
        engine: FactorEngine,
        factor: Factor,
        config: FactorEngineConfig,
        path: str | Path,
        opts: Any | None = None,
        *,
        pipeline_overrides: Any | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """落盘单个 config 项，返回 ``(factor_name, materialize_output)``。"""
        from runtime.config_runtime import resolve_materialize_kwargs_for_pipeline

        if opts is None:
            opts = resolve_materialize_kwargs_for_pipeline(config, pipeline_overrides)
        out = engine.materialize(factor, **opts.to_engine_materialize_kwargs())
        out["config"] = config
        out["config_path"] = str(path)
        return factor.name, out

    @classmethod
    def _materialize_batch_from_config_group(
        cls,
        group: list[tuple[FactorEngine, Factor, FactorEngineConfig, str | Path]],
        outputs: dict[str, Any],
        *,
        batch_run: bool,
        enable_cse: bool | None,
        pipeline_overrides: Any | None,
        parallel: bool,
        n_jobs: int | None,
    ) -> None:
        from runtime.config_runtime import (
            config_materialize_batch_key,
            resolve_materialize_kwargs_for_pipeline,
        )
        from runtime.materialize_service import (
            can_batch_materialize_compute,
            execute_materialize_from_resolved,
        )

        batches: dict[
            tuple[Any, ...],
            list[tuple[FactorEngine, Factor, FactorEngineConfig, str | Path]],
        ] = {}
        for item in group:
            batches.setdefault(
                config_materialize_batch_key(item[2], pipeline=pipeline_overrides),
                [],
            ).append(item)

        for batch in batches.values():
            base_opts = resolve_materialize_kwargs_for_pipeline(batch[0][2], pipeline_overrides)
            use_batch_run = batch_run and len(batch) > 1 and can_batch_materialize_compute(base_opts)
            if use_batch_run:
                engine, _, _, _ = batch[0]
                run_kwargs = {"enable_cse": enable_cse, **base_opts.to_run_kwargs()}
                factors = [f for _, f, _, _ in batch]
                meta = {
                    f.name: (f, config, path)
                    for (_, f, config, path) in batch
                }
                if not parallel:
                    # Phase 5 R17：流式物化——先编译一次拿 analyses，再
                    # run_many_iter 每算完一个因子立即落盘，结果不全部驻留。
                    dag, analyses = engine._dag_from_factors(
                        factors,
                        enable_cse=enable_cse,
                    )
                    for name, result, _ in engine.run_many_iter(
                        factors,
                        precompiled=(dag, analyses),
                        **run_kwargs,
                    ):
                        factor, config, path = meta[name]
                        opts = resolve_materialize_kwargs_for_pipeline(config, pipeline_overrides)
                        output = {
                            "factor": factor,
                            "analysis": analyses[name],
                            "result": result,
                        }
                        out = execute_materialize_from_resolved(engine, factor, output, opts)
                        out["config"] = config
                        out["config_path"] = str(path)
                        out["batched_run"] = True
                        outputs[name] = out
                    continue

                # 并行路径：整批计算后逐因子落盘（保留并发，结果驻留在 batch 期间）
                if parallel:
                    run_out = engine.run_many_parallel(
                        factors,
                        n_jobs=n_jobs,
                        **run_kwargs,
                    )
                else:
                    run_out = engine.run_many(factors, **run_kwargs)
                for _, factor, config, path in batch:
                    opts = resolve_materialize_kwargs_for_pipeline(config, pipeline_overrides)
                    output = {
                        "factor": factor,
                        "analysis": run_out["analyses"][factor.name],
                        "result": run_out["results"][factor.name],
                    }
                    out = execute_materialize_from_resolved(engine, factor, output, opts)
                    out["config"] = config
                    out["config_path"] = str(path)
                    out["batched_run"] = True
                    outputs[factor.name] = out
                continue

            for engine, factor, config, path in batch:
                name, out = cls._materialize_one_from_config_item(
                    engine,
                    factor,
                    config,
                    path,
                    pipeline_overrides=pipeline_overrides,
                )
                outputs[name] = out

    @classmethod
    def materialize_many_from_config(
        cls,
        config_paths: Sequence[str | Path],
        *,
        batch_run: bool = True,
        enable_cse: bool | None = None,
        profile: str | None = None,
        pipeline_overrides: Any | None = None,
        parallel: bool = False,
        n_jobs: int | None = None,
    ) -> dict[str, Any]:
        """从多个 YAML 配置文件批量物化因子。

        同数据源作用域且物化参数一致的因子可共享一次 ``run_many`` 计算，
        再逐因子落盘（见 ``can_batch_materialize_compute``）。

        Args:
            config_paths: 配置文件路径序列。
            batch_run: 是否尝试批跑共享计算。
            enable_cse: 是否启用 CSE。
            profile: 可选 profile 名称。
            pipeline_overrides: pipeline 层覆盖参数。
            parallel: 批跑时是否并行因子根。
            n_jobs: 并行 worker 数。

        Returns:
            含 ``materializations`` 字典的结果。
        """
        from runtime.config_runtime import config_data_scope_key

        loaded: list[tuple[FactorEngine, Factor, FactorEngineConfig, str | Path]] = []
        for path in config_paths:
            engine, factor, config = cls.from_config(path, profile=profile)
            loaded.append((engine, factor, config, path))
        if not loaded:
            return {"materializations": {}}

        groups: dict[str, list[tuple[FactorEngine, Factor, FactorEngineConfig, str | Path]]] = {}
        for item in loaded:
            key = config_data_scope_key(item[2])
            groups.setdefault(key, []).append(item)

        outputs: dict[str, Any] = {}
        for group in groups.values():
            cls._materialize_batch_from_config_group(
                group,
                outputs,
                batch_run=batch_run,
                enable_cse=enable_cse,
                pipeline_overrides=pipeline_overrides,
                parallel=parallel,
                n_jobs=n_jobs,
            )
        return {"materializations": outputs}

    @classmethod
    def materialize_many_from_config_parallel(
        cls,
        config_paths: Sequence[str | Path],
        *,
        batch_run: bool = True,
        enable_cse: bool | None = None,
        n_jobs: int | None = None,
        profile: str | None = None,
        pipeline_overrides: Any | None = None,
    ) -> dict[str, Any]:
        """从多个 YAML 配置文件批量物化（根节点并行）。

        等价于 ``materialize_many_from_config(..., parallel=True)``。
        """
        return cls.materialize_many_from_config(
            config_paths,
            batch_run=batch_run,
            enable_cse=enable_cse,
            profile=profile,
            pipeline_overrides=pipeline_overrides,
            parallel=True,
            n_jobs=n_jobs,
        )

    def materialize_sharded(
        self,
        factors: Sequence[Factor],
        *,
        factor_ids: Sequence[str] | None = None,
        shard_by: str = "factor_id",
        shard_index: int = 0,
        shard_count: int = 1,
        run_many_batch: bool = True,
        **materialize_kwargs: Any,
    ) -> dict[str, Any]:
        """分片物化：按因子 ID、资产 bucket 或时间月份切分执行范围。

        用于大规模因子生产任务的分布式/分片调度。

        Args:
            factors: 待物化因子序列。
            factor_ids: 对应因子 ID；缺省用 ``factor.name``。
            shard_by: 分片策略，可选 ``factor_id`` / ``asset_bucket`` / ``time_month``。
            shard_index: 当前分片索引（0-based）。
            shard_count: 分片总数。
            run_many_batch: 多分片因子是否共享 ``run_many``。
            **materialize_kwargs: 传递给 ``materialize`` 的参数。

        Returns:
            含 ``materializations``、``shard_index``、``shard_count`` 等的字典。

        Raises:
            ValueError: ``shard_by`` 不支持或 ``factor_ids`` 长度不匹配。
        """
        allowed = {"factor_id", "asset_bucket", "time_month"}
        if shard_by not in allowed:
            raise ValueError(f"不支持的 shard_by={shard_by!r}，可选 {sorted(allowed)}")

        ids = list(factor_ids) if factor_ids is not None else [f.name for f in factors]
        if len(ids) != len(factors):
            raise ValueError("factor_ids 长度必须与 factors 一致")

        mat = dict(materialize_kwargs)
        shard_meta: dict[str, Any] = {"shard_by": shard_by}
        engine = self

        if shard_by == "factor_id":
            from runtime.shard_materialize import shard_factor_ids

            selected_ids = set(
                shard_factor_ids(
                    ids,
                    shard_index=shard_index,
                    shard_count=shard_count,
                )
            )
            selected_pairs = [
                (f, fid)
                for f, fid in zip(factors, ids)
                if fid in selected_ids
            ]
            if not selected_pairs:
                return {
                    "shard_index": shard_index,
                    "shard_count": shard_count,
                    "shard_by": shard_by,
                    "materializations": {},
                    "factor_ids": [],
                }
            sel_factors = [p[0] for p in selected_pairs]
            sel_ids = [p[1] for p in selected_pairs]
        else:
            sel_factors = list(factors)
            sel_ids = list(ids)
            ds = self.data_source
            if shard_by == "asset_bucket":
                from runtime.shard_materialize import shard_bucket_values

                bucket_count = int(mat.pop("bucket_count", 64))
                buckets = shard_bucket_values(
                    bucket_count,
                    shard_index=shard_index,
                    shard_count=shard_count,
                )
                shard_meta["bucket_values"] = buckets
                if buckets and ds is not None and hasattr(ds, "params"):
                    import copy

                    scoped = copy.copy(ds)
                    scoped.params = {**getattr(ds, "params", {}), "bucket_values": buckets}
                    scoped._column_cache = {}
                    scoped._panel_cache = {}
                    engine = self.with_data_source(scoped, fresh_cache=True)
            elif shard_by == "time_month":
                from runtime.shard_materialize import (
                    month_date_bounds,
                    month_keys_between,
                    shard_time_months,
                )

                months = mat.get("time_months")
                if months is None:
                    months = month_keys_between(
                        mat.get("since"),
                        mat.get("end_date"),
                    )
                selected_months = shard_time_months(
                    months,
                    shard_index=shard_index,
                    shard_count=shard_count,
                )
                shard_meta["time_months"] = selected_months
                if not selected_months:
                    return {
                        "shard_index": shard_index,
                        "shard_count": shard_count,
                        "shard_by": shard_by,
                        "materializations": {},
                        "factor_ids": [],
                        **shard_meta,
                    }
                start, end = month_date_bounds(selected_months)
                if ds is not None:
                    import copy

                    scoped = copy.copy(ds)
                    scoped.start_date = start
                    scoped.end_date = end
                    if hasattr(scoped, "_column_cache"):
                        scoped._column_cache = {}
                    if hasattr(scoped, "_panel_cache"):
                        scoped._panel_cache = {}
                    engine = self.with_data_source(scoped, fresh_cache=True)

        outputs: dict[str, Any] = {}
        run_kw = {
            "input_dq_check": mat.pop("input_dq_check", False),
            "input_dq_strict": mat.pop("input_dq_strict", True),
            "input_dq_thresholds": mat.pop("input_dq_thresholds", None),
            "auto_warmup": mat.pop("auto_warmup", False),
            "trim_warmup": mat.pop("trim_warmup", True),
            "market": mat.pop("market", None),
            "pit_enforce": mat.pop("pit_enforce", False),
            "pit_forbid_forward_fill": mat.pop("pit_forbid_forward_fill", False),
        }
        if run_many_batch and len(sel_factors) > 1:
            run_out = engine.run_many(sel_factors, **run_kw)
            from runtime.materialize_service import execute_materialize

            for factor, fid in zip(sel_factors, sel_ids):
                output = {
                    "factor": factor,
                    "analysis": run_out["analyses"][factor.name],
                    "result": run_out["results"][factor.name],
                }
                out = execute_materialize(
                    engine,
                    factor,
                    output,
                    target=str(mat.get("write_target", "local")),
                    lake_root=mat.get("lake_root"),
                    staging_dataset=mat.get("staging_dataset", "factor_lake_staging"),
                    factor_id=fid,
                    author=mat.get("author"),
                    frequency=mat.get("frequency"),
                    description=mat.get("description"),
                    expression=mat.get("expression"),
                    dq_check=mat.get("dq_check", False),
                    dq_strict=mat.get("dq_strict", True),
                    dq_thresholds=mat.get("dq_thresholds"),
                    write_metadata=mat.get("write_metadata", True),
                    data_source_config=mat.get("data_source_config"),
                    resume_materialize=mat.get("resume_materialize", False),
                    isolate_partition_failures=mat.get(
                        "isolate_partition_failures", True
                    ),
                    preserve_invalid_rows=mat.get("preserve_invalid_rows", False),
                    value_dtype=str(mat.get("value_dtype", "float32")),
                    clickhouse_table=mat.get("clickhouse_table"),
                    ch_ensure_table=mat.get("ch_ensure_table", True),
                    ch_host=mat.get("ch_host"),
                    ch_port=mat.get("ch_port"),
                    ch_database=mat.get("ch_database"),
                    ch_username=mat.get("ch_username"),
                    ch_password=mat.get("ch_password"),
                    ch_secure=mat.get("ch_secure"),
                    storage_format=str(mat.get("storage_format", "long")),
                    partition_columns=mat.get("partition_columns"),
                )
                out["batched_run"] = True
                out["shard_index"] = shard_index
                out["shard_count"] = shard_count
                out["shard_by"] = shard_by
                outputs[factor.name] = out.get("materialization", out)
        else:
            for factor, fid in zip(sel_factors, sel_ids):
                mk = {**run_kw, **mat, "factor_id": fid}
                for key in ("time_months", "bucket_count"):
                    mk.pop(key, None)
                out = engine.materialize(factor, **mk)
                out["shard_index"] = shard_index
                out["shard_count"] = shard_count
                out["shard_by"] = shard_by
                outputs[factor.name] = out

        return {
            "shard_index": shard_index,
            "shard_count": shard_count,
            "shard_by": shard_by,
            "factor_ids": sel_ids,
            "materializations": outputs,
            **shard_meta,
        }

    def materialize_matrix(
        self,
        factors: Sequence[Factor],
        *,
        factor_ids: Sequence[str] | None = None,
        universe: str,
        frequency: str = "1d",
        matrix_root: str | Path | None = None,
        partition_columns: list[str] | None = None,
        value_dtype: str = "float32",
        **run_kwargs: Any,
    ) -> dict[str, Any]:
        """批量计算多因子并写入 factor_matrix 宽表格式。

        宽表格式面向训练/回测场景，减少多因子读取时的 join 开销。

        Args:
            factors: 待计算因子序列。
            factor_ids: 因子 ID 列表。
            universe: 标的 universe 标识。
            frequency: 因子频率。
            matrix_root: 宽表输出根目录。
            partition_columns: 分区列。
            value_dtype: 值 dtype。
            **run_kwargs: 传递给 ``run_many`` 的参数。

        Returns:
            物化摘要字典（由 ``matrix_service`` 返回）。
        """
        from runtime.matrix_service import execute_materialize_matrix

        return execute_materialize_matrix(
            self,
            factors,
            factor_ids=factor_ids,
            universe=universe,
            frequency=frequency,
            matrix_root=matrix_root,
            partition_columns=partition_columns,
            value_dtype=value_dtype,
            **run_kwargs,
        )

    def plan_incremental_from_event(
        self,
        event: Any,
        *,
        lake_root: str | Path | None = None,
        end_date: str | None = None,
        lookback_extra: int = 5,
        market: str | None = None,
    ) -> dict[str, Any]:
        """根据数据列更新事件生成受影响因子的增量重算计划。

        Args:
            event: 数据更新事件对象（含列名、时间范围等）。
            lake_root: 因子湖根目录（查 catalog 依赖）。
            end_date: 增量输出上界。
            lookback_extra: lookback 缓冲 bar 数。
            market: 市场标识。

        Returns:
            增量计划字典（受影响因子列表及窗口）。
        """
        from runtime.incremental_event_service import plan_incremental_from_event

        return plan_incremental_from_event(
            event,
            lake_root=lake_root,
            end_date=end_date,
            lookback_extra=lookback_extra,
            market=market,
        )

    def materialize_incremental_from_event(
        self,
        event: Any,
        *,
        lake_root: str | Path | None = None,
        end_date: str | None = None,
        lookback_extra: int = 5,
        market: str | None = None,
        dry_run: bool = False,
        **materialize_kwargs: Any,
    ) -> dict[str, Any]:
        """根据数据列更新事件自动增量物化受影响因子。

        Args:
            event: 数据更新事件对象。
            lake_root: 因子湖根目录。
            end_date: 增量输出上界。
            lookback_extra: lookback 缓冲 bar 数。
            market: 市场标识。
            dry_run: 为真时仅生成计划不落盘。
            **materialize_kwargs: 传递给 ``materialize_incremental`` 的参数。

        Returns:
            物化结果或 dry_run 计划字典。
        """
        from runtime.incremental_event_service import materialize_incremental_from_event

        return materialize_incremental_from_event(
            self,
            event,
            lake_root=lake_root,
            end_date=end_date,
            lookback_extra=lookback_extra,
            market=market,
            dry_run=dry_run,
            **materialize_kwargs,
        )

    @classmethod
    def materialize_incremental_many_from_config(
        cls,
        config_paths: Sequence[str | Path],
        *,
        profile: str | None = None,
        pipeline_overrides: Any | None = None,
    ) -> dict[str, Any]:
        """从多个 YAML 配置文件批量增量物化。

        逐配置解析 incremental 参数并调用 ``materialize_incremental``。
        """
        from runtime.incremental_event_service import (
            materialize_incremental_many_from_config,
        )

        return materialize_incremental_many_from_config(
            cls,
            config_paths,
            profile=profile,
            pipeline_overrides=pipeline_overrides,
        )

    @classmethod
    def materialize_from_config(
        cls,
        config_path: str | Path,
        *,
        lake_root: str | Path | None = None,
        factor_id: str | None = None,
        author: str | None = None,
        frequency: str | None = None,
        description: str | None = None,
        expression: str | None = None,
    ):
        """从 YAML 配置文件执行单因子并物化到因子湖（Parquet）。

        Args:
            config_path: YAML 配置文件路径。
            lake_root: 覆盖配置中的因子湖根目录。
            factor_id: 覆盖落盘因子 ID。
            author/frequency/description/expression: 覆盖元数据字段。

        Returns:
            含 ``materialization`` 及 ``config`` 的执行结果字典。
        """
        logger.info("开始从配置物化因子: %s", config_path)
        engine, factor, config = cls.from_config(config_path)
        from runtime.config_runtime import resolve_materialize_kwargs

        opts = resolve_materialize_kwargs(
            config,
            lake_root_override=str(lake_root) if lake_root is not None else None,
            factor_id_override=factor_id,
            author_override=author,
            frequency_override=frequency,
            description_override=description,
            expression_override=expression,
        )
        result = engine.materialize(factor, **opts.to_engine_materialize_kwargs())
        result["config"] = config
        logger.info("完成从配置物化因子: %s", factor.name)
        return result

    @classmethod
    def materialize_incremental_from_config(
        cls,
        config_path: str | Path,
        *,
        lake_root: str | Path | None = None,
        factor_id: str | None = None,
        since: str | None = None,
        end_date: str | None = None,
        lookback_extra: int | None = None,
        recompute_tail_bars: int | None = None,
    ):
        """从 YAML 配置文件执行增量物化（watermark + lookback + upsert）。

        Args:
            config_path: YAML 配置文件路径。
            lake_root: 覆盖因子湖根目录。
            factor_id: 覆盖落盘因子 ID。
            since: 显式增量起点（覆盖 watermark）。
            end_date: 输出区间上界。
            lookback_extra: lookback 缓冲 bar 数。
            recompute_tail_bars: 输出 tail 重算 bar 数。

        Returns:
            含 ``materialization``、``incremental`` 及 ``config`` 的结果字典。
        """
        logger.info("开始从配置增量物化因子: %s", config_path)
        engine, factor, config = cls.from_config(config_path)
        from runtime.config_runtime import resolve_materialize_kwargs

        opts = resolve_materialize_kwargs(
            config,
            lake_root_override=str(lake_root) if lake_root is not None else None,
            factor_id_override=factor_id,
            since_override=since,
            end_date_override=end_date,
            lookback_extra_override=lookback_extra,
            recompute_tail_bars_override=recompute_tail_bars,
        )
        result = engine.materialize_incremental(
            factor, **opts.to_incremental_materialize_kwargs()
        )
        result["config"] = config
        logger.info("完成从配置增量物化因子: %s", factor.name)
        return result

    def _make_context(
        self,
        *,
        shared_result_cache: dict[str, Any] | None = None,
        perf: PerfConfig | None = None,
    ) -> ExecutionContext:
        """构造单次执行上下文，注入缓存会话与 query budget。"""
        from cache.session import ExecutionCacheSession
        from storage.long_table_source import LongTableDataSource

        prefer_long = isinstance(self.data_source, LongTableDataSource)
        effective_perf = perf or PerfConfig.from_env()
        from cleaned_operators.registry import OperatorRegistry
        from backend.evidence_provenance import compute_payload_hash, load_verified_artifact
        try:
            evidence = load_verified_artifact()
            evidence_version = compute_payload_hash((evidence or {}).get("provenance") or {})
        except (FileNotFoundError, ValueError, TypeError, OSError):
            evidence_version = ""
        base = ExecutionContext(
            data_source=self.data_source,
            run_mode=self.run_mode,
            registry_version=OperatorRegistry.version(),
            evidence_version=evidence_version,
            cache=self.cache,
            shared_result_cache=shared_result_cache,
            shared_long_lazy_cache={},
            panel_cache={},
            materialized_series={},
            materialized_long_lazy={},
            prefer_long_table=prefer_long,
            perf=effective_perf,
            query_budget=effective_perf.build_query_budget(),
            runtime_stats={},
        )
        if self.cache is None and shared_result_cache is None:
            return base
        # #335：session 的 CSE / panel 字节预算从 perf/resource plan 传入；
        # 无 resource plan（构造失败/未配置）时预算为 None（session 不强制限制）。
        cse_budget_bytes: int | None = None
        panel_budget_bytes: int | None = None
        try:
            resource_plan = effective_perf.build_resource_plan()
            if resource_plan is not None:
                cse_budget_bytes = getattr(resource_plan, "cse_budget_bytes", None)
                panel_budget_bytes = getattr(resource_plan, "panel_budget_bytes", None)
        except Exception:  # pragma: no cover - resource probe 失败不阻塞执行
            logger.debug("ExecutionCacheSession 资源预算构造失败，使用 None 预算", exc_info=True)
            cse_budget_bytes = None
            panel_budget_bytes = None
        session = ExecutionCacheSession(
            plan_cache=self.cache,
            shared_result_cache=shared_result_cache,
            panel_cache={},
            cse_budget_bytes=cse_budget_bytes,
            panel_budget_bytes=panel_budget_bytes,
        )
        return session.wrap_context(base)

    @staticmethod
    def _resolve_lineage_expression(factor: Factor, expression: str | None) -> str | None:
        from runtime.lineage_service import resolve_lineage_expression

        return resolve_lineage_expression(factor, expression)

    @staticmethod
    def _composite_lineage_from_source(data_source: Any) -> dict[str, Any]:
        from runtime.lineage_service import composite_lineage_from_source

        return composite_lineage_from_source(data_source)

    @staticmethod
    def _lineage_extra(
        *,
        data_source_config: dict | None,
        snapshot_id: str | None,
        input_dq: dict | None = None,
        data_source: Any = None,
        **more: Any,
    ) -> dict[str, Any]:
        from runtime.lineage_service import build_lineage_extra

        return build_lineage_extra(
            data_source_config=data_source_config,
            snapshot_id=snapshot_id,
            input_dq=input_dq,
            data_source=data_source,
            **more,
        )

    @staticmethod
    def _data_snapshot_id_from_source(
        data_source: Any | None,
        data_source_config: dict | None,
    ) -> str | None:
        from runtime.lineage_service import resolve_data_snapshot_id

        return resolve_data_snapshot_id(data_source, data_source_config)

    @staticmethod
    def _resolve_parquet_write_target(write_target: str) -> str:
        from runtime.materialize_service import resolve_parquet_write_target

        return resolve_parquet_write_target(write_target)

    @staticmethod
    def _needs_clickhouse_write(write_target: str) -> bool:
        from runtime.materialize_service import needs_clickhouse_write

        return needs_clickhouse_write(write_target)

    def _append_clickhouse_to_summary(self, summary: dict[str, Any], **kwargs) -> dict[str, Any]:
        from runtime import dual_write_service

        return dual_write_service.append_clickhouse_to_summary(summary, **kwargs)

    def _dual_write_clickhouse(self, materializer, summary: dict[str, Any], **kwargs) -> dict[str, Any]:
        from runtime import dual_write_service

        return dual_write_service.dual_write_clickhouse(materializer, summary, **kwargs)

    @staticmethod
    def _prefetch_columns(data_source: Any, columns: set[str]) -> None:
        if not columns:
            return
        names = sorted(columns)
        prefetch_panels = getattr(data_source, "prefetch_panels", None)
        if callable(prefetch_panels):
            prefetch_panels(names)
            logger.info("预加载宽表 panel（prefetch_panels）: %s", names)
            return
        prefetch = getattr(data_source, "prefetch_columns", None)
        if callable(prefetch):
            prefetch(names)
            logger.info("预加载数据列（prefetch）: %s", names)
            return
        load_columns = getattr(data_source, "load_columns", None)
        if callable(load_columns):
            load_columns(names)
            logger.info("预加载数据列: %s", names)

    @staticmethod
    def _prepare_batch_data(
        data_source: Any,
        columns: set[str],
        *,
        input_dq_check: bool = False,
        input_dq_strict: bool = True,
        input_dq_thresholds=None,
        data_snapshot_id: str | None = None,
    ) -> Any | None:
        """批量 input_dq + 单次 prefetch（run_many 快路径）。"""
        if not columns:
            return None
        from storage.read_session import DataSourceReadSession

        session = DataSourceReadSession(data_source)
        input_report = session.prepare_batch(
            columns,
            input_dq_check=input_dq_check,
            input_dq_strict=input_dq_strict,
            input_dq_thresholds=input_dq_thresholds,
            data_snapshot_id=data_snapshot_id,
        )
        if input_dq_check:
            logger.info("批量输入 DQ 通过，列数=%d", len(columns))
        return input_report

    def run(
        self,
        factor: Factor,
        *,
        plan=None,
        analysis=None,
        input_dq_check: bool = False,
        input_dq_strict: bool = True,
        input_dq_thresholds=None,
        auto_warmup: bool = False,
        trim_warmup: bool = True,
        market: str | None = None,
        pit_enforce: bool = False,
        pit_forbid_forward_fill: bool = False,
    ):
        """编译并执行单因子，返回完整运行上下文。

        可选跳过重复编译（传入已有 ``plan`` / ``analysis``）。支持 input_dq、
        auto_warmup（按 lookback 扩展加载窗口后裁剪）、PIT 审计及 production
        fast path 运行时校验。

        Args:
            factor: 待执行因子。
            plan: 可选预编译逻辑计划。
            analysis: 可选预编译分析结果。
            input_dq_check: 是否校验输入列质量。
            input_dq_strict: 输入 DQ 失败是否中断。
            input_dq_thresholds: 输入 DQ 阈值。
            auto_warmup: 是否自动扩展 warmup 加载窗口。
            trim_warmup: warmup 后是否裁剪回用户请求的 ``start_date``。
            market: 市场标识（warmup 日历）。
            pit_enforce: 是否强制 PIT 安全审计（编译阶段）。
            pit_forbid_forward_fill: PIT 是否禁止前向填充。

        Returns:
            含 ``factor``、``analysis``、``plan``、``result`` 的字典；可选键包括
            ``input_dq``、``run_window``、``backend_path``、``production_pandas_fallbacks``
            及 SQL/Polars 路径诊断字段。
        """
        from runtime.production_policy import is_production_mode

        pit_enforce = bool(pit_enforce or is_production_mode(self.run_mode))
        started_at = time.perf_counter()
        logger.info("开始执行因子 '%s'", factor.name)
        from runtime.production_policy import (
            assert_no_stub_operators,
            assert_production_factors,
            assert_production_run_flags,
        )

        assert_production_factors([factor], mode=self.run_mode, context="run")
        if plan is None or analysis is None:
            plan, analysis = self.compile(
                factor,
                pit_enforce=pit_enforce,
                pit_forbid_forward_fill=pit_forbid_forward_fill,
            )
        # R9-P0-011: execution-contract gate —— 单因子执行路径与批跑同门，
        # scoped-universe 因子不得在全源上算截面。
        assert_execution_scope_contract(
            _scope_from_factor(factor),
            plan,
            factor_name=factor.name,
            data_source=self.data_source,
        )
        assert_production_run_flags(
            mode=self.run_mode,
            input_dq_check=input_dq_check,
            auto_warmup=auto_warmup,
            pit_enforce=pit_enforce,
            context="run",
        )
        assert_no_stub_operators(plan, mode=self.run_mode)
        from runtime.warmup_service import prepare_run_warmup

        warmup = prepare_run_warmup(
            self,
            factor,
            analysis,
            auto_warmup=auto_warmup,
            trim_warmup=trim_warmup,
            market=market,
        )
        run_window = warmup.run_window
        engine_to_use = warmup.engine
        s_bpd = warmup.bars_per_day

        input_report = None
        from planner.sql_io import should_skip_column_prefetch

        # DebugBackend renders plans without reading data; skip all source I/O.
        if engine_to_use.backend.__class__.__name__ == "DebugBackend":
            skip_prefetch = True
        else:
            skip_prefetch = should_skip_column_prefetch(
                [plan],
                input_dq_check=input_dq_check,
                backend=engine_to_use.backend,
            )
        if analysis.referenced_columns and not skip_prefetch:
            from storage.read_session import DataSourceReadSession

            session = DataSourceReadSession(engine_to_use.data_source)
            if input_dq_check:
                input_report = session.prepare_batch(
                    analysis.referenced_columns,
                    input_dq_check=True,
                    input_dq_strict=input_dq_strict,
                    input_dq_thresholds=input_dq_thresholds,
                )
                logger.info("因子 '%s' 输入 DQ 通过", factor.name)
            else:
                session.prefetch(analysis.referenced_columns)
        elif skip_prefetch:
            logger.debug(
                "因子 '%s' 跳过列 prefetch（fully_sql 或 prefers_native_scan）",
                factor.name,
            )
        ctx = engine_to_use._make_context()
        from runtime.production_policy import record_production_fastpath_check
        from runtime.resource_telemetry import record_resource_telemetry

        ctx.runtime_stats = record_resource_telemetry(ctx.runtime_stats)
        record_production_fastpath_check(ctx, plan, mode=engine_to_use.run_mode)
        result = engine_to_use.backend.execute(plan, ctx)
        from runtime.production_policy import assert_production_fastpath_runtime

        assert_production_fastpath_runtime(ctx, mode=engine_to_use.run_mode, context=f"run:{factor.name}")

        if run_window is not None and run_window.trim_output and run_window.requested_start:
            from storage.time_window import slice_series_time_window

            trim_start = pd.Timestamp(run_window.requested_start)
            trim_end = (
                pd.Timestamp(run_window.requested_end) if run_window.requested_end else None
            )
            # 日内源保留 timestamp 精度（不按 normalize 截断）
            if s_bpd > 1:
                trim_start = pd.Timestamp(run_window.requested_start)
            result = slice_series_time_window(
                result,
                start=trim_start,
                end=trim_end,
            )

        from runtime.perf_config import PerfConfig
        from runtime.result_budget import enforce_result_budget

        enforce_result_budget(
            result,
            PerfConfig.from_env(),
            factor_name=factor.name,
            run_mode=self.run_mode,
        )
        non_null_count = int(result.notna().sum()) if hasattr(result, "notna") else None
        logger.info(
            "完成执行因子 '%s'，结果行数=%s，非空=%s，耗时 %.2fs",
            factor.name,
            len(result),
            non_null_count,
            time.perf_counter() - started_at,
        )
        out = {
            "factor": factor,
            "analysis": analysis,
            "plan": plan,
            "result": result,
        }
        if input_report is not None:
            out["input_dq"] = input_report.to_dict()
        if run_window is not None:
            out["run_window"] = run_window.to_dict()
        from runtime.production_policy import (
            format_pandas_fallback_report,
            is_production_mode,
            summarize_pandas_fallbacks,
        )

        fallbacks = summarize_pandas_fallbacks(ctx)
        if fallbacks:
            out["production_pandas_fallbacks"] = fallbacks
            if is_production_mode(self.run_mode):
                report = format_pandas_fallback_report(fallbacks)
                if report:
                    logger.info("因子 '%s' %s", factor.name, report.replace("\n", " | "))
        runtime = dict(getattr(ctx, "runtime_stats", None) or {})
        if runtime.get("polars_expr"):
            out["polars_expr"] = True
        if runtime.get("polars_expr_fallback"):
            out["polars_expr_fallback"] = True
        if runtime.get("used_polars_long_path"):
            out["used_polars_long_path"] = True
        if runtime.get("used_polars_long_native"):
            out["used_polars_long_native"] = True
        if runtime.get("used_polars_long_python_rolling"):
            out["used_polars_long_python_rolling"] = True
        if runtime.get("used_polars_long_map_groups"):
            out["used_polars_long_map_groups"] = True
        if runtime.get("used_polars_long_registry"):
            out["used_polars_long_registry"] = True
        if runtime.get("used_polars_long_passthrough"):
            out["used_polars_long_passthrough"] = True
        if runtime.get("polars_long_fallback_reason"):
            out["polars_long_fallback_reason"] = runtime["polars_long_fallback_reason"]
        for key in (
            "used_sql_pushdown",
            "fully_sql",
            "sql_subtree_count",
            "sql_subtrees",
            "sql_fully_pushed",
            "sql_partial_pushed",
            "sql_long_lazy_subtrees",
            "sql_series_subtrees",
            "sql_fallback_subtree_count",
            "sql_dialect",
            "sql_query_count",
            "production_fastpath_ok",
            "production_fastpath_violations",
            "polars_long_native_ops",
            "polars_long_python_rolling_ops",
            "polars_long_map_group_ops",
            "polars_long_registry_ops",
            "polars_long_passthrough_ops",
            "polars_long_blocked_causal_ops",
            "polars_long_other_ops",
            "polars_long_columns",
        ):
            if key in runtime:
                out[key] = runtime[key]
        from backend.path_summary import build_backend_path_summary, snapshot_backend_path
        from backend.runtime_labels import resolve_runtime_backend_label
        from runtime.resource_telemetry import record_resource_telemetry

        ctx.runtime_stats = record_resource_telemetry(ctx.runtime_stats, finalize=True)
        runtime = dict(getattr(ctx, "runtime_stats", None) or {})
        from backend.runtime_labels import resolve_runtime_backend_label
        from backend.path_summary import build_backend_path_summary, snapshot_backend_path

        runtime.setdefault("backend", resolve_runtime_backend_label(getattr(engine_to_use, "backend", None)))
        if fallbacks and "production_pandas_fallbacks" not in runtime:
            runtime["production_pandas_fallbacks"] = fallbacks
        out["backend_path"] = snapshot_backend_path(runtime)
        out["backend_path_summary"] = out["backend_path"].get("backend_path_summary") or build_backend_path_summary(
            runtime
        )
        if runtime.get("events"):
            out["runtime_events"] = list(runtime["events"])
        return out

    def materialize(
        self,
        factor: Factor,
        *,
        lake_root: str | Path | None = None,
        factor_id: str | None = None,
        author: str | None = None,
        frequency: str | None = None,
        description: str | None = None,
        expression: str | None = None,
        dq_check: bool = False,
        dq_strict: bool = True,
        dq_thresholds=None,
        input_dq_check: bool = False,
        input_dq_strict: bool = True,
        input_dq_thresholds=None,
        data_source_config: dict | None = None,
        write_metadata: bool = True,
        resume_materialize: bool = False,
        isolate_partition_failures: bool = True,
        auto_warmup: bool = False,
        trim_warmup: bool = True,
        market: str | None = None,
        preserve_invalid_rows: bool = False,
        value_dtype: str = "float32",
        write_target: str = "local",
        pit_enforce: bool = False,
        pit_forbid_forward_fill: bool = False,
        ch_host: str | None = None,
        ch_port: int | None = None,
        ch_database: str | None = None,
        ch_username: str | None = None,
        ch_password: str | None = None,
        ch_secure: bool | None = None,
        clickhouse_table: str | None = None,
        ch_ensure_table: bool = True,
        staging_dataset: str = "factor_lake_staging",
        storage_format: str = "long",
        partition_columns: list[str] | None = None,
    ):
        """执行单因子并将结果物化到因子湖（及可选 ClickHouse）。

        内部先 ``run`` 再调用 ``materialize_service.execute_materialize``。
        ``write_target`` 支持 ``local``、``staging``、``clickhouse``、
        ``staging_clickhouse`` 等组合目标。

        Args:
            factor: 待物化因子。
            lake_root: 因子湖根目录。
            factor_id: 落盘因子 ID。
            author/frequency/description/expression: 元数据。
            dq_check/dq_strict/dq_thresholds: 产出 DQ 门禁。
            input_dq_check/input_dq_strict/input_dq_thresholds: 输入 DQ。
            data_source_config: 数据源配置快照（lineage）。
            write_metadata: 是否写入 run 元数据。
            resume_materialize: 断点续写分区。
            isolate_partition_failures: 单分区失败隔离。
            auto_warmup/trim_warmup/market: 传递给 ``run`` 的 warmup 参数。
            preserve_invalid_rows: 保留 inf 为 ``is_valid=0``。
            value_dtype: 落盘值 dtype。
            write_target: 写入目标。
            pit_enforce/pit_forbid_forward_fill: PIT 参数。
            ch_* / clickhouse_*: ClickHouse 连接参数。
            staging_dataset: staging 数据集名。
            storage_format: Parquet 格式（``long`` 等）。
            partition_columns: 自定义分区列。

        Returns:
            ``run`` 输出字典，附加 ``materialization`` 落盘摘要。
        """
        target = str(write_target or "local").lower()
        logger.info("开始落盘因子 '%s'，write_target=%s", factor.name, target)
        output = self.run(
            factor,
            input_dq_check=input_dq_check,
            input_dq_strict=input_dq_strict,
            input_dq_thresholds=input_dq_thresholds,
            auto_warmup=auto_warmup,
            trim_warmup=trim_warmup,
            market=market,
            pit_enforce=pit_enforce,
            pit_forbid_forward_fill=pit_forbid_forward_fill,
        )
        from runtime.materialize_service import execute_materialize

        return execute_materialize(
            self,
            factor,
            output,
            target=target,
            lake_root=lake_root,
            staging_dataset=staging_dataset,
            factor_id=factor_id,
            author=author,
            frequency=frequency,
            description=description,
            expression=expression,
            dq_check=dq_check,
            dq_strict=dq_strict,
            dq_thresholds=dq_thresholds,
            write_metadata=write_metadata,
            data_source_config=data_source_config,
            resume_materialize=resume_materialize,
            isolate_partition_failures=isolate_partition_failures,
            preserve_invalid_rows=preserve_invalid_rows,
            value_dtype=value_dtype,
            clickhouse_table=clickhouse_table,
            ch_ensure_table=ch_ensure_table,
            ch_host=ch_host,
            ch_port=ch_port,
            ch_database=ch_database,
            ch_username=ch_username,
            ch_password=ch_password,
            ch_secure=ch_secure,
            lineage_mode="full",
            storage_format=storage_format,
            partition_columns=partition_columns,
        )

    def materialize_clickhouse(
        self,
        factor: Factor,
        *,
        factor_id: str | None = None,
        table: str | None = None,
        factor_version: str = "",
        author: str | None = None,
        frequency: str | None = None,
        description: str | None = None,
        expression: str | None = None,
        data_source_config: dict | None = None,
        input_dq_check: bool = False,
        input_dq_strict: bool = True,
        input_dq_thresholds=None,
        auto_warmup: bool = False,
        trim_warmup: bool = True,
        market: str | None = None,
        dq_check: bool = False,
        dq_strict: bool = True,
        dq_thresholds=None,
        preserve_invalid_rows: bool = False,
        value_dtype: str = "float32",
        pit_enforce: bool = False,
        pit_forbid_forward_fill: bool = False,
        ch_host: str | None = None,
        ch_port: int | None = None,
        ch_database: str | None = None,
        ch_username: str | None = None,
        ch_password: str | None = None,
        ch_secure: bool | None = None,
        ensure_table: bool = True,
        lake_root: str | Path | None = None,
        staging_dataset: str = "factor_lake_staging",
        timestamp_column: str | None = None,
        instrument_column: str | None = None,
    ):
        """执行因子并写入 ClickHouse（catalog + watermark + CH）。

        等价于 ``materialize(..., write_target=\"clickhouse\")``，返回结构
        额外含 ``clickhouse_materialization`` 摘要。

        Args:
            factor: 待物化因子。
            factor_id: 落盘因子 ID。
            table: ClickHouse 表名。
            factor_version: 因子版本号。
            author/frequency/description/expression: 元数据。
            data_source_config: 数据源配置快照。
            input_dq_* / dq_*: 输入与产出 DQ 参数。
            auto_warmup/trim_warmup/market: warmup 参数。
            preserve_invalid_rows/value_dtype: 落盘格式参数。
            pit_enforce/pit_forbid_forward_fill: PIT 参数。
            ch_*: ClickHouse 连接参数。
            ensure_table: 是否自动建表。
            lake_root/staging_dataset: Parquet 侧参数（双写场景）。
            timestamp_column/instrument_column: 已废弃，请配置 ClickHouseMaterializer。

        Returns:
            含 ``materialization`` 与 ``clickhouse_materialization`` 的结果字典。
        """
        if timestamp_column is not None or instrument_column is not None:
            logger.warning(
                "materialize_clickhouse 的 timestamp_column/instrument_column 已废弃；"
                "列映射请配置 ClickHouseMaterializer（默认 trade_date/instrument）。"
            )
        output = self.materialize(
            factor,
            factor_id=factor_id,
            lake_root=lake_root,
            author=author,
            frequency=frequency,
            description=description,
            expression=expression,
            data_source_config=data_source_config,
            input_dq_check=input_dq_check,
            input_dq_strict=input_dq_strict,
            input_dq_thresholds=input_dq_thresholds,
            auto_warmup=auto_warmup,
            trim_warmup=trim_warmup,
            market=market,
            dq_check=dq_check,
            dq_strict=dq_strict,
            dq_thresholds=dq_thresholds,
            preserve_invalid_rows=preserve_invalid_rows,
            value_dtype=value_dtype,
            pit_enforce=pit_enforce,
            pit_forbid_forward_fill=pit_forbid_forward_fill,
            write_target="clickhouse",
            clickhouse_table=table,
            ch_host=ch_host,
            ch_port=ch_port,
            ch_database=ch_database,
            ch_username=ch_username,
            ch_password=ch_password,
            ch_secure=ch_secure,
            ch_ensure_table=ensure_table,
            staging_dataset=staging_dataset,
        )
        mat = output.get("materialization") or {}
        ch = mat.get("clickhouse") or {}
        output["clickhouse_materialization"] = {
            "factor_id": ch.get("factor_id") or mat.get("factor_id"),
            "table": ch.get("table"),
            "rows_written": ch.get("rows_written", mat.get("rows_written", 0)),
            "database": ch.get("database"),
            "write_target": "clickhouse",
            "dq_report": mat.get("clickhouse_dq_report") or mat.get("dq_report"),
            "run_id": (mat.get("run_lineage") or {}).get("run_id"),
        }
        return output

    @classmethod
    def materialize_clickhouse_from_config(cls, config_path: str | Path):
        """从 YAML 配置文件执行因子并写入 ClickHouse。

        Args:
            config_path: YAML 配置文件路径。

        Returns:
            含 ``clickhouse_materialization`` 的结果字典。
        """
        engine, factor, config = cls.from_config(config_path)
        from runtime.config_runtime import resolve_materialize_kwargs

        opts = resolve_materialize_kwargs(config)
        return engine.materialize_clickhouse(
            factor,
            factor_id=opts.factor_id,
            table=opts.clickhouse_table,
            expression=opts.expression,
            data_source_config=opts.data_source_config,
            auto_warmup=opts.auto_warmup,
            trim_warmup=opts.trim_warmup,
            market=opts.market,
            dq_check=opts.dq_check,
            dq_strict=opts.dq_strict,
            dq_thresholds=opts.dq_thresholds,
            input_dq_check=opts.input_dq_check,
            input_dq_strict=opts.input_dq_strict,
            input_dq_thresholds=opts.input_dq_thresholds,
            preserve_invalid_rows=opts.preserve_invalid_rows,
            value_dtype=opts.value_dtype,
            pit_enforce=opts.pit_enforce,
            pit_forbid_forward_fill=opts.pit_forbid_forward_fill,
            ch_host=opts.clickhouse_host,
            ch_port=opts.clickhouse_port,
            ch_database=opts.clickhouse_database,
            ch_username=opts.clickhouse_username,
            ch_password=opts.clickhouse_password,
            ch_secure=opts.clickhouse_secure,
        )

    def run_many(
        self,
        factors: Sequence[Factor],
        *,
        perf: PerfConfig | None = None,
        enable_cse: bool | None = None,
        auto_warmup: bool = False,
        trim_warmup: bool = True,
        market: str | None = None,
        input_dq_check: bool = False,
        input_dq_strict: bool = True,
        input_dq_thresholds=None,
        pit_enforce: bool = False,
        pit_forbid_forward_fill: bool = False,
        result_policy: str = "return",
        sink: Any | None = None,
        warmup_clusters: bool = False,
    ) -> dict[str, Any]:
        """多因子求值：先执行共享子树，再各因子根。

        编译多因子 DAG（可选 CSE）后，先物化 ``DAGPlan.shared_nodes`` 到
        ``shared_result_cache``，再按依赖图顺序执行各因子根。含 ``plan_ref``
        节点的计划必须使用此入口而非多次 ``run``。

        PIT 审计在编译期做（不退化批跑）；``auto_warmup`` 走共享 union 加载窗口，
        各因子独立 trim——不再退化为逐因子 ``run()``。

        Args:
            factors: 待求值因子序列。
            perf: 性能配置。
            enable_cse: 是否启用 CSE。
            auto_warmup/trim_warmup/market: warmup 参数。
            input_dq_check/input_dq_strict/input_dq_thresholds: 输入 DQ。
            pit_enforce/pit_forbid_forward_fill: PIT 参数。
            result_policy: Phase 5 R4 ``return``（全部驻留）| ``sink``（即算即写，
                结果不驻留）| ``yield``/``materialize``（与 ``return`` 累积）。
            sink: ``result_policy="sink"`` 时的回调 ``sink(factor_name, result)``。
            warmup_clusters: Phase 5 P1-4 按 lookback 成本聚类成 waves，各自独立
                union 窗口，避免一个 full-history 因子拖累整批。

        Returns:
            含 ``results``、``dag``、``analyses`` 及可选 ``batch_graph``、
            ``input_dq``、``backend_paths``、``plan_costs`` 等的字典。
        """
        from runtime.batch_service import execute_run_many

        return execute_run_many(
            self,
            factors,
            perf=perf,
            enable_cse=enable_cse,
            auto_warmup=auto_warmup,
            trim_warmup=trim_warmup,
            market=market,
            input_dq_check=input_dq_check,
            input_dq_strict=input_dq_strict,
            input_dq_thresholds=input_dq_thresholds,
            pit_enforce=pit_enforce,
            pit_forbid_forward_fill=pit_forbid_forward_fill,
            result_policy=result_policy,
            sink=sink,
            warmup_clusters=warmup_clusters,
        )

    def run_many_iter(
        self,
        factors: Sequence[Factor],
        *,
        perf: PerfConfig | None = None,
        enable_cse: bool | None = None,
        auto_warmup: bool = False,
        trim_warmup: bool = True,
        market: str | None = None,
        input_dq_check: bool = False,
        input_dq_strict: bool = True,
        input_dq_thresholds=None,
        pit_enforce: bool = False,
        pit_forbid_forward_fill: bool = False,
        precompiled: tuple | None = None,
    ):
        """Phase 5 R4：``run_many`` 的生成器形态——每算完一个因子根立即 ``yield``。

        ``run_many_iter`` 不把全部因子结果驻留在内存，适合「算一个 → DQ →
        筛选/落盘 → 释放 → 下一个」的生产挖因子管线。yield 元素为
        ``(factor_name, result, backend_path_dict)``。

        ``precompiled=(dag, analyses)``：跳过重复编译（materialize_many 流式路径用）。
        """
        from runtime.batch_service import execute_run_many_iter

        yield from execute_run_many_iter(
            self,
            factors,
            perf=perf,
            enable_cse=enable_cse,
            auto_warmup=auto_warmup,
            trim_warmup=trim_warmup,
            market=market,
            input_dq_check=input_dq_check,
            input_dq_strict=input_dq_strict,
            input_dq_thresholds=input_dq_thresholds,
            pit_enforce=pit_enforce,
            pit_forbid_forward_fill=pit_forbid_forward_fill,
            precompiled=precompiled,
        )

    def run_many_parallel(
        self,
        factors: Sequence[Factor],
        *,
        n_jobs: int | None = None,
        perf: PerfConfig | None = None,
        enable_cse: bool | None = None,
        auto_warmup: bool = False,
        trim_warmup: bool = True,
        market: str | None = None,
        input_dq_check: bool = False,
        input_dq_strict: bool = True,
        input_dq_thresholds=None,
        pit_enforce: bool = False,
        pit_forbid_forward_fill: bool = False,
        result_policy: str = "return",
        sink: Any | None = None,
    ) -> dict[str, Any]:
        """多因子求值：共享子树串行、因子根并行。

        在 ``run_many`` 基础上，同一依赖层内的因子根通过 joblib 线程池并行。
        共享子树仍必须先串行物化。资源设置包裹在 ``ExecutionResourceScope`` 中，
        执行结束自动恢复（R15）。

        Args:
            factors: 待求值因子序列。
            n_jobs: 并行 worker 数；缺省取 ``perf.max_workers``。
            result_policy/sink: 同 ``run_many``（R4）。
            其余参数同 ``run_many``。

        Returns:
            与 ``run_many`` 结构相同的批跑结果字典。

        Raises:
            ImportError: 未安装 joblib。
        """
        from runtime.batch_service import execute_run_many_parallel

        return execute_run_many_parallel(
            self,
            factors,
            n_jobs=n_jobs,
            perf=perf,
            enable_cse=enable_cse,
            auto_warmup=auto_warmup,
            trim_warmup=trim_warmup,
            market=market,
            input_dq_check=input_dq_check,
            input_dq_strict=input_dq_strict,
            input_dq_thresholds=input_dq_thresholds,
            pit_enforce=pit_enforce,
            pit_forbid_forward_fill=pit_forbid_forward_fill,
            result_policy=result_policy,
            sink=sink,
        )

    def with_data_source(self, data_source, *, fresh_cache: bool = False) -> FactorEngine:
        """返回共享 backend 的新引擎实例，用于切换或窄化数据源。

        典型场景：增量物化时按时间窗口 narrow 数据源。

        Args:
            data_source: 新的数据源实例。
            fresh_cache: 是否丢弃内存层子计划缓存；磁盘持久化缓存按新
                ``data_scope`` 隔离，仍可命中同窗口历史结果。

        Returns:
            新的 ``FactorEngine`` 实例（共享 ``backend`` 与 ``run_mode``）。
        """
        if self.cache is None:
            return FactorEngine(
                backend=self.backend,
                data_source=data_source,
                cache=None,
                run_mode=self.run_mode,
            )
        scope = compute_data_scope(data_source)
        if isinstance(self.cache, PersistentPlanCache):
            new_cache = self.cache.with_scope(scope, clear_memory=fresh_cache)
            return FactorEngine(
                backend=self.backend,
                data_source=data_source,
                cache=new_cache,
                run_mode=self.run_mode,
            )
        if fresh_cache:
            return FactorEngine(
                backend=self.backend,
                data_source=data_source,
                cache=CacheManager(data_scope=scope),
                run_mode=self.run_mode,
            )
        if getattr(self.cache, "data_scope", None) != scope:
            return FactorEngine(
                backend=self.backend,
                data_source=data_source,
                cache=CacheManager(data_scope=scope),
                run_mode=self.run_mode,
            )
        return FactorEngine(
            backend=self.backend,
            data_source=data_source,
            cache=self.cache,
            run_mode=self.run_mode,
        )

    def run_incremental(
        self,
        factor: Factor,
        *,
        factor_id: str | None = None,
        since: str | None = None,
        end_date: str | None = None,
        lookback_extra: int = 5,
        recompute_tail_bars: int | None = None,
        lake_root: str | Path | None = None,
        input_dq_check: bool = False,
        input_dq_strict: bool = True,
        input_dq_thresholds=None,
        market: str | None = None,
        pit_enforce: bool = False,
        pit_forbid_forward_fill: bool = False,
        auto_warmup: bool = False,
        trim_warmup: bool = True,
    ):
        """增量执行单因子：按 watermark 加载 lookback 窗口，仅输出 tail 区间。

        从 catalog 读取 watermark，构建 ``IncrementalPlan`` 后窄化数据源时间
        窗口，执行 ``run`` 并按计划裁剪结果。

        Args:
            factor: 待执行因子。
            factor_id: catalog 因子 ID；缺省用 ``factor.name``。
            since: 显式增量起点（覆盖 watermark）。
            end_date: 输出区间上界。
            lookback_extra: lookback 缓冲 bar 数。
            recompute_tail_bars: 输出 tail 重算 bar 数。
            lake_root: 因子湖根目录（查 watermark）。
            input_dq_check/input_dq_strict/input_dq_thresholds: 输入 DQ。
            market: 市场标识。
            pit_enforce/pit_forbid_forward_fill: PIT 参数（编译阶段生效）。
            auto_warmup/trim_warmup: warmup 参数（仅全量增量时生效 auto_warmup）。

        Returns:
            含 ``result``（已裁剪）、``incremental`` 计划及 ``analysis``、``plan`` 的字典。
        """
        from cleaned_operators.operator_policy import infer_source_bar_freq
        from runtime.incremental import (
            build_incremental_plan,
            slice_factor_result_for_incremental,
        )
        from storage.materializer import ParquetMaterializer
        from storage.time_window import narrow_data_source_for_window
        from storage.trading_calendar import get_trading_calendar, infer_market

        fid = factor_id or factor.name
        materializer = ParquetMaterializer(lake_root=lake_root)
        watermark = materializer.catalog.get_watermark(fid)

        dataset = getattr(self.data_source, "dataset", None)
        resolved_market = market or infer_market(
            universe=getattr(factor, "universe", None),
            dataset=str(dataset) if dataset else None,
        )
        source_bar_freq = infer_source_bar_freq(
            self.data_source,
            fallback=getattr(factor, "freq", None),
        )
        factor_freq = getattr(factor, "freq", None)

        plan, analysis = self.compile(
            factor,
            pit_enforce=pit_enforce,
            pit_forbid_forward_fill=pit_forbid_forward_fill,
        )
        cal = get_trading_calendar(resolved_market)
        # A segmented-execution stateful root resumes from an exact per-instrument
        # checkpoint, so the recompute-tail overlap is unnecessary (the boundary
        # state is exact, not re-derived).  Zeroing it makes output_start =
        # watermark + 1 and the terminal checkpoint usable by the next run.
        from runtime.stateful_incremental import segmented_incremental_available

        segmented_eligible = (
            getattr(analysis, "ir", None) is not None
            and segmented_incremental_available(ir=analysis.ir)
        )
        effective_tail = 0 if segmented_eligible else recompute_tail_bars
        inc = build_incremental_plan(
            factor_id=fid,
            analysis_lookback=getattr(analysis, "lookback", 0),
            watermark=watermark,
            since=since,
            end_date=end_date,
            lookback_extra=lookback_extra,
            recompute_tail_bars=effective_tail,
            market=resolved_market,
            calendar=cal,
            factor_freq=factor_freq,
            source_bar_freq=source_bar_freq,
        )

        if inc.is_full_run:
            scoped = self
        else:
            narrowed = narrow_data_source_for_window(
                self.data_source,
                start_date=inc.load_start,
                end_date=inc.load_end,
                bar_freq=source_bar_freq,
            )
            # 持久化缓存按 data_scope 隔离；内存缓存需 fresh_cache 避免跨窗口命中
            use_fresh = not isinstance(self.cache, PersistentPlanCache)
            scoped = self.with_data_source(narrowed, fresh_cache=use_fresh)

        logger.info(
            "增量因子 '%s' plan: full=%s load=[%s,%s] output=[%s,%s] lookback=%d freq=%s/%s",
            factor.name,
            inc.is_full_run,
            inc.load_start,
            inc.load_end,
            inc.output_start,
            inc.output_end,
            inc.lookback_bars,
            factor_freq,
            source_bar_freq,
        )

        # Audit §11: a segmented-execution stateful root resumes from a
        # per-instrument checkpoint instead of re-reading the look-back window.
        # A recursive operator cannot use a finite look-back window (its state
        # would be wrong), so when the checkpoint path is unavailable this block
        # forces a full-history replay instead of the narrow-and-replay path.
        from runtime.incremental import FULL_HISTORY_LOOKBACK_SENTINEL
        from runtime.stateful_checkpoint_store import StatefulCheckpointStore
        from runtime.stateful_incremental import (
            try_stateful_segmented_incremental,
        )

        if segmented_eligible and not inc.is_full_run:
            checkpoint_root = (
                Path(str(lake_root)) / "stateful_checkpoints"
                if lake_root is not None
                else None
            )
            store = StatefulCheckpointStore(root=checkpoint_root)
            stateful_output = None
            # (1) pure incremental resume over the output window.
            output_window = narrow_data_source_for_window(
                self.data_source,
                start_date=inc.output_start,
                end_date=inc.output_end,
                bar_freq=source_bar_freq,
            )
            stateful_output = try_stateful_segmented_incremental(
                factor_id=fid,
                ir=analysis.ir,
                source=output_window,
                store=store,
                start=inc.output_start,
                end=inc.output_end,
                bootstrap=False,
            )
            if stateful_output is None:
                # (2) bootstrap the terminal checkpoint over the full causal
                # history up to the output end: a recursive operator's state at
                # the output window must be seeded from the dataset origin, so
                # the first run is a full-history computation.
                bootstrap_window = narrow_data_source_for_window(
                    self.data_source,
                    start_date=None,
                    end_date=inc.output_end,
                    bar_freq=source_bar_freq,
                )
                stateful_output = try_stateful_segmented_incremental(
                    factor_id=fid,
                    ir=analysis.ir,
                    source=bootstrap_window,
                    store=store,
                    start=inc.load_start,
                    end=inc.output_end,
                    bootstrap=True,
                )
            if stateful_output is not None:
                result_series, mode = stateful_output
                result_series = slice_factor_result_for_incremental(result_series, inc)
                output = {
                    "result": result_series,
                    "incremental": {**inc.to_dict(), "market": resolved_market, **mode},
                    "analysis": analysis,
                    "plan": plan,
                }
                logger.info(
                    "增量因子 '%s' 走 stateful segmented 路径 (mode=%s)",
                    factor.name,
                    mode.get("mode"),
                )
                return output
            # No usable checkpoint anywhere: a recursive operator cannot resume
            # from an unseeded finite look-back window, so force full replay.
            inc = build_incremental_plan(
                factor_id=fid,
                analysis_lookback=FULL_HISTORY_LOOKBACK_SENTINEL,
                watermark=None,
                since=since,
                end_date=end_date,
                lookback_extra=lookback_extra,
                recompute_tail_bars=recompute_tail_bars,
                market=resolved_market,
                calendar=cal,
                factor_freq=factor_freq,
                source_bar_freq=source_bar_freq,
            )
            scoped = self
            logger.info(
                "增量因子 '%s' 无可用 checkpoint,回退 full-history replay",
                factor.name,
            )

        output = scoped.run(
            factor,
            plan=plan,
            analysis=analysis,
            input_dq_check=input_dq_check,
            input_dq_strict=input_dq_strict,
            input_dq_thresholds=input_dq_thresholds,
            auto_warmup=auto_warmup and inc.is_full_run,
            trim_warmup=trim_warmup,
            market=resolved_market,
            pit_enforce=False,
            pit_forbid_forward_fill=False,
        )
        sliced = slice_factor_result_for_incremental(output["result"], inc)
        output["result"] = sliced
        output["incremental"] = {**inc.to_dict(), "market": resolved_market}
        output["analysis"] = analysis
        output["plan"] = plan
        return output

    def materialize_incremental(
        self,
        factor: Factor,
        *,
        factor_id: str | None = None,
        since: str | None = None,
        end_date: str | None = None,
        lookback_extra: int = 5,
        recompute_tail_bars: int | None = None,
        lake_root: str | Path | None = None,
        dq_check: bool = False,
        dq_strict: bool = True,
        author: str | None = None,
        frequency: str | None = None,
        description: str | None = None,
        expression: str | None = None,
        input_dq_check: bool = False,
        input_dq_strict: bool = True,
        input_dq_thresholds=None,
        data_source_config: dict | None = None,
        write_metadata: bool = True,
        resume_materialize: bool = False,
        isolate_partition_failures: bool = True,
        market: str | None = None,
        preserve_invalid_rows: bool = False,
        value_dtype: str = "float32",
        write_target: str = "local",
        dq_thresholds=None,
        pit_enforce: bool = False,
        pit_forbid_forward_fill: bool = False,
        auto_warmup: bool = False,
        trim_warmup: bool = True,
        clickhouse_table: str | None = None,
        ch_host: str | None = None,
        ch_port: int | None = None,
        ch_database: str | None = None,
        ch_username: str | None = None,
        ch_password: str | None = None,
        ch_secure: bool | None = None,
        ch_ensure_table: bool = True,
        staging_dataset: str = "factor_lake_staging",
    ):
        """增量执行单因子并物化（watermark + lookback + upsert）。

        先 ``run_incremental`` 获取 tail 结果，再调用 ``execute_materialize``
        落盘；空结果跳过写入。支持 ``staging_clickhouse`` / ``clickhouse`` 双写。

        Args:
            factor: 待物化因子。
            factor_id: catalog 因子 ID。
            since/end_date/lookback_extra/recompute_tail_bars: 增量窗口参数。
            lake_root: 因子湖根目录。
            dq_check/dq_strict/dq_thresholds: 产出 DQ。
            author/frequency/description/expression: 元数据。
            input_dq_check/input_dq_strict/input_dq_thresholds: 输入 DQ。
            data_source_config: 数据源配置快照。
            write_metadata: 是否写入 run 元数据。
            resume_materialize/isolate_partition_failures: 分区写入策略。
            market: 市场标识。
            preserve_invalid_rows/value_dtype: 落盘格式。
            write_target: 写入目标。
            pit_enforce/pit_forbid_forward_fill: PIT 参数。
            auto_warmup/trim_warmup: warmup 参数。
            clickhouse_* / ch_*: ClickHouse 参数。
            staging_dataset: staging 数据集名。

        Returns:
            含 ``materialization`` 与 ``incremental`` 的结果字典；无新数据时
            ``materialization.skipped=True``。
        """
        target = str(write_target or "local").lower()
        output = self.run_incremental(
            factor,
            factor_id=factor_id,
            since=since,
            end_date=end_date,
            lookback_extra=lookback_extra,
            recompute_tail_bars=recompute_tail_bars,
            lake_root=lake_root,
            input_dq_check=input_dq_check,
            input_dq_strict=input_dq_strict,
            input_dq_thresholds=input_dq_thresholds,
            market=market,
            pit_enforce=pit_enforce,
            pit_forbid_forward_fill=pit_forbid_forward_fill,
            auto_warmup=auto_warmup,
            trim_warmup=trim_warmup,
        )

        if output["result"] is None or len(output["result"]) == 0:
            logger.warning("增量因子 '%s' 无新输出，跳过落盘", factor.name)
            output["materialization"] = {
                "factor_id": factor_id or factor.name,
                "rows_written": 0,
                "skipped": True,
                "incremental": output.get("incremental"),
            }
            return output

        from runtime.materialize_service import execute_materialize

        output = execute_materialize(
            self,
            factor,
            output,
            target=target,
            lake_root=lake_root,
            staging_dataset=staging_dataset,
            factor_id=factor_id,
            author=author,
            frequency=frequency,
            description=description,
            expression=expression,
            dq_check=dq_check,
            dq_strict=dq_strict,
            dq_thresholds=dq_thresholds,
            write_metadata=write_metadata,
            data_source_config=data_source_config,
            resume_materialize=resume_materialize,
            isolate_partition_failures=isolate_partition_failures,
            preserve_invalid_rows=preserve_invalid_rows,
            value_dtype=value_dtype,
            clickhouse_table=clickhouse_table,
            ch_ensure_table=ch_ensure_table,
            ch_host=ch_host,
            ch_port=ch_port,
            ch_database=ch_database,
            ch_username=ch_username,
            ch_password=ch_password,
            ch_secure=ch_secure,
            lineage_mode="incremental",
        )
        output["materialization"]["incremental"] = output.get("incremental")
        return output
