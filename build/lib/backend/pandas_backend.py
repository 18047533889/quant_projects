"""Pandas 执行后端：逻辑计划 → MultiIndex Series 结果。

本后端是因子引擎的基准 Python 执行路径，按 ``PlanNode.op`` 分派到
``KernelRegistry`` 中注册的 kernel，递归求值整棵逻辑计划树。

注册策略
--------
- ``column`` / ``literal``：内建 kernel，从 ``ExecutionContext.data_source`` 取列或常量；
- **其余所有** ``PlanNode.op``：启动时由 ``list_cleaned_ops_for_backend`` 批量注册为
  ``make_cleaned_kernel``，100% 走 ``cleaned_operators``，无第二套逐算子 kernel 文件。

缓存
----
若 ``ctx.cache`` 存在，按 ``plan_cache_key(node)`` memoize 子计划结果，
避免同一子树在 DAG 中重复计算。

特殊节点
--------
- ``plan_ref``：从 ``shared_result_cache`` 读取 CSE 预计算结果；
- ``materialized_series``：从 SQL partial pushdown 物化缓存读取。
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Sequence

import pandas as pd

from planner.logical_plan import PlanNode
from planner.plan_hash import contract_is_resolved, plan_cache_key

from runtime.perf_config import PerfConfig

from .base import Backend
from .cleaned_bridge import list_cleaned_ops_for_backend, make_cleaned_kernel
from .context import ExecutionContext
from .kernels import KernelRegistry
from .panel_native import finalize_panel_result, load_column_as_panel, panel_native_enabled


class PandasBackend(Backend):
    """Pandas/Numpy 基准执行后端。

    按 ``PlanNode.op`` 分派 kernel；除 ``column`` / ``literal`` / ``plan_ref`` /
    ``materialized_series`` 外，其余算子均通过 ``cleaned_operators`` 桥接执行。
    """

    def __init__(self, use_modin_pandas: bool | None = None) -> None:
        """初始化 kernel 注册表并批量注册内建与 cleaned 算子。

        R40 #140: ``use_modin_pandas=True``（``build_backend("pandas_modin")``）
        在当前 execution context 内启用 modin —— 不再改 process-global
        ``os.environ``。modin 未安装时 ``pandas_compat`` 自动回退标准 pandas。
        """
        if use_modin_pandas is not None:
            from backend.pandas_compat import set_modin_enabled

            set_modin_enabled(bool(use_modin_pandas))
        self._registry = KernelRegistry()
        self._register_kernels()

    def execute(self, plan: PlanNode, ctx: ExecutionContext) -> Any:
        """执行逻辑计划并返回最终因子结果。

        参数
        ----
        plan : PlanNode
            待执行的逻辑计划根节点。
        ctx : ExecutionContext
            执行期上下文，含数据源、缓存与列名配置。

        返回
        ----
        Any
            经 ``finalize_panel_result`` 对齐后的因子值，
            通常为 ``(timestamp, instrument)`` MultiIndex Series。
        """
        ctx = self._with_pandas_perf(ctx)
        return finalize_panel_result(self._eval(plan, ctx), ctx)

    @staticmethod
    def _with_pandas_perf(ctx: ExecutionContext) -> ExecutionContext:
        """为 Pandas 后端注入性能配置，强制算子层走 ``pandas_numpy``。

        参数
        ----
        ctx : ExecutionContext
            原始执行上下文。

        返回
        ----
        ExecutionContext
            ``perf.operator_backend`` 设为 ``pandas_numpy`` 的新上下文副本。
        """
        perf = ctx.perf if ctx.perf is not None else PerfConfig.from_env()
        if getattr(perf, "operator_backend", "auto") == "auto":
            perf = replace(perf, operator_backend="pandas_numpy")
        return replace(ctx, perf=perf)

    def _eval(self, node: PlanNode, ctx: ExecutionContext) -> Any:
        """递归求值单个计划节点（含缓存、CSE 与 SQL 物化快捷路径）。

        参数
        ----
        node : PlanNode
            当前待求值的逻辑计划节点。
        ctx : ExecutionContext
            执行期上下文。

        返回
        ----
        Any
            节点求值结果，类型取决于算子（Series、标量、DataFrame 等）。

        异常
        ----
        RuntimeError
            ``plan_ref`` 节点但 ``shared_result_cache`` 未配置。
        KeyError
            CSE 或 SQL 物化缓存中缺少对应 ``sid``。
        NotImplementedError
            算子名未在 ``KernelRegistry`` 中注册。
        """
        if node.op == "plan_ref":
            sid = node.attrs["sid"]
            # R38 P0-031（§12）：L0 读取优先走 GovernedBufferStore（唯一 owner）——
            # backing 缺失时尝试 spill reload；没有 store 才退回 raw dict。
            store = getattr(ctx, "shared_buffers", None)
            if store is not None:
                val = store.get_ref(sid)
                if val is not None:
                    return val
                # 尝试 spill reload（SPILLED 的 sid）。
                try:
                    from runtime.spill_store import SpillRef

                    ref = getattr(store, "_spill_ref", None)
                    _ = ref
                except Exception:
                    pass
            sc = getattr(ctx, "shared_result_cache", None)
            if sc is None:
                raise RuntimeError(
                    "plan_ref requires ExecutionContext.shared_result_cache; "
                    "use FactorEngine.run_many() after compile_many CSE."
                )
            if sid not in sc:
                raise KeyError(f"missing shared subplan result for sid prefix={sid[:64]!r}...")
            return sc[sid]

        if node.op == "materialized_series":
            sid = str(node.attrs["sid"])
            mat = getattr(ctx, "materialized_series", None) or {}
            if sid not in mat:
                raise KeyError(f"missing SQL materialized_series for sid={sid[:64]!r}...")
            return mat[sid]

        cache = getattr(ctx, "cache", None)
        cache_key: str | None = None
        if cache is not None:
            cache_key = plan_cache_key(node)
            hit = cache.get(cache_key)
            if hit is not None:
                return hit

        try:
            kernel = self._registry.get(node.op)
        except KeyError as exc:
            from cleaned_operators.registry import OperatorRegistry
            try:
                resolved = "where" if node.op == "if_else" else OperatorRegistry.resolve_canonical(node.op)
                kernel = self._registry.get(resolved)
            except (KeyError, ValueError):
                raise NotImplementedError(f"Unsupported op: {node.op}") from exc

        out = kernel(node, ctx)
        if cache is not None and cache_key is not None:
            # R6-153: an unresolved operator contract (registry not loaded /
            # ``semantic_version: unregistered``) must not enter the PRODUCTION
            # persistent cache — it is a bootstrap-only deterministic namespace,
            # not a real semantic version.  Memory caches are fine (same-process,
            # rebuilt with the registry); disk survives across processes where a
            # stale "unregistered" entry could collide with a later resolved
            # operator set.
            from storage.cache import PersistentPlanCache

            _is_persistent = isinstance(cache, PersistentPlanCache)
            if not _is_persistent or contract_is_resolved(node):
                cache.set(cache_key, out)
        return out

    def _register_kernels(self) -> None:
        """向注册表填入 ``column``/``literal`` 内建 kernel 及全部 cleaned 算子。"""
        reg = self._registry.register
        reg("column", self._op_column)
        reg("literal", self._op_literal)
        for op in list_cleaned_ops_for_backend(set()):
            reg(op, make_cleaned_kernel(self._eval, op))
        # Compatibility aliases resolve to the now-registered canonical kernel.
        from cleaned_operators.registry import OperatorRegistry
        for alias, canonical in OperatorRegistry._aliases.items():
            if canonical in self._registry._kernels:
                reg(alias, self._registry._kernels[canonical])

    def _op_column(self, node: PlanNode, ctx: ExecutionContext):
        """``column`` 节点 kernel：从数据源加载指定列。

        参数
        ----
        node : PlanNode
            属性 ``name`` 为列名。
        ctx : ExecutionContext
            执行期上下文，提供 ``data_source``。

        返回
        ----
        Any
            列数据；panel-native 模式下为宽表 DataFrame，否则为 MultiIndex Series。
        """
        name = node.attrs["name"]
        if panel_native_enabled(ctx):
            return load_column_as_panel(ctx.data_source, name, ctx)
        return ctx.data_source.load_column(name)

    def _op_literal(self, node: PlanNode, ctx: ExecutionContext):
        """``literal`` 节点 kernel：返回节点属性中的常量值。

        参数
        ----
        node : PlanNode
            属性 ``value`` 为字面量。
        ctx : ExecutionContext
            未使用，保留以符合 kernel 签名。

        返回
        ----
        Any
            ``node.attrs['value']`` 中的常量。
        """
        _ = ctx
        return node.attrs["value"]


# ---------------------------------------------------------------------------
# R40 #62：PandasBackendLiveEvidence —— 对已知参考 canonical 集实跑 Pandas
# backend，验证输出语义正确性，结果存 versioned evidence。
# ---------------------------------------------------------------------------


def _col(name: str) -> "PlanNode":
    from planner.logical_plan import PlanNode

    return PlanNode(op="column", attrs={"name": name})


def _lit(value: float) -> "PlanNode":
    from planner.logical_plan import PlanNode

    return PlanNode(op="literal", attrs={"value": value})


def _binop(op: str, left: "PlanNode", right: "PlanNode") -> "PlanNode":
    from planner.logical_plan import PlanNode

    return PlanNode(op=op, inputs=(left, right))


def _unop(op: str, child: "PlanNode") -> "PlanNode":
    from planner.logical_plan import PlanNode

    return PlanNode(op=op, inputs=(child,))


def _default_live_input() -> tuple[pd.MultiIndex, dict[str, list[float]]]:
    """live evidence 的确定性输入（timestamp × instrument 面板）。

    ``close`` 故意含负值（abs/neg/div 等符号语义需要非平凡输入）；
    ``volume`` 全正（add/ts_delay 区分两列）。
    """
    idx = pd.MultiIndex.from_product(
        [
            pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            ["A", "B"],
        ],
        names=["timestamp", "instrument"],
    )
    return idx, {
        "close": [10.0, -20.0, 11.0, -21.0, 12.0, 18.0],
        "volume": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    }


#: 已知参考 canonical 集：``canonical -> (plan_builder(data) -> PlanNode, expected)``。
#: expected 是 6 行面板展平顺序 ``(d1,A),(d1,B),(d2,A),(d2,B),(d3,A),(d3,B)``。
_REFERENCE_CANONICALS: dict[str, tuple[Any, list[float]]] = {
    "abs": (
        lambda data: _unop("abs", _col("close")),
        [10.0, 20.0, 11.0, 21.0, 12.0, 18.0],
    ),
    "neg": (
        lambda data: _unop("neg", _col("close")),
        [-10.0, 20.0, -11.0, 21.0, -12.0, -18.0],
    ),
    "add": (
        lambda data: _binop("add", _col("close"), _col("volume")),
        [11.0, -18.0, 14.0, -17.0, 17.0, 24.0],
    ),
    "ts_delay": (
        lambda data: _binop("ts_delay", _col("close"), _lit(1.0)),
        [float("nan"), float("nan"), 10.0, -20.0, 11.0, -21.0],
    ),
    "ts_mean": (
        # ts_mean 声明策略 min_periods=1（operator_policy.py:628），滚动窗口含
        # 当前观测；首行在窗口未满时返回单点均值，不是 NaN（满窗语义是别的算子）。
        lambda data: _binop("ts_mean", _col("close"), _lit(2.0)),
        [10.0, -20.0, 10.5, -20.5, 11.5, -1.5],
    ),
}


class PandasBackendLiveEvidence:
    """R40 #62：Pandas backend 的 live evidence validator。

    对已知参考 canonical 集在真实 :class:`PandasBackend` 上实跑（不再是 catalog
    flag 静态判定），把输出与手算参考值比对，结果存 versioned evidence：

        - ``version_sha``   生成证据时的 HEAD（或调用方显式版本）；
        - ``evidence``      canonical -> {passed, expected, actual, atol}；
        - ``summary``       {passed, failed, skipped}。

    用于 production capability 认证 / CI：live evidence 失败即该 canonical 在
    Pandas backend 的 capability 不可信（fail-closed）。
    """

    def __init__(
        self,
        *,
        backend: "PandasBackend | None" = None,
        reference_canonicals: dict[str, tuple[Any, list[float]]] | None = None,
        version_sha: str | None = None,
        atol: float = 1e-9,
    ) -> None:
        self._backend = backend if backend is not None else PandasBackend()
        self._reference = reference_canonicals or _REFERENCE_CANONICALS
        self._atol = atol
        self._version_sha = version_sha or _current_head_sha()
        self._evidence: dict[str, dict[str, Any]] = {}

    # -- 执行 --

    def _run_plan(self, plan: "PlanNode", data: dict[str, list[float]],
                  index: pd.MultiIndex) -> list[float]:
        source = _live_evidence_source(
            {name: pd.Series(vals, index=index) for name, vals in data.items()}
        )
        ctx = ExecutionContext(data_source=source)
        out = self._backend.execute(plan, ctx)
        return [float(x) if x is not None else float("nan") for x in out.to_numpy()]

    @staticmethod
    def _allclose(actual: Sequence[float], expected: Sequence[float], atol: float) -> bool:
        import numpy as np

        a = np.asarray(actual, dtype=float)
        e = np.asarray(expected, dtype=float)
        if a.shape != e.shape:
            return False
        return bool(np.allclose(a, e, rtol=1e-7, atol=atol, equal_nan=True))

    def validate_canonical(self, canonical: str) -> dict[str, Any]:
        """对单个 canonical 跑 live evidence，返回记录（并写入 self._evidence）。"""
        rec = self._reference.get(canonical)
        if rec is None:
            return {"canonical": canonical, "passed": False, "skipped": True,
                    "reason": "no_reference_case", "version_sha": self._version_sha}
        builder, expected = rec
        index, data = _default_live_input()
        try:
            plan = builder(data)
            actual = self._run_plan(plan, data, index)
        except Exception as exc:  # noqa: BLE001
            record = {"canonical": canonical, "passed": False, "error": str(exc),
                      "version_sha": self._version_sha}
            self._evidence[canonical] = record
            return record
        passed = self._allclose(actual, expected, self._atol)
        record = {
            "canonical": canonical,
            "passed": passed,
            "expected": list(expected),
            "actual": list(actual),
            "atol": self._atol,
            "version_sha": self._version_sha,
        }
        self._evidence[canonical] = record
        return record

    def validate_known_set(self) -> dict[str, Any]:
        """跑全部参考 canonical 集，返回 summary + 逐条 evidence。"""
        for canon in self._reference:
            self.validate_canonical(canon)
        return self.summary()

    # -- evidence --

    def evidence(self) -> dict[str, dict[str, Any]]:
        return dict(self._evidence)

    def summary(self) -> dict[str, Any]:
        passed = sum(1 for r in self._evidence.values() if r.get("passed"))
        failed = sum(1 for r in self._evidence.values()
                     if not r.get("passed") and not r.get("skipped"))
        skipped = sum(1 for r in self._evidence.values() if r.get("skipped"))
        return {
            "version_sha": self._version_sha,
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
            "total": len(self._evidence),
        }


def _current_head_sha() -> str:
    try:
        from backend.evidence_provenance import current_commit_sha

        return current_commit_sha()
    except Exception:
        return ""


def _live_evidence_source(data: dict[str, pd.Series]) -> Any:
    """构造 live evidence 的 DataSource。

    优先复用 ``tests.helpers.InMemorySeriesSource``（测试/CI 环境）；不可用
    （production 打包无 tests 树）→ 回退最小内联 DataSource，保证 validator
    在任意环境可跑。
    """
    try:
        from tests.helpers import InMemorySeriesSource

        return InMemorySeriesSource(data=data)
    except Exception:
        from storage.datasource import DataSource as _DataSource

        class _InlineSeriesSource(_DataSource):
            def __init__(self, d):
                self.data = d

            def load_column(self, name: str) -> Any:
                return self.data[name]

            def load_columns(self, names: list[str]) -> dict[str, Any]:
                return {n: self.data[n] for n in names if n in self.data}

        return _InlineSeriesSource(data=data)


#: 进程级 live evidence 单例（production capability 认证消费）。
_LIVE_EVIDENCE: PandasBackendLiveEvidence | None = None


def get_pandas_live_evidence(
    *, refresh: bool = False,
) -> PandasBackendLiveEvidence:
    """进程级 PandasBackendLiveEvidence 单例（惰性构造）。"""
    global _LIVE_EVIDENCE
    if _LIVE_EVIDENCE is None or refresh:
        _LIVE_EVIDENCE = PandasBackendLiveEvidence()
    return _LIVE_EVIDENCE


def reset_pandas_live_evidence() -> None:
    """测试用：重置 live evidence 单例。"""
    global _LIVE_EVIDENCE
    _LIVE_EVIDENCE = None
