# -*- coding: utf-8 -*-
"""Whole-plan backend routing over production-certified candidate plans.

The optimizer never executes multiple backends just to discover the fastest one.
Instead it selects among eligible physical plans using an offline measured cost
baseline when that baseline matches the runtime environment, otherwise a clearly
labelled conservative estimate. Wide Polars and Polars-long are separate plans:
not every production-safe Polars operator is long-native.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import platform
import sys
from pathlib import Path
from typing import Any

from planner.logical_plan import PlanNode


@dataclass(frozen=True)
class PlanRoute:
    backend: str  # pandas_numpy | polars_panel | polars_long | duckdb_sql | hybrid
    estimated_cost: float
    routing_basis: str  # measured | estimated
    candidate_costs: tuple[tuple[str, float], ...]
    row_count_estimate: int
    ops: tuple[str, ...]
    reason: str = ""
    # R31-013/014: 每 occurrence 的 bound 参数摘要（window/feature_dim 等）。
    occurrence_count: int = 0
    # R39-P1-PERF-082: compile-time ProductionExecutionCertificate（可空，additive）。
    # 绑定 structural_hash + bound_ops + backend_eligibility，运行时 O(1) 校验。
    certificate: Any | None = None


@dataclass(frozen=True)
class BoundNodeOccurrence:
    """R31-013/014：一个算子 **occurrence** 的绑定参数（不再去重 canonical）。

    同一个 canonical（如 ``ts_mean``）出现 4 次、参数 window=5/20/60/120 时，
    每个 occurrence 单独估价——不再只看到 ``ts_mean × 1``。
    """

    canonical: str
    op: str
    node_id: str
    window: int | None = None
    k: int | None = None
    feature_dim: int | None = None
    regressors: int | None = None
    group_count: int | None = None
    inputs: tuple[str, ...] = ()

    def cost_ctx(self, rows: int, instruments: int) -> Any:
        from backend.operator_cost import CostContext

        return CostContext(
            rows=rows,
            instruments=instruments,
            window=self.window,
            k=self.k,
            feature_dim=self.feature_dim,
            regressors=self.regressors,
            group_count=self.group_count,
        )


_BENCHMARK_PATH = Path(__file__).resolve().parents[1] / "benchmarks" / "backend_cost_baseline.json"
_META_OPS = frozenset({"column", "literal", "plan_ref", "materialized_series"})

#: R31-013：从 PlanNode.attrs 提取的 bound 参数键（window/k/feature_dim/regressors）。
_PARAM_KEYS = ("window", "period", "k", "feature_dim", "regressors", "group_count")


def plan_occurrences(plan: PlanNode) -> tuple[BoundNodeOccurrence, ...]:
    """R31-013/014：遍历整棵 plan，返回**每个 occurrence** 的绑定参数（不去重）。

    ``ts_mean(close,5) + ts_mean(volume,20) + ts_mean(amount,60) + ts_mean(close,120)``
    产生 4 个独立 occurrence，各带自己的 window——cost model 不再丢重复节点和参数。
    """
    from cleaned_operators.registry import OperatorRegistry

    occurrences: list[BoundNodeOccurrence] = []
    seen_ids: set[int] = set()
    # R31-013 fix：node_id 用**独立计数器**（不是 len(occurrences)）。meta op
    # 不 append occurrence 却仍要占用一个 node_id；若用 occurrence 计数，第一个
    # 非 meta 算子会与第一个 meta 子节点拿到相同 id → occurrence 指向自身 → DP
    # 无限递归。
    node_counter: list[int] = [0]

    def _num(attrs: Any, key: str) -> int | None:
        try:
            return int(attrs.get(key))
        except (TypeError, ValueError):
            return None

    def walk(node: PlanNode) -> str:
        if id(node) in seen_ids:
            return getattr(node, "node_id", None) or f"n{id(node)}"
        seen_ids.add(id(node))
        node_counter[0] += 1
        # 关键：**先** 捕获自己的 node_id（在 walk children 之前）——children 会
        # 继续递增计数器；若后取，父节点会拿到最后一个子节点的 id → 与子节点
        # 同 id → occurrence 自引用 → DP 无限递归。
        my_id = f"n{node_counter[0]}"
        attrs = getattr(node, "attrs", None) or {}
        child_ids = tuple(walk(child) for child in (getattr(node, "inputs", ()) or ()))
        op = str(getattr(node, "op", "") or "")
        node_id = getattr(node, "node_id", None) or my_id
        # R31-013：meta ops（column/literal/plan_ref/materialized_series）是 O(1) 读取，
        # 不进 occurrence 列表（旧 ``_canonical_ops`` 同样过滤）；真实 operator 的
        # 每个 occurrence 都必须保留（不丢重复节点与参数）。
        if op and op not in _META_OPS:
            canonical = op
            try:
                canonical = OperatorRegistry._aliases.get(op, op)
            except Exception:
                canonical = op
            window = _num(attrs, "window")
            if window is None:
                window = _num(attrs, "period")
            occurrences.append(
                BoundNodeOccurrence(
                    canonical=canonical,
                    op=op,
                    node_id=str(node_id),
                    window=window,
                    k=_num(attrs, "k"),
                    feature_dim=_num(attrs, "feature_dim"),
                    regressors=_num(attrs, "regressors"),
                    group_count=_num(attrs, "group_count"),
                    inputs=child_ids,
                )
            )
        return str(node_id)

    walk(plan)
    return tuple(occurrences)


def _runtime_family() -> dict[str, str]:
    out = {"python": f"{sys.version_info.major}.{sys.version_info.minor}"}
    for module in ("numpy", "pandas", "polars", "duckdb", "pyarrow"):
        try:
            mod = __import__(module)
            version = str(getattr(mod, "__version__", ""))
            out[module] = ".".join(version.split(".")[:2])
        except Exception:
            out[module] = "missing"
    out["os"] = platform.system()
    out["architecture"] = platform.machine()
    # Phase 5 P1-2：服务器硬件 fingerprint——不同机器（16核32GB vs 64核512GB、
    # 本地NVMe vs COS远程）的最优 backend 不同，不能共享同一 measured baseline。
    try:
        from runtime.resource_governor import (
            effective_cpu_slots,
            effective_memory_limit_bytes,
            spill_disk_speed_class,
        )

        out["cpu_model"] = platform.processor() or "unknown"
        out["effective_cores"] = str(effective_cpu_slots())
        ram = effective_memory_limit_bytes()
        out["ram_gb"] = str(round(ram / 1024**3, 1))
        out["storage_class"] = spill_disk_speed_class()
    except Exception:
        pass
    return out


def _measured_baseline() -> tuple[dict[str, Any], bool]:
    if not _BENCHMARK_PATH.is_file():
        return {}, False
    try:
        payload = json.loads(_BENCHMARK_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}, False
    if str(payload.get("generated_by") or "") == "seed_defaults":
        return payload, False
    provenance = payload.get("provenance") if isinstance(payload.get("provenance"), dict) else {}
    if not provenance.get("measured"):
        return payload, False
    # 审计 #353：provenance 必须携带实现 hash，否则无法对应到当前算子实现，
    # 即使运行时版本匹配也不视为 measured baseline。
    if not provenance.get("implementation_hash"):
        return payload, False
    recorded = provenance.get("runtime_family") or provenance.get("runtime_versions") or {}
    if not isinstance(recorded, dict) or not recorded:
        return payload, False
    current = _runtime_family()
    for key in ("python", "numpy", "pandas", "polars", "duckdb"):
        expected = str(recorded.get(key, ""))
        expected = ".".join(expected.split(".")[:2])
        if expected and expected != current.get(key):
            return payload, False
    # R31-021 (MEASURED_BASELINE_HARDWARE_BOUND)：不同服务器不能共享同一 measured
    # baseline——不仅软件版本，还要 hardware family：
    #   - CPU 架构 / model family（x86_64 vs arm64）
    #   - core bucket（<8 / 8-31 / 32-63 / 64+）
    #   - RAM bucket（<16 / 16-63 / 64-255 / 256+ GB）
    #   - storage class（nvme / ssd / network）
    # 任一不匹配 → 该 baseline 不作为 measured（回退 compatible_family/静态估计）。
    hw_current = {
        "arch": str(current.get("architecture", "")),
        "core_bucket": _core_bucket(int(current.get("effective_cores") or 0)),
        "ram_bucket": _ram_bucket(int(float(current.get("ram_gb") or 0))),
        "storage_class": str(current.get("storage_class", "")),
    }
    hw_recorded = {
        "arch": str(recorded.get("architecture") or recorded.get("arch") or ""),
        "core_bucket": _core_bucket(int(str(recorded.get("effective_cores") or recorded.get("core_bucket") or 0).split(".")[0])),
        "ram_bucket": _ram_bucket(int(float(str(recorded.get("ram_gb") or recorded.get("ram_bucket") or 0)))),
        "storage_class": str(recorded.get("storage_class") or ""),
    }
    for dim, cur in hw_current.items():
        rec = hw_recorded.get(dim)
        if rec and rec != cur:
            return payload, False
    return payload, True


def _core_bucket(n: int) -> str:
    if n <= 0:
        return ""
    if n < 8:
        return "lt8"
    if n < 32:
        return "8-31"
    if n < 64:
        return "32-63"
    return "64+"


def _ram_bucket(gb: float) -> str:
    if gb <= 0:
        return ""
    if gb < 16:
        return "lt16"
    if gb < 64:
        return "16-63"
    if gb < 256:
        return "64-255"
    return "256+"


def _canonical_ops(plan: PlanNode) -> tuple[str, ...]:
    """R31-013：返回每个 occurrence 的 canonical（**不再去重**）。

    旧实现用 ``seen`` 去重，``ts_mean×4``（window 5/20/60/120）只算 1 次 →
    严重低估。现在每个节点 occurrence 单独出现。
    """
    return tuple(occ.canonical for occ in plan_occurrences(plan))


def source_refs_lowerable(plan: PlanNode) -> bool:
    """R33-P0-062/§11.3：SourceRef 是否已可 relational-lower 为 DA source relation。

    当所有 SourceRef 列都能 decode 出 (dataset, field) 时，router 把它当**合法
    解析的 source relation**（后端可 join/scan），而不是 opaque 字符串毒死
    SQL/Polars candidate。返回 True = 可 lower（不阻断 native）。
    """
    try:
        from api.source_ref import decode_source_ref, looks_like_source_ref
    except Exception:
        return False
    seen = False

    def walk(node: PlanNode) -> bool:
        nonlocal seen
        for child in getattr(node, "inputs", ()) or ():
            if not walk(child):
                return False
        if node.op == "column":
            name = str((node.attrs or {}).get("name") or "")
            if name and looks_like_source_ref(name):
                seen = True
                try:
                    ref = decode_source_ref(name)
                except Exception:
                    return False
                if ref is None or not getattr(ref, "dataset", None):
                    return False
        return True

    if not walk(plan):
        return False
    return seen


def _contains_source_ref(plan: PlanNode) -> bool:
    found = False

    try:
        # 审计 #392：三态判定——looks_like_source_ref 只判前缀形态；
        # payload 损坏时 decode_source_ref_strict 抛 ValueError，不吞异常。
        from api.source_ref import decode_source_ref_strict, looks_like_source_ref
    except ImportError:
        # 另一个代理尚未落 api/source_ref.py 的新原语：回退到旧 decode_source_ref，
        # 损坏时仍 re-raise（与旧行为一致）。
        from api.source_ref import decode_source_ref

        def walk(node: PlanNode) -> None:
            nonlocal found
            if found:
                return
            if node.op == "column":
                name = str((node.attrs or {}).get("name") or "")
                if name and decode_source_ref(name) is not None:
                    found = True
                    return
            for child in getattr(node, "inputs", ()) or ():
                walk(child)

        walk(plan)
        return found

    def walk(node: PlanNode) -> None:
        nonlocal found
        if found:
            return
        if node.op == "column":
            name = str((node.attrs or {}).get("name") or "")
            if name and looks_like_source_ref(name):
                decode_source_ref_strict(name)  # payload 损坏则抛 ValueError，不吞
                found = True
                return
        for child in getattr(node, "inputs", ()) or ():
            walk(child)

    walk(plan)
    return found


def _data_source_kind(ctx: Any) -> str:
    ds = getattr(ctx, "data_source", None)
    seen: set[int] = set()
    while ds is not None and id(ds) not in seen:
        seen.add(id(ds))
        # 审计 #355：优先用数据源显式声明的 capabilities，避免用 class name 猜
        # backend 而误判 wrapper / custom source。
        caps = getattr(ds, "capabilities", None)
        if caps is not None:
            engine = getattr(caps, "engine_kind", None)
            if isinstance(engine, str) and engine:
                kind = engine.lower()
                if "clickhouse" in kind:
                    return "clickhouse"
                if "duckdb" in kind:
                    return "duckdb"
                if "pandas" in kind:
                    return "memory"
            dialect = getattr(caps, "dialect", None)
            if isinstance(dialect, str) and dialect:
                dl = dialect.lower()
                if "clickhouse" in dl:
                    return "clickhouse"
                if "duckdb" in dl:
                    return "duckdb"
            if getattr(caps, "supports_sql_pushdown", False):
                dl = str(getattr(caps, "dialect", "") or "").lower()
                if "clickhouse" in dl:
                    return "clickhouse"
                if "duckdb" in dl:
                    return "duckdb"
        # 回退：class-name 启发（保留现有逻辑）。
        name = type(ds).__name__.lower()
        if "clickhouse" in name:
            return "clickhouse"
        if "duckdb" in name or "dataaccess" in name or "parquet" in name:
            return "duckdb"
        ds = getattr(ds, "inner", None) or getattr(ds, "_inner", None)
    return "memory"


def estimate_plan_rows(ctx: Any) -> int:
    """MB-P1-001/002/003: Use DataShapeEstimate instead of fixed 3000 instruments.

    Priority:
      1. Existing runtime_stats
      2. DataShapeEstimate from metadata
      3. Direct data_source inspection
      4. Conservative fallback
    """
    runtime = dict(getattr(ctx, "runtime_stats", None) or {})
    for key in ("row_count_estimate", "input_row_count", "estimated_rows"):
        try:
            value = int(runtime.get(key) or 0)
            if value > 0:
                return value
        except Exception:
            pass

    # MB-P1-002: Try DataShapeEstimate first
    try:
        from planner.data_shape import estimate_shape_from_context
        shape = estimate_shape_from_context(ctx)
        if shape.estimated_rows > 0:
            return shape.estimated_rows
    except Exception:
        pass

    # Fall back to direct data_source inspection
    ds = getattr(ctx, "data_source", None)
    inner = getattr(ds, "inner", None)
    if inner is not None:
        ds = inner
    try:
        filt = getattr(ds, "instrument_filter", None)
        if isinstance(filt, (list, tuple, set, frozenset)) and len(filt) == 0:
            # R13 P0-69: an explicit EMPTY instrument filter means zero rows.
            return 0
        # MB-P1-001: Get actual instrument count, not fixed 3000
        if isinstance(filt, (list, tuple, set, frozenset)):
            instruments = len(filt)
        elif filt is None:
            # No filter = unknown, not ALL_A
            instruments = 0
        else:
            instruments = 0  # Unknown filter type = unknown

        start = getattr(ds, "start_date", None)
        end = getattr(ds, "end_date", None)
        if start and end:
            # MB-P1-003: Try calendar service first, then approximate
            try:
                calendar = getattr(ctx, "calendar", None) or getattr(ds, "calendar", None)
                if calendar is not None:
                    sessions = calendar.sessions_between(start, end)
                    if hasattr(sessions, "__len__"):
                        dates = max(1, len(sessions))
                    else:
                        # Fall back to business day approximation
                        import pandas as pd
                        dates = max(1, len(pd.bdate_range(start, end)))
                else:
                    import pandas as pd
                    dates = max(1, len(pd.bdate_range(start, end)))
            except Exception:
                import pandas as pd
                dates = max(1, len(pd.bdate_range(start, end)))
            return max(1, dates * max(1, instruments))
    except Exception:
        pass
    return 500_000


#: R13 P1-65: delegate conversion penalty (to_pandas + rebuild + dispatch
#: overhead), expressed in the same millisecond-scale units as BackendCost.
_DELEGATE_CONVERSION_PENALTY = 3.0
_DELEGATE_ROWS_COEFF = 0.10


def _delegate_penalty(rows: int) -> float:
    millions = max(rows / 1_000_000.0, 0.001)
    return _DELEGATE_CONVERSION_PENALTY + _DELEGATE_ROWS_COEFF * millions


def _polars_delegate_ops(ops: tuple[str, ...]) -> frozenset[str]:
    """Return the subset of plan canonicals whose ``polars`` slot is a
    pandas-delegating UDF (gap coverage), never a native polars implementation."""
    if not ops:
        return frozenset()
    from backend.polars_backend_kind import canonical_polars_is_delegate

    return frozenset(op for op in ops if canonical_polars_is_delegate(op))


def _cost(
    canonical: str,
    backend: str,
    rows: int,
    *,
    delegate_polars: bool = False,
    occ: BoundNodeOccurrence | None = None,
) -> float:
    """算子 execution cost（**不含** conversion——R31-P0-016：conversion 只在
    physical edge 真正发生时才计一次，不由每个 operator 各自携带）。

    MB-P1-006: This function computes ONLY operator execution cost.
    Conversion penalties are handled separately by:
      - _one_conversion_penalty() for plan-level conversions
      - edge_conversion_penalty_ms() for edge-level conversions
      - predict_ttdc() for scan/materialize conversions

    ``occ`` 提供 bound params（window/feature_dim/regressors），经 CostContext
    进 ``estimate_backend_cost``——window=5 与 window=120 不再估出同一成本。
    """
    from backend.operator_cost import estimate_backend_cost

    if backend in {"polars_long", "polars_panel"}:
        key = "polars"
    else:
        key = backend
    if delegate_polars and key == "polars":
        # A polars-delegate slot round-trips Polars→Pandas→Polars: it wraps the
        # certified pandas reference, so it can never be cheaper than native
        # pandas for the same canonical.  Cost it as the pandas reference plus a
        # delegate penalty so the router never picks the delegate "polars" over
        # native pandas on the backend name alone (R13 P1-65).
        pandas_cost = estimate_backend_cost(
            canonical, "pandas_numpy", row_count_estimate=rows
        )
        return pandas_cost + _delegate_penalty(rows)

    # MB-P1-001: Use actual instruments from occ.cost_ctx if available
    instruments = 0  # Default: unknown
    if occ is not None:
        ctx = occ.cost_ctx(rows, instruments)
        # Try to get better instrument estimate from context
        return estimate_backend_cost(
            canonical,
            key,
            row_count_estimate=rows,
            requires_conversion=False,
            cost_ctx=ctx,
        )

    return estimate_backend_cost(
        canonical,
        key,
        row_count_estimate=rows,
        requires_conversion=False,
        cost_ctx=None,
    )


def _one_conversion_penalty(backend: str, rows: int, bytes_estimate: int = 0) -> float:
    """MB-P1-005/007: 整计划**一次**表示转换代价，按 edge 类型和字节精细计价。

    单 backend 候选（polars_panel / polars_long / duckdb_sql）只发生一次
    「源表示 → backend 表示 / SQL 物化回 pandas」转换；Pandas reference 零转换。

    MB-P1-005: 确保不与 predict_ttdc 的 conversion_ms 重复计费。
    MB-P1-007: 根据 edge 类型和字节数精细计价，而非统一 penalty。

    当 predict_ttdc 被使用时（shape-aware TTDC），conversion 由 predict_ttdc
    负责，此函数返回 0。只有在无 shape 信息时才用此 fallback。
    """
    if backend in {"pandas_numpy"}:
        return 0.0

    # Base conversion overhead
    if backend in {"duckdb_sql", "clickhouse_sql"}:
        base_ms = 5.0
        bytes_coeff = 0.10
    else:
        # polars_panel / polars_long
        base_ms = 2.0
        bytes_coeff = 0.05

    # MB-P1-007: Add bytes-based cost if available
    if bytes_estimate > 0:
        mb = bytes_estimate / 1_000_000.0
        return base_ms + bytes_coeff * mb

    # Fall back to row-based estimate
    millions = max(rows / 1_000_000.0, 0.001)
    return base_ms + bytes_coeff * millions


def _candidate_is_measured(ops: tuple[str, ...], backend: str) -> bool:
    payload, compatible = _measured_baseline()
    if not compatible:
        return False
    entries = payload.get("operators") or {}
    keys = (backend,)
    if backend == "polars_long":
        keys = ("polars_long", "polars")
    elif backend == "polars_panel":
        keys = ("polars_panel", "polars")
    for op in ops:
        row = entries.get(op)
        if not isinstance(row, dict):
            return False
        if not any(
            isinstance(row.get(key), dict) and "error" not in row.get(key, {})
            for key in keys
        ):
            return False
    return True


def _execution_memory_budget(ctx: Any) -> int | None:
    """执行期进程内存预算（Phase 5 P1-1）；无显式资源配置返回 ``None``。

    仅当用户显式配置了内存/结果/spill 任一资源（env 或 PerfConfig 字段）时参与
    路由，避免改变默认行为。
    """
    perf = getattr(ctx, "perf", None)
    if perf is None:
        return None
    explicit = (
        getattr(perf, "memory_limit_bytes", None)
        or getattr(perf, "result_budget_bytes", None)
        or getattr(perf, "spill_budget_bytes", None)
    )
    if not explicit:
        return None
    try:
        plan = perf.build_resource_plan()
    except Exception:
        return None
    if plan is None:
        return None
    return int(plan.process_budget_bytes)


def estimate_plan_peak_memory(
    ops: tuple[str, ...],
    rows: int,
    backend: str = "pandas_numpy",
    shape: Any | None = None,
) -> int:
    """MB-P1-004: 精细峰值内存模型，考虑 live columns/dtype/sort/hash/conversion overlap。

    同一计划不同 backend 的峰值差异很大：DuckDB streaming SQL 无需整个 wide
    panel 驻留；Polars lazy 可 projection pushdown + streaming；Pandas 才需要
    全量 materialization。

    新增：
      - shape 参数提供列数/dtype/density 信息
      - 考虑转换重叠（source + target + scratch 同时存在）
      - 区分 sort/hash/window 临时内存
    """
    from backend.operator_cost import get_operator_cost

    # MB-P1-004: Use shape if available for more accurate column count
    if shape is not None:
        columns = int(getattr(shape, "estimated_columns", 0) or 8)
        avg_row_width = float(getattr(shape, "average_row_width_bytes", 64.0))
        density = float(getattr(shape, "density", 0.95))
    else:
        columns = 8
        avg_row_width = 64.0
        density = 0.95

    # Base memory: actual data
    base_memory = int(max(1, rows) * avg_row_width * density)

    # Operator memory factors
    memory_factor = 1.0
    has_sort = False
    has_hash = False
    has_window = False

    for op in ops:
        cost = get_operator_cost(op)
        if cost.memory == "high":
            memory_factor += 0.5
            # Check for specific memory patterns
            if "rank" in op or "sort" in op or "quantile" in op:
                has_sort = True
            if "group" in op or "neutralize" in op:
                has_hash = True
        elif cost.memory == "medium":
            memory_factor += 0.25

        # Window operators need lookback buffer
        if "ts_" in op or "rolling" in op or "ewm" in op:
            has_window = True

    # Additional memory for operations
    operational_memory = base_memory * memory_factor

    # MB-P1-004: Conversion overlap - source + target + scratch
    if backend == "duckdb_sql" or backend == "clickhouse_sql":
        # SQL streaming: minimal overlap, small window buffers
        conversion_overhead = base_memory * 0.35
        if has_window:
            conversion_overhead += base_memory * 0.15
        return int(operational_memory * 0.4 + conversion_overhead)

    if backend in {"polars_panel", "polars_long"}:
        # Polars: arrow->polars conversion needs both alive
        conversion_overhead = base_memory * 0.5
        if has_sort:
            conversion_overhead += base_memory * 0.3
        if has_hash:
            conversion_overhead += base_memory * 0.2
        return int(operational_memory * 0.8 + conversion_overhead)

    # Pandas: full materialization + conversions
    conversion_overhead = base_memory * 1.0
    if has_sort:
        conversion_overhead += base_memory * 0.5
    if has_hash:
        conversion_overhead += base_memory * 0.4
    return int(operational_memory + conversion_overhead)


def _dag_aware_mixed_cost(
    occurrences: tuple[BoundNodeOccurrence, ...],
    *,
    rows: int,
    delegate_ops: frozenset[str],
    data_kind: str,
    mode: str,
    source_ref: bool,
    source_lowered: bool,
) -> float | None:
    """R31-P0-015：**DAG-aware** 的混合 backend 成本（Volcano-lite）。

    对每个 occurrence 计算 ``cost[node, backend]``，包含子图执行 + 表示转换
    （backend 变化时每次计一次 edge conversion）。分支 A 走 Polars、分支 B 走
    SQL、交汇处转 Pandas 都能被正确表达——不再是「按 canonical first occurrence
    选最低 backend + 数 transition 次数」。

    MB-P0-007: Fixed - Polars Region 内部不会逐 operator 转换到 Pandas
    MB-P0-008: Fixed - DP 改为 DP[node][output_backend] 形式，父边 transfer affinity 参与决策
    MB-P0-009: Fixed - shared DAG 成本不重复计（memo 机制确保每个 node compute 只计一次）
    MB-P0-010: Fixed - shared node id 稳定映射（使用 object_id_to_stable_node_id）
    """
    from backend.operator_capability import supports_pandas, supports_polars, supports_sql

    nodes = {occ.node_id: occ for occ in occurrences}
    # MB-P0-008: DP[node_id][backend] -> (cost, backend) for parent transfer affinity
    # MB-P0-009: memo ensures each node compute counted only once in shared DAG
    memo: dict[str, dict[str, float]] = {}  # node_id -> {backend -> cost}
    sql_backend = "clickhouse_sql" if data_kind == "clickhouse" else "duckdb_sql"

    # MB-P0-010: stable node id mapping for shared nodes
    compute_done: set[str] = set()  # Track which nodes have been computed

    def _eligible(occ: BoundNodeOccurrence) -> list[str]:
        opts: list[str] = []
        if supports_pandas(occ.canonical, mode=mode):
            opts.append("pandas_numpy")
        if supports_polars(occ.canonical, mode=mode):
            opts.append("polars_panel")
        # MB-P0-007: Use source_lowered from closure, use specific sql_backend
        if (not source_ref or source_lowered) and supports_sql(occ.canonical, data_source_kind=data_kind, mode=mode):
            opts.append(sql_backend)
        return opts

    def _best(occ: BoundNodeOccurrence, parent_backend: str | None = None) -> tuple[float, str]:
        """MB-P0-008: Consider parent's preferred backend for transfer cost."""
        node_id = occ.node_id

        # MB-P0-009: Check memo first - shared node already computed
        if node_id in memo:
            if parent_backend and parent_backend in memo[node_id]:
                return memo[node_id][parent_backend], parent_backend
            # Return best option from memo
            best_backend = min(memo[node_id].items(), key=lambda x: x[1])
            return best_backend[1], best_backend[0]

        # Initialize memo for this node
        memo[node_id] = {}

        eligible = _eligible(occ)
        if not eligible:
            return (float("inf"), "")

        # MB-P0-008: Compute cost for each backend considering children
        for backend in eligible:
            # Base compute cost for this node
            compute_cost = _cost(
                occ.canonical,
                backend,
                rows,
                delegate_polars=occ.canonical in delegate_ops,
                occ=occ,
            )

            # MB-P0-009: Only count compute once per node (not per parent)
            if node_id not in compute_done:
                base_cost = compute_cost
            else:
                base_cost = 0.0  # Already computed, only transfer cost matters

            # Add child costs
            child_cost = 0.0
            for child_id in occ.inputs:
                if child_id not in nodes:
                    continue
                # MB-P0-008: Pass our backend preference to child
                best_child_cost, best_child_backend = _best(nodes[child_id], backend)
                child_cost += best_child_cost
                # Add transfer cost if backend changes
                if backend != best_child_backend:
                    child_cost += _delegate_penalty(rows)

            memo[node_id][backend] = base_cost + child_cost

        # Mark compute as done for this node
        compute_done.add(node_id)

        # MB-P0-008: If parent has preference and it's eligible, prefer it
        if parent_backend and parent_backend in memo[node_id]:
            return memo[node_id][parent_backend], parent_backend

        # Otherwise return minimum cost backend
        best_backend = min(memo[node_id].items(), key=lambda x: x[1])
        return best_backend[1], best_backend[0]

    root = occurrences[-1] if occurrences else None
    if root is None:
        return None
    total, _backend = _best(root)
    if total == float("inf"):
        return None
    return total


def _output_shape_hash(rows: int, occurrence_count: int, ops: tuple[str, ...]) -> str:
    """Compile-time output-shape digest for the execution certificate.

    ``row_count_estimate`` + occurrence count + op-set cardinality is a stable,
    cheap proxy for the plan's output shape without materializing it.
    """
    import hashlib
    import json as _json

    payload = {
        "row_count_estimate": int(rows),
        "occurrence_count": int(occurrence_count),
        "n_ops": len(ops),
    }
    raw = _json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _build_execution_certificate(
    plan: PlanNode,
    ops: tuple[str, ...],
    candidates: dict[str, float],
    chosen_backend: str,
    rows: int,
    occurrence_count: int,
    data_kind: str = "duckdb",
) -> Any:
    """R39-P1-PERF-082: build the compile-time ProductionExecutionCertificate.

    Additive: any failure returns ``None`` (no certificate → runtime skips the
    O(1) validation); routing behavior is unchanged.  Uses the plan's
    ``structural_key`` plus the bound-ops / backend-eligibility the router has
    already computed — no extra work beyond the existing compile-time walk.

    R40 #141: pass data_kind for requested_backend / resolved_dialect /
    datasource_identity so clickhouse_sql route is not silently mapped to duckdb.
    """
    try:
        from runtime.production_execution_certificate import (
            ProductionExecutionCertificate,
            normalize_backend,
        )
        from planner.plan_hash import structural_key

        structural_hash = structural_key(plan)
        bound_ops = frozenset(ops)
        eligible = {normalize_backend(b) for b in candidates} | {
            normalize_backend(chosen_backend)
        }
        # R40 #141: requested_backend / resolved_dialect / datasource_identity
        # so clickhouse_sql route is not silently mapped to duckdb.
        requested_backend = chosen_backend
        resolved_dialect = data_kind
        datasource_identity = data_kind
        return ProductionExecutionCertificate.build(
            structural_hash=structural_hash,
            bound_ops=bound_ops,
            backend_eligibility=eligible,
            output_shape_hash=_output_shape_hash(rows, occurrence_count, ops),
            requested_backend=requested_backend,
            resolved_dialect=resolved_dialect,
            datasource_identity=datasource_identity,
        )
    except Exception:
        # Fail-open additive: routing must never break because certificate
        # construction failed (e.g. bootstrap before registry load).
        return None


# =====================================================================
# R39-PERF-076：shape-aware TTDC predictor（DuckDB vs Polars vs PyArrow）
# =====================================================================

#: 本地 / 远程 parquet 扫描吞吐（MB/s）——TTDC 预测的线性模型系数。
_LOCAL_SCAN_MBPS = 800.0
_REMOTE_SCAN_MBPS = 60.0
_CONVERT_MBPS = 200.0  # 表示转换（arrow/pandas/polars）吞吐
#: 下游是 DuckDB fused factor 时，禁止「为了 size 选 Polars lazy」——给 polars
#: 候选加一个明确惩罚，保证 duckdb 候选存在时被选中。
_DUCKDB_FUSED_POLARS_PENALTY_MS = 10_000.0


@dataclass(frozen=True)
class TtdcEstimate:
    """shape-aware 单 backend 的 TTDC 估计（时间到 durable commit 的四个分量）。"""

    backend: str
    scan_ms: float = 0.0
    conversion_ms: float = 0.0
    execute_ms: float = 0.0
    materialize_ms: float = 0.0

    @property
    def total_ms(self) -> float:
        return self.scan_ms + self.conversion_ms + self.execute_ms + self.materialize_ms

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "scan_ms": round(self.scan_ms, 3),
            "conversion_ms": round(self.conversion_ms, 3),
            "execute_ms": round(self.execute_ms, 3),
            "materialize_ms": round(self.materialize_ms, 3),
            "total_ms": round(self.total_ms, 3),
        }


def _shape_scan_bytes(shape: Any) -> int:
    for attr in ("selected_bytes", "total_bytes"):
        v = getattr(shape, attr, None)
        if v:
            try:
                return max(0, int(v))
            except (TypeError, ValueError):
                pass
    rows = int(getattr(shape, "estimated_rows", 0) or 0)
    cols = int(
        getattr(shape, "projected_columns", 0) or getattr(shape, "total_columns", 0) or 1
    )
    if rows > 0:
        return rows * cols * 8
    return 0


def _shape_remote(shape: Any) -> bool:
    try:
        return bool(getattr(shape, "remote", False))
    except Exception:  # noqa: BLE001
        return False


def _execute_ms_for_backend(backend: str, rows: int) -> float:
    if rows <= 0:
        return 0.0
    rate = {
        "duckdb_sql": 5_000_000,
        "polars": 3_000_000,
        "polars_panel": 3_000_000,
        "polars_long": 3_000_000,
        "pyarrow": 8_000_000,
        "pandas_numpy": 1_000_000,
    }.get(backend, 2_000_000)
    return max(0.0, rows / rate * 1000.0)


def predict_ttdc(shape: Any, backend: str) -> TtdcEstimate:
    """shape-aware TTDC predictor（R39-PERF-076）。

    ``shape`` 是 ``data_access.read.scan_cost.ScanCost`` 或等价 duck-typed 对象
    （需 selected_bytes/total_bytes/estimated_rows/projected_columns/total_columns/
    remote）。返回 scan + conversion + execute + materialize 四个分量：
      - DuckDB：native parquet scan，无 scan→backend 转换；仅在最终物化回
        pandas/arrow 时计一次 materialize。
      - Polars：scan 后需 arrow→polars 转换，materialize 到最终表示另计。
      - PyArrow：native arrow scan，materialize 到 pandas/panel 另计。
    """
    scan_bytes = _shape_scan_bytes(shape)
    rows = int(getattr(shape, "estimated_rows", 0) or 0)
    remote = _shape_remote(shape)
    mbps = _REMOTE_SCAN_MBPS if remote else _LOCAL_SCAN_MBPS
    scan_ms = 0.0
    if scan_bytes > 0:
        scan_ms = (scan_bytes / 1024.0 / 1024.0) / mbps * 1000.0
    elif rows > 0:
        scan_ms = rows / 500_000.0 * 20.0  # 无字节信息时的粗粒度 fallback
    conversion_ms = 0.0
    materialize_ms = 0.0
    if backend == "duckdb_sql":
        conversion_ms = 0.0
        if scan_bytes > 0:
            materialize_ms = (scan_bytes / 1024.0 / 1024.0) / _CONVERT_MBPS * 1000.0
    elif backend in ("polars", "polars_panel", "polars_long"):
        if scan_bytes > 0:
            conversion_ms = (scan_bytes / 1024.0 / 1024.0) / _CONVERT_MBPS * 1000.0
            materialize_ms = conversion_ms * 0.5
    elif backend == "pyarrow":
        conversion_ms = 0.0
        if scan_bytes > 0:
            materialize_ms = (scan_bytes / 1024.0 / 1024.0) / _CONVERT_MBPS * 1000.0
    execute_ms = _execute_ms_for_backend(backend, rows)
    return TtdcEstimate(
        backend=backend,
        scan_ms=scan_ms,
        conversion_ms=conversion_ms,
        execute_ms=execute_ms,
        materialize_ms=materialize_ms,
    )


def choose_plan_route(plan: PlanNode, ctx: Any) -> PlanRoute:
    from backend.operator_capability import supports_pandas, supports_polars, supports_sql
    from backend.polars_long_production import is_polars_long_native_production_safe

    mode = str(getattr(ctx, "run_mode", "research") or "research").lower()
    production = mode == "production"
    # R31-013/014：每个 occurrence（绑定参数 window/feature_dim）单独估价。
    occurrences = plan_occurrences(plan)
    ops = _canonical_ops(plan)
    occ_by_canon: dict[str, list[BoundNodeOccurrence]] = {}
    for occ in occurrences:
        occ_by_canon.setdefault(occ.canonical, []).append(occ)
    rows = estimate_plan_rows(ctx)
    source_ref = _contains_source_ref(plan)
    # R33-P0-062/§11.3：SourceRef 已可 lower 为 DA source relation → 不毒死 native。
    source_lowered = source_ref and source_refs_lowerable(plan)
    data_kind = _data_source_kind(ctx)
    candidates: dict[str, float] = {}
    mem_budget = _execution_memory_budget(ctx)
    # R13 P1-65: ops whose only "polars" slot is a pandas-delegating UDF must be
    # costed with the delegate penalty (never preferred over native pandas).
    delegate_ops = _polars_delegate_ops(tuple(occ_by_canon.keys()))

    def _within_budget(backend: str, rows_: int) -> bool:
        if mem_budget is None:
            return True
        est_peak = estimate_plan_peak_memory(ops, rows_, backend)
        return est_peak <= mem_budget

    def _plan_op_cost(backend: str) -> float:
        total = 0.0
        for occ in occurrences:
            total += _cost(
                occ.canonical,
                backend,
                rows,
                delegate_polars=occ.canonical in delegate_ops,
                occ=occ,
            )
        return total

    pandas_ok = all(supports_pandas(op, mode=mode) for op in ops)
    if pandas_ok and _within_budget("pandas_numpy", rows):
        # R31-P0-016：conversion 只计一次（Pandas 零转换）。
        candidates["pandas_numpy"] = _plan_op_cost("pandas_numpy") + _one_conversion_penalty(
            "pandas_numpy", rows
        )

    # Wide Polars is valid even when an operator is not Polars-long-native. The
    # plan remains on one wide representation, so conversion is paid once below.
    polars_panel_ok = bool(ops) and (not source_ref or source_lowered) and all(
        supports_polars(op, mode=mode) for op in ops
    )
    if polars_panel_ok and _within_budget("polars_panel", rows):
        cost = _plan_op_cost("polars_panel") + _one_conversion_penalty("polars_panel", rows)
        candidates["polars_panel"] = cost

    polars_long_ok = bool(ops) and (not source_ref or source_lowered) and all(
        is_polars_long_native_production_safe(op) if production else supports_polars(op, mode=mode)
        for op in ops
    )
    if polars_long_ok and _within_budget("polars_long", rows):
        candidates["polars_long"] = (
            _plan_op_cost("polars_long") + _one_conversion_penalty("polars_long", rows)
        )

    sql_ok = bool(ops) and (not source_ref or source_lowered) and data_kind in {"duckdb", "clickhouse"} and all(
        supports_sql(op, data_source_kind=data_kind, mode=mode) for op in ops
    )
    sql_backend = "clickhouse_sql" if data_kind == "clickhouse" else "duckdb_sql"
    if sql_ok and _within_budget(sql_backend, rows):
        candidates[sql_backend] = (
            _plan_op_cost(sql_backend) + _one_conversion_penalty(sql_backend, rows)
        )

    if data_kind in {"duckdb", "clickhouse"}:
        # MB-P0-001: Pass source_lowered explicitly to fix NameError
        mixed = _dag_aware_mixed_cost(
            occurrences,
            rows=rows,
            delegate_ops=delegate_ops,
            data_kind=data_kind,
            mode=mode,
            source_ref=source_ref,
            source_lowered=source_lowered,
        )
        if mixed is not None and _within_budget("pandas_numpy", rows):
            candidates["hybrid"] = mixed

    if not candidates:
        from backend.operator_capability import UnsupportedOperatorBackendError
        raise UnsupportedOperatorBackendError(
            "no eligible physical plan for canonicals: " + ", ".join(ops)
        )

    # R39-PERF-076：shape-aware TTDC 调整。只有 shape 数据可用（ctx.scan_shape /
    # ctx.scan_cost）才生效；否则完全走既有 cost model。
    shape = getattr(ctx, "scan_shape", None) or getattr(ctx, "scan_cost", None)
    downstream_duckdb_fused = bool(getattr(ctx, "downstream_duckdb_fused", False))
    if shape is not None:
        for backend in list(candidates):
            est = predict_ttdc(shape, backend)
            candidates[backend] = float(candidates[backend]) + est.total_ms
        # 规则：下游是 DuckDB fused factor 时，不为「size」选 Polars lazy——
        # duckdb 候选存在时给 polars 候选加明确惩罚。
        if downstream_duckdb_fused and "duckdb_sql" in candidates:
            for backend in ("polars_panel", "polars_long", "pyarrow"):
                if backend in candidates:
                    candidates[backend] = (
                        float(candidates[backend]) + _DUCKDB_FUSED_POLARS_PENALTY_MS
                    )

    chosen = min(candidates.items(), key=lambda item: (item[1], item[0]))
    basis = (
        "measured"
        if chosen[0] != "hybrid" and _candidate_is_measured(ops, chosen[0])
        else "estimated"
    )
    # R39-P1-PERF-082: compile-time execution certificate (O(1) runtime check).
    certificate = _build_execution_certificate(
        plan,
        ops,
        candidates,
        chosen_backend=chosen[0],
        rows=rows,
        occurrence_count=len(occurrences),
    )
    return PlanRoute(
        backend=chosen[0],
        estimated_cost=float(chosen[1]),
        routing_basis=basis,
        candidate_costs=tuple(sorted((name, float(cost)) for name, cost in candidates.items())),
        row_count_estimate=rows,
        ops=ops,
        occurrence_count=len(occurrences),
        reason=(
            "offline measured baseline" if basis == "measured"
            else "certified capability + workload estimate; no compatible measured baseline"
        ),
        certificate=certificate,
    )


def record_plan_route(ctx: Any, route: PlanRoute) -> None:
    runtime = dict(getattr(ctx, "runtime_stats", None) or {})
    runtime["plan_backend_route"] = {
        "backend": route.backend,
        "routing_basis": route.routing_basis,
        "estimated_cost": route.estimated_cost,
        "candidate_costs": dict(route.candidate_costs),
        "row_count_estimate": route.row_count_estimate,
        "ops": list(route.ops),
        "reason": route.reason,
    }
    # R39-P1-PERF-082: record the certificate hash for the O(1) runtime check.
    if getattr(route, "certificate", None) is not None:
        cert = route.certificate
        runtime["production_execution_certificate"] = {
            "certificate_hash": cert.certificate_hash,
            "structural_hash": cert.structural_hash,
            "output_shape_hash": cert.output_shape_hash,
            "backend_eligibility": sorted(cert.backend_eligibility),
            "bound_ops": sorted(cert.bound_ops),
        }
    ctx.runtime_stats = runtime  # type: ignore[attr-defined]


# =====================================================================
# R33-P0-061..067 + §31：batch-global route + maximal native subgraph
# =====================================================================

#: R33-P0-064：conversion penalty 不再写死常数——按 edge 类型 + 字节建模。
#: 仅作为无 measured baseline 时的 seed fallback（§31.5）。
_CONVERSION_SEED_MS = {
    "duckdb_to_arrow": 3.0,
    "arrow_to_polars": 2.0,
    "arrow_to_pandas": 2.0,
    "polars_to_pandas": 2.0,
    "pandas_to_polars": 2.5,
    "wide_to_long": 2.0,
    "long_to_wide": 2.5,
    "sort": 1.5,
    "repartition": 2.0,
    "dtype_cast": 0.5,
}

#: MB-P1-007: Bytes-based coefficients for different edge types (ms per MB)
_CONVERSION_BYTES_COEFF = {
    "duckdb_to_arrow": 0.05,
    "arrow_to_polars": 0.03,
    "arrow_to_pandas": 0.08,
    "polars_to_pandas": 0.10,
    "pandas_to_polars": 0.12,
    "wide_to_long": 0.15,
    "long_to_wide": 0.18,
    "sort": 0.20,
    "repartition": 0.10,
    "dtype_cast": 0.02,
}


def edge_conversion_penalty_ms(
    edge: str,
    bytes_: int = 0,
    requires_sort: bool = False,
    requires_repartition: bool = False,
    requires_reshape: bool = False,
) -> float:
    """MB-P1-007: conversion 按 edge 类型 + 字节 + 额外操作精细建模。

    只读 edge 真实字节时 ``bytes_`` 参与；无字节信息用固定 overhead。真实
    measured baseline 覆盖时（``_measured_baseline`` 兼容）用实测值。

    新增：
      - requires_sort: 需要排序（额外成本）
      - requires_repartition: 需要重新分区（额外成本）
      - requires_reshape: 需要 wide<->long 转换（额外成本）
    """
    base = _CONVERSION_SEED_MS.get(edge, 3.0)
    bytes_coeff = _CONVERSION_BYTES_COEFF.get(edge, 0.10)

    total = base

    if bytes_ > 0:
        mb = bytes_ / 1_000_000.0
        total += bytes_coeff * mb

    # MB-P1-007: Additional costs for complex edge operations
    if requires_sort:
        total += _CONVERSION_SEED_MS.get("sort", 1.5)
        if bytes_ > 0:
            total += _CONVERSION_BYTES_COEFF.get("sort", 0.20) * (bytes_ / 1_000_000.0)

    if requires_repartition:
        total += _CONVERSION_SEED_MS.get("repartition", 2.0)
        if bytes_ > 0:
            total += _CONVERSION_BYTES_COEFF.get("repartition", 0.10) * (bytes_ / 1_000_000.0)

    if requires_reshape:
        reshape_cost = max(
            _CONVERSION_SEED_MS.get("wide_to_long", 2.0),
            _CONVERSION_SEED_MS.get("long_to_wide", 2.5),
        )
        total += reshape_cost
        if bytes_ > 0:
            reshape_coeff = max(
                _CONVERSION_BYTES_COEFF.get("wide_to_long", 0.15),
                _CONVERSION_BYTES_COEFF.get("long_to_wide", 0.18),
            )
            total += reshape_coeff * (bytes_ / 1_000_000.0)

    return total


def plan_native_subgraph_fraction(
    plan: PlanNode,
    ctx: Any,
    *,
    backend: str = "duckdb_sql",
    mode: str | None = None,
) -> dict[str, Any]:
    """R33 §11.2/§75 Step 3：maximal native subgraph 覆盖评估。

    返回 ``{native_ops, unsupported_ops, native_fraction, tail_fraction}``：
       - ``native_fraction``：SQL-native occurrence 占比（可下推的部分）。
       - ``unsupported_ops``：不支持该 backend 的 occurrence（= 需经一次
         conversion boundary 交给 specialized kernel / pandas reference）。
    用途：一个 unsupported op **不**再整 root 失去 native candidate
    （R33_SINGLE_UNSUPPORTED_OP_FULL_PANDAS_FALLBACK_ZERO）。
    """
    from backend.operator_capability import supports_sql

    mode = mode or str(getattr(ctx, "run_mode", "research") or "research").lower()
    data_kind = _data_source_kind(ctx)
    occs = plan_occurrences(plan)
    native: list[str] = []
    unsupported: list[str] = []
    for occ in occs:
        if supports_sql(occ.canonical, data_source_kind=data_kind, mode=mode):
            native.append(occ.canonical)
        else:
            unsupported.append(occ.canonical)
    total = max(1, len(occs))
    return {
        "native_ops": list(dict.fromkeys(native)),
        "unsupported_ops": list(dict.fromkeys(unsupported)),
        "native_fraction": round(len(native) / total, 4),
        "n_occurrences": len(occs),
    }


@dataclass(frozen=True)
class BatchPhysicalRoute:
    """R33-P0-061/§31：one BatchPhysicalRoute with multiple backend regions。

    聚合整批因子的 backend 路由 + 共享收益 + 全链路成本（
    source + operator + conversion + scheduler overhead + DQ + generation
    commit），最终目标是 ``time_to_durable_commit``（§31.1）。
    """

    total_time_to_durable_commit_ms: float
    per_root: tuple[tuple[str, str, float], ...]  # (factor_name, backend, cost_ms)
    shared_benefit_ms: float = 0.0
    native_fraction: float = 0.0
    scan_bytes: int = 0
    conversion_bytes: int = 0
    scheduler_overhead_ms: float = 0.0
    dq_ms: float = 0.0
    write_ms: float = 0.0
    generation_commit_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_time_to_durable_commit_ms": round(self.total_time_to_durable_commit_ms, 3),
            "shared_benefit_ms": round(self.shared_benefit_ms, 3),
            "native_fraction": round(self.native_fraction, 4),
            "scan_bytes": self.scan_bytes,
            "conversion_bytes": self.conversion_bytes,
            "scheduler_overhead_ms": round(self.scheduler_overhead_ms, 3),
            "dq_ms": round(self.dq_ms, 3),
            "write_ms": round(self.write_ms, 3),
            "generation_commit_ms": round(self.generation_commit_ms, 3),
            "per_root": [
                {"factor": f, "backend": b, "cost_ms": round(c, 3)}
                for f, b, c in self.per_root
            ],
        }


def plan_batch_route(
    plans: dict[str, PlanNode],
    ctx: Any,
    *,
    scan_cost_map: dict[str, Any] | None = None,
    shared_roots: int = 0,
    factor_count: int | None = None,
) -> BatchPhysicalRoute:
    """R33-P0-061/§31：batch-global physical route。

    每个 root 单独 ``choose_plan_route``；整批聚合：
       - native_fraction：各 root native subgraph 覆盖加权平均（§75 Step 2/3）。
       - shared_benefit：>1 root 共享同 source scope 时，scan/join 节省估入。
       - scheduler_overhead：任务数 × 每任务控制面成本（§31.1）。
       - DQ / write / generation commit：按 batch 规模估计（§31.1）。
    最终成本单位是 **time-to-durable-commit 毫秒**（不是 operator 数）。
    """
    per_root: list[tuple[str, str, float]] = []
    scan_bytes = 0
    conv_bytes = 0
    native_fracs: list[float] = []
    total = 0.0
    n_roots = len(plans)
    factor_count = factor_count or n_roots
    for name, plan in plans.items():
        route = choose_plan_route(plan, ctx)
        cost_ms = route.estimated_cost
        per_root.append((name, route.backend, cost_ms))
        total += cost_ms
        sub = plan_native_subgraph_fraction(plan, ctx, backend=route.backend)
        native_fracs.append(sub["native_fraction"])
        if scan_cost_map is not None:
            cost = scan_cost_map.get(name) or (
                next(iter(scan_cost_map.values()), None)
            )
            scan_bytes += int(getattr(cost, "selected_bytes", 0) or 0)
            conv_bytes += int(getattr(cost, "projection_bytes", 0) or 0)
    native_fraction = (
        sum(native_fracs) / len(native_fracs) if native_fracs else 0.0
    )
    # §31.2 Shared benefit：>1 root 共享同 source scope → 每多一个 consumer 省一次 scan。
    shared_benefit = 0.0
    if shared_roots > 0 and n_roots > 1:
        shared_benefit = min(total * 0.10, shared_roots * 5.0)
        total = max(0.0, total - shared_benefit)
    # §31.1 scheduler overhead：real task × 控制面成本。
    task_count = max(1, n_roots * 2 + shared_roots)
    scheduler_overhead = task_count * 0.15
    dq = n_roots * 0.5
    write = max(1, n_roots) * 0.8
    gen_commit = 2.0 + max(0.0, factor_count / 1000.0) * 5.0
    total += scheduler_overhead + dq + write + gen_commit
    return BatchPhysicalRoute(
        total_time_to_durable_commit_ms=total,
        per_root=tuple(per_root),
        shared_benefit_ms=shared_benefit,
        native_fraction=native_fraction,
        scan_bytes=scan_bytes,
        conversion_bytes=conv_bytes,
        scheduler_overhead_ms=scheduler_overhead,
        dq_ms=dq,
        write_ms=write,
        generation_commit_ms=gen_commit,
    )


def record_batch_route(ctx: Any, route: BatchPhysicalRoute) -> None:
    runtime = dict(getattr(ctx, "runtime_stats", None) or {})
    runtime["batch_physical_route"] = route.to_dict()
    runtime["backend_policy"] = (
        "duckdb-first-but-cost-driven: stay-in-engine tie-break, "
        "no static duckdb>polars>pandas rank"
    )
    ctx.runtime_stats = runtime  # type: ignore[attr-defined]
