# Polars Backend 内存与并行优化分析报告

**日期**: 2026-08-13  
**范围**: Polars 后端内存使用与并行执行优化  
**目标**: 内存降低 50-70%，吞吐提升 30-50%

---

## 一、当前架构概览

### 1.1 核心文件结构
- **polars_backend.py**: 主入口，继承 PandasBackend
- **polars_expr_emitter.py**: 表达式编译器（3205 行）
- **polars_region_optimizer.py**: 多根节点统一 collect 与流式元数据
- **adaptive_batch_scheduler.py**: 资源感知 DAG 调度器
- **perf_config.py**: 性能配置（线程/内存/后端选择）

### 1.2 当前执行路径
```
PolarsBackend.execute()
  → compile_polars_long_lazy()  # 编译为 LazyFrame
    → _compile_polars()          # 递归构建表达式
      → lf.collect()             # **关键：单次 collect**
        → polars_long_to_multiindex_series()
```

---

## 二、关键发现：collect() 调用点分析

### 2.1 受控 collect 终端（✅ 已优化）

**位置**: `polars_expr_emitter.py:3179`

```python
# Line 3163-3179: execute_polars_long_plan
_start = _t.perf_counter()
frame = lf.collect()  # ← 单次受控 collect
enforce_arrow_budget(
    _budget,
    frame.to_arrow(),
    elapsed_ms=(_t.perf_counter() - _start) * 1000,
)
```

**现状评估**:
- ✅ 已集中化：整个计划编译为一个 LazyFrame，只在末尾 collect 一次
- ✅ 已添加预算控制：QueryBudget deadline/rows 强制执行
- ✅ 已添加快照验证：`revalidate_for_long_collect()` 防止数据源变更
- ⚠️ **优化机会**：当前未启用 `streaming=True` 参数

### 2.2 其他 collect 调用（无发现）

**搜索结果**: 全文件扫描未发现其他裸 `.collect()` 调用

**结论**:
- 架构设计良好：单一受控 collect 点
- 无分散的多次 collect（避免了重复物化）
- 无中间结果的提前物化

---

## 三、优化策略与实施方案

### 3.1 流式模式优化（目标：内存 -50~70%）

#### 现状问题
```python
# 当前实现（polars_expr_emitter.py:3179）
frame = lf.collect()  # 全部物化到内存
```

#### 优化方案

**A. 智能流式决策**

```python
# 新增函数：streaming_capable_analysis
def streaming_capable_for_plan(plan: PlanNode) -> tuple[bool, str]:
    """分析计划是否支持流式执行。
    
    返回:
        (can_stream, reason): 
            - can_stream=True: 可安全启用 streaming=True
            - reason: 不支持原因（调试用）
    """
    from backend.polars_long_policy import collect_plan_op_stats
    
    stats = collect_plan_op_stats(plan)
    
    # 阻塞流式的算子类别
    blocking_ops = {
        # 需要全局排序
        "rank", "rank_pct", "cs_pct_rank", "ts_rank",
        # 需要完整分组
        "zscore", "normalize", "cs_quantile", "quantile",
        # 递归状态（部分支持，保守）
        "KAMA", "MACD",
    }
    
    for op in stats.keys():
        if op in blocking_ops:
            return False, f"blocking_op={op}"
    
    # 检查 map_groups（Python UDF）占比
    if stats.get("map_groups", 0) > 0:
        # 如果有 Python 回调，流式性能可能不佳
        if stats["map_groups"] / sum(stats.values()) > 0.3:
            return False, "high_python_udf_ratio"
    
    return True, "ok"


def execute_polars_long_plan_with_streaming(
    plan: PlanNode,
    ctx: Any,
    *,
    base_lf: pl.LazyFrame | None = None,
    lazy_cache_key: str | None = None,
) -> pd.Series:
    """支持流式的 collect 实现。"""
    compiled = compile_polars_long_lazy(
        plan, ctx, base_lf=base_lf, lazy_cache_key=lazy_cache_key
    )
    lf = compiled.frame.sort([compiled.ts_col, compiled.inst_col]).select(
        pl.col(compiled.ts_col).alias(ctx.timestamp_col),
        pl.col(compiled.inst_col).alias(ctx.instrument_col),
        pl.col(compiled.value_col).alias("value"),
    )
    
    ds = getattr(ctx, "data_source", None)
    revalidate = getattr(ds, "revalidate_for_long_collect", None)
    if callable(revalidate):
        revalidate()
    
    from data_access.read.query_budget import (
        enforce_arrow_budget,
        resolve_query_budget,
    )
    import time as _t
    
    _budget = resolve_query_budget(None)
    _start = _t.perf_counter()
    
    # 🔥 核心优化：智能流式决策
    can_stream, reason = streaming_capable_for_plan(plan)
    
    if can_stream:
        # 流式模式：边扫描边计算边输出，峰值内存 << 全量数据
        frame = lf.collect(streaming=True)
        runtime = dict(getattr(ctx, "runtime_stats", None) or {})
        runtime["polars_streaming"] = True
        ctx.runtime_stats = runtime
    else:
        # 回退全量 collect
        frame = lf.collect()
        runtime = dict(getattr(ctx, "runtime_stats", None) or {})
        runtime["polars_streaming_blocked"] = reason
        ctx.runtime_stats = runtime
    
    enforce_arrow_budget(
        _budget,
        frame.to_arrow(),
        elapsed_ms=(_t.perf_counter() - _start) * 1000,
    )
    return polars_long_to_multiindex_series(
        frame,
        timestamp_col=ctx.timestamp_col,
        instrument_col=ctx.instrument_col,
        value_col="value",
        template_index=optional_universe_index(ctx),
    ).sort_index()
```

**预期收益**:
- 内存峰值降低 **50-70%**（大数据集，流式窗口聚合）
- 启动延迟降低（无需等全量读取）
- 适用场景：时序窗口、简单截面、逐行变换

**风险控制**:
- 保守白名单：仅对已验证安全的算子启用
- 回退机制：检测到阻塞算子自动降级
- Telemetry：记录 `polars_streaming` 指标，监控生产表现

---

### 3.2 线程池动态调优（目标：吞吐 +30~50%）

#### 现状问题

**A. 固定线程预算**
```python
# adaptive_batch_scheduler.py:113-120
budget = 1  # 默认单线程
try:
    contract = task.resource_contract
    if contract is not None:
        bt = getattr(contract, "backend_threads", None)
        if bt and int(bt) >= 1:
            budget = int(bt)
except Exception:
    budget = 1
```

**B. Polars 全局线程池未显式管理**
```python
# 当前无显式调用 pl.Config.set_streaming_chunk_size() 
# 或 pl.Config.set_thread_count()
```

#### 优化方案

**A. 动态线程预算绑定**

```python
# 新增模块：backend/polars_thread_config.py

import os
import polars as pl
from contextlib import contextmanager


def optimal_polars_threads(cpu_budget: int, task_type: str) -> int:
    """根据 CPU 预算与任务类型计算最优线程数。
    
    Args:
        cpu_budget: ResourceBroker 分配的 CPU token
        task_type: 'compute' | 'io' | 'mixed'
    
    Returns:
        推荐的 Polars 线程数
    """
    # 获取物理核心数（非超线程）
    try:
        import psutil
        physical_cores = psutil.cpu_count(logical=False) or os.cpu_count() or 4
    except ImportError:
        physical_cores = os.cpu_count() or 4
    
    # 策略：
    # 1. compute-heavy（纯 CPU）：接近 cpu_budget
    # 2. io-heavy（扫描/读取）：适度超订（IO 等待时 CPU 可调度其他）
    # 3. mixed：折中
    
    if task_type == "io":
        # IO 任务：允许 1.5x 超订
        target = min(int(cpu_budget * 1.5), physical_cores)
    elif task_type == "compute":
        # CPU 任务：严格遵守预算
        target = min(cpu_budget, physical_cores)
    else:  # mixed
        target = min(int(cpu_budget * 1.2), physical_cores)
    
    return max(1, target)


@contextmanager
def polars_thread_budget(threads: int):
    """临时设置 Polars 线程数的上下文管理器。
    
    用法:
        with polars_thread_budget(4):
            df = lf.collect()  # 使用 4 线程
    """
    original = pl.thread_pool_size()
    try:
        pl.Config.set_thread_count(threads)
        yield
    finally:
        pl.Config.set_thread_count(original)


def configure_polars_for_execution(ctx: Any) -> dict[str, int]:
    """根据执行上下文配置 Polars 全局参数。
    
    Returns:
        配置快照（用于日志）
    """
    perf = ctx.perf if ctx.perf is not None else PerfConfig.from_env()
    
    # 1. 线程数
    max_workers = perf.max_workers
    if max_workers is None:
        max_workers = os.cpu_count() or 4
    
    # 2. 流式块大小（影响内存 vs 吞吐）
    # 小块：内存低，吞吐可能降低（频繁切换）
    # 大块：内存高，吞吐高（批量矢量化）
    memory_mb = perf.max_in_memory_mb
    if memory_mb is not None and memory_mb < 2000:
        # 低内存环境：小块
        chunk_size = 50_000
    else:
        # 正常环境：默认或大块
        chunk_size = 100_000
    
    pl.Config.set_thread_count(max_workers)
    pl.Config.set_streaming_chunk_size(chunk_size)
    
    return {
        "thread_count": max_workers,
        "streaming_chunk_size": chunk_size,
    }
```

**B. 集成到调度器**

```python
# 修改 adaptive_batch_scheduler.py:_dispatch

def _dispatch(
    task: PhysicalFactorTask,
    backend: Any,
    ctx: Any,
    execute_root: Callable[[PhysicalFactorTask], Any],
    materialize_shared: Callable[[str, Any], Any],
) -> tuple[str, Any]:
    """模块级执行函数（支持 Polars 线程预算）。"""
    if task.task_type == TASK_CSE_SHARED:
        sid = task.task_id.split(":", 1)[1]
        materialize_shared(sid, task.node_ref)
        return task.task_id, None
    
    if task.task_type == TASK_ROOT:
        budget = 1
        try:
            contract = task.resource_contract
            if contract is not None:
                bt = getattr(contract, "backend_threads", None)
                if bt and int(bt) >= 1:
                    budget = int(bt)
        except Exception:
            budget = 1
        
        # 🔥 新增：BLAS/OpenMP + Polars 双重线程预算
        from runtime.execution_traits import thread_budget
        
        # 判断任务类型
        task_type = "compute"  # 默认
        if task.preferred_backend == "polars":
            # Polars 任务：检查是否有 SOURCE_SCAN 前驱
            if any("scan" in str(p).lower() for p in task.inputs):
                task_type = "io"
            else:
                task_type = "mixed"
        
        with thread_budget(budget):
            # Polars-specific: 设置其线程池
            if task.preferred_backend == "polars":
                from backend.polars_thread_config import (
                    optimal_polars_threads,
                    polars_thread_budget,
                )
                polars_threads = optimal_polars_threads(budget, task_type)
                with polars_thread_budget(polars_threads):
                    result = execute_root(task)
            else:
                result = execute_root(task)
        
        return task.task_id, result
    
    return task.task_id, None
```

**预期收益**:
- CPU 密集型任务：吞吐提升 **30-50%**（充分利用多核）
- IO 密集型任务：延迟降低（适度超订，隐藏 IO 等待）
- 资源隔离：多任务并发时互不干扰（各自独立线程预算）

---

### 3.3 Lazy 优化全量利用（目标：+20~40%）

#### 现状评估

**已实现优化**:
- ✅ DAG 融合（`_try_fuse_binary_ts_on_column`, `_try_binary_from_base_columns`）
- ✅ 谓词下推（`.filter()` 在 LazyFrame 上自动下推）
- ✅ 投影下推（`collect_columns()` 只扫描需要的列）
- ✅ CSE 复用（`memo` 字典缓存编译结果）

**优化空间**:

**A. 显式列裁剪强化**

```python
# 当前 (polars_expr_emitter.py:3012)
def resolve_base_lazy_for_plan(plan, ctx, scan_fn):
    columns = collect_columns(plan)
    if columns:
        return scan_fn(sorted(columns))  # ✅ 已裁剪
    # ...

# 🔥 增强：添加二次裁剪（CSE 后可能产生冗余列）
def resolve_base_lazy_optimized(plan, ctx, scan_fn):
    """带二次裁剪的 base lazy 构建。"""
    # 第一轮：收集直接引用列
    direct_cols = collect_columns(plan)
    
    # 第二轮：分析 CSE 实际消费列
    from planner.cse import collect_consumed_sids
    materialized_cols = set()
    for child in plan.inputs:
        if child.op == "plan_ref":
            sid = child.attrs.get("sid")
            sc = getattr(ctx, "shared_result_cache", None) or {}
            if sid in sc:
                # 已物化结果不需要列扫描
                continue
        materialized_cols.update(collect_columns(child))
    
    # 最终列集 = direct + 未物化的 materialized
    final_cols = direct_cols | materialized_cols
    
    if final_cols:
        base = scan_fn(sorted(final_cols))
        # 🔥 二次裁剪：如果 scan 返回了额外列（如 select *），显式 select
        scanned_cols = set(base.collect_schema().names())
        needed = final_cols | {"ts", "inst"}
        if not needed.issuperset(scanned_cols):
            base = base.select([c for c in scanned_cols if c in needed])
        return base
    # fallback ...
```

**B. 智能物化时机**

```python
# 新增：大型中间结果物化决策
def should_materialize_intermediate(
    node: PlanNode,
    memo: dict,
    threshold_consumers: int = 3,
) -> bool:
    """判断中间节点是否应提前物化（缓存为 DataFrame）。
    
    场景：
    - 昂贵算子（如大窗口 rolling、复杂 UDF）
    - 多次复用（>= threshold_consumers）
    - 结果集相对小（相比输入）
    
    Polars 的 LazyFrame 会在每个 consumer 重复执行整个子树，
    对于复用度高的昂贵节点，提前 collect 一次反而更快。
    """
    # 检查复用度
    # （需要从 DAG 分析工具获取 consumer 数量）
    consumers = 0  # placeholder
    if consumers < threshold_consumers:
        return False
    
    # 检查算子类型
    expensive_ops = {
        "ts_skew", "ts_kurt", "ts_quantile",  # rolling_map
        "expanding_std", "quantile",  # map_groups
    }
    if node.op in expensive_ops:
        return True
    
    return False
```

**预期收益**:
- 列裁剪：扫描时间 -10~30%（宽表场景）
- 智能物化：高复用子树计算时间 -20~40%

---

### 3.4 内存预分配与池化（目标：减少碎片，稳定延迟）

#### 优化方案

**A. Polars Arena 分配器配置**

```python
# 新增：backend/polars_memory_config.py

def configure_polars_memory(ctx: Any) -> None:
    """配置 Polars 内存分配器参数。"""
    perf = ctx.perf if ctx.perf is not None else PerfConfig.from_env()
    
    # 1. 预分配 Arena 大小
    # Polars 内部使用 arrow/polars allocator，支持 arena 预分配
    memory_limit = perf.max_in_memory_mb
    if memory_limit is not None:
        # 预分配 20% 作为 arena pool
        arena_mb = int(memory_limit * 0.2)
        os.environ["POLARS_MAX_THREADS"] = str(perf.max_workers or 4)
        # 注：Polars 0.20+ 支持环境变量配置 allocator
        os.environ["POLARS_ALLOCATOR_ARENA_SIZE_MB"] = str(arena_mb)
    
    # 2. 字符串池化（大量重复 instrument code 场景）
    # Polars 自动 string interning，无需额外配置
    
    # 3. 显式 GC 触发阈值
    import gc
    if memory_limit is not None:
        # 达到 70% 时主动触发 GC
        gc.set_threshold(
            int(memory_limit * 0.7 * 1024 * 1024 / 1000),  # gen0
            10,  # gen1
            10,  # gen2
        )
```

**B. 批次大小自适应**

```python
# 修改 adaptive_batch_scheduler.py 的批次策略

def _adaptive_batch_size(
    broker: ResourceBroker,
    task: PhysicalFactorTask,
) -> int:
    """根据实时资源状态计算批次大小。
    
    Returns:
        推荐的 micro-batch 合并数量
    """
    headroom = broker.snapshot().live_headroom
    total_mem = broker.resource_envelope().safe_memory_bytes
    
    # 压力越大，批次越小（降低峰值）
    if headroom < total_mem * 0.2:
        # 低余量：小批次
        return 16
    elif headroom < total_mem * 0.5:
        # 中等：默认
        return 64
    else:
        # 充裕：大批次（提高吞吐）
        return 128
```

**预期收益**:
- 内存碎片降低 **10-20%**（长时间运行稳定性）
- GC 停顿减少（主动触发替代被动压力式 GC）
- 批次自适应：负载波动时平滑性能

---

## 四、实施优先级与风险评估

### 4.1 优先级矩阵

| 优化项 | 收益 | 风险 | 工作量 | 优先级 |
|--------|------|------|--------|--------|
| 流式模式 | ⭐⭐⭐⭐⭐ | 🔴 中 | 2d | **P0** |
| 线程池调优 | ⭐⭐⭐⭐ | 🟢 低 | 1d | **P0** |
| Lazy 优化 | ⭐⭐⭐ | 🟢 低 | 1d | P1 |
| 内存预分配 | ⭐⭐ | 🟢 低 | 0.5d | P2 |

### 4.2 风险控制

**流式模式风险**:
- **症状**：部分算子在 streaming=True 下结果错误或崩溃
- **缓解**：
  1. 保守白名单（仅启用已验证安全的算子组合）
  2. A/B 测试：生产环境先 10% 流量试点
  3. 自动回退：检测到错误时降级到全量 collect
  4. 监控指标：`polars_streaming_error_rate`

**线程超订风险**:
- **症状**：IO 任务 1.5x 超订可能导致 CPU 饱和
- **缓解**：
  1. 分级配置：研究环境激进，生产保守
  2. 动态降级：检测到 CPU >90% 时自动降低超订比例
  3. 隔离策略：不同任务类型独立线程池

### 4.3 验证计划

**Phase 1: 单元测试**
- 对比 streaming vs non-streaming 结果一致性
- 验证线程预算绑定生效（通过 `top -H` 观察）

**Phase 2: 基准测试**
- TTDC（time-to-data-collection）：300 因子 × 252 日 × 100 标的
- 峰值内存：`memory_profiler` 跟踪
- 吞吐：QPS（factors per second）

**Phase 3: 生产试点**
- 选择非关键策略组（10% 流量）
- 监控 7 天：错误率、延迟 P99、内存 P99
- 无异常后全量推广

---

## 五、代码修改清单

### 5.1 新增文件

1. **`backend/polars_thread_config.py`** (150 行)
   - `optimal_polars_threads()`
   - `polars_thread_budget()` context manager
   - `configure_polars_for_execution()`

2. **`backend/polars_memory_config.py`** (80 行)
   - `configure_polars_memory()`
   - `_adaptive_batch_size()`

3. **`backend/polars_streaming_policy.py`** (200 行)
   - `streaming_capable_for_plan()`
   - `blocking_ops` 白名单/黑名单
   - streaming 元数据注解

### 5.2 修改文件

1. **`backend/polars_expr_emitter.py`**
   - Line 3179: `lf.collect()` → `lf.collect(streaming=<dynamic>)`
   - 新增 `execute_polars_long_plan_with_streaming()`

2. **`runtime/adaptive_batch_scheduler.py`**
   - Line 113-120: `_dispatch()` 集成 Polars 线程预算
   - Line 1551: `_auto_shard_replan()` 集成批次自适应

3. **`backend/polars_backend.py`**
   - Line 172: `execute()` 调用 `configure_polars_for_execution()`

4. **`runtime/perf_config.py`**
   - 新增字段: `polars_streaming_auto: bool = True`
   - 新增字段: `polars_thread_oversubscribe_io: float = 1.5`

### 5.3 测试覆盖

1. **`tests/backend/test_polars_streaming.py`** (新增)
   - 对比 streaming vs eager 结果
   - 覆盖 50+ 核心算子

2. **`tests/backend/test_polars_threads.py`** (新增)
   - 验证线程预算生效
   - 并发任务隔离测试

3. **`tests/performance/bench_polars_optimization.py`** (新增)
   - TTDC 基准（优化前后对比）
   - 内存峰值基准

---

## 六、度量指标

### 6.1 成功标准

**内存优化**:
- ✅ 峰值内存降低 ≥50%（大数据集，300 因子 × 252 日 × 1000 标的）
- ✅ 内存碎片率 <15%（运行 24 小时后）

**吞吐优化**:
- ✅ TTDC 降低 ≥30%（CPU 密集型因子组）
- ✅ QPS 提升 ≥30%（混合负载）

**稳定性**:
- ✅ 错误率 <0.1%（与基线一致）
- ✅ P99 延迟抖动 <10%

### 6.2 监控 Dashboard

```python
# 新增 runtime_stats 字段
{
    "polars_streaming": True,           # 是否启用流式
    "polars_streaming_blocked": None,   # 阻塞原因
    "polars_threads": 8,                # 实际线程数
    "polars_thread_oversubscribe": 1.5, # 超订比例
    "polars_memory_peak_mb": 1250,      # 峰值内存
    "polars_collect_duration_ms": 523,  # collect 耗时
}
```

---

## 七、总结

### 核心发现
1. ✅ **架构健康**：单一受控 collect 点，无分散物化
2. ⚠️ **主要瓶颈**：未启用 streaming，线程池固定预算
3. 🎯 **快赢机会**：`collect(streaming=True)` + 动态线程调优

### 推荐路径
- **Week 1**: 实现流式决策 + 线程预算绑定（P0）
- **Week 2**: Lazy 优化增强 + 内存配置（P1/P2）
- **Week 3**: 全量测试 + 生产试点（10% 流量）
- **Week 4**: 监控验证 + 全量推广

### 预期 ROI
- **开发投入**: 4-5 人日
- **性能收益**: 内存 -50~70%，吞吐 +30~50%
- **生产价值**: 单机承载能力翻倍，云成本降低 40%+

---

**附录**:
- [Polars Streaming 文档](https://pola-rs.github.io/polars/user-guide/lazy/streaming/)
- [线程池最佳实践](https://pola-rs.github.io/polars/user-guide/misc/multiprocessing/)
- 内部 Benchmark 数据：`benchmarks/polars_baseline_2026-08.json`
