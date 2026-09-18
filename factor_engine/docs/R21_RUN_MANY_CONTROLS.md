# 批量落值入口与可选控制（2026-09-18）

## 默认入口

管理员已配置批准的数据作用域和产物目标时：

```python
from factor_engine import get_engine

with get_engine() as engine:
    receipt = engine.run_many(factors)
```

这是持久化入口，返回逐因子终态回执，不返回十万份驻留内存的结果。
默认策略选择 auto 后端、80% 有效剩余内存预算、有界读算写和 DAG/CSE。
预算不是内存预分配，也不保证所有算子原生支持所有后端。
原始数据完整性、PIT、单位和输出约束仍执行；计算不授权生产发布。

需要调整时，使用已有的严格校验策略参数，不需修改环境变量：

```python
with get_engine(execution={
    "backend": "auto",
    "memory_fraction": 0.80,
    "initial_lookahead_factors": 512,
    "result_queue_initial_cap_bytes": 256 * 1024**2,
}) as engine:
    receipt = engine.run_many(factors)
```

参数是规划/队列上限，实际并发和内存仍由同一资源管理器准入。
后端选择还可使用策略支持的 pandas_numpy、polars_long、duckdb_sql；
可接受选项不代表所有表达式都能在该后端执行，也不代表 GPU 已覆盖。
auto 是依据实现能力与成本的自动选择，不是对每个数据规模都绝对最快的保证。

## 已持有底层 FactorEngine 的研究调用

```python
from factor_engine.runtime.perf_config import PerfConfig

summary = engine.run_many(
    factors,
    result_policy="sink",
    sink=write_one_factor,  # 已配置的写入函数：接收 name, value
    # 以下均可省略，让现有调度器自动选择：
    wave_size=128,
    sink_queue_bytes=64 * 1024**2,
    perf=PerfConfig(max_workers=None, enable_cse=True, native_fusion=True),
)
```

本轮将 wave_size、sink_queue_bytes 接通到既有 streaming runner，
不再需要换入口才能选择这些控制。它们只允许用于 sink 模式，
非法值在执行前拒绝；不会静默忽略参数或丢弃已编译的全批 DAG。
max_workers=1 明确限制串行；正整数是上限，不是资源授权。

底层接口保留 result_policy="return" 的兼容默认值，适合小样本；
大批量请使用持久化入口或 sink，避免把全部因子结果保留在调用方内存。
源快照身份稳定且预算允许时，跨 wave 可复用有界缓存；预算不足允许回收和重算。
不能承诺在有限内存下永久保留十万个因子的全部中间值。

`native_fusion=True` 是允许使用经过验证的多根原生融合，不是强制开启。
当前检查未找到正式后端实现 `execute_multi_roots`；因此不能把开关开启
当作已经执行了多根 SQL/Polars 融合。现有 DAG/CSE 与有界分批仍可独立使用。

## 测试边界

本说明不代表全部算子、全部参数或全部数据源已验收。
测试日志与逐算子差分账本位于仓库 evidence；编译通过、小样本有值、
独立数值验证、原生后端覆盖和真实大规模吞吐是不同证据。
