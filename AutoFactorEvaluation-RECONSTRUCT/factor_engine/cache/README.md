# `cache` — 执行期分层缓存

因子引擎在 **一次 `run` / `run_many`** 内复用中间结果，避免重复读盘、重复 unstack、重复算同一子树。

## 协作者速览（约 3 分钟）

1. **L0 CSE**：多因子公共子树 → `ExpressionCache` / `shared_result_cache`
2. **L1 Panel**：MultiIndex Series → 宽表 panel → `PanelCache`
3. **L1 Column**：数据源列预加载 → `DataAccessSource._column_cache`（见 `storage/`）
4. **L2 子计划**：内存 plan 缓存 → `storage.cache.CacheManager`
5. **L3 磁盘**：持久化子计划 → `PersistentPlanCache`（production 默认开）

入口：`ExecutionCacheSession.wrap_context()` 把各层挂到 `backend.context.ExecutionContext`。

## 文件说明

| 文件 | 作用 |
|------|------|
| `cache_policy.py` | `CachePolicy`：各层默认开关；`for_mode("production")` |
| `layers.py` | `CacheLayer` 枚举、`CacheHitStats` 命中统计 |
| `expression_cache.py` | L0：`plan_ref` / CSE 共享结果读写 |
| `panel_cache.py` | L1：Series unstack 宽表缓存 |
| `column_cache.py` | 列缓存作用域键（dataset + 时间窗 + snapshot） |
| `session.py` | `ExecutionCacheSession`：统一绑定 L0–L3 与 runtime 统计 |

## 与 runtime 的关系

`runtime.engine.FactorEngine.run_many()` 在启用 CSE 时创建 `ExecutionCacheSession`，
多因子共享子树只算一次；`runtime/perf_config.py` 的 `PerfConfig` 可关闭部分层。

## 相关文档

- [`../runtime/perf_config.py`](../runtime/perf_config.py)
- [`../backend/context.py`](../backend/context.py)
- [`../docs/源码注释导读.md`](../docs/源码注释导读.md)
