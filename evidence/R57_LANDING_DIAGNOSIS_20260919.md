# 落值链路诊断报告 — R57 目录 / 全量落值性能

日期：2026-09-19  
工作树：`/home/sunhaiwei/quant_projects`（`qs-server-c`）  
本轮基线提交：`a31e77ec` → 新增 `583fdc69`、`97c6a5b0`

---

## 0. 直接结论

| 问题 | 结论 |
|---|---|
| "每个因子新建数据源"贵在哪 | 构造本身只要 **0.005s**（惰性）。贵的是**每个因子重新走一遍列读取**，以及单因子 `run()` 没有 DAG/CSE 共享 |
| 共享数据源能救多少 | 4 因子 162.0s → 129.2s，**仅 1.25x** |
| 装资源经纪人（开列缓存）能救多少 | 129.2s → 133.9s，**噪声内，0 收益** |
| **真正的杠杆是什么** | **换后端**：pandas 133.9s → `polars_long` **46.5s，2.8x**，且 CPU 从 114% 升到 358% |
| 全量落值的瓶颈 | 不是 IO，是**每因子编译+规划+执行的 CPU**。单核跑 11.4 万因子 ≈ 19 天量级；必须多核并行 |
| auto 自动选后端能用吗 | **不能用**。实测 `run_many` + `auto` 秒退：`PhysicalRegionPlan is not research-ready: row-count estimate unavailable`。根因见 §4 —— 扫描成本能算出 855 万行，但没回流到 `ctx` |
| 11.4 万里不能落值 | **125 行编译硬失败**（本轮修好 31 行 → 剩 94）+ **134 行生产准入缺口**。另有 19587 行是**探针口径误报**，不是真问题 |

---

## 1. "每个因子新建数据源"是什么意思

`jobs/land_evoalpha14.py:314-323`，循环体内部：

```python
for nm, ex in formulas.items():
    ds  = build_data_source({...})                              # ← 每个因子新建一个数据源
    eng = FactorEngine(build_backend("pandas"), ds, run_mode="research")
    out = eng.run(Factor(name=nm, expr=ex), market="ashare")     # ← 单因子 run，非 run_many
```

三件事被反复做：

1. **新建 `DataAccessSource`** —— 构造是惰性的（`refresh_snapshot` 只 0.68s），所以这一步本身不贵；
2. **重新读列** —— 每个新数据源都是空缓存，`AdjClose` 冷读 60.67s、页面缓存热读 5.89s，逐因子重付；
3. **单因子 `run()`** —— 走不了 `run_many` 的多因子 DAG、CSE 共享子表达式、自适应并发调度。

对 14 个因子这只是浪费；对 11.4 万因子这是**结构性错误**。

---

## 2. 落值速度分解（实测，4 因子 / 3 列）

同一份数据（`ashare_stock_daily_adj`，2019-01-02 ~ 2026-08-24，**855.9 万行 × 5461 标的**）。

| 模式 | 后端 | 数据源 | 资源经纪人 | 耗时 | 峰值 CPU |
|---|---|---|---|---|---|
| **A** 现状 | pandas | 每因子 1 个（×4） | 无 | **162.0s** | ~100% |
| **B** | pandas | 共享 1 个 | 无 | **129.2s** | ~100% |
| **C** | pandas | 共享 1 个 | 有（12.8 GB 预算） | **133.9s** | 114% |
| **D** | `polars_long` | 共享 1 个 | 有 | **46.5s** | **358%** |

所有模式 `runtime_stats.physical_plan` 均为 `primary_route='pandas'`、`used_polars_long_path=False`（B/C），即**默认路由把一切送到 pandas 参考路径**。

### 2.1 缓存确实是坏的，但装好后也没救

`factor_engine/storage/sources/data_access_source.py:593`：

```python
def _data_cache_authority():
    return peek_v2_resource_broker()      # 无 broker 时 → None
```

`_default_data_cache_budget(None)` → `0`，于是 `_put_cache()` 第一行条件直接短路：

```python
if (self._cache_broker is None or nbytes <= 0 or
        nbytes > self._max_cache_bytes or not owners):
    return False        # 所有列缓存写入被拒
```

**并且 `FACTOR_ENGINE_DATA_CACHE_MAX_BYTES` 环境变量在无 broker 时完全无效** —— 因为它被 `min(value, shared_budget)` 压制，`shared_budget=0` 时恒为 0：

```python
return min(value, shared_budget)     # shared_budget = broker.current_read_budget() or 0
```

装上 broker 后缓存立即生效（实测）：

```
load_column('AdjClose')  #1   60.67s
load_column('AdjClose')  #2    0.00s   ← 命中
load_column('AdjClose')  #3    0.00s
load_column('Volume')    #1    6.42s
load_column('AdjClose')  #4    0.00s
column_cache_stats: {'cached_columns': 2, 'cache_bytes': 205638978, 'max_cache_bytes': 12803349544}
```

**但 C 模式端到端只快 0 秒** —— 说明缓存不是这个批次的瓶颈。原因：整列只有 3 列被反复用，OS page cache 已经把它们留在内存里，第二次读本来就只要 5.89s。**真实的墙钟时间花在每个因子的编译/规划/执行上。**

### 2.2 摊销曲线：加大批量**没有**变便宜

| 因子数 | 后端 | 总耗时 | 每因子 |
|---|---|---|---|
| 4 | pandas | 133.9s | 33.5s |
| 4 | `polars_long` | 46.5s | 11.6s |
| 10 | `polars_long` | 172.8s | **17.3s** |

polars_long 从 4 → 10 因子，每因子成本从 11.6s **涨到** 17.3s（同批峰值 RSS 已到 19 GB）。`result_policy="return"` 把所有结果留在内存，批量越大越被内存压力拖累。

**推论：批量落值必须配 `result_policy="sink"` + 有界 `wave_size`，否则"更大一批"会变成"更慢一批"。**

### 2.3 真实瓶颈的量级

```
数据源构造            0.005 s
refresh_snapshot()    0.68  s
第一次冷读一列        60.67 s    ← 一次性，之后被 OS page cache 吸收
页面缓存热读一列       5.89 s
缓存命中              0.00 s
每因子编译+执行       剩余部分（D 模式 4 因子 46.5s，扣除读 ≈ 每因子 8-10s CPU）
```

11.4 万因子 × ~10s CPU ÷ 32 核 ≈ **10 小时**（理想并行、无 IO 抖动）；
单核串行则是 **~13 天**；现状（逐因子 + 单核 pandas）就是后者。

**所以"太慢"的根因是「后端选择 + 并行度」，不是「数据源/缓存」。**

---

## 3. 算子后端就绪矩阵（1823 canonical）

`evidence/factor_catalog_20260916/r57_backend_readiness.py` 实测：

```
=== polars slot classification ===
  SPEC:polars_pandas_delegate              940     ← 名义有 polars，实为 pandas 委托
  NO_EXPLICIT_PHYSICAL_SPEC                783     ← 生产模式分类 UNSUPPORTED
  SPEC:polars_numpy_kernel                  58     ← 真 numpy 内核
  SPEC:polars_native_expr                   24     ← 真原生 polars 表达式
  SPEC:delegate_python                      17
  NO_POLARS_SLOT                             1

=== explicit accelerator declarations ===
  none                                    1039     ← 1039 个算子没有任何加速器声明

=== backend sets ===
  1271  ('pandas_numpy', 'polars')
   548  ('pandas_numpy', 'polars', 'sql')

=== polars_long (streaming) ===
  POLARS_LONG_NATIVE          260
  POLARS_EXPR_CAPABLE         316
  POLARS_LONG_PYTHON_ROLLING   23
  POLARS_LONG_STATEFUL         19

=== sql slot ===
  canonicals with a sql-ish backend: 549
```

**直接回答"各算子有没有加速后端"：没有。**
- 1823 个里只有 **82 个**（58 numpy kernel + 24 native expr）有真正的加速实现；
- **940 个是 pandas 委托**（伪 polars，你明确反对的那种）；
- **783 个连物理规格都没有**，生产模式直接 UNSUPPORTED；
- **1039 个没有任何 accelerator 声明**；
- 549 个有 SQL 槽位 —— 但因为路由 bug（见 §5.3）**全部走不到**。

---

## 4. auto 自动选后端：代码看着接通了，实测是坏的

### 4.1 静态路径（为什么"看起来"能用）

- `build_backend("auto")` → `HybridBackend`。factory 的 docstring 仍写着"公共 run/run_many 尚未接通"，**该注释已过期**。
- 单因子 `FactorEngine.run()` → `_assert_backend_plan_authority()` **直接拒绝**：
  ```
  HybridBackend execution requires a planner-admitted PhysicalRegionPlan;
  FactorEngine has no physical-plan consumer wired
  ```
- `FactorEngine.run_many()` → `_assert_public_batch_authority(engine, research_physical_consumer=True)` 对 research + HybridBackend **放行**，随后 `batch_service.py:1737` 的 `elif ... == "HybridBackend"` 分支执行 `optimize_batch_global(...)` → `_admit_ready_single_region_batch(admission_mode="research")`。

### 4.2 实测（mode E：`auto` + broker + `run_many`，4 因子）

```
[E] ds=0.03s  run_many(4) = 1.64s  backend=auto
    → PhysicalPlanRequiredError: PhysicalRegionPlan is not research-ready:
      row-count estimate unavailable
```

**1.64 秒秒退 —— 一次执行都没发生。**

### 4.3 根因链（逐层实测定位）

| # | 环节 | 实测结果 |
|---|---|---|
| 1 | `batch_service` 是否传了扫描成本 | 传了：`column_scan_costs=batch_request.column_scan_costs` |
| 2 | `BatchSourceResolver._resolve(scope)` | `scope.dataset == anchor.dataset` 时返回 `anchor_source`，即 `DataAccessSource` 本身 ✓ |
| 3 | **`DataAccessSource.estimate_scan_cost`** | **完全正常**：`ScanCost(estimated_rows=8559273, projection_bytes=164369233, selected_bytes=1333284585, remote=True, cost_basis='physical_scope', file_count=2588, selected_files=1854, selectivity=0.3)` |
| 4 | optimizer 取全局行数 | `batch_global_optimizer.py:509` → `rows = self._known_rows(ctx)` |
| 5 | `_known_rows(ctx)` 的查询面 | 只查 **ctx**：`_known_positive_stat(ctx, ("row_count_estimate","input_row_count","estimated_rows"))` → `estimate_shape_from_context(ctx)` |
| 6 | 结果 | ctx 上没有这些字段 → `rows = None` |
| 7 | readiness 判定 | `_readiness(rows=None)` → `return False, "row-count estimate unavailable"`（`:1806`） |
| 8 | 准入 | `execution_ready=False` → 抛 `PhysicalPlanRequiredError` |

**断点：数据源的扫描成本能算出 855 万行，但这个行数没有回流到 `ctx`；而 optimizer 的全局行数只认 `ctx`。**

### 4.4 修复方向

`_known_rows(ctx)` 增加一条回退：ctx 无行数时读 `ctx.data_source.estimate_scan_cost(...).estimated_rows`；或在 `batch_service` 组装 ctx 时把 scan cost 的 `estimated_rows` 写进 `ctx.runtime_stats`。

**在修好之前，`auto` 路由不可用 —— 只能显式指定后端。** 显式 `polars_long` 实测可用且快 2.8x（见 §2）。这就是"后端感觉没增加"的直接原因：自动选路坏了，而默认落在 pandas。


---

## 5. 三个总闸的当前状态

### 5.1 证据工件 fail-closed（未修）

`evidence/primitive_verified.json` 与当前树哈希不符，实测：

```
evidence_artifact_valid() = False
validation errors (require_commit_match=True)  = 498
validation errors (require_commit_match=False) = 501
```

不匹配项集中在**共享层哈希**（`add`、`gt` 两个算子上重复出现同一组 expected 值）：

```
operator[add].implementation_hash_pandas                    expected=d3f13bbd… actual=5c2e0639…
operator[add].implementation_hash_duckdb                    expected=1e8e955d… actual=2732f830…
operator[add].semantic_contract_hash                       expected=0d5fb740… actual=7b61385c…
operator[add].bridge_parameter_hash                        expected=468513d6… actual=c469ecba…
operator[add].test_source_hash                             expected=76610d9d… actual=c91a48d6…
operator[add].composite_lowerings_python_tree_hash         expected=ab1c6a15… actual=f2e204e1…
emitter_hashes                                             expected={polars/duckdb emitter…}
```

`implementation_hash_duckdb` 的 expected 值与 `emitter_hashes` 里 `implementation_hash_duckdb_emitter` 相同 → **工件是在 emitter 层改动之前生成的**，属"证书过期"而非"代码回退"。

影响：`_load_verified_set()` 返回空集合 → 所有认证能力清零。**但不阻塞 research 模式的因子求值**，只阻塞"生产认证"路径。

修法：重跑 recert（本轮子代理因配额没跑成），或按变更面重签。

### 5.2 缓存授权为 0（未修，且有隐藏 bug）

见 §2.1。除 `_put_cache` 短路外，`min(value, shared_budget)` 使环境变量在无 broker 时静默失效。

修法二选一：
- 批量落值入口显式装 broker：`get_v2_resource_broker(DefaultExecutionPolicy())`（实测有效）；
- 或让 `_default_data_cache_budget` 在 authority 为 `None` 时把环境变量视为显式授权，而不是压到 0。

### 5.3 SQL 路由误判为 memory（未修）

两处独立实现，**都错**：

```python
# cleaned_bridge.py:399
name = type(ds).__name__.lower()          # "dataaccesssource"
if "duckdb" in name or "data_access" in name or "parquet" in name:
    return "duckdb"
return "memory"                            # ← 命中这里
```

`"data_access" in "dataaccesssource"` → **False**（类名没有下划线）→ 返回 `"memory"` → 549 个算子的 SQL 下推全部失效。

`plan_cost_router.py:367` 是审计 #355 的改良版，优先读 `ds.capabilities.engine_kind / dialect` —— 但 `DataAccessSource` **根本没有 `capabilities` 属性**（`grep -c capabilities` = 0），所以改良分支落空，同样回退到 class-name 启发、同样误判。

修法：给 `DataAccessSource` 加 `capabilities`（`engine_kind` 或 `dialect='duckdb'`），并清理 `cleaned_bridge` 的子串匹配。

---

## 6. 11.4 万里哪些不能落值

### 6.1 编译硬失败 125 行

| 行数 | 错误 | 状态 |
|---|---|---|
| 73 | `ts_vol_of_vol: min_periods must be <= inner_window` | 未修 |
| 24 | `ashare_limit_distance on continuous price close/high_limit` | **本轮已修** |
| 7 | `ashare_limit_open_failed on continuous price open/low/high_limit` | **本轮已修** |
| 5 | `cs_knn_tangent_residual: missing required parameters ['f2','f3']` | 缺参数 |
| 4 | `state_confidence_weighted_ema: missing ['confidence']` | 缺参数 |
| 4 | `state_cost_aware_slew: missing ['cost_proxy']` | 缺参数 |
| 4 | `state_uncertainty_deadband: missing ['uncertainty']` | 缺参数 |
| 4 | `state_cost_aware_deadband: missing ['cost_proxy']` | 缺参数 |

→ 本轮修好 31 行，**剩 94 行**：73 行 `ts_vol_of_vol` 参数校验 + 21 行缺必填参数。

### 6.2 生产准入缺口 134 行（能编译，但算子未获生产准入）

涉及 **30 个算子**，主要是统计检验族：

```
13  ts_signal_spectral_entropy
 7  holder_concentration_change / holder_count_change_rate
 4  chi_square_test / durbin_watson_test / jarque_bera_test / kpss_test / levene_test
 4  spearman_corr_test / ttest_one_sample / ttest_two_samples / ttest_paired / corr_test
 4  pacf / ACF / bartlett_test / kendall_corr_test / ks_test / lilliefors_test
 4  granger_causality / stationarity_test
 4  recipe_micro_* / recipe_vpmacd* / recipe_vp_* / recipe_downside_beta
```

### 6.3 ⚠️ 19587 行 `ops_missing` 是探针口径误报

```
COMPILED 但 ops_missing 非空：19587
  19102  field          ← DSL 内置列选择原语，不是算子
   1244  source_col     ← 同上
    690  holder_top10_pledge_ratio   ← dsl_parser 原生宏（compat_research surface）
```

`compile_r57_full.py` 的 `parser_allowed_names()` 已能识别 `field/source_col/holder_top10_*`，但 CSV 的 `ops_missing` 列仍把前两类记进去了 —— **需要修探针口径，否则会误判 1.9 万个因子"缺算子"**。

### 6.4 重要口径提醒

`r57_execution_status` **全部为 `NOT_RUN`**（113893/113893）：

> R57 is a compile-only manifest; R20 execution status is carried for reference only

**这份清单只证明"编译通过"，没有任何执行证据。** 严格说，现在还没有任何一行被证实"能落值"。

---

## 7. 本轮修复（已提交）

### `583fdc69` — R19 配方链接线（修 72 行）

`factor_engine/tools/catalog_recipe_migration.py` 的 `migrate_catalog_recipe_formula` **从未 import** `catalog_r19_operator_recipes`，R19 的 legacy-OHLCV 改写器是**死代码**。

本模块的 `_RecipeSpec` 只匹配短签名（`Indicator(price, window)`），而受影响的 72 行是五参模板
`Indicator(open, high, low, close, volume)`，两者不匹配 → 原样输出、`migration_changes` 为空。

修法：R19 pass 前置。因 R19 的每个替换都改变 arity，specs 无法在其输出上二次触发，组合幂等。

全量验证（113893 行）：**72 行修复 / 113821 行逐字节不变 / 0 非幂等 / 0 残留**。

```
ElderRay(field('open',Adj),field('high',Adj),field('low',Adj),field('close',Adj),field('volume',Adj))
  → ElderRay(field('high',Adj), field('low',Adj), field('close',Adj), 13)
CoppockCurve(5 参模板) → CoppockCurve(field('close',Adj), 14, 11, 10)
FisherTransform(5 参模板) → FisherTransform(field('high',Adj), field('low',Adj), 9)
```

### `97c6a5b0` — 涨跌停算子价格基表（修 31 行）

复权价迁移把每个 `field(..., table='StockDailyBar')` 无条件改写成 `StockDailyBarAdj`，但交易所涨跌停谓词**必须读官方未复权价**（typed-IR P0-30），于是这些行编译失败。

两个缺陷叠加：
1. `_OFFICIAL_LIMIT_PRICE_PARAMETERS` 里**没有** `ashare_limit_distance` / `ashare_limit_open_failed`；
2. leaf 识别只匹配**裸标识符**，而目录里全部写成 `field('close', table='StockDailyBarAdj')`，守卫即使列了也打不着。

修法：补两条表项 + 用 `_raw_price_leaf_name()` 同时接受两种形态，且只改当前读复权表的 leaf（保持幂等）。

全量验证：**31 行修复 / 113862 行不变 / 0 非幂等 / 0 残留复权 leaf**。同一公式里其他算子的复权参数**保持不动**（定向修复）：

```
OLD: multiply(ashare_limit_distance(field('close',Adj), field('high_limit',Adj)), …)
NEW: multiply(ashare_limit_distance(field("close",Raw), field("high_limit",Raw)), …)
     ↑ 同式里 ts_turnover_cost_quantile_distance 的 Adj 参数原样保留
```

单测：`factor_engine/tests/tools/` 修复前后同为 **172 passed / 13 failed**（13 个是既有的 `ModuleNotFoundError: evidence.factor_catalog_20260915` 收集错误，与本次改动无关）。

---

## 8. 建议的批量落值架构（DAG）

```python
from factor_engine.runtime.resource_broker import get_v2_resource_broker
from factor_engine.runtime.default_execution_policy import DefaultExecutionPolicy

broker = get_v2_resource_broker(DefaultExecutionPolicy())   # ① 必须先装，否则列缓存结构性失效

ds  = build_data_source({...})                              # ② 全程共享一个数据源
eng = FactorEngine(build_backend("auto"), ds, run_mode="research")   # ③ auto 逐因子选后端

eng.run_many(
    factors,
    market="ashare",
    result_policy="sink", sink=writer,       # ④ 即算即写，结果不驻留
    wave_size=256,                           # ⑤ 有界内存；>dag_width 自动走流式
    warmup_clusters=True,                    # ⑥ 按 lookback 聚类，避免全历史因子拖累整批
    enable_cse=True,                         # ⑦ 共享子表达式
)
```

四条实测依据：
- ① C 模式证明装 broker 后列缓存命中（#2/#3/#4 = 0.00s）；
- ③ D 模式证明后端是最大杠杆（pandas 133.9s → polars_long 46.5s，2.8x，CPU 114%→358%）；
- ④ `result_policy="return"` 下 16 因子 polars_long 峰值 RSS 已达 **14.5 GB**，11.4 万因子必须 sink；
- ⑤ 外层再按标的/时间段**多进程分片**吃满 32 核。

---

## 9. 未决清单（按优先级）

| # | 事项 | 影响面 |
|---|---|---|
| 1 | **修 `_known_rows(ctx)` 的扫描成本回退** → 让 `auto` 路由可用 | 全部自动后端选择 |
| 2 | 落值入口改 `run_many` + `sink` + `wave_size` + 多进程分片 | 全量落值速度（最大项） |
| 3 | 73 行 `ts_vol_of_vol` 参数校验 | 73 行 |
| 4 | `DataAccessSource.capabilities` + `_data_source_kind` 子串匹配 | 549 个算子的 SQL 下推 |
| 5 | `_default_data_cache_budget` 的 `min(value, 0)` 压制 | 环境变量静默失效 |
| 6 | 证据工件重签（recert） | 生产认证全部能力 |
| 7 | `ops_missing` 探针口径（`field`/`source_col` 应排除） | 1.9 万行误报 |
| 8 | 940 个 pandas delegate 算子转真 polars/numpy | 后端加速覆盖 |
| 9 | 21 行缺必填参数 | 21 行 |
| 10 | 134 行生产准入缺口（30 算子） | 134 行 |
| 11 | 重编译产出 R57d CSV（含本轮 103 行修复） | 交付清单 |
