# FactorEngine — 因子计算引擎概述

## 一、定位与职责

`factor_engine` 是一个**因子编译-优化-执行引擎**，负责将因子的数学定义（表达式 DSL 或 Python 代码）转换为实际计算结果（MultiIndex `(timestamp, instrument)` 面板数据）。

### 核心职责

```
因子定义 (DSL/Code) → 编译 → 优化 → 执行 → [落盘]
```

---

## 二、两种计算模式

### 2.1 表达式模式（`calc_mode="expr"` — 默认）

通过算子与显式表达式驱动计算，全链路享受编译优化。

**完整流水线：**

```
DSL 字符串: "rank(ts_mean(col('close'), 20))"
    │
    ▼ api.dsl_parser.parse_expr()
    │  使用 Python ast.parse(mode="eval")，受限子集
    │  函数名必须位于 build_dsl_allowlist() 白名单内
    │
    ▼ expr.CleanedCall / ColumnRef / Literal  (表达式树)
    │
    ▼ ir.Analyzer.lower()
    │  依赖分析：回溯期(lookback)、列引用、时序/截面算子识别
    │
    ▼ planner.Lowerer.to_logical_plan()
    │  表达式树 → 逻辑计划 (PlanNode)
    │
    ▼ planner.Optimizer.optimize()
    │  常量折叠等编译期优化
    │
    ▼ backend.PandasBackend.execute(plan, ctx)
    │  算子派发 → cleaned_bridge → OperatorRegistry
    │  列加载 → DataSource.load_column()
    │
    ▼ MultiIndex Series (timestamp × instrument)
```

**支持的操作符：** 441 个已注册算子、116 个别名，涵盖：
- 时序算子：`ts_mean`, `ts_std_dev`, `ts_delta`, `ts_rank`, `ts_sum` 等
- 截面算子：`rank`, `group_rank`, `group_zscore`, `winsorize` 等
- 数学运算：四则运算、比较、一元负号
- 数据清洗：`fillna`, `dropna` 等
- 技术指标：`ts_rsi`, `ts_macd`, `ts_bollinger` 等

### 2.2 代码模式（`calc_mode="code"` — 新增）

通过直接提供 Python 函数源码进行计算，不经过 IR/Planner 优化链路。

**流水线：**

```
Python 函数源码: "def my_factor(close, volume): ..."
    │
    ▼ compile_code_function()
    │  解析 AST → 提取参数名 → 编译为 code object
    │
    ▼ 确定所需数据列（从函数参数名推断）
    │
    ▼ DataSource.load_column() 加载各列
    │
    ▼ execute_code_function()
    │  在受限命名空间中执行函数
    │  注入: pd, np, 及所有数据列
    │
    ▼ MultiIndex Series (timestamp × instrument)
```

**注意：** 代码模式不经过 CSE 优化、不参与多因子共享子表达式消除。适用于：
- 快速验证/原型开发
- 无法用算子表达的自定义逻辑
- Gateway 阶段的小体量运行测试

---

## 三、三种执行规模

### 3.1 正常全量（`tiny_run=False`）

处理完整数据集的所有标的和所有时间范围。

### 3.2 小体量运行（`tiny_run=True` — 新增）

限制数据范围为：
- **前 5 个标的**（instrument）
- **前 10 个交易日**（timestamp）

用于 Gateway 阶段快速验证表达式/代码的可运行状态（语法正确性、字段可访问性、无运行时异常）。`_TruncatedDataSource` 透明包装原始数据源，对上层代码无侵入。

### 3.3 增量/批量落盘

通过 `ParquetMaterializer` 支持：
- **按日分区 Parquet**：`{output_path}/{factor_id}/data/{YYYY-MM-DD}.parquet`
- `output_path` **必须显式传入**，无默认值
- 幂等 Upsert（`[datetime, asset]` 去重）
- 原子写入（`.tmp` → `os.replace`，COS 路径则本地临时文件 → `admin-cos cp` 上传）
- SQLite Catalog 元数据跟踪
- 支持本地路径（`/data/...`）和 COS 路径（`cos://bucket/...`）两种落盘目标

---

## 四、多因子优化：CSE

### 公共子表达式消除

当多个因子共享相同的子表达式时（例如都使用了 `ts_mean(col("close"), 20)`），CSE 将该子树提取到 `DAGPlan.shared_nodes` 中，只计算一次，所有因子共享结果。

```
无 CSE:
  Factor A: ts_mean(close,20) → rank(...)
  Factor B: ts_mean(close,20) → ts_std_dev(...)
  计算量: ts_mean×2 + rank + ts_std_dev

有 CSE:
  shared:   ts_mean(close,20)  ← 只算 1 次
  Factor A: plan_ref(shared) → rank(...)
  Factor B: plan_ref(shared) → ts_std_dev(...)
  计算量: ts_mean×1 + rank + ts_std_dev
```

### 三层缓存 + 持久化磁盘缓存

| 层级 | 位置 | 缓存对象 | 作用域 |
|------|------|----------|--------|
| L1 列缓存 | `CacheManager` / `PersistentCache` | parquet 读取的列数据 | 进程内 + **跨进程（磁盘）** |
| L2 子树缓存 | `PandasBackend._eval` | 中间计算结果 (plan hash key) | 进程内 + **跨进程（磁盘）** |
| L3 CSE | `planner.cse` | 多因子间共享子树 | 单次 run_many 调用 |

**持久化缓存**（新增）：当指定 ``cache_root`` 时，引擎自动启用 ``PersistentCache``：
- 列数据和中间计算结果持久化到 ``{cache_root}/factor_cache/``
- 使用 SHA256 hash 作为缓存键，确保确定性
- 内置 LRU 驱逐（默认 1GB 上限）和 TTL（默认 24 小时）
- **即使因子逐一评估（非批量）**，后续评估也能复用之前缓存的列数据和中间结果

```yaml
engine:
  enable_cache: true
  cache_root: /data/factor_cache   # 持久化缓存目录（跨进程共享）
  tiny_run: false
```

---

## 五、数据源抽象

引擎内置 5 种数据源：

| 类型 | 类 | 适用场景 |
|------|-----|----------|
| `parquet_kline` | `KlineParquetSource` | K 线 OHLCV 数据，高效 DuckDB 批量读取 |
| `parquet` | `ParquetSource` | 通用 parquet 文件/目录，支持 Hive 分区 |
| `multi_parquet` | `MultiParquetSeriesSource` | 多文件（如基本面数据） |
| `cleaned_parquet` | `CleanedParquetSource` | 预清洗 parquet，带默认列映射 |
| `composite` | `CompositeDataSource` | 多表关联（asof join） |

所有数据源返回统一格式：`pd.Series` with `MultiIndex(timestamp, instrument)`。

**分区 parquet 支持：** `ParquetSource` 原生支持 Hive 分区目录读取。

**因子值落盘格式：** `{output_path}/{factor_id}/data/{YYYY-MM-DD}.parquet`，每天一个独立文件。
`output_path` **必须显式传入**，不提供任何默认值，防止静默路径导致的数据错乱。
支持本地路径（`/data/...`）和 COS 路径（`cos://bucket/...`）两种模式——传入 `cos://` 前缀时自动使用 `admin-cos` 命令读写。

---

## 六、外部算子库（`factor_engine_operators` — 新增）

### 配置方式

```yaml
engine:
  enable_cache: true
  factor_engine_operators: /path/to/external/operators
```

或环境变量：`FACTOR_ENGINE_OPERATORS=/path/to/external/operators`

### 加载流程

1. 扫描目标目录下所有 `.py` 文件
2. **预编译**：逐个 `py_compile.compile()`，显示编译进度（`[N/M] ✅`）
3. **加载到解释器**：逐个 `importlib.util.spec_from_file_location()` + `exec_module()`
4. 加载后的模块中的算子通过 `@register_operator` 自动注册到算子注册表

### 路径原则

所有路径（`output_path`、`output_root`、`lake_root` 等）**必须显式传入**，不提供任何默认值，
防止在服务内部由于静默路径导致崩溃或数据错乱。

---

## 七、后端与加速选项

| 选项 | 启用方式 | 效果 |
|------|----------|------|
| Numba | `FACTOR_ENGINE_USE_NUMBA=1` | 滚动算子 GPU-like 加速 |
| Bottleneck | `pip install factor-engine[accel]` | ts_mean/ts_max/ts_min 加速 |
| Modin | `backend.type: pandas_modin` | 透明多核/分布式 pandas |
| Polars Lazy | `backend.type: polars_lazy` | 惰性计算，查询优化 |
| Joblib 并行 | `run_many_parallel(n_jobs=N)` | 因子级线程并行 |

---

## 八、完整 YAML 配置示例

```yaml
# 表达式模式
factor:
  name: momentum_5d_rank
  expr: rank(ts_mean(col("close"), 5) / ts_mean(col("close"), 20) - 1)
  calc_mode: expr        # 默认值
  freq: 1d
  universe: equities

data_source:
  type: parquet
  root: /data/stock_daily_bar/
  timestamp_col: TradeDate
  instrument_col: Symbol
  fields:
    close: Close

backend:
  type: pandas

engine:
  enable_cache: true
  tiny_run: false
  factor_engine_operators: null   # 使用内置算子
```

```yaml
# 代码模式
factor:
  name: custom_alpha
  expr: |
    def custom_alpha(close, volume, vwap):
        import pandas as pd
        result = (close - vwap) / (close.rolling(20).std() + 1e-8)
        return result * volume.rank(pct=True)
  calc_mode: code
  freq: 1d

data_source:
  type: parquet
  root: /data/stock_daily_bar/
  fields:
    close: Close
    volume: Volume
    vwap: Vwap

engine:
  tiny_run: true    # Gateway 快速验证
```

---

## 九、典型使用场景

| 场景 | calc_mode | tiny_run | 推荐设置 |
|------|-----------|----------|----------|
| Gateway 表达式验证 | `expr` | `True` | 快速检查语法+字段可访问性 |
| Gateway 代码函数验证 | `code` | `True` | 快速检查代码可执行性 |
| 批量因子生产 | `expr` | `False` | 开启 CSE + 多 Worker 并行 |
| 单因子研究/调试 | `expr` | `False` | 正常全量运行 |
| 自定义逻辑因子 | `code` | `False` | 使用 Python 函数直接计算 |
| 外部算子库开发 | `expr` | `True/False` | 先 tiny 验证，再全量 |

---

## 十、参考文档

| 文档 | 内容 |
|------|------|
| `engine_config.md` | **配置参数手册** — 所有 YAML 字段和运行时参数的详细说明、必填标记、示例值 |
| `canonical_data_fields.md` | 标准数据字段定义与映射 |
