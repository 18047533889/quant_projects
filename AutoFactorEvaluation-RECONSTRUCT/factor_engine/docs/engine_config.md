# FactorEngine 配置参数手册

本文档列出运行 `factor_engine` 所需的所有外部传入参数。每个参数标明 **是否必填**、**类型**、**说明** 和 **示例值**。

---

## 一、YAML 配置文件结构

因子引擎通过 YAML 配置文件驱动。一个完整的配置文件由以下 4 个部分组成：

```yaml
factor:          # [必填] 因子定义
data_source:     # [必填] 数据源
backend:         # [可选] 后端类型（默认 pandas）
engine:          # [可选] 引擎参数
```

---

## 二、`factor` — 因子定义

```yaml
factor:
  name: momentum_5d_rank       # [必填] 因子名称
  expr: rank(ts_mean(col("close"), 5))   # [必填] 因子公式
  calc_mode: expr               # [可选] 计算模式: "expr"(默认) | "code"
  freq: 1d                      # [可选] 频率（默认 1d）
  universe: equities            # [可选] 股票池标签
  description: 5日动量因子       # [可选] 描述
```

| 字段 | 必填 | 类型 | 说明 | 示例 |
|------|:----:|------|------|------|
| `name` | ✅ | `string` | 因子名称，唯一标识 | `"momentum_5d_rank"` |
| `expr` | ✅ | `string` | 因子公式。`calc_mode=expr` 时为 DSL 表达式字符串；`calc_mode=code` 时为 Python 函数源码 | `"rank(ts_mean(col('close'), 5))"` |
| `calc_mode` | ❌ | `"expr"` \| `"code"` | 计算模式。`expr`=算子表达式驱动，享受全部优化；`code`=Python 函数直接计算 | `"expr"` |
| `freq` | ❌ | `string` | 因子频率标签，写入物化元数据 | `"1d"` |
| `universe` | ❌ | `string` | 股票池/标签，仅用于文档 | `"equities"` |
| `description` | ❌ | `string` | 人类可读描述 | `"5日动量因子"` |

### calc_mode="expr" 的 DSL 语法

- 支持：函数调用、四则运算、比较运算、一元负号、字面量
- 字段引用：`col("close")` 或裸写 `close`
- 算子名必须位于 `build_dsl_allowlist()` 白名单中（内建 440+ 算子 + 外部加载的算子）
- 不支持：`import`、属性访问、列表推导、`lambda`

### calc_mode="code" 的函数签名

```python
# expr 字段存放完整的 Python 函数源码
def my_factor(close, volume, vwap, tiny_run=False):
    import pandas as pd
    result = (close - vwap) / (close.rolling(20).std() + 1e-8)
    return result * volume.rank(pct=True)
```

- 函数参数名将自动匹配数据源的字段名
- 引擎会自动加载同名数据列传入
- `tiny_run` 参数可选接收，用于判断是否处于小体量模式

---

## 三、`data_source` — 数据源

```yaml
data_source:
  type: parquet                  # [必填] 数据源类型
  root: /data/stock_daily_bar/   # [必填] 数据根路径（绝对路径）
  timestamp_col: TradeDate       # [必填] 时间戳列名
  instrument_col: Symbol         # [必填] 标的列名
  fields:                        # [可选] 字段映射: DSL名 → 实际列名
    close: Close
    volume: Volume
    vwap: Vwap
  start_date: 2024-01-01         # [可选] 开始日期过滤
  end_date: 2024-12-31           # [可选] 结束日期过滤
  max_files: 5                   # [可选] 最大读取文件数
```

### 支持的数据源类型

| `type` | 说明 | 适用场景 | 必填 options |
|--------|------|----------|:-----------:|
| `parquet` | 通用 parquet 数据源 | 普通 parquet 文件/目录 | `root`, `timestamp_col`, `instrument_col` |
| `parquet_kline` | K 线专用（DuckDB 批量读） | OHLCV 标准 K 线数据 | `root`, `timestamp_col`, `instrument_col` |
| `multi_parquet` | 多文件 parquet（fundamentals） | 基本面等多文件数据集 | `root`, `timestamp_col`, `instrument_col` |
| `cleaned_parquet` | 预清洗 parquet | 已清洗的标准格式数据 | `root` |
| `composite` | 多表关联 | 主表+附表 asof join | `anchor`, `anchor_column`, `sources` |

### 字段映射说明

`fields` 是一个 DSL 字段名 → parquet 列名的映射字典。当 DSL 中的 `col("close")` 被调用时，引擎实际读取 parquet 中的 `Close` 列。

```yaml
fields:
  close: Close       # DSL名: close → 实际列名: Close
  volume: Volume
```

若 DSL 名与 parquet 列名一致，可省略映射。

---

## 四、`backend` — 后端类型（可选）

```yaml
backend:
  type: pandas       # [可选] "pandas" (默认) | "polars" | "polars_lazy" | "pandas_modin"
```

| 值 | 说明 |
|----|------|
| `pandas` | 标准 Pandas 执行（默认） |
| `polars` | Polars 立即执行 |
| `polars_lazy` | Polars 惰性执行（查询优化） |
| `pandas_modin` | Modin 多核/分布式 pandas |

---

## 五、`engine` — 引擎参数（可选）

```yaml
engine:
  enable_cache: true                # [可选] 启用计算缓存（默认 true）
  tiny_run: false                   # [可选] 小体量模式（默认 false）
  factor_engine_operators: null     # [可选] 外部算子库路径（默认 null=内置）
```

| 字段 | 类型 | 默认值 | 说明 |
|------|------|:------:|------|
| `enable_cache` | `bool` | `true` | 启用中间结果缓存。关闭后每次重新计算全部子树 |
| `tiny_run` | `bool` | `false` | 小体量模式。只取前 5 个标的 × 前 10 个交易日，用于 Gateway 快速验证 |
| `factor_engine_operators` | `string` \| `null` | `null` | 外部算子库目录的绝对路径。提供后引擎加载该目录下所有 `.py` 文件为算子 |

---

## 六、运行时必填参数（非 YAML，代码调用时传入）

以下参数在**代码调用**时必须显式传入，不提供默认值：

### 6.1 因子值落盘

```python
from runtime.engine import FactorEngine

engine, factor = FactorEngine.from_config("config.yaml")

# 必须传入 output_path — 支持本地路径和 COS 路径
result = engine.materialize(
    factor,
    output_path="/data/factors_output",         # 本地路径
    # output_path="cos://bucket/factors_output", # 或 COS 路径（自动使用 admin-cos 命令）
    factor_id="momentum_v1",
)
# 输出格式: {output_path}/{factor_id}/data/{YYYY-MM-DD}.parquet
# 本地示例: /data/factors_output/momentum_v1/data/2024-01-01.parquet
# COS 示例: cos://bucket/factors_output/momentum_v1/data/2024-01-01.parquet
```

| 参数 | 必填 | 说明 |
|------|:----:|------|
| `output_path` | ✅ | 因子值落盘根路径。支持本地绝对路径（`/data/...`）和 COS 路径（`cos://bucket/...`）。输出格式：`{output_path}/{factor_id}/data/{YYYY-MM-DD}.parquet` |

### 6.2 Pipeline 日志输出

```python
from pipeline import run_from_config

result = run_from_config(
    "config.yaml",
    output_root="/var/log/factor_pipeline",  # [必填] 日志、摘要 JSON 输出目录
    output_path="/data/factors_output",       # [可选] 因子值落盘路径
)
```

| 参数 | 必填 | 说明 |
|------|:----:|------|
| `output_root` | ✅ | Pipeline 输出根目录。生成 `results/*.json`、`config_snapshot.yaml`、`run_summary.json` |

### 6.3 直接使用 ParquetMaterializer

```python
from storage.materializer import ParquetMaterializer

materializer = ParquetMaterializer(
    lake_root="/data/factors_output"   # [必填] 落盘根目录
)
summary = materializer.materialize(
    factor_id="momentum_v1",
    result=series,
    output_path="/data/factors_output",  # [必填] 指定因子值写入位置
)
```

---

## 七、环境变量

| 环境变量 | 用途 | 说明 |
|----------|------|------|
| `FACTOR_ENGINE_OPERATORS` | 外部算子库路径 | 同 YAML `engine.factor_engine_operators` |
| `FACTOR_ENGINE_TINY_RUN` | 小体量模式 | `"1"` \| `"true"` 开启 |
| `FACTOR_ENGINE_DISABLE_CSE` | 关闭 CSE | `"1"` \| `"true"` 关闭公共子表达式消除 |
| `FACTOR_ENGINE_USE_NUMBA` | 启用 Numba | `"1"` \| `"true"` 尝试 Numba 加速滚动算子 |
| `FACTOR_ENGINE_MAX_WORKERS` | 并行 Worker 数 | 正整数，如 `"8"` |
| `FACTOR_ENGINE_MAX_MEMORY_MB` | 软内存上限 | 正浮点数，单位 MB |
| `FACTOR_ENGINE_ENABLE_CACHE` | 计算缓存 | `"1"` \| `"true"` 启用 |

---

## 八、完整配置示例

### 表达式模式 + 物化落盘

```yaml
# config.yaml
factor:
  name: momentum_5d_rank
  expr: rank(ts_mean(col("close"), 5) / ts_mean(col("close"), 20) - 1)
  freq: 1d

data_source:
  type: parquet
  root: /data/ashare/StockDailyBar/
  timestamp_col: TradeDate
  instrument_col: Symbol
  fields:
    close: Close
    volume: Volume

backend:
  type: pandas

engine:
  enable_cache: true
```

```python
# 调用代码
from runtime.engine import FactorEngine

engine, factor = FactorEngine.from_config("config.yaml")
result = engine.materialize(
    factor,
    output_path="/data/factors_output",      # 必填
    factor_id="momentum_5d_rank_v1",         # 可选，默认使用 factor.name
)
# 输出: /data/factors_output/momentum_5d_rank_v1/data/2024-01-01.parquet
#       /data/factors_output/momentum_5d_rank_v1/data/2024-01-02.parquet
```

### 代码模式 + Gateway 快速验证

```yaml
factor:
  name: custom_alpha_test
  expr: |
    def custom_alpha(close, volume, vwap):
        result = (close - vwap) / (close.rolling(20).std() + 1e-8)
        return result * volume.rank(pct=True)
  calc_mode: code

data_source:
  type: parquet
  root: /data/ashare/StockDailyBar/
  fields:
    close: Close
    volume: Volume
    vwap: Vwap

engine:
  tiny_run: true    # 小体量快速验证
```

---

## 九、常见错误

| 错误 | 原因 | 解决 |
|------|------|------|
| `KeyError: config.yaml 缺少必填字段: factor.name` | YAML 缺少 `factor.name` | 添加因子名称 |
| `ValueError: Config must include data_source.type` | YAML 缺少 `data_source.type` | 添加数据源类型 |
| `DSLParseError: Unsupported function: xxx` | 表达式中使用了未注册的算子 | 检查算子名是否在 `build_dsl_allowlist()` 中 |
| `TypeError: Materializer expected MultiIndex (timestamp, instrument) Series` | 代码模式返回值格式错误 | 确保函数返回 MultiIndex Series |
| `ValueError: output_path is required` | 落盘时未传入 `output_path` | 必须显式指定输出路径 |
