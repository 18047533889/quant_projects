# Assetization 模块功能概述

## 一、模块目的

**因子资产化（Assetization）** 是因子评估流水线的第二道环节，负责：

1. 从 `GatewayPassBase` 读取通过网关校验的候选因子（`afv.json`）
2. 从外部行情数据源读取行情数据
3. 生成全局唯一的 `factor_id`
4. 计算因子值
5. 输出到临时库 `tier0/assetization_temp/`，等待路由到 `RawFactorBase`

---

## 二、核心函数

### 2.1 generate_factor_id / extract_coordinates

**签名**:
```python
def extract_coordinates(config: dict) -> dict[str, str]
def generate_factor_id(coordinates: dict[str, str], seed: object | None = None) -> str
```

**功能**: 从 Config 提取坐标，生成全局唯一因子标识符。

**入参**:
| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `config` | dict | 是 | 含 signal_structure, asset_class, frequency_bucket, domain_root, domain |
| `seed` | object | 否 | 用于稳定哈希种子；传入时同一输入重复运行得到相同 factor_id |

**出参**:
| 字段 | 类型 | 说明 |
|------|------|------|
| `factor_id` | str | 格式 `{asset_class}_{frequency}_{signal}_{suffix8}` |
| `coordinates` | dict | 完整坐标字典（四元坐标 + domain） |

### 2.2 FormulaEvaluator.evaluate

**签名**: `evaluate(formula: str, market_df: pd.DataFrame) → np.ndarray`

**功能**: 解析 DSL 公式，对行情数据执行计算，返回因子值数组。

**入参**:
| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `formula` | str | 是 | DSL 表达式，如 `ts_mean(rank(vwap / close - 1), 3)` |
| `market_df` | pd.DataFrame | 是 | 行情数据，含 TradeDate, Symbol 及公式引用列 |

**出参**: `np.ndarray` — 一维 float64 因子值数组

### 2.3 process_factor_dir

**签名**:
```python
def process_factor_dir(
    factor_dir: str | Path,
    *,
    market_data_path: str | Path,
    output_base: str | Path | None = None,
    temp_base: str | Path | None = None,
) -> tuple[str, str]
```

**功能**: 处理单个已通过 Gateway 审查的因子目录。**所有路径必须从外部配置传入，无代码级默认值。**

**入参**:
| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `factor_dir` | str/Path | 是 | tier1/gateway_pass_base 中的因子目录 |
| `market_data_path` | str/Path | 是 | 行情数据根路径（StockDailyBar 等子目录的父目录） |
| `output_base` | str/Path | 否 | assetization_raw_factor_base 根目录 |
| `temp_base` | str/Path | 否 | tier0/assetization_temp 路径 |

**出参**:
| 字段 | 类型 | 说明 |
|------|------|------|
| `factor_id` | str | 生成的全局唯一因子 ID |
| `temp_dir` | str | 临时输出目录绝对路径 |

---

## 三、执行流程

```
① 读取 GatewayPassBase 输入
   ├── afv.json → 校验 Gateway.Label == "Pass"
   └── manifest.json → 读取 domain_root, frequency_bucket 等元信息

② 提取坐标，生成 factor_id
   └── extract_coordinates(config) + generate_factor_id(seed={candidate_id, formula, Config})

③ 读取行情数据
   ├── 路径: {market_data_path}/StockDailyBar/{YYYY-MM-DD}.parquet
   └── 通过 cache_utils.load_market_data_cached() 加载（带全局缓存）

④ 计算因子值
   └── FormulaEvaluator.evaluate(formula, market_df) → np.ndarray

⑤ 构建输出 DataFrame
   └── 列: TradeDate, Symbol, factor_id, factor_value

⑥ 写入 tier0/assetization_temp/
   ├── data/{YYYY-MM-DD}.parquet（按 TradeDate 分片为每日一个 parquet）
   ├── afv.json（保留上游 Gateway 信息，追加 Assetization 段）
   └── manifest.json（保留上游元信息，追加 factor_id）
```
