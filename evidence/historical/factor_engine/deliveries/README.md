# 因子 DSL 转换结果

## 输出内容

```
factor_delivery_converted/
├── converted_factors.csv          # 所有 474 个因子的转换结果
├── conversion_summary.json         # 转换统计摘要
├── factors_combined/              # 每个因子的完整 JSON（含原始代码）
│   ├── factor_breakout_...json
│   └── ...
├── convert_factors.py             # 转换脚本（可重复运行）
└── README.md                    # 本文件
```

## 转换结果总览

| 指标 | 数量 |
|---|---|
| 总因子数 | 474 |
| 完全转换 | 319（67%）|
| 含文字/pandas记号 | 155（33%）|

**按来源：**

| 来源 | 总数 | 完全转换 |
|---|---|---|
| original（日频量价） | 378 | 262（69%）|
| fundamental（基本面） | 56 | 51（91%）|
| dev（xalpha 独有） | 14 | 6（43%）|
| minute（分钟频） | 26 | 0（0%）|

## CSV 字段说明

| 字段 | 说明 |
|---|---|
| `factor_name` | 因子名 |
| `source` | 来源（original/fundamental/dev/minute）|
| `pool` | 池（qualified/elite）|
| `domain` | 领域（alpha-drawdown、alpha-market-cycle 等）|
| `ic` / `rank_ic` | IC / Rank IC 回测指标 |
| `source_formula` | 原始公式文本（来源 DSL）|
| `lqtp_formula` | 转换后的 LQTP DSL 公式 |
| `fe_formula` | 转换后的 factor_engine DSL 公式 |
| `lqtp_notes` | LQTP 转换备注 |
| `fe_notes` | factor_engine 转换备注 |

## DSL 转换规则

### 1. 窗口参数移至末尾
```
Source:  MA(close, 20)        → LQTP: ts_mean(20, close)
Source:  EMA(ret, 21)         → LQTP: ema(21, ret)
Source:  rolling_std(ret, 20) → LQTP: ts_std(20, ret)
```

### 2. 下标语法统一
```
Source:  ts_rank_{20}(x)      → LQTP: ts_rank(20, x)
Source:  rolling_mean_{20}(x) → LQTP: ts_mean(20, x)
Source:  EMA5(x)              → LQTP: ema(5, x)
Source:  SMA20(x)             → LQTP: ts_mean(20, x)
Source:  mean_5d(x)          → LQTP: ts_mean(5, x)
```

### 3. LQTP 与 factor_engine 的差异

| 操作 | LQTP | factor_engine |
|---|---|---|
| 截面排名 | `cs_rank` | `rank` |
| 延迟 | `delay` | `ts_delay` |
| 技术指标 | `ts_rsi`, `ts_macd` | `RSI`, `MACD` |
| 最大/最小 | `max`, `min`（元素级）| `flex_max`, `flex_min` |

## 剩余未转换的记号（LQTP 不支持）

以下几类因子包含 LQTP 公式层无法表达的记号，已保留在 `<...>` placeholder 中：

1. **分钟频自定义算子**：`EMA_volume`、`ewm_mean`、`vol_ratio`、`daily_skew` 等
2. **文字描述记号**：`I[...]`（指示函数）、`R(...)`（排名运算）、`C`（收盘价简写）
3. **pandas 方法引用**：`daily_std`、`daily_corr`、`ifnan` 等
4. **ADX 子指标**：`PLUS_DI`、`MINUS_DI`

这些因子仍然**保留了原始 Python 代码**，可在 `factors_combined/*.json` 的 `code` 字段中找到，通过 Python 方式执行。

## 使用方法

```bash
# 重新运行转换脚本
python convert_factors.py --output /path/to/output

# 读取转换结果
python -c "
import csv
rows = list(csv.DictReader(open('converted_factors.csv')))
for r in rows:
    print(r['factor_name'], r['lqtp_formula'][:80])
"
```
