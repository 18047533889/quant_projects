# LQTP 因子转换器

把 **factor_engine DSL** 或 **Python 落值因子** 转成 LQTP `RunFactor` 可提交的公式。

对照手册：LQTP-backtest 因子服务用户手册（**2026-07-19**）。

## 入口

| 方式 | 说明 |
|------|------|
| CLI | `python -m scripts.cogalpha_lqtp.convert_to_lqtp` |
| 库 | `from scripts.cogalpha_lqtp.lqtp_converter import convert_dsl, convert_python, convert_auto` |

底层命名改写：`ast_translator.dsl_to_lqtp`  
兼容检测：`lqtp_dsl_compat.is_lqtp_native_dsl`

## 快速用法

```bash
cd /home/shw/quant_projects

# 单条 FE DSL
python -m scripts.cogalpha_lqtp.convert_to_lqtp \
  --dsl 'tanh(clip(cs_rank(ts_pct(close, 5)), -3, 3))'

# Python 因子文件
python -m scripts.cogalpha_lqtp.convert_to_lqtp --python-file /path/to/factor.py

# 批量：报告 report_data.js → 提交清单
python -m scripts.cogalpha_lqtp.convert_to_lqtp \
  --report-js /home/shw/reports/report_data.js \
  --out-json /home/shw/reports/lqtp_submission_formulas.json \
  --out-md /home/shw/reports/lqtp_submission_formulas.md
```

## 状态

| status | 含义 |
|--------|------|
| `ready` | 可直接 `RunFactor --formula` |
| `review` | 有近似（如 `cs_rank_gaussian→rank`）或需平台验证的算子 |
| `blocked` | 含 ATR/ADX/RSI 等 FE-only，或 Python 无法自动转写 |

## 主要改写

- `clip` → `cap`
- `ts_delay` → `delay`
- `ts_ema` / `ewm_mean` → `ema`
- `cs_rank` → `rank`，`cs_zscore` → `zscore`
- `and_` / `or_` / `not_` → 中缀 `and` / `or` / `not`
- `tanh(x)` → `2 * sigmoid(2 * x) - 1`
- `cs_rank_gaussian(x)` → `rank(x)`（近似）
- `ts_median(x,d)` → `ts_quantile(x,d,0.5)`

## Python 手写覆盖

自动 AST 搞不定的因子（`groupby`/`ewm`/`cumcount` 等）可写入
`lqtp_converter.MANUAL_LQTP_OVERRIDES`（按 `factor_key` 或函数名），批量报告会优先用覆盖。

## 扩展评估（2026-08）

`eval_lake_fast.py` 默认启用扩展评估（可用 `--skip-extended` 关闭）：

| 维度 | 说明 |
|------|------|
| 行业中性 RankIC | 申万一级（sw_l1）组内去均值 |
| 市值中性 RankIC | 每日 OLS：因子 ~ log(MarketCap) 残差 |
| 行业+市值双中性 | 先行业中性，再市值中性 |
| IC 月度热力图 | 按年月聚合 Mean RankIC |
| IC 半衰期 | 自相关衰减拟合 |
| IC 自相关 | lag 1 / 5 / 20 |
| 因子 Rank 换手 | 截面排名日度自相关 |

辅助数据优先经 **`data_access`** 读取登记数据集（`ashare_stock_industry` / `ashare_stock_valuation_daily`，含 COS mirror）；失败时回退 `clean-cos-ro`。

**与 factor_engine / data_access 对齐（2026-08）：**

- 落值：`materialize.py` / `python_materialize.py` 统一走 `default_ashare_pv_data_source_config` → `DataAccessSource(ashare_stock_daily)`；后端默认 `auto`（可用 `FACTOR_ENGINE_OPERATOR_BACKEND` 覆盖）
- 评估：`eval_lake_fast.py` 支持 `--returns-source auto|data_access|lqtp`；无 LQTP 收益缓存时自动从 `ashare_stock_daily.Return` 构建
- 因子 parquet：兼容 `datetime/asset` 与 `trade_date/symbol`，以及 hive 分区 `year=*/data.parquet`

```bash
python -m scripts.cogalpha_lqtp.eval_lake_fast --work-dir data/cogalpha_lqtp_production
python -m scripts.cogalpha_lqtp.eval_lake_fast --returns-source data_access   # 纯 data_access 收益
python -m scripts.cogalpha_lqtp.eval_lake_fast --force-aux-cache   # 强制重建行业/市值缓存
```

TopK 回测已从默认报告移除（无 TopK 数据时不显示该区块）。

