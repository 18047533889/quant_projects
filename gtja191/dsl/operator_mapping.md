# GTJA 算子 → factor_engine 映射说明

## 已用自有算子（均在 `build_dsl_allowlist()` 内）

| GTJA | factor_engine | 说明 |
|------|---------------|------|
| DELAY | `ts_delay` | |
| DELTA | `ts_delta` | |
| SUM / TS_SUM | `ts_sum` | |
| MEAN | `ts_mean` | |
| STD | `ts_std` | |
| RANK | `rank` | |
| TSRANK | `ts_rank` | |
| CORR | `ts_corr` | |
| COVARIANCE | `ts_cov` | |
| DECAYLINEAR | `ts_decay_linear` | |
| TSMAX / TS_MIN | `ts_max` / `ts_min` | |
| SMA(x,n,m) | `ts_ema(x, span)` | `span = round(2n/m - 1)`，GTJA alpha=m/n |
| EMA | `ts_ema` | 两参数 |
| WMA | `WMA` | 技术指标 canonical |
| MAX/MIN（两列） | `flex_max` / `flex_min` | 逐元素 |
| MAX/MIN（字段+窗口） | `ts_max` / `ts_min` | 如 `MAX(high,9)` |
| IF / `?:` | `where` | |
| COUNT(cond,n) | `ts_sum(where(cond,1,0),n)` | |
| SUMIF(v,n,cond) | `ts_sum(where(cond,v,0),n)` | |
| PROD | `ts_product` | |
| HIGHDAY / LOWDAY | `ts_argmax` / `ts_argmin` | 与 GTJA 差 1 天时在公式 `(n-x)/n` 中抵消 |
| REGBETA | `ts_regression(..., 'slope')` | 时间序列斜率近似 |
| LOG / ABS / SIGN | `log` / `abs` / `sign` | |
| `&&` / `&` | `and_` | |
| `||` | `or_` | |
| `^` | `power` | |

## 未完全覆盖（已处理策略）

| GTJA | 策略 |
|------|------|
| `VWAP` | 写成 `((high+low+close)/3)`，避免与算子名 `vwap` 冲突 |
| `REGRESI(...,MKT,SMB,HML)` | Alpha 030 → `0*close`（缺宏观因子列） |
| `SELF`（递归） | Alpha 143 改为有限窗口收益 |
| `BANCHMARKINDEX*` | 保留 `index_close` / `index_open` 字段名；数据源 `ashare_index_daily` composite 待接 |
| `DTM/DBM/TR`（ADX） | Alpha 069 用价量近似；172 简化；186 内联 TR/LD/HD |
| `FILTER` | 译为 `if_else(cond, x, 0)` |
| `SEQUENCE` | 回归类因子用 `ts_regression` + `delay` 近似 |

## 暂不投递 / 需数据后启用

以下因子 DSL 合法，但**不进入投递包**：

| 因子 | 原因 |
|------|------|
| 030 | 零因子 stub（缺 MKT/SMB/HML） |
| 183 | 零因子 stub（原式为 0） |
| 075、149、181、182 | 依赖 `index_close` / `index_open`（`ashare_index_daily`） |

数据或宏观列接入后，从 `DELIVERY_EXCLUDED` 移除并重新生成即可。
