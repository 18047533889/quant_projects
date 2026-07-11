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

## factor_engine 别名陷阱（勿在 GTJA 投递 DSL 中使用）

| factor_engine 别名 | 实际语义 | GTJA 正确写法 |
|--------------------|----------|---------------|
| `SMA(x, n)` | 简单滚动均值 `ts_mean` | GTJA `SMA(x,n,m)` → `ts_ema(x, span)` |
| `delay` / `shift` | → `ts_delay` | 投递写 `ts_delay` |
| `if_else` / `IIF` | → `where` | 投递写 `where` |
| `max` / `min`（两列） | → `flex_max` / `flex_min` | 投递写 `flex_max` / `flex_min` |
| `neutralize` | → `group_neutralize`（组内去均值） | OLS 中性化用 `cs_resid` / `cs_regression` |

数据或 macro/benchmark 列接入后，重新跑 `convert_and_build_delivery.py` 即可扩展 catalog。
