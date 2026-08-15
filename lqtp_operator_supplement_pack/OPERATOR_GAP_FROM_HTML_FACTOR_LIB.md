# HTML / Catalog → LQTP 算子缺口（算子，不是因子）

- 扫描：`reports/factor_*.html`（172）+ `screening_reeval_catalog.json`（228）
- 对照：`lqtp_dsl_compat._LQTP_KNOWN_CALLS` + `is_lqtp_native_dsl` 黑名单

## 平台交不上的规模

| 指标 | 数量 |
|---|---:|
| catalog 总条数 | 228 |
| `lqtp_native=False` | 149 |
| `eval_route=local_dsl` | 115 |
| `eval_route=local_python` | 35 |
| `eval_route=lqtp_dsl` | 78 |
| HTML 能对上 catalog 的 | 115（其中非 native 85） |

## 建议新增算子（NEW=80，ALIAS=3，合计 83 名）

- 其中 **27** 个来自 HTML/Catalog 交不上的缺口（见下表）
- 另 **53** 个为 EXTRA 未来常用（非模型）：数学饱和 / OHLC 几何 / 时序统计扩展 / MACD·布林·KDJ 等
- 完整名单以 `lqtp_operators_to_add.py` 的 `NEW_OPERATORS` / `EXTRA_FUTURE_COMMON` 为准

### HTML/Catalog 缺口（27）

见 `lqtp_operators_to_add.py` → `NEW_OPERATORS`。

### 按拒单证据

| 算子 | 证据来源 | 说明 |
|---|---|---|
| `ATR_WILDER` | catalog ~35 | 最大缺口 |
| `ATR` / `true_range` | HTML ~22 + FE_ONLY | |
| `tanh` | HTML ~16 + 黑名单 | 勿用 sigmoid 替代 |
| `ts_median` | HTML ~16 + catalog | 或接受 `ts_quantile(x,w,0.5)` 同名 |
| `classify_volume_regime` | HTML ~27 | 可拆成 ema+where |
| `pow` | catalog ~15 | 有 `power`，缺同名 |
| `divide` | catalog ~14 | ≡ safe_div |
| `maximum`/`minimum`/`max`/`min` | catalog + 手册禁名 | 元素二元 |
| `ts_ema`/`ewm_mean` | DSL map 极多 + 黑名单 | ≡ ema |
| `RSI`/`RSI_WILDER`/`ADX`/`ROC`/`SMA` | FE_ONLY + HTML | |
| `rolling_vwap` | catalog | |
| `ts_zscore` / `exp` | catalog | |
| `decompose_overnight_intraday` | HTML ~3 | |
| `imbalance` | HTML | |
| `and_`/`or_`/`not_` | 黑名单 | |

## 可选别名（ALIAS=3）

| 名 | 已有 |
|---|---|
| `protected_div` | `safe_div` |
| `clip` | `cap` |
| `ts_delay` | `delay` |

## 交付

`lqtp_operators_to_add.py` 仅含算子函数实现，不含因子。
