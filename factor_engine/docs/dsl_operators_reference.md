# DSL 算子白名单参考（第 31 版 cleaned）

> **权威枚举**：[`dsl_allowlist.json`](dsl_allowlist.json)（`PYTHONPATH=. python scripts/export_dsl_allowlist.py` 生成）  
> **运行时校验**：`build_dsl_allowlist()` / `parse_expr(formula)` / [`../api/mining_integration.py`](../api/mining_integration.py)  
> **语义细节**：[`operators_semantics.md`](operators_semantics.md)

---

## 1. 如何判断「能不能写进 manifest」

| 条件 | 结果 |
|------|------|
| 名字在 `build_dsl_allowlist()` 中 | ✅ 可 `parse_expr`；美股 `expression_type: dsl` 可投递 |
| 名字仅在 `operators_catalog.md` 且 `status=stub` | ❌ 不可 parse |
| 名字在 catalog 标注 `api_expr_only` 或未注册 | ❌ 不可 parse（第 31 版前「能 parse 不能算」已取消） |
| `api.operator_registry.STUB_IR_OPS` | **恒为空集** — 勿再引用为占位集合 |

本地自检：

```bash
cd ~/quant_projects/factor_engine
PYTHONPATH=. python scripts/validate_delivery_formula.py "rank(ts_mean(close, 20))"
PYTHONPATH=. python scripts/export_dsl_allowlist.py   # 刷新 docs/dsl_allowlist.json
```

---

## 2. 白名单规模（约 512 名）

含 `col`、蛇形 DSL 名、以及 **`SMA`/`EMA`/`MACD` 等大写 canonical**（经 `_aliases.py` 映射）。  
**不要**手写穷举；投递侧以 JSON 为准。

---

## 3. 可投递 — 代表算子（按类）

### 3.1 算术 / 逻辑 / 截面

`add` `subtract` `multiply` `divide` `abs` `log` `sign` `sqrt` `power` `sin` `cos` `exp`  
`if_else` `where` `and_` `or_` `not_`  
`rank` `zscore` `normalize` `quantile` `scale` `winsorize` `neutralize`

### 3.2 时序（通用）

`ts_mean` `ts_std` / `ts_std_dev` `ts_sum` `ts_max` `ts_min`  
`delay` / `ts_delay` / `shift`（同义别名）  
`ts_delta` `ts_corr` / `ts_correlation` `ts_cov` / `ts_covariance`  
`ts_rank` `ts_decay_linear` `ts_regression` / `ts_regression_slope`  
`ts_skew` / `ts_skewness` `ts_kurt` / `ts_kurtosis`

### 3.3 分组

`group_rank` `group_neutralize` `group_zscore` `group_mean`  
（`group_neutralize` 与 `neutralize` 在 DSL 中有别名关系，见 `_aliases.py`）

### 3.4 清洗 / 数值安全（已在白名单）

`protected_div` `protected_log` `protected_sqrt`  
`nan_to_num` `fillna` `ffill` `bfill` `coalesce` `winsorize`

### 3.5 信号 / 条件

`trade_when` — 见 [`adr_trade_when.md`](adr_trade_when.md)  
`hump_decay` — 阈值衰减（**不是**旧名 `hump`）

### 3.6 技术指标（已在白名单 — 节选）

**命名注意**：均线用 **`SMA(x,d)` / `EMA(x,d)`** 或别名 `Mean`/`ts_mean`；**没有** `ts_sma` / `ts_ema` 别名。

| DSL 名 | 说明 |
|--------|------|
| `ts_rsi` `ts_macd` `ts_atr` `ts_bbands` | 单序列 / HLC |
| `ts_adx` `ts_adxr` `ts_aroon` | 趋势 |
| `ts_obv` `ts_mom` `ts_roc` `ts_trix` | 量价 / 动量 |
| `ts_cci` `ts_willr` `ts_stoch` `ts_stochf` | HLC |
| `ts_kama` | 无 TA-Lib 时 EMA 近似 |

完整 `ts_*` 列表见 `dsl_allowlist.json` 筛选前缀 `ts_`。

---

## 4. 不可投递 — 待迁回 cleaned（第 31 版缺口）

下列名字在 **旧版文档 / ADR / changelog** 中常出现，但 **当前不在** `build_dsl_allowlist()`。  
投递前 **必须** 改写为白名单内算子组合，或等维护者在 `cleaned_operators/` 补实现。

**清洗 / 变换：** `pasteurize` `tail` `bucket` `hump` `ts_step`

**上下文：** `orthogonalize` `change_instrument`

**分组 / 工程：** `group_backfill` `group_scale` `densify`

**Overlap / 均线别名：** `ts_sma` `ts_ema` `ts_dema` `ts_wma` `ts_tema` `ts_trima` `ts_t3` `ts_ma_envelope`

**通道 / 波动 / 扩展 TA：**  
`ts_donchian` `ts_keltner` `ts_trange` `ts_natr`  
`ts_ad` `ts_adosc` `ts_sar` `ts_mfi` `ts_ppo` `ts_apo` `ts_cmo` `ts_bop` `ts_ultosc` `ts_stochrsi`  
`ts_linearreg_slope` `ts_rocr` `ts_rocr100`

**向量 / stub：** `vec_avg` `vec_sum` 及全部 `*_stub`

**替代写法示例：**

| 原写法 | 替代 |
|--------|------|
| `pasteurize(x)` | `if_else(is_finite(x), x, 0)` 或 `nan_to_num(x)` |
| `ts_ema(close, 20)` | `EMA(close, 20)` 或 `ts_mean` 近似 |
| `ts_sma(close, 20)` | `SMA(close, 20)` 或 `ts_mean(close, 20)` |
| `ts_donchian(...)` | `(high - ts_max(high,d))` 等组合 |
| `ret` 列 | `close / delay(close, 1) - 1` |

---

## 5. 维护者：新增算子 checklist

1. 在 `cleaned_operators/*.py` 实现 `@register_operator`  
2. 如需 DSL 蛇形名，在 [`../cleaned_operators/_aliases.py`](../cleaned_operators/_aliases.py) 登记  
3. `PYTHONPATH=. pytest tests/test_cleaned_operators_comprehensive.py -q`  
4. `PYTHONPATH=. python scripts/export_dsl_allowlist.py`  
5. 更新本文 §3/§4 与 [`operators_semantics.md`](operators_semantics.md)

---

## 6. 相关文档

| 文档 | 用途 |
|------|------|
| [`miner_delivery_spec.md`](miner_delivery_spec.md) §4.1 | 美股公式与字段契约 |
| [`算子与导入教程.md`](算子与导入教程.md) | 挖掘组 import / 校验 |
| [`../cleaned_operators/operators_catalog.md`](../cleaned_operators/operators_catalog.md) | 全量审计（含历史 `api_expr_only` 标注） |
