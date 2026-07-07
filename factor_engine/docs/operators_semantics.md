# 算子语义与约定

> **第 5 版更改-shw**：与 WQ 算子库落地同步；语义与 DSL 限制见本文；完整变更列表见 [`changelog_shw.md`](changelog_shw.md)。
>
> **第 7 版更改-shw**：补充 **清洗 / 技术指标 / 上下文 / group_*** 的可执行语义与路线图链接；占位列表不再包含 `group_*`。
>
> **第 10 版更改-shw**（历史）：`bucket` / `trade_when` / `ts_step` / `hump` 曾在旧 `api/operators` 路径实装；**第 31 版** 起以 [`dsl_operators_reference.md`](dsl_operators_reference.md) 白名单为准（其中 `trade_when` ✅，`bucket`/`hump`/`ts_step` 待迁回 cleaned）。
>
> **第 31 版更改-shw**：**cleaned_operators 全量接入** — `STUB_IR_OPS` 恒为空；未在 cleaned 实现的算子 **不在 DSL 白名单**。占位与 catalog-only 名字见 [`../cleaned_operators/operators_catalog.md`](../cleaned_operators/operators_catalog.md) 中 `status=stub` 条目（约 79 个），**不可用于投递**。
>
> **第 19 版更改-shw**：技术指标 **第三批**（`ts_bop`/`ts_mom`/`ts_stochf`/`ts_trix`/`ts_adxr`/`ts_dx`/`ts_rocr`/`ts_rocr100`/`ts_linearreg_slope`/`ts_linearreg_angle`）；详见 `changelog_shw.md`「第 19 版」。

## Bar / 日历

- 引擎中间结果为 `(timestamp, instrument)` 的 MultiIndex Series。
- WorldQuant 文档中的 “days” 在本实现中对应 **数据行（bar）**，不是交易所日历日；`ts_*` 窗口长度 `d` 为 **每个标的上的连续 bar 数**。

## DSL（`parse_expr`）

- 基于 Python `ast.parse(..., mode="eval")`，因此 **`and` / `or` / `not` 不能作为函数名**；请使用 **`and_` / `or_` / `not_`**。
- **字段引用（挖掘侧标准）**：公式字符串里 **裸写列名** 即可，如 `rank(ts_mean(close, 20))`；解析器自动视为列引用。`col("close")` 与 `close` 等价。
- 比较运算写 **`close > open`** 或 **`col("x") < col("y")`**；**不支持链式比较**（如 `a < b < c`）。
- 多元算术请用 **`add(...)` / `multiply(...)`** 等函数，而非 `and` 类关键字。

## 依赖

- `ts_quantile` 与截面 `quantile` 使用 **scipy** 的逆正态 CDF；安装可选依赖：`pip install "factor-engine[pandas]"`（已包含 `scipy`）。
- **TA-Lib**（可选）：扩展技术指标（含 MACD/ATR/STOCH/MFI/OBV、**ADX/Aroon/AD/ADOSC/SAR**、**CMO/PPO/APO/UltOsc/StochRSI**、**TEMA/TRIMA/T3**、**BOP/MOM/STOCHF/TRIX/ADXR/DX/ROCR/ROCR100/线性回归** 等）在已安装时优先走 C 实现；否则为 pandas/numpy 退化（`ts_kama` 无库时用 EMA 近似；**`ts_sar` 无库时用简化 PSAR，与 TA-Lib 数值可能略有差异**；**`ts_bop` 无真实 open 时用上一根 close 近似 open（首 bar 用自身 close），与标准 BOP 列契约不同**；**`ts_trix` 无库时为三重 EMA 的 1 期 ROC×100，与 TA-Lib 在边界上可能略有差异**）。`pip install "factor-engine[talib]"`（部分环境需系统 TA-Lib 库）。
- **Bottleneck**（可选）：`pip install "factor-engine[accel]"` 后，`ts_mean` / `ts_max` / `ts_min` 可走 `move_*`；`FACTOR_ENGINE_DISABLE_BOTTLENECK=1` 强制走 pandas。**Numba** 仍在路线图阶段 B。

## 白名单与「已实现」语义（第 31 版）

**投递与 `parse_expr` 只认白名单。** 完整名单见 [`dsl_operators_reference.md`](dsl_operators_reference.md) 与 [`dsl_allowlist.json`](dsl_allowlist.json)。

### 已在白名单 — 清洗 / 截面 / 分组

- **数值安全**：`protected_div` / `protected_log` / `protected_sqrt`；`nan_to_num` `fillna` `ffill` `bfill` `coalesce`
- **截面**：`rank` `zscore` `normalize` `quantile` `scale` `winsorize` **`neutralize(x, y)`**（截面 OLS 残差）
- **分组**：`group_rank` `group_neutralize` `group_zscore` `group_mean`（≈ 行业去均值用 `group_neutralize`）

### 已在白名单 — 技术指标（节选）

- **单序列 / 别名**：`ts_rsi` `ts_macd` `ts_bbands` `ts_mom` `ts_roc` `ts_trix` `ts_kama`；均线用 **`SMA`/`EMA`**（**不是** `ts_sma`/`ts_ema`）
- **HLC / OHLCV**（须数据含对应列）：`ts_atr` `ts_adx` `ts_adxr` `ts_aroon` `ts_cci` `ts_stoch` `ts_stochf` `ts_willr` `ts_obv`
- **矩**：`ts_skew` / `ts_skewness` `ts_kurt` / `ts_kurtosis`

TA-Lib 可选加速；无库时部分算子有 pandas 退化（见上文「依赖」节）。

### 已在白名单 — 信号

- **`trade_when(trigger, alpha, exit_)`** — 见 [`adr_trade_when.md`](adr_trade_when.md)
- **`hump_decay(x, hump)`** — 阈值衰减（**不是**旧 DSL 名 `hump`）

### 待迁回 cleaned（勿投递 — 语义见 ADR / 旧版）

下列算子在 **旧架构** 或 **ADR** 中有完整语义，但 **当前不在白名单**。维护者迁回前，挖掘侧勿写入 manifest：

| 类别 | 名字 |
|------|------|
| 清洗 | `pasteurize` `tail` |
| 变换 | `bucket` `hump` `ts_step` |
- **上下文**：`orthogonalize` `change_instrument` — **当前不在白名单**（待迁回 cleaned；勿投递）
| 分组 | `group_backfill` `group_scale` |
| 扩展 TA | `ts_donchian` `ts_keltner` `ts_natr` `ts_trange` `ts_ad` `ts_adosc` `ts_sar` `ts_mfi` `ts_ppo` … |

**`bucket` 语义（迁回前参考）**：截面 rank 分桶；参数 `buckets` / `range` / `skipBoth` / `NaNGroup` — 见历史 ADR 与 changelog 第 10 版。

**`ts_step` / `hump`**：旧名；当前白名单用 **`hump_decay`**（**不是** `hump`）。`ts_step` 待迁回 cleaned。

## 子树缓存（MVP）

- 若 `FactorEngine(..., cache=CacheManager())` 或配置 `engine.enable_cache: true`，`PandasBackend` 对 **结构相同** 的子计划节点结果做内存复用（键为 `op + attrs + 子树形状`，与列指纹无关的 MVP）。

## 路线图

- 待迁移算子列表：见 [`dsl_operators_reference.md`](dsl_operators_reference.md) §4。

## 占位与 catalog-only 算子（第 31 版）

**cleaned 架构下**：

- **`build_dsl_allowlist()`** 只收录 **有 pandas runtime** 的名字（`cleaned_operators` 中 `status=implemented` 或 alias）。  
- **`api.operator_registry.STUB_IR_OPS`** 为 **空集**（兼容旧 Gateway 引用）。  
- 名称带 `*_stub`、`vec_avg`、`vec_sum` 等 **catalog 标注为 stub** 的算子：**不在白名单**，parse 会直接失败；若强行调用 runtime 则 `NotImplementedError`。

**权威列表**：[`cleaned_operators/operators_catalog.md`](../cleaned_operators/operators_catalog.md) — 筛选 `status` 列：

| status | 含义 |
|--------|------|
| `implemented` | 可 DSL、可执行 |
| `alias` | 别名，指向 canonical |
| `stub` | 语义预留，无 runtime（勿投递） |
| `api_expr_only` | 历史标注；第 31 版后多数已迁入 cleaned 或不再入白名单 |

### 远期 stub 数据契约（概要，仍适用语义设计）

| 约定 | 说明 |
|------|------|
| 输入形态 | 默认一个 child：与 `(timestamp, instrument)` 对齐的浮点列 |
| 频率 | 财报类季频对齐到日 bar；tick 类需数据源 schema 约定 |
| PiT | 引擎不代做 asof join |
| 与 `ts_*` | 通用时序用已有 `ts_delta` 等组合；stub 仅语义标签 |
