# Factor Engine

可扩展的量化因子引擎框架，支持表达式树构建、编译优化、多后端执行与配置驱动运行。

**文档索引**：[`docs/README.md`](docs/README.md) · 挖掘投递 [`docs/miner_delivery_spec.md`](docs/miner_delivery_spec.md) · 算子写法 [`docs/算子与导入教程.md`](docs/算子与导入教程.md) · 白名单 [`docs/dsl_operators_reference.md`](docs/dsl_operators_reference.md) · 语义 [`docs/operators_semantics.md`](docs/operators_semantics.md) · 回测 ADR [`docs/adr_backtest_target_position.md`](docs/adr_backtest_target_position.md) · 变更 [`docs/changelog_shw.md`](docs/changelog_shw.md)

### 协作者速览（新人约 5 分钟）

1. **数据流（本仓库在干什么）**：`api`（`Factor` / DSL / 算子工厂，经 **`cleaned_operators`**）→ `expr` → `ir` → `planner` → `backend` → **MultiIndex 因子序列**；**编排入口**是 **`runtime/FactorEngine`**。挖掘侧入门：**[`docs/算子与导入教程.md`](docs/算子与导入教程.md)**。
2. **规范从哪读**：算子 **能否写入 manifest** 以 [`docs/dsl_operators_reference.md`](docs/dsl_operators_reference.md) 为准；参数语义见 [`docs/operators_semantics.md`](docs/operators_semantics.md)。
3. **动手跑**：最小脚本 [`examples/simple_factor.py`](examples/simple_factor.py)；配置驱动见 `examples/` 与 **`FactorEngine.run_from_config`**；**目标仓位回测**见 [`../../backtest_layer/single_asset_backtest/README.md`](../../backtest_layer/single_asset_backtest/README.md) 文首 **「新人 5 分钟上手」**。
4. **版本与变更**：[`docs/changelog_shw.md`](docs/changelog_shw.md)。

> **第 5 版更改-shw**：更新「支持的算子」「项目结构」与 `docs/` 索引，以反映 `api/operators/` 包、算子注册表及 WQ 风格扩展；细节仍以 `changelog_shw.md` 分版条为准。

> **第 7 版更改-shw**：落地 **清洗 / 技术指标 / 上下文 / group_*** 与 **子树缓存 MVP**；README 本节与结构树同步，详见 `changelog_shw.md`「第 7 版」。

> **第 8 版更改-shw**：对齐 **`docs/factor_engine_llm_prompt`（.md / .txt）** 中算子字典与 **`enable_cache`** 说明；**`test_dsl_parser`** 覆盖新 DSL；`changelog` 为第 3 版 **`group_*` 占位** 补历史脚注。详见 `changelog_shw.md`「第 8 版」。

> **第 9 版更改-shw**：**Bottleneck** 加速 `ts_mean` / `ts_max` / `ts_min`（安装 `factor-engine[accel]` 后生效）；可用环境变量 **`FACTOR_ENGINE_DISABLE_BOTTLENECK=1`** 对照测试或与 pandas 完全一致路径。详见 `changelog_shw.md`「第 9 版」。

> **第 10–14 版更改-shw**：`bucket` / `trade_when` / `ts_step(d,anchor)` / `hump` 已在 Pandas 实装；**`vec_avg` / `vec_sum`** 按路线图 **路径 B** 暂缓；**PolarsBackend** 当前 **委托 PandasBackend**（保留 `build_backend("polars")` 入口，非独立 Polars 执行）；未注册进 `cleaned_operators` 的远期算子 **不在 DSL 白名单**，`parse_expr` 即失败；**`sin` / `cos`** 与 **Joblib** 多因子示例见 `examples/run_factors_joblib.py`。详见 `changelog_shw.md`。

> **第 15 版更改-shw**：为 `bucket` / `trade_when` / `ts_step` / `hump`、`ts_regression` / `ts_quantile`、`group_*`、`orthogonalize` / `change_instrument` 及远期 stub 等 **加长源码内中文说明**（`expr/`、`api/operators/`、`pandas_backend` 相关 docstring）。详见 `changelog_shw.md`「第 15 版」。

> **第 16 版更改-shw**：**技术指标大扩展**（`ts_atr`/`ts_donchian`/`ts_keltner`/`ts_macd`/`ts_cci`/`ts_stoch`/`ts_obv`/`ts_mfi`/`ts_dema` 等，见 `operators_semantics.md`）；**`neutralize`**（截面 OLS 残差）；**`ts_skew`/`ts_kurt`**。详见 `changelog_shw.md`「第 16 版」。
> **第 18 版更改-shw**：**技术指标第二波**（`ts_adx`/`ts_aroon`/`ts_ad`/`ts_adosc`/`ts_sar`/`ts_cmo`/`ts_ppo`/`ts_apo`/`ts_ultosc`/`ts_stochrsi`/`ts_tema`/`ts_trima`/`ts_t3`）。详见 `changelog_shw.md`「第 18 版」。

> **第 19 版更改-shw**：**技术指标第三批**（`ts_bop`/`ts_mom`/`ts_stochf`/`ts_trix`/`ts_adxr`/`ts_dx`/`ts_rocr`/`ts_rocr100`/`ts_linearreg_slope`/`ts_linearreg_angle`）。详见 `changelog_shw.md`「第 19 版」。

> **第 20 版更改-shw**（**历史**；华泰对照文档已删除）：曾新增华泰研报算子对照；现 **只认** `build_dsl_allowlist()`。详见 `changelog_shw.md`「第 20 版」。

> **第 21 版更改-shw**：曾新增独立华泰对照 py（已由 **第 22 版** 替代为融入 `api/operators` / `expr`）。

> **第 22 版更改-shw**（**历史**；`expr/intraday.py`、`api/operators/intraday.py` 已在 **第 31 版** 删除）：华泰算子融入旧模块；新增 **`exp`**、**`INTRADAY_STUB_OPS`**；删除 `api/htsc_factor_factory_reference.py`。分钟 stub 现见 **`cleaned_operators/intraday_microstructure.py`**（catalog `status=stub`）。详见 `changelog_shw.md`「第 22 版」。

> **第 23 版更改-shw**：**性能与后端选项**——[`planner/cse.py`](planner/cse.py) 多因子 **CSE** + [`FactorEngine.run_many`](runtime/engine.py)；[`backend/pandas_compat.py`](backend/pandas_compat.py) 可选 **Modin**（`pandas_modin` / `FACTOR_ENGINE_USE_MODIN`）；[`PolarsBackend`](backend/polars_backend.py) 扩展算子子集 + **`polars_lazy` / `FACTOR_ENGINE_POLARS_LAZY`**；[`runtime/perf_config.py`](runtime/perf_config.py)；脚本 [`scripts/profile_pandas_backend.py`](scripts/profile_pandas_backend.py)、[`scripts/bench_pandas_vs_modin.py`](scripts/bench_pandas_vs_modin.py)。详见 `changelog_shw.md`「第 23 版」。

> **第 24 版更改-shw**：新增 **`backtest/` 单标回测子系统**（Backtrader 可选依赖）、冻结 `target_position` 对接 ADR，并补充最小示例与测试。详见 `changelog_shw.md`「第 24 版」。
>
> **第 25 版更改-shw**：回测补齐 **D-3 策略注册/策略库**（`strategy_name`/`strategy_version`/`strategy_params`/`strategy_instance_id`）并引入 **分层指标 profile**（`core`/`standard`/`industrial`，覆盖 `sortino`/`calmar`/`var_95`/`cvar_95` 等扩展指标）。详见 `changelog_shw.md`「第 25 版」。
>
> **第 26 版更改-shw**：回测补充 **信号时点与防前视**（`BacktestConfig.target_lag_bars`、`portfolio_weight_lag_bars`）、**可复现指纹**说明，并整理 README 回测章节与 ADR §13。详见 `changelog_shw.md`「第 26 版」。
>
> **第 27 版更改-shw**：为 **`api/`、`backend/`、`expr/`、`ir/`、`planner/`、`runtime/`、`storage/`、`scripts/`、`tests/`、`examples/`、`docs/`** 等目录各增 **`README.md`**，并在 [`docs/README.md`](docs/README.md) 汇总索引。详见 `changelog_shw.md`「第 27 版」。
>
> **第 28 版更改-shw**：各包 **`README.md` 深度扩写**（架构说明、逐文件表、数据流、契约与测试索引）。详见 `changelog_shw.md`「第 28 版」。
>
> **第 29 版更改-shw**：**多资产回测文档与实现对齐**——`backtest/README.md`、`docs/adr_backtest_target_position.md`、根 `README.md` 统一叙述 **执行层 → `executed_weights` → 滞后 → 毛/净收益**；明确 **`portfolio_execution_engine: python`** 与 **`FACTOR_BACKTEST_EXECUTION_ENGINE`** 的关系及 **多资产 `data_fingerprint` 基于执行后权重**。详见 `changelog_shw.md`「第 29 版」。
>
> **第 30 版更改-shw**：根目录与各包 **`README.md`** 增加 **「协作者速览（约 5 分钟）」**；**`docs/README.md`** 说明该约定。详见 `changelog_shw.md`「第 30 版」。

> **第 31 版更改-shw**：**`cleaned_operators` 全量接入** — 删除 **`api/operators/`** 与强类型 **`expr/*`** 模块；DSL 白名单与 Pandas 执行均经 **`cleaned_operators`**；详见 [`api/README.md`](api/README.md)、[`cleaned_operators/README.md`](cleaned_operators/README.md)、[`docs/changelog_shw.md`](docs/changelog_shw.md)「第 31 版」。

---

## 快速开始

```python
from api import col, rank, ts_mean
from api.factor import Factor
from backend.pandas_backend import PandasBackend
from runtime.engine import FactorEngine
from storage.kline_parquet_source import KlineParquetSource

source = KlineParquetSource(root="/data/us_stocks_sip/day_aggs_v1", max_files=5)
engine = FactorEngine(backend=PandasBackend(), data_source=source)

factor = Factor(name="mom3_rank", expr=rank(ts_mean(col("close"), 3)))
result = engine.run(factor)
print(result["result"].head())
```

---

## 配置驱动运行

引擎支持通过 YAML 文件描述因子和数据源，无需编写 Python 代码即可运行。

**步骤：**
1. 编写包含 `factor`、`data_source`、`backend`、`engine` 四个字段的 YAML 文件。
2. 调用 `FactorEngine.run_from_config(path)` 或 `FactorEngine.from_config(path)` 获取引擎实例。

**支持的数据源类型：**

| 类型 | 说明 |
|---|---|
| `parquet_kline` | K线格式 parquet，适用于 `us_stocks_sip` |
| `multi_parquet` | 通用多文件 parquet，适用于各类 fundamentals |
| `parquet` | 单文件或简单目录 parquet |

**配置示例（日K线动量因子）：**

```yaml
factor:
  name: day_aggs_rank_ts_mean_close_3
  expr: rank(ts_mean(col("close"), 3))
  freq: 1d
  universe: equities
  description: 日K线 - 3日收盘价均线截面排名，动量方向因子

data_source:
  type: parquet_kline
  root: /data/us_stocks_sip/day_aggs_v1
  instrument_column: ticker
  timestamp_column: window_start
  fields:
    close: close
  max_files: 5

backend:
  type: pandas

engine:
  enable_cache: true
```

**配置示例（fundamentals 基本面因子）：**

```yaml
factor:
  name: balance_sheet_rank_total_assets
  expr: rank(col("total_assets"))
  freq: 1d
  description: 资产负债表 - 总资产截面排名，越高代表规模越大

data_source:
  type: multi_parquet
  root: /data/fundamentals/balance_sheet
  timestamp_col: period_end
  instrument_col: tickers
  max_files: 3

backend:
  type: pandas
```

`examples/configs/` 目录下收录了覆盖全部 11 个数据集的配置文件（见下文[数据集列表](#数据集列表)）。

---

## 支持的算子

**第 31 版起**，算子 runtime 统一在 **`cleaned_operators/`**（约 **440+** 个已实现 canonical + **116** 个别名）。DSL 白名单由 **`api.operator_registry.build_dsl_allowlist()`** 自动生成（`col` + 全部有 pandas runtime 的名字）。

| 层级 | 说明 |
|------|------|
| **已实现** | 算术 / 逻辑 / 时序 / 截面 / 分组 / 清洗 / 技术指标 / 上下文 / transformational 等 — 经 `cleaned_bridge` 在 `PandasBackend` 执行 |
| **catalog-only / stub** | 在 [`cleaned_operators/operators_catalog.md`](cleaned_operators/operators_catalog.md) 标注为 `stub` 或 `api_expr_only` 的名字 **不在 DSL 白名单**，投递勿用 |
| **PolarsBackend** | 当前 **委托** `PandasBackend`（同一 cleaned 路径） |

权威清单与状态：**[`cleaned_operators/operators_catalog.md`](cleaned_operators/operators_catalog.md)**；语义细节：**[`docs/operators_semantics.md`](docs/operators_semantics.md)**。本地枚举白名单：

```python
from api.operator_registry import build_dsl_allowlist
sorted(build_dsl_allowlist().keys())
```

**常用示例**（完整白名单见 [`docs/dsl_operators_reference.md`](docs/dsl_operators_reference.md)）：

| 算子 | 类型 | 说明 |
|---|---|---|
| `close` / `col("close")` | 字段 | 裸写列名是挖掘标准；`col()` 等价 |
| `rank(x)` | 截面 | 每个时间截面内百分位排名 [0,1] |
| `zscore(x)` | 截面 | 每个时间截面内 Z-score 标准化 |
| `ts_mean(x, d)` | 时序 | 滚动均值（窗口 `d` 为 bar 数） |
| `ts_std_dev(x, d)` / `ts_std(x, d)` | 时序 | 滚动标准差 |
| `ts_delay(x, d)` / `delay(x, d)` | 时序 | 滞后 |
| `group_rank` `group_neutralize` 等 | 分组 | 组内排名 / 去均值 |
| `protected_div` `protected_log` 等 | 清洗 | 除零 / log 安全 |
| `SMA(x,d)` / `EMA(x,d)` | 均线 | **不是** `ts_sma`/`ts_ema` |
| `ts_rsi` `ts_macd` `ts_atr` `ts_adx` `ts_obv` 等 | 技术指标 | HLC/V 列契约见 `operators_semantics.md` |
| `trade_when` `neutralize` | 信号 / 截面 | 条件持仓 / OLS 残差 |
| 四则 `+ - * /` | 算术 | 或 `add`/`multiply`/… |

**可选依赖**：`pip install "factor-engine[pandas]"`（含 scipy）；`[talib]`（部分技术指标优先 C 实现）；**`[polars]`**；**`[modin]`**（`FACTOR_ENGINE_USE_MODIN=1` 或 `backend.type: pandas_modin`）；**`[parallel]`**（Joblib + CSE，见 `runtime/perf_config.py`）；**`[backtest]`**（Backtrader，见 monorepo `backtest_layer`）。性能脚本：`scripts/profile_pandas_backend.py`、`scripts/bench_pandas_vs_modin.py`。

**回测接口与语义 ADR**：[`docs/adr_backtest_target_position.md`](docs/adr_backtest_target_position.md)（输出协议、策略版本、真实数据路径、分层指标、可复现字段、**信号时点 / 防前视** §13、**多资产执行与指纹** §14）。**逐步数据流与模块说明（可替代通读源码）**：[`../../backtest_layer/single_asset_backtest/README.md`](../../backtest_layer/single_asset_backtest/README.md)。

### 多资产组合回测（Portfolio Mode）

当前多资产路径采用**组合会计口径**（research/audit friendly），并保持冻结协议 `returns/metrics/summary` 必需键不变。

- 执行入口：`single_asset_backtest.runner.run_multi_asset_backtest`
- 目标权重输入：`target_weights`，支持：
  - DataFrame 列：`timestamp`, `symbol`, `target_weight`
  - 或 MultiIndex Series：`['timestamp','symbol']`
- 契约处理：按时间对齐后 `ffill`，缺失补 `0.0`，并做权重边界与每时点 gross leverage 校验
- 组合收益口径：`realized_weights = executed_weights.shift(portfolio_weight_lag_bars)` 后与当期资产收益相乘；**默认 `portfolio_weight_lag_bars=1`**（即 **t−1** 权重 × **t** 期收益），**不允许为 0**（避免组合层面零滞后前视）
- 组合执行约束：
  - `portfolio_min_trade_weight`：最小调仓阈值（小于阈值的 delta 直接忽略）
  - `portfolio_adv_participation_cap`：按 `price*volume*cap/initial_cash` 约束每 bar 可执行权重变化
- 成本模型（`portfolio_cost_model`）：
  - `simple_bps`：`(commission_bps + spread_bps) * turnover`
  - `linear_impact`：在 `simple_bps` 基础上叠加线性冲击项（受 `portfolio_impact_coeff` 与参与率影响）
  - `square_impact`：在 `simple_bps` 基础上叠加平方冲击项（受 `portfolio_impact_coeff` 与参与率影响）
- 执行内核选择（`portfolio_execution_engine`）：`python` / `numpy` / `numba` / `auto`。若 YAML 写 **`python`**，实际参与解析的请求来自环境变量 **`FACTOR_BACKTEST_EXECUTION_ENGINE`**（`PerfConfig.from_env().backtest_execution_engine`，默认 `python`），便于不改业务配置切换内核；若写 **`numpy`/`numba`/`auto`**，则按该字面值解析（`numba` 不可用时回退 `numpy`，`auto` 优先 `numba`）。`summary` 记录 **requested/resolved**

输出在不破坏冻结必需键前提下增量包含：
- `returns.portfolio_turnover`
- `returns.portfolio_cost`
- `returns.portfolio_participation`
- `metrics.portfolio_turnover_total`
- `metrics.portfolio_cost_total`
- `metrics.portfolio_participation_max`
- `summary.mode = "multi"`

### 单资产：信号时点与 `target_lag_bars`

单资产路径**不能**自动检测因子是否误用「当日收盘后才可得」的信息；若因子层已对信号滞后，请保持 **`target_lag_bars=0`**（默认），避免双重滞后。

- **`BacktestConfig.target_lag_bars`**：在与行情对齐并 `ffill` 之后，对目标仓位再 **`shift(target_lag_bars)`**（空缺填 `0`）。设为 **`1`** 时，第 `t` 根 K 线使用原序列在 `t−1` 的值。该字段写入 `summary.strategy_params`，并参与 **`data_fingerprint`**（与**滞后后的有效目标**一致）。

### 回测可复现元数据（single / multi）

`run_single_asset_backtest` 与 `run_multi_asset_backtest` 均会在 `summary` 注入审计字段：

- `run_id`：单次运行唯一 ID（每次运行不同）
- `mode`：`single` 或 `multi`
- `data_fingerprint`：对 OHLCV 与目标序列做**结构化统计摘要**后 SHA256；**同逻辑输入应稳定**；**不是**原始文件字节级 hash
- `dependency_versions`：至少包含 `python`、`pandas`、`numpy`、`backtrader`（未安装时为 `null`）
- `git_sha`：当前仓库提交（best-effort，获取失败时为 `null`）
- `signal_timestamp`：信号时间语义标注（当前为 `bar_close_t`）
- `decision_timestamp`：决策时间语义标注（当前为 `bar_close_t`）
- `execution_effective_lag_bars`：收益归因使用的有效滞后 bar 数（single 来自 `target_lag_bars`，multi 来自 `portfolio_weight_lag_bars`）
- `return_attribution`：收益归因公式字符串（如 `weights(t-1) * returns(t)`）
- `execution_engine_requested`：多资产执行层请求内核（来自配置或环境变量）
- `execution_engine_resolved`：当前实际执行内核（`python` / `numpy` / `numba`）


### 真实黄金回测（IBKR）

工业回测建议使用 `BacktestConfig.strict_real_data=True`，该模式下回测器只会从 `data_root` 加载真实 OHLCV，传入 inline `ohlcv` 会直接报错，不存在 synthetic/fallback 路径。

- 数据抓取脚本：各环境自备（如 IBKR 黄金数据脚本）
- 默认落盘目录：通过 `data_root=` 或 YAML 配置，支持 `~/quant_projects/data/...`
- 文件命名兼容：`XAU_1_hour_30_D.parquet`、`XAUUSD_*.parquet` 等（按 `symbol+frequency` 别名自动匹配）

最小配置示例：

```python
from single_asset_backtest.config import BacktestConfig

cfg = BacktestConfig(
    strict_real_data=True,
    data_root="~/quant_projects/data/ibkr",
    symbol="XAUUSD",
    frequency="1h",
    metrics_profile="industrial",
    include_trade_ledger=True,
)
```

运行示例（在 **monorepo 根目录**，且 `PYTHONPATH` 含 `backtest_layer` 与 `factor_engine`，见 [`../../backtest_layer/single_asset_backtest/README.md`](../../backtest_layer/single_asset_backtest/README.md) 文首）：

```bash
python backtest_layer/examples/backtest_single_asset.py
```


**完整列表、DSL 限制（如 `and_`/`or_`/`not_`）与待迁移算子**：见 [`docs/dsl_operators_reference.md`](docs/dsl_operators_reference.md) 与 [`docs/operators_semantics.md`](docs/operators_semantics.md)。

---

## 数据集列表

| 配置文件 | 数据集 | 因子示例 |
|---|---|---|
| `fundamentals_balance_sheet.yaml` | 资产负债表 | `rank(total_assets)` |
| `fundamentals_cash_flow_statement.yaml` | 现金流量表 | `zscore(net_cash_from_operating_activities)` |
| `fundamentals_financials_ratios.yaml` | 财务比率 | `rank(price_to_earnings)` |
| `fundamentals_income_statement.yaml` | 利润表 | `zscore(revenue)` |
| `fundamentals_short_interest.yaml` | 融券兴趣 | `rank(days_to_cover)` |
| `fundamentals_short_volume.yaml` | 融券成交量 | `zscore(short_volume_ratio)` |
| `fundamentals_stocks_floats.yaml` | 流通股 | `zscore(free_float_percent)` |
| `us_stocks_sip_day_aggs_v1.yaml` | 日K线 | `rank(ts_mean(close, 3))` |
| `us_stocks_sip_minute_aggs_v1.yaml` | 分钟K线 | `rank(ts_mean(close, 5))` |
| `us_stocks_sip_quotes_v1.yaml` | 报价 | `zscore(bid_price / ask_price)` |
| `us_stocks_sip_trades_v1.yaml` | 逐笔成交 | `rank(price)` |

详细字段说明见 `docs/massive_parquet_data_dictionary.md`。

---

## 项目结构

```
factor_engine/
│
├── api/                        # 用户接口层（薄封装）
│   ├── columns.py              #   col() 列引用
│   ├── cleaned_ops.py          #   CleanedCall 工厂生成
│   ├── operator_registry.py    #   build_dsl_allowlist() ← cleaned_operators
│   ├── factor.py               #   Factor 数据类
│   └── dsl_parser.py           #   parse_expr / parse_factor
│
├── cleaned_operators/          # 算子 runtime 库（唯一实现源）
│   ├── __init__.py             #   calculate() 分派
│   ├── operators_catalog.md    #   全量 catalog + 状态
│   └── …                     #   按业务域分模块
│
├── expr/                       # 表达式树（仅三种节点）
│   ├── base.py                 #   Expr + 四则 → CleanedCall
│   ├── column.py               #   ColumnRef
│   ├── literal.py              #   Literal
│   └── cleaned_call.py         #   CleanedCall(op, args, kwargs)
│
├── ir/                         # 中间表示层（IR）
│   ├── nodes.py                #   IR 节点定义
│   ├── types.py                #   类型系统
│   ├── schema.py               #   Schema 推导
│   └── analyzer.py             #   Expr → IR 转换与依赖分析
│
├── planner/                    # 编译与规划层
│   ├── logical_plan.py         #   逻辑计划节点（PlanNode）
│   ├── lowerer.py              #   IR → 逻辑计划（Lowerer）
│   ├── optimizer.py            #   逻辑计划优化（常量折叠等）
│   ├── plan_hash.py            #   计划子树结构化哈希（缓存键 / CSE）
│   ├── cse.py                  #   多因子公共子式消除（CSE → plan_ref）
│   ├── rules.py                #   优化规则抽象（Rule）
│   ├── physical_plan.py        #   物理计划
│   └── dag.py                  #   多因子 DAG 计划（DAGPlan / FactorPlan）
│
├── backend/                    # 执行后端
│   ├── pandas_backend.py       #   委托 cleaned_bridge（~70 行）
│   ├── cleaned_bridge.py       #   panel 转换 + kernel 注册
│   ├── polars_backend.py       #   委托 PandasBackend
│   ├── pandas_compat.py        #   可选 Modin
│   ├── debug_backend.py        #   打印计划树
│   ├── context.py              #   ExecutionContext
│   ├── kernels.py              #   KernelRegistry
│   └── factory.py              #   build_backend
│
├── storage/                    # 存储与数据源层
│   ├── datasource.py           #   DataSource 抽象基类
│   ├── kline_parquet_source.py #   KlineParquetSource（K线 parquet）
│   ├── parquet_source.py       #   ParquetSource（通用 parquet）
│   ├── factory.py              #   build_data_source()
│   ├── cache.py                #   CacheManager（列缓存）
│   ├── materializer.py         #   Materializer
│   └── result_store.py         #   ResultStore 抽象
│
├── runtime/                    # 运行时编排层
│   ├── engine.py               #   FactorEngine（compile / run / compile_many / run_many）
│   ├── perf_config.py          #   性能与环境变量（并行、CSE、Modin/Numba 提示）
│   ├── config.py               #   YAML 配置 + load_config（路径经 workspace_paths）
│   ├── exceptions.py           #   FactorEngineError
│   └── real_data_factor_smoke.py # DatasetSpec / MultiParquetSeriesSource / smoke 工具
│
├── workspace_paths.py          #   quant_projects 根 / ~ 展开 / 默认 data 目录
│
├── （回测实现已迁至 monorepo **`../../backtest_layer/single_asset_backtest/`**，示例见 **`../../backtest_layer/examples/`**，测试见 **`../../backtest_layer/tests/test_backtest_*.py`**）
│
├── examples/                   # 示例脚本与配置
│   ├── simple_factor.py        #   最简因子示例
│   ├── pandas_factor.py        #   Pandas 后端示例
│   ├── multi_factor_dag.py     #   多因子 DAG 示例
│   ├── run_factors_joblib.py   #   Joblib 多因子并行示例
│   ├── profile_pandas_backend.py # cProfile 热点（Pandas 路径）
│   ├── bench_pandas_vs_modin.py  # Pandas vs Modin 耗时对比（可选 modin）
│   ├── config_driven_factor.yaml   # 配置驱动示例（K线）
│   ├── notebook_config_smoke.yaml  # Notebook smoke 配置
│   └── configs/                #   11 个数据集的因子配置文件
│       ├── fundamentals_*.yaml
│       └── us_stocks_sip_*.yaml
│
├── tests/                      # 测试套件（回测专项在 monorepo ../../backtest_layer/tests/）
│   ├── test_expr.py            #   表达式树单元测试
│   ├── test_planner.py         #   编译规划测试
│   ├── test_backend.py         #   后端执行测试
│   ├── test_pandas_backend.py  #   Pandas 后端详细测试
│   ├── test_dsl_parser.py      #   DSL 解析器测试
│   ├── test_config_runtime.py  #   配置加载与运行时测试
│   ├── test_end_to_end.py      #   端到端集成测试
│   ├── test_factor_templates.py #  11 类数据集合成数据参数化测试
│   ├── test_cleaned_operators_comprehensive.py  # cleaned 全链路（DSL/IR/执行）
│   ├── test_cleaned_integration.py              # cleaned 集成冒烟
│   ├── test_polars_backend.py  #  Polars 子集对齐（importorskip polars）
│   ├── helpers.py              #  测试共享（如 InMemorySeriesSource）
│   └── test_real_data_factor_smoke.py # 真实 parquet 集成测试（需 RUN_REAL_PARQUET_SMOKE=1）
│
├── docs/
│   ├── README.md                # 文档索引
│   ├── miner_delivery_spec.md   # 投递 JSON 契约（组员必读）
│   ├── 算子与导入教程.md         # factor_engine DSL 写法与 import
│   ├── dsl_operators_reference.md  # 白名单与不可 parse 名
│   ├── dsl_allowlist.json       # 机器可读白名单
│   ├── operators_semantics.md   # 算子语义、DSL 限制
│   ├── canonical_data_fields.md # 字段四层命名
│   ├── adr_backtest_target_position.md
│   ├── adr_trade_when.md
│   ├── changelog_shw.md
│   └── massive_parquet_data_dictionary.md
│
└── factor_generation_process.ipynb  # 因子生成过程演示 Notebook
```

---

## 运行测试

```bash
# 运行全部单元测试（不含真实数据）
cd /path/to/factor_engine
PYTHONPATH=. pytest tests/

# 回测专项测试（位于 monorepo backtest_layer/tests/；需安装 factor-engine[backtest]）
cd /path/to/quantsociety_backend_project
pytest backtest_layer/tests/test_backtest_*.py -q

# 运行真实 parquet 集成测试（需要 massive_parquet 数据集）
RUN_REAL_PARQUET_SMOKE=1 pytest tests/test_real_data_factor_smoke.py -v
```

回测专项测试依赖 **`backtrader`**（`pip install "factor-engine[backtest]"`）；`backtest_layer/tests/conftest.py` 会注入 `PYTHONPATH`，一般无需手写。

---

## 执行流程

```
Factor(expr)
    │
    ▼ Analyzer
  IR Nodes  ─── 依赖列分析 ──▶ 数据源拉取
    │
    ▼ Lowerer
 LogicalPlan (PlanNode)
    │
    ▼ Optimizer
 OptimizedPlan
    │
    ▼ Backend.execute()
 pd.Series (MultiIndex: timestamp × instrument)
```

**多因子**：`compile_many` 可做 **CSE**（重复子式 → `DAGPlan.shared_nodes` + `plan_ref`），**`run_many`** 先算共享子式再算各因子根。YAML 中 `backend.type` 除 `pandas` / `polars` 外还可写 **`pandas_modin`**、**`polars_lazy`**（见上文可选依赖）。