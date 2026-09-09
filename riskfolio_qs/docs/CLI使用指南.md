# riskfolio_qs CLI 使用指南

优化器、风险模型、成本、约束和输出字段的完整数学定义见
[riskfolio_qs v0.3 功能与数学规范](riskfolio_qs_v0.3_功能与数学规范.md)。

本文对应 `riskfolio_qs 0.3.0`。当前 CLI 会先解析优化器，再按其强依赖装配输入：

- Alpha：调用方提供 Parquet/CSV；
- benchmark：仅指增模式从 `data_access/ashare_index_constituent` 读取；
- tradable：仅凸优化模式从 `data_access/ashare_stock_daily` 读取；
- 初始持仓：指增使用首日 benchmark，绝对收益从现金开始，TopN 不要求初始持仓；
- 后续持仓：自动使用上一回测日的优化目标权重；
- 线性成本：代码内固定为单边 `5 bps`；
- 风险：按模式使用预计算 Barra、`data_access` 历史收益，或不加载风险数据。

CLI 不再接受 benchmark、previous positions、tradable、linear cost 或 impact
cost 文件。这是有意的单一回测链路，不保留旧文件接口兼容。

## 1. 能力边界

正式支持：

- `adapter.type: auto | barra_precomputed | data_access`
- `meanvar_enhance_barra_precomputed`
- `minvar_enhance_barra_precomputed`
- `meanvar_enhance_hist`
- `minvar_enhance_hist`
- `meanvar_absolute_return`
- `topn_long_only_equal_weight`
- `scenario: index_enhancement`
- `scenario: conservative_index_enhancement`
- `scenario: absolute_return`
- `scenario: fallback`
- Alpha 的 long/wide Parquet 或 CSV
- `riskfolio-qs optimize` 与 `python -m riskfolio_qs optimize`

输入装配规则：

| 风险模式 | 自动读取 |
|---|---|
| `barra_precomputed` | benchmark、tradable、预计算 Barra B/F/D |
| `historical_cov` 指增 | benchmark、tradable、带 warm-up 的个股日收益 |
| `historical_cov` 绝对收益 | tradable、带 warm-up 的个股日收益 |
| `none`（TopN） | 只读取 Alpha，不访问 `data_access` 或 Barra |

`StockDailyBar.Return` 会在适配层除以 `10000` 转成小数收益，优化器直接计算
收益协方差，不会再次执行 `pct_change()`。CLI 不会生成真实订单或模拟成交偏差。
输出的是目标权重和权重变化。

## 2. 安装

`riskfolio_qs` 依赖工作区内部包 `data_access`。在当前工作区推荐：

```powershell
cd D:\quantsociety\whh_local_workspace
python -m pip install -e .\data_access
python -m pip install -e ".\riskfolio_qs[dev]"
python -m pip check
python -m riskfolio_qs --version
```

wheel 部署时，目标环境必须能从内部包源解析 `data_access>=0.1.0`。

`data_access` 的数据根目录由其自身环境和注册表管理。当前 A 股本地环境通常需要：

```powershell
$env:ASHARE_PARQUET_ROOT = "D:\quantsociety\whh_local_workspace\lqtp_data"
$env:DATA_ACCESS_SKIP_COS_MIRROR = "1"
```

不要在 riskfolio 配置中填写行情文件路径。数据集名称和读取规则由代码固定。

## 3. 快速测试

在工作区根目录执行：

```powershell
$env:DATA_ACCESS_SKIP_COS_MIRROR = "1"
$env:ASHARE_PARQUET_ROOT = "D:\quantsociety\whh_local_workspace\lqtp_data"

python -m riskfolio_qs optimize `
  --config .\riskfolio_qs\validation\ridge_h5_all_optimizers\cli_smoke.yaml `
  --output-dir .\outputs\my_backtest_result `
  --overwrite
```

正常输出类似：

```text
status=passed
optimizer=meanvar_enhance_barra_precomputed
dates=5
assets=488
max_constraint_violation=3.099e-07
output_dir=D:\quantsociety\whh_local_workspace\outputs\my_backtest_result
```

这套 smoke 配置按正式回测规则从首日 benchmark 初始化，随后逐日串联优化权重。
求解器版本和容差可能导致最后几位不同，不应把示例中的违例数值当作逐位匹配标准。

另外提供三份可直接运行的模式配置：

```powershell
# data_access 历史协方差指增
python -m riskfolio_qs optimize `
  --config .\riskfolio_qs\validation\ridge_h5_all_optimizers\cli_historical_smoke.yaml `
  --output-dir .\outputs\historical_result --overwrite

# data_access 历史协方差绝对收益（不需要 benchmark）
python -m riskfolio_qs optimize `
  --config .\riskfolio_qs\validation\ridge_h5_all_optimizers\cli_absolute_smoke.yaml `
  --output-dir .\outputs\absolute_result --overwrite

# Alpha-only TopN（不需要 data_access 环境变量）
python -m riskfolio_qs optimize `
  --config .\riskfolio_qs\validation\ridge_h5_all_optimizers\cli_topn_smoke.yaml `
  --output-dir .\outputs\topn_result --overwrite
```

`meanvar_enhance_index` 与 `minvar_enhance_index` 是旧的原始 Barra 名称。CLI 会将它们
分别路由到同目标的 precomputed 优化器，并在 `run_manifest.yaml.inputs.routing`
记录请求名称和实际优化器，避免重新估计一套较弱的残差风险模型。

## 4. 命令行

```text
riskfolio-qs optimize [-h] -c CONFIG [--alpha ALPHA]
                      [-o OUTPUT_DIR] [--overwrite] [--debug]
```

| 参数 | 必填 | 说明 |
|---|---:|---|
| `-c, --config` | 是 | YAML 配置文件 |
| `--alpha` | 否 | 覆盖 `inputs.alpha.path` |
| `-o, --output-dir` | 否 | 覆盖 `output.directory` |
| `--overwrite` | 否 | 覆盖输出目录中的标准 CLI 产物 |
| `--debug` | 否 | 失败时显示完整 traceback |
| `--version` | 否 | 显示版本 |

YAML 内相对路径相对于 YAML 所在目录；命令行覆盖路径相对于当前终端目录。

## 5. 完整配置

```yaml
config_version: 1

adapter:
  type: barra_precomputed
  risk_root: D:/risk_data/v1_sbi_fullA
  benchmark_index: 000905.SH
  minimum_benchmark_weight_coverage: 0.95
  alpha_input_type: expected_return
  alpha_horizon_days: 5
  calendar_name: XSHG
  risk_data_lag_periods: 1
  exposure_data_lag_periods: 1
  annualization_factor: 252
  load_factor_returns: false

inputs:
  alpha:
    path: D:/signals/Ridge_h5_pred.parquet
    layout: long
    date_column: TradeDate
    asset_column: Symbol
    value_column: pred
    start_date: 2025-12-24
    end_date: 2025-12-24

smoother:
  mode: never

run:
  optimizer_name: meanvar_enhance_barra_precomputed
  data_version_hash: ridge_h5_20251224
  allow_fallback: false
  runtime_overrides:
    tracking_error_cap_annual: 0.0295
    turnover_cap: 0.195
    single_name_max: 0.03
    active_weight_abs_max: 0.0195
    style_band: 0.095

output:
  directory: D:/portfolio_outputs/2025-12-24
  overwrite: false
```

配置出现未知字段会直接失败。旧版的以下字段现在属于非法配置：

```yaml
inputs:
  benchmark: ...
  previous_positions: ...
  tradable: ...
  linear_cost_bps: ...
  impact_cost: ...
```

### 5.1 按模式选择 `adapter`

| `run.optimizer_name` | `adapter.type` | `risk_root` | `benchmark_index` | 自动读取 |
|---|---|---:|---:|---|
| `meanvar_enhance_barra_precomputed` | `auto` 或 `barra_precomputed` | 必填 | 必填 | benchmark + tradable + B/F/D |
| `minvar_enhance_barra_precomputed` | `auto` 或 `barra_precomputed` | 必填 | 必填 | benchmark + tradable + B/F/D |
| `meanvar_enhance_hist` | `auto` 或 `data_access` | 不需要 | 必填 | Return + calendar + benchmark + tradable |
| `minvar_enhance_hist` | `auto` 或 `data_access` | 不需要 | 必填 | Return + calendar + benchmark + tradable |
| `meanvar_absolute_return` | `auto` 或 `data_access` | 不需要 | 不需要 | Return + calendar + tradable |
| `topn_long_only_equal_weight` | 任意 | 不需要 | 不需要 | 只读取 Alpha |

`adapter.type` 表示允许的数据来源，不再决定唯一优化器。CLI 始终先解析
`optimizer_name`/`scenario`，然后只装配该优化器的强依赖。设置
`type: data_access` 时会显式禁止 precomputed Barra，避免配置误用；TopN
无论填写哪种 type 都不会访问风险数据。

### 5.2 历史指增最小 YAML

```yaml
config_version: 1
adapter:
  type: data_access
  benchmark_index: 000905.SH
  minimum_benchmark_weight_coverage: 0.95
  alpha_input_type: expected_return
  alpha_horizon_days: 5
  risk_data_lag_periods: 1
inputs:
  alpha:
    path: alpha.parquet
    layout: wide
run:
  optimizer_name: meanvar_enhance_hist
  allow_fallback: false
output:
  directory: historical_output
  overwrite: false
```

`hist_cov_lookback_days` 来自冻结参数文件；CLI 会自动多读 warm-up，不要求 Alpha
文件包含这段历史。行情字段固定为 `StockDailyBar.Return / 10000`。

### 5.3 绝对收益最小 YAML

```yaml
config_version: 1
adapter:
  type: data_access
  alpha_input_type: expected_return
  alpha_horizon_days: 5
  risk_data_lag_periods: 1
inputs:
  alpha:
    path: alpha.parquet
    layout: wide
run:
  optimizer_name: meanvar_absolute_return
  allow_fallback: false
output:
  directory: absolute_output
```

该模式不读取 benchmark，回测首期按全零仓位（现金）初始化，后续日期使用上一期
优化权重。

### 5.4 TopN 最小 YAML

```yaml
config_version: 1
adapter:
  type: auto
inputs:
  alpha:
    path: alpha.parquet
    layout: wide
run:
  optimizer_name: topn_long_only_equal_weight
output:
  directory: topn_output
```

TopN 的持股数量由冻结参数 `topn_n` 控制；当前它不在 runtime override
白名单内。TopN 不加载 benchmark、tradable、历史行情或 Barra，第一期交易量
按全零仓位计算。

## 6. 配置字段

### 6.1 `adapter`

| 字段 | 默认值 | 说明 |
|---|---:|---|
| `type` | `auto` | `auto`、`barra_precomputed` 或 `data_access` |
| `risk_root` | 无 | 仅预计算 Barra 模式必填；历史与 TopN 模式忽略 |
| `benchmark_index` | 无 | 指增模式必填；绝对收益与 TopN 不需要 |
| `minimum_benchmark_weight_coverage` | `0.95` | Alpha 股票池至少覆盖的指数权重 |
| `alpha_input_type` | `score` | `score` 或 `expected_return` |
| `alpha_horizon_days` | `1` | Alpha 预测交易日数 |
| `calendar_name` | `XSHG` | 审计用交易日历名称 |
| `risk_data_lag_periods` | `0` | F/D 风险快照或历史收益滞后期数 |
| `exposure_data_lag_periods` | `0` | B 暴露快照滞后期数 |
| `annualization_factor` | `252` | 年化期数 |
| `load_factor_returns` | `false` | 是否读取可选 factor returns |
| `minimum_annual_specific_volatility` | `0.05` | 特异风险质量门槛 |

`benchmark_index` 是业务选择，不是数据源配置。CLI 会对目标指数全部成分权重求和，
再计算 Alpha 股票池保留的权重比例。低于覆盖率门槛时直接失败；通过后仅在 Alpha
股票池内归一化，最终每行 benchmark 权重和为 1。

### 6.2 `inputs`

只允许 `alpha`。Alpha 输入支持：

```yaml
path: 文件路径
layout: long | wide
date_column: 日期列
asset_column: 资产列
value_column: 值列
start_date: 起始日期
end_date: 结束日期
```

long 布局必须指定 `value_column`；wide 文件若未保留 `DatetimeIndex`，必须指定
`date_column`。日期过滤为闭区间。

### 6.3 `smoother`

```yaml
smoother:
  mode: never
  topn_n: 50
  turnover_threshold: 0.30
  turnover_window: 5
  ema_span: 5
```

`mode` 可为 `never`、`always`、`auto`。

### 6.4 `run`

`optimizer_name` 与 `scenario` 互斥；都不写时默认
`scenario: index_enhancement`。

| `scenario` | 实际默认优化器 |
|---|---|
| `index_enhancement` | `meanvar_enhance_barra_precomputed` |
| `conservative_index_enhancement` | `minvar_enhance_barra_precomputed` |
| `absolute_return` | `meanvar_absolute_return` |
| `fallback` | `topn_long_only_equal_weight` |

场景路由不会在缺数据时静默切换到另一种风险模型。缺少强依赖会直接失败；只有优化器
映射显式声明 fallback 且 `allow_fallback: true` 时才接受降级。

常用硬约束覆盖：

- `tracking_error_cap_annual`
- `turnover_cap`
- `single_name_max`
- `active_weight_abs_max`
- `style_band`
- `mcap_band`

常用目标和求解设置：

- `alpha_weight`
- `risk_weight`
- `turnover_penalty`
- `solver_name`
- `max_iters`
- `solver_tol`
- `on_solve_failure`

最终白名单和值域以包内 `parameters.v0.2.0.yaml` 为准。

`allow_fallback: false` 表示任何降级都令 CLI 失败。正式回测建议保持 `false`。

### 6.5 `output`

```yaml
output:
  directory: output
  overwrite: false
```

`--output-dir` 优先于配置。已有标准产物时默认拒绝覆盖；`--overwrite` 只覆盖已知
CLI 文件，不递归清空目录。

## 7. data_access 规则

### 7.1 Benchmark

固定读取：

```text
dataset = ashare_index_constituent
columns = TradeDate, IndexSymbol, Symbol, Weight
```

处理顺序：

1. 按 Alpha 日期范围读取；
2. 筛选 `adapter.benchmark_index`；
3. 校验每个日期存在且权重非负、有限；
4. 计算 Alpha 股票池的指数权重覆盖率；
5. 覆盖率不足则失败；
6. 在 Alpha 股票池中补零并归一化。

数据 snapshot ID、registry hash、文件 manifest hash 和逐日覆盖率会写入
`run_manifest.yaml`。

### 7.2 Tradable

固定读取：

```text
dataset = ashare_stock_daily
columns = TradeDate, Symbol, Volume, Amount
rule = Volume > 0 and Amount > 0
```

行情缺行、成交量为零或成交额为零都视为不可交易。不可交易表示该资产目标权重冻结
在上一期持仓。当前规则不包含实时涨跌停价、风控黑名单或账户级下单限制，因此 CLI
结果仍需经过交易执行系统的二次校验。

### 7.3 成本

CLI 固定构造每个日期、每个资产单边 `5 bps` 线性成本。当前回测 CLI 不接收外部
冲击成本矩阵，因此也拒绝在 `run.runtime_overrides` 中设置
`default_linear_cost_bps`、`default_impact_cost`、`impact_cost_penalty` 或
`enable_impact_cost`，避免出现配置存在但输入不完整的假生效状态。

### 7.4 历史收益

历史协方差模式固定读取：

```text
dataset = ashare_stock_daily
columns = TradeDate, Symbol, Return
scale = Return / 10000
```

CLI 根据冻结参数中的 `hist_cov_lookback_days` 和
`risk_data_lag_periods`，先从 `ashare_calendar` 确定 warm-up 起点，再读取
Alpha 股票池的历史收益。收益矩阵保留停牌、上市前等真实缺失，不做全表填零；
协方差估计器使用 pairwise 样本、方差回退、收缩和 PSD 投影处理覆盖差异。
行情及日历的 snapshot、读取区间、缩放比例和非空覆盖会写入
`run_manifest.yaml`。

### 7.5 回测持仓如何串联

不同模式的首期仓位如下：

| 模式 | 首期仓位 |
|---|---|
| precomputed / 历史指增 | 首个 Alpha 日归一化 benchmark |
| 绝对收益 | 全零仓位（现金） |
| TopN | 全零仓位 |

指增模式第一日交易量为：

```text
第一日目标权重 - 第一日 benchmark 权重
```

第二日及以后使用上一回测日的优化目标权重：

```text
当日交易量 = 当日目标权重 - 上一回测日目标权重
```

因此换手、成本和不可交易冻结都有明确基准，同时不需要持仓文件。这个假设只适用于
当前回测 CLI；将来接入实盘时必须改为账户实际成交后持仓，不能沿用目标权重。

## 8. Alpha 契约

### 8.1 Alpha long 格式

| TradeDate | Symbol | pred |
|---|---|---:|
| 2025-12-24 | 000001.SZ | 0.012 |
| 2025-12-24 | 600000.SH | -0.003 |

同一 `(date, asset)` 不得重复。

`alpha_input_type: expected_return` 时，`0.01` 表示预测期内 1% 收益；
`alpha_horizon_days` 必须对应信号预测期限。

### 8.2 回测日期序列

Alpha 可以包含一个或多个日期。多日输入会在一次运行内按日期升序求解并自动串联。
单日输入每次都会重新初始化：指增从当日 benchmark 开始，绝对收益和 TopN 从全零
仓位开始。因此不要用多个独立单日调用替代一次连续多日回测，否则会丢失跨日换手
和成本。

## 9. Barra 风险包

`risk_root` 至少包含：

```text
manifest.yaml
exposure.parquet
factor_cov.parquet
specific_risk.parquet
```

manifest 必须声明风险单位：

```yaml
product: barra_lite
estimation:
  units: daily_variance
files:
  exposure: exposure.parquet
  factor_cov: factor_cov.parquet
  specific_risk: specific_risk.parquet
```

允许 `daily_variance`、`horizon_variance`、`annual_variance`。`specific_var`
是方差，不是波动率。B/F/D 必须覆盖本次 Alpha 资产和所需 as-of 日期。

## 10. 输出与审计

| 文件 | 用途 |
|---|---|
| `target_positions.parquet` | `date × asset` 目标权重 |
| `trades.parquet` | `(date, asset)` 权重变化 |
| `summary.parquet` | 每日风险、成本、约束和求解诊断 |
| `metadata.json` | 优化器、fallback、时点和参数审计 |
| `run_manifest.yaml` | 输入来源、data_access 快照和产物清单 |
| `resolved_cli_config.yaml` | 命令行覆盖后的配置 |
| `resolved_mapping.yaml` | 实际优化器映射 |
| `resolved_params.yaml` | 实际冻结参数与 runtime override |

`run_manifest.yaml` 中：

- Alpha、Barra 风险包记录文件来源；
- 初始仓位记录 `backtest_starts_from_benchmark` 假设；
- benchmark、tradable 记录 `data_access` 数据集和 snapshot；
- 线性成本记录内部常数来源；
- benchmark 记录逐日股票池权重覆盖率。

## 11. 结果验收

```python
from pathlib import Path
import pandas as pd

root = Path(r"D:\portfolio_outputs\2025-12-24")
weights = pd.read_parquet(root / "target_positions.parquet")
summary = pd.read_parquet(root / "summary.parquet")

assert summary["solve_status"].isin(["optimal", "optimal_inaccurate"]).all()
assert summary["feasible_flag"].all()
assert summary["fallback_used"].eq(False).all()
assert summary["max_constraint_violation"].max() <= 1e-5
assert summary["tracking_error_cap_violation"].max() <= 1e-5
assert abs(weights.sum(axis=1) - 1.0).max() <= 1e-5
```

必须同时检查 `trades.parquet` 的持仓串联：

```python
trades = (
    pd.read_parquet(root / "trades.parquet")["delta_weight"]
    .unstack("asset")
    .reindex(index=weights.index, columns=weights.columns)
)

# 第二日以后：交易差额必须等于相邻目标仓位之差。
assert (
    trades.iloc[1:] - weights.diff().iloc[1:]
).abs().to_numpy().max() <= 1e-10
```

还应检查：

- `predicted_tracking_error_annual`
- `tracking_error_cap_slack`
- `turnover` / `turnover_slack`
- `active_share`
- `max_abs_industry_active_exposure`
- `max_abs_style_active_exposure`
- `expected_linear_cost`
- `risk_exposure_date`
- `risk_covariance_date`
- `specific_risk_date`

### 11.1 如何判断“贴约束”

以下情况表示约束正在主导结果，但本身不是求解错误：

- `turnover_slack` 接近 0：换手上限已用满；
- `tracking_error_cap_slack` 接近 0：TE 上限已用满；
- `max_abs_style_active_exposure` 接近 `style_band`：风格带宽已用满；
- `active_share` 持续上升：组合相对指数越来越主动。

如果多个约束长期同时贴边，应做参数敏感性分析；否则微小的输入变化可能引起较大的
持仓变化。

### 11.2 风险模型数值诊断

`covariance_condition_number` 很大表示因子协方差接近奇异。当前预计算 Barra 路径
使用因子形式和 PSD 修复，条件数大不会自动使结果不合格，但它是风险估计敏感性的
警告。建议持续观察：

- `covariance_min_eigenvalue` 是否为正；
- `covariance_condition_number` 是否突然恶化；
- TE 和风险贡献是否对日期或参数微调异常敏感。

### 11.3 防止时点前视

`risk_exposure_date`、`risk_covariance_date` 和 `specific_risk_date` 必须符合回测
决策时点。`lag=0` 会使用同日风险快照；只有该快照在决策时已经生成且未使用决策后
数据时才合格，否则应设置 `risk_data_lag_periods: 1` 和
`exposure_data_lag_periods: 1`。

## 12. 常见失败

### `No module named data_access` 或 `No module named duckdb`

```powershell
python -m pip install -e .\data_access
python -m pip check
```

必须使用执行 CLI 的同一个 Python 环境安装。

### data_access 找不到 A 股数据

检查 `ASHARE_PARQUET_ROOT` 和 `data_access/config/datasets.yaml`。不要把文件路径
重新放回 riskfolio YAML。

### benchmark coverage 不足

Alpha 股票池没有覆盖足够的指数权重。应修复 Alpha universe 或选择正确指数；
不要简单降低阈值掩盖代码体系或股票池错误。

### tradable 大量为 False

检查 Alpha 日期是否为交易日、证券代码是否与 `ashare_stock_daily.Symbol` 一致，
以及对应日期行情是否完整。

### 风险快照不可用

检查 `risk_data_lag_periods`、`exposure_data_lag_periods` 和风险包日期。只有同日风险
数据在决策前确实可用时才可设置 `lag=0`。

### 输出目录已有产物

确认允许覆盖后增加 `--overwrite`。该参数不会删除目录中的其他文件。

### 定位完整异常

```powershell
python -m riskfolio_qs optimize -c run.yaml --debug
```

CLI 退出码：成功为 `0`，运行失败为 `1`，参数语法错误为 `2`。
