# riskfolio_qs 仓位分析 CLI 使用指南

P0/P1 指标、风险分解、约束审计和能力边界的完整数学定义见
[riskfolio_qs v0.3 功能与数学规范](riskfolio_qs_v0.3_功能与数学规范.md)。

## 1. 能力边界

`analyze-positions` 分析 `riskfolio-qs optimize` 生成的目标仓位产物。

- P0 不访问外部数据，检查仓位、交易、集中度、换手、约束和输出自洽性；
- P1 按优化 manifest 重新加载基准、tradable 和 Barra B/F/D，计算主动仓位、
  因子暴露、风险贡献与可选流动性指标；
- 不计算实际成交、收益、净值和回撤，这些属于 `vectorbt_qs`。

## 2. 最小命令

```powershell
python -m riskfolio_qs analyze-positions `
  --config .\riskfolio_qs\examples\cli\position_analysis.yaml `
  --input-dir .\outputs\my_backtest_result `
  --output-dir .\outputs\my_backtest_result\analysis `
  --overwrite
```

安装 console script 后也可以使用：

```powershell
riskfolio-qs analyze-positions `
  --config .\riskfolio_qs\examples\cli\position_analysis.yaml
```

配置文件中的相对路径相对于配置文件目录解析。命令行覆盖的相对路径相对于当前
工作目录解析。

## 3. data_access 环境

P0 不要求 `data_access` 有可用行情。P1 的 benchmark、tradable 和流动性数据
由 `data_access` 装配，不接受外部替代文件。

如果使用本地完整镜像，需要先按本机实际路径配置数据根目录。例如：

```powershell
$env:ASHARE_PARQUET_ROOT = "D:\quantsociety\whh_local_workspace\lqtp_data"
$env:DATA_ACCESS_SKIP_COS_MIRROR = "1"
```

只有确认本地数据完整时才能设置 `DATA_ACCESS_SKIP_COS_MIRROR=1`。否则应安装并
配置团队 COS CLI，让 `data_access` 自行补齐缺失数据。riskfolio_qs 不会替用户
修改这些环境变量。

## 4. 完整 YAML

可运行模板见
[`examples/cli/position_analysis.yaml`](../examples/cli/position_analysis.yaml)。

### 4.1 `input`

```yaml
input:
  optimization_dir: ../../../outputs/my_backtest_result
```

目录必须至少包含：

- `target_positions.parquet`
- `trades.parquet`
- `summary.parquet`
- `metadata.json`
- `run_manifest.yaml`

`resolved_cli_config.yaml` 和 `resolved_params.yaml` 强烈建议保留。缺少它们时，
部分 P1 数据路由和约束边界无法恢复。

### 4.2 `validation`

```yaml
validation:
  reconstruction_tolerance: 1.0e-10
  comparison_tolerance: 1.0e-8
  binding_tolerance: 1.0e-6
  provenance_policy: strict
  allow_partial_enrichment: true
```

| 字段 | 含义 |
|---|---|
| `reconstruction_tolerance` | `target - previous - trade` 重建容差 |
| `comparison_tolerance` | 优化器上报值与分析器独立重算值的比较容差 |
| `binding_tolerance` | 判断约束是否贴边的容差 |
| `provenance_policy` | `strict` 时 data_access snapshot 不一致则停止对应 enrichment；`warn` 时继续但标记 partial |
| `allow_partial_enrichment` | 是否接受部分 P1 数据不可用；允许时 CLI 对 partial 返回退出码 2 |

首期初始仓位由 `first_target - first_trade` 重建。由于没有第二份独立初始仓位，
首期交易本身无法自证；第二期以后会严格检查交易串联关系。

### 4.3 `position`

```yaml
position:
  weight_epsilon: 1.0e-8
  topk: [5, 10, 20]
  include_full_universe: true
  concentration: true
  turnover: true
  holding_persistence: true
```

换手口径为：

```text
one_way_turnover = 0.5 * sum(abs(delta_weight))
gross_traded_weight = sum(abs(delta_weight))
```

当前优化器 `summary.turnover` 使用前者。

### 4.4 `benchmark`

```yaml
benchmark:
  mode: auto
  source: manifest
```

- `auto`：优化 manifest 声明 benchmark 时加载；
- `required`：缺失或加载失败时分析失败；
- `off`：不加载；
- `source` 当前只允许 `manifest`。

重新读取后会比较 optimizer manifest 中的 data_access `snapshot_id`。
基准 P1 同时从 `ashare_stock_industry` 的 `sw_l1` 口径和
`ashare_stock_valuation_daily.MarketCap` 装配行业分类与市值暴露；它们使用
各决策日可得的最近截面，并分别记录 data_access provenance。

### 4.5 `barra`

```yaml
barra:
  mode: auto
  source: manifest
  analyze_absolute_risk: true
  analyze_active_risk: true
```

分析器从优化 manifest 记录的 Barra risk package 读取 exposure、factor covariance
和 specific variance，并遵守 `summary.parquet` 中记录的 exposure/covariance/
specific-risk 日期。

如果旧优化 manifest 只保存 Barra 路径、没有保存内容 hash，分析器会记录本次
package hash，但只能标记为 `path_only`。此时发现风险值与优化器原上报值不一致，
结果会成为 `partial`，不会用新值覆盖旧值。

### 4.6 `liquidity`

```yaml
liquidity:
  mode: required
  portfolio_notional: 100000000
  adv_window_days: 20
  maximum_adv_participation: 0.10
```

只有提供正数 `portfolio_notional` 才计算：

- `trade_notional`
- `adv`
- `adv_participation`
- `estimated_days`

ADV 窗口严格使用决策日前的数据，不使用当日或未来成交额。

### 4.7 `output`

```yaml
output:
  directory: ../../../outputs/my_backtest_result/analysis
  parquet: true
  overwrite: false
```

默认不覆盖已有分析产物。`--overwrite` 只会删除并重写分析模块拥有的已知文件，
不会修改父目录中的优化产物。

## 5. 输出文件

```text
analysis/
├── position_summary.parquet
├── holdings_detail.parquet
├── turnover_detail.parquet
├── exposure_summary.parquet
├── risk_contribution.parquet
├── constraint_summary.parquet
├── quality_checks.parquet
├── resolved_analysis_config.yaml
├── analysis_manifest.yaml
└── position_report.html
```

最重要的验收项：

1. `analysis_manifest.yaml.status`；
2. `quality_checks.parquet` 中非 passed 的记录；
3. reported/recomputed 差异；
4. benchmark 和 tradable 的 `snapshot_match`；
5. Barra asset contribution 是否加总为 total variance；
6. `constraint_summary.parquet` 是否存在 violation。

## 6. 状态与退出码

| 状态 | 退出码 | 含义 |
|---|---:|---|
| `passed` | 0 | 核心检查和启用的 enrichment 均通过 |
| `partial` | 2 | 配置允许 partial，但存在数据不可用、snapshot 告警或复算差异 |
| `partial` | 1 | 配置不允许 partial |
| `failed` | 1 | 核心契约、required enrichment 或风险加总失败 |

PowerShell 可用 `$LASTEXITCODE` 获取退出码。

## 7. Python API

目录模式：

```python
from riskfolio_qs.analysis import analyze_position_output

result = analyze_position_output(
    config="riskfolio_qs/examples/cli/position_analysis.yaml",
    input_dir="outputs/my_backtest_result",
    output_dir="outputs/my_backtest_result/analysis",
    overwrite=True,
)

print(result.status)
print(result.position_summary)
print(result.quality_checks)
```

内存模式：

```python
from riskfolio_qs.analysis import analyze_output_bundle

result = analyze_output_bundle(
    output_bundle,
    config={
        "analysis_version": 1,
        "benchmark": {"mode": "off"},
        "barra": {"mode": "off"},
        "liquidity": {"mode": "off"},
    },
)
```

内存模式默认不写文件，并允许调用者通过 `AnalysisEnrichment` 注入带 provenance
的 P1 数据。
