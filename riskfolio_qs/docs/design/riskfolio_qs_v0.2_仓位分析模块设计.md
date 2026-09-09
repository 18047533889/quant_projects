# riskfolio_qs 仓位分析模块设计 v0.1

## 1. 文档状态

- 状态：P0/P1 已实现，P2 待实现
- 目标版本：riskfolio_qs v0.3
- 适用对象：`OutputBundle`、`riskfolio-qs optimize` 产物目录
- 核心原则：分析目标仓位，不伪装成回测或实盘成交分析

本文定义仓位分析模块的职责、数据契约、指标口径、CLI/YAML 协议、
结构化产物和验收标准。实现阶段必须遵守本文的时点、降级和审计规则。

## 2. 目标与边界

### 2.1 目标

仓位分析模块需要回答以下问题：

1. 优化产物是否完整、自洽、可被下游使用；
2. 每期组合持有什么、集中在哪里、相对上期如何变化；
3. 相对基准超配或低配了什么；
4. 行业、风格、市值和 Barra 风险集中在哪里；
5. 哪些硬约束接近边界，哪些指标与优化器上报值不一致；
6. 在给定组合规模后，交易是否可能受到流动性约束；
7. 所有结论使用了哪一版输入数据，是否可复现。

### 2.2 非目标

本模块不负责：

1. 重新求解组合；
2. 根据目标仓位推断实际成交；
3. 计算真实收益、净值、回撤、滑点或成交成本；
4. 代替 `vectorbt_qs` 的回测与执行分析；
5. 在输入缺失时静默填充基准、行业、因子暴露或成交数据。

`target_positions.parquet` 表示优化器希望持有的目标权重，不等于实际持仓。
真实持仓和目标持仓的偏离属于 `vectorbt_qs` 或实盘执行层。

## 3. 核心术语

| 术语 | 定义 |
|---|---|
| 目标仓位 | 优化器在决策日输出的资产权重 `target_positions` |
| 上期目标仓位 | 同一次连续优化中，上一决策日的目标仓位 |
| 初始仓位 | 第一个决策日前已有仓位；可由首期目标仓位减首期交易量重建 |
| 交易权重 | `delta_weight = target_weight - previous_weight` |
| 实际仓位 | 执行、停牌、涨跌停、成交量和撮合后真实持有的仓位；本模块 P0/P1 不拥有 |
| 基准仓位 | 决策日对应指数成分权重 |
| 主动仓位 | `active_weight = target_weight - benchmark_weight` |
| 决策日 | 目标仓位生效所对应的组合决策日期 |
| 数据时点 | 某类风险或暴露输入实际使用的 as-of 日期 |
| 上报指标 | 优化器写入 `summary.parquet` 的诊断值 |
| 重算指标 | 分析器仅根据落盘产物或重新装配的数据独立计算的值 |

## 4. 分层架构

```text
OutputBundle / optimize 产物目录
                |
                v
       ArtifactLoader + Validator
                |
        +-------+--------+
        |                |
        v                v
  Core Position      Enrichment
  Metrics (P0)       (P1, 可独立失败)
                         |
              +----------+----------+
              |          |          |
          benchmark   data_access   Barra B/F/D
              |          |          |
              +----------+----------+
                         |
                         v
              Structured Analysis Result
                         |
               +---------+---------+
               |                   |
             Parquet              HTML
```

依赖方向必须是：

```text
analysis -> core contracts / adapters
analysis -> data_access（仅 enrichment）
analysis -X-> optimizer implementation
optimizer -X-> analysis
```

分析模块不能调用优化器内部的私有计算函数重算指标，否则无法形成独立审计。

## 5. 建议代码结构

```text
src/riskfolio_qs/analysis/
├── __init__.py
├── contracts.py
├── artifact_loader.py
├── position_analyzer.py
├── report.py
├── enrichers/
│   ├── benchmark.py
│   ├── market.py
│   └── barra.py
└── metrics/
    ├── integrity.py
    ├── concentration.py
    ├── turnover.py
    ├── active.py
    ├── exposure.py
    ├── risk.py
    └── liquidity.py
```

首期实现应保持模块纯函数化。HTML 渲染只能消费结构化分析结果，不能在模板中
临时重新计算业务指标。

## 6. 输入契约

### 6.1 核心产物

分析器必须支持两种等价入口：

1. 内存中的 `OutputBundle`；
2. `riskfolio-qs optimize` 生成的产物目录。

目录入口的核心文件如下：

| 文件 | 是否必需 | 用途 |
|---|---:|---|
| `target_positions.parquet` | 是 | 目标仓位宽表，`date × asset` |
| `trades.parquet` | 是 | 交易明细，索引为 `(date, asset)`，列为 `delta_weight` |
| `summary.parquet` | 是 | 求解、风险、约束和组合级上报指标 |
| `metadata.json` | 是 | 优化器、求解器、版本和 fallback 审计信息 |
| `run_manifest.yaml` | 是 | 输入来源、数据快照、状态和产物清单 |
| `resolved_cli_config.yaml` | 建议 | 恢复 benchmark、Barra、时点和运行配置 |

如果通过 Python 直接传入 `OutputBundle`，P0 可以完整运行。需要 P1 enrichment 时，
调用者还必须传入等价的运行上下文或显式构造 `AnalysisEnrichment`。

### 6.2 核心校验

加载器必须执行以下检查：

1. 日期、资产代码和 MultiIndex 名称标准化；
2. 日期严格递增、资产列唯一、交易索引无重复；
3. 所有权重和交易量为有限数值；
4. `target_positions`、`trades` 和 `summary` 的日期集合一致；
5. 每个 `date × asset` 都有唯一交易量；
6. 交易重建关系在容差内成立；
7. manifest 中的日期数、资产数和实际文件一致；
8. manifest 状态、fallback 和 solve status 被原样保留；
9. 不因权重为零而删除资产，以免掩盖 universe 或交易缺口。

交易重建定义为：

$$
\hat w_{t-1} =
\begin{cases}
w_t - \Delta w_t, & t = t_0 \\
w_{t-1}, & t > t_0
\end{cases}
$$

并验证：

$$
\left\|w_t - \hat w_{t-1} - \Delta w_t\right\|_\infty
\le \epsilon_{\text{reconstruction}}
$$

首期的 `w_t - Δw_t` 是优化运行的初始仓位。它不一定是现金仓位：
指数增强通常从基准初始化，绝对收益模式通常从零仓位初始化。

### 6.3 Enrichment 输入

P1 enrichment 按以下优先级装配：

1. 优化产物 `run_manifest.yaml` 和 `resolved_cli_config.yaml` 声明的来源；
2. Python 调用显式传入的、带 provenance 的 `AnalysisEnrichment`；
3. 不允许通过猜测资产、指数或“最新文件”完成装配。

CLI 不接受外部 benchmark/tradable 文件路径。基准、交易状态和行情仍从团队
`data_access` 加载；Barra 预计算数据从优化运行记录的 risk package 加载。

## 7. 数据时点与可复现性

### 7.1 Point-in-time 规则

1. 基准权重使用对应决策日的指数权重；
2. 行业、风格、因子暴露使用优化器实际采用的 exposure date；
3. 因子协方差使用 `risk_covariance_date`；
4. 特异方差使用 `specific_risk_date`；
5. 不允许为方便分析而使用当前最新 Barra 截面替换历史截面；
6. 流动性分析只使用决策时已知的滚动窗口数据，不使用未来成交额。

优先从 `summary.parquet` 读取：

- `risk_exposure_date`
- `risk_covariance_date`
- `specific_risk_date`

缺失时只能按照已记录的 lag 配置重新推导，并在质量表中标记
`asof_date_derived=true`。

### 7.2 data_access snapshot 约束

当前 `data_access.read_result()` 会返回 snapshot identity，包括：

- `snapshot_id`
- `registry_hash`
- `schema_hash`
- `file_manifest_hash`

但当前接口不支持按历史 `snapshot_id` 直接回读。因此分析器的正确行为是：

1. 按原 dataset、时间范围、资产集合和参数重新读取；
2. 将新 snapshot 与优化 manifest 记录的 snapshot 比较；
3. `strict` 策略下不一致即终止对应 enrichment；
4. `warn` 策略下允许继续，但结果状态必须为 `partial`，所有相关表写入
   `snapshot_match=false`；
5. 不允许将新 snapshot 冒充原优化 snapshot。

这意味着当前体系可以检测数据漂移，但在上游未保留旧文件时，不能保证回读旧版本。
完整可复现能力需要 `data_access` 后续增加 snapshot pinning 或 immutable dataset
version。

### 7.3 Barra provenance

Barra enrichment 必须读取优化 manifest 记录的 risk package 路径，并校验：

1. 包内 manifest；
2. factor 名称和顺序；
3. 资产、日期覆盖率；
4. exposure、factor covariance、specific variance 的对应日期；
5. 可用时校验文件 checksum 或内容 hash。

如果优化产物只记录了路径而没有 hash，结果只能标记为
`provenance_status=path_only`，不能声称已验证与求解时完全一致。

## 8. Python 公共接口

建议公开两个入口：

```python
from riskfolio_qs.analysis import PositionAnalyzer, analyze_position_output

result = analyze_position_output(
    input_dir="outputs/my_optimization_result",
    config="position_analysis.yaml",
)

analyzer = PositionAnalyzer(config=analysis_config)
result = analyzer.analyze(output_bundle, enrichment=enrichment)
```

建议的数据契约：

```python
@dataclass(slots=True)
class OptimizationArtifacts:
    target_positions: pd.DataFrame
    trades: pd.DataFrame
    summary: pd.DataFrame
    metadata: pd.DataFrame
    run_manifest: dict[str, Any] | None = None
    resolved_config: dict[str, Any] | None = None


@dataclass(slots=True)
class AnalysisEnrichment:
    benchmark: pd.DataFrame | None = None
    tradable: pd.DataFrame | None = None
    market_amount: pd.DataFrame | None = None
    industry: pd.DataFrame | None = None
    style: pd.DataFrame | None = None
    market_cap: pd.DataFrame | None = None
    factor_exposure: pd.DataFrame | None = None
    factor_cov: pd.DataFrame | None = None
    specific_var: pd.DataFrame | None = None
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PositionAnalysisResult:
    position_summary: pd.DataFrame
    holdings_detail: pd.DataFrame
    turnover_detail: pd.DataFrame
    exposure_summary: pd.DataFrame
    risk_contribution: pd.DataFrame
    constraint_summary: pd.DataFrame
    quality_checks: pd.DataFrame
    metadata: dict[str, Any]
```

没有对应 enrichment 时，相关结果表保持合法空表并包含固定 schema，而不是返回
`None`。调用者可以稳定消费所有字段。

## 9. P0 指标定义

P0 只依赖优化产物，不访问外部数据。

### 9.1 基础仓位

对每个决策日：

$$
\text{net}_t = \sum_i w_{t,i}
$$

$$
\text{gross}_t = \sum_i |w_{t,i}|
$$

$$
\text{unallocated}_t = b_t - \text{net}_t
$$

其中 `b_t` 是预算约束目标值。只有明确是 long-only 且预算为 1 时，
`unallocated` 才可展示为“现金权重”；通用表中不得直接命名为 cash。

使用 `weight_epsilon` 判断：

- `long_count`：`w > epsilon` 的资产数；
- `short_count`：`w < -epsilon` 的资产数；
- `zero_count`：其余资产数；
- `max_weight`、`min_weight`、`max_abs_weight`；
- 最大多头、最大空头和最大绝对仓位资产。

### 9.2 集中度

令：

$$
p_i = \frac{|w_i|}{\sum_j |w_j|}
$$

定义：

$$
\text{HHI} = \sum_i p_i^2
$$

$$
\text{effective\_n} = \frac{1}{\text{HHI}}
$$

$$
\text{topk\_abs\_share} =
\frac{\sum_{i \in \operatorname{TopK}(|w|)} |w_i|}
{\sum_j |w_j|}
$$

当 gross exposure 小于容差时，HHI、effective N 和 top-k share 为 `NaN`，
同时写入质量告警，不使用零值掩盖无仓位状态。

### 9.3 换手与持仓迁移

$$
\text{one\_way\_turnover}_t =
\frac{1}{2}\sum_i |\Delta w_{t,i}|
$$

同时保留不除以 2 的：

$$
\text{gross\_traded\_weight}_t = \sum_i |\Delta w_{t,i}|
$$

当前优化器 `summary.turnover` 使用 `one_way_turnover` 口径。分析器不得用
`gross_traded_weight` 与它直接比较。

同时输出：

$$
\text{buy\_weight}_t = \sum_i \max(\Delta w_{t,i}, 0)
$$

$$
\text{sell\_weight}_t = \sum_i \max(-\Delta w_{t,i}, 0)
$$

以及：

- `entry_count`：上期不持有、本期持有；
- `exit_count`：上期持有、本期不持有；
- `increase_count`、`decrease_count`；
- 最大买入和最大卖出资产；
- `holding_overlap`：相邻两期非零持仓集合的 Jaccard 相似度；
- `weight_cosine_similarity`：相邻两期权重向量余弦相似度。

首期换手相对重建出的初始仓位计算，不相对零仓位计算。

### 9.4 优化诊断与独立复核

分析器保留 `summary.parquet` 的全部求解诊断，同时独立重算可由产物得到的指标：

- gross exposure；
- net exposure；
- one-way turnover；
- active share（仅 P1 基准可用时）；
- 约束 violation 中可由权重直接复核的部分。

结果字段必须成对保留：

```text
reported_gross_exposure
recomputed_gross_exposure
gross_exposure_difference
```

差异超过 `comparison_tolerance` 时写入 `quality_checks`，不得用重算值覆盖原值。

## 10. P1 指标定义

### 10.1 相对基准

$$
a_t = w_t - w_t^b
$$

$$
\text{active\_share}_t = \frac{1}{2}\sum_i |a_{t,i}|
$$

输出：

- benchmark weight；
- active weight；
- 最大超配和最大低配资产；
- benchmark weight coverage；
- index constituent / off-benchmark 标记；
- 组合与基准的持仓重合度。

基准在 alpha universe 外的权重不能静默丢弃。必须同时报告：

- `full_benchmark_weight`；
- `selected_universe_benchmark_weight`；
- `benchmark_weight_coverage`；
- 是否发生过再归一化。

### 10.2 行业、风格和市值暴露

对任一暴露矩阵 `B`：

$$
e_p = B^\top w
$$

$$
e_b = B^\top w^b
$$

$$
e_a = B^\top (w - w^b)
$$

行业暴露必须先经过 adapter 的分类校验，不能在分析模块临时将多行业或缺失行业
强制变成 one-hot。缺失暴露时输出 coverage，并按配置执行 `required` 失败或
`auto` 降级。

市值分析至少包括：

- portfolio weighted log market cap；
- benchmark weighted log market cap；
- active log market-cap exposure；
- 自定义或数据源定义的市值分组权重。

### 10.3 Barra 风险分解

Barra 模型：

$$
\Sigma = B F B^\top + D
$$

分析绝对风险时使用 `v = w`；分析跟踪误差时使用 `v = w - w^b`。

组合方差：

$$
\sigma_v^2 = v^\top \Sigma v
$$

系统性方差：

$$
\sigma_{\text{factor}}^2 = v^\top B F B^\top v
$$

特异方差：

$$
\sigma_{\text{specific}}^2 = v^\top Dv
$$

资产边际风险和方差贡献：

$$
\operatorname{MRC}_i = (\Sigma v)_i
$$

$$
\operatorname{RC}_i = v_i(\Sigma v)_i
$$

因子方差贡献：

$$
\operatorname{FRC}_k = (B^\top v)_k
\left[F(B^\top v)\right]_k
$$

必须验证：

$$
\sum_i \operatorname{RC}_i \approx \sigma_v^2
$$

以及 factor contribution 与 specific contribution 之和近似总方差。
有做空或主动仓位时，单个 contribution 可以为负，不能强制取绝对值。

风险结果必须同时记录：

- daily / horizon / annualized 口径；
- annualization factor；
- absolute risk 或 active risk；
- factor/exposure/covariance/specific-risk 日期；
- 资产和因子覆盖率。

### 10.4 流动性与容量

流动性分析只有在配置 `portfolio_notional` 后才可计算金额指标。

$$
\text{trade\_notional}_{t,i}
= |\Delta w_{t,i}| \times \text{portfolio\_notional}
$$

$$
\text{ADV}_{t,i}
= \operatorname{mean}\left(\text{Amount}_{s,i}\right),
\quad s \in [t-L,t-1]
$$

$$
\text{ADV\_participation}_{t,i}
= \frac{\text{trade\_notional}_{t,i}}{\text{ADV}_{t,i}}
$$

$$
\text{estimated\_days}_{t,i}
= \frac{\text{trade\_notional}_{t,i}}
{\text{ADV}_{t,i}\times \text{maximum\_adv\_participation}}
$$

窗口必须截止到决策日前一个可用交易日。`portfolio_notional` 缺失时只输出权重换手，
不得假设固定 AUM。

## 11. 约束分析

约束分析分三层：

1. 优化器上报的 `violation` 和 `slack`；
2. 可由目标权重独立重算的约束；
3. 需要 benchmark、Barra 或市场数据才能重算的约束。

`constraint_summary` 使用长表：

| 字段 | 含义 |
|---|---|
| `date` | 决策日 |
| `constraint_name` | 稳定约束名称 |
| `scope` | portfolio / asset / industry / style / risk |
| `reported_value` | 优化器上报值 |
| `recomputed_value` | 分析器重算值 |
| `limit` | 约束边界 |
| `slack` | 与边界的距离 |
| `utilization` | 边界使用比例，定义不适用时为空 |
| `is_binding` | `abs(slack) <= binding_tolerance` |
| `is_violated` | 超过 violation tolerance |
| `recompute_status` | passed / unavailable / failed |

不同方向的约束必须使用明确公式计算 slack，不能统一使用 `limit - value`：

- 上界：`upper - value`；
- 下界：`value - lower`；
- 区间：分别输出 lower/upper 两条记录；
- 等式：输出绝对偏差和容差。

## 12. 结构化输出

默认目录：

```text
analysis/
├── position_summary.parquet
├── holdings_detail.parquet
├── turnover_detail.parquet
├── exposure_summary.parquet
├── risk_contribution.parquet
├── constraint_summary.parquet
├── quality_checks.parquet
├── analysis_manifest.yaml
└── position_report.html
```

### 12.1 `position_summary.parquet`

索引为 `date`，至少包含：

```text
net_exposure
gross_exposure
unallocated_weight
long_count
short_count
zero_count
max_weight
min_weight
max_abs_weight
hhi
effective_n
top5_abs_weight_share
top10_abs_weight_share
top20_abs_weight_share
one_way_turnover
gross_traded_weight
buy_weight
sell_weight
entry_count
exit_count
holding_overlap
weight_cosine_similarity
active_share
systematic_variance
specific_variance
total_variance
predicted_volatility
predicted_tracking_error
quality_status
```

不适用的 P1 字段保留为 `NaN`，同时由 `quality_checks` 解释原因。

### 12.2 `holdings_detail.parquet`

索引为 `(date, asset)`，至少包含：

```text
weight
previous_weight
delta_weight
abs_weight
weight_rank
abs_weight_rank
benchmark_weight
active_weight
is_long
is_short
is_entry
is_exit
is_tradable
industry
market_cap
asset_variance_contribution
asset_risk_contribution_pct
```

不得只落盘非零仓位。完整 universe 是发现异常零仓位、不可交易资产变动和基准覆盖
问题的必要条件。HTML 可以只展示 top-N。

### 12.3 `turnover_detail.parquet`

索引为 `(date, asset)`，包含交易方向、权重变化、是否进入/退出、交易额、
ADV participation 和 estimated days。它与 `holdings_detail` 有意部分重叠，
用于下游交易审计稳定消费。

### 12.4 `exposure_summary.parquet`

索引为 `(date, exposure_type, exposure_name)`，包含：

```text
portfolio_exposure
benchmark_exposure
active_exposure
asset_coverage
weight_coverage
exposure_date
snapshot_match
```

### 12.5 `risk_contribution.parquet`

索引为 `(date, risk_basis, component_type, component_name)`。

- `risk_basis`：absolute / active；
- `component_type`：asset / factor / specific / total；
- 同时输出 variance contribution 和占总方差比例；
- 不强制 contribution 为正。

### 12.6 `quality_checks.parquet`

每条检查一行：

```text
check_id
date
scope
severity
status
observed
expected
difference
message
```

`severity` 为 `info / warning / error`，`status` 为
`passed / failed / unavailable`。

### 12.7 `analysis_manifest.yaml`

至少记录：

- analysis status：`passed / partial / failed`；
- riskfolio_qs 和 analysis schema 版本；
- 原优化目录和原 run manifest identity；
- 各输入 snapshot/hash/path；
- snapshot 是否匹配；
- 实际启用和跳过的模块；
- 配置文件绝对路径及解析后配置；
- 产物列表、行列数；
- warnings 和 errors 摘要；
- 创建时间和总耗时。

## 13. CLI 设计

命令：

```powershell
python -m riskfolio_qs analyze-positions `
  --config .\position_analysis.yaml `
  --input-dir .\outputs\my_optimization_result `
  --output-dir .\outputs\my_optimization_result\analysis `
  --overwrite
```

安装 console script 后等价：

```powershell
riskfolio-qs analyze-positions `
  --config .\position_analysis.yaml `
  --input-dir .\outputs\my_optimization_result
```

参数：

| 参数 | 必需 | 说明 |
|---|---:|---|
| `--config` | 是 | 分析 YAML |
| `--input-dir` | 否 | 覆盖 `input.optimization_dir` |
| `--output-dir` | 否 | 覆盖 `output.directory` |
| `--overwrite` | 否 | 覆盖已知分析产物 |
| `--debug` | 否 | 失败时显示 traceback |

退出码：

- `0`：passed；
- `2`：partial 且配置允许 partial；
- `1`：failed、核心产物错误或不允许的 partial。

CLI 标准输出应打印：

```text
status=passed|partial
dates=...
assets=...
quality_errors=...
quality_warnings=...
snapshot_match=true|false|not_applicable
output_dir=...
```

## 14. YAML 配置协议

```yaml
analysis_version: 1

input:
  optimization_dir: ../../outputs/my_optimization_result

validation:
  reconstruction_tolerance: 1.0e-10
  comparison_tolerance: 1.0e-8
  binding_tolerance: 1.0e-6
  provenance_policy: strict       # strict | warn
  allow_partial_enrichment: false

position:
  weight_epsilon: 1.0e-8
  topk: [5, 10, 20]
  include_full_universe: true
  concentration: true
  turnover: true
  holding_persistence: true

benchmark:
  mode: auto                      # auto | required | off
  source: manifest                # v1 只允许 manifest

barra:
  mode: auto                      # auto | required | off
  source: manifest                # v1 只允许 manifest
  analyze_absolute_risk: true
  analyze_active_risk: true

liquidity:
  mode: 'off'                     # auto | required | off
  portfolio_notional: null
  adv_window_days: 20
  maximum_adv_participation: 0.10

report:
  html: true
  top_holdings: 20
  top_trades: 20

output:
  directory: ../../outputs/my_optimization_result/analysis
  parquet: true
  overwrite: false
```

配置规则：

1. 相对路径相对于分析配置文件所在目录解析；
2. CLI 覆盖路径相对于当前工作目录解析；
3. 未知字段直接报错；
4. `mode=required` 缺失或校验失败时整次分析失败；
5. `mode=auto` 只有在 manifest 声明该输入时启用；
6. `mode=off` 不访问对应数据源；
7. `portfolio_notional` 必须为正数，否则流动性金额分析不可用；
8. `include_full_universe=false` 只影响展示，不得改变底层结构化计算。

## 15. 失败与降级策略

| 场景 | 行为 |
|---|---|
| 核心文件缺失或 schema 错误 | failed，不生成误导性报告 |
| 交易无法重建目标仓位 | failed |
| 优化器上报值与重算值不一致 | 结构化结果照常生成，质量状态至少 warning；严重差异可配置为 error |
| auto enrichment 不存在 | 跳过，记录 unavailable |
| required enrichment 不存在 | failed |
| snapshot strict 不匹配 | 对应 enrichment 失败；required 时整次失败 |
| snapshot warn 不匹配 | 允许计算，整体状态 partial |
| Barra 某期覆盖不足 | 该期风险结果为空并记录 error，不用零填充 |
| HTML 渲染失败 | Parquet 已成功时状态 partial，不删除结构化产物 |
| 优化 run 本身 status=failed | 默认拒绝生成普通报告；诊断模式可生成并显著标注 failed source |

任何降级都必须进入 manifest 和 `quality_checks`，不能只写日志。

## 16. HTML 报告信息架构

HTML 是结构化结果的只读视图，建议包含：

1. 运行状态、优化器、日期范围、资产数、fallback 和数据版本；
2. 仓位总览：净/总敞口、持仓数、集中度；
3. 换手：单边换手、买卖、进入退出和最大变动；
4. 相对基准：active share、最大超配/低配；
5. 暴露：行业、风格和市值主动暴露；
6. 风险：绝对/主动风险、系统性/特异风险、因子和资产贡献；
7. 约束：binding、violation 和 reported/recomputed 差异；
8. 流动性：ADV participation 和预计退出天数；
9. 数据质量：snapshot、覆盖率、缺失、时点和全部告警。

图表只用于展示趋势和分布。报告中的每个图必须能追溯到某个 parquet 表，
不得拥有独立指标口径。

## 17. 实施阶段

### P0：产物级仓位审计

1. contracts 和 artifact loader；
2. 核心 schema、重建和完整性检查；
3. 仓位、集中度、换手和持仓迁移；
4. summary reported/recomputed 对账；
5. constraint summary 基础版；
6. parquet + analysis manifest；
7. CLI 和 YAML 严格解析；
8. 最小 HTML 报告。

P0 完成后，即使没有 Barra 或 data_access，也能判断优化产物是否自洽。

### P1：外部数据增强

1. manifest 驱动的 benchmark 重载；
2. data_access snapshot 对账；
3. 行业、风格和市值暴露；
4. Barra 绝对/主动风险分解；
5. 流动性与容量；
6. 完整 HTML 报告；
7. 部分失败和 provenance 审计。

### P2：目标与实际仓位联动

1. 定义 `vectorbt_qs` realized position / fills 输入契约；
2. 目标与实际权重偏差；
3. 未成交、延迟成交和不可交易漂移；
4. 将仓位分析结果链接至回测绩效报告。

P2 仍不应在 riskfolio_qs 内复制收益、净值和回撤计算。

## 18. 测试矩阵

### 18.1 单元测试

- long-only、long-short、全零仓位；
- 首期从零、首期从 benchmark、首期非零自定义初始仓位；
- 买入、卖出、进入、退出和不变仓位；
- HHI、effective N、top-k；
- active share 和 benchmark coverage；
- Barra total = factor + specific；
- asset RC sum = portfolio variance；
- 上界、下界、区间和等式 slack；
- ADV 窗口不使用未来数据。

### 18.2 契约测试

- 宽表目标仓位和 MultiIndex trades；
- 重复日期/资产、错位日期、非有限数值；
- manifest 日期数和实际不一致；
- reported/recomputed 指标故意不一致；
- enrichment 合法空表 schema；
- YAML 未知字段拒绝；
- 已有输出目录不允许静默覆盖。

### 18.3 集成测试

至少覆盖当前 CLI 支持的：

- TopN / alpha-only；
- 历史绝对收益；
- 历史指数增强；
- Barra precomputed 指数增强；
- minvar 和 meanvar；
- 有/无 benchmark；
- data_access snapshot 匹配/不匹配；
- Barra 数据缺失、因子错位和资产覆盖不足；
- 优化器 fallback 和 source status=failed。

### 18.4 验收标准

1. P0 对所有合格 optimize 产物可运行；
2. `target - previous - trade` 最大误差不超过配置容差；
3. P0 重算 gross/net/turnover 与当前 summary 一致；
4. Barra 分解加总误差不超过数值容差；
5. 所有 unavailable/partial 都有结构化原因；
6. 同一输入和同一数据 snapshot 产生相同结构化结果；
7. HTML 删除后可仅由 parquet 和 manifest 重建；
8. 分析模块不修改原优化产物。

## 19. 实现前必须补强的上游信息

为获得可靠 P1，建议优化 CLI manifest 后续补充：

1. 每个 data_access 读取的完整 lineage，而不仅是 snapshot hash；
2. Barra risk package 的文件 hash 或 immutable version；
3. 预算约束目标值和所有实际生效约束边界；
4. 初始仓位来源及首日初始化模式；
5. benchmark 归一化前覆盖率和归一化规则；
6. 输出 artifact 自身的 checksum；
7. 优化运行唯一 `run_id`。

这些字段不阻塞 P0，但缺失时 P1 只能报告相应 provenance 限制。

## 20. 设计结论

仓位分析模块应首先是一套独立、结构化、可审计的产物检查器，其次才是可视化报告。

正确边界是：

```text
riskfolio_qs:
    alpha -> 目标仓位 -> 目标仓位风险与约束分析

vectorbt_qs:
    目标仓位 -> 模拟执行 -> 实际仓位 -> 收益、成本和回撤分析
```

第一阶段应先实现 P0，建立稳定数据契约和独立复核能力；随后实现 P1，
用 manifest 驱动 data_access 与 Barra enrichment。只有在结构化输出和
provenance 稳定后，才扩展 HTML 和 vectorbt_qs 联动。
