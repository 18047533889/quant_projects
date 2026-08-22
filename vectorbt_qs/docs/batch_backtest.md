# Batch 参数网格回测

如果目标是从空环境直接跑冻结的 `standard_accurate_v2`，请先阅读
[从零跑通 accurate-batch](accurate_batch_quickstart.md)。本文其余部分用于说明通用
Batch 参数网格和完整配置。

Batch 模式用于在一次命令中遍历多组目标权重和回测参数。每个遍历条件必须在
`batch.grid` 中显式定义为列表，各列表之间按照笛卡尔积展开。

## 配置方式

```yaml
market: ashare

input:
  positions:
    group_1: path/to/group_1.parquet
    group_2: path/to/group_2.parquet

backtest:
  start: 2024-01-01
  end:
  execution_mode: fast
  execution_lag: 1
  price_type: Open_adj
  fees: 0.0
  fixed_fees: 0.0
  slippage: 0.0

batch:
  grid:
    freq: [1D, 1W]
    fast_tradeability: [ignore, approximate]
    limit_check_mode: [execution, strict]
  max_runs: 256
  on_error: raise
  workers: 1

output:
  dir: examples/output/factor_batch
  plot: true
  plot_each: false
```

这个例子有两个仓位文件和 `2 × 2 × 2` 组参数，因此总共运行 16 次回测。
展开顺序保持 YAML 中参数和值的书写顺序，每次运行使用稳定编号
`run_0001`、`run_0002` 等。遍历参数和仓位来源记录在参数清单中，每组回测的
完整生效配置另存为 JSON，不依赖文件名反推配置。

运行命令：

```bash
python -m vectorbt_qs batch configs/config.batch.example.yaml
```

## 参数范围

`batch.grid` 中的键默认对应 `backtest` 配置字段。嵌套字段使用点路径：

```yaml
batch:
  grid:
    price_type: [open, vwap, close]
    costs.commission: [0.00015, 0.00025]
    costs.minimum_commission: [0.0, 5.0]
```

也允许写成 `backtest.price_type`，其含义与 `price_type` 相同。Batch 只负责组合
参数，不放宽单次回测约束。例如 fast 模式仍固定使用其 factor research 价格和
费用口径；与该口径冲突的组合会由单次回测校验拒绝。

`input.positions` 支持三种形式：

- 单个 parquet 路径；
- parquet 路径列表；
- `{组合名称: parquet 路径}` 映射，推荐用于多个因子分层。

多个仓位输入同样参与笛卡尔积。例如 5 个因子分层和 6 组参数会产生 30 次回测。

## 错误与规模控制

- `max_runs` 默认 256，参数组合超过上限时不会开始回测；
- `on_error: raise` 在任意组合失败时立即停止；
- `on_error: continue` 记录失败原因并继续后续组合；
- `workers: 1` 使用串行；大于 1 时使用多进程并行回测；
- 参数列表不能为空，标量也不会被隐式转换成单元素列表。

多进程只并行最耗时的单组回测阶段，结果会重新按 `run_0001`、`run_0002`
的参数顺序排列；图表和 Barra 暴露导出仍在主进程按目录顺序执行。Windows 使用
spawn 模式，每个进程会各自加载行情并初始化一次 Numba，通常建议从
`workers: 2` 或 `workers: 4` 开始，避免同时占用过多内存。

## 输出

Batch 输出目录包含：

| 文件 | 内容 |
|---|---|
| `batch_parameters.csv` | 每次运行的仓位来源、遍历参数、状态和错误 |
| `effective_configs.json` | 每次运行合并基础参数和遍历条件后的完整配置 |
| `performance_comparison.csv` | 参数与 vectorbt 绩效指标合并后的对比表 |
| `nav_curves.parquet` | 所有成功组合的标准化 NAV |
| `batch_nav_curves.png` | 静态 NAV 对比图 |
| `batch_nav_curves.html` | Plotly 交互式 NAV 对比图 |
| `runs/run_xxxx/performance_stats.csv` | 每组回测的独立绩效文件 |

如果启用 `analysis.exposure`，每个成功组合的暴露分析会写入对应的
`runs/run_xxxx` 目录。`output.plot_each: true` 可以额外生成每组独立回测图；
默认关闭，避免大批量任务产生过多图像。

## Python API

```python
from vectorbt_qs.mvp.engine import run_backtest_batch

result = run_backtest_batch(
    "ashare",
    {
        "group_1": weights_1,
        "group_2": weights_2,
    },
    {
        "freq": ["1D", "1W"],
        "fast_tradeability": ["ignore", "approximate"],
    },
    base_config={
        "execution_mode": "fast",
        "fees": 0.0,
        "fixed_fees": 0.0,
        "slippage": 0.0,
    },
)

print(result.parameters)
print(result.performance_table())
print(result.nav_curves())
```

Batch 表示“一次提交、统一产出”。`workers=1` 时按组合顺序串行调用经过验证的
单次回测入口；`workers>1` 时多进程并行执行单组回测，并在主进程按稳定的 run ID
顺序汇总输出。两种方式不改变单组回测口径。

## 冻结准确口径：standard_accurate_v2

需要统一交付准确回测时，推荐使用版本化的冻结 profile。用户只提供仓位文件、
Barra 因子目录和结果根目录：

```yaml
input:
  positions: path/to/target_positions.parquet
  barra_root: path/to/v2_sbi_mvl_fullA

output:
  root: path/to/backtest_results

batch:
  profile: standard_accurate_v2
  workers: 4
```

也可以使用三路径 CLI：

```bash
python -m vectorbt_qs accurate-batch \
  --positions path/to/target_positions.parquet \
  --barra-root path/to/v2_sbi_mvl_fullA \
  --output-root path/to/backtest_results \
  --workers 4
```

该 profile 固定 `execution_mode=accurate`，并冻结其余基础参数。只有以下四个显式
维度参与笛卡尔积：

| 参数 | 固定候选值 |
|---|---|
| `benchmark_index` | `000300.SH`、`000905.SH`、`000852.SH` |
| `price_type` | `open`、`vwap` |
| `cost.commission` | `0.002`、`0.001` |
| `freq` | `1D`、`1W` |

总计固定生成 `3 × 2 × 2 × 2 = 24` 组。YAML 可以像
[`configs_module.yaml`](../configs/configs_module.yaml) 一样把完整 `batch.grid` 和冻结
`backtest` 参数显式写在正文中，便于审阅和归档；但这些值必须与
`standard_accurate_v2` 契约完全一致，不能增删候选值或覆盖冻结参数。
`start/end`、`workers` 和数据路径仍可按每次任务修改。

其中 benchmark 只是分析维度，不改变订单和 NAV。系统实际执行 8 组交易计算，再
复用每组交易结果生成三个 benchmark 报告。准确执行全程使用 `StockDailyBar` 的
原始价格和真实股数；`StockDividend` 的现金分红进入现金，送股与转增进入股数。
复权因子仅用于数据校验，不再代替公司行为。

其他冻结参数为：

| 参数 | standard_accurate_v2 |
|---|---:|
| `execution_mode` | `accurate` |
| `init_cash` | `10000000` |
| `signal_time` | `close` |
| `execution_lag` | `1` |
| `planner_engine` | `numba` |
| `limit_check_mode` | `strict` |
| `limit_price_rtol` | `0.00001` |
| `limit_price_atol` | `0.00000001` |
| `lot_size` | `100` |
| `slippage` | `0.001` |
| `costs.stamp_tax` | `0.0005` |
| `costs.transfer_fee` | `0.00001` |
| `costs.minimum_commission` | `5.0` |
| `cash_sharing` | `true` |
| `allow_partial` | `false` |
| `strict_symbols` | `true` |
| `performance_year_days` | `252` |
| `risk_free_rate` | `0.0` |
| `max_participation_rate` | `0.10` |

绩效以日度 NAV 简单收益率计算。冻结口径的夏普为
`(日收益均值 - 日无风险收益) / 日收益样本标准差 × sqrt(252)`，其中年化无风险
收益率为 0%。benchmark 不进入夏普分子；相对 benchmark 的表现另用跟踪误差、
信息比率和年化主动收益表示。由于夏普使用算术平均日收益，而区间收益使用复合收益，
高波动路径中二者符号可能不同。

Barra 暴露分析采用 `missing_policy: report_unknown`，不把缺失证券重归一化为已覆盖
证券。冻结口径的组合覆盖率门槛为 98%，基准覆盖率门槛为 99%。中证1000包含较多
尚处于风险模型初始化窗口的新股，因此不使用通用分析接口更严格的 99.5% 门槛；
未覆盖权重仍会完整写入 `coverage.csv`。

冻结交付只输出 B 暴露，不启用实验性的同日 B×f 事后归因。若回测区间超出风险
模型日期，交易回测仍保留完整区间，但暴露报告只取交集，并将运行状态写为
`complete_with_exposure_truncation`，不会静默伪装成完整暴露。

`start` 默认是 `2024-01-01`，`end` 默认使用仓位数据末日：

```yaml
backtest:
  start: 2024-01-01
  end: 2025-12-31
```

结果目录严格采用：

```text
用户结果根目录/
└── 仓位数据文件名/
    ├── benchmark_index_000300.SH_price_type_open_cost.commission_0.002_freq_1D/
    ├── benchmark_index_000300.SH_price_type_open_cost.commission_0.002_freq_1W/
    └── ...
```

目录名只包含四个显式参数。冻结的其他参数、日期区间和数据路径记录在
`effective_config.json`，不进入目录名。每个参数目录包含绩效、NAV、静态图、
Plotly 报告以及 Barra 风格和行业暴露分析。
