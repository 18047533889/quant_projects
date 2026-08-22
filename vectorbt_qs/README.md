# vectorbt_qs — MVP 量化回测系统

> 如果需要进行 `accurate` 准确回测，请先阅读
> [从零跑通 accurate-batch](docs/accurate_batch_quickstart.md)。该文档是环境配置、
> 数据路径、仓位准备、短区间验证和正式 Batch 回测的标准入口。
> 如果只需要用一套固定口径快速验证仓位，请使用
> [单配置 Accurate Benchmark](docs/accurate_benchmark.md)。

基于 [vectorbt](https://github.com/polakowo/vectorbt) 的 A 股/美股组合回测系统。
项目在外层实现数据适配、市场约束和统一接口，并对仓库内 vendored vectorbt
扩展公司行动现金/资产流，以保证 Accurate 模式的真实股数与净值核算一致。

## v0.4 更新 (2026-07-23)

- **可复现安装**：新增项目包元数据、CLI 命令和与 vendored vectorbt 对齐的依赖
- **准确执行口径**：收盘信号在下一交易日原始开盘价/VWAP 执行，逐日维护真实现金和股数
- **冻结准确 Batch**：三条输入路径固定生成 24 组标准参数结果，支持多进程执行
- **单配置 Benchmark**：VWAP、日频、中证 1000 和固定费率一键生成准确回测报告
- **A 股约束**：先卖后买、100 股买入整手、停牌/涨跌停拒单、方向性税费和最低佣金
- **B/f 暴露分析**：支持实际组合、基准和主动风格/行业暴露，以及实验性事后因子归因
- **数据修复**：COS URI、`full.parquet` 日期过滤、ADR 布尔解析和非交易日退市
- **测试**：覆盖约束、数据适配、费用、整手、基准收益和周频行为的回归测试

## v0.3 更新 (2026-07-18)

- **CLI 入口**：`python -m vectorbt_qs backtest --benchmark B001` 一键回测
- **配置文件模式**：`python -m vectorbt_qs run config.yaml` YAML 驱动
- **Benchmark 集成**：内置 B001-B008 全部 benchmark 路径映射
- **性能优化**：消除行情重复加载，2294 天 × 5122 股回测从 89s → 46s

## v0.2 更新 (2026-07-17)

- **接入 data_access**：A 股数据优先走 `data_access`（DuckDB 列裁剪 + 谓词下推），不可用时 fallback 到 COS reader
- **修复涨跌停复权**：HighLimit/LowLimit 同步乘 Factor
- **runner 增强**：支持 `fees`/`fixed_fees` config 显式覆盖约束层费率

---

## 项目结构

```
vectorbt_qs/
├── vectorbt/                  # vectorbt 源码（从上游克隆）
├── pyproject.toml             # 安装、依赖与 vectorbt-qs 命令
├── __init__.py                # 包版本
├── __main__.py                # python -m vectorbt_qs 入口
├── cli.py                     # CLI 主逻辑
├── configs/                   # 可复用 YAML 配置模板与说明
├── mvp/
│   ├── data/
│   │   ├── cos_reader.py      # COS 读取器（CLI 优先 → SDK fallback）
│   │   └── adapter.py         # 数据适配器（data_access 优先 → COS fallback）
│   ├── constraints/
│   │   ├── ashare.py          # A 股约束：停牌/涨跌停/分方向费率
│   │   └── us_stock.py        # 美股约束：SSR/退市/ADR排除
│   ├── engine/
│   │   ├── execution.py       # 交易日调度、现金/股数订单规划、A 股费用
│   │   └── runner.py          # 统一回测入口：数据→执行→回测→报告
│   ├── analysis/               # Barra-lite B/f 组合暴露与实验性归因
│   ├── visualization.py       # vectorbt Plotly 仪表板与 HTML 导出
│   └── examples/
│       ├── ashare_demo.py     # A 股 Demo
│       ├── ashare_demo.ipynb  # A 股 Demo (Notebook)
│       ├── us_stock_demo.py   # 美股 Demo
│       ├── us_stock_demo.ipynb# 美股 Demo (Notebook)
│       ├── backtest_benchmark_v1_1.py  # benchmark 回测 + 可视化
│       └── output/            # 回测图表输出
├── examples/output/           # 全局输出目录
├── docs/
│   ├── README.md                       # 文档索引
│   ├── accurate_batch_quickstart.md      # 从零跑通冻结准确 Batch
│   ├── accurate_benchmark.md             # 单配置 Accurate Benchmark
│   ├── target_positions_format.md         # 目标仓位数据格式
│   └── benchmark_performance_report.md   # 性能测试报告
├── scripts/
│   ├── backtest_from_pred.py           # 预测数据 → 目标权重 → 回测
│   ├── test_benchmark_positions.py     # benchmark 仓位对比回测
│   ├── test_target_weight_backtest.py  # 目标权重端到端测试
│   └── benchmark_perf_test.py          # 性能压测
├── requirements.txt
├── tests/                     # 外层适配与约束回归测试
├── .gitignore
└── README.md
```

---

## 快速开始

### 1. 环境准备

```bash
git clone https://github.com/WuHaohai06/vectorbt_qs.git
cd vectorbt_qs

pip install -r requirements.txt
pip install -e .

# data_access 本地读取（可选）
pip install -e ../data_access
```

### 2. 设置环境变量

```powershell
$env:DATA_ACCESS_SKIP_COS_MIRROR = "1"
$env:ASHARE_PARQUET_ROOT = "D:\path\to\lqtp_data"
```

---

## 三种使用方式

### 方式 1 — CLI 命令行（推荐）

```bash
# 列出所有 benchmark
python -m vectorbt_qs list

# 运行 benchmark 回测
python -m vectorbt_qs backtest --benchmark B001

# 指定 parquet 文件 + 图表
python -m vectorbt_qs backtest -p path/to/target_positions.parquet --plot -o output/

# 查看帮助
python -m vectorbt_qs backtest --help
```

可选参数：

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--benchmark`, `-b` | Benchmark ID (B001-B008) | — |
| `--positions`, `-p` | target_positions.parquet 路径 | — |
| `--init-cash` | 初始资金 | 10,000,000 |
| `--fees` | 兼容参数：双边统一总费率（建议配置 `costs`） | A 股默认费用模型 |
| `--slippage` | 滑点 | 0.001 |
| `--plot` | 生成净值曲线图 | — |
| `--output-dir`, `-o` | 输出目录 | examples/output |
| `--risk-model-root` | 启用 B/f 暴露分析并指定风险模型目录 | — |
| `--risk-data-root` | `lqtp_data` 根目录，用于读取指数成分权重 | `../lqtp_data` |
| `--exposure-attribution` | 生成实验性事后 B×f 归因 | 关闭 |

### 方式 2 — 配置文件模式

```bash
python -m vectorbt_qs run configs/config.example.yaml
```

```yaml
# configs/config.example.yaml
market: ashare
input:
  positions: path/to/target_positions.parquet
backtest:
  init_cash: 10000000
  execution_mode: accurate
  signal_time: close
  execution_lag: 1             # 信号后的下一交易日
  price_type: open
  limit_check_mode: execution  # 默认；VWAP/Close 保守回测可改为 strict
  limit_price_rtol: 0.00001
  limit_price_atol: 0.00000001
  costs:
    commission: 0.00025
    stamp_tax: 0.0005
    transfer_fee: 0.00001
    minimum_commission: 5.0
  slippage: 0.001
output:
  dir: examples/output
analysis:
  exposure:
    enabled: true
    risk_model_root: ../v2_sbi_mvl_fullA
    benchmark_data_root: ../lqtp_data
    benchmark_index: 000300.SH
    weight_source: realized_close
    missing_policy: report_unknown
    min_portfolio_coverage: 0.98
    min_benchmark_coverage: 0.995
    include_attribution: true
```

### 方式 3 — Python API

```python
import pandas as pd
from vectorbt_qs.mvp.engine.runner import run_backtest, portfolio_report

# 你的策略输出 → 目标权重矩阵
target_weights = pd.DataFrame(
    data=...,                     # 权重值
    index=pd.DatetimeIndex(...),  # 日期
    columns=[...],                # 股票代码
)

pf = run_backtest("ashare", target_weights,
                  config={"init_cash": 10_000_000})

stats = portfolio_report(pf)
print(stats[["Start Value", "End Value", "Total Return [%]", "Sharpe Ratio"]])

from vectorbt_qs.mvp.analysis import analyze_portfolio_exposure

exposure = analyze_portfolio_exposure(
    pf,
    risk_model_root="../v2_sbi_mvl_fullA",
    benchmark_index="000300.SH",
    benchmark_data_root="../lqtp_data",
)
exposure.export("output/exposure")
```

### Accurate：状态机与加速引擎

`accurate` 会按交易日维护实际现金和持仓，先卖后买，并处理整手、停牌、涨跌停、方向性费用、最低佣金和滑点。订单规划默认使用 Numba 编译的状态机：

两种回测口径的完整业务说明见
[Fast 与 Accurate 回测模式说明](docs/backtest_modes.md)。

```yaml
backtest:
  execution_mode: accurate
  planner_engine: numba        # 默认；生产和批量回测
```

原始 Python 状态机作为可读的回归基准保留。排查边界场景时可显式切换，业务口径和输出结构不变：

```yaml
backtest:
  execution_mode: accurate
  planner_engine: python       # 参考实现；速度较慢
```

Numba 引擎遇到非法动态执行数据时，会交回 Python 参考实现生成完全相同的异常诊断。
当前测试会逐订单比较 Python 与 Numba，包括现金分红、送转股、成交容量、费用、
滑点、停牌和涨跌停场景。`tests/baselines/B002_meanvar_1D_python` 仅保留为 V1
复权价格口径的历史审计材料，不是 V2 的正确性基准。

### Fast：无摩擦因子研究口径

`execution_mode: fast` 专门用于大量因子、分组或参数组合的横向筛选，不再表示旧版近似实盘回测。它固定采用同一套可比较假设：

- 收盘生成信号，在下一个真实交易日以复权开盘价执行；
- 周频先选每周最后一个实际信号日，再在下一交易日执行；
- 使用完整日频交易日历计算收益、波动率和回撤；
- 使用分数股、共享现金，消除股价和初始资金造成的整手偏差；
- 手续费、固定费用和滑点全部为零；
- A 股只做多，各行目标权重之和不能超过 1。

```yaml
backtest:
  execution_mode: fast
  freq: 1D
  signal_time: close
  execution_lag: 1
  price_type: Open_adj
  fast_tradeability: ignore
  fees: 0.0
  fixed_fees: 0.0
  slippage: 0.0
```

`fast_tradeability` 有两种显式假设：

| 值 | 含义 | 用途 |
|---|---|---|
| `ignore` | 忽略停牌和涨跌停，假定所有有开盘价的目标都可成交 | 默认；纯粹比较因子选股能力 |
| `approximate` | 在目标权重层近似阻止停牌、涨停买入和跌停卖出 | 敏感性检查；不等同于真实订单状态 |

fast 结果用于比较组合排序、分组收益和多空组收益差。绝对收益、容量和可成交性仍应由 `accurate` 复核。完整模板见 [config.factor_fast.example.yaml](configs/config.factor_fast.example.yaml)。

### Batch：参数列表笛卡尔积

`batch` 模式可以在一次命令中遍历多个目标权重文件和多组回测参数。每个遍历条件
在 `batch.grid` 中显式写成列表，不同列表按笛卡尔积展开：

```yaml
input:
  positions:
    group_1: path/to/group_1.parquet
    group_2: path/to/group_2.parquet

batch:
  grid:
    freq: [1D, 1W]
    fast_tradeability: [ignore, approximate]
    limit_check_mode: [execution, strict]
  max_runs: 256
  on_error: raise
```

```bash
python -m vectorbt_qs batch configs/config.batch.example.yaml
```

系统统一输出运行参数清单、绩效对比、所有组合的 NAV 数据和静态/交互式 NAV
对比图。嵌套参数可使用 `costs.commission` 形式。完整配置与输出说明见
[Batch 参数网格回测](docs/batch_backtest.md)，可直接复制
[Batch 示例配置](configs/config.batch.example.yaml)。

需要统一的准确回测交付时，可使用冻结的 `standard_accurate_v2` profile。用户只需
提供仓位 parquet、Barra 因子目录和结果根目录，系统固定生成沪深300/中证500/
中证1000、open/vwap、两档佣金和日频/周频的 24 组结果：

```bash
python -m vectorbt_qs accurate-batch \
  --positions path/to/target_positions.parquet \
  --barra-root path/to/v2_sbi_mvl_fullA \
  --output-root path/to/backtest_results \
  --workers 4
```

默认 `start=2024-01-01`，可通过 `--start/--end` 修改日期边界。结果按
`根目录/仓位文件名/参数名_参数值_...` 保存。YAML 模板见
[冻结 Accurate Batch 示例](configs/config.accurate_batch.example.yaml)。首次使用请直接阅读
[从零跑通 accurate-batch](docs/accurate_batch_quickstart.md)。

V2 明确使用原始交易价格和真实股数执行整手、费用、涨跌停与组合估值，并从
`lqtp_data/StockDividend` 按除权除息日计入现金分红、送股和转增股；不再把包含现金
分红的复权因子误当作股数变化。滑点后的最终价格必须位于当日 High/Low 与涨跌停
边界内。绩效统一按 A 股 252 个交易日、年化无风险利率 0% 计算。24 份报告中，真正
影响交易的参数组合只有 8 组，三个 benchmark 仅复用同一交易结果生成相对绩效与
暴露报告。单股每次订单最多成交当日市场成交量的 10%，未完成部分保留在真实持仓
状态中，并在后续调仓日按新目标重新计算。

### 交互式 Plotly 可视化

`mvp.visualization` 统一封装当前 vendored vectorbt 的 Plotly 图表。图表数据和 trace 由 vectorbt 生成；vectorbt_qs 负责图表注册、现金共享组合的安全默认值、单标的组合方式和 HTML 导出。

组合级默认仪表板包含：

- 策略 NAV、基准指数 NAV 和超额 NAV；
- 累计收益和水下回撤；
- 现金余额；
- 总敞口和净敞口。

其中三条 NAV 均以回测首日归一化为 1，超额 NAV 定义为
`策略 NAV / 基准 NAV`。A 股基准行情读取自 `lqtp_data/IndexDailyBar`，
由配置项 `backtest.benchmark_index` 选择，例如 `000300.SH`。

```python
from vectorbt_qs.mvp.visualization import (
    build_plotly_dashboard,
    export_plotly_dashboard,
)

# 现金共享的多股票组合总览
fig = build_plotly_dashboard(pf, title="Portfolio Overview")
fig.show()

# 单只股票的订单、交易盈亏、资产流和持仓数量
asset_fig = build_plotly_dashboard(pf, column="000001.SZ")
asset_fig.show()

# 导出可离线打开的完整 HTML
export_plotly_dashboard(
    pf,
    "output/portfolio_dashboard.html",
    include_plotlyjs=True,
)
```

也可以选择任意兼容的 vectorbt 子图：

```python
fig = build_plotly_dashboard(
    pf,
    charts=["value", "cum_returns", "drawdowns", "cash"],
)
```

支持的图表包括策略/基准/超额 NAV、订单、交易、交易/持仓盈亏、资产流、现金流、持仓数量、现金、持仓市值、组合净值、累计收益、回撤、水下回撤及总/净敞口。运行 CLI `--plot` 或配置文件中的 `output.plot: true` 时，会在保留原 Matplotlib PNG 的同时生成 `portfolio_dashboard.html`。

### Barra-lite B/f 组合暴露

`mvp.analysis` 使用 `pf.asset_value(group_by=False) / pf.value()` 计算每日
实际收盘权重，再与风险模型长表 B 连接，输出组合、基准和主动风格/行业
暴露。计算采用 DuckDB 长表聚合，不会构造全 A 的三维稠密暴露矩阵。

风险模型与回测日期自动取交集；未匹配股票不会填零，而会进入
`coverage.csv` 的 `unknown_weight` 和 `coverage_weight`。指数权重读取自
`lqtp_data/IndexConstituent`，原始百分数会转换为小数权重。

配置 `analysis.exposure.plot: true` 时还会输出组合、基准和主动风格暴露
PNG 时序图；每个风格因子使用独立子图，并与交互式
`exposure_dashboard.html` 同时生成。行业部分默认输出主动暴露热力图、
最新超配/低配排序、组合/基准 TopN 对比和 TopN+Other 配置堆叠图；
行业中文名读取自 `lqtp_data/StockIndustry`。

可选的 B×f 归因明确标记为 `experimental_ex_post_contemporaneous_B`：
它使用期初持仓和当日回归暴露，仅用于研究性相对归因。当前模块不读取
F/D，也不输出绝对波动率、VaR 或正式风险贡献。完整说明见
[组合暴露分析](docs/exposure_analysis.md)。

---

## 可用 Benchmarks

| ID | 信号源 | 优化器 | 天数 | 标的 |
|---|---|---|---|---|
| B001 | gtja191_alpha191 | topn 等权 | 2,294 | 5,122 |
| B002 | gtja191_alpha191 | meanvar_hist v1.1 | 2,294 | 300 |
| B002v1 | gtja191_alpha191 | meanvar_hist v1（等权） | 2,294 | 300 |
| B003 | composite169_ic | topn 等权 | 2,294 | 5,122 |
| B004 | composite169_ic | meanvar_hist | 2,294 | 300 |
| B005 | linear_ridge_h1 | topn 等权 | 1,433 | 5,112 |
| B006 | linear_ridge_h1 | meanvar_hist | 1,433 | 300 |
| B007 | tree_xgboost_h1 | topn 等权 | 599 | 5,086 |
| B008 | tree_xgboost_h1 | meanvar_hist | 599 | 300 |

---

## 输入格式

核心输入是一个 **目标权重矩阵**（`pandas.DataFrame`）：

| 维度 | 含义 | 要求 |
|---|---|---|
| index | 交易日 | `DatetimeIndex`，递增 |
| columns | 标的代码 | A 股用 Symbol（如 `000001.SZ`），美股用 Ticker（如 `AAPL`） |
| values | 目标权重 | `0.05`=做多5%, `-0.03`=做空3%, `0`=平仓, `NaN`=跳过 |

详见 [目标仓位数据格式](docs/target_positions_format.md)。

---

## 当前覆盖的交易规则

| 规则 | A 股 | 美股 | 实现层级 | 实现方式 |
|---|---|---|---|---|
| 停牌过滤 | ✅ | — | 订单规划 | 订单拒绝并保持实际持仓 |
| 涨跌停 | ✅ | — | 订单规划 | 涨停不买、跌停不卖 |
| 收盘信号隔日执行 | ✅ | — | 交易日调度 | `execution_lag=1` 表示下一交易日 |
| T+1 可卖股数 | ✅* | — | 每日单次执行 | 日频模型不会卖出当日新买股份 |
| 印花税 | ✅ | — | 订单费用 | 仅卖出收取 |
| 佣金及最低佣金 | ✅ | ✅ | 订单费用 | 按实际成交额逐单计算 |
| 滑点 | ✅ | ✅ | 订单价格 | 买卖方向分别处理 |
| 整手约束（100 股） | ✅ | — | 订单规划 | 买入和部分减仓整手，清仓允许零股 |
| 现金分红与送转股 | ✅ | — | 公司行为流 | 除权除息日按 `StockDividend` 更新现金/真实股数 |
| 成交量容量 | ✅ | — | 订单规划 | V2 单股订单上限为当日 `Volume` 的 10% |
| SSR 限制 | — | ✅ | ① 输入矩阵 | 空头 `target_weights` → 0 |
| ADR 排除 | — | ✅ | ① 输入矩阵 | 按明确解析后的 ADR 标志排除 ticker |

> A 股默认使用 `accurate` 模式。周频先选择每周最后一个实际信号日，再按交易日历
> 延迟执行，因此 `freq=1W, execution_lag=1` 是周末信号后的下一交易日，而不是下一周。
> `fast` 是固定零摩擦、分数股的因子研究模式。只有设置
> `fast_tradeability=approximate` 时才启用目标权重层的近似交易约束。

---

## 性能

| 场景 | 规模 | 耗时 |
|---|---|---|
| B001 topn 等权 | 2294天 × 5122股 | ~46s |
| B006 meanvar | 1433天 × 300股 | ~20s |
| B007 XGBoost | 599天 × 5086股 | ~15s |

> 环境：Windows 11, Python 3.11, DuckDB + 本地 parquet  
> 详见 [docs/benchmark_performance_report.md](docs/benchmark_performance_report.md)

---

## 上下游链路

```
信号源                    riskfolio_qs              vectorbt_qs
┌──────────────┐     ┌─────────────────┐     ┌──────────────────┐
│ alpha_191     │  →  │ meanvar / topn   │  →  │ run_backtest()    │
│ composite169  │     │ target_positions │     │ → 净值 + Sharpe   │
│ ridge/OLS ... │     │ trades.parquet   │     │ → 图表输出        │
└──────────────┘     └─────────────────┘     └──────────────────┘
```

---

## 示例输出

运行 `python -m vectorbt_qs backtest -b B002 --plot` 会生成净值与回撤、月度收益
热力图、年度收益、持仓数量等 PNG，以及 `portfolio_dashboard.html` 交互式报告。
这些回测产物写入 `examples/output/`，属于本地生成文件，不纳入 Git；需要长期保留
的结果应复制到独立的结果存储。

---

## 依赖

```text
Python >= 3.11, < 3.13
numpy >= 1.26, < 2
pandas >= 2.2, < 3
numba >= 0.60, < 0.61
pyarrow >= 15
plotly >= 4.12
PyYAML >= 6.0
```

完整依赖与版本约束以 `pyproject.toml` 和 `requirements.txt` 为准。
