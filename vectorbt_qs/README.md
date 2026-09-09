# vectorbt_qs — A 股/美股组合回测系统（fast/accurate 双口径）

基于 vendored [vectorbt](https://github.com/polakowo/vectorbt) **1.1.0** 的组合回测系统。
vendored 意味着上游源码整个拷进 `vectorbt/` 子目录（无 git 依赖、可改源码），外层包一层
生产化适配：data_access 读数、A 股交易约束、公司行动处理、numba 订单状态机、统一 CLI。

**定位:** 企业级 A 股横截面日频多因子量化项目的**回测执行层** —— 上游
riskfolio_qs 产出 `target_positions.parquet` 目标权重，本库负责把它变成真实成交、
真实现金、真实费后净值，并给出 24 组固定报告。

**版本:** 0.4.0 ｜ **仓库:** https://github.com/HKUST-QUANT-SOCIETY/vectorbt_qs (私有)
**文档索引:** [`docs/README.md`](docs/README.md) ｜ 从零跑通: [`docs/accurate_batch_quickstart.md`](docs/accurate_batch_quickstart.md) ｜ 目标仓位格式: [`docs/target_positions_format.md`](docs/target_positions_format.md)

---

## 它是什么 / 不是什么

**做什么：** 接收目标权重/目标仓位矩阵 → 按所选口径模拟成交（fast 分数股瞬时再平衡 /
accurate 逐日整手真实成交）→ 应用 A 股约束（涨跌停/停牌/T+1/整手/费税）→ 计入公司行动
（现金分红/送转）→ 输出组合净值、逐笔成交、订单台账、Barra-lite 暴露。

**不做什么：** 不产生信号/权重（riskfolio_qs、lightgbm_qs 产）、不评估因子（quant_evaluator
做）、不读原始行情库之外的数据源（数据一律走 data_access 优先 + COS fallback）。

## 两种执行口径（本库核心设计）

| | **Fast**（因子筛选） | **Accurate**（候选复核） |
|---|---|---|
| 用途 | 大量因子/分组的快速筛选 | 入库前候选策略的真实复核 |
| 成交单位 | 分数股（vectorbt 原生） | **A 股 100 股整手** |
| 现金/持仓 | 按目标权重瞬时再平衡 | **逐日维护真实现金与股数**（numba 状态机） |
| 费用/滑点 | 固定为零 | 方向性费率、最低佣金 5 元、滑点 |
| 停牌/涨跌停 | 忽略或目标层近似 | **按真实状态拒单**（涨停不买/跌停不卖） |
| 公司行动 | — | 除权除息日计入现金分红/送转（`StockDividend` 表） |
| 执行价 | VWAP 信号 → 次日执行 | VWAP / Open 执行，真实股数 |

两者共同点：**收盘生成信号 → 下一交易日执行**（与全平台 vwap-to-vwap 决策时钟一致）；
A 股默认只做多（各行权重和 ≤ 1）。

> 历史教训（已修）：wmat 曾把 `NaN=平仓` 处理，正确语义是 **NaN=维持现有持仓、0=平仓**。
> 输入矩阵的 NaN/0 语义见 [`docs/target_positions_format.md`](docs/target_positions_format.md)。

## 安装与运行

```bash
git clone https://github.com/HKUST-QUANT-SOCIETY/vectorbt_qs.git
cd vectorbt_qs
```

> **安装阻塞：** 外层 `requirements.txt` 声明 `numpy<2`、`pandas<3`、`numba<0.61`，
> vendored `vectorbt/pyproject.toml` 则声明 `numpy>=2.4.6`、`pandas>=3.0.3`、
> `numba>=0.66`，两套元数据互斥且当前没有经过验证的统一安装组合。`--no-deps`
> 只能绕过解析，不能证明运行兼容。协调并验证约束前，不推荐默认一键安装；后续应在
> 独立环境完成依赖求解，再按需安装 data_access 与 GPU/Rust 可选依赖。

### CLI（`python -m vectorbt_qs`）

| 命令 | 作用 |
|---|---|
| `list` | 列出内置 benchmark（B001 topN 等权 / B006 meanvar / B007 XGB 等） |
| `backtest -b B001 --plot` | 跑单个内置 benchmark，可出图 |
| `run configs/config.example.yaml` | 按 YAML 配置跑单次回测 |
| `batch configs/config.batch.example.yaml` | 参数网格批量回测（fast 口径） |
| `accurate-batch --positions target_positions.parquet --barra-root ... --output-root out --workers 4` | accurate 口径批量复核，**每组输出 24 组固定报告** |
| `accurate-benchmark` | accurate 口径跑内置 benchmark |

### Python API

```python
from vectorbt_qs.mvp.engine.runner import run_backtest, portfolio_report

target_weights = pd.DataFrame(data=..., index=pd.DatetimeIndex(...), columns=[...])
pf = run_backtest("ashare", target_weights, config={"init_cash": 10_000_000})
stats = portfolio_report(pf)
```

## 输入格式：目标权重矩阵

- `index` = 交易日、`columns` = 证券代码（`000001.SZ`）、`values` = 目标权重
- `0.05` = 持有 5% 多头、`-0.03` = 空头（美股）、`0` = **平仓**、`NaN` = **维持现状**
- accurate 模式也接受目标仓位 DataFrame（股数口径）

## A 股交易规则覆盖（accurate 口径全实现）

- **停牌过滤**：停牌日拒单
- **涨跌停**：涨停不买、跌停不卖（按 StockDailyBar 真实状态）
- **T+1**：当日买入次日才可卖（可卖数量单独跟踪）
- **印花税**：仅卖出收取（现行 0.05%）
- **佣金**：双边 0.025%，单笔最低 5 元
- **整手约束**：100 股整数倍（numba 订单生成器 `plan_ashare_orders`）
- **现金分红/送转**：除权除息日自动入账（`StockDividend` 表驱动）
- **成交量限制**：单股单日不超过当日成交量 10%（max_participation）
- **滑点**：可配置方向性滑点

## 关键模块

```
cli.py                # 全部 CLI 入口
configs/              # 示例 YAML（fast/accurate/batch profile）
mvp/engine/           # runner / execution（numba 状态机 plan_ashare_orders + Python parity oracle 对拍）/ batch / profiles（frozen 标准 profile standard_accurate_v2）
mvp/data/             # data_access read_arrow 优先（5-10× 快）+ COS fallback
mvp/constraints/      # A 股 / 美股约束规则（涨跌停/停牌/T+1/整手）
mvp/analysis/         # Barra-lite 风格暴露分析（exposure/factor_returns/factor_cov/specific_risk）+ 实验性归因
contracts/            # BacktestRequest / BacktestArtifact / ledger / orders / trades DTO（与 quant_platform 对接）
vectorbt/             # vendored vectorbt 1.1.0 源码（无上游 git，可改）
docs/                 # accurate 快速上手 / 双模式说明 / batch / 暴露分析 / 目标仓位格式
tests/                # 83 个回归测试（含 numba vs Python parity oracle 对拍）
```

## 数据需求（accurate 口径，lqtp COS 表）

| 表 | 用途 |
|---|---|
| `StockDailyBar` | 原始 OHLCV + 复权因子 + 停牌 + 涨跌停价 |
| `StockDividend` | 现金分红、送转股（公司行动入账） |
| `IndexDailyBar` + `IndexConstituent` | 基准净值曲线 / 成分股 |
| `StockIndustry` | 行业归属（暴露分析用） |
| Barra-lite 风险模型（可选，`v2_sbi_mvl_fullA`） | exposure / factor_returns / factor_cov / specific_risk 四件套 |

## 性能参考（32 核机器实测）

- B001 topN 等权：2294 天 × 5122 股 ≈ **46s**
- B006 meanvar：1433 天 × 300 股 ≈ 20s
- B007 XGB：599 天 × 5086 股 ≈ 15s

## 与 lightgbm_qs 的对接（上游全链路）

`lightgbm_qs/backtest_final_vectorbt.py` 调用本库：
`set_data_root()` 设数据根 → `run_backtest()` 跑组合 → `portfolio_report()` 出统计 →
`build_nav_curves()` 生成净值曲线。组合参数：topk=30、单股上限 5%、双边成本 5bp。

## 硬性规则

- 收益/决策口径与全平台一致：收盘信号、次日执行、vwap-to-vwap
- 回测结果**好得离谱先查泄露**（读 `MISTAKES_AND_LEAKAGE_LESSONS.md` 自查表）
- vendored vectorbt 源码可改，但改动要在 tests/ 里加对拍

## 依赖与接口（谁 import 谁）

- **依赖**：`data_access`（`mvp/data/adapter.py`，read_arrow 优先 + COS fallback）。
- **被谁调用**：`lightgbm_qs`（最终回测链）、`quant_platform`（BacktestRequest ↔ contracts DTO）。

## 相关仓库

- **data_access** — 数据层（唯一读通道）
- **riskfolio_qs** — 上游组合优化（产出 target_positions.parquet）
- **lightgbm_qs** — 上游模型链（调用方）
- **factor_engine / quant_evaluator** — 上游因子计算与评估
