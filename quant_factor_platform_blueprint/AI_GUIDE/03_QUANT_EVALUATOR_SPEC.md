# 03 — QuantEvaluator Development Specification

## 3.1 定位

`quant_evaluator` = 高性能、独立、Batch-First 的 Quantitative Evidence Engine。

只回答：

- 信号是否有统计/经济证据？
- 证据在哪些时间、股票池、流动性、行业、执行条件下成立？
- 因子有什么 shape/decay/fragility/dependency？

不回答：

- 因子公式如何计算（FE）
- 因子怎么修改（FO）
- 是否进入资产库（FA）
- 模型如何训练（Modeling）

## 3.2 Public API

必须支持轻量函数与统一 evaluate：

```python
from quant_evaluator import rank_ic, evaluate

x = rank_ic(factors, returns)

bundle = evaluate(
    factors,
    labels=returns,
    context=context,
    metrics=["coverage", "ic.rank.mean", "ic.rank.ir"],
    where={"year": 2026},
)
```

核心模型：`Metric × Slice × GroupBy`，禁止为年份/股票池复制 metric 名称。

## 3.3 Input Contracts

### FactorBatch

推荐字段：

- factor IDs
- dates
- assets
- values
- layout metadata
- dtype
- optional validity mask
- optional value reference/hash

值可以是：Arrow/NumPy/Polars/Pandas adapter。Core 内部标准化为高效 block view。

### LabelBundle

明确：

- horizon
- label semantics
- label start/end timing
- return convention
- optional execution convention

QE **不自行猜测** `shift(+5)` 还是 `shift(-5)`；label contract 不明确则拒绝高级评估。

### EvaluationContext

可选提供：

- universe mask
- industry
- size/free-float size
- liquidity/turnover
- beta/volatility
- ST/suspend/limit/buyable/sellable
- benchmark membership
- IPO age/board

QE 不负责从数据库组装这些；DA adapter 可以帮助构造。

## 3.4 Metric Registry

每个 `MetricSpec` 至少：

- `name`
- `version`
- `tier`: T0/T1/T2/T3
- required inputs
- shared intermediates
- supported layouts/dtypes
- streaming support
- groupby support
- estimated cost class
- reference implementation
- fast implementation
- numerical tolerance
- minimum observations

注册表是逻辑清单，不要让 metric 装饰器产生隐式不可追踪副作用。

## 3.5 Shared Intermediate Resolver

只做轻量 dependency resolution：

```text
factor values
  -> valid mask
  -> cross-sectional ranks
  -> daily rank IC
     -> mean / IR / yearly / rolling / HAC
```

共享中间量优先内存 cache；第一版不建跨任务磁盘 Artifact Platform。

建议共享：

- valid masks
- factor ranks
- label ranks
- daily IC series
- quantile assignments
- top/bottom masks
- simple probe weights
- exposure matrices
- horizon labels

## 3.6 Performance Layout

不能整体构造 `T×N×K=2500×5000×10000`。

使用：

- date chunk
- factor block
- asset axis

典型 block：64–512 因子；运行时根据内存预算决定。

Fast kernel 原则：

- 大 scan/join：交给 DataAccess/DuckDB
- 表面转换：Arrow/Polars
- 数学 hot path：NumPy/Numba
- 统计 reference：NumPy/SciPy/statsmodels
- 累积统计默认 float64，即使输入 factor value 可用 float32

禁止每因子 Python `groupby(date).corr()`。

## 3.7 RankIC Semantics

默认建议：

- 每日横截面 Spearman：先 average-rank，再 Pearson corr ranks
- NaN pairwise drop
- cross-section 有效资产数 `< min_assets` -> NaN
- factor 当日常数/label 常数 -> NaN
- ties 使用 average rank
- 汇总时不把 NaN 当 0
- `ICIR = mean(IC) / std(IC)`；是否年化必须用不同 metric 名称/parameter 明示，禁止隐式年化

所有规则必须写进 MetricSpec 与 tests。

## 3.8 Metric Tiers

### T0 Mining Fast Gate

- coverage / valid count / unique/tie/zero/inf
- cross-section dispersion
- Pearson IC / RankIC mean
- IC std / ICIR / sign ratio
- simple quantile/top-bottom spread
- rank turnover proxy
- simple correlation novelty

### T1 Core

- yearly/monthly/rolling IC
- Kendall（按需）
- multi-horizon / decay / delay
- quantile monotonicity / slope
- shape diagnosis
- long/short/LS probe
- size/industry/liquidity dependency
- tradable / buyable / sellable diagnostics
- microcap/board/IPO-age robustness
- drawdown/Sharpe/Sortino/Calmar for probe portfolio

### T2 Admission/Robustness

- HAC t/p
- block bootstrap CI
- specification robustness
- neutralization survival
- residual/conditional novelty
- cost/capacity proxies
- turnover netting
- change point
- missingness robustness
- FDR/q-value

### T3 Research

- DSR/PBO/SPA/Reality Check
- advanced conditional dependence
- large bootstrap
- partial distance correlation / mutual information
- advanced multiple testing

## 3.9 Diagnosis

输出 `FactorDiagnosis`，不要只输出数字表。

建议字段：

- predictive shape
- horizon peak / half-life
- turnover class
- size/industry/liquidity dependency
- microcap dependence
- tradability dependence
- long-short asymmetry
- stability/change-point state
- nonlinear gain
- residual information class
- missing/outlier sensitivity
- parameter fragility（若由 optimizer feedback 提供）

Shape classes：LINEAR_MONOTONIC / TOP_TAIL / BOTTOM_TAIL / U_SHAPE / INVERTED_U / CONVEX / CONCAVE / THRESHOLD / SATURATING / BIPOLAR / NON_MONOTONIC / NOISE。

## 3.10 A-share Module

仅使用现有数据可支持能力：

- liquidity / turnover / market cap / free-float cap dependence
- ST / suspension
- actual high/low limit derived tradability context
- buyable/sellable conditional metrics
- board / IPO age
- benchmark membership robustness
- minute-to-daily factor evaluation（factor values 由 FE 生成）

不要假装拥有 Level2、订单簿、分析师一致预期、可靠新闻全文等数据。

## 3.11 Output

`EvaluationBundle`：

- summary metrics
- grouped/sliced metrics
- diagnostics
- warnings
- metric versions/config
- optional series refs (daily IC etc.)
- data/evaluation context hash

大 series 不强制嵌入 JSON，可返回 Arrow/Parquet ref 或 in-memory object。

## 3.12 Tests

必须：

- reference/fast parity
- ties/NaN/constant/min-assets
- sign inversion metamorphic test
- positive monotonic transform RankIC invariance
- batch/chunk parity
- order invariance
- float32/float64 tolerance
- streaming/batch parity（支持 streaming 的指标）
- external parity sample：Alphalens/LQTP/旧 reference 仅作交叉检查，不作为绝对真值

## 3.13 Migration Priority

P0：coverage, IC/RankIC, IC summary, quantile, turnover, probe portfolio, exposure basics。

P1：horizon/decay, shape, A-share robustness, neutralization survival, HAC/bootstrap。

P2：conditional novelty, specification robustness, multiple testing, change points。

P3：research-only tests。
