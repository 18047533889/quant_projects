# 19 — Detailed QuantEvaluator Metric Registry Blueprint

本页用于 AI 建 `MetricRegistry` 时逐项落地。Metric 名称可在 Contract Freeze 时微调，但语义、依赖和 tier 不应随意改变。

## A. Quality & Coverage（T0/T1）

| Metric | Tier | Shared inputs | Notes |
|---|---:|---|---|
| `quality.coverage` | T0 | valid_mask | 全面板有效率 |
| `quality.valid_count` | T0 | valid_mask | 有效观测数 |
| `quality.daily_coverage.mean/std/min` | T0/T1 | valid_mask by date | 日覆盖稳定性 |
| `quality.missing_rate` | T0 | valid_mask | 不填 0 |
| `quality.inf_rate` | T0 | values | Inf/-Inf |
| `quality.zero_rate` | T0 | values | 识别稀疏信号 |
| `quality.unique_ratio` | T0 | values | 横截面离散度 |
| `quality.tie_ratio` | T0 | rank stats | RankIC 重要 |
| `quality.constant_day_ratio` | T0 | daily dispersion | 常数截面比例 |
| `quality.stale_ratio` | T1 | optional age/freshness | 基本面等 |
| `quality.coverage.industry/size/liquidity` | T1 | context groups | 偏置诊断 |
| `quality.missingness_predictiveness` | T2 | missing mask + labels | 缺失本身是否带信息 |
| `quality.imputation_sensitivity` | T2 | alternative outputs | 由 FP policy 对照 |

## B. Distribution / Geometry（T0/T1）

- mean / median / std / MAD / IQR
- skewness / excess kurtosis
- entropy / concentration proxy
- percentile range p01-p99 / p05-p95
- outlier ratio by robust threshold
- winsor sensitivity
- cross-sectional dispersion mean/stability
- effective breadth / concentration
- positive/negative mass ratio
- tail mass ratio

## C. IC / Cross-sectional Forecast（T0–T2）

- `ic.pearson.daily/mean/median/std/ir/sign_ratio`
- `ic.rank.daily/mean/median/std/ir/sign_ratio`
- `ic.kendall.*`（成本更高，T1/T2）
- `ic.rank.tstat`
- `ic.rank.hac_tstat`, `hac_pvalue`
- bootstrap CI / block bootstrap CI
- cumulative IC
- rolling 20/60/120/252（参数化 window，不复制 metric 名）
- groupby year/quarter/month
- cross-sectional OOS R2
- cross-sectional MSE/MAE
- pairwise ordering accuracy
- top/bottom quantile classification accuracy

## D. Horizon / Decay / Delay（T1）

参数化 horizon，推荐 1/2/3/5/10/20/40/60：

- IC(h), RankIC(h), tail spread(h)
- peak horizon
- half-life
- decay slope / decay AUC
- zero-cross/reversal horizon
- horizon sign stability
- execution delay survival: delay=1/2/3/... with fixed holding semantics

必须区分 `horizon`（预测终点）与 `delay`（信号生成后晚几天才执行）。

## E. Quantile / Shape / Tail（T1）

- quantile returns Q1..Q10（quantile count 参数化）
- top-bottom spread
- quantile monotonicity Spearman
- linear slope / rank slope
- adjacent consistency
- top-tail gain / bottom-tail gain
- middle noise
- tail concentration
- long-tail/short-tail asymmetry
- linear fit R2
- isotonic fit score
- spline/piecewise gain（T2）
- shape class + confidence

## F. Long / Short / Probe Portfolio（T1）

分别对 long, short, LS：

- mean/annualized return
- annualized volatility
- Sharpe / Sortino / Calmar
- max drawdown / duration
- hit rate / positive month ratio
- gain-loss ratio / downside deviation
- long contribution / short contribution / asymmetry

Probe portfolio 必须明确 weighting/rebalance/execution assumptions；不是正式 portfolio optimizer。

## G. Turnover / Cost / Capacity（T0–T2）

- rank turnover
- topK/bottomK membership turnover
- probe portfolio turnover
- average holding period
- one-way/two-way turnover convention（明确）
- ADV/Amount participation proxy
- free-float participation proxy
- cost break-even
- simple net alpha under configured cost
- pair/cluster turnover netting benefit（T2）
- capacity proxy / crowding proxy（仅现有字段能支持的版本）

## H. Stability / Change / Drift（T1/T2）

- annual/quarterly consistency
- rolling mean/std
- worst 20/60/120-day window
- sign survival
- rank of factor quality over time
- change-point score / last break / pre-post delta
- PSI / Wasserstein drift
- exposure drift
- coverage drift
- health score components（不强制合成万能 score）

## I. Exposure / Neutralization Survival（T1/T2）

Exposure：

- size / free-float size
- industry
- liquidity / turnover
- beta / volatility（若 context 提供）
- benchmark membership

Diagnostics：

- exposure magnitude/stability/concentration
- raw IC vs industry residual
- raw vs size residual
- raw vs liquidity residual
- raw vs combined residual
- survival ratio / sign survival

## J. A-share Robustness（T1/T2）

- All A / CSI-style benchmark slices（按可用 index membership）
- exclude bottom 10/20/30% market cap
- top 50/70% liquidity
- Main/ChiNext/STAR
- IPO age thresholds
- exclude ST
- exclude suspended
- tradable universe
- buyable long / sellable short
- limit-up/down dependence
- free-float size dependence
- execution-delay robustness

## K. Novelty / Incremental Information（T0–T3）

Cheap：

- Pearson/Spearman candidate-neighbor
- PnL corr
- TopK/BottomK overlap
- exposure/horizon/regime vector distance

Exact/conditional：

- linear residual IC
- conditional IC
- nonlinear residual IC（advanced）
- portfolio PnL residual alpha
- incremental CS OOS R2
- group/family ablation delta（未来模型端接入后）

## L. Statistical Search Control（T2/T3）

- raw p-value
- BH q-value / FDR
- family/campaign adjusted result
- bootstrap significance
- DSR
- PBO
- SPA / Reality Check
- specification distribution / sign survival / non-standard error summary

这些不应成为 T0 默认指标。

## M. Production Health（T1/T2 Streaming subset）

当天可立即：coverage, dispersion, rank turnover, exposure drift, value distribution drift, graph membership drift（FA）。

Label 成熟后：realized IC/horizon performance/decay。禁止用尚未成熟 label 更新今天的 factor 状态。

## Registry Implementation Status

建议 registry 支持 `status = planned/reference_ready/fast_ready/production/research_only/deprecated`，这样可以先把 150+ capability 定义完整，再分批实现，不需要为了“名义支持”写低质量代码。
