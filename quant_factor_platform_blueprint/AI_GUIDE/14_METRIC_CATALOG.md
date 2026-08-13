# 14 — QuantEvaluator Metric / Analyzer Catalog

本 Catalog 是 capability target，不代表第一版全部计算。所有 metric 必须有 tier/cost/dependencies。

## Quality / Distribution

- coverage, valid_count, missing/inf/zero/tie/unique ratios
- daily/industry/size coverage
- mean/std/MAD/IQR/skew/kurtosis/entropy
- outlier sensitivity, winsor sensitivity
- cross-sectional dispersion/effective breadth
- missingness predictiveness / missing treatment sensitivity

## IC / Forecast

- Pearson / Spearman / Kendall
- mean/median/std/ICIR/sign/hit
- t/HAC/block-bootstrap CI
- rolling/year/month/quarter
- CS OOS R2 / MSE / MAE
- pairwise ordering / tail classification

## Horizon / Timing

- h=1/2/3/5/10/20/40/60
- peak horizon / half-life / decay AUC/slope
- zero crossing / reversal
- T+1/T+2/T+3 execution delay survival

## Quantile / Shape / Tail

- Q1..Q10 / top-bottom spread
- monotonicity / slope / adjacent consistency
- top/bottom tail gain / middle noise
- linear/isotonic/spline fit diagnostics
- shape classification

## Long / Short / Economic Probe

- long / short / LS return
- Sharpe / Sortino / Calmar
- MDD / drawdown duration
- hit rate / gain-loss
- long-short asymmetry

## Stability / Drift

- year/quarter stability
- rolling worst windows
- sign/rank stability
- change point / pre-post break
- PSI / Wasserstein / rank drift

## Exposure / Neutralization

- size/free-float/industry/liquidity/turnover/beta/vol exposure
- exposure stability/concentration
- raw vs residual survival

## A-share Implementation

- microcap dependency
- liquidity buckets
- ST/suspension dependency
- limit dependency
- tradable/buyable/sellable IC
- board / IPO age
- benchmark relative robustness

## Turnover / Cost / Capacity

- rank turnover / TopK turnover
- probe portfolio turnover
- holding period
- ADV/amount/free-float participation proxies
- cost break-even / capacity proxy
- turnover netting benefit

## Novelty / Redundancy

- Pearson/Spearman
- PnL corr
- TopK/BottomK overlap
- exposure/horizon/regime similarity
- linear residual IC
- nonlinear residual IC
- conditional IC
- incremental OOS R2 / downstream utility (later)

## Multiple Testing / Robustness

- raw p / q / BH FDR
- family/campaign FDR
- DSR/PBO/SPA/Reality Check (T3)
- specification robustness distribution
- sign/specification survival
- worst-quantile specification performance

## Complexity / Mechanism

Complexity profile 通常由 FO/FE 提供，QE 可把其作为 evidence dimension。

Mechanism falsification analyzer：根据 hypothesis 的 expected signatures 检查 slice/horizon/exposure 结果，输出 SUPPORTED/MIXED/REJECTED，不让 LLM 自证机制。
