# 15 — FactorOptimizer Diagnosis/Mutation Catalog

## Mapping Principles

- Diagnosis 是优先级，不是自动答案。
- 一次 mutation 尽量只改变一个机制，便于 attribution。
- parent/child 必须有 edit distance / changed parameters。
- child 必须重新 QE；LLM 文本解释不算验证。

## Common Mappings

| Diagnosis | Candidate Families | Avoid |
|---|---|---|
| Top-tail only | percentile, sigmoid, clipped tail score | high-degree polynomial search |
| Bottom-tail only | bottom percentile/tail score | symmetric transform forcing |
| U-shape | abs distance, two-sided | simple monotonic rank |
| Outlier sensitive | rank, MAD/quantile winsor, robust scale | extreme clipping without test |
| High turnover | EMA, hysteresis, persistent component | arbitrary longer window only |
| Fast decay | delay-aware smoothing, faster execution tag | assuming T close execution |
| Industry dependent | raw+residual/soft neutralization | blanket 100% neutralization |
| Size/liquidity dependent | residual/scaling/channel | deleting factor solely for exposure |
| Long-only alpha | asymmetric long representation | forcing symmetric LS |
| Fundamental stale | freshness/report-age conditioning | using report period as publication time |
| Parameter spike | neighboring search / plateau | keeping single best point |

## A-share Data-aware Grammar

Production allowed domains based on current known data：

- price/volume daily
- minute OHLCV -> daily aggregation
- valuation/market cap/free float/turnover
- PIT fundamentals by publication time
- industry/index membership
- ownership/top holders/pledge/freeze
- ST/suspend/limits/board/IPO age

Production-disallowed without new data：Level2/order book/analyst consensus/news full text/northbound true flow 等。

## Search Budget

每个 campaign 显式：

- max candidates
- max full-history evaluations
- max T2 robust evaluations
- max LLM calls
- max compute time/cost

Search engine 永远不得静默突破预算。
