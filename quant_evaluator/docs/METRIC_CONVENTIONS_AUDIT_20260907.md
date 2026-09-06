# QuantEvaluator calculation conventions audit — 2026-09-07

## Assessment: needs revision before claiming execution-realistic results

This is a source-path audit, not a certification of every operator or all CPU/GPU
implementations. Existing HTML/results need recomputation after a convention change.
Tests: `quant_evaluator/tests/test_metric_conventions.py`,
`quant_evaluator/tests/test_gross_exposure_contract.py`, and
`tests/test_incremental_report_mainline.py`. No new parallel repair pipeline is used.

## Definitions and ownership

| Quantity | Contract / authority | Limitation or remaining work |
|---|---|---|
| Signal / label | Report uses signal at t and adjusted VWAP(t+2)/VWAP(t+1)-1; named date/security alignment | Not an executable VWAP fill guarantee; timezone Asia/Shanghai; audit instrument history and adjustment revisions |
| Direction | Training window alone, fixed afterward; optimize already-directed inputs with direction +1 | General adapter defaults can still choose direction on its entire supplied sample; non-report callers must provide a training window/fixed direction |
| RankIC | Daily cross-sectional Spearman, average ties; summarize daily IC | Whole-universe rank association, not tail-portfolio return or profit assurance |
| IC IR | Mean daily IC / sample standard deviation, unannualized report ratio | Not Sharpe; overlapping labels and serial correlation invalidate naive significance assumptions |
| Universe | Signal-time observed adjusted price; no forward-return-dependent reselection | Listing, delisting, suspensions, industry, ST, price limits and borrowability require independent point-in-time controls |
| Long/short positions | `equal_gross_weights`: equal amount per selected stock across both sides; sum absolute weights=1; full short margin | Equal amount is not equal share count or volume. Equal legs only when counts match. No integer lots or borrow availability simulation |
| Missing side | New equal-gross API stays in cash if either side is empty; report additionally rejects invalid quantile observations | Do not fabricate portfolio performance from an undefined bucket |
| Fees | Absolute target-weight changes, including first entry and sign changes, times one-way commission | Not drift-aware executed turnover. Taxes, impact, slippage, borrow, financing excluded. Do not label all-in net |
| Missing held return | New equal-gross API invalidates day; optional explicit zero-fill is a flat-mark assumption | Does not renormalize portfolio using future validity. Decile averages still exclude missing labels: not the same completeness policy as strict LS |
| Wealth | `compute_wealth_curve` / aligned equivalent: initial equity 1, periodic compound return, zero absorbing | No recapitalization or margin debt model. Missing aligned observations remain gaps |
| CAGR | `compute_compound_annualized_return`: exp(sum(log1p(r))*periods_per_year/n)-1; <=-100% return makes terminal loss absorbing | Default drop annualizes observed periods; requires coverage disclosure, not elapsed-calendar annualization |
| Maximum drawdown | Initial capital included, max(1 - wealth/running peak), positive magnitude; zero equity =>100% | Not the difference between two cumulative leg NAVs |
| Calmar | Same CAGR authority / positive maximum drawdown | Zero/undefined drawdown =>NaN, not an infinite score |
| Sharpe | Arithmetic mean excess return / sample volatility * sqrt(periods/year); annual RF divided by frequency | Serial correlation, overlapping returns and deterministic/short samples need separate treatment; RF convention differs from geometric annual RF conversion |
| Drawdown duration | Library period-based duration; missing time policies vary by caller | Report drops missing returns before some metrics, which compresses time; not certified elapsed-trading-day duration with internal gaps |
| Quantile spread | `registry_adapters.compute_quantile_spread_value`: top minus bottom, a statistical spread | Intentionally not a funded 100%-gross account return; must not compound or label it as that portfolio |
| Multi-day cohorts | `probe_portfolio/_core.compute_cohort_pnl`: overlapping H-period cohorts, default +/-0.5 per cohort | Separate clock/weight contract; unequal buckets are equal amount per side, not globally equal amount. Holding-window edges, truncation, daily weight drift vs fees require dedicated tests |
| Optimization | Validation-only profiles; no test-window selection; training direction locked; default winner FE DSL replay checked | Alternatives only candidates until replay; neutralization/PIT exposure data not universally certified; multiple testing remains |
| CPU/GPU | Report falls back to CPU when GPU quantile/minimum-universe contract differs | GPU parity not certified merely by GPU availability; do not disable gates for speed |
| Publication | Manifest + hashed series artifact, source identity/time; no render-time metric recomputation | Legacy artifacts do not inherit a new portfolio convention. Full library fingerprint and raw-matrix identity should be persisted, not just main-script hash |

## Reproduced errors corrected in this pass

1. Calmar used mean(r)*252 while headline annual return used CAGR. They now use
   a single CAGR implementation. Repeated [+10%, -8%] previously gave Calmar
   31.5 vs CAGR/MDD 43.6894.
2. Headline annual return still compounded through negative equity after NAV
   had been fixed. [+10%, -120%, -200%, +1000%] previously produced huge positive
   annual return; now -100%.
3. A 3-D factor batch with a documented 2-D validity mask indexed a nonexistent
   third axis. It now broadcasts explicitly, with shape/dtype checks.
4. Empty Sharpe input returned a drawdown-shaped tuple. It now returns the same
   scalar/vector contract as nonempty input.

## Further high-priority risks (not declared fixed)

- Decile missing-label exclusion vs strict portfolio invalidation: synchronize
  before asserting every decile and LS point has identical observation coverage.
- Historical flat modules/cap-weighted APIs and cohort diagnostics have different
  capitalization conventions. Preserve names explicitly or route funded-return
  interfaces through the canonical weights API with migration tests.
- Cohort implementation uses target weights on each day but entry/exit costs only;
  verify intended constant-share vs daily rebalanced strategy before changing it.
- New-stock extreme contributions, adjustment breaks and stale financial vintages:
  trace top daily PnL contributors and source timestamps; do not clip unexplained
  large results or flip historical losing subperiods.
- Calendar-month metrics vs fixed 21-observation blocks: fixed blocks are not
  calendar months. Rename or supply dates before interpreting monthly hit rates.
- Validate all-in costs, tradability, short margin/account financing, share rounding,
  and constrained fills in an execution backtester; factor evaluation alone is
  insufficient evidence of realizable A-share returns.

## Release checks

### Verified regression result

2026-09-07: **142 passed, 2 warnings in 73.62 seconds** using the server-c
project virtual environment. The suites were `test_metric_conventions.py`,
`test_gross_exposure_contract.py`, `test_drawdown_evidence.py`,
`test_portfolio_stats_registry.py`, `quant_evaluator/tests/metrics`, and
`tests/test_incremental_report_mainline.py`.
The registry hand oracle previously expected an unfunded 200%-gross spread;
it now independently sums count-weighted leg returns divided by total selected
names, matching the authorized equal-amount 100%-gross portfolio contract.
Both warnings concern a test operator without PhysicalImplementationSpec;
this does not certify production operator eligibility or all GPU implementations.
The run does not establish that historical artifacts were recomputed, nor that
the rendered HTML uses this version. Publication requires new evaluation evidence.

Run the above regression suites plus risk/registry/metric suites. For each change,
check finite/missing/empty/constant/zero-equity inputs, masks and axis permutations,
cash entry/exit and reversals, uneven bucket sizes and ties. Reconcile direct daily
PnL with persisted wealth and metric cards. Recompute, rather than relabel, old results.
Record unresolved failures separately; passing the selected suite is not full-library
certification.
