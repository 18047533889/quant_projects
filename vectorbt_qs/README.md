# vectorbt_qs

A-share / US portfolio backtesting toolkit built on a vendored
[vectorbt](https://github.com/polakowo/vectorbt) **1.1.0**, adding market
constraints, corporate actions (cash dividends / stock splits), and unified
CLI/config entry points for accurate production-grade backtests.

**Version:** 0.4.0 ｜ **Repo:** https://github.com/HKUST-QUANT-SOCIETY/vectorbt_qs (private)
**Docs:** [`docs/README.md`](docs/README.md) ｜ Start here: [`docs/accurate_batch_quickstart.md`](docs/accurate_batch_quickstart.md)

## Two execution modes

| | Fast | Accurate |
|---|---|---|
| Purpose | factor screening (many factors/groups) | candidate strategy validation |
| Units | fractional shares | A-share 100-share lots |
| Cash/positions | weight rebalance | day-by-day real cash & shares |
| Fees/slippage | zero | directional fees, min commission, slippage |
| Halt / price-limit | ignored or approximated | rejected per real status |
| Corporate actions | — | cash dividends & splits booked on ex-date |
| Return basis | VWAP signal → next-day execution | VWAP / Open execution, real shares |

Both modes: signal at close → execute next trading day; A-share long-only by
default (sum of target weights ≤ 1).

## Install & run

```bash
git clone https://github.com/HKUST-QUANT-SOCIETY/vectorbt_qs.git
cd vectorbt_qs
pip install -r requirements.txt && pip install -e .
# data layer (optional but recommended): pip install -e ../data_access
```

```bash
python -m vectorbt_qs list                       # list built-in benchmarks
python -m vectorbt_qs backtest -b B001 --plot    # run benchmark
python -m vectorbt_qs run configs/config.example.yaml     # config-driven
python -m vectorbt_qs batch configs/config.batch.example.yaml  # grid search
python -m vectorbt_qs accurate-batch --positions target_positions.parquet \
    --barra-root /path/v2_sbi_mvl_fullA --output-root out --workers 4  # 24 fixed reports
```

Python API:

```python
from vectorbt_qs.mvp.engine.runner import run_backtest, portfolio_report

target_weights = pd.DataFrame(data=..., index=pd.DatetimeIndex(...), columns=[...])
pf = run_backtest("ashare", target_weights, config={"init_cash": 10_000_000})
stats = portfolio_report(pf)
```

## Input format

Target-weight matrix: index = trading days, columns = symbols (`000001.SZ`),
values = weight (0.05 = long 5%, -0.03 = short, 0 = close, NaN = hold).
See [`docs/target_positions_format.md`](docs/target_positions_format.md).

## Key modules

```
cli.py                # CLI: backtest / run / batch / list / accurate-batch
configs/              # example YAMLs (accurate/fast/batch profiles)
mvp/engine/           # runner, execution (numba state machine), batch, profiles
mvp/data/             # data_access-first adapter, COS fallback
mvp/constraints/      # A-share / US constraint rules
mvp/analysis/         # Barra-lite B/f exposure + experimental attribution
contracts/            # BacktestRequest/Artifact, ledger, orders, trades DTOs
vectorbt/             # vendored vectorbt 1.1.0 (source, no upstream git)
docs/                 # accurate quickstart, modes, batch, exposure, target format
tests/                # regression tests
```

## Data requirements (accurate)

- `lqtp_data/StockDailyBar` (raw OHLCV + adj factor + halt + price limits)
- `lqtp_data/StockDividend` (cash dividends, splits — booked on ex-date)
- `lqtp_data/IndexDailyBar` + `IndexConstituent` (benchmark NAV / constituents)
- `lqtp_data/StockIndustry` (industry names)
- Optional Barra-lite risk model (`v2_sbi_mvl_fullA`): exposure/factor_returns/
  factor_cov/specific_risk for B/f analysis

## A-share trading rules covered

Halt filter, price-limit (no buy at limit-up / sell at limit-down), T+1 sellable,
stamp tax (sell only), commission + min commission, slippage, 100-share lots,
cash dividends & splits, per-order volume cap (10% of day volume).

## Performance

- B001 topN equal-weight: 2294 days × 5122 stocks ≈ 46s
- B006 meanvar: 1433 × 300 ≈ 20s; B007 XGB: 599 × 5086 ≈ 15s
(Windows 11, Py 3.11, DuckDB + local parquet)

## Related repos

- **data_access** — data layer (preferred read path)
- **riskfolio_qs** — upstream portfolio optimizer that writes `target_positions.parquet`
- **factor_engine / quant_evaluator** — factor computation & evaluation upstream
