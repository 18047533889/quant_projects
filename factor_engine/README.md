# factor_engine

DSL-based factor computation engine for A-share / US equity research.
Compile factor formulas (DSL / expression trees), execute on vectorized backends
(Pandas / Polars / DuckDB / ClickHouse SQL), and materialize to the factor lake.

**Version:** 0.3.x ｜ **Repo:** https://github.com/HKUST-QUANT-SOCIETY/factor_engine (private)

> Chinese full guide → [`docs/FactorEngine完全指南.md`](docs/FactorEngine完全指南.md)
> HTTP service → [`service/README.md`](service/README.md)
> Operator reference → [`docs/dsl_operators_reference.md`](docs/dsl_operators_reference.md)

---

## What it does

You write a formula as DSL, e.g. `rank(ts_mean(close, 20))`; the engine reads
market data from **data_access**, computes factor values on the `(date ×
instrument)` grid, and optionally materializes them to the factor lake.

It does **not** do: data cleaning, factor evaluation/backtesting, model training.
It only computes factors and writes the factor lake.

## Install & first factor

```bash
git clone https://github.com/HKUST-QUANT-SOCIETY/factor_engine.git
cd factor_engine
pip install -e .
# also recommended: pip install "data-access @ git+https://<TOKEN>@github.com/HKUST-QUANT-SOCIETY/data_access.git"
```

```python
from factor_engine.api import col, rank, ts_mean
from factor_engine.api.factor import Factor
from factor_engine.backend.factory import build_backend
from factor_engine.runtime.engine import FactorEngine
from factor_engine.storage.factory import build_data_source

source = build_data_source({
    "type": "data_access",
    "dataset": "ashare_stock_daily",
    "fields": {"close": "Close"},
    "start_date": "2024-01-01",
    "end_date": "2024-03-31",
})
engine = FactorEngine(backend=build_backend("auto"), data_source=source)
factor = Factor(name="mom20_rank", expr=rank(ts_mean(col("close"), 20)))
out = engine.run(factor)
print(out["result"].head())   # MultiIndex Series (TradeDate, Symbol)
```

Or config-driven: `FactorEngine.run_from_config("examples/config_driven_factor.yaml")`.
Or HTTP: `pip install -e ".[service]"` then `factor-engine-serve --port 8088`
(routes: `/factor-engine/operators`, `/validate-spec`, `/jobs/compute`,
`/jobs/materialize`, `/jobs/{id}/artifacts`).

## Architecture

```
Factor(expr) → expr AST → ir.Analyzer → planner (Lowerer + Optimizer + CSE)
    → backend.execute (auto: SQL pushdown → Polars → Pandas) → MultiIndex factor values
```

| Layer | Dir | Role |
|---|---|---|
| DSL API | `api/` | `col()`, `rank()`, `ts_mean()`, `Factor`, `parse_expr` |
| AST | `expr/` | ColumnRef / Literal / CleanedCall |
| Analysis | `ir/` | Expr → IR, lookback derivation |
| Planning | `planner/` | IR → plan, CSE, cost model |
| Execution | `backend/` | pandas / polars / duckdb_sql / clickhouse_sql / auto |
| Operators | `cleaned_operators/` | single implementation source (1737 canonical, daily surface 1238) |
| Storage | `storage/` | data_access source, parquet/CH, materializer → factor lake |
| Runtime | `runtime/` | `FactorEngine`, batch scheduler, resource broker, materialize services |
| Service | `service/` | thin FastAPI adapter (port 8088) |

## Operators

- Canonical catalog: `cleaned_operators/docs/operators_catalog.json` (1737; daily 1238,
  extended 480, research 8). Machine-readable allowlist: `docs/dsl_allowlist.json` (1421).
- Semantics: `docs/operators_semantics.md`. Local enumeration:
  `from factor_engine.api.operator_registry import build_dsl_allowlist`.
- Common: `rank`, `zscore`, `ts_mean`, `ts_std`, `delay`, `group_rank`,
  `group_neutralize`, `protected_div`, `SMA/EMA`, `ts_rsi/ts_macd/ts_atr/...`, `trade_when`, `neutralize`.

## Reading data / writing factors

- Read: `data_source.type: data_access`, dataset names in data_access `config/datasets.yaml`
  (e.g. `ashare_stock_daily`, `ashare_stock_minute_adj`, `factor_lake`).
- Write: `engine.materialize(factor)` → local lake / `factor_lake_staging` →
  `lake_publish.publish_factor_lake()`. Production profile: `examples/profiles/prod.yaml`.

## Return / timing caliber (hard rule)

This repo computes **values only**. Evaluation/training labels must be
**vwap-to-vwap** (`VWAP_{t+H}/VWAP_t - 1`, 后复权); decision clock = close signal →
next-day VWAP execution. See `docs/adr_backtest_target_position.md` §13 (no look-ahead)
and the platform `modeling` package for `DecisionClock` / `LabelContract`.

## Tests

```bash
cd factor_engine && PYTHONPATH=. pytest tests/ -q    # 1144+ tests
RUN_REAL_PARQUET_SMOKE=1 pytest tests/test_real_data_factor_smoke.py -v  # needs real data
```

## Related repos

- **data_access** — the data layer this engine reads (https://github.com/HKUST-QUANT-SOCIETY/data_access)
- **quant_evaluator** — evaluate factors / produce evidence
- **factor_optimizer** — factor mutation/search (uses FE to validate & hash formulas)
- **factor_assets** — factor identity/registry (uses FE canonical hash)
- **alphaprobe** — DL/RL mining (fe_bridge evaluates mined expressions through FE)
