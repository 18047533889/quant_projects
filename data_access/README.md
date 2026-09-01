# data_access

Unified data read/write layer for the quant team. All code that reads or writes
parquet/CSV/factor values should go through this package.

**Version:** 0.10.x ｜ **Repo:** https://github.com/HKUST-QUANT-SOCIETY/data_access (private)
**User manual (中文):** [`docs/用户使用手册.md`](docs/用户使用手册.md) ｜ Index: [`docs/README.md`](docs/README.md)

## Why this exists

Before, everyone did `pd.read_parquet` + glob + merge with hard-coded paths and
no write discipline. This package provides one entry point, registry-based
datasets, cost-aware routing, PIT semantics, and governed writes.

## Install & first read

```bash
git clone https://github.com/HKUST-QUANT-SOCIETY/data_access.git
cd data_access
pip install -e ".[all]"
```

```python
from data_access import get_store
store = get_store()

df = store.read_frame(
    "ashare_stock_daily",
    columns=["TradeDate", "Symbol", "Close"],
    time_range=("2024-01-01", "2024-01-31"),
)

# unified read → handle (to_arrow / to_pandas / to_lazy / stream)
handle = store.read("ashare_stock_daily", columns=["TradeDate", "Symbol", "Close"],
                    time_range=("2024-01-01", "2024-03-31"),
                    filters={"Symbol": ["600000.SH"]})

# read many factors in one query
handle = store.read_factors(["mom_3d", "vol_20"], time_range=("2024-01-01","2024-12-31"),
                            layout="wide")  # datetime, asset, mom_3d, vol_20
```

HTTP service: `data-access-server --host 0.0.0.0 --port 8765` + `DataAccessClient`.

## Core capabilities

- **Registry**: `config/datasets.yaml` (73 datasets), logical fields in
  `config/semantic_fields.yaml` (56 fields: close/open/high/low/vwap/return_bp/
  adj_factor/...). Never bare `pd.read_parquet`.
- **Read**: `read`/`read_uri` (parquet/csv/feather/arrow/jsonl), Filter AST with
  DuckDB/Polars/PyArrow compilers, partition pruning + manifest min/max.
- **Cost routing**: `read_auto` routes by estimated scan cost (rows/bytes/columns/
  files/remote/selectivity).
- **Factors**: `read_factors` (UNION ALL / PIVOT), `get_factor_catalog`.
- **PIT**: `PITContract`, `read_cos_events_asof`, `read_asof`; manifest snapshot token
  (`manifest_version` / `is_snapshot_stale`).
- **Multi-dataset joins**: `read_joined` (exact / pit_asof) in DuckDB; `sql_relation`.
- **Write & publish**: `write_arrow` / `upsert` / `publish_from_staging` /
  `delete_rows` / `compute_and_write` → staging → publish.
- **COS**: mirror / remote / auto modes; `cos://` URIs; DuckDB httpfs via Secret Manager.
- **ClickHouse**: panel table reads, `ensure_factor_table` / `verify_factor_write`.
- **Governance**: `QueryBudget` (deadline cancel), `ManagedBatchReader`, security
  authorization, `data-access-quality` CLI.

## Layout

```
store.py          # public facade (DataAccessStore, get_store)
core/             # DuckDB engine, exceptions, audit
registry/         # datasets.yaml
read/             # read contracts, ReadHandle, PIT, SQL, factors, object_store
write/            # publish, upsert, generation, mutation lock
cos/              # mirror / remote / s3_duckdb / serving
clickhouse/       # panel read/write helpers
service/          # FastAPI + client
quality/          # quality CLI
config/           # datasets.yaml, semantic_fields.yaml
docs/             # 用户使用手册, PIT/COS contracts, architecture
tests/            # 161+ unit tests
```

## Data caliber (hard rules)

- **复权**: adjustment factors come from `adj_factor` / `adjusted_price_backward`
  (single source of truth). Factor computation uses 后复权 (backward-adjusted).
- **Returns**: `return_bp` is in basis points (×1/10000); vwap-to-vwap forward
  returns are the label convention everywhere downstream (see `label_bundle`).
- **PIT**: history-as-of reads must use PIT/asof paths, never future values.

## Related repos

- **factor_engine** — the computation engine that reads through this layer
- **riskfolio_qs** — reads benchmark / tradable / Return through this layer
- **vectorbt_qs** — A-share backtest adapter reads data here first
- **quant_evaluator / factor_preprocess / quant_platform** — use universe/calendar
  providers or storage adapters from here
