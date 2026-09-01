# data_access — 团队统一数据读写层

**版本:** 0.10.x ｜ **仓库:** https://github.com/HKUST-QUANT-SOCIETY/data_access (私有)
**用户手册（中文）:** [`docs/用户使用手册.md`](docs/用户使用手册.md) ｜ 文档索引: [`docs/README.md`](docs/README.md) ｜ 历史报告: [`docs/archive/`](docs/archive/)

## 它是什么

全团队统一的量化数据读写入口。所有读/写 parquet、csv、因子值的代码都应走这里，
而不是直接 `pd.read_parquet`。提供：数据集注册表、成本路由读、PIT 语义、受控写、COS/ClickHouse。

## 安装与第一个读

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

# 统一读 → ReadHandle（to_arrow / to_pandas / to_lazy / stream）
handle = store.read("ashare_stock_daily", columns=["TradeDate", "Symbol", "Close"],
                    time_range=("2024-01-01", "2024-03-31"),
                    filters={"Symbol": ["600000.SH"]})

# 一次读多个因子（单查询）
handle = store.read_factors(["mom_3d", "vol_20"], time_range=("2024-01-01","2024-12-31"),
                            layout="wide")  # datetime, asset, mom_3d, vol_20
```

HTTP 服务：`data-access-server --host 0.0.0.0 --port 8765` + `DataAccessClient`。

## 核心能力

- **注册表**：`config/datasets.yaml`（73 个数据集），逻辑字段在 `config/semantic_fields.yaml`
  （56 个：close/open/high/low/vwap/return_bp/adj_factor/...）。
- **读**：`read`/`read_uri`（parquet/csv/feather/arrow/jsonl），Filter AST（DuckDB/Polars/PyArrow 三编译器），
  分区裁剪 + manifest min/max。
- **成本路由**：`read_auto` 按预估扫描成本（行/字节/列/文件/远端/选择性）路由。
- **因子**：`read_factors`（UNION ALL / PIVOT）、`get_factor_catalog`。
- **PIT**：`PITContract`、`read_cos_events_asof`、`read_asof`；快照令牌 `manifest_version` / `is_snapshot_stale`。
- **多数据集 join**：`read_joined`（exact / pit_asof，DuckDB 内）；`sql_relation`。
- **写与发布**：`write_arrow` / `upsert` / `publish_from_staging` / `delete_rows` / `compute_and_write`。
- **COS**：mirror / remote / auto 三模式；`cos://` URI；DuckDB httpfs + Secret Manager。
- **ClickHouse**：面板表读、`ensure_factor_table` / `verify_factor_write`。
- **治理**：`QueryBudget`（deadline 主动取消）、`ManagedBatchReader`、安全授权、`data-access-quality` CLI。

## 目录

```
store.py          # 对外门面（DataAccessStore, get_store）
core/             # DuckDB 引擎、异常、审计
registry/         # datasets.yaml
read/             # 读契约、ReadHandle、PIT、SQL、因子、object_store
write/            # publish、upsert、generation、mutation lock
cos/              # mirror / remote / s3_duckdb / serving
clickhouse/       # 面板读写
service/          # FastAPI + client
quality/          # 质量 CLI
config/           # datasets.yaml, semantic_fields.yaml
docs/             # 用户手册、PIT/COS 契约、架构、（archive 历史报告）
tests/            # 161+ 单测
```

## 口径（红线）

- **复权**：复权因子唯一来源是 `adj_factor` / `adjusted_price_backward`；因子计算用后复权。
- **收益**：`return_bp` 单位是 bp（×1/10000）；下游标签一律 **vwap-to-vwap** 后复权。
- **PIT**：历史快照读走 PIT/asof，禁止未来函数。

## 依赖与接口

- **被谁调用**：
  - `factor_engine` → `storage/data_access_source.py`（读数）
  - `riskfolio_qs` → benchmark / tradable / Return
  - `vectorbt_qs` → `mvp/data/adapter.py`（行情）
  - `quant_evaluator` → `adapters/data_access.py`（universe/日历 provider）
  - `factor_preprocess` → `adapters/data_access.py`
  - `quant_platform` → `app/adapters/data_access_storage.py`、`data_access.read.object_store`
  - `alphaprobe` / `lightgbm_qs` → 行情与标签

## 相关仓库

- **factor_engine** — 通过本层读数据的计算引擎
- **riskfolio_qs / vectorbt_qs / quant_evaluator / factor_preprocess / quant_platform / alphaprobe** — 见上
