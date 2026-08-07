# `data_access` — 团队统一数据读写入口

> **一句话定位**：所有读写 parquet 的代码，都应该走这里。  
> **仓库**：https://github.com/HKUST-QUANT-SOCIETY/data_access （组织私有仓，需有权限）  
> **磁盘目录**：本仓库根目录即包内容；代码里永远 `import data_access`。  
> **对外使用说明**：**[docs/用户使用手册.md](docs/用户使用手册.md)**（只看这一份即可；含完整 HTTP 服务）。  
> 文档索引：[docs/README.md](docs/README.md)  
> 维护：量化基础平台组｜更新：2026-08-07｜包版本：**0.4.0**

## 为什么有这个模块

以前各处都是 `pd.read_parquet` → 拼 glob → 合并 DataFrame，路径硬编码、性能不齐、写盘无规范。

**当前能力（0.4.x）—— Universal Quant Data IO Layer**：

1. **统一读入口**：`read()` / `read_uri()` 返回 `ReadHandle`（`.to_arrow()/.to_pandas()/.to_polars()/.to_lazy()/.stream()`）
2. **多文件格式**：`format:` 字段 + FormatAdapter——parquet / csv / csv.gz / tsv / jsonl / arrow / feather；`_build_select_sql` 不再写死 `read_parquet`
3. **通用过滤**：`filters=` 支持 Filter AST（Eq/Ne/Lt/Le/Gt/Ge/Between/In/NotIn/IsNull/NotNull/And/Or/Not），DuckDB / Polars / PyArrow 三编译器
4. **路径裁剪**：Partition Planner 按 time_range 展开 hive/日期路径 + **Dataset Manifest**（`_manifest.parquet`）文件级 min/max 裁剪，避免 `**/*.parquet` 全量 glob
5. **成本路由**：`read_auto` / `read()` 按 `estimated_scan_cost`（rows/bytes/columns/files/remote/selectivity）路由，不只按行数
6. **因子批量读**：`read_factors(factor_ids=[...])` 单查询 UNION ALL / 宽表 PIVOT + `FactorCatalog`
7. **StorageBackend**：local / s3 / cos 统一 `storage:` 声明；COS/S3 DuckDB 接入走 **Secret Manager**（`CREATE SECRET`，老版本回退 `SET s3_*`）；DuckDB 版本能力层自动跳过 deprecated PRAGMA
8. **企业级治理**：QueryBudget 超时**主动取消**（deadline + interrupt）、`ManagedBatchReader` 流资源生命周期、升级版质量契约
9. **读 API 兼容**：`read_arrow` / `read_frame` / `load_columns` / `read_arrow_stream` / `read_auto` / `scan_polars` 全部保留
10. **DuckDB 共享引擎**：跨线程 buffer pool / footer cache；HTTP `/v1/read_uri` `/v1/factors` `/v1/factors/read`
11. **写与发布**：`write_arrow` / `upsert` / `publish_from_staging` / `delete_rows` / `compute_and_write`
12. **COS**：mirror / remote / auto；可执行面板与 **PIT 契约**；HTTP 服务 + ClickHouse

**与 factor_engine**：读数一律 `data_source.type: data_access`；计算默认 `backend.type: auto`。见 [factor_engine/README.md](../factor_engine/README.md)。

## 目录结构

```
dataaccess/                 # 磁盘名；Python 包 = data_access
├── store.py                # 对外 API 门面
├── __init__.py             # get_store / QueryBudget / COS 契约导出
├── core/                   # 引擎、异常、命名空间、审计
├── registry/               # datasets.yaml
├── read/                   # 读契约、预算、SQL
├── write/                  # publish / upsert
├── cos/                    # 镜像、远程直读、PIT
├── clickhouse/
├── service/                # FastAPI + client
├── quality/                # 质量 CLI
├── deploy/                 # Docker / K8s / systemd
├── ops/
├── config/datasets.yaml
├── docs/
└── tests/
```

入口：`from data_access import get_store`。

---

## pip 安装

```bash
# 从组织仓安装（推荐）
git clone https://github.com/HKUST-QUANT-SOCIETY/data_access.git
cd data_access
pip install -e ".[all]"

# 私有仓一次性安装（把 <TOKEN> 换成有 repo 权限的 GitHub PAT）
# pip install "data-access[all] @ git+https://<TOKEN>@github.com/HKUST-QUANT-SOCIETY/data_access.git"
# pip install "data-access[client] @ git+https://<TOKEN>@github.com/HKUST-QUANT-SOCIETY/data_access.git"

# 读数 HTTP 服务
pip install -e ".[service]"
export DATA_ACCESS_API_KEY=quantsociety
data-access-server --host 0.0.0.0 --port 8765
# 完整接口与路径见 docs/用户使用手册.md §10
```

| extra | 包含 |
|-------|------|
| 默认 | 本地 `get_store()` |
| `[client]` | HTTP 客户端 |
| `[service]` | FastAPI 服务 |
| `[polars]` | Polars lazy |
| `[all]` | 全部 |

### COS 远程直读

```bash
export DATA_ACCESS_COS_READ_MODE=remote   # 或 auto
export DATA_ACCESS_COS_S3_ENDPOINT=cos.ap-guangzhou.myqcloud.com
export COS_SECRET_ID=...
export COS_SECRET_KEY=...
```

| 模式 | 行为 |
|------|------|
| `mirror` | 先拉到本地根再读（默认） |
| `remote` | DuckDB/httpfs 或 cli 缓存直连 COS |
| `auto` | 本地有则本地，否则 remote |

语义细节：[docs/COS语义与PIT契约.md](docs/COS语义与PIT契约.md)。

### 最小读示例

```python
from data_access import get_store
store = get_store()
df = store.read_frame(
    "ashare_stock_daily",
    columns=["TradeDate", "Symbol", "Close"],
    time_range=("2024-01-01", "2024-01-31"),
)
```

### HTTP 客户端

```python
from data_access.service.client import DataAccessClient
client = DataAccessClient("http://host:8765", api_key="quantsociety")
df = client.read_frame("ashare_stock_daily", columns=["Close"], time_range=("2024-01-01", "2024-01-31"))
```

### 部署

见 [deploy/README.md](deploy/README.md)：`Dockerfile`、`docker-compose.yml`、`kubernetes.yaml`、`data-access.service`。

### 质量检查

```bash
data-access-quality --help
```

---

## 常用 API（摘要）

| 方法 | 用途 |
|------|------|
| `read` / `read_uri` | 统一读入口，返回 `ReadHandle`（多形态转换） |
| `read_frame` / `read_arrow` | 表格式读取（兼容旧 API） |
| `load_columns` | 宽表列（给 factor_engine） |
| `read_auto` / `scan_polars` | 成本路由 / lazy |
| `read_factors` | 一次读多因子（UNION ALL / PIVOT） |
| `get_factor_catalog` / `refresh_factor_catalog` | 因子目录 |
| `build_dataset_manifest` | 构建数据集 `_manifest.parquet` |
| `sql` | 只读 SELECT |
| `compute_and_write` | 读→SQL→写 staging |
| `write_arrow` / `upsert` | 写草稿 |
| `publish_from_staging` | 晋升发布 |
| `read_cos_panel` 等 | COS 面板 / 事件 / asof（契约层） |

### 统一读示例

```python
from data_access import get_store
store = get_store()

# 统一读：按成本自动路由引擎/结果形态
handle = store.read("ashare_stock_daily",
                    columns=["TradeDate", "Symbol", "Close"],
                    time_range=("2024-01-01", "2024-01-31"),
                    filters={"Symbol": ["600000.SH", "000001.SZ"]})
tbl = handle.to_arrow()     # Arrow Table
df  = handle.to_pandas()    # pandas
plf = handle.to_lazy()      # Polars LazyFrame

# 临时文件直接读（dev 白名单下），不写 YAML
h = store.read_uri("/tmp/foo.csv", columns=["a", "b"])   # csv 自动识别
h = store.read_uri("/tmp/foo.feather")                    # arrow/feather 走 PyArrow

# 一次读多个因子（单查询）
handle = store.read_factors(["mom_3d", "vol_20"],
                            time_range=("2024-01-01", "2024-12-31"),
                            layout="wide")  # wide → datetime, asset, mom_3d, vol_20
```

完整场景与字段表：**[docs/用户使用手册.md](docs/用户使用手册.md)**。

## 与 factor_engine 配合

```yaml
data_source:
  type: data_access
  dataset: ashare_stock_daily
  fields: { close: Close, volume: Volume }
  start_date: "2019-01-01"
  end_date: "2026-06-30"
```

或 `from api.mining_integration import default_ashare_pv_data_source_config`。

## 文档与变更

- 用户手册：[docs/用户使用手册.md](docs/用户使用手册.md)
- 文档索引：[docs/README.md](docs/README.md)
- 变更：[CHANGELOG.md](CHANGELOG.md)
- 组织仓同步说明：[docs/SOURCE_SYNC.md](docs/SOURCE_SYNC.md)
