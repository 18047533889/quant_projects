# `data_access` — 团队统一数据读写入口

> **一句话定位**：所有读写 parquet 的代码，都应该走这里。  
> **磁盘目录**：本仓库为 **`dataaccess/`**；代码里永远 `import data_access`。  
> **对外使用说明**：**[docs/用户使用手册.md](docs/用户使用手册.md)**（只看这一份即可）。  
> **Monorepo 总览**：[docs/量化平台使用总览.md](../docs/量化平台使用总览.md)  
> 文档索引：[docs/README.md](docs/README.md)  
> 维护：量化基础平台组｜更新：2026-07-19｜包版本：**0.3.0**

## 为什么有这个模块

以前各处都是 `pd.read_parquet` → 拼 glob → 合并 DataFrame，路径硬编码、性能不齐、写盘无规范。

**当前能力（0.3.x）**：

1. **读入口**：`read_arrow` / `read_frame` / `load_columns` / `read_arrow_stream` / `read_auto` / `scan_polars`
2. **DuckDB 共享引擎**：跨线程 buffer pool / parquet footer cache
3. **数据集登记**：`config/datasets.yaml`
4. **namespace**：namespaced / staging / published
5. **写与发布**：`write_arrow` / `upsert` / `publish_from_staging` / `delete_rows` / `compute_and_write`
6. **有限 SQL**：`sql()` + QueryBudget
7. **COS**：mirror / remote / auto；可执行面板与 **PIT 契约**
8. **HTTP**：`data-access-server` + `DataAccessClient`；`/v1/datasets`、`/v1/read`
9. **部署**：Docker / compose / K8s / systemd（`deploy/`）
10. **质量**：`data-access-quality` CLI（`quality/`）
11. **ClickHouse**：panel 读 / 因子写与校验

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
# 本地 editable
cd dataaccess && pip install -e ".[all]"
# 或在 monorepo 根：
pip install -e "./dataaccess[all]"

# 只装远程客户端
pip install "data-access[client] @ git+https://github.com/18047533889/quant_projects.git#subdirectory=dataaccess"

# 读数服务
pip install -e "./dataaccess[service]"
export DATA_ACCESS_API_KEY=团队密钥
data-access-server --host 0.0.0.0 --port 8765
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
client = DataAccessClient("http://host:8765", api_key="...")
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
| `read_frame` / `read_arrow` | 表格式读取 |
| `load_columns` | 宽表列（给 factor_engine） |
| `read_auto` / `scan_polars` | 自动路由 / lazy |
| `sql` | 只读 SELECT |
| `compute_and_write` | 读→SQL→写 staging |
| `write_arrow` / `upsert` | 写草稿 |
| `publish_from_staging` | 晋升发布 |
| `read_cos_panel` 等 | COS 面板 / 事件 / asof（契约层） |

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
