# quant_projects

A 股 / 美股量化因子研究 monorepo：**data_access 统一读数据** + **factor_engine 计算因子** + **COS Parquet 镜像**。

> **总使用文档（从这里开始）→ [docs/量化平台使用总览.md](docs/量化平台使用总览.md)**  
> **Agent loop 控制面 → [loop/README.md](loop/README.md)**

## 路径约定

| 磁盘目录 | Python import | 说明 |
|----------|---------------|------|
| **`dataaccess/`** | `from data_access import get_store` | 包目录无下划线；import 有下划线 |
| `factor_engine/` | `from api import ...`（需 `PYTHONPATH`） | DSL / 落盘引擎 |

## 快速开始

```bash
cd ~/quant_projects
bash scripts/setup_quant_projects.sh   # 若有
source env.sh
pip install -e "./dataaccess[all]"
```

## 目录概览

| 目录 | 作用 |
|------|------|
| `dataaccess/` | 统一数据读写（DuckDB + datasets.yaml + COS） |
| `factor_engine/` | 因子 DSL 引擎与落盘 |
| `ashare_lqtp_kit/` | A 股 LQTP gRPC 客户端与报告包 |
| `factor-pool-standard/` | manifest 字段契约与校验 |
| `factor_layer/` | 因子评估、准入、Agent（可选） |
| `gtja191/` | GTJA191 因子公式包 |
| `week2_pv_factors/` | Week2 价量因子包 |
| `scripts/` | 同步、落盘、检查脚本 |
| `data/` | 本地数据与因子湖（大数据默认不同步） |
| `docs/` | **总使用文档**与团队规范 |

## 环境变量

```bash
source env.sh
```

| 变量 | 默认 |
|------|------|
| `QUANT_PROJECTS_ROOT` | 本仓库根 |
| `QUANTSOCIETY_WORKSPACE_DATA_ROOT` | `data/` |
| `FACTOR_LAKE_ROOT` | `data/factors/lake` |
| `ASHARE_PARQUET_ROOT` | `data/a_share/lqtp_data` |
| `US_MASSIVE_ROOT` | `data/us_stock/massive_data` |
| `DATA_ACCESS_SKIP_COS_MIRROR` | 设 `1` 关闭读前 COS 拉取 |
| `DATA_ACCESS_COS_READ_MODE` | `mirror` / `remote` / `auto` |

### 读 COS → 运算 → 写到别处

```bash
export DATA_ACCESS_COS_READ_MODE=remote
export DATA_ACCESS_COS_S3_ENDPOINT=cos.ap-guangzhou.myqcloud.com
export COS_SECRET_ID=...
export COS_SECRET_KEY=...
```

```python
from data_access import get_store
store = get_store()
store.compute_and_write(
    "SELECT TradeDate AS datetime, Symbol AS asset, Close AS value, "
    "CAST(strftime(TradeDate, '%Y') AS INTEGER) AS year FROM {{ashare_stock_daily}}",
    read_datasets=["ashare_stock_daily"],
    read_time_ranges={"ashare_stock_daily": ("2024-01-01", "2024-01-31")},
    write_dataset="factor_lake_staging",
    factor_id="close_raw_v1",
    partition_by=["year"],
)
```

复杂因子用 `factor_engine` + `data_source.type: data_access`。详见总使用文档第 3 章。

## 同步 COS 数据

```bash
bash scripts/sync_ashare_lqtp_cos.sh StockDailyBar
bash scripts/sync_us_stock_cos.sh StockDailyBar
```

需已配置 `clean-cos-ro`（或等价 COS 凭证）。

## 读取数据（Python）

```python
from data_access import get_store
store = get_store()
cols = store.load_columns(
    "ashare_stock_daily",
    columns=["Close"],
    time_range=("2016-01-01", "2016-12-31"),
)
```

## 远程读数 HTTP 服务

```bash
# 服务端
pip install -e "./dataaccess[service]"
export DATA_ACCESS_API_KEY=团队密钥
data-access-server --host 0.0.0.0 --port 8765

# 客户端（任意机器）
pip install "data-access[client] @ git+https://github.com/18047533889/quant_projects.git#subdirectory=dataaccess"
```

```python
from data_access.service.client import DataAccessClient
client = DataAccessClient("http://读数服务器:8765", api_key="团队密钥")
df = client.read_frame("ashare_stock_daily", columns=["Close"], time_range=("2024-01-01", "2024-01-31"))
```

| extra | 用途 |
|-------|------|
| （默认） | 本地 `get_store()` |
| `[client]` | HTTP 远程客户端 |
| `[service]` | 读数 API 服务 |
| `[polars]` | Polars lazy |
| `[all]` | 全部 |

Docker / K8s：见 [`dataaccess/deploy/README.md`](dataaccess/deploy/README.md)。

## 因子落盘（示例）

```bash
source env.sh
python3 scripts/materialize_gtja191_factors.py --skip-existing
python3 scripts/materialize_week2_factors.py
```

结果：`data/factors/lake/factors/{factor_id}/year=YYYY/data.parquet`

## 投递校验

```bash
python3 factor-pool-standard/scripts/check_manifest_fields.py path/to/manifest.json
python3 scripts/check_data_access_allowlist.py
bash scripts/verify_repo_tracking.sh
```

## Git 同步

公共代码 `dataaccess/`、`factor_engine/` 须在仓库顶层被跟踪。详见 [docs/REPO_SYNC.md](docs/REPO_SYNC.md)。

```bash
bash scripts/verify_repo_tracking.sh
```

## 测试

```bash
source env.sh
pytest dataaccess/tests/ -q
cd factor_engine && pytest tests/ -q
```

## 更多文档

| 文档 | 说明 |
|------|------|
| **[docs/量化平台使用总览.md](docs/量化平台使用总览.md)** | **总使用文档（推荐）** |
| [dataaccess/docs/用户使用手册.md](dataaccess/docs/用户使用手册.md) | data_access 完整手册 |
| [dataaccess/README.md](dataaccess/README.md) | data_access 模块说明 |
| [factor_engine/docs/FactorEngine完全指南.md](factor_engine/docs/FactorEngine完全指南.md) | 因子引擎总指南 |
| [factor_engine/docs/README.md](factor_engine/docs/README.md) | 因子引擎文档索引 |
| [STRUCTURE.md](STRUCTURE.md) | 目录结构 |
| [ashare_lqtp_kit/README.md](ashare_lqtp_kit/README.md) | LQTP 客户端与报告 |
