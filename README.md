# quant_projects

A 股 / 美股量化因子研究 monorepo：**data_access 统一读数据** + **factor_engine 计算因子** + **COS Parquet 镜像**。

详细目录说明见 [STRUCTURE.md](STRUCTURE.md)。

## 快速开始（给他人分发后）

```bash
cd ~/quant_projects
bash scripts/setup_quant_projects.sh
source env.sh
```

## 目录概览

| 目录 | 作用 |
|------|------|
| `data_access/` | 统一数据读写（DuckDB + datasets.yaml + COS 镜像） |
| `factor_engine/` | 因子 DSL 引擎与落盘 |
| `factor-pool-standard/` | manifest 字段契约与校验 |
| `factor_layer/` | 因子评估、准入、Agent（可选） |
| `gtja191/` | GTJA191 因子公式包 |
| `week2_pv_factors/` | Week2 价量因子包 |
| `data/` | 本地数据与因子湖 |
| `scripts/` | 同步、落盘、检查脚本 |

## 环境变量

```bash
source env.sh   # 设置 PYTHONPATH、数据路径、DuckDB 参数
```

| 变量 | 默认 |
|------|------|
| `QUANT_PROJECTS_ROOT` | 本仓库根 |
| `QUANTSOCIETY_WORKSPACE_DATA_ROOT` | `data/` |
| `FACTOR_LAKE_ROOT` | `data/factors/lake` |
| `ASHARE_PARQUET_ROOT` | `data/a_share/lqtp_data` |
| `US_MASSIVE_ROOT` | `data/us_stock/massive_data` |
| `DATA_ACCESS_SKIP_COS_MIRROR` | 设 `1` 关闭读前 COS 拉取 |
| `DATA_ACCESS_COS_READ_MODE` | `mirror`（默认）/ `remote`（直连 COS）/ `auto` |

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

COS / published 源只读；结果落到 staging。复杂因子用 `factor_engine` + `data_source.type: data_access`。

## 同步 COS 数据

```bash
# A 股日线（全量 20 表或指定表）
bash scripts/sync_ashare_lqtp_cos.sh StockDailyBar

# 美股
bash scripts/sync_us_stock_cos.sh StockDailyBar
```

需已配置 `clean-cos-ro` 命令。

## 读取数据（Python）

```python
source env.sh  # 或手动 export PYTHONPATH

from data_access import get_store
store = get_store()
cols = store.load_columns(
    "ashare_stock_daily",
    columns=["Close"],
    time_range=("2016-01-01", "2016-12-31"),
)
```

## 远程读数 HTTP 服务（团队无需 clone 全仓库）

### pip 安装（推荐）

```bash
# 从 GitHub 安装轻量客户端（连远程读数服务）
pip install "data-access[client] @ git+https://github.com/18047533889/quant_projects.git#subdirectory=data_access"

# 本地开发：editable 安装完整 data_access
cd quant_projects/data_access
pip install -e ".[all]"
```

```python
from data_access.service.client import DataAccessClient

client = DataAccessClient("http://读数服务器:8765", api_key="团队密钥")
df = client.read_frame("ashare_stock_daily", columns=["Close"], time_range=("2024-01-01", "2024-01-31"))
```

| extra | 用途 |
|-------|------|
| （默认） | 本地 `get_store()` 读数 |
| `[client]` | HTTP 远程客户端 |
| `[service]` | 启动读数 API 服务 |
| `[polars]` | `store.scan()` / Polars lazy |
| `[all]` | 全部可选依赖 |

**服务端**（部署在有 COS/数据盘的一台机器，如学校服务器或风控专用读数机）：

```bash
source env.sh
pip install -e "./data_access[service]"   # 或 pip install -r requirements-service.txt
export DATA_ACCESS_API_KEY=你的团队密钥   # 可选但建议设置
data-access-server --host 0.0.0.0 --port 8765
# 等价: python scripts/run_data_access_server.py --host 0.0.0.0 --port 8765
```

**客户端**（任意机器，pip 安装即可，不必 clone 全仓库）：

```bash
pip install "data-access[client] @ git+https://github.com/18047533889/quant_projects.git#subdirectory=data_access"
```

```python
from data_access.service.client import DataAccessClient

client = DataAccessClient("http://读数服务器:8765", api_key="你的团队密钥")
df = client.read_frame(
    "ashare_stock_daily",
    columns=["TradeDate", "Symbol", "Close"],
    time_range=("2024-01-01", "2024-12-31"),
)
print(df.head())
```

返回格式支持 `parquet`（默认）、`arrow_ipc`、`json`（小结果调试）。响应头带 `X-Data-Snapshot-Id` 供 lineage 追溯。

## 因子落盘

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

## Git 同步到 GitHub（必读）

公共代码 `data_access/`、`factor_engine/` **必须在仓库顶层被 Git 跟踪**。  
勿仅用 `git add -u`（不会添加新文件）。详见 [docs/REPO_SYNC.md](docs/REPO_SYNC.md)。

```bash
bash scripts/verify_repo_tracking.sh
bash scripts/git_sync_public_code.sh --dry-run
bash scripts/git_sync_public_code.sh --commit "你的说明" --push
```

## 从上游 sparse clone 同步（可选，需 GITHUB_TOKEN）

```bash
export GITHUB_TOKEN=ghp_xxx
bash scripts/sync_quantsociety_backend.sh
```

不会覆盖本地定制的 `data_access/`、`factor_engine/`。

## 测试

```bash
source env.sh
pytest data_access/tests/ -q
cd factor_engine && pytest tests/ -q
```

## 更多文档

- **[data_access 用户使用手册](data_access/docs/用户使用手册.md)** — 给外部使用者（随 `data_access` 包分发）：功能、数据集、COS、自选路径、运算、写发布、HTTP
- [data_access/README.md](data_access/README.md) — 模块说明与 API 摘要
- [STRUCTURE.md](STRUCTURE.md) — 目录结构
- [docs/REPO_SYNC.md](docs/REPO_SYNC.md) — Git 同步与 GitHub 可见性
