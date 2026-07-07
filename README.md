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

## 因子落盘

```bash
source env.sh
python3 scripts/materialize_gtja191_factors.py --skip-existing
python3 scripts/materialize_week2_factors.py
```

结果：`data/factors/lake/factors/{factor_id}/year=YYYY/data.parquet`

## 投递校验

```bash
# DSL 语法
cd factor_engine && python3 scripts/validate_delivery_formula.py --manifest path/to/manifest.json

# 字段 / domain
python3 factor-pool-standard/scripts/check_manifest_fields.py path/to/manifest.json

# 代码规范（禁止直读 parquet）
python3 scripts/check_data_access_allowlist.py
```

## 测试

```bash
source env.sh
pytest data_access/tests/ -q
cd factor_engine && pytest tests/ -q
```

## 从上游同步（需 GITHUB_TOKEN）

```bash
export GITHUB_TOKEN=ghp_xxx
bash scripts/sync_quantsociety_backend.sh
```

不会覆盖本地定制的 `data_access/`、`factor_engine/`。
