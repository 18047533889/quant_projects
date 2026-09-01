# Factor Engine — DSL 因子计算引擎

**版本:** 0.3.x ｜ **仓库:** https://github.com/HKUST-QUANT-SOCIETY/factor_engine (私有)
**中文完整指南:** [`docs/FactorEngine完全指南.md`](docs/FactorEngine完全指南.md) ｜ **HTTP 服务:** [`service/README.md`](service/README.md) ｜ **算子参考:** [`docs/dsl_operators_reference.md`](docs/dsl_operators_reference.md)

## 它是什么

把因子公式写成 DSL（如 `rank(ts_mean(close, 20))`），引擎从 **data_access** 读行情，
在 `(日期 × 标的)` 网格上算出因子值（MultiIndex Series），并可**落盘到因子湖**。

它**不做**：数据清洗、因子评估/回测、模型训练 —— 只负责**算因子 + 写因子湖**。

## 安装与第一个因子

```bash
git clone https://github.com/HKUST-QUANT-SOCIETY/factor_engine.git
cd factor_engine
pip install -e .
# 推荐同时装读数层：pip install "data-access @ git+https://<TOKEN>@github.com/HKUST-QUANT-SOCIETY/data_access.git"
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

或 YAML 一键跑：`FactorEngine.run_from_config("examples/config_driven_factor.yaml")`。
或 HTTP 服务：`pip install -e ".[service]"` 后 `factor-engine-serve --port 8088`。

## 架构

```
Factor(expr) → expr AST → ir.Analyzer → planner（Lowerer + Optimizer + CSE）
    → backend.execute（auto：SQL 下推 → Polars → Pandas）→ 因子值
```

| 层 | 目录 | 作用 |
|---|---|---|
| DSL API | `api/` | `col()`, `rank()`, `ts_mean()`, `Factor`, `parse_expr` |
| AST | `expr/` | ColumnRef / Literal / CleanedCall |
| 分析 | `ir/` | Expr → IR，推导 lookback |
| 规划 | `planner/` | IR → Plan、CSE、成本模型 |
| 执行 | `backend/` | pandas / polars / duckdb_sql / clickhouse_sql / auto |
| 算子 | `cleaned_operators/` | 单一实现源（canonical 1737，daily surface 1238） |
| 存储 | `storage/` | data_access 数据源、parquet/CH、物化 → 因子湖 |
| 运行时 | `runtime/` | `FactorEngine`、批量调度、资源代理、物化服务 |
| 服务 | `service/` | 薄 FastAPI 适配层（端口 8088） |

## 算子

- 权威清单：`cleaned_operators/docs/operators_catalog.json`（1737；daily 1238 / extended 480 / research 8）；
  机器可读白名单：`docs/dsl_allowlist.json`（1421）。
- 常用：`rank`、`zscore`、`ts_mean`、`ts_std`、`delay`、`group_rank`、`group_neutralize`、
  `protected_div`、`SMA/EMA`、`ts_rsi/ts_macd/ts_atr/...`、`trade_when`、`neutralize`。

## 读数据 / 写因子

- 读：`data_source.type: data_access`，数据集名见 data_access 的 `config/datasets.yaml`
  （如 `ashare_stock_daily`、`ashare_stock_minute_adj`、`factor_lake`）。
- 写：`engine.materialize(factor)` → 本地 lake / `factor_lake_staging` →
  `lake_publish.publish_factor_lake()`。生产 profile：`examples/profiles/prod.yaml`。

## 依赖与接口（谁 import 谁）

- **依赖**：`data_access`（读数，如 `from data_access.clickhouse.panel import ...`）。
- **被谁调用**：
  - `quant_evaluator` → `adapters/factor_engine.py`（把 FE 结果转 FactorBatch）
  - `factor_optimizer` → `adapters/factor_engine.py`（canonical_hash / validate_mutation / estimate_complexity）
  - `factor_assets` → `adapters/factor_engine.py`（身份 hash）
  - `alphaprobe` → `fe_bridge/`（表达式转 DSL 并用 FE 评估）

## 收益/时序口径（红线）

本库**只算值**。评估/训练标签一律 **vwap-to-vwap**（`VWAP_{t+H}/VWAP_t - 1`，后复权）；
决策时钟 = 收盘信号 → 下一交易日 VWAP 成交。详见 `docs/adr_backtest_target_position.md` §13（防前视）。

## 测试

```bash
cd factor_engine && PYTHONPATH=. pytest tests/ -q    # 1144+ tests
RUN_REAL_PARQUET_SMOKE=1 pytest tests/test_real_data_factor_smoke.py -v  # 需真实数据
```
