---
name: quant-factor-landing
description: 落因子值的正确方式 — 用 data_access 读后复权数据 + factor_engine 算值并落盘（DSL / FactorEngine.run / factor_lake_staging / compute_and_write），禁止手写 pandas 落值。当任务涉及"算因子、落因子值、写因子湖、backfill"时先读此 skill。
version: 1.0.0
---

# 落因子值（factor landing）标准流程

**前提**：已读 quant-platform-workflow skill（主趁库约束 + 后复权口径）。

## 路线选择

| 场景 | 路线 |
|---|---|
| 单因子/少量因子，DSL 能表达 | **factor_engine DSL**（首选） |
| 上百因子批量 + 复杂逻辑（需要 Python 代码块） | jobs/ 现有 backfill 框架（`jobs/backfill_28_factors.py` 的 `run_dsl_pages` / HASCODE `run_hascode_parallel`） |
| SQL 能表达的简单衍生 | `store.compute_and_write()` |
| 已有因子值合并/筛选 | `lightgbm_qs/scripts/merge_all_factors.py` 模式 |

## 路线 1：factor_engine DSL（标准）

```python
import sys; sys.path.insert(0, '/home/sunhaiwei/quant_projects')
# 必须 .venv/bin/python：系统 python3 无 polars，load_all 会 fail

import factor_engine.cleaned_operators as co
co.load_all(include_research=False)          # 先装载算子注册表（P0-B1 修复后独立可用）

from factor_engine.api import Factor, col, rank, ts_mean   # 其余算子走 __getattr__ 白名单懒加载
from factor_engine.backend.factory import build_backend
from factor_engine.storage.factory import build_data_source
from factor_engine.runtime.engine import FactorEngine

# 数据源：data_access（后复权表）
source = build_data_source({
    "type": "data_access",
    "dataset": "ashare_stock_daily_adj",     # 后复权表（硬性）
    "fields": {
        "close": "AdjClose", "open": "AdjOpen", "high": "AdjHigh",
        "low": "AdjLow", "vwap": "AdjVwap", "volume": "Volume",
    },
    "start_date": "2024-01-01",
    "end_date": "2024-12-31",
})
engine = FactorEngine(backend=build_backend("auto"), data_source=source)

factor = Factor(name="mom20_rank", expr=rank(ts_mean(col("close"), 20)),
                freq="1d", universe="ALL",
                description="20日动量截面秩", source_expr="rank(ts_mean(close, 20))")
out = engine.run(factor)          # 返回 dict: {factor, analysis, plan, result}
values = out["result"]            # MultiIndex Series / DataFrame（因子值）
```

要点：
- **算子必须在白名单**（1478 个）：`build_dsl_allowlist()` 查；不在 → 去库内加实现，不许手写。
- `Factor.name` 走 domain validator（非法 id 直接抛错）。
- production 模式有 PIT/算子白名单硬门；research 模式默认 fail-open。
- DSL 语法验证：`from factor_engine.api import parse_expr` / `validate_factor_engine_dsl("rank(ts_mean(close,5))")`。

## 路线 2：批量落值（jobs/ 现成框架）

- 日线 HASCODE 并行：`jobs/backfill_adj_parallel.py` → 复用 `extend_all_456._compute_factor_chunk`（ProcessPool）
- DSL 分页：`jobs/backfill_28_factors.py` 的 `run_dsl_pages`
- 落盘目录：`weekly_backtest_output/factor_matrices_all/<factor_id>.parquet`（date×asset wide）
- 临时市场缓存：`/tmp/mkt_*`（后复权 Adj 已重建，`backfill_28_factors.DAILY` 已切 StockDailyBarAdj）

**新因子加进来**：写 factor 函数（`def factor_xxx(df) -> Series`）→ 走 HASCODE 分发；DSL 公式 → 走 `run_dsl_pages`。参考 `jobs/backfill_13_dsl_factors.py`、`jobs/backfill_61_fund.py`。

## 路线 3：SQL 衍生（compute_and_write）

```python
from data_access import get_store
store = get_store()
store.compute_and_write(
    "SELECT CAST(TradeDate AS TIMESTAMP) AS datetime, Symbol AS asset, AdjClose/AdjPreClose-1 AS value "
    "FROM {{ashare_stock_daily_adj}} WHERE TradeDate >= DATE '2024-01-01' AND TradeDate <= DATE '2024-06-30'",
    read_datasets=["ashare_stock_daily_adj"],
    read_time_ranges={"ashare_stock_daily_adj": ("2024-01-01", "2024-06-30")},   # COS remote 必带
    write_dataset="factor_lake_staging", factor_id="ret1d_v1",
    mode="overwrite", partition_by=["year"],
)
store.publish_from_staging("factor_lake_staging", "factor_lake", factor_id="ret1d_v1")  # 全员可见（需权限）
```

## 环境变量（COS remote 读）

```bash
export DATA_ACCESS_COS_READ_MODE=remote
export DATA_ACCESS_COS_REMOTE_BACKEND=cli
export QUANT_RUN_NAMESPACE=<你的namespace>     # 写数据必设
```

**COS remote cli 后端读任何表必须带 `time_range`（或 `read_time_ranges`），否则整表拉取。**

## 落值后自检（必做）

1. 值非全 NaN / 覆盖率合理（tradable 资产上 ≥ 某阈值）
2. 与 label 口径对齐：评估收益用 `jobs/adj_vwap_common.adj_fwd_return()`（后复权 AdjVwap）
3. 无 look-ahead：因子值在 t 日只用 ≤t 信息（DSL 的 ts_* 算子天然安全；HASCODE 手写段检查 shift 方向）
4. 落盘 parquet index=TradeDate(date), columns=Symbol

## 禁止

- ❌ 手写 pandas 版动量/波动率等核心算子绕过 factor_engine
- ❌ 未复权 `ashare_stock_daily` 算收益类因子
- ❌ 直接写 `factor_lake`（只能写 `factor_lake_staging` 再 publish）