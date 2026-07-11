# Quant platform integration

`AutoFactorEvaluation-RECONSTRUCT` 已移除目录内复制的旧 FactorEngine，统一复用
`quant_projects` 根目录的生产组件。

## 组件边界

- DSL parser、IR、planner、算子、Pandas/Polars/DuckDB 后端：根目录 `factor_engine/`。
- 行情读取、COS mirror、schema、查询预算、DataSnapshot：根目录 `data_access/`。
- AutoFactorEvaluation 唯一平台适配入口：`integrations/quant_platform.py`。
- A 股默认数据集：`ashare_stock_daily`；美股默认数据集：`us_stock_daily`。
- 因子值先幂等 upsert 到 `factor_lake_staging`，只有显式 `publish=True` 才晋升
  `factor_lake`。
- `expression_type=python/code` 在该接入链路中 fail-closed，候选必须先转换为
  FactorEngine DSL。

## Gateway

Gateway Step2 使用 `build_factor_engine_config()` 生成 `type: data_access` 的配置，
Step4 使用当前 FactorEngine 真实编译候选。设置 `AUTOFACTOR_TINY_START` 和
`AUTOFACTOR_TINY_END` 后，Step4 会通过 DataAccess 在指定小区间真实执行；未设置
时执行 compile-only gate，不会伪造计算成功。

## Assetization

```python
from assetization.scripts.compute import run_assetization

result = run_assetization(
    {
        "candidate_id": "mom_20d",
        "formula": "close / ts_delay(close, 20) - 1",
        "market": "ashare",
        "frequency_bucket": "1d",
        "start_date": "2020-01-01",
        "end_date": "2026-06-30",
    },
    backend="pandas",
    run_mode="research",
    materialize_staging=True,
    publish=False,
)
```

返回值包含 `data_snapshot_id`、MultiIndex 因子序列、按日结果及 DataAccess staging
写入摘要。生产发布必须显式执行 `publish=True`，并受 DataAccess 的 staging →
published 原子晋升约束。

## 独立公式执行

```python
from integrations.quant_platform import execute_factor_formula

execution = execute_factor_formula(
    "ts_zscore(close, 20)",
    factor_name="price_zscore_20d",
    market="ashare",
    start_date="2024-01-01",
    end_date="2025-12-31",
    run_mode="research",
)
```

## 运行环境

应从仓库根目录启动命令。CI 已配置：

```bash
export PYTHONPATH="$PWD:$PWD/factor_engine:$PWD/AutoFactorEvaluation-RECONSTRUCT"
```

非标准部署路径可设置：

```bash
export QUANT_PROJECTS_ROOT=/path/to/quant_projects
```

业务模块不得重新添加旧引擎路径，也不得直接使用 `pd.read_parquet` 扫描正式行情；
正式数据身份由 DataAccess dataset、参数和 `DataSnapshot.snapshot_id` 决定。
