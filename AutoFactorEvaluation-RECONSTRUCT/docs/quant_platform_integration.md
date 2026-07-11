# Quant platform integration

`AutoFactorEvaluation-RECONSTRUCT` 不再携带私有 FactorEngine 副本。

- DSL parser、IR、planner、算子和执行后端：仓库根 `factor_engine/`。
- 行情、快照、schema、查询预算和因子 staging：仓库根 `data_access/`。
- 唯一适配入口：`integrations/quant_platform.py`。
- A 股默认数据集：`ashare_stock_daily`；美股默认：`us_stock_daily`。
- 因子产物先写 `factor_lake_staging`，只有显式 `publish=True` 才晋升。
- `expression_type=python/code` 在生产接入中 fail-closed，必须转换为 DSL。

运行时需令仓库根和 `factor_engine/` 可导入。根 CI 已设置对应 `PYTHONPATH`；
服务器脚本可从仓库根启动，或设置 `QUANT_PROJECTS_ROOT`。
