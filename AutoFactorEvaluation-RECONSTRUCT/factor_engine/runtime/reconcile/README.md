# `runtime/reconcile` — 双写对账

本地 parquet 与 `factor_lake_staging` 双写场景下的补偿与对账。

| 模块 | 作用 |
|------|------|
| `runtime/dual_write_reconcile.py` | 比较本地 vs staging，生成 repair 计划 |
| 本目录 | 与根模块镜像，供 reconcile CLI 引用 |

典型命令见 [`../../run_pipeline.py`](../../run_pipeline.py) 的 `--reconcile` 子命令。
