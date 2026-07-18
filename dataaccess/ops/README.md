# dataaccess/ops — 运维脚本

| 脚本 | 用途 |
|------|------|
| [`refresh_dataset_stats.py`](refresh_dataset_stats.py) | 刷新 `datasets.yaml` 静态数据集的 `.data_access_stats.json` sidecar |

```bash
cd quant_projects
PYTHONPATH=. python3 dataaccess/ops/refresh_dataset_stats.py --dataset ashare_stock_daily
```

一次性 bucket 分区迁移脚本已移除（迁移已完成）；历史说明见 `docs/team_docs/01_data_access_duckdb_全量变更与迁移指南.md`。
