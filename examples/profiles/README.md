# `examples/profiles` — 生产级因子配置模板

与 **`runtime/config.py`** 的 **profile** 机制配合：继承 `dev` / `staging` / `prod` 的物化、DQ、telemetry 默认值，再覆盖因子与数据源。

## 使用

```bash
cd factor_engine
PYTHONPATH=..:. python -m pipeline run \
  --config examples/profiles/us_sip_day_aggs.yaml
```

## Profile 索引

| 文件 | 数据源形态 | 说明 |
|------|-----------|------|
| `dev.yaml` | — | 本地开发默认 |
| `staging.yaml` | — | staging 物化 |
| `prod.yaml` | — | 生产默认（`staging_clickhouse` 双写） |
| `prod_clickhouse.yaml` | — | ClickHouse 写端 |
| `factor_cs_pipeline.yaml` | `data_access` | A 股截面流水线（winsorize→neutralize→zscore） |
| `us_polygon_daily.yaml` | `data_access` | Polygon 日线价量 |
| `us_polygon_floats.yaml` | `composite` | Polygon 日线 + 流通股 asof |
| `us_sip_day_aggs.yaml` | `data_access` | SIP 日 K 价量 |
| `us_sip_fundamental.yaml` | `composite` | SIP 日 K + 财务比率 asof |
| `us_sip_balance_sheet.yaml` | `composite` | SIP 日 K + 资产负债表 asof |
| `us_sip_cash_flow.yaml` | `composite` | SIP 日 K + 现金流量表 asof |
| `us_sip_quotes.yaml` | `data_access` | SIP 报价 tick |
| `us_sip_trades.yaml` | `data_access` | SIP 逐笔成交 tick |

Mining preset 与 **`datasets.yaml`** 契约见 **`scripts/validate_datasets_mining_alignment.py`**。
