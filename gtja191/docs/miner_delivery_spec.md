# 自动化因子挖掘 — 统一交付规范（GTJA 包内索引）

> **规范正文已迁至 monorepo factor_engine。**  
> 请以 **[`../../factor_engine/docs/miner_delivery_spec.md`](../../factor_engine/docs/miner_delivery_spec.md)** 为准（disk.v1 / AFV Gateway）。

## GTJA185 相对规范的约定

| 项 | 本包取值 |
|----|----------|
| 包目录 | `gtja191/`（投递子集 **185** 条，AutoFactor pack 名 `gtja185`） |
| `expression_type` | `dsl` |
| **`dsl_surface`** | **`compat`**（含 `ts_ema` / `flex_max` / `ts_time_slope` 等，勿用默认 `daily` 校验） |
| Campaign `data_source` | `{local, cos}` 路径提示（§2.8）；**执行** 用 `data_access` + `ashare_stock_daily` |
| 算子策略 | `lqtp_pv_daily` + 解析面 `compat` |
| 重建投递 | `scripts/convert_and_build_delivery.py` → `candidate_pool/manual_ashare_pv_202607041600/` |

相关：

- [FactorEngine 完全指南](../../factor_engine/docs/FactorEngine完全指南.md)
- [量化平台使用总览](../../docs/量化平台使用总览.md)
- 本包 [README.md](../README.md)
