# Backend 三层覆盖清单

> 自动生成：`python scripts/report_backend_coverage.py --write-backend-doc docs/backend_coverage.md`

## 摘要

- runtime 已实现: **378**
- Polars 注册: **334**
- parity verified: **106**
- production Polars safe: **97**
- Polars expr long-table: **141**
- Polars long ∩ SQL emitter: **99**
- Polars long only (无 SQL): **243**
- SQL only (无 Polars long): **0**
- PRODUCTION_CORE: **84**（Polars gap: 无）
- PolarsLong native: **110**
- Production fast path（三后端 parity）: **30**
- DuckDB triple parity: **14**
- PRODUCTION_CORE fast path gap: **50**
- production 允许但仅 pandas: **35**
- 有 Polars 未进 production safe: **235**

## 晋级路径

```text
runtime polars 注册 → parity CI → POLARS_PARITY_VERIFIED → POLARS_PRODUCTION_SAFE
```

## production 允许但 production auto 仍走 pandas

- `BollingerBands`
- `BollingerLower`
- `BollingerUpper`
- `DPO`
- `MOM`
- `OBV`
- `ROC`
- `StochasticD`
- `StochasticK`
- `WilliamsR`
- `c_mean`
- `c_std`
- `c_sum`
- `coalesce`
- `group_mean`
- `group_normalize`
- `group_rank`
- `group_std`
- `group_zscore`
- `maximum`
- `minimum`
- `normalize`
- `scale`
- `ts_beta`
- `ts_corr`
- `ts_cov`
- `ts_mean`
- `ts_sharpe`
- `ts_std`
- `ts_var`
- … 共 35 个
