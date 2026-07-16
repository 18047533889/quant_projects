# Backend 三层覆盖清单

> 自动生成：`python scripts/report_backend_coverage.py --write-backend-doc docs/backend_coverage.md`

## 摘要

- runtime 已实现: **386**
- Polars 注册: **332**
- parity verified: **36**
- production Polars safe: **53**
- PRODUCTION_CORE: **50**（Polars gap: 无）
- production 允许但仅 pandas: **5**
- 有 Polars 未进 production safe: **279**

## 晋级路径

```text
runtime polars 注册 → parity CI → POLARS_PARITY_VERIFIED → POLARS_PRODUCTION_SAFE
```

## production 允许但 production auto 仍走 pandas

- `Corr`
- `Median`
- `Var`
- `cum_last`
- `if_else`
