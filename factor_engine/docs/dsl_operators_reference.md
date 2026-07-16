# DSL 算子白名单参考

> **权威枚举**：[`dsl_allowlist.json`](dsl_allowlist.json)，由 `build_dsl_allowlist()` 导出。  
> **完整 runtime 与分类**：[`../cleaned_operators/docs/operators_catalog.md`](../cleaned_operators/docs/operators_catalog.md)。  
> **分层说明**：[`operator_surfaces.md`](operator_surfaces.md)。

## 日常因子 DSL

正常 manifest 只能使用 `surface=daily` 的算子，即能够生成按
`(date, instrument)` 对齐的因子值且默认满足因果约束的算子。代表性能力：

- 算术与安全数学：`add`, `subtract`, `multiply`, `divide`, `safe_div`, `log`, `sqrt`, `power`；
- 时序：`delay`, `ts_delta`, `ts_mean`, `ts_std`, `ts_sum`, `ts_min`, `ts_max`, `ts_corr`, `ts_rank`；
- 截面：`rank`, `zscore`, `winsorize`, `normalize`, `cs_demean`, `cs_regression`；
- 分组：`group_rank`, `group_neutralize`, `group_zscore`, `group_mean`；
- 技术与领域算子：保留能直接生成标量因子面板的技术、基本面和微观结构算子。

## 不在公共 DSL 中

- 已从运行时移除：`Lead`, `next`, `bfill`, `fillna_interpolate`, `interpolate`, `shuffle`, `sample`, `rand_*`, `norm*`；
- 研究侧因果外推：`causal_linear_extrapolate`（仅用历史两点，不在 daily DSL）；
- 研究诊断：假设检验、PDF/CDF、矩阵分解、PCA、复数、FFT、wavelet 和 filtering；
- 冗余旧名：`inv`, `reciprocal`, `fmax`, `fmin`, `sqr`, `cube`, `cumulative_*`。

研究工具通过 `research_operators` 显式访问，不得写入正常因子 manifest。

```bash
cd factor_engine
PYTHONPATH=. python scripts/export_dsl_allowlist.py
PYTHONPATH=. python scripts/generate_operators_catalog.py
```
