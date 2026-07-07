# factor_pool

这个目录当前是预留位，用于未来沉淀“已评估、已准入、可复用”的因子集合。

## 当前状态

- 目录尚未形成稳定实现
- 当前仓库里的实际因子生产和筛选流程仍以 `factor_engine`、`factor_evaluation`、`factor_admission` 为主

## 建议理解方式

可以把 `factor_pool` 看成后续要承接下列结果的位置：

1. `factor_engine` 产出的因子值或 factor lake
2. `factor_evaluation` 产出的评估摘要
3. `factor_admission` 产出的准入结论

在这个目录形成稳定实现前，建议直接使用：

- [../factor_engine/README.md](../factor_engine/README.md)
- [../factor_evaluation/README.md](../factor_evaluation/README.md)
- [../factor_admission/README.md](../factor_admission/README.md)