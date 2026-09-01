# FactorEngine 冷启动因子库 V9 后端审计报告

## 结论

- V8 输入：**6292** 条。
- 最新 DSL / IR / Plan 静态审计：**6292/6292** 通过。
- 真实 FactorEngine Pandas 后端执行：**6292/6292** 通过，执行失败 0、全空 0、无穷值 0。
- 修正复权因子与盘口压力数据后，仅 **2** 条极端阈值事件公式持续退化为常数，已移出默认池并放入事件归档。
- V9 可用库：**6266** 条；默认核心 **6137** 条；可选数据池 **129** 条。
- 多输入统计公式中 **709** 条改为显式 `window=` 参数，解决复杂子表达式的 SQL 参数误绑定。

## 后端能力

- 具备完整 Polars 算子级路径：**5360** 条。
- 整条 DAG 成功生成 SQL：**5934** 条。
- 同时满足 Pandas路径、Polars算子路径和整式SQL编译：**5263** 条。
- 按当前合同快照同时满足 production status 与 PIT 标记：**5662** 条。
- 三后端静态路径且合同严格：**4936** 条。

这里的“静态路径”表示算子注册和SQL编译均成立，**不等于已经完成真实 Polars/DuckDB 数值一致性运行**。

## 当前环境限制

当前 Python 为 3.13.5，环境中没有 `polars`、`duckdb`、`pyarrow`。内部软件源不提供对应包，公网 DNS/下载被阻断；临时 GitHub Actions 任务也在 runner 启动前失败，没有生成 wheel artifact。因此本轮没有伪造原生 Polars 或 DuckDB 的运行结果。

包内提供 `scripts/verify_native_backends_v9.py`。依赖可用后，它会拒绝导入stub，并对三后端静态子集执行 Pandas、原生Polars和真实DuckDB SQL数值对比。

## 分层使用

1. 默认使用 `production_default_core_v9.json`。
2. 字段合同满足后并入 `production_optional_v9.json`。
3. 需要跨后端部署时，优先使用 `three_backend_static_path_v9.json`；真实部署前仍需运行原生复验脚本。
4. `event_archive_v9.json` 不进入默认模型，仅用于事件型研究。
5. `contract_pending_v9.json` 的公式数值可执行，但使用了合同快照中仍标记为 experimental 或 PIT 未最终确认的算子。
