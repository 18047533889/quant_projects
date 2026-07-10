# `benchmarks` — 后端性能基准

手工/CI 可选运行的算子与后端耗时对比，**不是** pytest 默认套件。

| 文件 | 作用 |
|------|------|
| `backend_operator_bench.py` | 按 manifest 抽样算子，对比 Pandas / Polars / SQL 路径 |
| `backend_cost_baseline.json` | 基线耗时快照（`backend/operator_cost.py` 消费） |
| `operator_manifest.json` | 算子清单副本（与 `scripts/export_operator_manifest.py` 同步） |

运行示例：

```bash
cd factor_engine
python benchmarks/backend_operator_bench.py --help
```
