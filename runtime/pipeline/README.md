# `runtime/pipeline` — 批量流水线

YAML / 目录批量跑因子的薄封装，CLI 入口 [`../../run_pipeline.py`](../../run_pipeline.py)。

| 文件 | 作用 |
|------|------|
| `batch.py` | 扫描配置目录、并行/串行调度 `FactorEngine.run_from_config` |

根目录 [`pipeline.py`](../../pipeline.py) 仅为兼容 shim → `runtime.pipeline.batch`。
