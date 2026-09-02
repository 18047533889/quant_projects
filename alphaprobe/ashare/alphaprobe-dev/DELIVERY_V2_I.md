# DELIVERY_V2_I — 集成/Leakage/LongShort/Calibration 测试矩阵 + Done Definition 核验

日期：2026-09-02
范围：任务书 §68（集成/Leakage/LongShort/Calibration 四行）+ §69（Done Definition 核验）
角色：测试与核验（未改任何 src/ 生产代码）
工作目录：`/home/sunhaiwei/quant_projects/alphaprobe/ashare/alphaprobe-dev`

## 一、交付物

- 新增测试文件：`tests/test_v2_integration.py`（20 个用例，全合成数据、不读 COS、stub 注入、不跑真实 LLM/模型）
- 回归结果：`PYTHONPATH=src python3 -m pytest tests/ -q` → **425 passed + 1 skipped**（基线 405 + 新增 20；faiss skip 为既有，非失败）

## 二、测试矩阵行覆盖（§68）

| 行 | 用例 | 结论 |
|----|------|------|
| 集成行 | `TestEndToEndIntegration`（3 用例）：stub LLM → SearchPipeline 一轮（structured_generation=False 默认路径）→ dedup 过滤 → L0-L2 → QuantEvaluatorAdapter 评估 → factor_fitness_v2 出分 → ActivePool Pareto 入池 | DONE |
| Leakage 行 | `TestLeakage`（3 用例）：泄漏因子 rankic≈1 红旗 + 合法因子 rankic 温和 + 无未来函数契约（label shift 方向正确） | DONE |
| LongShort 行 | `TestLongShort`（3 用例）：overlapping 直接年化 Sharpe 显著高于 cohort（inflation 存在）+ adapter 只输出 cohort 口径 + L 维 9 项指标全有值且在合理范围 | DONE |
| Calibration 行 | `TestCalibration`（4 用例）：warmup ordinal → min_warmup_n 后 z → freeze 参数不变 + Sharpe 单调映射 | DONE |

## 三、Done Definition 16 条逐条状态（§69）

| # | 条目 | 状态 | 证据 |
|---|------|------|------|
| 1 | 全量基线 405 passed + 1 skipped 保住 | DONE | 回归 425 passed + 1 skipped（含新增） |
| 2 | FactorFitness V2 六维 P/Q/L/S/N/R 拆分存在 | DONE | `src/alphaprobe/fitness/factor_fitness.py::factor_fitness_v2`（六维 + 三惩罚） |
| 3 | RetrieverScore 拆分存在 | DONE | `src/alphaprobe/retrieval/bayesian_retriever.py::compute_retriever_score` |
| 4 | ExportScore 拆分存在 | DONE | `src/alphaprobe/fitness/__init__.py::export_score`（§21.5） |
| 5 | L 维 20d cohort 非 overlapping | DONE | `evaluator_adapter._cohort_metrics` 走 `compute_metrics_from_cohort`；测试断言 cohort 路径被调用且无 overlapping 键 |
| 6 | 评估只走 QE（无手写 calc_sharpe/calc_long_short/calc_mdd） | DONE | grep 全 src 0 命中；`evaluator_adapter` 只读 QE registry + probe_portfolio |
| 7 | FE fail-closed（build_identity_view 抛错路径） | DONE | `authority.build_identity_view` 空公式/坏 provider 抛 `FactorIdentityAuthorityError` |
| 8 | LabelContract embargo ≥ label_days | DONE | `vwap_20d_label_contract`：horizon=20、embargo=20、overlapping=True、basis=vwap_to_vwap |
| 9 | structured_generation 默认关闭行为不变 | DONE | `PipelineConfig.structured_generation=False` 默认；orchestrator 同步 False |
| 10 | Pareto 淘汰非单一 IC | DONE | `pool/pareto.py` non-dominated rank；测试断言低 fitness 高 novelty 不被单维 IC 淘汰 |
| 11 | QuantEvaluatorAdapter 统一评估入口 | DONE | `evaluator_adapter.py`（P/S/R 走 QE registry，L 20d 走 cohort，fail-closed） |
| 12 | factor_assets_adapter + pool/pareto | DONE | `factor_assets_adapter.py` + `pool/pareto.py`（既有 19+1skip 测试） |
| 13 | generation/structured（action JSON→AST→FE validate） | DONE | `generation/structured.py`（既有 29 测试） |
| 14 | authority wiring（FE identity + LabelContract + DedupClient） | DONE | `authority.py` + `dedup_client.py`（既有 12+27 测试） |
| 15 | pipeline.py SearchPipeline 主链 + stub llm | DONE | `pipeline.py`（既有 16 测试 + 本文件集成用例） |
| 16 | dedup_client identity 三级降级链 | DONE | `dedup_client.py`（FE identity → alphaprobe.identity → 文本 canonical） |

**16/16 全部 DONE**，无 PARTIAL / MISSING。

## 四、发现的缺陷清单

无生产代码缺陷。测试过程中确认以下既有行为符合预期（非缺陷）：
- `SearchPipeline` 默认 evaluator 走 `make_fe_evaluate_fn`（fe_bridge 真算），本文件集成用例显式注入 `make_qe_evaluate_fn`（QE 口径）以覆盖 §68 集成行；两条路径均可用。
- 泄漏因子（factor=label）rankic_valid≈1.0、net_sharpe≈34，被识别为极端值红旗；合法动量因子 rankic≈0.08、net_sharpe≈2.2，区分显著。
- overlapping 直接年化（sqrt(252)）Sharpe 显著高于 cohort 真实 PnL Sharpe（inflation 存在），验证了 adapter 只输出 cohort 口径的必要性。

## 五、遗留项（不能自动核验 / 需人工确认）

1. **ExportScore 的「正式输出分」语义**：`export_score` 已存在且被 `export/__init__.py::decide_export` 消费，但「ExportScore 是否作为独立 Score 拆分对外暴露」的完整链路（含 ExportRegistry 落库）未在本文件覆盖——建议在导出端到端验收中补。
2. **真实数据端到端**：本文件全部合成数据；真实 COS 数据下的 QE cohort 指标数值口径需在数据接入验收中复核（不属本测试范围）。
3. **faiss 依赖**：`test_factor_assets_adapter.py` 的 faiss 用例为既有 skip，非本任务引入。

## 六、回归命令

```
PYTHONPATH=src OMP_NUM_THREADS=31 python3 -m pytest tests/ -q
# 425 passed, 1 skipped
```
