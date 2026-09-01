# AlphaPROBE 全量重构 — 基础层完成快照（给后续 subagents 的上下文）

**对象**：/home/sunhaiwei/quant_projects/alphaprobe/ashare/alphaprobe-dev/
**任务书**：/home/sunhaiwei/quant_projects/alphaprobe/AlphaPROBE_全量重构_AI开发实施总任务书.md
**代码地图**：ashare/alphaprobe-dev/ALPHAPROBE_CURRENT_CODE_MAP.md（P0 已完成，339 行）

## 已落盘的新基础层（主模型完成，勿重复建设）

```
src/alphaprobe/
├── contracts.py            # §5 ExperimentContext / §3.3 ResearchSplitSpec / §6.2 FactorBatch /
│                           # §8 EvaluationRecord / §26 SearchActionType / §75 全部契约 / §83 RejectionReason
├── research_protocol/      # §3.3 LeakageGuard + SealedTestAccess（L0-L4 sealed test 零读取）
│   ├── __init__.py         #   + §88.1 assert_sealed_test_zero_reads
│   ├── orientation.py      # §10 Train-only 方向标准化（f/-f 同 signal id）
│   └── survival.py         # §88.4/§44.1 survival cutoff 闸门（代码防线）
├── evalcache.py            # §12 EvaluationCache（全维度 cache key）
├── dedup.py                # §11 四层去重（canonical/sign-invariant/param family/256bit fingerprint）
│                           #   + §11.6 GlobalSeenIndex（atomic reservation + topk fingerprint neighbors）
├── integration/
│   ├── __init__.py         # §3.1/§8/§59/§67 四个 Protocol
│   ├── evaluator_client.py # §8 legacy_compat Evaluator（once-compute + cache + leakage guard）
│   └── refinement_client.py# §59/§67 client-only stub（未注入外部引擎时显式失败）
├── fitness/__init__.py     # §13-§21 MetricCalibrator(robust MAD+sigmoid+round freeze) /
│                           # P/Q/L/S/N 五维 / D10 cliff / isotonic / HardGates /
│                           # SearchValue/PoolUtility/ExportScore/survival shrinkage
├── pool/__init__.py        # §53-§55 ActivePool(Pareto+QD, 禁 pop lowest IC) + QDArchive + EmbeddingCache
├── memory/__init__.py      # §33-§43 SQLite GlobalMemoryStore（factor_nodes/aliases/action_edges(多父)/
│                           # attempts/evaluations/exploration/failure/survival/direction/regime/
│                           # export_registry/seed ingest/MemoryPacket）
├── lineage/__init__.py     # §37 ActionEdge 多父视图
├── search/__init__.py      # §25.5 GenerationRepairArm + §29 Thompson/UCB ActionScheduler +
│                           # §30 model_route + §56.3 LineagePatience
├── checkpoint/__init__.py  # §57 checkpoint v2 + §57.1 migrate_v1（test metrics 隔离）
├── export/__init__.py      # §58 ExportDecision gate（无 Top-N）
└── observability/__init__.py # §72-§74 RunManifest/CostLedger/Counters
```

**测试**（40 个全绿，`PYTHONPATH=src python3 -m pytest tests/ -q`）：
- tests/test_taskbook_units.py（§87 unit：canonical/orientation/calibrator/D10/SearchValue/Pool/Scheduler）
- tests/test_leakage_gates.py（§88：sealed test 零读取 + §3.2 eval/exec 静态 AST gate）

**已修**：trainer/pool.py 四处 eval() 已替换为 parse_mining_expression（§3.2）。
shared/alphagen_qlib/utils.py:27 eval 保留（legacy baselines 路径，gate 允许清单）。

## 关键现状（P0 侦察结论）

- monkey-patch：fe_bridge/bootstrap.py enable_factor_engine_evaluation（Expression.evaluate）仍在
- 批执行空白：fe_bridge/stock_data.py 硬编码 build_backend("pandas") + 逐公式 engine.run；
  FE 已有 run_many/run_many_parallel/CSE/AdaptiveBatchScheduler（签名见代码地图 §3.1）
- graph 单父字符串 identity：shared/alphagen/data/expression_knowledge_graph.py
- logger 读 test 写回节点：trainer.py AlphaKnowledgeLogger（load_test_res 硬编码 qlib）
- pool capacity 2000 pop lowest single IC：trainer/pool.py try_new_expr/_pop
- exporter 三窗全量重算：delivery/exporter.py → fe_bridge/metrics.py
- continuous 无跨轮记忆：src/alphaprobe/continuous.py
- quant_evaluator 可用但零引用（runtime/evaluator.py Evaluator + adapters/factor_engine.py FactorEngineAdapter）
- 冷启动库 data/ashare/ 目录为空（用户尚未放文件）→ 只做兼容，不阻塞

## 硬规矩（每个 subagent 必守）

1. 不修改 factor_engine/、data_access/ 内部语义；AlphaPROBE 侧适配
2. 不跑任何模型训练/LLM 调用（禁模型；测试用 stub）
3. 不新建第二套评估库：指标走 EvaluatorClient protocol
4. 旧路径先兼容再切默认；不 Big Bang；每步可 import
5. vwap-to-vwap 20d label 口径；purge>=20 trading days
6. Python 3.12；子进程 ≤31 核；不用 pytest 之外的框架
7. 改完必须：PYTHONPATH=src python3 -m pytest tests/ -q 全绿 + 逐模块 import 冒烟
8. 写进度到 /tmp/ap_refactor_progress.log（append）