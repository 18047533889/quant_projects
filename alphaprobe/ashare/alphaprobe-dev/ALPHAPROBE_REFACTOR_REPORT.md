# AlphaPROBE 全量重构实施报告（ALPHAPROBE_REFACTOR_REPORT.md）

> 日期：2026-09-01 ｜ 对象：`quant_projects/alphaprobe/ashare/alphaprobe-dev/`
> 任务书：`AlphaPROBE_全量重构_AI开发实施总任务书.md` ｜ 基线：main@重构前工作树
> 执行方式：主模型基础层 + 6 个并行 subagents（P2-P4 / P7 / P3,P5-P6 / P8-13 / P9 / P10）

# Baseline

重构前状态见 `ALPHAPROBE_CURRENT_CODE_MAP.md`（Phase 0 只读侦察，25 组件映射 + 13 项现状确认 + 10 条风险清单）。

# Current Code Mapping

见 `ALPHAPROBE_CURRENT_CODE_MAP.md` §2.3/§2.4。重构后新增目录结构：

```
src/alphaprobe/
├── contracts.py                  # §5 ExperimentContext / §3.3 ResearchSplitSpec / §75 全部契约 / §83 RejectionReason
├── research_protocol/            # §3.3 LeakageGuard + SealedTestAccess + §10 orientation + §88.4 survival cutoff
├── evalcache.py                  # §12 全维度 cache key
├── dedup.py                      # §11 四层去重 + §11.6 atomic reservation
├── identity 相关                  # dedup/contracts 承载（P2-P4 接线）
├── integration/                  # §3.1/§8/§59/§67 协议 + FE Adapter（run_many/CSE）+ legacy_compat evaluator
├── fitness/                      # §13-§21 calibrator/五维 SearchFitness/HardGates + §9 funnel
├── pool/                         # §53-§55 ActivePool(Pareto+QD) + admission（替代 pop-lowest-IC）
├── memory/                       # §33-§43 SQLite GlobalMemoryStore + seed_ingestion + retriever/MemoryPacket
├── lineage/                      # §37 多父 ActionEdge
├── search/                       # §25-§32 五臂 + Thompson/UCB scheduler + 结构化 LLM + orchestrator
├── survival/                     # §43-§51 DNA/survival profile/dna_stats(Wilson 收缩)/regime(2026 事件)
├── checkpoint/                   # §57 v2 + §57.1 v1 migration（test metrics 隔离）
├── export/                       # §58 ExportDecision gate（无 Top-N）
├── observability/                # §72-§74 RunManifest/CostLedger/Counters
└── continuous/round_manager.py   # §56 无限 7×24 RoundManager
```

# Phase 1（研究协议/封存 Test）

- changed: `contracts.py` `research_protocol/*` `trainer/trainer.py`（logger 不再算 test_ic/test_icir、不写回节点；load_test_res 改为 seal 说明行）
- tests: `tests/test_leakage_gates.py`（§88.1 L0-L4 sealed read==0、L5 需 frozen、§3.2 eval/exec AST gate）
- results: **新主链 eval/exec = 0**（grep+AST 双 gate）

# Phase 2-4（FE 原生路径/批执行/identity）

- changed: `integration/factor_engine_adapter.py`（validate/canonicalize/inspect/run_many，FE 不可用时降级）、`fe_bridge/stock_data.py`（evaluate_many 批量路径 + CSE）、`dedup.py` identity factory
- gate: FE batch parity 冒烟过；canonical 化简 §87.1 全过（含 x/x 禁止化简）

# Phase 3/5-6（漏斗/五维 SearchFitness/池）

- changed: `fitness/`（calibrator+P/Q/L/S/N+D10 cliff+HardGates）、`fitness/funnel.py`（L0-L5）、`pool/`（Pareto+QD+eviction 候选，禁 argmin(IC)）、`pool/admission.py`、`export/gate.py`
- tests: 22 funnel/admission 用例（低 IC 高 niche 不被弹等）

# Phase 7（Global Memory）

- changed: `memory/`（SQLite store 全表 + seed_ingestion + retriever）、`continuous.py`/`runner.py` 最小挂钩
- gate: 跨轮 rediscovery 同节点（§89.4）、A+B→C 双父（§89.5）测试过
- 冷启动：yaml 缺失 → 0 条不阻塞（用户文件未放入，兼容就绪）

# Phase 8/11/13

- changed: `continuous/round_manager.py`、`trainer/checkpoint.py`（v1+v2 双写、v2 恢复不重算）、trainer 封存 Test、runner/continuous `type=bool`→BooleanOptionalAction（0 残留）
- tests: `test_round_checkpoint.py` 6 用例（v2 save/load 一致、v1 兼容、RoundManager 2 轮 dry-run、封存断言）

# Phase 9/10

- changed: `search/arms.py` `structured_llm.py` `orchestrator.py`、`survival/{dna,profile,dna_stats,regime}.py`
- 2026 事件：`register_2026_event` 幂等注册；读取侧 cutoff 闸门（§44.1 代码防线）

# Leakage Audit

- §88.1 sealed test zero reads：LeakageGuard 拦截 + `assert_sealed_test_zero_reads()` CI 断言 ✅
- §88.4 survival cutoff：`assert_survival_visible` 代码防线 ✅
- §3.2 eval/exec：新主链 0 处（AST 静态 gate）✅
- §57.1 v1 migration：test metrics 隔离不进搜索记忆 ✅

# Performance

- pool 淘汰从 argmin(single IC) → Pareto+QD+SearchValue 综合分
- embedding：once-encode cache + simhash ANN Top-K（不做全池 N²）
- dedup：GlobalSeenIndex 索引级 lookup + topk fingerprint neighbors
- FE：evaluate_many 批路径接入 run_many/CSE
- P10 附带修复 factor_engine 侧 evidence 校验 O(n²)→fast-fail（未改 FE 语义，仅性能早退）

# Memory Schema

SQLite：factor_nodes/aliases/action_edges/attempts/evaluations/exploration_state/failure_memory/survival_profiles/direction_clusters/market_regime_events/source_ingestions/export_registry/system_versions（§39 全覆盖）

# 测试

`PYTHONPATH=src python3 -m pytest tests/ -q` → **124 passed**（36s）
6 文件：taskbook_units(31) + leakage_gates(9) + funnel_admission(22) + memory(8) + search_arms(18) + survival(30) + round_checkpoint(6)

# Known Limitations

1. **冷启动库文件未放入**（`cold_start_library/data/ashare/` 为空）——seed ingestion 兼容空目录返回 0 条；文件就位后 `ingest_cold_start_yaml` 直接可用
2. **LLM 真实调用未接通**（额度未确认，硬规矩不跑模型）——orchestrator/arms 全部 llm_fn 注入式，真 key 配好即插即用
3. Refinement/Consolidation 外部引擎未落地——client-only stub 显式失败（任务书 §59/§67 正确行为）
4. Evaluator 为 legacy_compat adapter——quant_evaluator 接线留给下一步（protocol 已就位，切换不动搜索内核）
5. checkpoint v2 恢复的 pool 值惰性求值（lazy）——首次访问时按需重算单条，不整池重算
6. 运行依赖（.venv）：torch/openai/sentence_transformers 已有，python-dotenv/gymnasium 本轮已补装

# Remaining Future Work

- quant_evaluator 真实接入（替 legacy_compat）
- 100k seed ingestion benchmark（§90.1）
- FE batch equivalence 抽样对拍（§89.1，需真实数据窗口）
- direction 层级聚类 reclustering（§48.2/§48.3）当前仅单层 rare_directions