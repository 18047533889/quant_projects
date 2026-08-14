# Quant Factor Platform — Continuous Deep Refactor & Audit Plan
## 给 Claude Code / AI 的一次性持续开发任务书

> **目标**：对当前本地量化因子基础设施做持续、系统、可验证的深度整改。不要只修首批已知问题；要持续发现新问题、持续分配给并行 Agent、完成后独立复核、失败则返工，然后继续补任务。
>
> **本地工程根目录**：`/home/shw/quant_projects`  
> **唯一代码事实源**：当前本地 working tree。远端镜像只作为只读参考。  
> **禁止**：不要创建远端 PR，不要 push，不要在远端修改文件。  
> **持续时间**：环境允许时目标至少约 6 小时 wall-clock；若 session/平台提前结束，必须持久化任务账本和下一批 READY 任务，以便下一会话立即续跑。

---

# 0. 主 Claude 开工顺序

1. `cd /home/shw/quant_projects`
2. 先扫描当前本地：
   - 目录树
   - `git status`
   - 用户未提交改动
   - import graph
   - 各 package `pyproject.toml`
   - 测试基线
   - `.claude/`
   - README / IMPLEMENTATION / COMPLETION 报告
3. 不得覆盖用户已有改动。
4. 建立：
   - `AI_REFACTOR_TASK_LEDGER.md`
   - `AI_REFACTOR_FINDINGS.md`
   - `AI_REFACTOR_DECISIONS.md`
5. 本文已知问题全部转成任务；审计 Agent 继续发现新问题并追加。
6. 四个新库并行推进，不要串行“一个包全部做完再下一个”。
7. 每项实现都由**不同 Agent**独立 review。
8. worker 完成后，Lead 立即分配下一 READY 任务。
9. 每 30–45 分钟执行一次全局 audit sweep。
10. 每 60–90 分钟执行一次 integration wave。
11. 首批 checklist 清空后继续 property / metamorphic / fuzz / benchmark / extraction / dead-code / dependency audit，直到满足收敛条件。

---

# 1. 架构边界

```text
DataAccess
= 数据读取/写入/PIT/快照/存储/数据路由

FactorEngine
= 因子 DSL/AST/IR/算子/编译/计算/因子物化

QuantEvaluator
= Evidence / Metrics / Diagnosis

FactorOptimizer
= 因子修复/变异/搜索：U-shape、tail、threshold、窗口、平滑、interaction 等

FactorAssets
= Identity / Seen / Screening / Novelty / Similarity / Clustering /
  Selection / Aggregation / Lifecycle

FactorPreprocess
= 模型前表示：Winsor / Rank / Z-score / Neutralization /
  Fitted Transform / Model-specific View
```

## 禁止重新实现

四个新库禁止再造：
- Data Lake / Storage Backend
- Parquet / DuckDB / COS 通用读取层
- PIT / Calendar / Universe / Snapshot Engine
- Factor DSL / Parser / AST / IR / Operator Engine
- Factor Materialization
- 第二套通用缓存平台
- 第二套通用分布式 DAG Scheduler

写新基础设施前必须证明 DA/FE 没有对应能力。

---

# 2. 持续 Subagent / Agent Team 调度

## 2.1 推荐组织

如果当前 Claude Code 支持 Agent Teams：

```text
Program Lead
├── Task Queue Manager
├── Integration Controller
├── Audit Controller
├── Simplification Auditor
├── QE workers
├── FO workers
├── FA workers
├── FP workers
└── Independent Red Teams
```

若 Agent Teams 未启用，则由主 Claude 作为唯一调度者，多 Subagents 并行完成独立任务。Task Queue Manager 负责分析任务队列和建议下一批工作；主 Claude 负责真正 dispatch / resume。

环境允许时保持约 **6–12 个独立工作单元**活跃；不要让大量 Agent 同时编辑 shared files。

共享文件只由 integrator/lead 修改：
- `pyproject.toml`
- public `__init__.py`
- 公共 contracts
- 根配置
- integration fixtures
- `.claude` 共享规则

## 2.2 必须有的管理 Agent

### `program-director`
- 持全局依赖图
- 冻结/管理公共合同
- 防止重复实现 DA/FE
- 每轮 integration 后重排优先级

### `task-queue-manager`
持续扫描：
- READY / BLOCKED / REVIEW / REWORK
- failed tests
- TODO/FIXME
- NotImplemented
- dead code
- docs drift
- dependency drift
- benchmark regression
- extraction failure

READY 少于 5 时主动挖新任务。

### `integration-controller`
检查跨包 API、contracts、imports、循环依赖、package isolation。

### `audit-controller`
任何实现：
`IMPLEMENTED -> REVIEW -> PASS/DONE 或 FAIL/REWORK`

### `simplification-auditor`
只找可删除内容：
- duplicate cache
- duplicate planner
- duplicate IO/PIT
- unused wrapper
- dead compatibility
- placeholder
- generic manager/service/base/core/utils 过度抽象

---

# 3. 建议 Agent 池

### QE
`qe-contract-auditor`, `qe-math-auditor`, `qe-runtime-auditor`,
`qe-stream-auditor`, `qe-cache-auditor`, `qe-performance-engineer`,
`qe-metric-catalog-worker`, `qe-diagnosis-worker`, `qe-regression-tester`

### FO
`fo-split-safety-worker`, `fo-shape-repair-worker`, `fo-grammar-worker`,
`fo-search-worker`, `fo-multifidelity-worker`, `fo-plateau-worker`,
`fo-llm-worker`, `fo-overfit-auditor`, `fo-trial-ledger-worker`

### FA
`fa-identity-worker`, `fa-seen-index-worker`, `fa-registry-worker`,
`fa-screening-worker`, `fa-novelty-worker`, `fa-similarity-worker`,
`fa-ann-worker`, `fa-leiden-worker`, `fa-selection-worker`,
`fa-aggregation-worker`, `fa-lifecycle-worker`, `fa-adapter-worker`

### FP
`fp-transform-worker`, `fp-neutralization-worker`, `fp-fitted-state-worker`,
`fp-representation-worker`, `fp-backend-worker`, `fp-performance-worker`,
`fp-leakage-auditor`

### Red Team
`pit-leakage-redteam`, `math-stat-redteam`, `cache-correctness-redteam`,
`concurrency-determinism-redteam`, `numerical-stability-redteam`,
`dependency-boundary-redteam`, `package-extraction-redteam`,
`api-contract-redteam`, `performance-memory-redteam`,
`docs-version-redteam`, `license-redteam`, `legacy-miner`,
`corpus-regression-redteam`

---

# 4. 任务账本格式

每个任务必须记录：

```text
ID
Package
Severity
Status: TODO/READY/IN_PROGRESS/REVIEW/REWORK/DONE/BLOCKED
Owner
Reviewer
Dependencies
Write Scope
Problem
Evidence
Required Change
Tests
Benchmark
Acceptance Criteria
Notes
```

P0–P3 全部最终处理，priority 只是顺序，不代表 P3 可永久忽略。

---

# 5. 全平台审计维度

每项输出 PASS / FAIL / N/A + 证据。

1. Architecture boundaries
2. Cross-package dependencies
3. Duplicate DA/FE capability
4. Public API
5. Contracts/schema/version
6. Package version vs schema version
7. Factor identity/canonical hash
8. PIT/information time
9. Label timing/start/end
10. Execution delay
11. Train/validation/test split
12. Sealed test
13. SplitUsage/upstream selection history
14. Fitted preprocessing leakage
15. Same-day vs matured labels
16. Mathematical correctness
17. Pearson/Spearman/ties
18. IC/ICIR/HAC/bootstrap
19. Quantile/tail/monotonicity
20. Turnover/probe portfolio
21. Exposure regression
22. Residual/conditional novelty
23. Axis T/N/F semantics
24. Mask alignment
25. NaN/Inf/constant/near-constant
26. Numerical stability/condition number
27. Cache identity/invalidation
28. Chunk merge correctness
29. Streaming exactness
30. Parallel determinism
31. RNG isolation
32. multiprocessing memory copies
33. BLAS/Numba oversubscription
34. Performance/vectorization
35. Factor blocks
36. O(K²) paths
37. Screening hard vs soft gates
38. Correlation/redundancy
39. ANN recall
40. Sparse graph
41. Leiden correctness
42. Small-universe clustering fallback
43. Selection/aggregation
44. Lifecycle/governance
45. Persistence/transactions
46. Raw factor preservation
47. Neutralization/standardization
48. Model-specific representation
49. Packaging/wheel contents
50. Standalone extraction
51. Optional dependencies
52. Namespace collisions
53. Path/env assumptions
54. Secrets/security/pickle/eval
55. Docs/code drift
56. License/provenance
57. Legacy migration
58. Corpus regression
59. Property/metamorphic tests
60. Benchmark/memory regression

---

# 6. QuantEvaluator 已确认问题与任务

## QE-001 Public API 文档漂移
README/顶层说明声称有稳定 `evaluate` facade 时，必须确认实际 public export 和实现一致。冻结真实 API，examples/tests/README 同步。

## QE-002 Cache key 身份不足
必须绑定：
- factor value/snapshot identity
- label identity/content
- validity mask
- LabelSpec
- SplitPlan
- evaluation view
- metric version
- config
- universe/snapshot

同 IDs/shape、不同 values/labels 必须 miss。

## QE-003 Generic chunk aggregation 不安全
禁止通用“scalar 平均 + array concatenate”。MetricSpec 必须声明：
`NOT_CHUNKABLE / WEIGHTED_REDUCIBLE / STATEFUL_MERGE / CONCAT_TIME / CONCAT_FACTOR / CUSTOM`

## QE-004 Cross-sectional metric 不可随意切 asset axis
RankIC、quantile、tail 必须看到完整当日横截面。

## QE-005 Streaming chunk 不能靠 shape heuristic 判断坐标
chunk 必须显式携带 time/asset/factor start/end。

## QE-006 Streaming coverage 不能平均 chunk fraction
累加 valid_count / total_count。

## QE-007 Pearson / Spearman streaming 语义分离
原始 sufficient statistics 只能证明 Pearson；RankIC 需要完整 rank 语义。

## QE-008 “constant memory / 100k factors” 必须 benchmark 证明
测 100/1k/10k/100k、peak RSS、block size。

## QE-009 Parallel RNG 非确定性
统一 master seed -> deterministic child seeds。

## QE-010 Multiprocessing 可能复制大型 FactorBatch
审计 pickle/fork/spawn，避免每 worker 拷完整矩阵。

## QE-011 Speedup 报告必须真实
区分 measured sequential baseline、parallel wall time、throughput、peak memory。

## QE-012 Worker error 不只 `str(e)`
保存 exception type、metric、block、request id、traceback ref。

## QE-013 QE 缓存过度平台化
当前若存在 simple cache + advanced L1/L2/Redis/compression/warming，应大幅收缩。QE core 默认只需 request/runtime scoped intermediate cache。没有 benchmark 证明价值的 disk/Redis 从 core 移除。

## QE-014 Metric Registry 单一真值
MetricSpec 应有：
- id/version
- reference/fast fn
- dependencies
- min observations
- axis requirements
- chunk/stream semantics
- matured-label requirement
- determinism
- cost tier
- tolerance

## QE-015 Metric tiers
建议 `T0_FAST / T1_CORE / T2_ROBUST / T3_RESEARCH`。

## QE-016 half-life 命名拆分
IC AR(1) persistence 与 horizon-decay half-life 不得混为同一 metric。

## QE-017 same-day health 与 future-label performance 分开
立即可算 coverage/drift/turnover/exposure/breadth；RankIC 等必须等 label 成熟。

## QE-018 QE 不做 universal admission
类似 coverage<0.5 不能作为全局资产拒绝政策；QE 提 evidence，FA 决策。

## QE-019 Constant check
不要只 `std == 0`；增加 tolerance、unique ratio、tie ratio、effective breadth、per-date diagnostics。

## QE-020 全历史 flatten 诊断不足
同时输出 per-date distribution/coverage/ties/drift。

## QE-021 Python per-factor diagnosis loop
改 batch/block fast path。

## QE-022 QE core 不构造第二套 forward return
只接受 explicit LabelBundle；LabelBuilder 若有只能 optional adapter。

## QE-023 Probe portfolio 命名
QE long/short quick test 统一叫 `probe_portfolio.*`，不可伪装真实 execution backtest。

## QE-024 Reference/Fast parity
旧 FactorAnalyzer/Exposures/batch_metrics 只做 reference mine，不能整体复制。

## QE-025 防旧 bug 回流
测试覆盖旧 `if type == "ic" or "rank_ic"`、hardcoded symbol column、label shift 方向问题。

## QE-026 Wheel inventory
当前 explicit package list 模式必须有 wheel inventory/extraction test，防新增子包漏打包。

---

# 7. FactorPreprocess 已确认问题与任务

## FP-001 严重边界：FP 不应 import QE runtime cache
若当前 `factor_preprocess` 直接 import `quant_evaluator.runtime.cache_v2.MultiLevelCache`，立即解耦。FP core 必须能在未安装 QE 的 fresh venv 中 import。

## FP-002 FP 多级缓存大概率过度设计
Disk/Redis/compression/warming/global singleton 全部交 simplification auditor。没有实际 benchmark 证明就删/隔离。

## FP-003 Global cache 污染 tests/config
优先无 cache 或 dependency injection。

## FP-004 若保留 cache，key 必须绑定 fitted state/exposure/split/snapshot/transform version。

## FP-005 Stateless/Fitted 类型级分离
Stateless: rank/zscore/MAD winsor/quantile winsor/demean  
Fitted: PCA/ICA/Box-Cox λ/learned neutralizer/supervised transforms

## FP-006 严禁 fit_transform(full_data)
必须 `fit(train)`、`transform(validation/test)`。

## FP-007 Raw 永远保留
任何 transform 不得 inplace 覆盖 source factor。

## FP-008 OLS neutralization T×F Python loop
保留 Reference；开发 batched/fast path。

## FP-009 Neutralization 数学健壮性
intercept、industry dummy k-1、rank deficiency、condition number、small N、weights、exposure missing、failure policy。

## FP-010 Exposure PIT
industry/size/liquidity/beta 全部由 caller 显式提供且必须 PIT。

## FP-011 中性化不是全局强制
支持 RAW/RANK/ZSCORE/INDUSTRY_RESIDUAL/SIZE_RESIDUAL/LIQUIDITY_RESIDUAL/FULL_RESIDUAL 多 view。

## FP-012 Neutralization survival
可由 QE 比较 `abs(IC_residual)/(abs(IC_raw)+eps)`，FP 不负责 reject。

## FP-013 Model-specific policies
至少 LinearReady / TreeReady / NeuralReady。

## FP-014 Transform Registry
记录 stateless/fitted、causal、params、dtype、group requirement、backend、production tier。

## FP-015 Backend 不允许“Polars -> Pandas -> Polars”伪 fast path。

## FP-016 Edge cases
zero std、ties、single-member industry、empty group、NaN、Inf、unseen category。

## FP-017 Feature lineage
source factor、transform、fitted state、exposure spec、version、fit split。

## FP-018 Experimental methods 默认关闭
RF/GBDT/PCA/ICA neutralization 等 research-only、fitted-only。


---

# 8. FactorOptimizer 已确认问题与任务

## FO-001 External SplitPlan 第一优先
上游已有 Train/Validation/Test 就直接使用，FO 不重新切。

## FO-002 默认不使用模型式 rolling/expanding walk-forward
本库主任务是因子表达修复：U-shape、tail、threshold、window、smoothing、interaction 等。默认使用 Train/Validation/Sealed Test。只有 mutation 内部本身含 fitted predictive model / regime classifier 时再加强时序验证。

## FO-003 Split 权限
Train：fit/diagnosis/orientation/hypothesis  
Validation：child vs parent、参数选择、plateau、Pareto、early stop  
Test：candidate freeze 后 final confirmation，搜索过程完全不可见。

## FO-004 Sealed Test API
正常 search objects 不得出现 test metric、full-history including test、test rank。

## FO-005 Test contamination ledger
记录 first access、count、candidate hash、transform hash、split plan。test 后定义/参数/preprocess 改变 => `TEST_CONTAMINATED`。

## FO-006 SplitUsage
记录 validation 是否被上游用过、candidate count seen、hyperparameter use、selection depth。

## FO-007 SearchRunner `max_concurrency`
若配置存在但 runner 实际串行：实现真实并发或移除误导字段。

## FO-008 `_validate_trial` 不能 placeholder
必须执行 grammar、FE legality/type、PIT、field availability、parameter bounds、complexity、duplicate/seen、target consistency。

## FO-009 Multi-fidelity 真正接入 runner
不能只定义 L0/L1/L2 而 run_search 永远 L0。

## FO-010 Fidelity score 不可裸比较
不同样本/metric set 必须使用 comparable utility / confidence / parent-relative delta / deterministic cohort。

## FO-011 Pareto 真正使用
目标至少含 validation alpha、complexity、turnover、fragility、novelty、data dependency，而非单一 best_score。

## FO-012 Search plateau vs Parameter plateau
前者是搜索停滞，后者是参数邻域稳定，必须分别实现。

## FO-013 RepairMapper 缺 Shape Diagnosis
必须支持：
`LINEAR, MONOTONIC, U_SHAPE, INVERTED_U, TOP_TAIL, BOTTOM_TAIL, THRESHOLD, SATURATING, CONVEX, CONCAVE, BIPOLAR, NON_MONOTONIC, NO_SIGNAL`

## FO-014 U-shape repair
候选：
- absolute distance from train-fitted center
- centered square
- symmetric tail rank
- piecewise/two-tail score

center 不得由 test 决定。

## FO-015 Tail repair
TOP/BOTTOM tail 生成 percentile threshold、smooth sigmoid/tanh、piecewise score；threshold 只用 validation 选。

## FO-016 Timing violation 不能自动“+1 lag”
出现 future/PIT violation 默认 quarantine/reject，重新修 definition timing 并完整验证。不能假设多 lag 一天自动合法。

## FO-017 Poor coverage 不能默认 FILTER_UNIVERSE
否则会通过删难算股票“优化”表现。Universe 改动必须是 explicit strategy-level policy。

## FO-018 High turnover 不能默认改 prediction horizon
Horizon 是研究 target。FO 应优先 smoothing/EMA/persistence/hysteresis，而不是静默换 label。

## FO-019 LOW_SIGNAL 不要无依据 ADD_INTERACTION
必须有 diagnosis/mechanism/domain constraints，控制 combinatorial explosion。

## FO-020 Normalization/Winsor 与 FP 边界
最终 representation 属于 FP。只有把 transform 明确定义成新 signal candidate 时 FO 才生成新 factor lineage。

## FO-021 Repair “confidence” 不可伪装统计置信度
纯 heuristic 就命名 `heuristic_priority`。

## FO-022 Mutation type 使用 typed registry
不要裸 string。

## FO-023 LLM 只是 proposal
必须经过 schema/grammar/FE/domain/timing/complexity/data/seen validation。

## FO-024 LLM robustness
schema repair、retry limit、candidate cap、unknown field reject、model/prompt/version provenance。

## FO-025 FO 的 `AdmissionPolicy` 易与 FA admission 混淆
FO 的语义其实是 candidate/trial preflight。建议改名 `TrialPreflightPolicy` 或 `CandidateEvaluationGate`。

## FO-026 Parent quality 不做普遍硬门槛
弱 parent 可能经过 nonlinear repair 变强。

## FO-027 Search dedup
接 GlobalSeen/campaign seen，避免重复算 canonical same candidate / same params / known failed candidate。

## FO-028 Multiple testing ledger
campaign、parent、generation、tested count、mutation family、parameter trials 必须记录。

## FO-029 Negative controls
random signal、label shuffle、time shuffle、wrong lag、cross-section shuffle，评估搜索是否在挖噪声。

---

# 9. FactorAssets 已确认问题与任务

## FA-001 Registry 持久化
若当前还是 in-memory，第一版上 SQLite WAL + migration + transactions；暂不需要 PostgreSQL/Neo4j。

## FA-002 SeenIndex 持久化
保存 active/failed/rejected/shadowed/retired/corpus 的 first seen + repeated encounters。

## FA-003 duplicate encounter 不能丢 provenance
同 canonical hash 再出现，记录 encounter event。

## FA-004 FE identity adapter string contract
如果 signature 接受 str 但内部直接 NotImplemented，真实接 FE public parser/definition API，或把 public contract 改成只接受实际支持类型。

## FA-005 Canonical hash 由 FE 真值提供
FA 不再重造 hash semantics。

## FA-006 FE compiler generation 不硬编码
从 provider metadata 获取。

## FA-007 DA adapter stub
如果 constructor/method 永远 NotImplemented：
- 真正实现 optional adapter；或
- 移出 production public surface。
不能报告“完成”但入口必然失败。

## FA-008 QE adapter primary metric
不能“字典里第一个 valid metric”作为 primary。Evaluation objective 显式传入。

## FA-009 schema version / QE package version 分离
Evidence provenance 分字段。

## FA-010 类型/default 清理
例如 `Dict[str, any]`、mutable/None defaults 等。

## FA-011 Sparse graph 升级 multiview
Edge 至少可携带 signal/PnL/tail/exposure/horizon/regime/structure components。

## FA-012 Isolated nodes
无高相关边的独立 factor 必须留在 graph universe。

## FA-013 当前 simplified modularity 不可冒充生产 Leiden
生产主算法使用成熟 Leiden 实现（例如 igraph/leidenalg 或等价库）并记录 version/seed/resolution。自写近似只保留 baseline/reference。

## FA-014 Hierarchical dense n×n 只允许小规模
大规模自动 route sparse pipeline。

## FA-015 Ward 与 `1-|corr|` arbitrary distance
限制不合法组合；小规模 reference 优先 average linkage。

## FA-016 Sign orientation
先处理 factor orientation，再定义 abs/signed similarity，不允许隐式全 abs。

## FA-017 Scale-aware organizer
```text
<20       no formal clustering
20–100    hierarchical optional
100–1k    hierarchical / sparse graph
1k+       fingerprint -> ANN -> exact similarity -> sparse graph -> Leiden
```
阈值为 config，不是硬编码金融真理。

## FA-018 HNSW 只做 neighbor recall
必须 exact multiview rerank 后再建 mutual-kNN/threshold graph。

## FA-019 ANN recall benchmark
小样本 brute-force 对照 Recall@K、memory、latency。

## FA-020 Correlation 不作单独硬删除
示例 policy：
- <.90：通常无 redundancy action
- .90–.98：review
- >=.98：near-duplicate candidate
最终还看 residual IC、PnL corr、tail overlap、exposure、horizon、regime。阈值全部可配置。

## FA-021 相关性主定义
主线为 daily cross-sectional Spearman，再 aggregate mean/median/P10/P90/recent/stability；不要只 pooled flatten Pearson。

## FA-022 Conditional Novelty
实现 linear/nonlinear/portfolio residual novelty，并严格区分 in-sample 与 OOF。

## FA-023 novelty provider ≠ novelty algorithm
如果当前只是 evidence provider，不能把模块名当已完成功能；确认真实算法存在。

## FA-024 Representative 不默认 MAX_IC
生产代表因子看 stability、novelty、coverage、PIT、turnover/cost、fragility、complexity、evidence freshness。

## FA-025 MIN_CORRELATION O(K²)
大规模只对 graph neighbors / cluster / ANN shortlist。

## FA-026 “EQUAL_WEIGHT” 语义
若只是按注册顺序等间隔抽样，必须修。Equal family/subfamily representation 应基于 membership。

## FA-027 RANDOM 不修改 global RNG
使用 local Generator。

## FA-028 Screening decision
使用：
`KEEP_CORE, KEEP_DIVERSE, OPTIMIZE, REGIME_CONDITIONAL, EXPOSURE_CONDITIONAL, SHADOWED, QUARANTINED, REJECTED`

## FA-029 Hard Gate
Hard reject 只针对 PIT/future、numeric failure、catastrophic coverage、illegal data、near-constant、implementation bug、exact identity duplicate handling等。禁止 `RankIC < threshold => reject`。

## FA-030 Repairable factors 先送 FO
U-shape、tail、threshold、outlier、high-turnover 不应被 linear IC gate 误杀。

## FA-031 Lifecycle
补 SHADOWED、REGIME/EXPOSURE conditional、QUARANTINED、revival/re-evaluation/re-approval 等。

## FA-032 Shadow != Delete
保存 shadowed_by、reason、evidence、date、revival conditions。

## FA-033 FA 不存 T×N×F 大矩阵
只存 refs、metadata、fingerprints、relations、evidence refs、lifecycle。

---

# 10. DataAccess / FactorEngine 边界任务

## CORE-001
持续扫描四库是否复制 DA/FE。

## CORE-002
FE 历史 generic 顶层 namespace（如 `expr`, `api`, `backend`）若仍存在，记录 collision debt。不要在四库重构同时无计划发动巨型 namespace 迁移。

## CORE-003
Adapters 最终只用 FE stable public API，不用 private top-level internals。

## CORE-004
审计 FE workspace/root/env resolver，core 不应假定 parent 一定是 `quant_projects`。

## CORE-005
DataAccess package metadata 是版本真值；README/version drift 修正。

## CORE-006
DA/其他显式 package list 有 wheel inventory test，防子包漏打包。

---

# 11. 因子筛选必须落地的流程

```text
RAW FACTORS
    ↓
Exact / Structural Identity
    ↓
Hard Quality Gate
    ↓
QE Evidence + Shape Diagnosis
    ↓
GOOD                 REPAIRABLE
 │                       │
 │                 FactorOptimizer
 └──────────────┬────────┘
                ↓
           Re-evaluate
                ↓
        Information Gate
                ↓
     Correlation Cheap Screen
                ↓
       Conditional Novelty
                ↓
      KEEP / SHADOW / REJECT
                ↓
       Fingerprint / ANN
                ↓
 Exact Multi-view Similarity
                ↓
         Sparse Graph
                ↓
 Leiden / small-set fallback
                ↓
       Family / Cluster
                ↓
 Selection / Aggregation
                ↓
      FactorPreprocess
                ↓
      ModelInputFactorSet
```

关键规则：聚类前筛掉“没有任何可利用信息”的因子，而不是筛掉“RankIC 不高”的因子。

---

# 12. 相关性与硬约束

相关性是 cheap gate，不是最终裁判。

主相似度：
- daily CS Spearman
- PnL correlation
- top/bottom K Jaccard
- exposure cosine
- horizon IC profile
- regime IC profile
- structural/genealogy

Near-duplicate Hard Shadow 可要求组合条件，例如：
- signal rank corr 极高
- PnL corr 极高
- residual/conditional IC 很小
- tail overlap 很高

所有阈值由 policy config 管理，不写死进算法。

高 corr 但 residual IC 高 => KEEP。  
corr 没那么高但 residual IC≈0、PnL/tail 极相似 => 仍可 SHADOW。

---

# 13. 聚类

生产主线：

```text
Factor Fingerprint
 -> HNSW/ANN candidate neighbors
 -> Exact multiview similarity
 -> Mutual-kNN sparse graph
 -> Leiden
```

Leiden 是主 clustering；ANN 不是 clustering。

小因子数量时不要强行 Leiden：
- <20：不正式聚类
- 20–100：hierarchical optional
- 中等：hierarchical/sparse
- 大规模：ANN+sparse+Leiden

保留 hierarchical 作为小规模 reference/audit；HDBSCAN 可做辅助，不做唯一真值。

---

# 14. FactorPreprocess 模型前处理

必须明确存在并可独立使用。

同一个 factor 可以输出：
- RAW
- RANK
- WINSORIZED
- ZSCORE
- ROBUST_ZSCORE
- INDUSTRY_RESIDUAL
- SIZE_RESIDUAL
- LIQUIDITY_RESIDUAL
- INDUSTRY_SIZE_RESIDUAL
- FULL_RESIDUAL
- MISSING_INDICATOR
- FRESHNESS

不要覆盖 raw。

Model policy：
- Linear：更偏 rank/robust scale/residual
- Tree：raw/rank/exposure/missingness
- NN：normalized raw/rank/residual/exposure/missing/freshness

Neutralization 不是全系统硬约束；记录 exposure dependency，交模型/策略 policy 选择。

---

# 15. FO Synthetic Acceptance Tests

## U-shape
构造有噪声、非退化数据：
- raw RankIC ≈ 0
- 两端 future return 高、中间低

必须：
1. QE 诊断 U_SHAPE
2. FO 不直接 abandon
3. 生成 center-distance/square/two-tail
4. fit 参数只看 train
5. validation 选 child
6. test freeze 前不可见
7. transformed validation evidence 改善

## Top-tail
raw IC 可弱但 top decile 有稳定 alpha；FO 生成 smooth threshold，禁止阈值过拟合到极少样本。

## Plateau
稳定的窗口邻域优先于孤立 spike。

---

# 16. FA Acceptance Tests

### High correlation + real novelty
corr=.94，residual IC=.018 => 不可仅因 corr shadow。

### Moderate correlation + no novelty
corr=.89，PnL=.97，tail overlap=.92，residual IC=.001 => 可 shadow。

### Small set
12 factors 不强行 ANN/Leiden。

### Scale
100k fingerprint/metadata 下不得创建 full O(K²) matrix。

---

# 17. FP Acceptance Tests

### Future poisoning
修改 validation/test 极端值，train fitted state/output 完全不变。

### Neutralization synthetic
factor = size + industry + true residual + noise；neutralized 后 size/industry exposure 显著下降，residual 保留。

### Raw preservation
任何 pipeline 后 source raw 可追溯。

---

# 18. QE Test Matrix

必须至少：
- Unit
- Golden
- Property
- Metamorphic
- Reference/Fast parity
- Batch/Factor-block parity
- Chunk parity（只对声明可 merge metric）
- Streaming/Batch parity（只对 exact streamable）
- Determinism
- Numerical stability
- Edge cases
- Performance/Memory

RankIC metamorphic：
- `f' = 10f + 5` -> unchanged
- `f' = -f` -> sign reversed
- aligned asset permutation -> unchanged

---

# 19. 必须捕获的“能跑但错”案例

1. same IDs/shape + different values => cache miss
2. same factor values + different labels => cache miss
3. same data + different split => fitted/cache identity 分离
4. uneven chunks => exact coverage
5. factor chunk 2 不可被 streaming 当 time chunk 2
6. asset-chunk local quantile/rank 必须 refuse 或采用正确全横截面算法
7. turnover time chunk 保留 boundary state
8. multiprocessing same seed deterministic
9. representative random 不污染 global RNG
10. FO search object 不含 test metric
11. optimizer 不可访问 `all=train+validation+test`
12. mutation after test => contamination
13. FP fresh venv 无 QE 可 import
14. FA fresh venv 无 QE/FE/DA 可 import core
15. QE fresh venv 无 FE/DA
16. FO core 无 adapter extras 可 import

---

# 20. 性能任务

先 baseline，后优化。

至少 factor counts：
- 100
- 1,000
- 5,000
- 10,000

测：
- coverage
- RankIC
- quantile
- turnover
- decay
- neutralization
- factor fingerprint/similarity

记录：
wall time、CPU、peak RSS、throughput、allocations、block size。

流程：
`Reference -> Golden -> Profile -> Fast Kernel -> Parity -> Benchmark`

---

# 21. Package Extraction

四包逐个：

1. 复制到 temp
2. 不把 monorepo 放 PYTHONPATH
3. fresh venv
4. pip install
5. import
6. tests
7. inspect wheel
8. optional extras 分别安装测试

特别：
- QE 无 DA/FE
- FO core 无 QE/FE
- FA core 无 QE/FE/DA
- FP **无 QE runtime cache dependency**

---

# 22. Legacy Migration

旧代码逐 symbol 分类：
`REUSE / REWRITE / REFERENCE_ONLY / DISCARD`

重点采矿：
- old factor evaluation
- FactorAnalyzer
- Exposures
- AlphaPurifier
- APr_utils
- toolkit cross-sectional/registry
- admission catalog
- Gateway contracts/complexity
- factor_agent orchestration

禁止旧 AutoFactor 大系统的 embedded DA/FE/cache/materialization/gateway IO 回流。

---

# 23. Corpora / Regression

长期用：
- GTJA
- Week2
- 当前真实 factor manifests
- synthetic edge cases
- 许可允许的 external corpora

用作 FE/QE/FO/FA/FP regression workload，不作为 runtime dependency。

---

# 24. 每 30–45 分钟 Audit Sweep

执行：
1. tests
2. import graph
3. TODO/FIXME/NotImplemented
4. duplicated capabilities
5. dead code
6. cross-package private imports
7. benchmark regression
8. leakage scan
9. docs drift
10. extraction sample

新问题立即进 ledger。

---

# 25. 每 60–90 分钟 Integration Wave

- 四包 tests
- integration tests
- contract compatibility
- extraction smoke
- benchmark sample
- simplification audit

失败 -> REWORK。不得无理由 skip/xfail。

---

# 26. “持续至少约 6 小时”工作循环

不是 sleep，而是持续有价值工作：

```text
while environment_allows:
    refresh_local_state()
    ingest_worker_results()
    send_completed_to_independent_review()
    reopen_failed_reviews()
    discover_new_tasks()
    unblock_ready_tasks()
    fill_idle_workers()
    run_periodic_regression()
    run_periodic_integration()
    update_ledger()
```

首批任务完成后继续：
1. property tests
2. metamorphic tests
3. fuzz
4. 10k/100k scale benchmark
5. memory profile
6. extraction
7. API compatibility
8. legacy corpus
9. dead code removal
10. duplicate abstraction removal
11. doc/code drift
12. optional dependency matrix
13. thread/process determinism
14. cache adversarial tests
15. future poisoning
16. negative research controls
17. cluster stability
18. parameter perturbation robustness

不得制造无价值 TODO 来凑时间。

---

# 27. 收敛条件

不以“6 小时到了”为成功。至少：

- 所有已知 P0–P3 DONE 或有真实外部阻塞
- core tests green
- integration green
- extraction green
- no accidental DA/FE duplication
- no undeclared imports
- no known test leakage
- no known mathematically invalid chunk merge
- no known cache false-hit
- FO U-shape/tail/split tests green
- FA real Leiden path + small-set fallback green
- FP fitted leakage tests green
- benchmark 无灾难性退化
- **连续两轮独立 audit sweep 没有新的高/中优先级 actionable defect**

若平台提前结束，持久化 ledger + READY queue + RESUME 指令。

---

# 28. 禁止“假完成”

不得：
- README 宣称 complete 但代码 stub
- 用 NotImplementedError 占位后说完成
- 把 failing tests xfail 掉
- 删除困难测试
- hardcode sample
- 用 test 选参数
- 静默改 label/horizon
- 过滤 universe 抬表现
- 为 tests green 偷改 metric semantics
- 复制旧实现不做 parity
- 加抽象层掩盖重复
- 把已知重要 bug 标 future improvement 就停止

---

# 29. 第一波并行 READY 任务

### Architecture
- local import graph
- DA/FE duplication scan
- extraction baseline
- all-package test baseline

### QE
- cache identity
- chunk/stream correctness
- registry/public API
- cache_v2 simplification
- performance baseline

### FO
- external split + sealed test
- RepairMapper shape redesign
- SearchRunner concurrency/multifidelity
- parameter plateau
- LLM grammar validation

### FA
- durable registry/seen
- multiview similarity
- actual Leiden
- screening/conditional novelty
- representative redesign

### FP
- remove QE cache dependency
- stateless/fitted
- OLS math/performance
- model representation
- backend parity

### Red Team
- leakage
- math
- dependency
- simplification
- determinism

---

# 30. 最终本地报告

持续更新：

### `AI_REFACTOR_TASK_LEDGER.md`
所有任务状态。

### `AI_REFACTOR_FINDINGS.md`
问题和证据。

### `AI_REFACTOR_DECISIONS.md`
架构决策及原因。

结束/中断前给用户：

```text
Completed
Reworked
Still Blocked
New Issues Found
Tests
Benchmarks
Extraction
Architecture Debt Remaining
Recommended Next Session
```

---

# 31. 最终目标

```text
DataAccess
    ↓
FactorEngine
    ↓
QuantEvaluator
    ├── FactorOptimizer feedback loop
    ↓
FactorAssets
    ↓
FactorPreprocess
    ↓
ModelInputFactorSet
```

角色：
- DA = Data Truth
- FE = Compiler / Compute Engine
- QE = Judge
- FO = Research Optimizer
- FA = Asset Governance
- FP = Model Representation
- LLM = Researcher, not authority

**不要把本文当静态 checklist。本文只是 initial backlog。真正职责是持续从本地代码发现问题、创建任务、并行修、独立审、失败返工、再继续发现，直到系统收敛。**
