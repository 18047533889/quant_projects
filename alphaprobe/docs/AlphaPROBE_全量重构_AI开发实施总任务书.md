# AlphaPROBE 全量重构：AI 开发实施总任务书

> **用途**：本文件不是研究建议或概念说明，而是可直接交给编码 AI 执行的 AlphaPROBE 全量改造任务书。  
> **目标仓库**：`https://github.com/18047533889/quant_projects`  
> **本次核对基线**：`main@0b7c3726c4da58ca685162423a3443aea64b7808`（2026-09-01 核对到的最新 main 提交）  
> **重要说明**：仓库近期持续重构。编码 AI 在动手前必须完成 Phase 0 的“路径/符号重新定位”，不得因为本文引用了历史文件名就假设最新版目录完全不变。  
> **总原则**：尽量在 AlphaPROBE 自身及 AlphaPROBE 侧适配层完成改造；不要为了方便而修改 `factor_engine/`、`data_access/` 的内部语义。若公共 Evaluator / Refinement / Consolidation 当前还不存在或接口未完全落地，先定义清晰的 client/protocol 与临时兼容 adapter，不要在 AlphaPROBE 内再造一套永久分叉实现。

---



# 0. 本任务最终要把 AlphaPROBE 改成什么

AlphaPROBE 不再只是：

```text
Seed Factor
→ LLM 改公式
→ IC / ICIR
→ 入 Pool
→ 再改
```

而要升级为面向企业级 A 股日频横截面研究的长期自动 Alpha Discovery System：

```text
                             ┌──────────────────────────────┐
                             │ Global Factor Intelligence   │
                             │ Memory                       │
                             │                              │
                             │ Seen / Lineage / Search      │
                             │ Failure / Exploration        │
                             │ Survival / Direction / Regime│
                             └──────────────┬───────────────┘
                                            │
                                            ▼
┌──────────────┐      ┌──────────────────────────────┐
│ Seed Library │─────▶│       AlphaPROBE Search      │
│ 10w+ roots   │      │                              │
└──────────────┘      │ Multi-Arm Search             │
                      │ SearchValue                   │
Other Miners ────────▶│ Multi-Fidelity Evaluation    │
                      │ Active Evolution Pool        │
                      └──────────────┬───────────────┘
                                     │
                                     ▼
                           FactorEngine DSL / AST
                                     │
                                     ▼
                              FactorEngine
                              batch execution
                                     │
                                     ▼
                              Unified Evaluator
                                     │
                ┌────────────────────┴───────────────────┐
                │                                        │
                ▼                                        ▼
           SearchFitness                          Alpha Survival
                │                                   Intelligence
                │
                ▼
          Candidate Archive
                │
                ▼
       Pre-Export Refinement
                │
                ▼
          Re-Evaluation
                │
                ▼
            Export Gate
                │
                ▼
       Global Factor Archive
                │
                ▼
    Factor Consolidation Engine
     clustering / synthesis / model input
```

最终必须具备：

1. **FactorEngine 原生 DSL/AST** 作为公式、语义、执行唯一真值；
2. **DataAccess** 作为数据/PIT/股票池/snapshot/字段能力唯一真值；
3. **统一 Evaluator** 作为 RankIC、分层、多空、稳定性、相关性、成本等事实指标唯一真值；
4. AlphaPROBE 只组合出自己的 **SearchFitness / SearchValue / PoolUtility / ExportScore**；
5. 真正的 **7×24 Round + Campaign** 持续进化；
6. **10 万冷启动 + 其他算法历史因子 + AlphaPROBE 新因子** 统一进入 Global Memory；
7. 跨轮去重、跨算法去重、符号等价、参数族、rank-equivalent 去重；
8. Persistent 多父节点 lineage；
9. 记住“不仅挖出了什么”，还记住“哪些方向已经挖烂、哪些失败、哪些更容易活过 2026 衰减”；
10. 对 2026 年 6–7 月等大规模衰减事件形成 **Alpha Survival Intelligence**，但严格防止 Test 泄漏；
11. 每个 Miner 在正式输出前调用公共 **Factor Refinement Engine**；
12. 聚类/PCA/VIF/簇合成等不塞入 AlphaPROBE，留给独立 **Factor Consolidation Engine**；
13. 能在几十万、百万级历史 factor 下继续运行，而不是 O(N²) 崩溃；
14. 所有重要运行结果、版本、token、成本、数据快照、知识截止时间可复现。

---



# 1. 不可违反的架构边界

这是本重构最重要的部分。编码 AI 不得为了“快速完成”把所有功能重新堆回 `trainer.py`。

## 1.1 FactorEngine 的职责

FactorEngine 是：

```text
Factor Language + Operator Semantics + AST + Validation + Execution Runtime
```

AlphaPROBE 需要调用 FactorEngine 的公开能力完成：

- DSL parse；
- production/research allowlist 检查；
- canonical AST；
- field/operator inspect；
- lookback/complexity/static risk 分析；
- `run_many`；
- `run_many_parallel`；
- Common Subexpression Elimination（CSE，共同子表达式消除）；
- batch source planning；
- auto/hybrid backend；
- Polars/SQL/Pandas fallback 等现有能力。

**不要在 AlphaPROBE 自己重新实现一套 DSL parser、operator semantics 或数据计算引擎。**

## 1.2 DataAccess 的职责

DataAccess 是：

```text
Data + PIT + Snapshot + Universe + Field Capability + Dataset Contract
```

AlphaPROBE 不得直接绕过 DataAccess 去扫描 COS 文件来自己拼研究面板。

必须通过 DataAccess 的公开能力或适配层取得：

- 数据快照 ID；
- 交易日历；
- 历史股票池；
- 字段能力；
- PIT contract；
- 数据 coverage；
- market/universe identity；
- dataset/source version。



## 1.3 Unified Evaluator 的职责

Evaluator 是“测量系统”，不是 AlphaPROBE 内部工具函数集合。

所有算法统一使用同一套事实指标：

```text
RankIC
RankICIR
Rolling IC
Yearly/Subperiod IC
D1-D10
Monotonicity
D10 Cliff
Long / Short / Long-Short
Turnover
Cost
MDD
Drawdown Duration
TUW
Exposure
Coverage
Tradability
Correlation
Decay
OOS
Multiple Testing
...
```

AlphaPROBE 不得永久维护自己的一套：

```python
batch_spearmanr()
batch_sharpe()
batch_max_drawdown()
...
```

如果公共 Evaluator 尚未完全接入，则：

1. 定义 `EvaluatorClientProtocol`；
2. 用临时 adapter 包现有逻辑；
3. 所有上层代码只依赖 protocol；
4. 后续切换真实 Evaluator 时不改搜索内核；
5. 临时 adapter 标记 `legacy_compat`，不得继续扩展成第二套评估库。



## 1.4 Factor Refinement Engine 的职责

Refinement 解决的是：

> 一个已经挖出来的因子本身，有没有可诊断、低自由度、可验证的表达缺陷？能否修复？

例如：

- RankIC 方向；
- U-shape；
- D10 cliff；
- extreme driven；
- numerical fragility；
- coverage；
- selective neutralization；
- high turnover smoothing；
- shape compression；
- model preprocess profile。

AlphaPROBE 只调用：

```text
refinement_client
```

不要把完整 Refinement Engine 再写进 AlphaPROBE。

## 1.5 Factor Consolidation Engine 的职责

Consolidation 解决的是：

> 几千、几万、几十万个好因子之间，如何聚类、去冗余、选代表、合成簇、形成模型输入？

因此以下逻辑不要继续留在 AlphaPROBE：

- VIF；
- PCA；
- SVD 选列；
- clustering；
- cluster composite；
- model-specific factor reduction；
- factor portfolio synthesis。

AlphaPROBE 最多消费 Consolidation 提供的：

```text
cluster_id
cluster_saturation
nearest_cluster_corr
cluster_novelty
```

辅助 SearchValue。

---



# 2. Phase 0：编码前必须完成的最新版仓库侦察



## 2.1 为什么必须做

本文基于：

- 当前 2026-09-01 `main@0b7c3726...`；
- 以及前几轮对 AlphaPROBE 的详细代码审计。

但仓库持续有更新，历史路径可能移动。因此编码 AI **第一步禁止修改代码**，必须先输出一份：

```text
ALPHAPROBE_CURRENT_CODE_MAP.md
```



## 2.2 必须定位的历史对象

在当前 commit 中搜索下列文件/符号的最新位置：

```text
runner.py
trainer.py
pool.py
continuous.py
checkpoint.py
exporter.py

fe_bridge/stock_data.py
fe_bridge/bootstrap.py
fe_bridge/dsl_convert.py

AlphaKnowledgeTrainer
AlphaKnowledgeLogger
AlphaKnowledgePool
ExpressionKnowledgeGraph
ExpressionNode
FactorEngineStockData
enable_factor_engine_evaluation

get_ic
get_ic_icir_mutics
get_mutl_ic
try_new_expr
search

Expression.evaluate
ExpressionParser
shared.alphagen
shared.alphagen_qlib
```



## 2.3 输出映射

格式：

```markdown
| 历史组件 | 当前路径 | 当前符号 | 状态 |
|---|---|---|---|
| trainer.py | xxx | AlphaKnowledgeTrainer | 仍存在 |
| pool.py | xxx | AlphaKnowledgePool | 已重构/仍存在 |
| fe_bridge | xxx | ... | ... |
```



## 2.4 必须确认的现状

编码前确认：

- 当前默认 execution backend；
- 是否仍 monkey-patch `Expression.evaluate`；
- 是否仍有 `eval(expr)`；
- Test 是否仍在 logger/search 中被读取；
- Pool 是否仍固定 capacity / pop lowest IC；
- graph 是否仍单父；
- exporter 是否仍重新回测；
- continuous 是否已有真正 global memory；
- checkpoint 是否按对象值重算；
- embedding 是否每次 search 全池重算；
- formula identity 是否仍字符串；
- 当前 label / split / universe；
- Evaluator / Refinement / Consolidation 是否已经有新公共包可直接调用。

**若现状与本文历史审计不同，按“目标职责”迁移，不机械按旧行号改代码。**

---



# 3. 现有 AlphaPROBE 中必须消除的 P0 架构问题

---



## 3.1 消除“双公式/双执行真值”

历史代码存在：

```text
legacy Expression / ExpressionParser
+
FactorEngine bridge
+
Expression.evaluate monkey patch
```

这会导致：

- 公式 identity 不稳定；
- DSL/AST 和 legacy expression 可能语义不一致；
- 去重困难；
- batch execution 难接；
- cache key 难稳定；
- 未来 operator 语义升级容易静默错算。



### 目标

新候选从生成到执行只有：

```text
LLM/Action Output
→ FactorEngine DSL
→ FE validate
→ FE canonicalize
→ Canonical AST
→ FactorIdentity
→ FE run_many
```



### 新接口

建议：

```python
class FactorEngineAdapter(Protocol):
    def validate(
        self,
        formula: str,
        *,
        surface: str,
        mode: str,
    ) -> ValidationResult:
        ...

    def canonicalize(self, formula: str) -> CanonicalFactor:
        ...

    def inspect(self, canonical: CanonicalFactor) -> FormulaInspection:
        ...

    def run_many(
        self,
        formulas: list[CanonicalFactor],
        context: "ExperimentContext",
    ) -> "FactorBatch":
        ...
```



### Legacy 的处理

旧 `Expression` 只允许：

- checkpoint v1 migration；
- legacy baseline 回归测试；
- 历史结果读取。

新 path 不得反向依赖旧 expression。

---



## 3.2 删除新主链所有 `eval(expr)` / `exec`

历史 `pool.py` 等路径存在公式字符串经 `eval` 的逻辑。

新主路径：

```text
禁止 eval
禁止 exec
禁止 Python expression 动态执行
```

公式只能：

```text
FactorEngine parse
FactorEngine validate
FactorEngine AST
```



### 测试

```python
def test_no_python_eval_in_alphaprobe_main_path():
    ...
```

可做静态 grep gate：

```text
eval(
exec(
```

只允许在明确 `legacy/` 测试 fixture 中存在。

---



## 3.3 完全封存 Test

这是最优先修复的研究协议问题。

历史代码存在：

- `runner` 创建 test data 后传入 trainer/logger；
- logger 每轮/每个 log frequency 计算 `test_ic/test_icir`；
- `node.test_ic/test_icir` 写回图；
- `load_test_res()` 再跑一套 test；
- logger 内历史 test label/date 口径还可能和主 runner 不一致。

这会造成：

1. Test 间接参与人类/LLM 调参；
2. Test 指标污染 Search Memory；
3. 长期 7×24 后 test 实际变成 validation；
4. 计算浪费。



### 新研究协议

必须支持：

```text
Train
Search Validation
Audit Validation
Sealed Test
```



### 语义

**Train**

- sign orientation；
- repair 参数拟合；
- 搜索 action 局部反馈；
- 参数 family 比较。

**Search Validation**

- 高频 SearchFitness；
- parent selection；
- promotion。

**Audit Validation**

- L3/L4 少量 finalist；
- 检查长期搜索是否过拟合 Search Validation；
- 不完整暴露给 LLM；
- 不允许反复人工调到它。

**Sealed Test**

- 版本/campaign 冻结后一次；
- 不进入当代搜索；
- 只进入正式 research report；
- 以后若成为历史知识，必须跨“研究版本”使用，见 Survival Memory 章节。



### 新组件

```text
research_protocol/
├── split_spec.py
├── leakage_guard.py
├── audit_validation.py
└── sealed_test.py
```



### 必须实现

```python
@dataclass(frozen=True)
class ResearchSplitSpec:
    train: DateRange
    search_valid: DateRange
    audit_valid: DateRange | None
    sealed_test: DateRange
    purge_trading_days: int
    embargo_trading_days: int
```

20 日 forward VWAP label 必须至少处理 label overlap：

```text
purge >= 20 trading days
```

embargo 具体值配置化，首版可用 5 trading days 作为额外缓冲，不得写死为不可配置。

### LeakageGuard

所有 datasource access 注入：

```python
class LeakageGuard:
    def assert_access_allowed(
        self,
        *,
        stage: FidelityLevel,
        segment: Segment,
        caller: str,
    ) -> None:
        ...
```

L0-L4：

```text
sealed_test read count == 0
```

必须成为 CI。

---



# 4. A 股数据语义硬规则

编码 AI 必须以 DataAccess / 当前 A 股数据契约为准，不能凭一般金融知识自己拼。

已知关键规则：

## 4.1 Return 单位

A 股日线：

```text
Return = bp
decimal_return = Return / 10000
```

不能把 `-550` 当 `-550%`。

## 4.2 财务 PIT

财务表：

```text
StockBalance
StockIncome
StockCashFlow
StockIndicator
```

是事件表，真正可知时间是：

```text
PubDate
```

不是：

```text
ReportPeriodEndDate
```

必须做 backward as-of：

```text
PubDate <= decision timestamp
```



## 4.3 行业表

`StockIndustry` 一只股票同一天有多个 `IndustrySource`。

中性化/行业分析前必须指定一个体系。

默认建议：

```text
sw_l1
```

但仍应从 config 读取。

否则 join 可能约 6 倍膨胀。

## 4.4 财务 NaN

财务 NaN 表示：

```text
未披露 / 不适用 / 会计科目差异
```

禁止默认：

```python
fillna(0)
```



## 4.5 UpdateTime

`UpdateTime` 是数据同步/写入 freshness：

```text
不是行情时间
不是公告 PIT
不是 decision timestamp
```



## 4.6 Tradability / Universe

参考基础条件：

```text
PublicStatus == "正常上市"
AND IsSuspend == False
AND Volume > 0
```

但最终股票池必须由 `UniverseSpec` / DataAccess 决定，不要 AlphaPROBE 自己散落 hardcode。

## 4.7 当前研究 label

历史确认主任务为：

```text
20 trading-day VWAP → VWAP forward return
```

所有：

- cohort；
- purge；
- embargo；
- holding horizon；
- turnover/cost；
- train/valid/test；

必须共享同一个 LabelSpec。

---



# 5. 建立 ExperimentContext：消灭散落参数

建议所有关键模块只接收统一 `ExperimentContext`。

```python
@dataclass(frozen=True)
class ExperimentContext:
    run_id: str
    round_id: str
    campaign_id: str

    market: str
    data_snapshot_id: str
    universe_snapshot_id: str

    label_spec: LabelSpec
    split_spec: ResearchSplitSpec

    factor_engine_version: str
    operator_semantics_version: str
    data_access_version: str
    evaluator_version: str

    search_config_hash: str
    prompt_version: str
    llm_model_policy_version: str

    knowledge_cutoff_date: date
    survival_memory_cutoff: date

    rng_seed: int
```

所有 cache / artifact / memory / export 必须可追溯到它。

---



# 6. FactorEngine 批量执行：从单因子循环改成 Batch-First

历史 `FactorEngineStockData` 的问题：

- dense Torch `[date,feature,stock]`；
- 可能写死 pandas；
- 每个 factor 调 `engine.run()`；
- 不能充分复用 CSE；
- 重复 IO；
- 难配 Evaluator Artifact DAG。



## 6.1 新主链

```text
Candidates
→ canonicalize/dedup
→ batch by data dependency / fidelity
→ FE run_many
→ FactorBatch
→ Evaluator
```



## 6.2 FactorBatch

建议：

```python
@dataclass
class FactorBatch:
    factor_ids: list[str]
    values_ref: ArtifactRef
    trade_dates: DateRange
    universe_snapshot_id: str
    data_snapshot_id: str
    coverage_summary: dict[str, float]
```

不要默认把所有大矩阵永久驻内存。

优先：

```text
Arrow / Parquet / memory map / artifact reference
```



## 6.3 批处理策略

按：

- field dependency；
- common lookback；
- source plan；
- fidelity；
- date segment；

分 batch。

首版：

```text
batch_size: 32~64
```

后续 benchmark 调。

---



# 7. 消除候选因子的重复计算

历史常见：

```text
get_ic_icir_mutics()
→ 已执行因子
→ try_new_expr()
→ 再 evaluate
→ exporter
→ 再 train/valid/test evaluate
```

必须重构成：

```text
FactorCandidate
      ↓
Evaluator.evaluate_once()
      ↓
EvaluationRecord / ArtifactRef
      ├─ SearchFitness
      ├─ Pool admission
      ├─ logger
      ├─ checkpoint
      └─ exporter
```

核心规则：

```text
Candidate Once Computed, Many Consumers
```

Pool 不得自己重新运行 FactorEngine。

Exporter 不得重新跑全回测。

Logger 不得重新计算指标。

---



# 8. Unified Evaluator 接入协议

建议：

```python
class EvaluatorClient(Protocol):
    def evaluate(
        self,
        candidates: list["FactorCandidate"],
        *,
        profile: str,
        fidelity: "FidelityLevel",
        context: "ExperimentContext",
    ) -> list["EvaluationRecord"]:
        ...

    def get_artifact(
        self,
        artifact_id: str,
    ) -> ArtifactRef:
        ...
```

`EvaluationRecord`：

```python
@dataclass(frozen=True)
class EvaluationRecord:
    factor_id: str
    segment: str
    fidelity: str

    metric_bundle: dict[str, float | None]
    artifact_refs: dict[str, str]

    evaluator_version: str
    data_snapshot_id: str
    universe_snapshot_id: str
    label_spec_hash: str

    created_at: datetime
```

AlphaPROBE 内部不得偷偷定义同名不同义 RankIC/Sharpe。

---



# 9. 多保真评估漏斗

长期 7×24 不可能所有 candidate 都跑完整报告。

必须分级。

## L0 — Static

不跑市场回测。

检查：

- FE DSL；
- research/production allowlist；
- PIT field legality；
- AST；
- exact duplicate；
- sign equivalence；
- parameter family；
- complexity；
- lookback；
- static numerical risk；
- obvious divide-by-zero structure；
- unsupported fields；
- approximate compute cost。

拒绝：

```text
INVALID_DSL
PIT_VIOLATION
UNSUPPORTED_FIELD
EXACT_DUPLICATE
SIGN_EQUIVALENT_DUPLICATE
UNSAFE_NUMERIC
```



## L1 — Scout

使用完整 universe + 少量固定日期 block。

计算：

- cheap RankIC；
- coverage；
- NaN/Inf；
- runtime；
- rank fingerprint；
- approximate nearest corr；
- rough dispersion。

目的：

```text
大量淘汰明显垃圾
```



## L2 — Full Train

计算：

- Train RankIC；
- RankICIR；
- D1-D10；
- monotonicity；
- D10 cliff；
- cohort LS；
- turnover；
- base cost；
- coarse stability；
- fragility。



## L3 — Search Validation

计算：

- Search Valid RankIC；
- MDD；
- DD Duration；
- TUW；
- rolling Sharpe；
- rolling RankIC；
- subperiod；
- exposure；
- tradability；
- optional horizon；
- cost stress。



## L4 — Pool Utility / Audit

只给少量 finalist：

- Audit Validation；
- residual RankIC；
- exact top-neighbor correlation；
- low-rank pool basis；
- cross-fitted incremental utility；
- multiple-testing diagnostics；
- model incremental proxy（若需要）。



## L5 — Sealed Test

只在：

```text
search/campaign/version 冻结
```

后运行。

---



# 10. Train-only 因子方向标准化

这是新的硬规则。

Train：

```text
RankIC_train(f) >= 0
→ canonical oriented factor = f

RankIC_train(f) < 0
→ canonical oriented factor = -f
```

Valid/Test：

```text
禁止重新翻方向
```

如果 Valid/Test RankIC 变负：

```text
这是 instability
不是重新乘 -1
```



## 10.1 Identity

以下必须同一 `SignalEquivalenceID`：

```text
f
-f
0-f
(-1)*f
-(-(-f))  （归一符号后）
```

最终 Active Pool / Export 只保留 Train RankIC 正向版本。

## 10.2 解决历史 `abs(IC)` 语义混乱

禁止：

```text
一处 abs(IC)
另一处 abs(ICIR)
graph 存 abs
pool 存 raw
```

新系统统一：

```text
raw metric
+
orientation
+
oriented metric
```

并显式字段区分。

---



# 11. Factor Identity 与四层去重

长期挖矿的第一性问题不是再找多少公式，而是防止重复计算已有信息。

---



## 11.1 Level 1：Canonical AST

安全化简：

```text
(x)            -> x
((x))          -> x
+x             -> x
x * 1          -> x
1 * x          -> x
x / 1          -> x
x + 0          -> x
0 + x          -> x
x - 0          -> x
0 - x          -> -x
x * (-1)       -> -x
(-1) * x       -> -x
-(-x)          -> x
pow(x, 1)      -> x
constant fold
```

对于明确满足交换律的 operator：

```text
canonical operand order
```

**禁止危险化简：**

```text
x/x -> 1
```

因为：

- x=0；
- NaN；
- Inf；
- mask；

下语义可能不同。

---



## 11.2 Level 2：Sign-invariant Signal ID

```python
SignalEquivalenceID = hash(
    sign_normalized_canonical_ast
)
```

---



## 11.3 Level 3：Parameter Family

例如：

```text
ts_mean(close,19)
ts_mean(close,20)
ts_mean(close,21)
```

属于同一：

```text
ParameterFamily
```

不要当成三个完全独立 alpha 成果。

每 family 保留：

- representative；
- best parameter；
- explored parameter ranges；
- saturation。

---



## 11.4 Level 4：Rank-equivalent Fingerprint

为固定的、跨年份 sample dates 生成：

```text
cross-sectional rank signature
```

量化后形成：

```text
256-bit SimHash / LSH fingerprint
```

10 万 factor 仅保存 256 bit：

约：

```text
100,000 × 32 bytes ≈ 3.2 MB
```

可快速召回近似排序信号。

主要识别：

```text
f
2f
100f + 5
monotonic transform(f)
rank(f)
```

等对横截面排序几乎相同的信号。

---



## 11.5 Level 5：Nearest Full Confirmation

只对 fingerprint / ANN 找出的 Top-K 邻居运行：

```text
value correlation
rank correlation
```

不要候选和 10 万/100 万全库做全相关矩阵。

---



## 11.6 Atomic Reservation

多 worker 同时生成同一 factor 时：

```text
canonicalize
→ SignalID
→ DB INSERT UNIQUE / reservation
```

状态：

```text
NEW
→ RESERVED
→ EVALUATING
→ EVALUATED
```

其他 worker 如果 insert 冲突：

```text
立即停止昂贵回测
```

---



# 12. Evaluation Cache

Cache key 至少包含：

```text
canonical_formula_hash
orientation
data_snapshot_id
universe_snapshot_id
FactorEngine version
operator semantics version
label spec hash
segment
fidelity
Evaluator version
preprocess profile version
```

不要只按公式 hash。

以下变化必须 cache miss：

- 数据快照；
- operator semantics；
- PIT contract；
- universe；
- label；
- evaluator；
- preprocessing；
- neutralization semantics。

---



# 13. 最终 SearchFitness 公式

所有原始指标先统一 utility 化，不能直接：

```text
RankIC + Sharpe - MDD
```

因为单位不同。

---



## 13.1 MetricCalibrator

默认：


z_m=
\frac{x_m-\operatorname{Median}(m)}
{1.4826 \cdot MAD(m)+\epsilon}


然后：


U(m)=\sigma(z_m/T_m)


或者用 robust quantile 映射。

越小越好：

```text
MDD
DD Duration
TUW
Cost
Correlation
Turnover
```

转换为：

```text
U(-metric)
```



### Round Freeze

每个 major round 开始：

```text
freeze calibrator
```

本轮相同原始 metric 必须映射成相同 utility。

Round 结束才更新：

- median；
- MAD；
- quantile sketch。

大规模下不要把全历史 metric 放内存，使用 streaming quantile sketch / robust summary。

---



## 13.2 总公式

# 
\boxed{
F_{search}

## 0.26P
+
0.20Q
+
0.25L
+
0.11S
+
0.18N

## P_{cost}

## P_{complexity}

P_{fragility}
}


---



# 14. Predictive：P


\boxed{
P=
0.70U(RankIC_{valid})
+
0.30U(MedianSubperiodRankIC)
}


说明：

- 主预测力仍以 Search Validation RankIC 为主；
- 子时期中位数抑制某一段偶然很好；
- RankICIR 不重复塞 P，主要归 Stability。

---



# 15. Stratification：Q

每天按 oriented factor：

```text
low → high
D1 ... D10
```

十组收益：


G_1,\ldots,G_{10}


---



## 15.1 Group Monotonicity

# 
M_{rank}

Spearman(
[1,\ldots,10],
[G_1,\ldots,G_{10}]
)


---



## 15.2 Isotonic Fit Quality

拟合单调递增 isotonic：


\hat G_{iso}


定义：

# 
M_{iso}

1-
\frac{SSE(G,\hat G_{iso})}
{SST(G)+\epsilon}


这里只用于**评价 shape**。

不要在 AlphaPROBE 里面自动修改 factor。

真正 shape repair 交给 Refinement。

---



## 15.3 D10 自身强度

# 
Top10Excess

## G_{10}

\frac{1}{10}\sum_{i=1}^{10}G_i


还必须记录：

```text
D10 absolute
D10 - universe
D10 - market/benchmark
D10 - D1
```

原因：

> D10-D1 很强可能仅仅因为 D1 非常差，D10 本身并不赚钱。

---



## 15.4 D10 Cliff

允许：

```text
D10 比 D7/D8/D9 稍微低
```

不能机械要求：

```text
D10 永远最高
```

定义：


d_{top}=
\frac{
Median[
(G_7-G_{10})*+,
(G_8-G*{10})*+,
(G_9-G*{10})*+
]
}{
|G*{10}-G_1|+\epsilon
}


允许 tolerance：


\tau \approx 0.10\sim0.15


# 
CollapsePenalty

\max(0,d_{top}-\tau)



TopTailQuality=U(-CollapsePenalty)


---



## 15.5 Q 最终


\boxed{
Q=
0.30U(M_{rank})
+
0.25U(M_{iso})
+
0.20U(Top10Excess)
+
0.25U(TopTailQuality)
}


另外保存：

```text
EffectiveSignalWidth
EffectiveBreadth
Top contribution concentration
```

作为 metadata / fragility / tradability。

---



# 16. Long-Short：L

当前 label 是：

```text
20d VWAP → VWAP
```

因此**禁止**把每天高度重叠的 20d forward D10-D1 序列直接当日频 PnL 年化 Sharpe。

---



## 16.1 20 日 Cohort Portfolio

每天：

1. 按 factor 分 D1-D10；
2. Long D10；
3. Short D1；
4. 新建一个持有 20 trading days 的 cohort；
5. 约投入 `1/20` 总资本；
6. 每天叠加当前还活着的 20 个 cohort；
7. 得到真实日频 cohort PnL。

必须严格遵守 decision/execution time contract。

---



## 16.2 A 股 Short 的语义

D1 short：

```text
主要作为 factor diagnostic portfolio
```

不等于实盘可完全融券执行。

因此同时保存：

- D10 long-only；
- D10-universe；
- D10-market；
- D10-D1 diagnostic。

---



## 16.3 低交易成本

净收益：

# 
r_t^{net}

## r_t^{gross}

Turnover_t \cdot c_{base}


`c_base` 配置化。

保存：

```text
1x
2x
3x
```

cost stress。

2x/3x 不进入主 SearchFitness。

---



## 16.4 L 最终


\boxed{
L=
0.30U(NetSharpe)
+
0.10U(Sortino)
+
0.15U(Calmar)
+
0.10U(-MDD)
+
0.08U(-MaxDDDuration)
+
0.07U(-TUW)
+
0.15U(Q20RollingSharpe)
+
0.05U(PositiveMonthRatio)
}


其中：

```text
TUW = Time Under Water
```

低分位 rolling Sharpe 是为了鼓励：

> 多空曲线持续向上，而不是靠少数幸运阶段。

---



# 17. Stability：S


\boxed{
S=
0.35U(RankICIR)
+
0.25U(Q20RollingRankIC)
+
0.15U(PositiveSubperiodRatio)
+
0.15U(TrainValidRetention)
+
0.10U(WorstSubperiodRankIC)
}


其中：

# 
TrainValidRetention

clip\left(
\frac{RankIC_{valid}}
{RankIC_{train}+\epsilon}
\right)


clip 范围配置化，防 denominator 太小爆炸。

---



## 17.1 20 日重叠 label 的 RankICIR 修正

Raw daily RankICIR 可能因 20 日 overlap 被自相关虚高。

因此 L3/Audit 增加：

- HAC/Newey-West adjusted RankICIR；
- 或 non-overlapping 20d IC；
- 或 block/bootstrap confidence。

便宜 raw RankICIR 可继续用于 L2。

---



# 18. Novelty：N

保留并升级 AlphaPROBE 原本“相关性 + offspring sparsity”优势。


\boxed{
N=
0.45N_{corr}
+
0.20N_{structural}
+
0.20U(ResidualRankIC)
+
0.15N_{schema}
}


---



## 18.1 Correlation Novelty

不要再用：

```text
1 - mean(corr(candidate, all pool))
```

因为一个 factor 和某个已有因子 0.999 重复，但和剩下 10 万个都低相关，平均相关仍很低。

改为：

```text
rho_max
mean_top_5_abs_corr
mean_top_20_abs_corr
```

同时计算：

```text
Pearson/value corr
Spearman/rank corr
```

横截面 alpha 对 rank corr 尤其重要。

---



## 18.2 Structural Novelty

看：

- FE canonical AST；
- operator multiset；
- field set；
- subtrees/motifs；
- ParameterFamily；
- formula roles；
- schema。

---



## 18.3 Residual RankIC

L4 才做。

候选：

```text
f
```

对已有低维 pool basis / nearest factors residualize 后：

```text
RankIC(residual)
```

用于判断真正新增信息。

这个计算可以来自 Evaluator/Consolidation 支持，不要给所有 L1/L2 candidate 跑。

---



# 19. Penalties

---



## 19.1 CostPenalty

成本已经在 LongShort net PnL 中体现。

额外只轻罚：

```text
P_cost <= 0.02
```

不能让低券商费率环境下成本权重压死好 alpha。

但必须额外看：

- turnover；
- impact proxy；
- liquidity；
- limit-up/down；
- suspension；
- tradability。

---



## 19.2 ComplexityPenalty

不是简单“越复杂越坏”。

首版：

```text
AST nodes 0-24    ≈ no penalty
25-40             small
41-64             medium
>64               optional hard reject
depth             up to 10-12
```

进化时记录：

# 
ComplexityEfficiency

\frac{
\Delta SearchFitness
}{
1+\Delta ASTNodes
}


复杂公式如果显著提高稳定性/新颖性可以保留。

---



## 19.3 FragilityPenalty

惩罚：

- mild winsor 后 IC 崩塌；
- extreme contribution concentration；
- numerical instability；
- near-zero denominator；
- low coverage；
- coverage instability；
- 单一年份独撑；
- 极少股票驱动；
- 极端 tail dependence。

建议：

```text
P_fragility cap ≈ 0.03~0.05
```

---



# 20. Hard Gates：不能靠高 Fitness 救回

至少：

```text
DSL invalid
PIT violation
forbidden field
insufficient coverage
NaN/Inf catastrophic
exact/sign duplicate
seed library duplicate at export
already exported
sealed-test leakage
untradeable catastrophic
numerical invalid
```

Hard Gate 与 SearchFitness 分开。

---



# 21. 不要把所有东西都叫 Fitness

必须拆生命周期评分。

---



## 21.1 SearchFitness

回答：

> 这个 candidate 现在表现怎么样？

就是前面的五维公式。

---



## 21.2 AssetQualityScore

回答：

> 把它当长期 alpha 资产，本身质量如何？

更偏：

- stability；
- survival；
- tradability；
- robustness；
- OOS。

---



## 21.3 SearchValue

回答：

> 从这个节点继续花 token/算力挖，值得吗？

建议：

# 
SearchValue

w_1 Fertility
+w_2 OffspringNovelty
+w_3 DescendantPoolGain
+w_4 CoverageGap
+w_5 Uncertainty
+w_6 Frontierness
+w_7 SurvivalOpportunity
-w_8 SearchSaturation
-w_9 HistoricalComputeCost


所有项先 normalization。

---



## 21.4 PoolUtility

回答：

> 它是否值得占有限 Active Pool 的一个 slot？

包含：

- Pareto quality；
- niche rarity；
- novelty；
- SearchValue；
- representative value。

---



## 21.5 ExportScore

回答：

> 能否作为正式新增因子输出？

比 SearchFitness 更严格。

---



## 21.6 ProductionHealth

正式上线后的实时健康指标。

不属于搜索本身，但统一 Evaluator 应支持。

---



# 22. 保留 AlphaPROBE 原生优点，但全部升级

历史 `pool.search()` 已经有值得保留的思想：

```text
sigmoid(ICIR zscore)
× depth decay
× retrieval/times decay
× diversity
```

leaf：

- factor value corr；
- semantic similarity；
- edit distance。

non-leaf：

- child ICIR improvement；
- parent-child corr；
- sibling corr。

这些不要删。

升级为：

```text
ICIR quality
→ SearchFitness quality

child ICIR gain
→ child SearchFitness gain

mean pool correlation
→ top-K nearest value + rank correlation

sibling sparsity
→ offspring novelty

node quality
→ SearchValue
```

保留：

- depth decay；
- times/retrieval decay；

避免一个老节点永远霸占预算。

---



# 23. Fertility：让“会生好后代”的因子被记住

Factor 自己质量和作为 parent 的价值不同。

定义历史 parent：

```text
Attempts
SuccessfulOffspring
EliteOffspring
NovelOffspring
DescendantPoolGain
```

可定义：

```text
FertilityScore
```

使用 empirical Bayes shrinkage，避免：

```text
只试 1 次成功 1 次
→ 100% fertility
```

被高估。

---



# 24. ExplorationState：记住某个因子的哪些方向已经挖过

每个：

```text
(factor_id, action_family)
```

保存：

```text
attempt_count
success_count
elite_count
mean_delta_fitness
mean_novelty_gain
mean_compute_cost
last_attempt
saturation_score
version_key
```

例如：

```text
Factor F123

WINDOW_SCALE       95% explored
REFINE             78%
FIELD_SUBSTITUTION 63%
STATE_CONDITION    22%
CROSSOVER          17%
SCHEMA_EXPLORE      5%
```

下一轮再抽到 F123：

> 不从 95% 饱和的窗口继续机械搜索。

---



# 25. Search Arms

至少建立 5 个正式 arm。

---



## 25.1 RefineArm

小步修改现有信号：

- operator substitute；
- field substitute；
- window；
- normalization；
- context；
- quality filter。

---



## 25.2 BranchArm

Tree-of-Thought 风格：

```text
一个 parent
→ 多个机制假设
→ 多个 branch
```

先便宜 scout，再只深挖最优 branch。

---



## 25.3 EvolutionArm

支持：

- mutation；
- role-aware crossover。

Crossover 不做字符串拼接。

---



## 25.4 SchemaExploreArm

通过语义方向主动寻找“还没有被挖烂”的机制。

Schema 维度：

```text
Event
Context
Qualities
Direction
Output
DataDomain
Horizon
Normalization
Tradability
```

---



## 25.5 GenerationRepairArm

只修：

- parser；
- type；
- field；
- DSL syntax；
- unsupported operator。

注意：

```text
GenerationRepairArm
!=
Factor Refinement Engine
```

前者只把“无效生成”改成合法公式。

后者是对已经有效因子的经济/统计形状做后处理。

---



# 26. 标准 SearchAction

```python
class SearchActionType(str, Enum):
    REFINE = "REFINE"
    WINDOW_SCALE = "WINDOW_SCALE"
    FIELD_SUBSTITUTION = "FIELD_SUBSTITUTION"
    OPERATOR_SUBSTITUTION = "OPERATOR_SUBSTITUTION"
    STATE_CONDITION = "STATE_CONDITION"
    CROSSOVER = "CROSSOVER"
    SCHEMA_EXPLORE = "SCHEMA_EXPLORE"
    ROBUSTIFY = "ROBUSTIFY"
    GENERATION_REPAIR = "GENERATION_REPAIR"
```

---



# 27. Role-aware Formula Representation

给公式 AST 标角色：

```text
signal_core
context_condition
quality_filter
normalization
direction
output_transform
```

Crossover 示例：

```text
A.signal_core
+
B.context_condition
```

而不是：

```python
f"({formula_a}) + ({formula_b})"
```

---



# 28. 把历史 `%d = 5/10/20/30` 暗搜索改成显式 Action

历史模板参数逻辑如果还存在：

```text
%d
→ 5 / 10 / 20 / 30
→ Train IC 最大者
```

必须改为：

```text
WINDOW_SCALE Action
```

并：

- 所有窗口 variant 归 ParameterFamily；
- attempt ledger 记录；
- 预算可控；
- 不把每个窗口当全新独立 alpha；
- 不无限网格搜索。

---



# 29. ActionScheduler：让搜索策略自己学习

实现：

```text
UCB / Thompson Sampling
```

调度：

```text
parent × action
```

Reward：

# 
Reward

## \Delta PoolUtility
+
ValidQualityGain
+
NoveltyGain
+
SchemaCoverageGain
+
SurvivalGain

## EvaluationCost

## LLMCost

FailurePenalty


支持 delayed trajectory credit：

某个 action 当代不是 elite，但后代最终形成优秀 factor，则给祖先一定 credit。

---



# 30. LLM 模型路由

不要所有 action 都用最强模型。

建议：

**Cheap model**

- routine refine；
- syntax repair；
- field substitute；
- simple mutation；
- schema proposal。

**Strong model**

- complex crossover；
- stagnation escape；
- multiple-defect reasoning；
- hypothesis/formula alignment dispute；
- rare mechanism synthesis。

记录真实：

```text
prompt tokens
completion tokens
latency
model
estimated cost
action
factor_id
```

---



# 31. LLM 输出必须结构化

优先：

```text
Pydantic / JSON schema constrained output
```

例如：

```python
class GeneratedCandidate(BaseModel):
    formula: str
    explanation: str
    hypothesis: str
    action_type: SearchActionType
    parent_ids: list[str]
    schema_tags: dict[str, str]
```

不要依赖：

```text
regex 从大段自然语言硬抓 JSON
```

regex fallback 只做兼容并打 warning。

---



# 32. Prompt 不再塞完整 lineage

历史 `path_to_root()` / traces 在 7 层尚可。

未来：

```text
30~80 generations
```

会直接爆 token。

替换：

```text
Full Global Memory
→ Retriever
→ MemoryPacket
→ LLM
```

每次 2k–4k token 左右。

---



# 33. Global Factor Intelligence Memory 总架构

这是本次重构的核心。

不要把“记忆”理解成：

```text
更多 prompt 历史文本
```

应该是结构化机器记忆。

---



## 33.1 Memory 子系统

至少：

1. Factor Registry；
2. Factor Alias/Identity；
3. Global Seen/Dedup；
4. Persistent Lineage；
5. Search Memory；
6. Failure/Saturation Memory；
7. Exploration State；
8. Factor DNA；
9. Direction/Motif/Cluster Memory；
10. Alpha Survival Memory；
11. Market Regime/Event Memory；
12. Source Ingestion Registry；
13. Export Registry；
14. System Version Registry。

---



# 34. Global Factor Registry

所有来源统一成唯一 FactorNode。

建议字段：

```text
factor_id
canonical_formula
canonical_ast_hash
signal_equivalence_id
factor_family_id

source_system
source_external_id
source_snapshot
source_type

orientation

field_set
operator_set
schema
mechanism
horizon
complexity
lookback

first_seen_at
last_seen_at
times_seen

is_seed
is_exportable
is_exported
status
```

---



# 35. 10 万冷启动怎么进 Memory

用户会一次性提供约 10 万公式，**没有迭代过程**。

正确做法：

每个 seed 是：

```text
root FactorNode
```

字段：

```text
source = seed_library
parents = []
lineage_root = true
exportable = false
```

不要伪造 lineage。

初始化时批量：

```text
formula
→ FE parse
→ canonical AST
→ sign-independent identity
→ parameter family
→ fields/operators
→ schema/DNA
→ complexity/lookback
→ rank fingerprint（若历史值可算）
→ GlobalSeenIndex
```

以后 AlphaPROBE 从 seed 开始：

```text
Seed F123
→ Action A
→ Child F100001
```

真实 lineage 从这里产生。

---



# 36. 其他算法 / COS 中已有因子的接入

未来可能有：

- CogAlpha；
- FactorMiner；
- EvoAlpha；
- AlphaCFG；
- 其他算法；
- 人工因子；
- COS 中历史 metadata；
- 只有公式；
- 公式+metrics；
- 公式+lineage。

因此建立：

```python
class FactorSourceAdapter(Protocol):
    def iter_factors(
        self,
        snapshot: str,
    ) -> Iterator[ExternalFactorRecord]:
        ...
```

支持：

```text
FORMULA_ONLY
FORMULA_WITH_METRICS
FORMULA_WITH_LINEAGE
ARTIFACT_REFERENCE
```

Idempotent key：

```text
(source_system, source_snapshot, external_factor_id)
```

如果不同来源归一后是同一 signal：

```text
同一个 FactorNode
+
多个 source aliases
+
rediscovery event
```

不要创建多个节点。

---



# 37. Persistent Multi-Parent Lineage

历史 `ExpressionNode.parent: Optional[...]` 必须升级。

FactorNode 唯一。

ActionEdge：

```python
@dataclass
class ActionEdge:
    action_id: str
    action_type: str

    parent_factor_ids: list[str]
    child_factor_id: str

    round_id: str
    campaign_id: str
    generation: int

    prompt_version: str
    llm_model: str | None

    cost: float
    latency_ms: int

    outcome: str
```

支持：

```text
A + B -> C
```

跨轮再发现 B：

```text
仍然连接到同一个 B 节点
```

不因 round 不同复制 factor。

---



# 38. Global Memory 存储方案

前期不需要为了 DAG 上 Neo4j。

建议：

```text
DuckDB / SQLite：事务、索引、状态
Partitioned Parquet / Arrow：大体量 evaluation/history
ANN / LSH：embedding/fingerprint nearest retrieval
COS：持久对象、快照、报告、外部 metadata
```

Active Subgraph 才进 RAM。

---



# 39. 建议 Global Memory 表

```text
factor_nodes
factor_aliases

factor_families
factor_family_members

action_edges
attempts
evaluations

rediscovery_events

pool_snapshots

factor_dna

factor_period_performance
factor_survival

factor_embeddings

direction_clusters
direction_cluster_versions
factor_direction_membership

market_regime_events
regime_factor_response

exploration_state
search_action_stats
failure_memory

source_ingestions

export_registry

system_versions
```

---



# 40. Machine Memory 与 LLM Working Memory 分离

**Machine Memory**：

保存完整：

- formula；
- AST；
- metrics；
- fingerprints；
- evaluations；
- actions；
- failure；
- lineage；
- survival；
- cluster；
- cost。

LLM 不直接读取。

**LLM Working Memory**：

Retriever 只给当前必要信息。

---



# 41. MemoryPacket

建议：

```python
class MemoryPacket(BaseModel):
    parent: FactorSummary

    structural_neighbors: list[FactorSummary]
    numerical_neighbors: list[FactorSummary]

    successful_offspring: list[AttemptSummary]
    representative_failures: list[AttemptSummary]

    ancestry_summary: str

    unexplored_actions: list[str]
    saturated_actions: list[str]

    rare_directions: list[DirectionSummary]
    survival_exemplars: list[FactorSummary]

    allowed_fields: list[str]
    allowed_operators: list[str]

    cluster_context: ClusterContext | None
```

典型数量：

```text
parent                         1
structural nearest            3~5
rank/value nearest            3~5
successful offspring          3
representative failures       3
rare directions               3~5
```

总 token：

```text
约 2k~4k
```

factor 总库从 10 万到 1000 万，prompt 大小基本不增长。

---



# 42. Failure Memory：负知识必须保留

不要只记成功。

例如：

```text
F123 × WINDOW_SCALE
124 次
成功率 0.8%
```

和：

```text
F123 × STATE_CONDITION
17 次
成功率 19%
```

下一轮应明显降低前者预算。

---



## 42.1 Failure 的寿命

**结构性失败**

例如：

- DSL 永远非法；
- field semantics 不兼容；
- exact duplicate；

可以长期保存，但仍带 operator/version key。

**统计失败**

例如：

- 2020-2025 当前样本表现差；

应该 TTL / decay。

**Search Saturation**

和：

```text
DataSnapshot
OperatorSetVersion
SearchGrammarVersion
```

绑定。

FactorEngine 新增 operator 后可以重新打开过去饱和 parent 的新方向。

---



# 43. Alpha Survival Memory：专门学习“哪些因子更不容易死”

用户特别关注：

```text
2026 年 6–7 月
大规模因子衰减
```

这必须成为 Global Memory 的正式一部分。

但是必须严格处理 leakage。

---



# 44. 绝不能拿 2026 Test 直接指导同一版搜索

假设版本：

```text
AlphaPROBE_vN
```

Test 包含：

```text
2026-06
2026-07
```

那么：

```text
vN 搜索过程中
禁止使用 2026 survival 信息
```

只有 vN：

```text
完全冻结
→ Sealed Test 完成
```

后，2026 表现才可以作为**历史已知市场结果**进入：

```text
Survival Memory for vN+1
```

而 vN+1 必须重新定义未来 holdout。

---



## 44.1 加知识截止时间

每次 run：

```text
knowledge_cutoff_date
survival_memory_cutoff
```

Retriever 不允许读取：

```text
event_date > cutoff
```

这条必须由代码防，不靠 prompt 自觉。

---



# 45. Factor Survival Profile

每个 factor 保存跨时期：

```text
monthly RankIC
quarterly RankIC
rolling RankIC

RankICIR

D1-D10
D10 cliff

cohort LS Sharpe
rolling Sharpe q20

MDD
DD Duration
TUW

turnover
coverage
breadth
cost sensitivity
exposure
```

---



## 45.1 Survival Labels

建议：

```text
PERSISTENT_ALPHA
HEALTHY
DEGRADING
BROKEN
RECOVERED
REGIME_DEPENDENT
UNCLASSIFIED
```

---



## 45.2 Survival Metrics

至少：

```text
Train/Valid -> Recent Retention
rolling performance slope
max deterioration
sign flip frequency
breach duration
recovery time
positive month/subperiod ratio
low-quantile performance
worst subperiod
drawdown in alpha quality
```

---



## 45.3 Half-life

可以保存描述性：


IC_t \approx IC_0e^{-\lambda t}



HalfLife=\ln(2)/\lambda


但：

**不要强迫所有 factor 用指数衰减模型。**

大量 alpha：

- regime switching；
- broken then recovered；
- nonlinear；
- sudden cliff；

并不适合 exponential decay。

因此 half-life 只是一个辅助特征。

---



# 46. Factor DNA：统计“什么样的因子更容易失效”

不能只统计：

```text
含 ts_rank 的因子失效率
```

因为 operator 与字段、horizon、source、complexity、crowding 强烈混杂。

给每个 factor 建 DNA。

---



## 46.1 Data DNA

```text
price
volume
liquidity
volatility
valuation
fundamental
earnings
balance_sheet
cashflow
shareholder
industry
index
intraday-derived
event
...
```

允许多标签。

---



## 46.2 Operator DNA

记录：

```text
operator multiset
operator depth
operator role
subtree motifs
```

---



## 46.3 Mechanism / Schema DNA

例如：

```text
momentum
reversal
liquidity
volatility
information_arrival
market_impact
behavioral
quality
growth
valuation
risk_premium
fundamental_surprise
state_conditioned
cross-domain
```

但不能完全依赖人工标签。

后面要自动发现方向。

---



## 46.4 Temporal DNA

```text
lookback
effective horizon
signal persistence
turnover
fast/medium/slow
```

---



## 46.5 Interaction DNA

```text
single-domain
price×volume
fundamental×price
liquidity×volatility
sector-relative
market-state conditional
cross-domain residual
```

---



## 46.6 Behavior DNA

从实际 factor value / return response 提取：

```text
monotonic
U-shape
inverted-U
tail-only
plateau
extreme-driven
state-dependent
high-turnover
broad
narrow
```

---



# 47. 2026 衰减统计不能做“朴素因果归因”

错误做法：

```text
2026 失效因子中 60% 有 ts_rank
→ ts_rank 容易失效
```

因为：

```text
全库可能 80% 都有 ts_rank
```

且它可能和 price/short-horizon 高度共现。

---



## 47.1 至少做置信度收缩

每个 operator/motif/direction 输出：

```text
support_count
raw_survival_rate
shrunk_survival_rate
confidence_interval
recent_retention
```

小样本不下强结论。

---



## 47.2 更正式的归因

建议在 Survival Lab 使用：

- hierarchical logistic regression；
- elastic net；
- group lasso；
- empirical Bayes；
- mixed effects；
- permutation importance（谨慎）；
- SHAP 仅作为预测解释，不直接宣称因果。

控制：

```text
field family
horizon
source
complexity
factor age
crowding
operator interactions
mechanism cluster
```

目标：

> 找“在控制其他因素后，哪些结构和 survival 显著相关”。

---



# 48. 自动发现“方向”：不要靠人工把所有方向提前列完

用户未来会有：

```text
几十万/几百万 factor
```

大量方向可能从未人工命名。

因此 Direction Memory 必须：

```text
machine-discovered first
human/LLM naming second
```

---



## 48.1 Factor Embedding

组合：

```text
deterministic AST / operator / field features
+
behavior/performance embedding
+
optional semantic/hypothesis embedding
```

不要只用 formula 文本 embedding。

---



## 48.2 Hierarchical Direction Clustering

建议层级：

```text
Coarse Mechanism
    ↓
Sub-Mechanism
    ↓
Motif
    ↓
Parameter Family
```

机器稳定 ID：

```text
DIR_C012
DIR_C012/SUB_04
DIR_C012/SUB_04/MOTIF_001
```

LLM 名称：

```text
"Volume-Price Divergence"
```

只是 alias，可改名，不影响 ID。

---



## 48.3 稳定 Cluster ID

Periodic recluster 后：

- 计算 old/new member overlap；
- Jaccard；
- centroid similarity；
- Hungarian matching；

尽量保持：

```text
DIR_C012 v3
→ DIR_C012 v4
```

而不是每次换随机 ID。

---



# 49. 自动生成“方向统计卡片”

系统才能真正生成：

```text
Volume-price divergence

seen: 13,820
active elite: 137
historic success rate: 7.4%
recent success rate: 1.8%
2026 Jun-Jul survival rate: 21%
median retention: 0.37
crowding: 0.95
search saturation: HIGH
uncertainty: LOW
```

而不是人工手写。

每个 Direction/Profile 至少：

```text
seen_count
unique_signal_count
active_elite_count
exported_count

historic_success_rate
recent_success_rate

median_search_fitness
median_asset_quality

survival_rate
2026_event_survival_rate
median_retention
q20_retention

crowding
saturation
novelty_gap

operator_distribution
field_distribution
horizon_distribution

support_count
uncertainty
```

---



# 50. Market Regime / Event Memory

2026 年 6–7 月不只应该作为一个日期标签。

创建：

```text
MarketRegimeEvent
```

例如：

```text
event_id
start_date
end_date
description

volatility_state
breadth_state
liquidity_state
style_dispersion
cross_sectional_dispersion
turnover_state
valuation_dispersion
index/regime context
```

然后统计：

```text
哪些 factor family 大幅衰减
哪些方向 survivor
哪些 operator/motif 被影响
哪些因子后来 recovered
```

未来遇到相似 regime：

SearchValue 可增加：

```text
RegimeRobustnessOpportunity
```

但权重不能过大，防止只为了一个两个月 event 过拟合。

---



# 51. SurvivalOpportunity 进入 SearchValue，但必须置信度收缩

例如：

```text
Direction A：
2026 survival 90%
但只有 7 个 factor
```

不能大幅奖励。

Direction B：

```text
支持 3000 个 factor
survival 72%
```

才可信。

使用：

```text
confidence-shrunk survival score
```

并设置最大贡献上限。

---



# 52. Seed Sampling：从随机变成记忆驱动

首版分配可用：

```text
60% quality acceptable + under-explored
20% historically fertile
10% rare direction/schema
10% pure random
```

概率：


P(seed)
\propto
Quality^\alpha
\cdot
UnderExplored^\beta
\cdot
Fertility^\gamma
\cdot
NoveltyNeed^\eta
\cdot
(1+TimesUsed)^{-\delta}


未来由 scheduler 动态学习，不永远固定 60/20/10/10。

---



# 53. Pool 架构彻底拆分

历史固定 `capacity=2000` 本身不是最大问题。

最大问题是：

> 一个 Pool 同时承担历史记忆、搜索工作集、输出集合。

改成：


| 层                          | 容量       | 是否常驻 RAM | 用途                        |
| -------------------------- | -------- | -------- | ------------------------- |
| Seed Library               | 无上限      | 否        | 已知 roots                  |
| Global Archive             | 无上限      | 否        | 所有唯一历史 factor             |
| Campaign Candidate Archive | 无上限/disk | 否        | 当前研究记录                    |
| Active Evolution Pool      | 有上限      | 是        | parent selection / search |
| Export Eligible Set        | 无 quota  | 否        | 正式输出候选                    |


---



## 53.1 Active Pool

首版建议：

```text
target_size = 1024
max_size = 2048
niche_elite_max = 32~64
```

这些不是硬真理，需 benchmark。

理由：

- nearest correlation；
- embedding；
- ranking；
- parent selection；

不能无限增大工作集。

---



## 53.2 淘汰逻辑

禁止：

```text
Pool full
→ pop lowest IC
```

改为：

```text
Pareto
+
Quality Diversity
+
SearchValue
+
Niche Coverage
```

即便某 factor IC 不是最高，如果：

- 很稳；
- 低相关；
- rare direction；
- offspring fertility 高；

仍可保留。

---



# 54. QD（Quality-Diversity）Archive

建立 niche：

可按：

```text
mechanism
schema
horizon
field family
turnover bucket
survival bucket
cluster
```

分 cell。

每个 cell 保留少量 elite。

这样 Pool 不会全部被：

```text
一种高 IC 量价结构
```

占满。

---



# 55. Embedding 性能优化

历史若每次 `pool.search()`：

```text
encode entire pool
→ full similarity matrix
```

必须改。

新规则：

Factor 第一次出现：

```text
embedding once
→ cache
```

查询：

```text
ANN Top-K
```

不要每轮重 encode 全池。

---



# 56. 7×24：真正的 RoundManager

目标：

```text
Round 001
→ Round 002
→ Round 003
→ ...
```

默认无限运行：

```text
max_hours = null
max_rounds = null
```

用户显式配置才停止。

---



## 56.1 Round 生命周期

```text
Load Global Memory
↓
Freeze Calibrator / Knowledge Cutoff
↓
Seed Sampling
↓
Campaign Initialization
↓
30~60+ Generations
↓
Pre-Export Refinement
↓
Export
↓
Update Search/Failure/Survival/Direction Memory
↓
Persist
↓
Next Round
```

---



## 56.2 Generation 深度

首版：

```text
target_generations ≈ 50
max_generations ≈ 80+
```

但不强制每条 lineage 跑满。

---



## 56.3 Lineage Early Stop

如果连续：

```text
patience ≈ 6 generations
```

没有：

- SearchFitness gain；
- novelty gain；
- PoolUtility gain；
- schema coverage gain；
- survival-related gain；

则停止该 branch。

强 lineage 获得更多预算。

---



# 57. Checkpoint v2

历史 checkpoint 如果保存：

- pool values；
- graph objects；
- test metrics；

恢复时又全重算，必须升级。

Checkpoint v2：

```text
run_id
round_id
campaign_id
generation

active_pool_factor_ids

pending_actions

scheduler_state
budget_state
rng_state

calibrator_version
knowledge_cutoff
survival_cutoff

global_memory_snapshot_id
evaluation_cache_snapshot_id

config_hash
system_version_key
```

不要把海量 factor matrix 直接塞 checkpoint。

---



## 57.1 v1 Migration

保留：

```text
checkpoint/migrate_v1.py
```

做：

- old expression -> FE canonical；
- old node -> FactorNode；
- single parent -> one-element parent list；
- old pool -> factor IDs；
- old metrics -> legacy evaluation record；
- **丢弃/隔离 test metrics，不进入搜索记忆**。

---



# 58. Exporter 重写

Exporter 不再算指标。

正确流程：

```text
EvaluationRecord
+
RefinementResult
+
GlobalSeenIndex
+
ExportRegistry
→ ExportDecision
```

---



## 58.1 不设 Top-N

正式：

```text
ExportEligible =
    HardGatePass
AND ExportScore >= floor
AND AuditPass
AND Novel
AND NotSeed
AND NotPreviouslyExported
AND RefinementComplete
```

本轮：

```text
0 个合格 → 输出 0
237 个合格 → 输出 237
```

不要强制 top 20 / top 100。

---



# 59. Pre-Export Factor Refinement 接入

所有准备正式输出的 AlphaPROBE factor：

```text
Candidate
→ Evaluator Diagnostics
→ Refinement PRE_EXPORT
→ Raw + Refined Variants
→ Re-evaluate
→ FactorFamily Representative
→ ExportGate
```

---



## 59.1 AlphaPROBE 只实现 Client

```python
class RefinementClient(Protocol):
    def refine(
        self,
        candidate: FactorCandidate,
        evaluation: EvaluationRecord,
        context: ExperimentContext,
        mode: str = "pre_export",
    ) -> RefinementResult:
        ...
```

不要复制完整 refinement logic。

---



# 60. Refinement 要覆盖的缺陷

公共 Refinement 应能够处理，AlphaPROBE 只接结果：

```text
orientation
D10 cliff
U-shape
inverted-U
tail-only
plateau
extreme-driven
distribution skew
numeric instability
coverage
high turnover
signal noise
industry exposure
size exposure
breadth/concentration
horizon specialist
regime specialist
```

---



# 61. 自动中性化：全诊断，选择性中性

绝对不要：

```text
所有 factor → industry+size neutralize
```

Router：

```text
exposure low
→ Raw

size high
→ try size-neutral

industry high
→ try industry-neutral

both high
→ try dual-neutral

style factor genuine
→ Raw allowed + STYLE_DEPENDENT

neutralization improves stability/LS
→ choose refined

neutralization destroys alpha
→ do not force
```

Factor-level neutralization：

```text
!=
Portfolio-level exposure constraint
```

避免 double neutralization。

---



# 62. PreprocessState：不重复加处理

从 FE AST inspect：

```text
rank already?
winsor already?
zscore already?
industry neutral already?
size neutral already?
EMA already?
shape transform already?
```

避免：

```text
rank(rank(f))
winsor(winsor(f))
neutralize(neutralize(f))
```

---



# 63. 高换手自动平滑

只有诊断为：

```text
HIGH_TURNOVER / HIGH_NOISE
```

才生成少量：

```text
EMA
EWM
rolling mean
rolling median
```

variant。

评价：

# 
SmoothingUtility

## \Delta Stability
+
\Delta LongShort
+
\Delta Cost

\lambda \Delta PredictiveLoss


短反转/事件型 signal 默认不机械平滑。

---



# 64. Shape Repair

U-shape：


g=
|CSRank(f)-c|


倒 U：


g=
-|CSRank(f)-c|


right tail：


g=
\max(CSRank(f)-c,0)


left tail：


g=
\max(c-CSRank(f),0)


D10 cliff：

- clip；
- compression；
- hump。

`c` / threshold / compression 参数：

```text
Train-only estimate
→ freeze
→ Valid only verify
```

---



# 65. Repair Budget

禁止：

```text
1 factor × 100 transforms × 100 参数
```

默认：

```text
最多 2 类 defect
总 variants <= 3
```

Raw 永远是 baseline。

# 
RepairUtility

## \Delta Fitness

## ComplexityIncrease

## TurnoverIncrease

OverfitRisk


---



# 66. Model Preprocessing 与 Factor Refinement 分开

例如：

```text
CS zscore
robust scale
rank-gauss
missing mask
```

如果只是为了模型数值训练，不要产生新 Factor ID。

输出：

```text
PreprocessProfile
```

---



## 66.1 Tree

推荐：

- raw/rank；
- optional robust winsor；
- 可保留 NaN routing；
- missing flag；
- 通常不强制 zscore。



## 66.2 NN

推荐：

- robust winsor；
- robust zscore / rank-gauss；
- simple fill；
- missing mask；
- freshness；
- fold-only fit；
- causal sequence preprocessing。

---



# 67. Consolidation 接口：不要在 AlphaPROBE 内聚类/PCA

AlphaPROBE 输出进入 Global Archive 后：

```text
Factor Consolidation Engine
```

负责：

- SimilarityEngine；
- incremental cluster；
- periodic recluster；
- representative；
- cluster composite；
- PCA/Ridge 等；
- model input registry。

AlphaPROBE 只需可选：

```python
class ConsolidationClient(Protocol):
    def get_cluster_context(
        self,
        factor_ids: list[str],
    ) -> dict[str, ClusterContext]:
        ...
```

用于 SearchValue：

- crowded cluster；
- rare cluster；
- cluster saturation。

---



# 68. 删除/迁移历史 VIF / SVD / 线性相关代码

历史 `remove_linearly_dependent_cols()` 如果仍是：

```text
SVD 算 rank=r
→ 直接取前 r 列
```

这是数学上不可靠的。

这类代码：

```text
VIF
linear dependence
linear regression combination
PCA
```

全部从 AlphaPROBE 搜索主链移出。

未来 Consolidation 若需要 rank-revealing selection：

```text
pivoted QR
rank-revealing QR
SVD + proper column subset
```

而不是“前 r 列”。

---



# 69. AlphaPROBE 与 Consolidation 的闭环

如果 Consolidation 发现：

```text
Factor F
Corr with Cluster17 = 0.94
但 SearchFitness 很好
```

可把它路由到：

```text
Refinement DEEP_REFINE
```

尝试 residual/new representation。

但这属于后续增强，不应该让 AlphaPROBE 直接做 pool-wide residualization。

---



# 70. 2026 Survival Intelligence 如何反馈搜索

不能直接：

```text
2026 活得好的 operator → 无限奖励
```

应形成：

```text
SurvivalOpportunity
```

输入：

- support；
- confidence；
- retention；
- recent survival；
- regime match；
- crowding；
- saturation。

SearchValue 中只占中等权重，并设置 cap。

---



# 71. Anti-Overfit：百万次尝试后的多重检验

7×24 以后，即使没有 Test 泄漏：

> 固定 Search Validation 也会被百万次尝试“拟合”。

所以：

1. Search Validation 高频；
2. Audit Validation 只给 L3/L4；
3. Audit 反馈不能完整给 LLM；
4. Sealed Test 最终；
5. 最终 finalist 可算：
  - Deflated Sharpe Ratio；
  - Probabilistic Sharpe Ratio；
  - PBO；
  - multiple-testing adjusted confidence。

这些是昂贵指标，只对少量 finalist。

---



# 72. Source / Data / Version 的完整可复现

每个 run 保存：

```text
Git commit
FactorEngine build
DataAccess build
Evaluator build
Refinement build
Consolidation build

data snapshot
universe snapshot

label spec
split spec
PIT contract

prompt version
LLM model
temperature
seed

knowledge cutoff
survival cutoff

search config
fitness config
```

---



# 73. 运行产物

每个 run 至少：

```text
run_manifest.json
attempts.jsonl
actions.jsonl
evaluations.parquet
lineage.json / parquet
pool_snapshot.json
budget_stats.json
checkpoint_latest.json
export_manifest.json
```

额外：

```text
survival_update.parquet
direction_update.parquet
memory_update_summary.json
```

---



# 74. Observability

运行面板至少统计：

```text
generated candidates
L0 rejects
duplicate rejects
seed duplicate
rank duplicate

L1 promoted
L2 promoted
L3 promoted
L4 promoted
exported

cache hit rate
FE batch size
FE runtime
Evaluator runtime

LLM tokens
LLM cost
LLM latency

action success rate
parent fertility
lineage depth

active pool size
niche occupancy

memory retrieval latency
ANN latency
```

---



# 75. Dataclasses / Pydantic Contracts

建议至少有以下正式类型。

---



## 75.1 FactorIdentity

```python
class FactorIdentity(BaseModel):
    factor_id: str
    canonical_formula: str
    canonical_ast_hash: str
    signal_equivalence_id: str
    parameter_family_id: str | None
    orientation: int
```

---



## 75.2 FactorCandidate

```python
class FactorCandidate(BaseModel):
    identity: FactorIdentity

    source: str
    parent_ids: list[str]
    action_id: str | None

    schema: dict[str, str]
    hypothesis: str | None

    field_set: list[str]
    operator_set: list[str]

    complexity: int
    depth: int
    lookback: int

    status: str
```

---



## 75.3 SearchAction

```python
class SearchAction(BaseModel):
    action_id: str
    action_type: SearchActionType

    parent_ids: list[str]

    target_role: str | None
    target_schema: dict[str, str] | None

    budget_class: str
    llm_model_class: str

    created_round: str
    created_generation: int
```

---



## 75.4 AttemptRecord

```python
class AttemptRecord(BaseModel):
    attempt_id: str
    action: SearchAction

    candidate_factor_id: str | None

    outcome: str
    rejection_reason: str | None

    delta_search_fitness: float | None
    novelty_gain: float | None
    pool_utility_gain: float | None

    eval_cost: float
    llm_cost: float

    latency_ms: int
```

---



## 75.5 FactorDNA

```python
class FactorDNA(BaseModel):
    field_families: list[str]
    operators: list[str]
    ast_motifs: list[str]

    mechanisms: list[str]
    schema: dict[str, str]

    horizon_bucket: str
    persistence_bucket: str
    turnover_bucket: str

    response_shape: str | None

    exposure_profile: dict[str, float]
    tradability_profile: dict[str, float]

    complexity_bucket: str
```

---



## 75.6 SurvivalProfile

```python
class SurvivalProfile(BaseModel):
    factor_id: str

    status: str

    long_term_retention: float | None
    recent_retention: float | None

    rolling_slope: float | None
    q20_rolling_rankic: float | None
    q20_rolling_sharpe: float | None

    sign_flip_rate: float | None

    max_deterioration: float | None
    max_breach_duration: int | None
    recovery_time: int | None

    descriptive_half_life: float | None

    confidence: float
    support_periods: int
```

---



## 75.7 ExplorationState

```python
class ExplorationState(BaseModel):
    factor_id: str
    action_family: str

    attempts: int
    successes: int
    elites: int

    mean_delta_fitness: float
    mean_novelty_gain: float

    saturation: float
    uncertainty: float

    version_key: str
```

---



## 75.8 ExportDecision

```python
class ExportDecision(BaseModel):
    factor_id: str

    hard_gate_pass: bool
    export_score: float | None
    audit_pass: bool

    novel: bool
    seed_duplicate: bool
    already_exported: bool

    refinement_complete: bool

    accepted: bool
    reasons: list[str]
```

---



# 76. 推荐目录

若当前包结构允许，重构为：

```text
alphaprobe/
├── runner.py
├── contracts.py
│
├── integration/
│   ├── factor_engine_adapter.py
│   ├── data_context.py
│   ├── evaluator_client.py
│   ├── refinement_client.py
│   └── consolidation_client.py
│
├── research_protocol/
│   ├── split_spec.py
│   ├── leakage_guard.py
│   ├── sealed_test.py
│   └── audit_validation.py
│
├── search/
│   ├── orchestrator.py
│   ├── actions.py
│   ├── scheduler.py
│   ├── schema.py
│   ├── role_parser.py
│   ├── stagnation.py
│   └── arms/
│       ├── refine.py
│       ├── branch.py
│       ├── evolution.py
│       ├── schema_explore.py
│       └── generation_repair.py
│
├── fitness/
│   ├── calibrator.py
│   ├── search_fitness.py
│   ├── search_value.py
│   ├── pool_utility.py
│   └── export_score.py
│
├── pool/
│   ├── active_pool.py
│   ├── qd_archive.py
│   └── admission.py
│
├── memory/
│   ├── global_store.py
│   ├── factor_registry.py
│   ├── seen_index.py
│   ├── source_ingestion.py
│   ├── retriever.py
│   ├── memory_packet.py
│   ├── exploration_state.py
│   ├── search_memory.py
│   ├── failure_memory.py
│   ├── survival_memory.py
│   ├── direction_memory.py
│   └── regime_memory.py
│
├── dedup/
│   ├── canonical.py
│   ├── signal_identity.py
│   ├── parameter_family.py
│   ├── fingerprint.py
│   └── reservation.py
│
├── lineage/
│   ├── graph.py
│   └── ledger.py
│
├── continuous/
│   └── round_manager.py
│
├── checkpoint/
│   ├── checkpoint_v2.py
│   └── migrate_v1.py
│
├── export/
│   └── exporter.py
│
├── observability/
│   ├── run_manifest.py
│   ├── events.py
│   └── cost.py
│
└── legacy/
    ├── old_expression_adapter.py
    ├── stock_data.py
    └── legacy_trainer.py
```

如果最新 repo 已经有更好的 package layout：

> 保留现有风格，按职责映射，不为“符合本文树形图”做无意义大搬家。

---



# 77. 推荐配置文件

```yaml
experiment:
  market: ashare
  run_mode: research

data:
  snapshot_id: auto
  universe:
    provider: data_access
    name: current_ashare_research_universe

label:
  type: vwap_to_vwap
  horizon_trading_days: 20

splits:
  train:
    start: "2016-01-01"
    end: "2021-12-31"

  search_valid:
    start: "2022-01-01"
    end: "2023-12-31"

  audit_valid:
    enabled: true
    # 实际日期根据当前研究协议重新划分，禁止和 sealed test 重叠

  sealed_test:
    # 当前 config 为权威；历史常用 2024-01-01 ~ 2026-07-31
    access: sealed

  purge_trading_days: 20
  embargo_trading_days: 5

factor_engine:
  native_dsl: true
  mode: research
  batch_size: 32
  parallel: true

evaluator:
  provider: unified
  batch_first: true

orientation:
  train_only: true
  keep_positive_rankic_only: true

fitness:
  calibrator:
    method: robust_mad_sigmoid
    freeze_per_round: true

  search:
    predictive: 0.26
    stratification: 0.20
    long_short: 0.25
    stability: 0.11
    novelty: 0.18

  penalties:
    cost_max: 0.02
    fragility_max: 0.05

stratification:
  groups: 10
  top_collapse_tolerance: 0.12

long_short:
  holding_days: 20
  cohort_portfolio: true

  transaction_cost:
    base: configurable
    stress_multipliers: [1.0, 2.0, 3.0]

  rolling_sharpe_window: 252

complexity:
  soft_free_nodes: 24
  light_to: 40
  medium_to: 64
  max_depth: 12
  hard_reject_above: configurable

search:
  target_generations: 50
  max_generations: 80
  lineage_patience: 6

  arms:
    refine: true
    branch: true
    evolution: true
    schema_explore: true
    generation_repair: true

scheduler:
  method: thompson
  cost_aware: true
  delayed_credit: true

llm:
  structured_output: true
  cheap_model: configurable
  strong_model: configurable

  memory_packet_max_tokens: 4000

cold_start:
  source: external_seed_library

  sampling:
    quality_underexplored: 0.60
    fertile: 0.20
    rare_direction: 0.10
    random: 0.10

memory:
  metadata_backend: duckdb_or_sqlite
  artifact_backend: parquet
  vector_index: ann_or_lsh

  knowledge_cutoff_required: true
  survival_cutoff_required: true

dedup:
  canonical_ast: true
  sign_invariant: true
  parameter_family: true
  rank_fingerprint: true
  atomic_reservation: true

active_pool:
  target_size: 1024
  max_size: 2048
  niche_elite_max: 64

  selection:
    pareto: true
    quality_diversity: true
    search_value: true

survival:
  enabled: true
  confidence_shrinkage: true

  event_memory:
    enabled: true

refinement:
  pre_export: true
  deep_refine: optional

export:
  quota: null
  threshold_based: true

  reject_seed_duplicate: true
  reject_previously_exported: true
  require_audit: true
  require_refinement_complete: true

continuous:
  enabled: true
  max_hours: null
  max_rounds: null

checkpoint:
  version: 2
  resume: true

observability:
  tokens: true
  llm_cost: true
  runtime: true
  cache: true
```

---



# 78. Performance：明确禁止的 O(N²) 路径

当历史 factor 数量达到：

```text
100k
500k
1m+
```

以下禁止：

```text
candidate × entire archive full correlation
entire archive embedding matrix
all factors kept as Python objects
entire lineage loaded to RAM
global reclustering per new factor
exporter full re-evaluation
checkpoint restore full re-evaluation
```

允许：

```text
ANN/LSH candidate retrieval
top-K exact confirmation
bounded Active Pool
disk-backed Global Archive
incremental statistics
cache/artifact refs
active subgraph only
```

---



# 79. Global Direction / Cluster 更新策略

AlphaPROBE 的 memory-level direction clustering 不等于最终 Model Consolidation。

它只用于：

- understanding search space；
- saturation；
- survival；
- search guidance。

新因子：

```text
DNA / embedding
→ nearest direction prototypes
```

三种情况：

1. 明确属于已有 direction；
2. 不属于任何 direction → 新 micro direction；
3. 同时接近多个 → bridge/ambiguous。

Periodic recluster：

```text
按新增数量 / 时间 / drift 触发
```

不每个 factor 全局重聚。

---



# 80. Survival Lab 的周期任务

AlphaPROBE 本身不需要每天都重做全历史。

建议 Memory 更新任务：

```text
Daily/round:
  new factor health
  recent rolling metrics

Weekly/threshold:
  direction aggregates
  operator/motif survival
  saturation

Monthly/major event:
  full survival report
  regime event update
  direction recluster
```

实际运行频率配置化。

---



# 81. 2026 June–July Event 分析要输出什么

至少：

```text
Total factors
Decayed factors
Broken factors
Recovered factors
Persistent factors

by:
  source
  field family
  operator
  operator pair
  AST motif
  mechanism direction
  horizon
  turnover
  complexity
  crowding
  cluster

survival:
  raw rate
  shrinkage rate
  confidence
```

再输出：

```text
Most robust directions
Most fragile directions
High-support survivor motifs
High-support decay motifs
Possible confounders
```

禁止把低样本现象直接标成规则。

---



# 82. 记忆系统还应记录“新颖性来源”

一个新因子为什么被认为新？

保存：

```text
structural novelty
rank novelty
value novelty
schema novelty
cluster novelty
residual novelty
```

后续可以统计：

> 哪种 novelty 最容易转化成长期 survival。

---



# 83. 记忆系统应记录“失败原因”

统一枚举：

```text
INVALID_DSL
PIT_VIOLATION
MISSING_FIELD
EXACT_DUPLICATE
SIGN_DUPLICATE
PARAMETER_REDUNDANT
RANK_NEAR_DUPLICATE

LOW_COVERAGE
NUMERICAL_FRAGILITY
LOW_PREDICTIVE
LOW_STABILITY
D10_COLLAPSE
BAD_LONG_SHORT

HIGH_CORRELATION
NO_INCREMENTAL_UTILITY

HIGH_COST
HIGH_TURNOVER
LOW_TRADABILITY

SEARCH_SATURATED
REFINEMENT_FAILED

AUDIT_FAIL
EXPORT_FAIL
```

这样 failure memory 可查询，不靠解析自然语言 log。

---



# 84. Logging：禁止把日志当计算入口

Logger 只写：

```text
factor_id
action
evaluation_ref
score
decision
cost
```

禁止：

```text
logger 中调用 evaluate()
logger 中读取 Test
logger 中重新回测
```

---



# 85. `argparse type=bool` 历史问题

如果最新版仍存在：

```python
parser.add_argument(..., type=bool)
```

必须改。

因为：

```text
bool("False") == True
```

使用：

```python
argparse.BooleanOptionalAction
```

或者配置系统的真正 bool 类型。

---



# 86. Universe/Instrument 必须被真正执行

历史 `FactorEngineStockData` 接收 `instrument`，但需确认是否真正用于过滤。

新系统：

```text
UniverseSnapshotID
```

必须参与：

- DataAccess query；
- Evaluator；
- cache key；
- artifact key；
- run manifest。

禁止仅在配置里写 universe name，但计算实际是全 A。

---



# 87. 测试体系

这是重构是否能交付的关键。

---



## 87.1 Unit — Identity / Canonical

测试：

```text
(f) == f
f*1 == f
1*f == f
f+0 == f
0+f == f
f-0 == f
0-f == -f
-(-f) == f
pow(f,1) == f

f 与 -f：
same SignalEquivalenceID
different orientation

x/x：
不得 unsafe simplify
```

---



## 87.2 Unit — Orientation

- Train RankIC positive → sign +1；
- Train negative → sign -1；
- Valid negative 不重新翻；
- Test 不影响 sign。

---



## 87.3 Unit — MetricCalibrator

- MAD；
- zero MAD safety；
- lower-is-better；
- per-round freeze；
- round change update；
- same raw metric in same round -> same utility。

---



## 87.4 Unit — D10

构造：

```text
D1 < ... < D9
D10 仅略低 D9
```

必须：

```text
small/no penalty
```

构造：

```text
D10 断崖低于 D7-D9
```

必须：

```text
large cliff penalty
```

---



## 87.5 Unit — Cohort PnL

用人工小样例验证：

```text
holding=3
每天开新 cohort
```

手算和代码完全一致。

再扩到 20d。

---



## 87.6 Unit — SearchValue

- factor 自己 SearchFitness 高，但 saturated/expensive → SearchValue 可以低；
- 中等 SearchFitness，但 fertile/rare → SearchValue 可以高；
- survival support 小时奖励必须 shrink。

---



## 87.7 Unit — ExplorationState

重复：

```text
F×WINDOW_SCALE
```

大量失败：

```text
saturation ↑
```

换 operator/search grammar version：

```text
可重新打开
```

---



# 88. Leakage Tests

必须成为最硬 CI。

---



## 88.1 Sealed Test Zero Reads

L0-L4：

```text
sealed_test datasource read = 0
```

---



## 88.2 No Test in Prompt

Prompt / MemoryPacket：

```text
不包含 test metric
不包含 post-cutoff survival
```

---



## 88.3 Repair Train-only

以下参数：

```text
U-shape c
D10 threshold
winsor percentile
EMA half-life
neutralization selection fitting
```

不得从 Valid/Test 拟合。

---



## 88.4 2026 Survival Cutoff

如果当前 run：

```text
knowledge_cutoff = 2026-01-01
```

则 2026-06/07 survival memory：

```text
Retriever 不可见
```

---



## 88.5 Financial PIT

测试：

```text
PubDate > decision date
→ 不可见
```

哪怕：

```text
ReportPeriodEndDate < decision date
```

也不允许。

---



## 88.6 UpdateTime

确保：

```text
UpdateTime
```

不会进入 PIT decision。

---



# 89. Integration Tests

---



## 89.1 FactorEngine Batch Equivalence

抽样历史 formula：

```text
legacy execution
vs
new FE batch
```

在合法等价语义下值一致/容差内一致。

---



## 89.2 Once-Compute

同 factor：

```text
Pool
Logger
Exporter
SearchFitness
```

消费同一个 EvaluationRef。

统计：

```text
FactorEngine expensive execution count == 1 per cache key
```

---



## 89.3 Seed Duplicate

导入 seed：

```text
F
```

后再生成：

```text
((F))*1
```

应：

```text
SEED_LIBRARY_DUPLICATE
```

并且：

```text
not export
```

---



## 89.4 Cross-round Rediscovery

Round 1：

```text
F
```

Round 20 再生成 F：

```text
same FactorNode
rediscovery_count += 1
```

不是新 node。

---



## 89.5 Multi-parent

```text
A + B -> C
```

lineage 查询必须同时返回两个 parent。

---



## 89.6 Crash Resume

在 generation N 强制 kill。

恢复：

```text
same round
same active pool
same pending actions
same RNG state if possible
same cache refs
```

不能重新从 cold start。

---



## 89.7 No Re-export

已经 export：

```text
F
```

未来再发现等价：

```text
ALREADY_EXPORTED
```

---



## 89.8 Refinement Round-trip

Raw：

```text
F
```

Refinement：

```text
F1/F2
```

Re-evaluate 后选择 best family representative。

lineage 必须记录：

```text
REFINEMENT
```

---



# 90. Performance Benchmarks

至少：

## 90.1 100k Seed Ingestion

测：

- parse/canonical；
- identity；
- registry insert；
- SeenIndex；
- fingerprint storage。

记录：

```text
wall time
peak RAM
disk
throughput
```

---



## 90.2 Seen Lookup

100k / 1m simulated：

```text
exact identity lookup
```

应是索引级，不遍历 Python list。

---



## 90.3 ANN / Fingerprint

Top-K retrieval latency。

---



## 90.4 FE Batch

对 1/8/32/64 factor 比：

```text
single run loop
vs
run_many
```

至少证明 batch path 正常。

---



## 90.5 Cache Hit

第二次同 context：

```text
不重新执行
```

---



## 90.6 MemoryPacket

无论：

```text
archive size=100k/1m
```

prompt packet 不超过配置 token。

---



## 90.7 Active Pool

确保任何 search path 不会：

```text
Global Archive N² correlation
```

---



# 91. Migration Strategy：禁止 Big Bang

按下面顺序做，每一阶段都能跑 baseline。

---



# Phase 0 — Current Repo Reconnaissance

交付：

```text
ALPHAPROBE_CURRENT_CODE_MAP.md
```

不改行为。

---



# Phase 1 — Research Protocol / Sealed Test

实现：

- `ExperimentContext`；
- `ResearchSplitSpec`；
- LeakageGuard；
- Test seal；
- logger 移除 test compute；
- label contract；
- purge/embargo。

Gate：

```text
L0-L4 test read == 0
```

---



# Phase 2 — FactorEngine / DataAccess Native Path

实现：

- `FactorEngineAdapter`；
- FE DSL canonical；
- batch execution；
- DataAccess context；
- universe snapshot；
- legacy adapter。

Gate：

```text
sampled FE vs legacy value parity
```

---



# Phase 3 — Evaluator Client + Once-Compute Cache

实现：

- EvaluatorClient；
- EvaluationRecord；
- artifact reuse；
- Pool/Logger/Exporter 全消费同一 record。

Gate：

```text
no repeated expensive evaluation
```

---



# Phase 4 — Identity / Dedup / Orientation

实现：

- canonical AST；
- sign ID；
- parameter family；
- GlobalSeenIndex；
- atomic reservation；
- fingerprint。

Gate：

```text
seed/sign/parameter duplicate tests pass
```

---



# Phase 5 — SearchFitness / Lifecycle Scores

实现：

- MetricCalibrator；
- P/Q/L/S/N；
- penalties；
- SearchFitness；
- SearchValue；
- PoolUtility；
- ExportScore interfaces。

Gate：

```text
old IC-only path no longer controls admission
```

---



# Phase 6 — Active Pool / QD / Native AlphaPROBE Upgrade

实现：

- Global Archive vs Active Pool；
- Pareto/QD；
- fertility；
- top-K correlation；
- embedding cache/ANN；
- descendant gain。

Gate：

```text
pool no longer pop-lowest-IC
```

---



# Phase 7 — Persistent Lineage / Global Memory / Source Ingestion

实现：

- Factor Registry；
- multi-parent edges；
- attempts；
- exploration；
- failures；
- seed ingestion；
- other miner source adapter；
- MemoryPacket.

Gate：

```text
cross-round rediscovery = same factor node
```

---



# Phase 8 — RoundManager / Checkpoint v2

实现：

- indefinite 7×24；
- resume；
- lineage patience；
- memory update；
- v1 migration。

Gate：

```text
forced crash resumes exactly
```

---



# Phase 9 — Multi-Arm Search / Scheduler / Structured LLM

实现：

- Refine；
- Branch；
- Evolution；
- Crossover；
- Schema；
- GenerationRepair；
- Thompson/UCB；
- structured outputs；
- model routing；
- action ledger。

---



# Phase 10 — Survival / Direction / Regime Intelligence

实现：

- Factor DNA；
- period performance；
- survival；
- 2026 event；
- confidence-shrunk operator/motif attribution；
- direction embeddings；
- hierarchical clusters；
- stable IDs；
- survival-aware SearchValue。

Gate：

```text
post-cutoff knowledge impossible to retrieve
```

---



# Phase 11 — Pre-Export Refinement / Export Gate

实现：

- RefinementClient；
- raw/refined family；
- audit gate；
- no top-N；
- export registry。

---



# Phase 12 — Consolidation Interface

只实现 AlphaPROBE 侧：

```text
ConsolidationClient
```

不要在 AlphaPROBE 写完整 clustering engine。

---



# Phase 13 — Legacy Cleanup / Performance / Documentation

把：

```text
old Expression
old StockData
old trainer
old metrics
```

移到：

```text
legacy/
```

新默认不走。

删除 dead code。

完善：

- benchmark；
- docs；
- config；
- runbook。

---



# 92. 旧功能删除清单

编码 AI 必须主动搜索和删除/隔离以下旧模式：

```text
搜索期间计算 test_ic/test_icir

node.test_ic
node.test_icir
作为 search feedback

load_test_res()
参与搜索

eval(expr)
exec(expr)

monkey patch Expression.evaluate
作为新默认执行

单 candidate 重复 engine.run

Pool 自己重新 evaluate

Logger 自己 evaluate

Exporter 重新 train/valid/test 全算

Pool full -> pop lowest IC

global mean corr 作为主要 novelty

每次 search 全池 embedding encode

single-parent graph

raw formula string = graph identity

whitespace hash = export identity

完整 lineage 全塞 prompt

%d 隐式窗口 brute force

argparse type=bool

VIF/PCA/SVD/linear-combination 混在 Trainer

Test knowledge 进入同版本 Global Memory
```

---



# 93. 必须保留并升级的 AlphaPROBE 原生能力

不要因为重构删掉：

```text
parent retrieval decay
depth decay
correlation-aware search
semantic novelty
edit distance idea
parent-child similarity
sibling similarity
child improvement
offspring sparsity
knowledge graph / lineage 思想
cold-start seed parent
continuous running
checkpoint resume
```

这些都升级到新体系。

---



# 94. Definition of Done

只有全部满足才算“AlphaPROBE 本次总重构完成”。

## 94.1 Research Correctness

- FactorEngine DSL/AST 唯一 formula truth；
- DataAccess 唯一 data/PIT truth；
- new main path 无 `eval`；
- Test 搜索阶段零读取；
- 20d label purge；
- Train-only orientation；
- Valid/Test 不翻方向；
- financial PIT 使用 PubDate；
- universe snapshot 明确；
- data/operator/evaluator versions 可追踪。



## 94.2 Performance

- batch FE；
- once compute；
- evaluator artifacts reuse；
- embedding cache；
- ANN/LSH；
- no global N²；
- checkpoint restore 不重算全 pool；
- exporter 不重算全 metrics。



## 94.3 Search

- SearchFitness 五维公式实现；
- D10 slight drop tolerant；
- severe D10 cliff detected；
- 20d cohort LS；
- cost light penalty；
- SearchValue 独立；
- fertility；
- multi-arm；
- scheduler；
- lineage early stop。



## 94.4 Pool

- Global Archive unlimited；
- Active Pool bounded；
- Pareto/QD；
- no lowest-IC pop；
- Export no quota。



## 94.5 Memory

- 10 万 seed 作为 roots；
- 其他算法可 ingest；
- aliases/source provenance；
- GlobalSeen；
- persistent multi-parent lineage；
- ExplorationState；
- Failure/Saturation；
- MemoryPacket；
- cross-round same signal no duplicate node。



## 94.6 Survival

- factor period performance；
- Factor DNA；
- 2026 Jun-Jul event；
- operator/motif/direction survival stats；
- confidence shrinkage；
- automatic direction discovery；
- stable direction IDs；
- post-cutoff leakage impossible；
- survival enters SearchValue only with uncertainty control。



## 94.7 Refinement

- pre-export hook；
- selective neutralization；
- D10/U-shape repair integration；
- high-turnover smoothing integration；
- duplicate preprocessing detection；
- raw vs refined re-evaluation；
- repair params Train-only。



## 94.8 Export

- HardGate；
- Audit；
- ExportScore；
- Novel；
- NotSeed；
- NotPreviouslyExported；
- RefinementComplete；
- 0 output is legal；
- hundreds output is legal。



## 94.9 Reproducibility

一个历史 factor 必须能回答：

```text
它的公式是什么？
来自哪里？
第一次哪轮出现？
是不是 seed？
有哪些 alias？
父节点是谁？
哪种 action 生成？
用了哪个 LLM？
花了多少 token/钱？
在哪个 data snapshot 评估？
RankIC/分层/多空是多少？
为什么入 Pool？
为什么被淘汰？
是不是曾经重新发现？
2026 是否衰减？
属于哪个 direction/cluster？
为什么被认为稳定/不稳定？
是否做过 refinement？
最终是否 export？
```

回答不了这些，就说明 Global Memory 还不完整。

---



# 95. 编码 AI 的执行要求

1. **不要一次性大重写。** 按 Phase 开 branch/commit。
2. 每个 Phase 先写测试，再迁移行为。
3. 旧路径先兼容，再切默认，再删除。
4. 不要修改 FactorEngine/DataAccess 内部语义来迁就 AlphaPROBE。
5. 不要在 AlphaPROBE 内复制 Evaluator 指标库。
6. 不要在 AlphaPROBE 内实现完整 Refinement Engine。
7. 不要在 AlphaPROBE 内实现完整 Consolidation Engine。
8. 不要用 Test 修当前版本搜索。
9. 不要为了“survival intelligence”破坏研究隔离。
10. 所有大型历史记忆 disk-backed。
11. 所有大规模相似性计算 Top-K/ANN-first。
12. 所有 LLM 调用记录真实 token/cost/model/prompt version。
13. 所有变更跑 regression + leakage + performance tests。
14. 每个 Phase 输出：
  - changed files；
    - migration notes；
    - tests；
    - benchmark；
    - remaining TODO。
15. 如果最新版仓库某个历史问题已经解决：
  - 不重新改回；
    - 在实施报告中标记 `ALREADY_RESOLVED`；
    - 继续执行其余目标。

---



# 96. 建议的最终开发汇报格式

编码 AI 完工后必须生成：

```text
ALPHAPROBE_REFACTOR_REPORT.md
```

包含：

```markdown
# Baseline
commit

# Current Code Mapping

# Phase 1
changed files
tests
results

...

# Compatibility
legacy support

# Performance
before / after

# Leakage Audit

# Memory Schema

# 100k Seed Benchmark

# Search Benchmark

# Known Limitations

# Remaining Future Work
```

---



# 97. 最终原则总结

这次改造不是“给 AlphaPROBE 再加几十个指标”。

真正要完成的是：

```text
旧：
一个 campaign 内的 LLM 因子进化器

新：
拥有统一执行、统一评估、多保真漏斗、
跨轮知识、跨算法历史、去重、失败记忆、
方向发现、因子生存智能和长期 7×24 调度的
Self-Improving Alpha Discovery System
```

必须长期坚持四条边界：

```text
FactorEngine      = 因子语言与执行
Evaluator         = 测量与事实
AlphaPROBE        = 搜索与学习
Refinement        = 单因子自动修复
Consolidation     = 因子集合整理与入模压缩
```

而 Global Factor Intelligence Memory 则把：

```text
冷启动 10 万历史公式
+
AlphaPROBE 全部 lineage
+
其他算法历史因子
+
失败尝试
+
搜索饱和度
+
2026 等衰减事件
+
survivor 方向
```

统一成可以被机器持续学习、但不会把海量历史直接塞给 LLM 的长期研究记忆。

完成上述全部内容后，再继续扩大 LLM 生成规模、generation depth、worker 数量，才具有真正的边际价值。