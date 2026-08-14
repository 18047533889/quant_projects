# 量化平台 6H+ 持续自治审计与开发任务书

**范围：FactorEngine / DataAccess / FactorAssets（FactorAssembly）/ Modeling / QuantEvaluator / FactorPreprocess / FactorOptimizer**  
**审计日期：2026-08-14**  
**当前代码快照标识：`c1f3564b5f9ee4b542f34abc1a974d01fff4f633`**

> 本任务书不是一次性 bug list。目标是建立一个持续运行的多 Subagent 开发组织：不断发现问题、复现、修复、独立复核、回归，再继续补充任务。只要运行环境允许，持续至少 6 小时 wall-clock（墙钟时间）。如果首批问题提前清完，不得停工，转入更深的性质测试、故障注入、跨模块集成、性能、并发和耐久审计。

---

# 1. 总原则

生产环境统一遵守：

```text
UNKNOWN != 0
UNKNOWN != EMPTY
UNKNOWN != False
UNKNOWN != DEFAULT
UNKNOWN != VERIFIED
```

以下情况必须 fail-closed（失败即阻断），不得静默返回空值、默认值或继续运行：

- PIT（Point-in-Time，时点一致性）无法证明；
- snapshot（数据快照）无法证明；
- universe（股票池）身份无法证明；
- semantic catalog（语义目录）不可用；
- backend capability（后端能力）不可确定；
- implementation evidence（实现证据）不完整；
- writer / materializer 写入失败；
- cache identity（缓存身份）不完整；
- memory/cost estimate 不可确定；
- q/Polars/DuckDB 输出 schema 不符合 contract；
- model split / fit window / label interval 无法证明；
- evaluator label timing 不完整。

平台责任边界：

```text
DataAccess
  → 数据、PIT、revision、snapshot、universe、calendar、semantic identity

FactorEngine
  → canonical factor semantics、operator、IR/DAG、backend planning/execution

FactorAssets / FactorAssembly
  → inventory、dedup、taxonomy、clustering、representative、composite、novelty、campaign

Modeling
  → split、fit/validate/freeze/predict、walk-forward、purge/embargo、model artifact

QuantEvaluator
  → factor/model/portfolio evaluation、statistics、robustness、production health
```

任何新 registry / policy / identity / planner 建立后，必须迁移 consumer 并删除旧 authority，禁止长期“两套并存”。

---

# 2. 自治 Subagent 组织

## 2.1 ChiefCoordinator

创建长期存在的 `ChiefCoordinator`。它尽量不直接修改大量生产代码，职责是：

1. 维护总任务池；
2. 给专业 Worker 分任务；
3. 防止两个 Worker 同时改相同模块；
4. 跟踪每项任务状态；
5. 完成后强制送 IndependentReviewer；
6. Reviewer reject 后重新入队；
7. runnable queue 不足时自动启动 Audit Miner；
8. 某 Worker 完成后立即补发下一项；
9. 每 20–30 分钟重新评估任务供给与风险排序；
10. 记录真实开始时间、结束时间和工作状态；
11. 上下文即将压缩/终止时先写恢复状态。

必须持续维护：

```text
AUTONOMOUS_MASTER_QUEUE.md
AGENT_STATUS.md
FINDINGS_LEDGER.md
AUDIT_COVERAGE_MATRIX.md
CERTIFICATION_MATRIX.md
REGRESSION_MATRIX.md
DECISIONS_AND_CONTRACTS.md
RESUME_STATE.md
```

## 2.2 IndependentReviewer

至少一个独立 Reviewer。**写代码的人不能自己宣布自己的任务完成。**

Reviewer 必须：

```text
复现原 bug
→ 检查修复
→ 自己再构造一个反例
→ 检查相邻 contract
→ 检查 silent fallback
→ 检查 PIT/identity
→ 跑 targeted regression
→ PASS / REJECT
```

## 2.3 Worker 池

建议角色：

```text
Worker-FE-Semantics
Worker-FE-Backend
Worker-Q
Worker-DA-PIT
Worker-DA-Identity
Worker-FactorAssets
Worker-Modeling
Worker-Evaluator
Worker-Runtime
Worker-Packaging
Worker-FaultInjection
Worker-ArchitectureMiner
```

高内存/测试重型 Worker 默认最多同时 2 个；BLAS 线程设为 1，避免测试本身争抢全部资源。

---

# 3. 任务状态机

所有任务必须：

```text
DISCOVERED
→ REPRODUCED
→ FIXING
→ LOCAL_TESTED
→ INDEPENDENT_REVIEW
→ REGRESSION_TESTED
→ CLOSED
```

无法复现则：

```text
DISCOVERED → NEEDS_PROOF
```

不能“看起来没问题 → CLOSED”。

---

# 4. 自动补货机制：让 AI 一直往下干

ChiefCoordinator 每 20–30 分钟检查：

```text
Runnable tasks < 15
```

立即启动新的 Audit Miner。

若：

```text
Runnable tasks < 8
```

至少同时启动两个：

```text
Miner-A：静态架构/代码异味
Miner-B：性质测试/故障注入
```

如果已知 P0/P1 全部完成，**禁止结束**，自动开启下一轮：

```text
Deep Audit Round N+1
```

继续检查之前没有获得证据的 contract。

Audit Miner 必搜：

```text
TODO / FIXME / NotImplemented / pass
placeholder / proxy / approximation / simplified
fallback
except Exception / bare except
return [] / {} / 0 / None
warnings.warn
default=str / repr / str(obj)
private attribute (_xxx)
global singleton / lru_cache / mutable global
monkey-patch
sys.path
absolute home paths
environment-driven production mode
hardcoded version/date
duplicate Registry/Catalog/Policy/Spec/Identity
```

每个发现必须判断是合法 research 行为还是 production correctness 风险。

---

# 5. 最新代码：已完成事项不要机械重做

最新代码相较上一轮已经集中整改了一批 P0。以下应当进入“独立复核/回归证明”，而不是重新从零实现：

```text
Parameter active_when production fail-closed
Runtime calibration typed factors
Memory zero-estimate guard
SemanticCatalogIdentity
Strict SHA-256 correctness identity
BucketHashRegistry
Layout write/read symmetry
DataAccess I/O boundary checker
Unified backend contracts module
DuckDB ts_skew / ts_kurt parity
```

原则：

```text
已有实现
→ review
→ negative test
→ integration proof
→ 才能 CLOSED
```

已有“完成报告”“零失败报告”都不作为最终 authority，只认实际代码和可重复的测试。

---

# 6. 当前确认的首批任务

## 6.1 DataAccess I/O Authority

### ARCH-P0-001：FactorEngine 的 I/O boundary hook 目前不是硬阻断

local hook 调 checker 时没有启用 strict；checker 只有 strict + violation 才返回失败。

**改：**

```text
violations > 0 → non-zero
```

生产 architecture gate 不允许“检测出违规但流程仍成功”。

### ARCH-P0-002：checker 扫描失败 fail-open

当前 SyntaxError / 扫描异常可被转成“没有 violation”。

**改：**

```text
scan failure → CHECK_INFRASTRUCTURE_FAILURE → gate fail
```

### ARCH-P0-003：存在两套 I/O boundary authority

目前 root allowlist/checker 与 FactorEngine local checker 各有一套规则。

**改：**

建立唯一 `PhysicalIOAuthorityPolicy`；checker 只消费它。豁免必须分类为：

```text
SOURCE_READ
RESULT_PERSISTENCE
MAINTENANCE
TEST_FIXTURE
BENCHMARK
MIGRATION
```

尤其市场/raw/source 数据读取应逐步收回 DataAccess。

---

## 6.2 Backend Capability

### FE-BE-P0-001：所谓“统一 backend contract”仍有 legacy BackendKind

Polars classification 模块仍自己定义 `BackendKind`。

**目标：**

```text
BackendKind
ExecutionKind
CapabilityLevel
PhysicalImplementationSpec
```

全平台只有一个 authority。

### FE-BE-P0-002：BackendCapability 与 BackendCapabilityRecord 仍重复

一个使用字符串 execution kind，一个使用 enum。

**目标：**合并为唯一 typed record。

### FE-BE-P0-003：Polars capability_quality 可与 production classifier 矛盾

可能出现：

```text
production = UNSUPPORTED
quality report = polars_native_kernel
```

因为 quality path 仍可 source inspection。

**改：**production router/report/evidence 全部只读 explicit PhysicalImplementationSpec；heuristic 仅 research diagnostics。

### FE-BE-P0-004：physical_spec() 抛异常不能被当“没有 spec”

区分：

```text
MISSING
INVALID
INFRASTRUCTURE_ERROR
```

INVALID/ERROR 在 production 必须 hard fail。

### FE-BE-P0-005：is_production_eligible() 目前验证不足

真正 production eligibility 至少绑定：

```text
semantic contract
param domain
null/NaN/Inf
min_periods
group/window
implementation hash
runtime version
parity evidence
```

若只做结构判断，请改名 `is_structurally_eligible()`。

### FE-BE-P0-006：q 还没进入主 BackendName capability authority

统一 enum 已有 q，但主 capability type 仍主要是 Pandas/Polars/DuckDB/ClickHouse。

**目标：**q 同等进入 registry/planner/cost/telemetry/evidence/runtime。

### FE-BE-P0-007：SQL emitter hash 失败仍会 `"unknown"`

生产 evidence identity 不能：

```text
hash unavailable → "unknown" → continue
```

改为 typed infrastructure error。

---

# 7. q/K Backend 首批任务

### Q-P0-001：手工 `_PHASE1_NATIVE_OPS` 不能成为 capability authority

自动计算：

```text
DeclaredNative
LoweringExists
CompilePass
RuntimePass
ParityPass
```

production 必须五者闭环。

### Q-P0-002：生成 Native Set - Lowering Set 差集

要求：

```text
Q_NATIVE_WITHOUT_LOWERING = 0
```

### Q-P0-003：compiler generic lambda 分支可能使 special lowering 永远到不了

重点：

```text
ts_beta
wma
clip
```

改成显式：

```text
Q_LOWERING_REGISTRY
canonical → exactly one lowering
```

### Q-P0-004：ts_quantile / cs_quantile 等声明与真实 map/lowering 对齐

Gate：

```text
Q_CAPABILITY_IMPLIES_REACHABLE_LOWERING
```

### Q-P0-005：rolling corr/cov/beta 重新认证

证明逐时点：

\[
Corr_t(x,y;W), \quad
Cov_t(x,y;W), \quad
\beta_t=\frac{Cov_t(x,y;W)}{Var_t(y;W)}
\]

不是 aggregate 或错误切片。

### Q-P0-006：统一 rolling wrapper

冻结：

```text
window
min_periods
warmup
ddof
null
NaN
Inf
ordering
group
session
```

### Q-P0-007：lag/rolling instrument boundary

故意将两只股票拼接，确认不会把 A 的末行带到 B 的首行。

### Q-P0-008：QBackend 仍把整个 logical plan 当单 region

生产 QBackend 只接受 `PhysicalBackendRegion`，region 由统一 planner 决定。

### Q-P0-009：基础数据缺失不能返回 empty DataFrame

改成 typed `DataUnavailableError`。

### Q-P0-010：输出 schema 不符合预期不能猜第一列/空 Series

建立 `QOutputContract`：

```text
keys
value column
dtype
row_count
grain
ordering
nullability
```

### Q-P0-011：global q backend singleton 有 research→production 配置污染风险

第一调用若允许 fallback，后续 production 可能拿到同一个实例。

改成 immutable-config keyed instance，或取消带运行模式的全局 singleton。

### Q-P0-012：object NULL 与 empty string/symbol 不得合并

禁止 `fillna("")` 作为通用 production conversion。

### Q-P0-013：short/int/long 各自 null sentinel

### Q-P0-014：nullable boolean 保持 True / False / NULL 三态

### Q-P0-015：zero-copy 失败的 fallback 只 catch 明确 zero-copy error

semantic/type conversion error 不得被第二次 `toq()` 掩盖。

### Q-P1-016：Resident handle identity 增加

```text
process_id
session_generation
connection_generation
adapter_version
runtime_version
semantic_hash
source_snapshot
universe_snapshot
PIT identity
schema_hash
ownership/refcount
```

### Q-P1-017：fan-in / fan-out / last-use release / process restart invalidation

---

# 8. BackendRegionPlanner / Polars / DuckDB

### REGION-P0-001：统一 BackendRegionPlanner 成为主执行 authority

```text
Canonical DAG
→ capability annotation
→ region partition
→ compute + transfer cost
→ region execution
```

### REGION-P0-002：禁止 Polars recursive fallback 的逐算子 pandas↔Polars ping-pong

Gate：

```text
BACKEND_ZERO_OPERATOR_LEVEL_CROSS_ENGINE_PINGPONG
```

### REGION-P0-003：Region 内保留 native intermediate

```text
Pandas → DataFrame/ndarray
Polars → LazyFrame/Expr
DuckDB → Relation/SQL pipeline
q → resident q object
```

只在 region boundary 转换。

### REGION-P1-004：成本模型包含 boundary cost

\[
TotalCost = Compute + Transfer + Materialization + Serialization
\]

### REGION-P1-005：DuckDB SQL fragmentation telemetry

记录 query count、region count、materialized bytes、Arrow transfer、compile time、execution time。

---

# 9. Runtime / Resource

### RT-P1-001：校准 persistence 是否真的接入主链

`load_server_calibration` / `save_server_calibration` 虽存在，但要证明真实 consumer。没有 consumer 就不能声称 persistent calibration。

### RT-P0-002：校准 load/save 的 broad except 要清理

corrupt JSON、permission denied、schema mismatch 要可见。

### RT-P0-003：server fingerprint 失败不能统一 `"unknown"`

避免多个环境共享 `unknown` calibration。

### RT-P0-004：Unknown memory → 8MB 不一定 conservative

建立 task-class upper bound，无法估计时 production 可拒绝 admission。

### RT-P1-005：定义 server fingerprint 真正包含哪些硬件/软件维度

避免 hostname 造成无意义碎片化。

### RT-P0-006：Host-level ResourceAuthority

多进程不能各自认为拥有整机 RAM/CPU。

### RT-P0-007：typed backend/runtime errors

```text
BackendUnavailableError
BackendOOMError
BackendTimeoutError
BackendContractError
BackendSemanticError
BackendTransportError
BackendRuntimeCrash
ResourceAuthorityUnavailable
```

---

# 10. DataAccess Identity / Layout / Build

### DA-ID-P1-001：correctness identity caller 不允许 strict=None 自动猜环境

### DA-ID-P0-002：审计所有 `hash_cache_key` / 64-bit/non-strict caller

凡影响结果复用的全部迁到 `hash_correctness_identity`。

### DA-ID-P1-003：datetime identity 统一到 canonical UTC instant

明确同一 instant、不同 timezone representation 是否同 identity。

### DA-ID-P1-004：list vs tuple、set vs frozenset 的 identity 等价是否是有意设计

### DA-ID-P0-005：production correctness caller 禁止直接 `hash_identity(bits=64)`

### DA-LAYOUT-P1-001：`_strict_count` 文档说 float reject，但 integral float 仍可能接受

统一 contract。生产建议只收真正 `int`，bool/float/string 全拒绝。

### DA-LAYOUT-P0-002：legacy raw bucket API 不得被 production caller 绕开 validated policy

production 只走 `*_from_policy`。

### DA-LAYOUT-P1-003：`stable_bucket(str(key))` 会把 `123` 与 `"123"` 合并

instrument key 若应为 str，则 reject 非 str，不要强制转换。

### DA-BUILD-P0-001：artifact version/build revision 应在构建时冻结

### DA-BUILD-P0-002：production 安装后不依赖 runtime source tree 执行 `git rev-parse`

### DA-BUILD-P0-003：外部 revision 必须严格格式验证

### DA-BUILD-P0-004：`DataSnapshotIdentity` 与 `ExecutionBuildIdentity` 分离

代码变化不等于数据内容 snapshot 变化；provenance 同时记录二者即可。

---

# 11. FactorAssets / FactorAssembly

### FA-P0-001：DataAccess adapter import 名称核对并修正为正式 package API

当前 adapter 与 DataAccess 发布 import contract 不一致的风险必须消除。

### FA-P0-002：Optional dependency detection 不应吞 TypeError

TypeError 可能是 integration bug，不是“依赖没装”。

### FA-P0-003：`check_factor_availability()` 不能拿到 handle 就 True

真正 materialize/validate，并带 snapshot/PIT identity。

### FA-P0-004：Catalog adapter 不能用 fabricated metadata / empty tuple placeholder

直接消费 DataAccess 正式 catalog API。

### FA-P0-005：FactorAssets 不应自己重新定义 FactorEngine factor identity

FactorEngine 签发 `FactorIdentityArtifact`；FactorAssets 只消费。

### FA-P1-006：compiler generation 不要硬编码版本字符串

### FA-P1-007：complex expression parsing 的 public contract 明确化

要么接正式 parser，要么 public API 只接受 Expr/FactorArtifact，不留半实现字符串接口。

---

# 12. Campaign / Contamination / Multi-Fidelity

### CAMP-P0-001：`max_duration_s` 必须由 monotonic clock 自动推进

### CAMP-P0-002：evaluation/cost reservation 原子化，防并发超预算

### CAMP-P0-003：Objective 必须声明 direction=max/min

### CAMP-P0-004：NaN/Inf metric 不能进入 best/plateau/stopping

### CAMP-P1-005：validate positive budget/patience/config

### CAMP-P1-006：长 campaign 需要 persistence/checkpoint/resume

### CAMP-P1-007：显式状态转移表，非法 transition hard fail

### LEDGER-P0-001：`seal_test_splits` 必须真实 enforce，而不只是写一条 seal metadata

### LEDGER-P0-002：test split 参与任何 adaptive search 都算 contamination

包括 feature selection、hyperparameter、threshold、model choice、factor mutation。

### MF-P0-001：`confidence_threshold` 必须实际使用或删除假参数

### MF-P0-002：`max_candidates_per_tier` 真正 enforce

### MF-P0-003：NaN metric 先 `isfinite`，否则 Python comparison 可绕过 threshold

### MF-P0-004：NaN stability/survival 同样 fail-closed

### MF-P1-005：预算连 L0 都负担不起时返回 NO_AFFORDABLE_TIER，不要仍给 L0

### MF-P1-006：fidelity sample 必须 deterministic + date/PIT-safe，禁止 random row sampling

---

# 13. Plateau / Pareto

### PLAT-P0-001：没有 neighbor evidence 不能返回 perfect stability/survival

应该 `INSUFFICIENT_EVIDENCE`。

### PLAT-P0-002：base_metric==0 不能自动得到零 sensitivity/完美稳定

### PLAT-P0-003：NaN/Inf neighbor metric hard handling

### PLAT-P0-004：参数扰动必须服从 ParamSpec/ParamRole/bounds/type/active_when

例如 window=20 不应被随意变成 float。

### PLAT-P1-005：所有 neighbor 使用同一 frozen snapshot/universe/label/cost/split contract

### PARETO-P0-001：不能默认所有 objective 都 maximize

例如：

```text
IC ↑
Sharpe ↑
Turnover ↓
MaxDrawdown ↓
Cost ↓
```

引入 `ObjectiveSpec(direction, transform, missing_policy, normalization)`。

### PARETO-P0-002：NaN objective reject

### PARETO-P0-003：缺 objective 不要默认 -Inf

### PARETO-P1-004：所谓 normalized distance 必须真正标准化各 objective 尺度

### PARETO-P1-005：empty objective list / duplicate point_id reject

---

# 14. Modeling 时序泄漏

### MODEL-P0-001：`SplitSpec.gap_days` 已定义但必须真正 enforce

### MODEL-P0-002：按 label interval 做 purge

仅 `train_end <= val_start` 不够。若标签覆盖 `t→t+5`，train 最后几天仍会进入 validation 信息区间。

### MODEL-P0-003：显式 Embargo contract

### MODEL-P0-004：VWAP-to-VWAP LabelContract 与 split 绑定

生产默认评估明确：

```text
DecisionClock
signal_available_time
execution VWAP
label_start VWAP
label_end VWAP
horizon
```

### MODEL-P0-005：`FittedTransform.transform(..., apply_start_time=None)` 不得成为 OOS 绕过口

public OOS transform 必须要求 `ApplicationWindow`；in-sample bypass 只允许内部 `fit_transform`。

### MODEL-P0-006：FitWindow 必须与实际输入 timestamps 绑定

不能 caller 给全样本 X，却声称是 train window。

### MODEL-P1-007：`CrossSectionalScaler` 名称与数学语义重新核对

当前若是 axis=0 跨时间训练统计量，更像 train-fitted feature scaler；若真正想 cross-sectional，则实现语义需改。

### MODEL-P0-008：ddof=1 + 样本过少产生 NaN std 的处理

### MODEL-P1-009：OOF anchored strategy 重新定义

anchored 通常固定起点、扩展终点，不应简单等价 rolling。

### MODEL-P0-010：ModelArtifact identity 完整绑定

```text
feature set
preprocess
split
label
data snapshot
universe
semantic catalog
hyperparameters
random seed
code build
library versions
```

---

# 15. Modeling Adapter

### MODEL-AD-P0-001：fit window translation 不丢 data_snapshot_ref

### MODEL-AD-P1-002：transform spec 由 Dict[str,Any] 升级 typed/versioned contract

### MODEL-AD-P0-003：feature_names 与 feature matrix feature 维度必须一致

### MODEL-AD-P1-004：producer version 绑定真实 artifact build identity

---

# 16. QuantEvaluator Quantile / Statistics

### QE-Q-P0-001：`assign_quantiles(method="average")` 的 method 目前必须证明真实参与 tie policy

建立 `QuantileTiePolicy`。

### QE-Q-P0-002：NumPy 与 Numba 对 boundary equality 的归组必须 exact parity

一边 `searchsorted(side="right")`，另一边 `val > boundary` 会产生 tie 差异风险。

### QE-Q-P0-003：T=1 / F=1 的 `.squeeze()` 不得把内部二维 shape 压成一维

### QE-Q-P0-004：Numba `fastmath=True` 与 NaN/Inf correctness 做正式 parity proof；证明不了就关闭

### QE-Q-P1-005：

```text
n_quantiles >= 2
min_assets >= 1
shape validation
```

### QE-Q-P1-006：tie-heavy property tests

例如 100 只股票中 80 只因子值相同。

### QE-P0-001：完整 suite / integration 重新执行，不以报告文本替代

### QE-P0-002：Batch vs Streaming equivalence

同 FactorBatch + LabelBundle + MetricSpec，结果 exact/tolerance-equivalent。

### QE-P0-003：Metric identity 包含 tie/min_periods/annualization/cost 等语义

### QE-P0-004：VWAP-to-VWAP ReturnSpec 成为 production 一级 contract

---

# 17. Packaging / Deployment

### PKG-P0-001：真实执行 clean-wheel smoke，不是“脚本存在”即完成

对：

```text
FactorEngine
DataAccess
FactorAssets
Modeling
QuantEvaluator
FactorPreprocess
FactorOptimizer
```

执行：

```text
build wheel
→ clean venv
→ install
→ 离开源码目录
→ public import
→ minimal execution
```

### PKG-P0-002：清理 production `sys.path` injection

所有引用分类：production/test/script/benchmark/migration；production 清零。

### PKG-P0-003：扫描硬编码 `/home/...`、用户目录、机器特定路径

### PKG-P1-004：明确 FactorOptimizer / FactorPreprocess 是长期独立 package，还是并入 FactorAssets / Modeling

不能职责双重拥有。

---

# 18. End-to-End Identity

### E2E-P0-001

完整链：

```text
DataAccess
→ FactorEngine
→ FactorAssets
→ Modeling
→ QuantEvaluator
```

以下 identity 不能丢：

```text
DataSnapshotIdentity
UniverseSnapshotIdentity
SemanticCatalogIdentity
CalendarIdentity
FactorIdentity
FeatureSetArtifact
PreprocessArtifact
ModelArtifact
LabelContract
EvaluationContract
```

### E2E-P0-002

只改变 actual snapshot，其余相同：

```text
factor cache miss
model identity changes
evaluation identity changes
```

---

# 19. 全维度 Audit Coverage

ChiefCoordinator 要把以下每项放入 `AUDIT_COVERAGE_MATRIX.md`，每一格记录 Static / Unit / Property / Fault / Integration 是否有证据。

## DataAccess

- event_time / knowledge_time / effective_time；
- announcement after close；
- financial revision/restatement；
- as-of join；
- universe historical membership / rebalance / IPO / delisting；
- trading calendar / holiday / half-day / DST / session；
- minute→daily boundary；
- corporate actions / adjusted price；
- unit / currency / FX PIT；
- duplicate / schema evolution；
- missingness / coverage；
- snapshot/content digest；
- change impact / invalidation；
- atomic generation；
- remote/local fallback；
- data-quality fail-closed；
- physical I/O authority。

## FactorEngine

- canonical / alias；
- ParamRole / default / active_when / bounds；
- formula authenticity；
- unit / grain / frequency；
- rolling min_periods / warmup / ddof；
- NaN / NULL / Inf；
- tie rank / quantile；
- group/instrument/session boundary；
- complete cross-section universe；
- sharding axis；
- recursive state / checkpoint；
- CSE；
- dependency extraction；
- factor identity；
- expression parser；
- no-future poison；
- evidence authenticity；
- precision policy。

## Backend

- Pandas oracle；
- Polars native vs delegate；
- DuckDB SQL；
- q；
- capability single authority；
- BackendRegionPlanner；
- transfer cost；
- native residency；
- fallback/error taxonomy；
- dtype coercion；
- serialization/IPC；
- runtime version；
- evidence hash；
- benchmark。

## Materialization / Cache

- DataReadIdentity；
- universe/snapshot/catalog/calendar/code identity；
- invalidation；
- atomic commit；
- partial write；
- corruption；
- lineage；
- idempotency；
- concurrent writer；
- GC/version retention。

## FactorAssets / Assembly

- exact/semantic/empirical dedup；
- taxonomy；
- correlation clustering；
- representative；
- composite train-only；
- orthogonalization；
- novelty；
- seen index；
- campaign budget；
- multi-fidelity；
- Pareto；
- plateau；
- split contamination；
- frozen candidate；
- persistence/resume；
- multiple testing。

## Modeling

- date split；
- label interval purge；
- embargo；
- walk-forward；
- hidden holdout；
- train-only preprocess；
- fitted artifacts；
- train-only feature selection；
- hyperparameter contamination；
- OOF；
- retrain/cutover；
- model artifact identity；
- seed/reproducibility；
- missing/sample weight；
- prediction timing。

## QuantEvaluator

- VWAP-to-VWAP；
- decision/signal/execution/label timing；
- Pearson IC / RankIC；
- rank/quantile ties；
- ICIR annualization；
- turnover；
- transaction cost/slippage；
- capacity；
- Sharpe / MDD / Calmar；
- coverage；
- bootstrap/confidence；
- regime/subperiod robustness；
- multiple testing / FDR / deflated Sharpe；
- streaming/batch equivalence；
- deterministic output。

## Packaging / Ops

- clean wheel；
- source-tree-independent import；
- dependency constraints；
- Python version；
- build identity；
- no sys.path hack；
- no absolute home；
- optional dependencies；
- import-time side effects；
- reproducible environment。

## Performance / Resource

- peak RSS；
- memory upper bound；
- global broker；
- BLAS oversubscription；
- q process pool；
- DuckDB connections；
- Polars pool；
- scan cost；
- boundary transfer bytes；
- resident GC；
- cancellation/timeout；
- resource leaks；
- soak test。

## Security

- SQL sandbox/injection；
- path traversal；
- secret logging；
- credentials；
- unsafe pickle/deserialization；
- temp permissions；
- service auth；
- query budget bypass；
- unbounded resource requests；
- expression parser safety。

---

# 20. Property Test 矩阵

核心 operator/backend/metric 必测：

```text
T=0,1
N=0,1
F=0,1
all NaN
all +Inf / -Inf
mixed NaN/Inf
constant / nearly constant
heavy ties
duplicate date
duplicate instrument
unsorted index
sparse dates
halted assets
changing universe
window=1
window>history
window=0
negative window
huge magnitude
tiny magnitude
float32 / float64
nullable integer
nullable boolean
timezone-aware / naive
```

---

# 21. Fault Injection

至少主动注入：

```text
missing file
corrupt manifest
schema drift
snapshot unavailable
partial partition
duplicate PK
disk full
permission denied
rename failure
writer interruption
DuckDB connection failure
q unavailable
q process crash/restart
IPC timeout
backend OOM
memory estimator crash
resource authority unavailable
calibration corruption
worker cancellation
cache corruption
wrong snapshot cache
network timeout
missing optional dependency
```

Production 不得把这些变成“空结果但成功”。

---

# 22. Concurrency / Global State 专项

创建 `Worker-Concurrency`，扫描：

```text
global singleton
mutable registry
lru_cache
threading.local
weakref
shared connection
shared writer
global stats
global process manager
```

至少测试：

```text
2 threads
4 threads
2 processes
concurrent read/write
research→production mode switch
process restart
```

重点：q singleton、OperatorRegistry、DataAccess store、calibration、campaign、seen index。

---

# 23. PIT Poison Tests

### PIT-001 Financial restatement

```text
original EPS=1.00
later restated EPS=1.12
```

重述前历史 as-of 必须始终读 1.00。

### PIT-002 After-close

```text
announcement 18:00
decision 15:00
```

当日 signal 不得看到。

### PIT-003 Future universe member

未来加入指数的股票不能回填历史。

### PIT-004 Future extreme

把未来输入改成 `1e30`，过去因子结果保持不变。

### CACHE-001 Snapshot mutation

相同请求参数但 actual snapshot S1→S2，必须 cache miss。

### MODEL-001 Future validation/test mutation

改变未来 validation/test label，不得改变 train fitted preprocess/model。

### QE-001 Label horizon mutation

改变 label_end 之后的数据，不得改变当前 label。

---

# 24. Performance / Soak

Correctness 通过后再优化。

Telemetry 至少：

```text
wall time
CPU
peak RSS
scan bytes
output bytes
conversion bytes
materialized bytes
backend switches
region count
SQL query count
q IPC bytes
```

Benchmark：

```text
1 / 100 / 1,000 / 10,000 factors
252 / 1,260 / 2,500 days
500 / 3,000 / 6,000 stocks
minute→daily
```

若时间允许，安排 2–4 小时连续 soak，监控 RSS、fd、connections、q resident object、cache/temp/thread/process 数量是否持续增长。

---

# 25. 6 小时执行节奏

不是到点才切阶段，始终流水化。

```text
0–30m:
  建 Master Queue、Agent Status、baseline smoke、首批 reproduction

30–120m:
  P0 architecture / q / backend capability / DataAccess PIT+identity /
  Modeling leakage / Evaluator statistics

120–240m:
  BackendRegion / cross-backend parity / PIT poison /
  FactorAssets optimizer+contamination

240–300m:
  full integration / clean wheel / E2E identity / VWAP-to-VWAP

300–360m+:
  property / fault / concurrency / performance / soak /
  independent re-audit / new issue mining
```

P0 未结束时不要为了时间表转去做低优先级工作。

---

# 26. Coordinator 调度伪代码

```python
while wall_clock_elapsed < target_duration:
    collect_worker_results()

    for task in completed_tasks:
        send_to_independent_reviewer(task)

    for task in reviewer_rejected:
        requeue(task, priority="higher")

    close_only_fully_verified_tasks()

    if runnable_tasks < 15:
        launch_audit_miner()

    if runnable_tasks < 8:
        launch_second_audit_miner()

    assign_every_idle_worker()

    if no_p0():
        promote_p1()

    if no_known_tasks():
        start_deep_audit_round()

    update_master_queue()
    update_findings_ledger()
    update_certification_matrix()
    update_resume_state()
```

---

# 27. Worker 任务模板

每项任务必须包含：

```text
Task ID
Priority
Owner
Module
Current evidence
Risk
Exact reproduction
Expected contract
Likely files
Do-not-touch scope
Required tests
Negative tests
Definition of Done
Reviewer
```

Worker 完成时返回：

```text
Root cause
Reproduction
Code changed
Why correct
New regression tests
Existing regression results
Edge cases
Remaining uncertainty
New issues found
Suggested next task
```

---

# 28. Reviewer 必须重新验证

不能只看 Worker 结果。

```text
A. 重跑旧 reproduction
B. 确认修复
C. 自己构造新 counterexample
D. 检查相邻 contract
E. 搜 silent fallback
F. 检 PIT
G. 检 identity
H. 检 API/docs drift
I. targeted regression
J. PASS/REJECT
```

---

# 29. “没活以后继续干什么”

如果前述任务做完，按顺序进入：

```text
Round A：Contract Drift Audit
Round B：Duplicate Authority Audit
Round C：Silent Failure Audit
Round D：Global State Audit
Round E：Identity Closure Audit
Round F：Numerical Stability Audit
Round G：Cross-process Reproducibility
Round H：Mutation Testing
Round I：Performance Regression
Round J：Long-running Soak
```

禁止通过反复跑同样测试、只改注释、只格式化、创建无人调用 abstraction 来“凑 6 小时”。

---

# 30. Task Queue 目标

尽量维持：

```text
20–40 queued
5–15 reproduced
2–4 active（受机器资源限制）
1–3 reviewer
```

一个 Worker 完成后立即发下一项，不需要等整批结束。

---

# 31. Resume 机制

如果环境提前终止，先写：

```text
RESUME_STATE.md
```

其中必须包含：

```text
current wall-clock progress
open P0
open P1
active workers
last completed tasks
review pending
blockers
next 20 runnable tasks
commands/tests needed for resume
```

下一次只读控制文件就可以继续。

---

# 32. 最终 Certification Matrix

至少逐项给：

```text
CERTIFIED
PARTIALLY_VERIFIED
OPEN
BLOCKED
UNKNOWN
```

需要覆盖：

```text
DATA_PIT
DATA_REVISION
DATA_UNIVERSE
DATA_SNAPSHOT_IDENTITY
FACTOR_CANONICAL
FACTOR_NO_FUTURE
OPERATOR_AUTHENTICITY
PANDAS_REFERENCE
POLARS_PARITY
DUCKDB_PARITY
Q_PARITY
BACKEND_REGION
NO_PINGPONG
CACHE_IDENTITY
MATERIALIZATION_ATOMICITY
FACTOR_ASSETS_CONTAMINATION
MODEL_SPLIT
MODEL_PURGE_EMBARGO
MODEL_PREPROCESS_LEAKAGE
EVALUATOR_LABEL
EVALUATOR_METRIC
EVALUATOR_STREAM_BATCH
CLEAN_WHEEL
END_TO_END_IDENTITY
```

---

# 33. 首轮立即分配

```text
ChiefCoordinator
  → 建控制文件、导入本任务书、开始调度

Backend/Q Agent
  → FE-BE-P0-* + Q-P0-*

DataAccess Agent
  → ARCH-P0-* + DA-ID + DA-LAYOUT + DA-BUILD + PIT poison

FactorAssets Agent
  → FA / Campaign / Ledger / MultiFidelity / Plateau / Pareto

Modeling Agent
  → gap_days / purge / embargo / VWAP-to-VWAP /
     FittedTransform bypass / OOF / ModelArtifact

Evaluator Agent
  → quantile tie / NumPy-Numba / squeeze / fastmath /
     batch-stream / VWAP-to-VWAP

Packaging Agent
  → clean-wheel / sys.path / absolute path / E2E identity

IndependentReviewer
  → 持续复核所有 LOCAL_TESTED task
```

---

# 34. 最终判断标准

平台不是“能算出因子”就 production-ready。

真正要求是，同一个：

```text
factor definition
data PIT
snapshot
universe
calendar
parameters
label contract
evaluation contract
```

在：

```text
Pandas
Polars
DuckDB
q
不同进程
不同服务器
重新运行
```

都得到可解释、可追踪、可复现且在契约容差内一致的结果。

任何无法回答下面问题的模块都继续留在任务池：

```text
这个数据当时真的已经知道了吗？
这个股票当时真的属于 universe 吗？
为什么这个 cache 会命中？
这个 q/Polars/DuckDB 结果为什么等价于 Pandas oracle？
这个模型有没有通过 preprocessing/selection 间接看到未来？
这个 RankIC 的 tie policy 到底是什么？
这个收益标签是不是严格 VWAP-to-VWAP？
这个 artifact 能不能从 identity 完整复现？
这个错误发生时系统会 hard fail，还是偷偷给空结果？
```

优先级始终：

```text
Correctness
→ PIT
→ Identity
→ Reproducibility
→ Leakage Prevention
→ Backend Parity
→ Failure Safety
→ Performance
```

在这些基础 contract 没有封死之前，不要优先继续扩大 operator 数量和模型数量。
