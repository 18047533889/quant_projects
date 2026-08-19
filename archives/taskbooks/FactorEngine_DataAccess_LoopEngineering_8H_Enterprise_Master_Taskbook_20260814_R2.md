# FactorEngine + DataAccess Loop Engineering 8H 多 Agent 企业级自治整改总纲（R2）
## Latest Main Re-Audit + Continuous Certification + Failed-Agent Recovery Taskbook
### Current Snapshot: `d78ed761b7d098d27e3acf916f3961d75096b3b3`
### Previous Mother Snapshot: `88237c561d641f7fc0d1351642a52fec88e6c558`
### Date: 2026-08-14

---

# 0. 给 Loop Engineering ChiefCoordinator 的总指令

你不是在完成一轮普通“修 bug”任务。你要把以 FactorEngine + DataAccess 为核心的量化研究基础设施持续推进到：

```text
ZERO KNOWN CORRECTNESS DEFECTS
+ ZERO KNOWN PIT / LOOKAHEAD DEFECTS
+ ZERO KNOWN BACKEND SEMANTIC DIVERGENCES
+ ZERO KNOWN SILENT FALLBACK / SILENT CORRUPTION
+ ALL PRODUCTION OPERATORS CERTIFIED
+ ALL SELECTABLE PHYSICAL IMPLEMENTATION PATHS CERTIFIED
+ END-TO-END PIPELINE CERTIFIED
+ PERFORMANCE REGRESSION CONTROLLED
+ FAULT / CONCURRENCY / RESTART BEHAVIOR CERTIFIED
+ CLEAN-WHEEL / CI / EVIDENCE CLOSED
```

任何软件系统都不可能数学上证明“以后永远不会有任何 bug”。因此终态不是 `NO POSSIBLE BUGS`，而是：

```text
ZERO KNOWN DEFECTS
+ 100% 当前生产面认证覆盖
+ 持续 adversarial audit miner
+ UNKNOWN 永远不能伪装成 PASS
```

只要还有 `UNKNOWN / UNTESTED / PARTIAL / TODO / HEURISTIC / FALLBACK / UNVERIFIED / NOT PROVEN`，就不得宣称整个系统 `PRODUCTION READY`。

---

# 1. 当前 main 的事实基线（R2：本节覆盖旧快照口径）

本任务书当前绑定：

```text
HEAD:
d78ed761b7d098d27e3acf916f3961d75096b3b3

message:
Sync Q-backend P0 gates, modeling OOS/gap enforcement,
and return_decomp ABI fixes.
```

上一版母文档绑定：

```text
88237c561d641f7fc0d1351642a52fec88e6c558
```

从上一版到当前 HEAD，仓库继续合入了：

```text
FactorEngine:
  ts_corr / ts_cov / ts_std / ts_var parity 修复
  return_decomp ABI 修补
  Q capability/evidence/adapter/backend 多处修改
  Q physical implementation registry 雏形
  Modeling authority / OOS / gap 检查
  多份 Loop Engineering 状态与完成报告

QuantEvaluator:
  top-bottom spread 方向修复
  quantile tie policy / Numba parity 相关改动

Modeling:
  split/gap/OOS transform enforcement 改动

FactorAssets:
  仍存在 DataAccess adapter 未闭环

CI:
  当前 HEAD 未发现新的 root production gate
  当前 HEAD 没有可用 GitHub Actions/combined-status 证据
```

**最重要的新事实：**

这一轮 11 个后台 Agent 中大量 Agent 报：

```text
API Error: Content block not found
```

但“Agent 报失败”并不等于“它没有留下源码修改”。

当前 HEAD 已经出现：

```text
失败 Agent 的半成品 source 被同步进 main
```

最严重例子：

```text
factor_engine/backend/q_backend/q_executor.py
```

当前被截成：

```text
imports
+
4 个硬编码 "PASS" 常量
```

而完整的：

```text
QExecutor
QExecutionFallbackPolicy
get_q_executor
execute_region
resident fan-in
workspace lifecycle
```

等实现已经丢失。

与此同时：

```text
q_backend.py
```

仍然 import：

```text
QExecutionFallbackPolicy
get_q_executor
```

因此 current HEAD 的 Q backend **不是“尚未优化完”这么简单，而是存在明显 import/runtime 断裂风险。**

所以 R2 的第一原则改为：

```text
RECOVER REPOSITORY INTEGRITY FIRST
THEN FIX CORRECTNESS
THEN CERTIFY
THEN OPTIMIZE PERFORMANCE
THEN EXPAND CAPABILITY
```

不能在 import-broken / half-written HEAD 上继续堆新功能。

本 R2 文档保留上一版所有设计与问题背景；若旧章节状态描述与本节及后续 `R2 CURRENT-HEAD OVERRIDE` 冲突：

```text
R2 CURRENT-HEAD OVERRIDE 优先。
```

---

# 2. 从 canonical certification 升级为 Physical Implementation Certification

以后禁止只记录：

```text
ts_skew = parity verified
```

必须记录：

```text
canonical = ts_skew

physical implementations:
  pandas_reference_kernel
  polars_long_expr_emitter
  polars_panel_registered_native
  polars_delegate
  duckdb_sql_emitter
  clickhouse_sql_emitter
  q_lowering/runtime
  optional_numba_kernel
```

真正认证对象：

```text
PhysicalImplementationID
=
canonical
+ backend
+ execution_kind
+ implementation_source_hash
+ emitter/kernel identity
+ parameter_domain_hash
+ semantic_contract_hash
```

一个 canonical 可以有多条物理实现，每一条 Planner/Router 有可能选到的路径都必须独立认证。

`Polars Long parity green` 绝不能推出 `所有 Polars implementation 都正确`。

---

# 3. 当前 Loop Engineering 本身需要整改（R2：API 失败恢复版）

原来的：

```text
Dimension Analysis
→ Audit
→ Implementation
→ Validation
→ Consolidation
```

框架保留。

但最新一轮约 2 小时执行暴露了一个更严重的系统性问题：

```text
高并发 Agent API 失败
+
失败 Agent 可留下 partial writes
+
Coordinator 最终统一 sync
=
半成品进入 main
```

今晚 8 小时任务必须先整改 Loop Engineering 自身，否则增加 Agent 数量只会放大损坏速度。

## LOOP-P0-001：API 失败不是“任务未执行”，必须审计工作区残留

任何 Agent 状态：

```text
FAILED
API ERROR
CONTENT BLOCK NOT FOUND
CANCELLED
TIMEOUT
```

Coordinator 必须立即执行：

```text
1. 获取该 Agent base_sha
2. 列出它声明的 write_scope
3. 比较 base_sha → 当前 worktree
4. 检测异常文件尺寸变化
5. python -m compileall
6. import touched modules
7. 检查 exported symbols
8. 检查是否存在硬编码 PASS / TODO / truncated class
9. 不通过则 quarantine/revert/repair
```

禁止：

```text
Agent failed
→ 假设它没改东西
→ git add -A
→ sync main
```

## LOOP-P0-002：Writer Agent 必须隔离 worktree/branch

首选：

```text
one writer Agent
=
one git worktree
+
one temporary branch
```

Agent 只在自己的 worktree 写。

完成后必须交：

```text
AgentChangeManifest
base_sha
branch
touched_files
intended_symbols
tests_run
test_results
new_commit
known_unknowns
```

ChiefCoordinator 只 cherry-pick：

```text
已通过 local acceptance 的 commit
```

Agent API 失败：

```text
默认丢弃该 worktree 的未认证修改
```

除非 Recovery Agent 对其独立验证通过。

如果环境不能创建 worktree：

```text
退化为 WRITE_LEASES + 每文件独占写锁
```

但绝不能多人共享写同一个 source tree。

## LOOP-P0-003：API 不稳定时降低并发，不是继续开 11 个 writer

若最近 30 分钟：

```text
API error rate > 20%
```

则切换：

```text
SAFE SERIAL MODE
```

配置：

```text
writer_agents <= 1
read_only_auditors <= 2
reviewer <= 1
```

若：

```text
API error rate <= 10%
```

可升至：

```text
writer_agents <= 2
```

今晚不追求“同时开最多 Agent”，追求：

```text
每个小时可验证关闭多少真实缺陷。
```

## LOOP-P0-004：每个微任务必须原子 checkpoint

一个 Writer Agent 每次只领取：

```text
一个 issue
或
一个 tightly-coupled family
```

完成：

```text
edit
→ targeted tests
→ import smoke
→ commit
→ manifest
```

后才领取下一个。

禁止长达 1–2 小时：

```text
改 30 个文件
最后一起提交
```

因为 API 断开会产生不可恢复半状态。

## LOOP-P0-005：任何 commit 合入后的 Mandatory Integrity Gate

每次 cherry-pick/merge 后：

```text
python -m compileall relevant_packages
import factor_engine
import data_access
import modeling
import factor_assets
import quant_evaluator
load_all operator registry
build minimal DataAccess request
compile minimal FactorEngine factor
run 1 pandas reference
run 1 polars path
```

Q 包单独：

```text
import backend.q_backend.q_backend
```

无 q runtime 也必须能正常 import。

任一失败：

```text
停止后续 wave
优先修 repository integrity
```

## LOOP-P0-006：禁止“硬编码 PASS = 证据”

类似：

```python
Q_FAN_IN_INPUT_PRESERVATION = "PASS"
```

不构成任何证据。

Hard Gate 必须：

```text
运行验证逻辑
或
读取绑定当前 HEAD 的不可变 evidence artifact
```

禁止测试只 assert：

```text
CONSTANT == "PASS"
```

## LOOP-P0-007：Validator 禁止自证

Writer Agent：

```text
最多声明 IMPLEMENTED_LOCAL
```

Independent Reviewer/Red-Team：

```text
才能给 REDTEAM_GREEN
```

Evidence Agent：

```text
才能生成 CERTIFIED artifact
```

ChiefCoordinator：

```text
才有权更新 production truth matrix。
```

## LOOP-P0-008：任务不能因局部 queue 暂空提前结束

今晚的任务预算是：

```text
8 小时自治开发窗口
```

某 wave 提前完成：

```text
自动进入下一 audit dimension
```

若所有显式问题暂时关闭：

```text
启动 AuditMiner / Fuzz / PIT poison / fault / perf / clean-wheel
```

不能提前收尾。

## LOOP-P0-009：状态数字全部机器生成

禁止再出现：

```text
实际修 4 个
报告写成功标准 >=5 已 PASS
```

以下全部由 ledger 自动统计：

```text
issues_discovered
issues_reproduced
issues_fixed
issues_certified
physical_impl_certified
operator_coverage
PIT_coverage
backend_coverage
tests
benchmarks
```

## LOOP-P0-010：任何 FINAL/COMPLETE 文档只是历史记录

仓库里已经有大量：

```text
FINAL
COMPLETION_REPORT
PRODUCTION_READY
ALL_PASS
```

文档。

它们不得成为 runtime truth。

唯一 truth：

```text
current HEAD
+
machine ledger
+
current evidence
+
current CI
```

---

# 4. 本轮明确 P0：FactorEngine `ridge`

文件：`factor_engine/cleaned_operators/common/statistics.py`

## FE-RIDGE-P0-001：双输入 Ridge 实际未实现

当前 `_ridge_trend()` 只有 `y is None` 路径有计算；只要传入第二输入 y，最终直接 `return np.nan`。这意味着 API 声称支持双输入 Ridge，但实际输出全 NaN。

## FE-RIDGE-P0-002：位置参数 off-by-one

当前对 `(y, window, alpha)` 解析时，alpha 应在 `args[2]`，代码却只有 `len(args)>3` 时才读 `args[3]`。

## FE-RIDGE-P0-003：broad except 静默全列 NaN

整列 rolling 捕获 `Exception` 后直接设 NaN，会把 API/shape/dtype/实现 bug 都隐藏成“没有信号”。

### Ridge 整改

禁止继续靠 `*args` 猜语义。建议拆成：

```text
ts_ridge_trend(x, window, alpha)
ts_ridge_beta(y, x, window, alpha, fit_intercept)
```

旧 `ridge` 只做兼容 alias/dispatcher。

双输入 rolling Ridge 明确定义：

\[
\hat\beta_t
=
\arg\min_\beta
\sum_{i=t-W+1}^{t}(y_i-\alpha_0-x_i^\top\beta)^2
+
\lambda\|\beta\|_2^2
\]

并冻结 intercept 正则化、min_periods、finite-pair policy、constant-x、alpha domain、输出含义。用 closed-form/sklearn oracle + golden + property test 认证。

---

# 5. 本轮新发现：Polars Native 的“窗口局部统计”系统性风险

文件：`factor_engine/cleaned_operators/common/polars_ts_stats.py`

危险模式：

```text
rolling statistic_t
→ pointwise(x_t, statistic_t)
→ 再 rolling
```

对很多“当前窗口 W_t 的整体统计量”并不等价。

## POL-WINDOW-P0-001：`ts_skew`

当前近似：

```python
mean = val.rolling_mean(w)
std = val.rolling_std(w)
m3 = ((val - mean) ** 3).rolling_mean(w)
skew = m3 / std**3
```

真正当前窗口三阶中心矩：

\[
m_{3,t}=\frac{1}{n_t}\sum_{i\in W_t}(x_i-\bar x_t)^3
\]

所有历史样本都必须减当前 `mean_t`。当前实现却让每个历史 i 减自己的 `mean_i`，数学对象不同。

## POL-WINDOW-P0-002：`ts_trimmed_mean`

当前先用 rolling quantile 对每个时点裁剪，再 rolling mean。这更像时间变化阈值下的 winsorized mean，不是“当前窗口删掉两端样本后的 trimmed mean”。

## POL-WINDOW-P0-003：`ts_qn_scale`

当前 `(Q75-Q25)/1.349` 是 IQR-normalized scale，不是 Rousseeuw-Croux Qn。要么改 canonical 为 `ts_iqr_scale`，要么真正实现 Qn。

## POL-WINDOW-P0-004：`ts_expected_shortfall`

当前先按每个历史时点自己的 VaR_i 选中 x_i，再 rolling mean。真正 ES_t 应使用当前窗口统一的 VaR_t：

\[
VaR_t(\alpha)=Q_\alpha(W_t)
\]

\[
ES_t(\alpha)=E[x\mid x\le VaR_t(\alpha),x\in W_t]
\]

## POL-WINDOW-P0-005：全族扫同构错误

扫描 `rolling_quantile/rolling_mean/rolling_std` 后接 pointwise mask/clip，再二次 rolling 的所有算子。重点：expected_shortfall_asymmetry、tail_mean、tail conditional moments、custom skew/kurt、trimmed/winsorized families。

---

# 6. 为什么现有 parity test 可能抓不到

新增 statistical parity 测试走的是 `polars_long`，并检查 `used_polars_long_path=True`。这认证的是 Polars Long Expr Emitter，不自动认证 `common/polars_ts_stats.py` 的 registered Polars classes。

因此必须把 parity matrix 从：

```text
canonical × backend
```

升级成：

```text
canonical × PhysicalImplementationID
```

---

# 7. Polars Cross-Sectional 全族 finite policy 重审

Pandas reference 的 `CrossSectionSampleMask` 明确用 `np.isfinite`，所以 NaN、+Inf、-Inf 都不计入截面统计。Polars native long transform 当前明显处理 NaN→NULL，但必须逐算子证明 Inf 也一致。

重点：

```text
cs_rank
cs_zscore
cs_demean
cs_scale
cs_quantile
cs_bucket
group equivalents
```

必须覆盖：

```text
[1,1,2,+Inf,-Inf,NaN]
constant cross-section
all-null
singleton
ties
```

`cs_zscore` 在 std=0 时到底 zero-fill 还是 null，必须由 canonical semantic authority 决定，不能照 ts_zscore 猜。


---

# 8. q/K 仍是未闭环的旁路 backend

最新 4 commits 没有实际修改 q backend 核心，因此上一轮 q P0 不得自动关闭。

## Q-P0-001：`production_safe` 定义仍自相矛盾

`QCapabilityEvidence.production_safe` 要求 declared/lowering/compile/runtime/parity 全 PASS；但 `get_q_production_safe_ops()` 当前只看 declared + lowering，compile/runtime/parity 仍是 False/TODO。

修复：

```text
get_q_production_safe_ops()
必须且只能读取
ev.production_safe
```

## Q-P0-002：`production_ready` 仍只看 lowering gap

当前：

```text
native_without_lowering_count == 0
→ production_ready=True
```

最多只能叫：

```text
DECLARATION_LOWERING_CLOSED
```

不能叫 production ready。

## Q-P0-003：q 尚未进入主 BackendName / BackendRouter

当前主 `operator_capability.BackendName` 与 `BackendRouter` 都没有 q_kdb。q 仍是平行系统，不是统一 planner/routing authority 的一部分。

## Q-P0-004：QBackend 仍自己 whole tree → one region

文档说 PhysicalPlanner 是唯一 authority，代码却明确写 simplified whole-tree region。生产入口必须改成消费 `PhysicalBackendRegion`，QBackend 自己不能重新决定 region。

## Q-P0-005：missing base data 仍返回 empty DataFrame

必须 typed `DataUnavailableError`，不能让 empty panel 流进计算。

## Q-P0-006：malformed q output 仍猜第一列

建立 `QOutputContract`，校验 timestamp/instrument/value/dtype/grain/sort/duplicate/nullability。

## Q-P0-007：runtime exception 仍可能 fallback Pandas

生产 fallback 只能发生在 planning admission 之前。执行中 compiler/runtime/schema/PIT/semantic error 一律 hard fail。

## Q-P0-008：resident fan-in 明确丢输入

当前 `execute_batch_regions()` 在上一 region 输出被下一 region 使用时，会把 `current_input` 整体替换为单个 resident handle，导致 base input 或第二 predecessor 丢失。必须支持任意 DAG fan-in。

必测：

```text
A ─┐
   ├→ C
B ─┘

A→B→D
A→C→D
```

## Q-P0-009：q global workspace 并发隔离

建立：

```text
ExecutionNamespace(run_id, region_id)
ConnectionGeneration
ProcessGeneration
SessionGeneration
execution lock / pool
resident refcount/liveness
failure cleanup
```

## Q-P0-010：carry-forward compiler semantics

逐项重新证明：

```text
ts_beta
ts_corr
ts_cov
lag(periods>1)
ts_median
ts_quantile
wma
clip
rank/tie
round negative half values
backend-owned default window/span
```

禁止“能生成 q 字符串”就算 implementation。

---

# 9. DataAccess 当前版本/Build Identity 是 P0

仓库终审报告与 current main 代码不一致。

## DA-BUILD-P0-001：pyproject 仍静态版本

当前实际：

```toml
version = "0.10.2"
```

不是终审报告里说的 dynamic SCM version。

## DA-BUILD-P0-002：tracked `_build_info.py` 又写另一版本

当前：

```text
0.11.0.dev0+untagged
commit_id=None
```

出现两个版本事实：

```text
0.10.2
0.11.0.dev0+untagged
```

## DA-BUILD-P0-003：`_build_meta.py` 仍 runtime git

当前依然是：

```text
env
→ git rev-parse HEAD
→ None
```

不是报告声称的 build-time frozen SHA。

## DA-BUILD-P0-004：package `__init__` 仍接旧 `_build_meta`

因此 clean wheel/container/无 `.git` 环境的 provenance 仍未闭环。

## DA-BUILD-P0-005：又存在 `core/build_metadata.py`

DataAccess 至少出现：

```text
pyproject version
_build_info.py
_build_meta.py
core/build_metadata.py
```

四套 build/version 事实源。

必须收敛：

```text
ONE_BUILD_IDENTITY_AUTHORITY
```

## DA-BUILD-P0-006：`"unknown"` 不能算合法 production identity

所有：

```text
None
"unknown"
"dev"
"local"
"untagged"
```

要 typed 状态，生产 gate 明确 fail。

---

# 10. DataAccess Canonical Identity 尚未真正全仓唯一

`core/identity_encoder.py` 已经有比较严格的 CanonicalIdentityEncoder，但 `read/read_contract.py` correctness-critical snapshot 路径仍维护第二套 `canonicalize_params/_jsonable`。

## DA-ID-P0-001：repr fallback

未知对象不能：

```text
repr(value)
```

进入 snapshot/cache identity。

## DA-ID-P0-002：`str(key)` typed collision

`1` 与 `"1"` 不应意外变同 key。

## DA-ID-P0-003：list/tuple 语义

必须由统一 encoder 明确类型是否保留，不可各模块各自折叠。

## DA-ID-P0-004：禁止 `json.dumps(default=str)`

correctness identity 一律 strict serializer。

## DA-ID-P0-005：critical hash 位宽统一

当前 read_contract 中可见：

```text
schema_hash      64 bit
manifest_hash    64 bit
snapshot_id      96 bit
```

与“correctness identity >=128 bit”原则不一致。

内部全部用 full SHA-256；短 hash 只用于 UI/log display。

## DA-ID-P1-006：datetime instant canonicalization

等价 instant：

```text
2026-08-14 10:00 +08
2026-08-14 02:00 UTC
```

如果语义是同一时刻，应 canonicalize 到 UTC ns。

另设明确类型表示：

```text
SessionLocalTimestamp
TradingSessionLabel
```

避免把“交易所本地时刻”与“绝对时刻”混成一类。

---

# 11. DataAccess remote snapshot / credentials

低层 `_remote_object_meta()` 是 best-effort，不等于整个 production pipeline 一定 fail-open；外层还有 SourceSnapshotResolver / SnapshotVerifier。所以这里的任务是**证明 production strict path 最终 fail-closed**。

必须 fault test：

```text
HEAD timeout
403
404
500
credential expired
credential rotates
same key overwritten
ETag changes
version_id changes
wildcard unresolved
LIST partial
network partition
negative cache hit
```

## DA-SNAP-P0-001：credential_generation 仍 TODO

remote metadata cache key 当前 credential_generation 还是 None。真正 credential rotation 必须换 cache namespace。

## DA-SNAP-P0-002：typed remote errors

不能把所有 remote error 都收敛成 None。至少：

```text
NOT_FOUND
AUTH_FAILURE
TIMEOUT
NETWORK
RATE_LIMIT
SERVER
MALFORMED_RESPONSE
```

外层 production verifier 必须知道为何无法证明 snapshot。

## DA-SNAP-P0-003：unresolved remote identity 不能进入 correctness cache

如果 exact object 没有可证明 version/etag/manifest generation，production cached result 不得使用它作为稳定 snapshot。

---

# 12. DataAccess CompiledDataRequest “冻结”还有 live object 风险

当前 `_immutable/_deep_freeze`：

```text
deepcopy fail
→ copy fail
→ return original value
```

这与“CompiledDataRequest 是不可变独立 IR”的承诺冲突。

## DA-REQ-P0-001

production compiler 对无法 canonicalize/freeze 的任意 object：

```text
raise UnsupportedRequestValueError
```

禁止回传原对象。

## DA-REQ-P0-002

Compiled IR 只允许明确 typed nodes：

```text
PredicateIR
JoinSpec
AggregationSpec
TransformSpec
SourceParamSpec
FieldParamSpec
```

## DA-REQ-P0-003：mutation-after-plan property test

构造拒绝 deepcopy 的可变 custom object，plan 后修改原对象，验证 explain/execute/snapshot identity 完全不变或 compile 直接拒绝。

---

# 13. `fail_if_changed` 语义需分级

当前 `pin` 无权威 manifest 会 hard fail；但 `fail_if_changed` 在 plan/execute 都无 manifest 时可能继续。

建议显式分：

```text
latest
best_effort_fail_if_changed
verified_fail_if_changed
pin
```

其中 production correctness path 只能用：

```text
verified_fail_if_changed
pin
```

规则：

```text
无法证明 == FAIL
```

---

# 14. `ReadLineage.instrument_filter` 默认值审计

当前默认是：

```python
instrument_filter = ()
```

注释却定义：

```text
None = 不限制/全市场
()   = 空股票池
```

必须扫描所有构造者。如果不传 filter 表示“不限制”，默认就应是 None。否则 lineage provenance 会虚假记录 empty universe。

---

# 15. Local file snapshot fidelity

本地 FileVersion 主要依赖：

```text
path
size
mtime_ns
```

按 dataset policy 分级：

```text
IMMUTABLE_GENERATION_MANIFEST
CONTENT_HASH
SIZE_MTIME_BEST_EFFORT
```

production-critical 数据优先 generation manifest/content digest。

加入：

```text
replace file preserving size
restore mtime
partial overwrite
atomic rename
```

故障测试，明确 best-effort 与 cryptographic snapshot 的边界。

---

# 16. FactorEngine Performance：不要把“统一 to_pandas helper”误认为性能闭环

最新 P1-05 把几处手写 `.to_pandas().set_index()` 改为共享 helper，代码质量有改善，但报告自身也写 performance impact neutral。

真正要优化的是：

```text
backend switches
materialization count
collect count
bytes transferred
boundary time
full-panel conversion
per-factor final conversion
CSE native reuse
DAG fusion ratio
SQL query count
q upload/download
peak RSS
```

---

# 17. Unified Backend Region Planner 是最终架构

```text
Canonical DAG
      ↓
Semantic/PIT Contracts
      ↓
Capability Annotation
      ↓
PhysicalBackendRegionPlanner
      ↓
Pandas Region | Polars Region | DuckDB Region | q Region
      ↓
conversion only at region boundary
      ↓
FactorMatrixArtifact / Canonical Result
```

Planner 不能 operator-by-operator greedy。

总成本模型：

\[
TotalCost
=
\sum_r ComputeCost(r)
+
\sum_b TransferCost(b)
+
MaterializationCost
+
SerializationCost
+
MemoryPressureCost
+
ContentionCost
\]

同 backend 连续节点应尽量融合为 maximal compatible region。

---

# 18. Native intermediate 与 CSE

Region 内部保持 native：

```text
Pandas    → ndarray/DataFrame
Polars    → LazyFrame/Expr
DuckDB    → Relation/Arrow/SQL pipeline
q         → q table/vector
```

CSE shared subtree：

```text
如果所有 consumers 同 backend
→ 保持 native shared node

只有跨 region consumer
→ 才 materialize/convert
```

---

# 19. 不应永远每个 factor 都返回 pd.Series

兼容 API 可以保留：

```text
single factor → pd.Series
```

但批量生产应新增：

```text
FactorMatrixArtifact
```

可承载：

```text
Arrow table
Polars long/wide
Parquet block
memory-mapped matrix
DuckDB relation
q-resident handle
```

Evaluator / FactorAssembly / Modeling 优先消费 batch-native artifact，避免几万因子逐个 MultiIndex Series 构造。

---

# 20. Production Environment Toggle 不能偷偷改变语义

例如：

```text
FACTOR_ENGINE_POLARS_LONG_ALIGN_UNIVERSE
```

如果环境变量会改变：

```text
输出 index
缺失行
universe alignment
```

它就不是单纯性能开关，而是语义配置。

必须：

```text
进入 ExecutionConfigIdentity
```

或者取消环境隐式开关，改成显式 typed contract。

规则：

```text
任何会改变结果值/shape/index/grain/PIT 的 env var
必须进入 semantic/execution identity。
```

---

# 21. 当前主 Backend Capability 仍有几个 fail-soft 点

## BE-P0-001：q 不在 BackendName

统一 BackendKind / BackendName / Router / Planner。

## BE-P0-002：`_emitter_source_hash()` 不能返回 `"unknown"`

production capability identity 无源码 hash → CapabilityInfrastructureError。

## BE-P0-003：production signature import error 不能 `pass`

registry/signature infrastructure broken ≠ 没有 signature。

## BE-P0-004：SQL emitter exception 不能全部变 `supported=False`

区分：

```text
UNSUPPORTED
PARAM_DOMAIN_REJECTED
COMPILER_BUG
REGISTRY_FAILURE
INFRA_FAILURE
```

## BE-P0-005：production evidence loader failure 也必须可观察

不能把证据文件损坏悄悄降成 implemented 而不发高等级 diagnostic。

---

# 22. Operator Certification 必须覆盖所有生产算子，不只是 33 个高优先级

当前 BackendParityFindings 只抽 33 个 high-priority operators。

最终目标必须自动枚举：

```text
OperatorRegistry.list_canonical()
→ production surface
→ DirectUse surface
→ backend-selectable physical implementations
```

对**每一个 production canonical**生成 certification record。

不要人工维护 operator list。

---

# 23. 单算子生产认证标准

每个 production canonical 必须有以下机器可读字段：

```text
canonical
semantic_version
semantic_hash
economic_role
input_slots
input_types
input_units
input_grain
output_type
output_unit
output_grain
timing_kind
PIT_contract
missing_policy
NaN_policy
Inf_policy
tie_policy
ddof
window_alignment
min_periods
warmup
parameter_schema
parameter_roles
parameter_domains
active_when
searchability
reference_implementation
physical_implementations
golden_cases
property_cases
fuzz_seed_set
backend_parity
stream_batch_parity
shard_parity
checkpoint_parity
performance_profile
evidence_hash
production_status
```

---

# 24. Operator 认证状态机

每个 operator / implementation 必须经过：

```text
DISCOVERED
→ SPEC_DEFINED
→ ORACLE_GREEN
→ IMPLEMENTATION_REPRODUCED
→ EDGE_GREEN
→ PROPERTY_GREEN
→ FUZZ_GREEN
→ BACKEND_GREEN
→ DAG_GREEN
→ PIT_GREEN
→ STREAM_GREEN
→ PERF_GREEN
→ REDTEAM_GREEN
→ CERTIFIED
```

任意一步代码/semantic hash/parameter-domain 变化：

```text
下游状态自动失效
```

禁止手工保留旧 CERTIFIED。



---

# 25. 多 Agent 组织架构

不要让所有 Agent 都叫“修代码 Agent”。

必须有职责隔离。

---

## Agent-00：ChiefCoordinator

职责：

```text
唯一 Task Ledger authority
唯一 Agent lease authority
唯一 merge/revalidation scheduler
唯一 production status publisher
```

禁止：

```text
直接大规模修改业务 source
自己实现自己 review
根据单个 Agent 文案改成 PASS
```

每 20–30 分钟：

```text
refresh HEAD
compare base SHA
invalidate stale findings
收集 Agent result
检测 write conflict
更新 queue
触发 broad tests
重分配下一轮
```

---

## Agent-01：Operator Truth Inventory

任务：

```text
枚举所有 canonical
aliases
DirectUse status
research_only
production target
ParamSpec
physical backend slots
source file/class
physical spec
evidence
cold-start usages
```

输出：

```text
operator_truth_matrix.json
operator_truth_matrix.parquet
```

它不修数学实现，只负责“现在到底有什么”。

---

## Agent-02：Semantic Oracle — TS / Rolling / Statistics

覆盖：

```text
ts_*
rolling stats
moments
quantiles
tail
risk
decay
correlation/covariance
drawdown
distance
trend
```

必须重点查本轮发现的：

```text
window-local statistic trap
```

---

## Agent-03：Semantic Oracle — Cross-Section / Group

覆盖：

```text
cs_*
group_*
row_*
neutralize
rank
bucket
winsor
zscore
```

重点：

```text
finite sample policy
ties
group NULL
singleton
asset permutation
```

---

## Agent-04：Semantic Oracle — Technical Indicators

覆盖：

```text
EMA
SMA
WMA
DEMA
TEMA
HMA
RSI
RSI_WILDER
MACD
Bollinger
ATR
ADX/DMI/DX
Keltner
Donchian
Ichimoku
PSAR
Supertrend
candlestick
breakout
```

不要只验证“公式像不像”。

要冻结：

```text
seed/warmup
adjust
ignore_nulls
Wilder smoothing
window rounding
TA-Lib compatibility contract
```

---

## Agent-05：Semantic Oracle — Regression / Model / Stateful

覆盖：

```text
ridge
OLS
beta
residual
r_squared
rolling regression
forecast error
Kalman
filters
recursive state
regime
model operators
```

重点：

```text
fit vs score
state update order
no future label
checkpoint/resume
```

---

## Agent-06：Semantic Oracle — Intraday / Microstructure / Minute→Daily

覆盖：

```text
intraday operators
minute aggregates
VWAP
OHLC aggregation
realized volatility
volume profile
microstructure
time-of-day
session state
```

重点：

```text
交易日边界
午休
集合竞价
停牌
涨跌停
缺分钟
重复分钟
timezone
session calendar
```

---

## Agent-07：Semantic Oracle — Fundamental / Event / Financial PIT

覆盖：

```text
financial
fiscal
report
announcement
event
analyst
index membership
industry
holdings
fund flow
```

这里不是普通 numeric parity。

首要：

```text
PIT
revision vintage
available_at
announcement timestamp
restatement
calendar
currency/accounting basis
```

---

## Agent-08：Pandas Reference Authority

职责：

```text
让 Pandas reference 真正符合 canonical semantic spec
```

不能假设：

```text
Pandas = 真理
```

先有数学/业务 oracle，再决定 reference。

它必须接受 Semantic Oracle Agent 的 spec。

---

## Agent-09：Polars Long/Expr Implementation

只负责：

```text
polars_expr_emitter
polars_long
LazyFrame/Expr
fusion
```

不碰 registered panel-native classes。

---

## Agent-10：Polars Registered Native/Panel

只负责：

```text
common/polars_*
polars_native/*
panel Polars implementation
```

防止：

```text
Polars Long test green
→ 错把 Panel native 当 certified
```

---

## Agent-11：DuckDB/SQL

覆盖：

```text
DuckDB SQL emitter
partial subtree pushdown
batch SQL
null/NaN/Inf
window SQL
parameter-domain
real SQL proof
```

需要确认：

```text
真执行 DuckDB
不是 Python UDF
不是 fallback
```

---

## Agent-12：q/K

覆盖：

```text
capability truth
lowering registry
runtime
PyKX type adapter
resident tables
DAG fan-in
thread/process safety
null/time
parity
performance
```

q 未闭环前：

```text
NOT_PRODUCTION_CERTIFIED
```

---

## Agent-13：Backend Region Planner / DAG

只负责：

```text
PhysicalBackendRegion
region partition
boundary cost
fusion
CSE
native sharing
fallback policy
planner telemetry
```

它不能改 operator semantics。

---

## Agent-14：DataAccess PIT / Semantic Catalog

覆盖：

```text
DataRequest
SemanticFieldCatalog
available_at
revision vintage
universe
calendar
join policy
pit_asof
unit normalization
```

---

## Agent-15：DataAccess Snapshot / Identity / Build

覆盖：

```text
CanonicalIdentityEncoder
DataSnapshot
ExecutionBuildIdentity
lineage
file manifest
remote object identity
package build provenance
cache identity
```

优先修本轮 DataAccess P0。

---

## Agent-16：DataAccess Storage / Transaction / Fault

覆盖：

```text
read/write
publish
upsert
mutation lock
manifest generation
atomicity
concurrent read/write
crash
disk-full
rename failure
process restart
```

---

## Agent-17：Differential / Property / Fuzz Lab

不能改 source。

只生成：

```text
golden cases
metamorphic tests
randomized differential tests
Hypothesis/property cases
adversarial panels
```

它是最重要的独立 bug miner 之一。

---

## Agent-18：End-to-End Integration

负责：

```text
DataAccess
→ FactorEngine
→ filter/preprocess
→ FactorAssembly
→ Modeling
→ Evaluator
```

验证：

```text
PIT identity
FactorArtifact
FeatureSetArtifact
ModelArtifact
evaluation label
VWAP-to-VWAP
```

不能只单包测试。

---

## Agent-19：Performance / Resource Engineering

只有 correctness certification 后才优化。

覆盖：

```text
wall time
RSS
bytes scanned
conversion
query count
fusion
CSE
cache
thread/process
q transfer
```

---

## Agent-20：Evidence / CI / Clean Wheel

负责：

```text
evidence generation
implementation closure hash
CI gates
wheel inventory
clean env
package dependencies
production reports
```

---

## Agent-21：Independent Red-Team Reviewer

任何 source 实现 Agent 都不能 review 自己。

Red Team 任务：

```text
假设修复仍是错的
找最小反例
找边界
找旁路 API
找 stale evidence
找另一个 physical path
找 silent fallback
找 performance regression
```

它只有：

```text
REJECT
REWORK
REDTEAM_GREEN
```

不能直接替 Coordinator 宣布整个模块 production-ready。

---

## Agent-22：AuditMiner Continuous

当 queue 变少时自动扩大搜索。

持续扫描：

```text
except Exception
pass
TODO
FIXME
NotImplemented
return None
return []
return {}
return 0
return np.nan
warnings.warn
fallback
default=str
repr(
str(
astype(int)
int(
float(
squeeze
fillna("")
date.today()
datetime.utcnow()
global singleton
os.environ
sys.path
/home/
production_ready
research_only
supported=True
allow_unverified
```

并且做 AST/semantic pattern mining，而不是简单 grep。

---

# 26. 文件写权限与并发冲突规则

长时间多 Agent 最大风险之一是互相覆盖。

建立：

```text
WRITE_LEASES.json
```

字段：

```text
path / module
owner_agent
base_sha
lease_started_at
lease_generation
reason
```

规则：

```text
同一个 source file 同时只能一个 writer
reviewer 永远 read-only
shared authority file 由专门 owner 管
其他 Agent 只能提交 change request
```

禁止：

```text
两个 Agent 同时改 registry.py
两个 Agent 同时重写 statistics.py
Agent A 用旧 HEAD 覆盖 Agent B 新修改
```

每个 implementation result 必须报告：

```text
base_sha
current_head_seen
files touched
semantic contracts affected
certifications invalidated
```

如果 HEAD 已移动：

```text
重新 diff
重新 review
重新 test
```

---

# 27. Task Lifecycle

所有 issue 统一状态：

```text
DISCOVERED
→ REPRODUCED
→ SPECIFIED
→ ASSIGNED
→ IMPLEMENTED_LOCAL
→ FAMILY_GREEN
→ CROSS_IMPL_GREEN
→ DAG_GREEN
→ PIT_GREEN
→ PERF_GREEN
→ REDTEAM_GREEN
→ CERTIFIED
```

另有：

```text
FALSE_POSITIVE
DUPLICATE
SUPERSEDED
BLOCKED
RESEARCH_ONLY
DELETE
```

禁止：

```text
DISCOVERED → DONE
LOCAL TEST PASS → PRODUCTION READY
```

---

# 28. 每个修复的最低证据包

每个 issue 必须产生：

```text
issue_id
base_sha
minimal reproduction
expected semantic result
old actual result
root cause
changed files
new tests
exact test commands
new actual result
affected canonicals
affected physical implementations
affected evidence
performance delta
red-team result
final state
```

没有 minimal reproduction：

```text
不能开始“凭感觉修”
```

---

# 29. 每个算子的 Golden Oracle

对所有 production canonical 建：

```text
operator_golden_cases/
```

每个 canonical 至少有：

```text
normal case
constant case
small sample
all missing
mixed missing
+Inf
-Inf
ties
window boundary
invalid params
dtype variants
```

复杂算子再加业务特例。

---

# 30. Property / Metamorphic Testing 规则

比普通 example tests 更重要。

## 时间因果性

对 causal operator：

```text
修改 t+1 以后所有数据
→ <=t 的输出完全不变
```

这是防未来函数的基础 property。

## Prefix invariance

```text
run(x[:T])
==
run(x[:T+K])[:T]
```

适用于 causal non-revising operators。

## Asset permutation equivariance

截面算子：

```text
打乱股票列
→ 输出按同一 permutation 打乱
```

不允许结果依赖物理 column order，除非 contract 明确 tie-break by canonical instrument key。

## Rank monotonic invariance

严格单调变换：

```text
rank(x)
==
rank(a*x+b)
```

对 a>0。

## Covariance identity

```text
ts_cov(x,x)
≈ ts_var(x)
```

同 ddof/min_periods 语义。

## Correlation identity

非零方差有效窗口：

```text
ts_corr(x,x) ≈ 1
```

## Z-score property

在 contract 有效窗口：

```text
窗口均值/标准差性质
```

但必须按 canonical 定义验证 current output，不要错误假设全窗口输出序列本身均值=0。

## Scale invariance

对于理论上无量纲的算子：

```text
f(c*x) == f(x)
```

若 c>0。

## Translation invariance

例如：

```text
std(x+c) == std(x)
```

## Window-local direct oracle

对 ES/trimmed mean/skew/Qn 等：

```python
for t in range(T):
    w = finite(x[max(0,t-W+1):t+1])
    expected[t] = direct_definition(w)
```

禁止用另一个 rolling API 自己验证自己。

---

# 31. Edge Matrix

每个生产算子最少自动跑：

```text
T:
  0
  1
  2
  window-1
  window
  window+1
  2*window
  252
  1000

N assets:
  1
  2
  3
  10
  100
  3000

window:
  1
  2
  3
  5
  20
  60
  252
  >history

missing:
  none
  leading
  middle holes
  trailing
  alternating
  all

Inf:
  +Inf
  -Inf
  both

ties:
  no ties
  50% ties
  90% ties
  all equal

dtype:
  float64
  float32
  int64
  nullable integer
```

当然并非所有组合全 Cartesian；用 coverage design / pairwise / property generator 降低成本。

---

# 32. 参数空间不能只测 default

对每个 ParamSpec：

```text
min
min+epsilon / min+1
default
middle
max-1 / max-epsilon
max
invalid below
invalid above
wrong type
NaN
Inf
None
bool-as-int
float-as-int
```

对 conditional params：

```text
active_when true
active_when false
```

任何 backend 自己偷偷：

```text
int(5.9)
```

都必须被抓出来。

---

# 33. 统计语义 Authority

为避免不同 backend 各猜：

```text
ddof
quantile interpolation
rank ties
zero variance
min_periods
Inf
```

建立统一：

```text
NumericSemanticsRegistry
```

至少包含：

```text
std_ddof
var_ddof
corr_pair_validity
cov_pair_validity
rank_tie_method
quantile_method
zero_std_policy
divide_by_zero
Inf_policy
NaN_policy
rolling_current_row_policy
warmup_policy
```

每个 operator 只引用 authority。

---

# 34. Pandas Reference 也必须被质疑

禁止：

```text
pandas == oracle
```

Pandas kernel 只是：

```text
reference implementation candidate
```

真正 Oracle 层级：

```text
数学定义
/ 官方算法定义
/ 财务业务语义
/ 交易时钟 contract
```

然后用 direct Python/Numpy/SciPy/Statsmodels/TA-Lib（适用时）生成 independent oracle。

---

# 35. Technical Indicator 专项

技术指标最容易出现“名字一样但 seed/adjust 不同”。

必须明确：

```text
EMA:
  alpha
  adjust
  seed
  ignore_nulls
  warmup

Wilder:
  seed = SMA first window?
  recursive update
  first valid output

RSI:
  zero loss
  zero gain
  both zero
  gaps

MACD:
  fast/slow ordering
  EMA contract
  signal seed

Bollinger:
  ddof
  mid basis
  std multiplier

ATR:
  true range definition
  first close
  Wilder smoothing

ADX/DMI:
  tie in +DM/-DM
  smoothing
  zero ATR

Ichimoku:
  displacement
  future-looking plotted line vs causal factor value
```

特别注意 Ichimoku：

图表中的 Senkou Span 常被“向未来位移显示”，FactorEngine 生产因子绝不能因此产生未来函数。必须区分：

```text
computed value at t
plot displacement metadata
```

---

# 36. Candlestick / Pattern 专项

binary/event operator 不应该被矿工当连续 alpha terminal。

同时要验证：

```text
OHLC invalid geometry
high < low
open outside range
close outside range
zero range
limit-up one-price
suspension
adjusted vs raw price
```

所有 pattern 都要：

```text
Bool/Event role
```

与 DirectUse status 对齐。

---

# 37. Regression / Statistical Model 专项

除 Ridge 外，全扫：

```text
regress
residual
r_squared
slope
intercept
ts_beta
rolling_beta
ts_regression_slope
ts_regression_resid
forecast_error
PACF/ACF
```

检查：

```text
x/y argument order
intercept
ddof
pairwise finite
min obs
rank deficiency
singular matrix
regularization
current in-sample residual
forecast residual
window output semantic
```

尤其当前 `residual` 的文档是“样本内当前残差的窗口均值”，名字很容易误导。DirectUse/LLM mining catalog 必须写清，避免模型把它理解成原始 residual series。

---

# 38. Stateful / Filter 专项

必须证明：

```text
batch == stream
one-shot == shard
resume checkpoint == uninterrupted
```

对：

```text
EMA
KAMA
recursive filters
deadband
quantile state
Kalman
adaptive slew
regime state
```

建立：

```text
StateArtifact
state_schema_hash
last_timestamp
snapshot_id
semantic_hash
operator_version
```

checkpoint 不能跨 incompatible semantics 恢复。

---

# 39. Minute→Daily 专项

严格按交易 session：

```text
auction
morning
lunch break
afternoon
after-close
half-day special calendar
```

每日 OHLC/VWAP：

```text
open = first valid tradable minute
high = max valid
low = min valid
close = last valid
volume = sum
turnover = sum
VWAP = sum(amount)/sum(volume)
```

明确：

```text
zero volume
missing minute
duplicate minute
out-of-session print
late correction
suspension
limit state
```

避免用普通 resample 代替交易日历语义。

---

# 40. DataAccess PIT Poison Test Suite

这是 DataAccess 最关键的长期红队测试。

构造：

```text
financial statement original
later restatement
announcement after close
future revision
future index member
future industry reclassification
future corporate action
future analyst estimate
future holding disclosure
```

然后验证：

```text
asof = historical date
```

任何 future poison value 都绝不能改变历史结果。

---

# 41. Universe PIT

股票池必须是：

```text
UniverseSnapshot(asof)
```

而不是：

```text
今天的成分股回填历史
```

测试：

```text
IPO
delisting
ST
suspension
index entry/exit
industry change
```

Feature/Factor/Model artifact 都绑定 `UniverseSnapshotIdentity`。

---

# 42. Calendar Identity

所有“bars”语义：

```text
lag 20
horizon 20
embargo 20
rolling 20
```

都需要明确：

```text
ExchangeSessionCalendar
market
session
timezone
calendar_version
```

不能从 panel 实际出现日期猜交易日。

---

# 43. DataAccess Fault Injection

Agent-16 每轮持续注入：

```text
disk full
permission denied
rename failure
partial parquet
corrupt footer
manifest corrupt
manifest missing
writer crash
reader during commit
two writers
process killed
network timeout
remote 403/404/500
credential expiry
clock skew
deadline/cancel
```

需要证明：

```text
old generation remains readable
or new generation atomically visible
never partial committed state
```

---

# 44. DataAccess Cache Identity

每个 cache key 都要回答：

```text
什么变化必须 miss？
什么变化可以 hit？
```

至少绑定：

```text
dataset snapshot
schema
query/request semantics
fields
filters
instruments/universe
PIT asof
join policy
unit normalization
transform
aggregation
calendar
build/execution identity where appropriate
```

不要“加几个 tuple 字段”就停止，要从结果函数的全部决定变量反推 cache key。

---

# 45. Cache Dependency Invalidation

L1/L2/L3 cache 必须有 dependency graph：

```text
source snapshot changes
→ prepared read invalid
→ factor result invalid
→ evaluation intermediate invalid
→ report invalid
```

不能只依赖 TTL eventual consistency 对 correctness cache。

区分：

```text
ephemeral performance cache
correctness result cache
```

两者要求不同。

---

# 46. End-to-End Certification

Agent-18 建立真实 pipeline：

```text
DataAccess snapshot
→ fields
→ FactorEngine DAG
→ factor values
→ preprocessing/filter
→ FactorAssembly
→ model fit
→ model artifact
→ OOS predict
→ evaluator
```

每一步输出 artifact identity，并能串成 Artifact DAG。

---

# 47. E2E anti-lookahead

随机选历史 cutoff T：

```text
删除 T 后未来数据
vs
保留全部未来数据
```

对 T 以前：

```text
DataAccess output
Factor values
filtered factor
model training samples
OOS score
evaluation labels available before maturity
```

必须符合各自时钟 contract。

---

# 48. Evaluation Return Contract

如果策略统一使用 VWAP-to-VWAP，则 label authority 必须唯一。

验证：

\[
r_t^{(H)}
=
\frac{VWAP_{t+H}}{VWAP_t}-1
\]

绑定：

```text
entry VWAP completeness
exit VWAP completeness
label maturity
DecisionClock
purge interval
embargo
```

禁止某些 evaluator/helper 又默认为 close-to-close。

---

# 49. Cold-start Factor Library 重认证

任何 operator：

```text
demote research_only
semantic change
parameter change
backend production status change
```

都必须触发：

```text
ColdStartLibraryRevalidation
```

每个 factor：

```text
parse DAG
→ transitive canonical dependencies
→ production admission
→ DataAccess field availability/PIT
→ parameter-domain
→ backend implementation evidence
```

输出：

```text
strict-active before/after
newly invalid
newly staged
replacement candidate
affected microclusters
affected macroclusters
```

---

# 50. Mining Catalog 与 Operator Semantics 联动

LLM 挖因子消费的 catalog 不能只写名字。

必须暴露：

```text
description
economic_role
DirectUseStatus
terminal_allowed
input slots
units
grain
parameter role
searchable ranges
PIT restrictions
cost tier
semantic redundancy group
replacement
```

如果算子语义修了：

```text
catalog hash
```

必须变化。



---

# 51. Performance Engineering 总原则

性能 Agent 必须遵守：

```text
Correctness First
Semantic Identity Frozen
Then Optimize
```

禁止：

```text
为了快改 min_periods
为了快把 NaN 当 0
为了快去掉 PIT join
为了快变更 warmup
为了快降低精度
为了快 silent fallback
```

任何 optimization 如果改变 semantic hash：

```text
它不是 optimization
它是 semantic change
```

必须重新走完整 certification。

---

# 52. 真实 benchmark workload

不要只测：

```text
1000 rows
1 factor
```

至少维护以下 benchmark profiles。

## DAILY-SMALL

```text
252 dates
500 stocks
50 factors
```

用于 PR smoke。

## DAILY-REALISTIC

```text
5 years
~1250 sessions
3000–5000 stocks
200–1000 factors
```

用于常规 performance suite。

## DAILY-LARGE-DAG

```text
10 years
A-share full universe
5,000–20,000 factor DAG roots
高 CSE 比例
多 backend eligible
```

用于 nightly。

## INTRADAY-REALISTIC

```text
1m bars
A-share universe
minute→daily feature bundle
multiple sessions
missing/suspension
```

## COLD-START-MASS

```text
当前 strict-active/candidate factor universe
```

用于验证大规模 compile/plan/materialize。

---

# 53. 每次 benchmark 记录

```text
wall_clock_ms
cpu_time
peak_rss
peak_python_heap
bytes_scanned
files_scanned
rows_read
rows_output
backend_regions
backend_switches
pandas_to_polars_bytes
polars_to_pandas_bytes
arrow_transfer_bytes
sql_materialized_bytes
q_upload_bytes
q_download_bytes
collect_count
sql_query_count
CSE_hits
CSE_misses
native_shared_nodes
fusion_ratio
cache_hits
cache_misses
threads
processes
open_file_descriptors
time_to_first_result
```

---

# 54. Performance Regression Gate

每个 benchmark 有：

```text
certified_baseline.json
```

比较：

```text
median
p95
peak RSS
bytes transferred
```

默认：

```text
>10% regression → investigate
>20% regression → gate fail
```

阈值可按 workload 调，但不能由实现 Agent 临时放宽。

---

# 55. Planner 最重要的性能 telemetry

每次 FactorEngine run 输出类似：

```text
Plan ID
Canonical DAG nodes: 1432
CSE unique nodes: 617

Backend regions:
  Pandas: 2
  Polars: 5
  DuckDB: 3
  q: 0

Backend switches: 8

Transfers:
  Pandas→Polars: 1.8GB / 2
  Polars→Pandas: 0.4GB / 1
  DuckDB→Polars: 2.2GB / 3

Materializations: 7
Collects: 6

Compute: 11.2s
Transfer: 4.9s
Materialize: 2.7s
Total: 18.8s
```

没有这个 telemetry，性能 Agent 只能盲猜。

---

# 56. Region Planner hard goals

```text
BACKEND_ZERO_OPERATOR_LEVEL_CROSS_ENGINE_PINGPONG
BACKEND_MAXIMAL_COMPATIBLE_REGION_EXECUTION
BACKEND_TRANSFER_COST_INCLUDED_IN_ROUTING
BACKEND_NATIVE_INTERMEDIATE_PRESERVED
BACKEND_CONVERSION_ONLY_AT_REGION_BOUNDARY
CSE_SAME_BACKEND_SHARED_NODE_STAYS_NATIVE
```

---

# 57. Memory Governance

长时间多 Agent + 大测试最容易 OOM。

统一：

```text
OPENBLAS_NUM_THREADS=1
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
```

大测试：

```text
serial or bounded concurrency
```

不要同时跑多个 full-suite/high-memory benchmark。

Coordinator 维护：

```text
resource_leases.json
```

包括：

```text
test memory budget
benchmark slot
DuckDB slot
q slot
CPU slots
```

---

# 58. Performance 不只看计算快

还要检查：

```text
startup cost
JIT compile cost
q process startup
SQL planning
Arrow conversion
Polars collect
MultiIndex construction
serialization
artifact write
```

对 research loop：

```text
首次冷启动
重复热运行
```

都要测。

---

# 59. Backend-specific benchmark

## Pandas

重点：

```text
unstack/stack
wide panel allocation
rolling
NumPy copy
MultiIndex
```

## Polars

重点：

```text
LazyFrame fusion
collect count
unpivot/pivot
group window
streaming
Arrow zero-copy
```

## DuckDB

重点：

```text
scan pruning
query count
CTE/subquery duplication
window sort
materialization
Arrow export
```

## q

重点：

```text
upload/download
resident reuse
IPC
workspace memory
thread/process
sort
vector ops
```

---

# 60. 不要把 q 默认当最快

q 加入以后必须基于真实 workload。

例如：

```text
Polars compute = 5ms
q compute = 2ms
Python→q = 12ms
q→Python = 8ms
```

则：

```text
q total 22ms
Polars total 5ms
```

Planner 应选 Polars。

---

# 61. Evidence Artifact 总体设计

任何 certification 必须生成不可手改的 evidence artifact。

建议：

```text
evidence/
  operator/
  backend/
  dataaccess/
  integration/
  performance/
```

每个 evidence：

```text
artifact_schema_version
generated_at
source_commit
build_identity
semantic_hash
parameter_domain_hash
implementation_closure_hash
test_case_registry_hash
runtime_versions
platform
results
failures
benchmark
```

---

# 62. Implementation Closure Hash

不能只 hash 当前 operator 文件。

例如：

```text
ts_zscore.py
```

还依赖：

```text
numeric_semantics
rolling helpers
bridge
base class
parameter validation
```

所以：

```text
ImplementationClosureHash
```

至少覆盖：

```text
direct source
transitive helper modules
compiler/emitter
shared semantic authority
type adapter
```

这些任一变化：

```text
旧 evidence invalid
```

---

# 63. Evidence 生成规则

禁止：

```text
旧 evidence
+ 修改 commit SHA
= 新 evidence
```

必须：

```text
run certification
→ capture real result
→ generate artifact
```

Evidence generator 自己也进入 CI。

---

# 64. Production Truth Matrix

生成：

```text
PRODUCTION_TRUTH_MATRIX.md
PRODUCTION_TRUTH_MATRIX.json
```

示例：

| Capability | Declared | Implemented | Runtime | Parity | PIT | Perf | Admitted |
|---|---:|---:|---:|---:|---:|---:|---:|
| pandas/ridge | yes | FAIL dual-input | no | no | n/a | no | NO |
| polars-long/ts_skew | yes | ? | test | ? | n/a | ? | UNKNOWN |
| polars-native/ts_skew | yes | suspect math | no | no | n/a | no | NO |
| q/ts_mean | yes | lowering | unproven | unproven | unproven | unproven | NO |

这比：

```text
supported=True
```

有意义得多。

---

# 65. Root CI 统一 Production Gates

当前 `.github/workflows` 有若干专项 workflow，但还需要一个总 production workflow。

建议：

```text
.github/workflows/quant-platform-production-gates.yml
```

组成：

```text
1. static architecture
2. operator registry ABI
3. semantic contract
4. production surface
5. implementation-path certification
6. Pandas oracle
7. Polars long
8. Polars native
9. DuckDB
10. q (environment capable 时)
11. DataAccess PIT
12. snapshot/identity
13. model leakage
14. evaluator contract
15. cold-start dependency
16. clean wheel
17. package boundary
18. integration smoke
```

---

# 66. CI 分层

## PR Fast Gate

目标：

```text
分钟级
```

跑：

```text
changed-family tests
static ABI
golden cases
targeted property
small parity
identity tests
```

## Merge Gate

跑：

```text
all production family certification
all backend available suites
PIT poison small
clean-wheel
integration
```

## Nightly

跑：

```text
large randomized fuzz
large factor library
fault injection
performance
soak
concurrency
```

## Weekly

跑：

```text
deep audit miner
all physical implementation inventory
stale evidence
dependency update compatibility
```

---

# 67. q CI

q 可能受 license/runtime 限制。

所以分：

```text
Q_STATIC_GATE
Q_COMPILE_GATE
Q_RUNTIME_GATE
Q_PARITY_GATE
Q_PERF_GATE
```

如果 CI 环境没有 q：

```text
runtime = UNAVAILABLE
```

不是：

```text
PASS
```

production q admission 必须依赖一份来自有合法 q runtime 的真实 evidence artifact。

---

# 68. Clean Wheel

每个 production package：

```text
build wheel
new venv
install wheel
cd outside source tree
unset PYTHONPATH
import
run public API
run minimal real calculation
```

至少：

```text
DataAccess
FactorEngine
FactorAssets
Modeling
QuantEvaluator
FactorPreprocess
FactorOptimizer
```

禁止：

```text
sys.path.insert(source)
```

作弊。

---

# 69. DataAccess clean-wheel build identity

这是本轮特别需要的测试：

```text
git repo 内运行
vs
只安装 wheel 无 .git
```

必须得到：

```text
同一个 package build identity
```

且不是：

```text
unknown
None
runtime current repo HEAD
```

---

# 70. Package Boundary

生产包不应依赖 monorepo 偶然 import。

检测：

```text
import graph
distribution dependencies
optional extras
```

所有 adapter：

```text
DA↔FE
FA↔DA
FA↔QE
Modeling↔FA
```

在 clean env 分别验证。

---

# 71. Audit Miner：数学语义模式扫描

除了 grep TODO，专门识别数学反模式。

例如：

```text
rolling threshold
→ pointwise mask
→ second rolling
```

```text
replace std=0 with 1
```

```text
return raw input on exception
```

```text
return NaN for unsupported branch
```

```text
approximate named statistic with different formula
```

```text
current-row threshold vs current-window threshold
```

```text
expanding accidentally代替rolling
```

---

# 72. Audit Miner：silent corruption 模式

持续找：

```python
except Exception:
    pass

except Exception:
    return None

except Exception:
    return np.nan

except Exception:
    return input

except Exception:
    result[:] = np.nan
```

每个点分类：

```text
expected data-domain failure
vs
programming/infrastructure failure
```

只有前者可以转成数值缺失，而且要 typed reason telemetry。

---

# 73. Typed Failure Model

Operator output 不应该只能表达：

```text
number / NaN
```

很多失败需要 telemetry：

```text
INSUFFICIENT_SAMPLE
CONSTANT_INPUT
SINGULAR_MATRIX
INVALID_DOMAIN
UNSUPPORTED_PARAMETER
DATA_MISSING
PIT_UNAVAILABLE
INFRASTRUCTURE_FAILURE
```

至少 evaluation/report 能统计：

```text
NaN 为什么产生
```

否则大面积 NaN 可能是 bug，也可能是正常 warmup，无法区分。

---

# 74. Run-level Quality Telemetry

每次生产 factor batch：

```text
factors_requested
factors_success
factors_all_nan
factors_partial_nan
operators_failed
failure_reasons
fallbacks
PIT_rejections
snapshot_failures
backend switches
```

设置 hard gate：

```text
unexpected all-NaN factor > 0
→ batch fail
```

白名单的 warmup/structural missing 单独处理。

---

# 75. Factor Result Sanity

不是用统计指标判断 alpha 好坏，而是发现计算异常。

每个 factor 监测：

```text
finite coverage
cross-sectional unique count
time variation
constant days
Inf count
extreme magnitude
sudden scale jump
day-over-day missing jump
universe coverage
```

例如：

```text
以前 coverage 95%
今天 0%
```

应直接触发 computation incident。

---

# 76. DirectUse / LLM Mining Safety

所有 mining candidate 在入 Evaluator 前先做：

```text
operator production certification
PIT eligibility
field availability
parameter validity
semantic role
cost budget
```

LLM 不能因为 registry 里“有这个名字”就使用。

---

# 77. All-Operator Closure Program

不要人工一个个挑。

自动生成 work shards：

```text
Shard TS-001
Shard TS-002
...
Shard CS-001
...
```

按：

```text
family
source module
complexity
backend paths
```

每 shard 20–50 canonical。

Shard 只有全部：

```text
CERTIFIED
RESEARCH_ONLY
DELETE
```

之一时才能 closed。

---

# 78. Research-only 也要有明确原因

不能把不会修的算子都：

```text
research_only=True
```

然后假装完成。

每个 research-only：

```text
reason
known defect
replacement
required data
future plan
mining visibility
```

如果它不值得保留：

```text
DELETE
```

---

# 79. Duplicate / Alias Closure

所有 alias：

```text
resolve canonical
parameter mapping
semantic equivalence
```

验证：

```text
alias(expr)
==
canonical(mapped expr)
```

不允许 alias 指向：

```text
不存在 canonical
不同语义 canonical
```

---

# 80. ABI Closure

对所有 implementation：

```text
metadata.param_names
ParamSpec
Python signature
DSL binding
IR attrs
backend lowering params
```

必须完全一致。

新增 gate：

```text
OPERATOR_ABI_ZERO_VIOLATIONS
```

不只看函数能运行。

---

# 81. Parameter Injectivity

如果两个 parameter 值：

```text
p1 != p2
```

但在所有合理输入上 operator 完全一样，

则参数可能是：

```text
dead parameter
metadata-only
implementation ignored
```

Audit Agent 对 searchable parameter 做 injectivity probe。

尤其：

```text
window
alpha
threshold
quantile
ddof
lag
```

---

# 82. Semantic Dedup

两个 canonical 如果：

```text
同输入
同参数映射
同输出
同 role
```

长期完全重复，

合并成：

```text
one canonical + aliases
```

减少 mining search space 和 maintenance burden。

---

# 83. Backend Capability 必须 parameter-aware

不能：

```text
ts_quantile supports SQL = True
```

而要：

```text
ts_quantile(q=0.5, interpolation=linear)
```

这一调用组合是否 certified。

即：

```text
Capability(canonical, bound_parameter_domain, backend)
```

---

# 84. Backend Fallback Hard Rule

Production：

```text
planner route
```

确定后，

执行中：

```text
backend A 失败
```

不能偷偷切：

```text
backend B
```

除非整个 region：

```text
重新 planning
重新记录 execution identity
重新从干净输入执行
```

且结果明确标：

```text
rerouted
```

默认应 fail。

---

# 85. No-Fallback Proof

每个 backend parity test 必须证明：

```text
selected implementation path
actual runtime path
fallback_count == 0
transition events expected
```

不要只比较输出值。

---

# 86. CSE Correctness

公共子表达式缓存必须绑定：

```text
semantic hash
input artifact identity
backend region identity
parameter values
PIT snapshot
```

测试：

```text
CSE enabled
vs
CSE disabled
```

结果完全一致。

---

# 87. Fusion Correctness

测试：

```text
fused DAG
vs
unfused node-by-node reference
```

特别：

```text
binary ops
rolling ops
group ops
missing values
```

不能因为 expression fusion 改 null propagation。

---

# 88. Time Sharding

长历史计算：

```text
2016-2020
2021-2026
```

分 shard 后拼起来，必须与一次性全历史一致，前提补足 warmup/state overlap。

建立每 operator：

```text
required_history_bars
stateful flag
checkpoint support
```

---

# 89. Incremental Daily Update

Production 最终每天只新增一天。

验证：

```text
full recompute
vs
previous state + new day
```

当前日及需要修订窗口内结果一致。

如果某因子依赖：

```text
revision-vintage
```

新 revision 到达时只重算合法受影响区间。

---

# 90. Streaming Evaluation

Evaluator streaming 输入最新一天时：

```text
不能重新 fit 未来数据
不能改变历史 evaluation
```

Factor health 只消费当时已知：

```text
factor_t
label matured through t-H
```

---

# 91. Reproducibility

同：

```text
code build
DataSnapshot
UniverseSnapshot
Calendar
FactorSet
parameters
seed
```

运行两次：

```text
bitwise 或 defined tolerance equivalent
```

如果 algorithm stochastic：

```text
seed
library version
thread settings
```

进入 identity。

---

# 92. Concurrency Determinism

单线程与允许的多线程：

```text
结果一致
```

检测：

```text
global registry contamination
global singleton config leak
environment variable mutation
shared temp files
q workspace collision
DuckDB connection state
```

---

# 93. Thread / Process Leak

循环执行：

```text
100
1000
```

次 factor batch。

记录：

```text
thread count
process count
fd count
RSS
q resident table count
DuckDB temp
```

不能单调泄漏。

---

# 94. Soak Test

Nightly/weekly：

```text
连续执行多个小时
随机 factors
随机 parameter legal combos
随机 backend eligible routes
```

看：

```text
memory
deadlock
resource leak
stale cache
nondeterminism
```

---

# 95. Failure Recovery

故意中断：

```text
FactorEngine batch
DataAccess read
materialization
q region
evaluation
```

重启后：

```text
checkpoint
artifact
cache
```

不能把 partial result 当完成。

---

# 96. Atomic Artifact Publish

Factor result/model/evidence：

```text
write temp
fsync
validate
atomic publish
manifest commit
```

失败时：

```text
old artifact remains
new partial invisible
```

---

# 97. Security / Path / SQL

DataAccess production：

```text
path authorization
credential isolation
SQL sandbox
dataset allowlist
```

也要持续 red-team：

```text
../ path
symlink escape
malformed S3 URI
SQL injection through identifier
unauthorized dataset
```

---

# 98. Logging 不得泄漏 credentials

审计：

```text
S3 secret
session token
database password
private path
```

exception/report/evidence 不能落敏感凭据。

---

# 99. R2 当前 HEAD 第一优先级队列（今晚必须先按此顺序）

## RECOVERY-P0：先恢复仓库完整性

```text
R2-P0-001  修复 q_executor.py 半提交/截断
R2-P0-002  current HEAD 全包 compileall/import smoke
R2-P0-003  failed-agent partial-write scrub
R2-P0-004  禁止 hard-coded PASS evidence
R2-P0-005  建立 writer worktree + change manifest + safe serial mode
```

## Q Backend

```text
R2-P0-006  Q capability 三套 authority → ONE authority
R2-P0-007  Q PlanNode ABI: children/operator/params → inputs/op/attrs
R2-P0-008  Q compiler dispatch-order/unreachable specialized lowering
R2-P0-009  ts_corr/ts_cov/ts_beta/wma/median/var/skew/kurt/round/lag 真正 q 语义
R2-P0-010  Q NULL/type contract
R2-P0-011  Q DataAccess/PIT input bundle
R2-P0-012  Q output schema/grain/nullability parameterization
R2-P0-013  Q physical region authority / no whole-tree self-routing
R2-P0-014  Q fan-in/resident/workspace/concurrency
R2-P0-015  q runtime 缺失时 production admission = DISABLED，不是假 PASS
```

## FactorEngine 算子 correctness

```text
R2-P0-016  ridge dual-input/ABI/broad-except
R2-P0-017  全仓 PhysicalImplementation inventory
R2-P0-018  ts_corr canonical 修复后的旧 Polars registered path 清理
R2-P0-019  ts_cov 旧 Polars registered path
R2-P0-020  rolling regression slope/intercept/resid 同类错误
R2-P0-021  ts_moment/ts_kurt/ts_skew window-local direct oracle
R2-P0-022  trimmed mean / winsorized mean 语义分离
R2-P0-023  Qn vs IQR-scale 正名/实现
R2-P0-024  Expected Shortfall / tail family current-window 语义
R2-P0-025  ts_zscore NaN/Inf/zero-std/min_periods
R2-P0-026  canonical param ABI: d/window/min_periods/ddof single authority
R2-P0-027  technical indicators全族 seed/warmup/Wilder/EMA parity
R2-P0-028  return_decomp 最新 ABI 二次 red-team
R2-P0-029  all selectable implementations match canonical signature
```

## DataAccess

```text
R2-P0-030  build/version identity ONE authority
R2-P0-031  clean wheel 无 .git identity稳定
R2-P0-032  read_contract serializer → CanonicalIdentityEncoder
R2-P0-033  correctness identity 全 SHA-256 / >=128bit
R2-P0-034  FE source_scope_hash 禁止 default=str + 64bit correctness identity
R2-P0-035  CompiledDataRequest strict typed freeze
R2-P0-036  remote snapshot typed failures + credential_generation
R2-P0-037  fail_if_changed verified policy
R2-P0-038  Universe/Calendar/Revision vintage identity
R2-P0-039  PIT poison suite
R2-P0-040  DataAccess↔FactorAssets real adapter
```

## Cold-start / Operator Rehabilitation

```text
R2-P0-041  重新计算当前 HEAD 的 research/production/mining truth
R2-P0-042  不再沿用“678 已降级”的旧数字
R2-P0-043  placeholder audit 与真实 register_operator ABI 对齐
R2-P0-044  所有可构造经济因子的 placeholder 分批真实实现
R2-P0-045  duplicate/alias/operator recipe 去重
R2-P0-046  unavailable-data operators 保持 gated，不伪实现
R2-P0-047  model/training structural operators 不作为 DirectUse factor terminal
R2-P0-048  cold-start transitive dependency 全量重认证
R2-P0-049  因算子降级失效的冷启动因子做 replacement/rebuild
```

## Modeling / Evaluator / FactorAssets

```text
R2-P0-050  Modeling 真正单 authority，不只是 AUTHORITY.md
R2-P0-051  gap/purge/embargo 基于 trading sessions + label maturity
R2-P0-052  train→val / train→test / val→test 全边界
R2-P0-053  QuantEvaluator assign_quantiles*.squeeze shape bug
R2-P0-054  quantile tie-heavy / empty-bin diagnostics
R2-P0-055  QE top-bottom orientation 二次 property test
R2-P0-056  FactorAssets DA adapter 不允许 date.today/broad except/fake catalog
```

## Backend Planner / Performance

```text
R2-P0-057  BatchGlobalOptimizer 由 scaffold 升级真实 full DAG
R2-P0-058  SQL/q 加入 eligible backend assignment
R2-P0-059  TransferEdge 真正生成
R2-P0-060  contiguous region 而不是 backend-name grouping
R2-P0-061  realistic row/bytes/memory estimates
R2-P0-062  production engine 真正消费 PhysicalRegionPlan
R2-P0-063  executor 不得重新 route
R2-P0-064  NativeBuffer/CSE 单 authority
R2-P0-065  batch-native FactorMatrixArtifact
R2-P0-066  backend conversion/materialization telemetry
R2-P0-067  realistic benchmark + regression gates
```

## CI / Evidence

```text
R2-P0-068  root production CI aggregator
R2-P0-069  current HEAD clean-wheel
R2-P0-070  stale evidence invalidation
R2-P0-071  completion-report claim integrity
R2-P0-072  final red-team
R2-P0-073  final current-SHA evidence regen
```

完成以上显式 queue 后：

```text
AuditMiner 自动继续
```

不停止。

---

# 100. 第二优先队列

```text
P1-01  ReadLineage instrument_filter default audit
P1-02  fail_if_changed policy levels
P1-03  local file content/generation fidelity
P1-04  env semantic toggle identity
P1-05  alias/dedup
P1-06  parameter injectivity
P1-07  all technical indicators
P1-08  regression family
P1-09  stateful checkpoint
P1-10  minute→daily
P1-11  cache dependency invalidation
P1-12  planner boundary telemetry
P1-13  performance realistic workload
P1-14  concurrency/restart
P1-15  clean-wheel full matrix
```

---

# 101. 长时间 Loop 的运行算法

ChiefCoordinator 应循环执行：

```python
while task_budget_active:

    refresh_current_head()

    invalidate_stale_findings_and_evidence()

    inventory = rebuild_truth_inventory()

    unresolved = prioritize(
        known_P0,
        reproducible_P1,
        uncertified_production_surface,
        stale_evidence,
        coverage_gaps,
    )

    if unresolved:
        allocate_nonconflicting_agents(unresolved)
    else:
        launch_audit_miners_on_new_dimensions()

    collect_agent_outputs()

    require_reproduction_before_fix()

    merge_nonconflicting_changes()

    run_targeted_tests()

    run_independent_review()

    update_machine_ledgers()

    if cycle_count % N == 0:
        run_broad_regression()
        run_PIT_smoke()
        run_cold_start_revalidation()
        run_performance_smoke()
        run_clean_wheel_subset()

    if no_new_issue_found:
        rotate_dimension()
        increase_adversarial_depth()
        do_not_stop_early()
```

---

# 102. “没找到问题”后的自动升级路径

如果连续一轮 0 issue：

第一层：

```text
换 operator family
```

第二层：

```text
换 physical implementation
```

第三层：

```text
property/fuzz
```

第四层：

```text
PIT poison
```

第五层：

```text
fault/concurrency
```

第六层：

```text
performance/soak
```

第七层：

```text
packaging/build/evidence
```

不能：

```text
0 issue → session success → stop
```

---

# 103. 每 30 分钟 Coordinator Checkpoint

输出：

```text
HEAD
issues discovered
issues reproduced
issues fixed
issues rejected
certified implementations
uncertified production implementations
current write leases
tests run
failures
performance delta
new risks
next assignments
```

所有数字机器生成。

---

# 104. 每 60–90 分钟 Broad Gate

至少：

```text
production operator ABI
changed-family parity
PIT poison smoke
DataAccess identity
cold-start dependency
cross-package import
memory smoke
```

避免多个 Agent 局部 green 叠加后整体 broken。

---

# 105. Session 不允许的终止原因

以下不能作为结束理由：

```text
“时间还很多但暂时没 P0”
“已有几百测试通过”
“某个 package 看起来 robust”
“当前 Agent 没找到问题”
“报告写 production ready”
```

---

# 106. 合法终止

只有：

```text
外部任务运行时间结束
```

或真正：

```text
all production surface CERTIFIED
all hard gates green
no known issue queue
red-team no issue under current audit budget
```

才结束。

结束时必须报告：

```text
KNOWN OPEN
UNKNOWN
DEFERRED
RESEARCH_ONLY
```

不能藏。

---

# 107. 当前 Production Status（R2 / HEAD d78ed761）

当前不能用“改了很多”替代 production certification。

建议真实状态：

```text
Repository Integrity:
  FAIL / BLOCKED
  reason:
    q_executor.py partial-write truncation
    q_backend imports removed symbols

FactorEngine Pandas Reference:
  PARTIALLY_VERIFIED
  known blocker:
    ridge dual-input implementation仍错误
    部分 operator semantics/evidence仍未current-head全认证

Polars Long / Expr:
  PARTIALLY_VERIFIED
  recent ts_std/var/cov/corr fixes are valuable
  but cannot代表 registered Polars native path

Polars Registered Native:
  NOT_CERTIFIED
  known blockers:
    ts_corr/ts_cov old wrong rolling formulation
    rolling regression family
    ts_moment/ts_kurt/window-local statistics
    ts_zscore/Inf and ABI gaps
    duplicate physical paths

DuckDB SQL:
  PARTIALLY_VERIFIED
  parameter-domain / physical-path / full-surface certification incomplete

Q/K:
  BROKEN_IN_CURRENT_HEAD
  production admission MUST BE OFF
  known blockers:
    executor truncated
    multiple capability authorities
    PlanNode ABI mismatch
    compiler semantics
    DataAccess not wired
    NULL/type
    region authority
    no current runtime/parity evidence

DataAccess:
  PARTIALLY_VERIFIED
  known blockers:
    build identity split-brain
    read_contract identity duplicate serializer
    hash-width / default=str issues
    request freeze
    remote snapshot credential generation
    PIT poison complete proof

Modeling:
  IMPROVED_BUT_NOT_SINGLE_AUTHORITY
  known blockers:
    standalone package still contains executable contracts/preprocess
    gap_days calendar-day semantics
    missing all split-boundary enforcement
    label-maturity coupling

QuantEvaluator:
  IMPROVED
  top-bottom spread direction fixed
  new blocker:
    assign_quantiles_batch / Numba squeeze shape instability
    tie-heavy/empty-bin diagnostics

FactorAssets:
  PARTIALLY_VERIFIED
  DataAccess adapter currently incompatible/fake in multiple places

Cold-start:
  STATUS MUST BE RECOMPUTED
  historical “678 research_only” cannot be reused as current truth

Root CI:
  NOT_PROVEN
  no root aggregator observed
  current HEAD no workflow/combined status evidence

Overall:
  NOT PRODUCTION CERTIFIED
  First objective tonight = restore repository integrity.
```

---

# 108. Hard Gates 总表

新增/维持：

```text
LOOP_ZERO_EARLY_TERMINATION
LOOP_MACHINE_GENERATED_COUNTS
LOOP_ZERO_SELF_REVIEW_CERTIFICATION

PHYSICAL_IMPLEMENTATION_IDENTITY_COMPLETE
ALL_SELECTABLE_IMPLEMENTATIONS_CERTIFIED
OPERATOR_ABI_ZERO_VIOLATIONS
OPERATOR_ZERO_SILENT_UNSUPPORTED_BRANCH
OPERATOR_ZERO_BROAD_EXCEPTION_TO_ALL_NAN
OPERATOR_PARAMETER_INJECTIVITY_PROVEN

POLARS_WINDOW_LOCAL_STATISTICS_DIRECT_ORACLE
POLARS_CS_FINITE_POLICY_PARITY
POLARS_ZERO_UNCERTIFIED_NATIVE_ROUTE

Q_PRODUCTION_SAFE_SINGLE_DEFINITION
Q_ZERO_DUPLICATE_CAPABILITY_AUTHORITIES
Q_MAIN_ROUTER_INTEGRATED
Q_PHYSICAL_REGION_ONLY
Q_FANIN_INPUT_PRESERVATION
Q_WORKSPACE_ISOLATION
Q_OUTPUT_SCHEMA_EXACT
Q_RUNTIME_PARITY_CERTIFIED

DA_ONE_BUILD_IDENTITY_AUTHORITY
DA_BUILD_IDENTITY_CLEAN_WHEEL_STABLE
DA_ONE_CANONICAL_IDENTITY_ENCODER
DA_CORRECTNESS_HASH_MIN_128
DA_ZERO_REPR_STR_IDENTITY_FALLBACK
DA_COMPILED_REQUEST_STRICT_FREEZE
DA_REMOTE_SNAPSHOT_FAIL_CLOSED_PROVEN
DA_PIT_POISON_ZERO_LEAK

BACKEND_REGION_MAXIMAL
BACKEND_BOUNDARY_COST_ROUTING
BACKEND_ZERO_OPERATOR_PINGPONG
CSE_NATIVE_REUSE
FUSION_REFERENCE_PARITY

COLD_START_ZERO_UNCERTIFIED_OPERATOR_DEPENDENCY
MINING_ZERO_RESEARCH_ONLY_TERMINALS

ROOT_PRODUCTION_GATES_MANDATORY
ALL_PACKAGES_CLEAN_WHEEL
EVIDENCE_IMPLEMENTATION_CLOSURE_HASH
EVIDENCE_REGENERATED_NOT_REBOUND

PERFORMANCE_ZERO_UNEXPLAINED_REGRESSION
RESOURCE_ZERO_LEAK_SOAK
```

---

# 109. 禁止的“修法”

禁止：

```text
1. 把错误算子直接 research_only 就当功能修完。
2. 为了绿测试修改 expected 去迎合错误代码。
3. xfail / skip production bug。
4. broad except → None/NaN。
5. runtime backend silently fallback。
6. 只改报告/状态文件。
7. 手工改 evidence SHA。
8. 加 dataclass 但 consumer 不接。
9. 缺 identity 用 ""/"unknown"。
10. clean wheel 用 sys.path 注入源码。
11. 为了性能改变 semantic contract 却不升级 semantic hash。
12. 只测 default parameters。
13. 只测 canonical，不确认 physical implementation path。
14. 只跑单包测试就写 production ready。
15. 多 Agent 并发改同一 authority file。
16. 发现难修就扩大 exception catch。
17. 把“没复现”写成 false positive。
18. 把“无 q runtime”写成 q PASS。
```

---

# 110. Independent Red-Team 最小问题集

每次准备 certification，Red Team 至少问：

```text
有没有另一 physical path 没测？
有没有 parameter combination 没测？
有没有未来数据能影响过去？
有没有 Inf/null/tie 边界？
有没有 all-NaN 掩盖 bug？
有没有 fallback？
有没有环境变量改变结果？
有没有 stale cache/evidence？
有没有 clean wheel 路径不同？
有没有 concurrent execution 不同？
有没有 split/shard 不同？
有没有 benchmark regression？
```

任一答案 UNKNOWN：

```text
不能 CERTIFIED
```

---

# 111. 最终验收问卷

整个 FactorEngine + DataAccess 只有能够回答下面的问题，才允许向上层说“当前生产面已认证”：

```text
1. 当前到底有多少 production canonical？
2. 每个 canonical 有几条 selectable physical implementation？
3. 每条都实际 runtime 验证过吗？
4. 每条都和 independent oracle 一样吗？
5. 全 parameter domain 都被覆盖了吗？
6. NaN/Inf/null/tie/ddof/min_periods 是否统一？
7. future poison 是否不能影响历史？
8. DataSnapshot identity 是否只有数据事实？
9. Build identity 是否离开 .git 仍稳定？
10. Universe/Calendar/Revision vintage 是否进入 identity？
11. DAG fusion/CSE/shard/stream 是否和 reference 一致？
12. backend fallback 是否全部可观察且生产禁止 silent？
13. 10k+ factor batch 是否不发生 conversion explosion？
14. q 是否进入主 planner authority，而不是旁路？
15. cold-start strict-active 是否只依赖 certified operators？
16. clean wheel 是否能独立运行？
17. evidence 是否真实重跑生成？
18. fault/concurrency/restart 是否不会产生 partial/corrupt artifact？
19. performance 是否有基线和回归 gate？
20. 当前还有哪些 UNKNOWN？
```

第 20 项如果不是：

```text
NONE on current production surface
```

就不得宣称 fully certified。

---

# 112. 给 ChiefCoordinator 的今晚 8 小时立即执行 Prompt（R2）

将以下内容作为最高优先级 instruction：

```text
MISSION BUDGET:
8 HOURS AUTONOMOUS DEVELOPMENT / AUDIT / REMEDIATION

CURRENT BASE:
Resolve HEAD yourself before every wave.
At authoring time: d78ed761b7d098d27e3acf916f3961d75096b3b3.

READ THIS ENTIRE R2 TASKBOOK FIRST.

Do not trust:
- Agent SUCCESS
- Agent FAILED
- commit message
- FINAL REPORT
- PASS constant
without inspecting current source and machine evidence.

PHASE ZERO IS REPOSITORY RECOVERY.

The current HEAD contains evidence of failed-agent partial writes:
q_executor.py is truncated while q_backend.py imports symbols that no longer exist.

Before any feature work:
1. create clean current-head snapshot
2. run compileall
3. run package imports
4. repair/revert partial writes
5. establish writer worktrees/branches
6. establish AgentChangeManifest
7. establish safe serial mode

API INSTABILITY POLICY:
If Content block not found / API error rate >20%:
- writer concurrency = 1
- read-only auditors <=2
- reviewer =1
- retry failed issues serially
Never sync failed worktree automatically.

Every writer issue:
worktree
→ reproduce
→ edit
→ targeted tests
→ import smoke
→ local commit
→ manifest
→ independent review
→ coordinator cherry-pick.

Never git add -A a failed-agent shared workspace.

After every merge/cherry-pick:
compileall + import smoke + targeted integration.

Then execute R2-P0 queue in priority order.

Do not spend the night only on Q.
Q must be repaired enough that:
- package imports
- capability truth is honest
- production routing is disabled until certified
Then move on to FactorEngine/DataAccess production surface.

For operators:
do not certify canonical names.
Enumerate every selectable PhysicalImplementationID.
Delete obsolete duplicate implementations where safer than maintaining two copies.

Every operator that can economically construct a factor from available fields should end in one of:
CERTIFIED_PRODUCTION
CERTIFIED_HIGH_COST
CERTIFIED_STATE/CONDITION
CERTIFIED_RECIPE_INTERNAL
or, only with a real reason:
RESEARCH_ONLY / DATA_GATED / DELETE.

Do not leave useful operators as passthrough/TODO merely because a prior bulk script demoted them.
Do not blindly promote all historical 678 operators either.
Classify, implement, oracle-test, and certify.

For model-like operators:
never hide training inside score().
Use fit/validate/freeze/predict artifacts and PIT-safe walk-forward lifecycle.

For DataAccess:
PIT/snapshot/build identity/revision/universe/calendar correctness comes before speed.

For performance:
correct semantics first.
Then optimize:
- projection/predicate pushdown
- batch reads
- native regions
- CSE/fusion
- Arrow zero-copy
- Polars Lazy
- DuckDB SQL
- Numba kernels
- q only when transfer-adjusted benchmark wins
- optional GPU/distributed backends only after profiling proves value.

Do not proliferate backends simply to claim more backends.

Every 20–30 min:
- refresh HEAD
- inspect failed-agent worktrees
- update machine ledger
- recalculate unresolved certification surface
- assign next microtasks.

Every 60–90 min:
- broad imports
- operator ABI
- parity
- PIT poison smoke
- cold-start dependency check
- performance smoke
- clean-wheel subset.

If explicit queue gets small:
launch AuditMiner / property fuzz / fault injection / concurrency / soak.
Do not finish early.

FINAL OUTPUT MUST INCLUDE:
- final SHA
- exact changed files
- issues fixed
- issues not fixed
- physical implementations certified
- operators promoted/research-only/deleted
- cold-start before/after
- backend performance before/after
- PIT/fault results
- clean-wheel results
- CI status
- evidence paths
- UNKNOWN / NOT_RUN explicitly

NEVER write “all problems solved” unless every current production hard gate is machine-green.

Objective:
ZERO KNOWN DEFECTS
+ COMPLETE CURRENT PRODUCTION-SURFACE CERTIFICATION
+ HONEST UNKNOWN SET
+ MEASURABLE PERFORMANCE IMPROVEMENT.
```

---

# 113. 建议新增机器文件

```text
loop_engineering/
  task_ledger.json
  agent_leases.json
  resource_leases.json
  production_truth_matrix.json
  certification_matrix.json
  stale_evidence.json
  current_head.json
  cycle_metrics.jsonl

factor_engine/evidence/
  operator_semantics/
  physical_implementations/
  parity/
  property/
  performance/

dataaccess/evidence/
  identity/
  pit/
  snapshot/
  build/
  fault/
```

---

# 114. Task Ledger Schema

```json
{
  "issue_id": "POL-WINDOW-P0-004",
  "status": "REPRODUCED",
  "severity": "P0",
  "subsystem": "factor_engine",
  "canonical": "ts_expected_shortfall",
  "physical_impl": "polars_registered_native",
  "base_sha": "...",
  "owner": "Agent-10",
  "reviewer": "Agent-21",
  "write_scope": ["..."],
  "reproduction": "...",
  "expected": "...",
  "actual": "...",
  "tests": [],
  "invalidated_evidence": [],
  "updated_at": "..."
}
```

---

# 115. Certification Record Schema

```json
{
  "canonical": "ts_skew",
  "semantic_hash": "...",
  "parameter_domain_hash": "...",
  "physical_implementation_id": "...",
  "implementation_closure_hash": "...",
  "oracle": "direct_window_reference_v1",
  "golden_pass": true,
  "property_pass": true,
  "fuzz_pass": true,
  "backend_parity_pass": true,
  "no_fallback_pass": true,
  "dag_pass": true,
  "pit_pass": true,
  "stream_pass": true,
  "performance_pass": true,
  "redteam_pass": true,
  "status": "CERTIFIED"
}
```

---

# 116. 结论

当前代码相比上一轮明显进步：

```text
更多 fail-closed
更多 parity test
更多 explicit contracts
更多 clean-wheel意识
更多多 Agent audit
```

但也正因为大规模并发修复开始加速，当前最大风险已经变成：

```text
局部修复速度 > 全局事实收敛速度
```

典型表现：

```text
状态报告与 main 不一致
task queue 滞后
同 canonical 多 implementation 但只测一条
数学命名与实现不一致
局部 test green 被升级为 production-ready
```

下一阶段不应该继续以“再写更多代码”为唯一 KPI。

真正 KPI 应变成：

```text
每小时关闭多少可复现 correctness risk
每小时增加多少 production certification coverage
每小时减少多少 UNKNOWN physical implementation
性能在正确语义下提升多少
PIT/fault/concurrency coverage 提升多少
```

只有这样，Loop Engineering 连续运行数小时才不会变成：

```text
很多 Agent 很忙
很多 tests 很绿
很多报告写 DONE
但系统里仍藏着未被真正执行过的错误路径
```

本总纲应作为 ChiefCoordinator 的长期任务母文档。后续每个 session 只追加增量发现和 certification 状态，不再重新发明一套并行的“FINAL REPORT truth”。


---

# 117. R2 CURRENT-HEAD OVERRIDE：失败 Agent 已造成 Q Executor 半提交

这是今晚最高优先级 blocker。

当前文件：

```text
factor_engine/backend/q_backend/q_executor.py
```

已经不再包含完整执行器。

当前主要只剩：

```text
imports
Q_FAN_IN_INPUT_PRESERVATION = "PASS"
Q_WORKSPACE_ISOLATED = "PASS"
Q_CONNECTION_THREAD_SAFE = "PASS"
Q_RESIDENT_HANDLE_BOUNDED_LIFETIME = "PASS"
```

但：

```text
factor_engine/backend/q_backend/q_backend.py
```

仍 import：

```python
QExecutionFallbackPolicy
get_q_executor
```

所以当前状态属于：

```text
PARTIAL-WRITE CORRUPTION
```

不是：

```text
Q optimization incomplete
```

## R2-Q-RECOVERY-P0-001

Recovery Agent 必须先：

```text
1. 对照 q_executor 最后一个已知完整版本
2. 识别 d78ed 里哪些 q_executor 改动是 intended
3. 恢复完整 executor
4. 再把 intended fan-in/workspace improvements 以最小 patch 重做
5. 不允许直接恢复旧文件后把新需求忘掉
```

## R2-Q-RECOVERY-P0-002

恢复后的最小 public ABI：

```text
QExecutionFallbackPolicy
QExecutionResult
QExecutor
get_q_executor
get_q_executor_telemetry
reset_q_executor_telemetry
```

具体名字以现有调用者 inventory 为准。

先扫描所有：

```text
from backend.q_backend.q_executor import ...
```

再冻结 ABI。

## R2-Q-RECOVERY-P0-003

新增：

```text
tests/q_backend/test_q_executor_public_abi.py
```

即使没有 PyKX/q runtime，也要：

```text
import module
resolve exported symbols
construct policy
construct executor
```

通过。

## R2-Q-RECOVERY-P0-004

新增源码完整性 gate：

```text
SOURCE_FILE_TRUNCATION_GUARD
```

对关键大模块维护：

```text
expected exported symbols
minimum structural elements
```

不是简单 LOC 阈值，而是 AST symbol inventory。

---

# 118. Q capability 并没有实现真正 Single Authority

当前同时存在：

```text
A. q_capability.py
   _PHASE1_NATIVE_OPS
   QBackendCapability

B. q_capability_evidence.py
   get_declared_native_ops()
   compute_q_capability_evidence()

C. q_physical_implementation_registry.py
   QPhysicalImplementationRegistry
```

而 C 当前还是：

```text
TODO: Populate from evidence ledger
```

因此不能称：

```text
single authority closed
```

## R2-Q-P0-001：最终 authority

最终只保留：

```text
QPhysicalImplementationRegistry
```

其记录不是单纯 canonical 名，而是：

```text
canonical
lowering_id
parameter_domain_id
implementation_hash
q_version_range
pykx_version_range
compile_evidence
runtime_evidence
parity_evidence
null_semantics_evidence
performance_evidence
```

## R2-Q-P0-002：声明层与认证层分离

可以有：

```text
Q_DECLARED_TARGETS
```

表示“希望实现”。

但 production admission 只能：

```text
registry.production_ready
```

不能：

```text
declared target
→ supports_native
```

## R2-Q-P0-003：删除假的 single-authority gate

如果 gate 逻辑类似：

```python
if hasattr(q_capability, "_PHASE1_NATIVE_OPS"):
    pass
return True, "PASS"
```

必须删除。

正确 gate：

```text
静态依赖扫描
+
runtime router dependency proof
+
registry population proof
```

## R2-Q-P0-004：compiler admission

当前：

```python
return op_name in self._operator_map
```

最多表示：

```text
LOWERING_EXISTS
```

不能等于：

```text
production compilable
```

新增分层：

```text
has_lowering(op)
is_compile_certified(call)
is_runtime_certified(call)
is_production_certified(call)
```

---

# 119. Q Backend 与 Canonical PlanNode ABI 仍不一致

Canonical `PlanNode` 当前字段是：

```text
op
inputs
attrs
semantic_attrs
node_id
```

Q backend 仍使用类似：

```text
node.children
node.operator
node.params
```

这是直接 runtime ABI mismatch。

## 修复

禁止 Q backend 自己重新定义 LogicalPlan 视图。

建立一个共享：

```text
PhysicalRegionNodeView
```

或直接消费 canonical PlanNode。

转换：

```text
node.inputs
node.op
node.attrs
node.semantic_attrs
```

## 必测

```text
column
literal
unary
binary
rolling
multi-input
shared DAG
```

全部使用真正：

```text
planner.logical_plan.PlanNode
```

而不是 test-local fake object。

Hard Gate：

```text
Q_CANONICAL_PLANNODE_ABI_EXACT
```

---

# 120. Q Compiler dispatch order 会让 specialized lowering 永远到不了

当前 `compile_operator()` 先判断：

```text
q_func.startswith("{")
```

再进入某些 specialized branch。

这意味着映射成 lambda 的：

```text
ts_beta
wma
clip
ts_median
ts_var
ts_skew
ts_kurt
...
```

可能先被 generic lambda 分支截获。

结果：

```text
参数被忽略
window 被忽略
专门语义分支 unreachable
```

## 正确结构

不要根据 q code string 推断 dispatch class。

使用 typed lowering：

```python
@dataclass(frozen=True)
class QLoweringSpec:
    canonical: str
    lowering_kind: QLoweringKind
    compiler_fn: Callable
    parameter_schema: ...
```

例如：

```text
BINARY_INFIX
UNARY_FUNCTION
ROLLING_WINDOW
ROLLING_PAIRWISE
CROSS_SECTIONAL
GROUP
CONDITIONAL
STATEFUL
CUSTOM
```

然后：

```text
compiler_fn(bound_call, contract)
```

## Hard Gate

```text
Q_ZERO_STRING_SHAPE_DISPATCH
Q_ZERO_UNREACHABLE_SPECIALIZED_LOWERING
```

---

# 121. Q rolling semantics 需要重写，不是补字符串

以下当前 mappings 必须全部重新以 direct oracle 验证：

```text
ts_mean
ts_sum
ts_std
ts_var
ts_median
ts_skew
ts_kurt
ts_corr
ts_cov
ts_beta
ts_rank
ts_zscore
wma
ema
lag
round
```

重点：

## ts_corr / ts_cov

q：

```text
cor
cov
```

本身是 aggregate。

不能假设：

```text
window cor[x;y]
```

就天然变 rolling correlation。

必须真正使用：

```text
q rolling/adverb/window formulation
```

并处理：

```text
finite pair
min_periods
ddof
current-row missing policy
```

## ts_std

必须对齐：

```text
ddof
```

不能把 q `dev/mdev` 默认语义当成 canonical。

## round

类似：

```text
`long$x+0.5
```

对：

```text
negative
half values
large float
NaN
Inf
```

都需要定义。

## lag

period > 1 必须用真实 q oracle。

## rank

冻结：

```text
tie method
null
Inf
percentile denominator
range
```

---

# 122. Q DataAccess 输入仍未建立真正生产 contract

虽然 missing `ctx.base_data` 已从：

```text
empty DataFrame
```

改成 typed error，这是进步。

但生产结构仍不能是：

```text
QBackend
→ ctx.base_data
```

正确：

```text
DataAccess ReadArtifact
+
DataSnapshotIdentity
+
UniverseSnapshotIdentity
+
CalendarIdentity
+
FieldSchema
+
PhysicalRegion input schema
→ QRegionInputBundle
```

## QRegionInputBundle

至少：

```text
input_name
logical_field_ids
physical columns
timestamp column
instrument column
grain
timezone
calendar
sortedness
uniqueness
null policy
data snapshot id
universe id
source artifact representation
```

Q backend 只消费 bundle。

不能：

```text
自己去 DataAccess 查字段
自己猜时间列
自己重做 PIT
```

---

# 123. Q Output Contract 当前过度硬编码

当前如果固定：

```text
timestamp
instrument
value
float64
daily
allow_nulls=False
```

不适用于所有 operator。

例如：

```text
event/bool
bucket/int
intraday
warmup NaN
research state
```

正确 contract 由：

```text
PhysicalRegionPlan
```

带入：

```text
OutputSchemaContract
```

字段：

```text
timestamp_col
instrument_col
value_cols
dtype family
grain
timezone
nullable
allow_nan
allow_inf
sortedness
unique_key
```

生产不能：

```text
缺 value → 猜第一列
```

Deprecated guessing path 应删除或 research-only。

---

# 124. Q NULL / Type Semantics 仍是明确 P0

当前 object conversion：

```python
fillna("").astype(str)
```

会把：

```text
missing symbol
```

和：

```text
真实空字符串
```

合并。

这对：

```text
industry
group key
event label
symbol
categorical state
```

是 correctness bug。

## 必须建立 QTypeContract

覆盖：

```text
bool / nullable bool
byte
short
int
long
real
float
symbol
char
string
date
timestamp
timespan
datetime
```

映射：

```text
q null sentinel
↔
Arrow nullable type
↔
Pandas extension dtype
```

## 禁止

```text
null → ""
null int32 sentinel applied to all integer widths
timezone silently dropped
```

## Runtime certification

没有合法 PyKX/q runtime：

```text
Q_NULL_RUNTIME = NOT_RUN
```

而不是 PASS。

---

# 125. 当前 ts_corr“已修”只覆盖了一条路径

最近修复：

```text
ts_corr min_periods + root bug
```

是有价值的。

但 current HEAD 仍存在：

```text
cleaned_operators/common/polars_ts_rolling.py
TSCorrNative
```

使用：

```text
rolling_mean
→ pointwise demean
→ second rolling_mean
```

这不是当前窗口 Pearson correlation 的直接定义。

因此：

```text
canonical ts_corr = fixed
```

仍然是假命题。

真正状态应：

```text
Pandas/selected Polars-long path: improved
registered Polars native path: FAIL
```

## 修复策略优先级

第一选择：

```text
如果该 registered implementation 与 Polars Long 重复且不必要：
DELETE / DE-REGISTER
```

第二选择：

```text
若保留：
用 canonical oracle 重写并独立认证
```

不要维护 3 套一样的 rolling corr 实现。

---

# 126. ts_cov / rolling regression 同样存在 window-local 错误

同文件：

```text
TSCovNative
TSRegressionSlopeNative
TSRegressionInterceptNative
TSRegressionResidNative
```

都需要重点检查：

```text
历史点自己的 rolling mean_i
```

是否被错误用来构造：

```text
当前窗口 t 的 covariance/regression
```

正确 rolling OLS 当前窗口：

\[
\bar x_t =
\frac{1}{n_t}\sum_{i\in W_t}x_i
\]

\[
\bar y_t =
\frac{1}{n_t}\sum_{i\in W_t}y_i
\]

\[
\beta_t =
\frac{
\sum_{i\in W_t}(x_i-\bar x_t)(y_i-\bar y_t)
}{
\sum_{i\in W_t}(x_i-\bar x_t)^2
}
\]

不能：

```text
每个 i 都减 mean_i
再 rolling
```

---

# 127. Polars `ts_moment` / `ts_kurt` 仍存在相同数学反模式

current HEAD：

```text
mean = rolling_mean(x)
((x - mean)**n).rolling_mean()
```

并不等于：

\[
\frac{1}{n_t}\sum_{i\in W_t}(x_i-\bar{x}_t)^k
\]

因此今晚新增 AST miner：

```text
RollingStatOfCurrentPoint
```

模式：

```text
rolling_stat(x)
→ elementwise(x, rolling_stat)
→ rolling_aggregate(...)
```

全部列出。

至少检查：

```text
moment
skew
kurt
corr
cov
regression
expected shortfall
tail conditional mean
trimmed/winsorized
conditional moments
```

---

# 128. Recent ddof 修复后又暴露 Canonical Parameter ABI 漂移

不同 physical implementation 当前存在：

```text
window
d
min_periods
ddof
```

不同命名/不同缺省。

例如某 Polars native：

```text
param_names = x,d,ddof
```

另一条：

```text
x,window,ddof,min_periods
```

不能让 backend 物理实现拥有自己的逻辑 signature。

## 目标

唯一：

```text
CanonicalOperatorContract
```

例如：

```text
ts_std(
    x,
    window,
    *,
    min_periods,
    ddof
)
```

所有 legacy：

```text
d
period
n
```

只是 alias。

Physical implementation 输入必须是：

```text
BoundCanonicalCall
```

不能再自己解析 aliases。

Hard Gate：

```text
ALL_PHYSICAL_IMPLEMENTATIONS_MATCH_CANONICAL_BOUND_SIGNATURE
```

---

# 129. `ridge` P0 在最新 HEAD 完全仍在

当前明确问题仍是：

```text
双输入 y 分支 return NaN
alpha positional off-by-one
*args 模糊 ABI
broad Exception → entire column NaN
```

今晚必须修。

但不建议继续把两种语义塞在：

```text
ridge
```

推荐：

```text
ts_ridge_trend
ts_ridge_regression_slope
```

如果旧 DSL 需要：

```text
ridge
```

则旧名只是 compatibility dispatcher。

---

# 130. FactorAssets ↔ DataAccess Adapter 当前不是“小 bug”，而是接口错位

current：

```text
factor_assets/adapters/data_access.py
```

存在：

```text
from dataaccess import ...
```

而 DataAccess 的实际安装 package：

```text
data_access
```

其次 adapter 构造：

```text
DataRequest(source=..., start_date=..., end_date=...)
```

但实际 DataRequest 是：

```text
fields
start
end
instruments
universe
pit
...
```

因此 adapter 不是 production-compatible。

另外：

```text
check_factor_availability()
```

只要拿到 handle 就返回 True。

没有：

```text
execute
finite coverage
dataset existence
snapshot
PIT
field availability
```

证明。

## 禁止

```text
date.today()
```

作为未指定 asof 的默认 production 语义。

Availability 必须返回 typed result：

```text
AVAILABLE
NOT_FOUND
PIT_UNAVAILABLE
DATA_GATED
SNAPSHOT_UNAVAILABLE
SCHEMA_MISMATCH
INFRA_FAILURE
```

并携带：

```text
asof
snapshot_id
source
reason
```

## Catalog

不能再返回伪造：

```json
{
  "factor_id": "...",
  "available": true,
  "source": "..."
}
```

必须真正读取：

```text
FactorAssets registry
或
DataAccess semantic catalog
```

明确 ownership。

---

# 131. Modeling “双包统一”目前仍偏文档级，不是物理级

current repo 仍有：

```text
factor_engine/modeling/*
modeling/modeling/*
```

且 standalone 包仍有：

```text
contracts
preprocess
exposure
adapter
```

所以：

```text
AUTHORITY.md
```

不能证明 split-brain 已消失。

## 最终结构二选一

### A. FactorEngine is authority

Standalone `modeling`：

```text
只 re-export / adapter
不再拥有 timing/split/preprocess correctness logic
```

### B. Shared canonical modeling-core

抽出：

```text
modeling_core
```

FactorEngine 与 standalone 都消费同一实现。

但不能：

```text
两份 SplitSpec
两份 OOS validator
两份 preprocessing lifecycle
```

---

# 132. Modeling gap_days 仍不是机构级时序泄漏 contract

自然日：

```text
(validation_start - train_end).days
```

不是量化时序泄漏最稳妥定义。

需要区分：

```text
calendar days
exchange sessions
bars
label maturity interval
purge
embargo
```

例如 H=20 日 VWAP-to-VWAP label：

```text
训练样本 t 的 label 使用到 t+20
```

即使：

```text
gap_days=1
```

仍会泄漏。

## 最终 contract

```text
LabelInterval
PurgeSpec
EmbargoSpec
ExchangeCalendar
```

共同决定 split legality。

## 必须检查三条边界

```text
train → validation
train → test
validation → test
```

以及：

```text
无 validation 的 train→test
```

---

# 133. QuantEvaluator 修复后的新 shape bug

current：

```python
return quantiles.squeeze() if quantiles.shape[2] == 1 else quantiles
```

当：

```text
T=1
或
N=1
```

时会把不该删除的 axis 一起 squeeze。

Numba 路径也相同。

正确：

```python
return quantiles[:, :, 0] if F == 1 else quantiles
```

## 必测 shape

```text
1 × 1 × 1
1 × N × 1
T × 1 × 1
T × N × 1
T × N × F
```

API 返回 shape 必须稳定。

---

# 134. Quantile tie-heavy / empty-bin 是生产诊断问题

当前 assignment 允许 ties 按：

```text
left/right
```

进入相同边界。

对于：

```text
大面积相同值
binary factor
event factor
```

可能出现：

```text
某些 quantile 空
```

这不一定是 bug。

但 Evaluator 必须输出：

```text
effective_bin_count
bin_counts
empty_bins
tie_fraction
unique_value_count
quantile_assignment_method
```

不能只返回：

```text
NaN return
```

否则用户无法判断：

```text
因子没收益
还是分层根本没形成。
```

---

# 135. Root CI 当前仍不存在真正总门

latest workflows 仍然只有专项：

```text
autofactor-platform-integration
factor-cold-start-library
factor-pack-evaluation
gtja191-production-gate
operator-surface-gate
```

没有：

```text
quant-platform-production-gates
```

latest HEAD 也没有可用 workflow evidence。

今晚必须补。

但 root CI 不要：

```text
一个 job 跑 pytest 全仓
```

而要成为：

```text
aggregator
```

每个 domain 独立 job：

```text
repo-integrity
dataaccess
factorengine-pandas
factorengine-polars-long
factorengine-polars-native
duckdb
q-static
modeling
evaluator
factor-assets
cold-start
clean-wheel
evidence
```

最后：

```text
production-gate
needs: all
```

---

# 136. DataAccess build identity P0 在最新 HEAD 仍完全存在

当前仍同时有：

```text
pyproject:
  version = 0.10.2

_build_info.py:
  0.11.0.dev0+untagged
  commit_id=None

_build_meta.py:
  env / runtime git rev-parse

core/build_metadata.py:
  又一套 SCM/build-time resolver
```

属于：

```text
BUILD IDENTITY SPLIT-BRAIN
```

## 更严重的一点

legacy fallback：

```text
build_sha="unknown"
build_time="unknown"
dirty=False
```

而 production validator 若只判断：

```python
if not build_sha
```

那么字符串：

```text
"unknown"
```

会被当成合法。

## 目标

只有一份：

```text
BuildIdentity
```

在 build time 生成：

```text
package_version
commit_sha_full
tree_dirty
build_id
build_timestamp
dependency_lock_hash
```

安装运行时：

```text
只读
```

不再：

```text
git rev-parse
```

---

# 137. FE 自己也仍有 correctness identity fail-soft

`runtime/engine.py` 当前：

```text
_data_source_to_scope_config
_stable_config
```

对未知对象会：

```text
str(value)
```

然后：

```text
compute_source_scope_hash
```

使用：

```text
json.dumps(... default=str)
sha256[:16]
```

这与 DataAccess canonical identity 的 strict 原则冲突。

对于 CSE scope correctness：

```text
不同对象 repr/str collision
```

可能让不应共享的 source scope 共享。

## 修复

FactorEngine 不再自己发明 source identity。

直接消费：

```text
DataAccessSourceIdentity
```

或共同：

```text
CanonicalIdentityEncoder
```

内部 full SHA-256。

短 16 hex：

```text
仅 UI/log
```

---

# 138. BatchGlobalOptimizer 仍是 scaffold，旧问题没有被这轮修掉

current 仍有：

```text
all_nodes 只直接放 shared + roots
每 root 单独 _optimize_tree
SQL backend TODO
estimated_rows=10000
estimated_memory_bytes=0
TransferEdge=[]
native_fraction=1.0
按 backend name 分组
```

所以它不能称：

```text
true batch-global region optimizer
```

## 目标

先建立完整 DAG：

```text
node_id → PlanNode
parents
children
consumer_count
estimated shape
eligible physical implementations
```

再做：

```text
global assignment
connected component region formation
boundary edge construction
topological ordering
memory/liveness
```

## 不允许

```text
相同 backend 但 DAG 不相连
→ 硬塞进一个 region
```

Region 必须：

```text
connected
execution-compatible
representation-compatible
```

---

# 139. q 仍不在主 Backend Factory / Router

当前：

```text
backend.factory
```

没有：

```text
q
q_kdb
```

`BackendRouter.RequestedBackend` 也没有 q。

因此即使 q package 写了很多：

```text
它仍不是统一 production routing 的一部分。
```

今晚不要急于把 q 塞进 production。

正确顺序：

```text
1. 修完整性
2. 修语义
3. 完成静态/compile/runtime/parity evidence
4. RegionPlanner 支持
5. Factory/Router research route
6. performance benchmark
7. production admission
```

没有第 3 步：

```text
第 7 步永远 NO。
```

---

# 140. “678 research_only”历史状态必须彻底重算

历史 commit 的确：

```text
Added research_only=True to 678 operators
```

但真实：

```text
register_operator(...)
```

当前没有：

```text
research_only
production_ready
```

这些 keyword。

而 latest HEAD 的相关源码中：

```text
research_only=True
```

又已经不在很多文件里。

因此：

```text
678
```

不能继续当当前 production truth。

## 更根本的问题：audit script 自己与 Registry ABI 不一致

当前 placeholder auditor 假设：

```text
decorator production_ready/research_only
```

决定生产状态。

但真实 registry 是：

```text
status
operator policy
mining role
evidence
```

等组合。

所以 audit script 会给出：

```text
“0 critical violations”
```

但这个数字本身未必代表 production admission 安全。

## 今晚必须重写 audit

真实查询：

```text
OperatorRegistry
OperatorPolicy
MiningRole
PhysicalImplementationSpec
Evidence
```

生成：

```text
OperatorAdmissionTruth
```

---

# 141. 对所有历史 placeholder 的正确处理不是“一键降级”

用户要求：

```text
凡是能构造因子的，尽可能做成真正可用。
```

这个目标是合理的。

但不能：

```text
所有 678 都 promotion
```

必须分 5 类。

## A — 有经济含义 + 数据可用 + 算法清晰

```text
IMPLEMENT + CERTIFY
```

优先级最高。

例如：

```text
rolling shape
event frequency
distribution shift
state lifecycle
intraday daily aggregation
robust statistics
```

只要真实字段/语义支持。

## B — 已经有等价 canonical

```text
DEDUPE → ALIAS / DELETE
```

不要再实现第二份。

## C — 需要未接入数据

```text
DATA_GATED
```

如：

```text
期权
分析师
新闻
产业链
高频订单簿
```

如果当前 DataAccess 没字段：

```text
不能伪造。
```

保留 contract + required_fields。

## D — 模型/训练/诊断结构

不适合直接作为 alpha terminal：

```text
MODEL_COMPONENT
STATE
CONDITION
DIAGNOSTIC
RECIPE_INTERNAL
```

可用，但 Mining AST position 限制。

## E — 统计上/经济上不值得保留

```text
DELETE
```

目标是：

```text
更强的算子库
```

不是：

```text
更多名字。
```

---

# 142. 8 小时算子 Rehabilitation（修复与恢复）目标

今晚不要试图一次把数千算子做昂贵人工证明。

用分层：

## Tier 0：Core Primitive

必须 100%：

```text
math
lag/delta/returns
rolling mean/sum/std/var/min/max/median
corr/cov
rank/zscore
cs/group
neutralize primitives
```

## Tier 1：High-value Factor Operators

必须优先真实实现：

```text
trend
reversal
volatility
liquidity
volume/turnover
distribution/tail
drawdown
shape
breakout/compression
event/state
technical indicators
```

## Tier 2：Fundamental / PIT

只在字段/vintage/available_at contract 真正存在时 promotion。

## Tier 3：Intraday→Daily

高价值但必须 session-safe。

## Tier 4：Research/high-cost

```text
mutual information
entropy
graph
topology
matrix profile
advanced nonlinear
```

保留：

```text
HIGH_COST / RESEARCH
```

不要求今晚全部 DirectUse。

---

# 143. 模型算子不能被“因子算子化”到 fit 泄漏

用户要求模型算子也尽量可用。

正确做法不是：

```text
operator.calculate()
里面临时 fit model
```

而是：

```text
TrainingSpec
→ fit
→ Validation
→ freeze ModelArtifact
→ score_asof
→ ModelDerivedFactor
```

## 生产模型核心优先级

```text
Ridge
Huber
ElasticNet
PCR
PLS
Regime Model
MoE
LightGBM
CatBoost
```

其中：

```text
Pandas/Sklearn-like baseline
```

先保证 lifecycle。

再扩：

```text
MLP
TabNet
TabPFN（小样本/适用条件）
Transformer
TCN
FactorVAE
```

但模型复杂度不是今晚核心 KPI。

## Model-derived factor 必须绑定

```text
training cutoff
feature set
label contract
split plan
hyperparameters
model artifact hash
preprocess artifact
universe
calendar
data snapshot
```

---

# 144. Backend 扩展原则：百亿私募级不是“后端越多越好”

当前核心建议仍是：

```text
Pandas/NumPy = reference
Numba = targeted kernel acceleration
Polars Lazy/Expr = in-memory columnar production
DuckDB SQL = scan/pushdown/window production
Arrow = interchange
ClickHouse = remote analytical backend where deployed
q/K = high-fit research backend until fully certified
```

可评估但不应今晚强推 production：

```text
cuDF/CuPy GPU
JAX
Ray/Dask distributed
```

## GPU 只有在：

```text
data resident on GPU
or
compute intensity >> PCIe transfer cost
```

才有意义。

## Distributed 只有在：

```text
single-node RAM/CPU 实际成为瓶颈
```

才引入。

A 股日频多因子通常：

```text
单机 columnar + batch DAG + 良好 CSE
```

先能吃掉很大规模。

不要为了“百亿”过度架构。

---

# 145. DataAccess 性能今晚需要做的不是重写

在 correctness 闭环后，优先：

```text
projection pushdown
predicate pushdown
partition pruning
field coalescing
batch read
one scan multiple factors
Arrow-native result
Polars Lazy scan
DuckDB relation
manifest/resolution job cache
resource governor
```

## 必须测

```text
files scanned
bytes scanned
rows scanned
columns scanned
read amplification
time to first batch
peak RSS
```

## DataAccess → FactorEngine

目标：

```text
一个 batch 的字段需求
→ Analyzer 合并
→ DataAccess 一次/少数 scan
→ 原生 columnar artifact
→ Factor DAG 多因子共享
```

不要：

```text
一个 factor
→ read once
```

---

# 146. 企业级 Observability（可观测性）补强

百亿级别真正重要的是：

```text
知道哪里坏了
```

每个 production factor batch 输出：

```text
run_id
code_build_id
data_snapshot_id
universe_snapshot_id
calendar_id
factor_set_id
backend_plan_id
factor_count
success_count
all_nan_count
PIT_rejection_count
backend_fallback_count
bytes_scanned
peak_memory
elapsed
```

每个 operator：

```text
input coverage
output coverage
Inf
NaN
constant
exception reason
backend
implementation id
```

## 告警

```text
coverage collapse
all-NaN
sudden scale change
backend fallback
PIT unavailable
snapshot changed
performance regression
```

这些要进入以后实盘 streaming health。

---

# 147. 报告/文档 Truth Integrity（事实完整性）

仓库已经积累很多：

```text
FINAL
COMPLETION
ALL PASS
PRODUCTION READY
```

历史文档。

增加 script：

```text
scripts/audit_report_claim_integrity.py
```

扫描：

```text
PRODUCTION READY
ALL PASS
COMPLETE
FINAL
```

如果文档没有：

```text
bound_commit_sha
evidence_artifact
evidence_hash
generated_by
```

标记：

```text
HISTORICAL_UNVERIFIED
```

不删除历史。

但用户/Agent 以后不能把它当 current truth。

---

# 148. Failed-Agent Recovery Manifest

新增：

```text
loop_engineering/failed_agent_recovery.jsonl
```

每次 API failure 记录：

```json
{
  "agent_id": "...",
  "base_sha": "...",
  "worktree": "...",
  "write_scope": [],
  "error": "Content block not found",
  "dirty_files": [],
  "unexpected_truncations": [],
  "import_status": "...",
  "recovery_action": "discard|repair|keep",
  "recovery_commit": null
}
```

长期来看这能判断：

```text
API failure 到底多大概率产生 partial write。
```

---

# 149. Writer Agent 的文件截断探测

不能只用：

```text
file size < previous
```

但可以做 heuristic：

```text
source file LOC 下降 >30%
exported class/function 数下降
public symbols消失
AST top-level nodes大幅减少
```

则自动：

```text
QUARANTINE
```

除非任务本来就是删除/重构。

这次 `q_executor.py`：

```text
-338 lines
+17 lines
```

就是非常典型的触发案例。

---

# 150. 晚上 8 小时 Wave Schedule（ChiefCoordinator 的执行编排）

这里的小时只是你给 Loop Engineering 的**任务预算分配**，不是外部承诺。

原则：

```text
wave 结束早 → 自动进入下一 wave
wave 被 blocker 卡住 → 留最小 recovery agent，其他无依赖工作继续
```

## Wave 0 — Repository Recovery（0:00–0:30）

目标：

```text
current HEAD 可 import
失败 Agent 不再留下半文件
```

任务：

```text
q_executor recovery
compileall
package import
registry load
minimal FE run
minimal DA plan/read fixture
safe serial mode
worktree policy
```

DoD：

```text
REPO_INTEGRITY_GREEN
```

---

## Wave 1 — Q honesty + blocker closure（0:30–1:30）

不是把 q “强行 production”。

完成：

```text
one capability authority
PlanNode ABI
compiler typed dispatch
NULL/type contract scaffold
input/output contract
fan-in
workspace lifecycle
```

若无 q runtime：

```text
production Q = disabled
runtime/parity = NOT_RUN
```

不拖住其他后端。

---

## Wave 2 — Core FactorEngine operator correctness（1:30–3:00）

并行最多：

```text
1 writer
1 oracle/auditor
1 reviewer
```

重点：

```text
ridge
corr/cov
moment/skew/kurt
regression
zscore
tail/ES
trimmed/Qn
technical indicators
return_decomp
canonical parameter ABI
duplicate physical path removal
```

DoD：

```text
Tier-0 core primitive production surface certified
Tier-1 highest-value families substantially increased
```

---

## Wave 3 — DataAccess + PIT（3:00–4:15）

重点：

```text
build identity
canonical identity
request freeze
snapshot
credential generation
calendar
universe
revision vintage
PIT poison
FactorAssets adapter
```

DoD：

```text
PIT poison smoke green
clean identity path
```

---

## Wave 4 — Operator Rehabilitation + cold-start（4:15–5:15）

重新计算当前：

```text
registered
production candidate
mining eligible
research
data-gated
duplicate
```

对可构造因子的：

```text
implement/certify
```

然后：

```text
cold-start dependency graph
```

重算。

DoD：

```text
没有 placeholder 被误当 production
有价值 placeholder 不因“一刀切降级”永久丢失
```

---

## Wave 5 — Planner / Backends / Performance（5:15–6:15）

重点：

```text
real batch DAG
region planner
TransferEdge
native buffer
CSE
Arrow
Polars Lazy
DuckDB
Numba
q research region
```

benchmark：

```text
daily realistic
multi-factor batch
```

DoD：

```text
correctness unchanged
backend switches/materializations下降
```

---

## Wave 6 — Modeling / Evaluator / FactorAssets（6:15–7:00）

重点：

```text
single modeling authority
session-based purge/embargo
label maturity
QE shape
quantile degeneracy
top-bottom property
FA adapter
```

DoD：

```text
no known leakage
stable array contracts
```

---

## Wave 7 — Full Gates / CI / Evidence / Red Team（7:00–8:00）

执行：

```text
compileall
package imports
targeted full suites
root CI
clean wheel
PIT
cold-start
benchmark
report integrity
stale evidence
red-team
```

最终必须生成：

```text
FINAL_SHA
OPEN_ISSUES
UNKNOWN
NOT_RUN
CERTIFIED_MATRIX
PERF_DELTA
```

禁止：

```text
为了在 8 小时结束前看起来全绿
而 xfail/skip/demote。
```

---

# 151. 8 小时 Agent 配置建议

API 稳定性当前差。

默认：

```text
ChiefCoordinator: 1
Writer: 1
Read-only Audit/Oracle: 1
Independent Reviewer: 1
```

最多同时：

```text
4 个 Agent
```

其中真正改源码：

```text
最多 1 个
```

若 60 分钟内稳定：

```text
Writer 可升到 2
```

但两者必须：

```text
不同 worktree
不重叠 write scope
```

**不要再一次发 11 个 writer。**

对于纯 scan：

```text
可多一些 read-only Agent
```

但结果由 Coordinator 合并。

---

# 152. 今晚新 Hard Gates

加入：

```text
REPO_ZERO_PARTIAL_SOURCE_FILE
REPO_PUBLIC_ABI_IMPORTS_CLEAN
FAILED_AGENT_DIRTY_WORKTREE_NOT_AUTO_MERGED
HARD_GATE_ZERO_LITERAL_PASS_AS_EVIDENCE

Q_EXECUTOR_PUBLIC_ABI_COMPLETE
Q_CAPABILITY_ONE_AUTHORITY
Q_PLANNODE_ABI_EXACT
Q_TYPED_LOWERING_DISPATCH
Q_ZERO_UNREACHABLE_SPECIALIZED_LOWERING
Q_NULL_EMPTY_STRING_DISTINCT
Q_DATAACCESS_INPUT_CONTRACT
Q_OUTPUT_CONTRACT_PARAMETRIC

ALL_CANONICALS_HAVE_PHYSICAL_IMPLEMENTATION_INVENTORY
ALL_SELECTABLE_PHYSICAL_IMPLS_CERTIFIED_OR_DISABLED
ALL_PHYSICAL_IMPL_SIGNATURES_CANONICAL
POLARS_ZERO_NESTED_ROLLING_CENTER_MOMENT_BUG
POLARS_CORR_COV_DIRECT_WINDOW_ORACLE

FA_DA_ADAPTER_PUBLIC_API_REAL
FA_DA_AVAILABILITY_ZERO_FAKE_TRUE
FA_DA_ZERO_WALL_CLOCK_DEFAULT

MODELING_ONE_EXECUTION_AUTHORITY
MODELING_SPLIT_LABEL_MATURITY_SAFE
MODELING_ALL_SPLIT_BOUNDARIES_SAFE

QE_QUANTILE_SHAPE_STABLE
QE_TIE_DEGENERACY_EXPLICIT

DA_BUILD_ONE_AUTHORITY
FE_DA_IDENTITY_ENCODER_SHARED
FE_ZERO_DEFAULT_STR_CORRECTNESS_IDENTITY

COLD_START_CURRENT_HEAD_STATUS_RECOMPUTED
PLACEHOLDER_AUDIT_USES_REAL_REGISTRY_LIFECYCLE

ROOT_CI_PRESENT_AND_CURRENT_SHA_BOUND
REPORT_CLAIM_INTEGRITY_GREEN
```

---

# 153. 今晚最后的 DoD（Definition of Done，完成定义）

8 小时结束时，不要求虚假承诺：

```text
世界上没有任何未知 bug
```

要求：

```text
1. current HEAD 没有已知 import/runtime broken state
2. 所有发现的 P0 有明确 FIX / DISABLED / OPEN
3. Tier-0 FactorEngine primitives 全部当前实现路径有 truth matrix
4. 有价值 Tier-1 算子显著提升真实可用性
5. DataAccess PIT/build/snapshot identity 关键链闭环或明确 blocker
6. q 不再是假 production-safe；能修则修，不能 runtime 验证就关闭 production
7. cold-start 当前状态重新计算
8. Modeling/QE/FA 的新红队 bug处理
9. performance 有真实 before/after
10. root CI/evidence 已绑定最终 SHA
11. Failed Agent 不再能把 partial writes 自动带进 main
12. UNKNOWN / NOT_RUN 明确列出
```

这才接近真正大型量化机构需要的工程标准。

