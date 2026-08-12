# FactorEngine × DataAccess R41 最新 HEAD 全域终审与一次性全量整改任务书
## —— 模型数学正确性、Artifact 生命周期、Evidence Truth、生产硬门、服务安全、构建依赖、DataAccess R30、长期生产化

> **可直接整份发送给 Coding AI / Codex / Cursor / Claude Code 执行。**
>
> 仓库：`https://github.com/18047533889/quant_projects`
>
> 本轮审计时最新 `main`：
>
> ```text
> 4b1577fce14c097fba48619885b38d6c596242eb
> ```
>
> **执行前必须重新读取最新 main HEAD。** 如果执行时 HEAD 已变化，以执行时 HEAD 为真实基线。
>
> 本文件是对仓库现有 `FactorEngine_最新HEAD_260项全量整改提示词_20260811.md` 的后续独立终审。旧 1–260 项不作废，必须在执行时最新 HEAD 上重新验证。

# 0. 最终目标

最终系统必须满足：

```text
Factor / Model Request
→ immutable ExecutionSemanticContext
→ typed DSL / IR / ModelScoreIR
→ exact operator + field + source + universe + calendar contracts
→ parameter / backend / model-artifact certification
→ cost-driven physical plan
→ pinned DataAccess snapshot
→ real stage execution
→ FactorBlock / ModelScoreBlock
→ DQ / leakage / axis / numeric gates
→ streaming durable materialization
→ atomic catalog/artifact publication
→ current-build executable evidence
```

预测模型必须拥有独立生命周期：

```text
ModelSpec
→ TrainingDatasetCertificate
→ WalkForwardPlan
→ train-only sequential preprocessing
→ approved hyperparameter search
→ validation-only selection
→ refit with exact effective cutoff
→ label maturity / publish time
→ immutable signed/checksummed Artifact
→ durable ArtifactCatalog
→ as-of legal resolution
→ pure FrozenScorer
→ OOS evaluation
→ promotion / rollback
→ drift / OOD monitoring
```

# 1. 禁止“假修复”

以下一律不得算完成：

1. 只改注释、README、ledger、evidence JSON。
2. 只把 gate 写成 `True`。
3. `except Exception: pass` 后继续把 evidence 标 PASS。
4. production hard gate 失败只 warning/telemetry 但仍执行。
5. production trainer 报错后 evidence runner 自动走另一套 fallback trainer。
6. 只证明类/函数存在，不执行其负控。
7. stale evidence 仍写 `CURRENT_HEAD_FRESH=true`。
8. wheel 源码树可 import 就当 clean install 可用。
9. dataclass `frozen=True` 但内部 dict/list/ndarray 仍可变，却宣称 artifact immutable。
10. 对 PIT / availability / market / calendar / source identity 使用字符串猜测。
11. 用今天的 universe / adjustment / artifact 回放历史。
12. 对未知 feature availability 默认“已可用”。
13. 没有 validation 时用 in-sample score 选生产模型。
14. 不收敛模型仍发布，只在 metadata 写 `converged=False`。
15. 证书校验失败只计数不阻断 production。
16. `NOT_RUN` 在生产 CI 中当成功。
17. 测试 fixture 自己构造非法/未初始化对象，任意异常都算“负控触发成功”。
18. empty allowlist 解释为 allow-all。
19. `default=str` / `repr()` 进入 production semantic identity。
20. 通过 fallback 掩盖真实 API 或数学错误。
21. 为追求速度关闭 PIT/DQ/certification。
22. 只改 FactorEngine 不验证 DataAccess ABI；反之亦然。
23. 只改 repo checkout，不验证 wheel/container 部署。
24. P0 无 current-SHA 可执行证据就宣布 CLOSED。
25. 旧 ledger 写 CLOSED 不等于最新 HEAD 已闭环。

# 2. 旧 ledger 明确未闭项

## 2.1 R39

```text
PERF-017 PARTIAL   native backend 源头 emit slice 未做
PERF-020 OPEN      process worker / PlanId lane
PERF-073 OPEN      DataAccess ScopedConnectionPool
PERF-078 OPEN      NativePipelineRegion optimizer
PERF-079 OPEN      RollingStateBlock multi-output native kernel
PERF-080 OPEN      PrimitiveBlock 跨因子 primitive extraction / materialization
```

另需重新复核：

```text
PERF-032 TemporaryArrayArena 标 CLOSED，但 ledger 写明未接 hot consumer
PERF-082 ProductionExecutionCertificate 标 CLOSED，但 HybridExecutor 当前明确失败只 warning，不阻断
```

## 2.2 R40 PARTIAL / OPEN

至少重新处理：

```text
#96 #137 #153 #169 #173 #175 #178 #179 #181 #189 #190 #191
#199 #200 #205 #206 #209 #211 #212 #252 #254 #256
```

# 3. 本轮新增问题

从 R41-261 开始。

---

# 4. 构建、打包、版本与 Evidence 基础设施（R41-261～280）

## R41-261 P0 — `factor_engine/modeling/` 没有进入 FactorEngine wheel

### 当前代码

`factor_engine/pyproject.toml` 的 packages.find 包含：

```text
api*
backend*
...
market*
mining*
...
runtime*
security*
semantic*
service*
storage*
```

但没有：

```text
modeling*
```

而仓库已经正式新增：

```text
factor_engine/modeling/
factor_engine/modeling/learners/
```

### 风险

repo checkout 中：

```text
pytest 能 import modeling
```

但 wheel 安装后：

```text
ModuleNotFoundError: modeling
```

导致模型层只在源码环境“看起来可用”。

### 必须怎么改

在 package discovery 中加入：

```toml
"modeling*",
```

不要只改 include 文本。CI 必须真实执行：

```bash
python -m build
python -m venv /tmp/fe-r41-clean
pip install dist/*.whl
python -c "import modeling; import modeling.trainer; import modeling.predictor"
```

然后在 clean venv 做最小闭环：

```text
create PanelDataset
→ train PCR
→ save artifact
→ new process load artifact
→ predict
```

### Hard Gate

```text
FE_CLEAN_WHEEL_MODELING_IMPORT == PASS
FE_CLEAN_WHEEL_MODEL_TRAIN_SAVE_LOAD_SCORE == PASS
```

---

## R41-262 P0 — DataAccess `r30/` 已存在，但没有进入 wheel 显式 package list

### 当前代码

`dataaccess/r30/__init__.py` 已存在，并包含大量：

```text
calendar_snapshot
universe_snapshot
experiment_snapshot
coverage_service
partition_index
session
change_impact
resolution_lease
...
```

但 `dataaccess/pyproject.toml` 的 packages 只列：

```text
data_access
data_access.core
...
data_access.snapshot
```

没有：

```text
data_access.r30
```

### 风险

源码环境可用：

```python
import data_access.r30
```

正式 wheel 不一定包含。

更严重的是 FE 最新代码已经开始依赖 R30/R40 DataAccess 能力，repo 测试与部署能力会分叉。

### 必须怎么改

显式加入：

```toml
"data_access.r30",
```

及 package-dir：

```toml
"data_access.r30" = "r30"
```

如果 r30 下未来形成子包，wheel inventory 必须递归发现。

### Hard Gate

```text
DA_CLEAN_WHEEL_IMPORT_DATA_ACCESS_R30 == PASS
DA_R30_ALL_PHYSICAL_MODULES_PRESENT_IN_WHEEL == PASS
```

---

## R41-263 P0 — DataAccess 版本事实源分裂：pyproject=0.10.2，r30.__version__=0.10.3

### 风险

以下可能分别报告不同版本：

```text
pip metadata
data_access.__version__
data_access.r30.__version__
service /health
FE DataAccess capability handshake
evidence
```

版本不再能唯一定位运行时行为。

### 必须怎么改

只留一个版本 authority：

推荐：

```text
distribution metadata
+
BuildManifest
```

内部模块禁止独立手写版本。

例如：

```python
from importlib.metadata import version
__version__ = version("data-access")
```

`r30` 若需要 API generation：

```text
R30_API_GENERATION = 30
```

不要伪装成 package version。

### Hard Gate

```text
DATA_ACCESS_VERSION_AUTHORITIES == 1
```

---

## R41-264 P0 — DataAccess wheel inventory 脚本本身会漏检 r30

`dataaccess/scripts/check_wheel_inventory.py` 的 `ALLOWED_PKG_DIRS` 是硬编码列表，并不包含 r30。

因此：

```text
r30 没进 wheel
+
wheel inventory 脚本仍 PASS
```

### 必须怎么改

不要维护“允许检查的目录名单”。

改为：

```text
扫描项目根目录
→ 找所有应打包 package
→ 与 wheel 实际 members 比较
```

若需要排除：

```text
EXPLICIT_NON_PACKAGE_DIRS
```

只维护排除项。

并增加 mutation test：

```text
创建临时 fake package
→ 不改 pyproject
→ inventory 必须 FAIL
```

---

## R41-265 P0 — FactorEngine 对 DataAccess 的最低版本约束严重落后于真实 API 依赖

当前 FE 仍允许：

```text
data-access>=0.2.0
```

但当前仓库 DataAccess 已是 0.10.x，并且 FE 使用了：

```text
snapshot
scan_cost
semantic_catalog
resolution/read governance
R30 objects
```

### 必须怎么改

不要仅靠版本号猜 ABI。

新增：

```python
DataAccessCapabilityHandshake
```

至少返回：

```text
api_version
build_manifest_digest
supported_read_contract_version
snapshot_contract_version
query_graph_version
resource_lease_version
semantic_catalog_version
```

FE startup：

```text
required capabilities ⊆ provided capabilities
```

否则 hard-fail。

版本 constraint 同步收紧到实际最低验证版本。

---

## R41-266 P1 — 生产依赖没有 reproducible lock / constraints

当前大量依赖只给下限：

```text
numpy>=
pandas>=
pyarrow>=
scipy>=
polars>=
duckdb>=
sqlglot>=
```

### 风险

同一 commit：

```text
今天 pip install
vs
三个月后 pip install
```

可能得到不同数值语义、Arrow 转换、DuckDB SQL、Pandas rolling 行为。

### 必须怎么改

建立：

```text
constraints/production-py310-linux-x86_64.txt
constraints/production-py311-linux-x86_64.txt
...
```

BuildManifest 记录：

```text
resolved dependency versions
wheel hashes
Python ABI
platform
BLAS vendor/version
```

CI 至少维护：

```text
locked production
minimum supported
latest supported
```

三组矩阵。

---

## R41-267 P0 — `psutil/threadpoolctl` 被描述为生产资源约束，却仍是 optional extra

如果 production 没装：

```text
HostResourceCoordinator / live headroom
BLAS thread governance
```

可能降级。

### 必须怎么改

二选一：

A. 生产 base dependency；
B. startup gate：

```text
production + missing psutil/threadpoolctl
→ ProductionRuntimeDependencyMissing
```

不能静默“功能少一点”。

---

## R41-268 P0 — 最新 HEAD 没有 connector 可见的 CI status / workflow run

审计时：

```text
HEAD = 4b1577...
combined status = []
workflow runs = []
```

### 必须怎么改

给 main 建立 permanent CI：

```text
unit
integration
clean-wheel
PIT
multi-backend parity
R40/R41 hard gates
current-SHA evidence
security negatives
benchmark smoke
```

main branch protection 必须要求关键 checks。

---

## R41-269 P0 — Model Layer evidence 已 stale，却仍声称 CURRENT_HEAD_FRESH=true

当前：

```text
MODEL_HARD_GATES.json.git_sha = 6932ffd...
current HEAD = 4b1577...
```

但 JSON 中：

```text
MODEL_CURRENT_HEAD_EVIDENCE_FRESH = true
```

### 必须怎么改

Evidence 不允许自行写“fresh”。

CI 外部验证：

```python
assert evidence.build_manifest_digest == current_build_manifest.digest
assert evidence.source_commit == current_build_manifest.source_commit
```

freshness 是比较结果，不是生成脚本里的常量字段。

---

## R41-270 P1 — R39 closure report / ledger 绑定旧实施 HEAD，不能作为最新 HEAD 生产证明

R39 closure report 记录实施期 head 为旧 commit，但仓库已经继续演进。

### 必须怎么改

旧 report 保留历史审计价值，但新增：

```text
EvidenceApplicability
```

状态：

```text
CURRENT
STALE_BUT_REPRODUCIBLE
INVALIDATED_BY_DEPENDENCY_CHANGE
SUPERSEDED
```

最新 release 只能接受 `CURRENT`。

---

## R41-271 P0 — R40 hard-gate 脚本把 NOT_RUN 当 CI 成功

当前脚本规则：

```text
exit code = FAIL 数量
NOT_RUN 不算失败
```

### 风险

最关键证据：

```text
参数域证据缺失
registry 不能 import
calendar infrastructure error
```

都可能：

```text
NOT_RUN
+
exit 0
```

### 必须怎么改

模式分离：

```text
research audit:
  NOT_RUN allowed

production CI:
  PASS only
  FAIL or NOT_RUN → nonzero
```

命令：

```bash
python scripts/audit_r41_hard_gates.py --mode production
```

---

## R41-272 P0 — Model evidence freshness gate 实际只检查 `git_sha is not None`

`report_hard_gate_set()` 的逻辑相当于：

```python
MODEL_CURRENT_HEAD_EVIDENCE_FRESH = git_sha is not None
```

只要调用者传一个字符串，就 PASS。

### 必须怎么改

传入：

```text
ExpectedBuildIdentity
```

并读取生成物自己记录的：

```text
source_commit
build_manifest_digest
evidence_input_digest
```

做真实 equality。

---

## R41-273 P0 — Model evidence runner 在正式 trainer 任意异常时自动换 fallback trainer

当前逻辑：

```text
authoritative trainer
→ any Exception
→ _fallback_train_predict
→ evidence 继续生成
```

### 风险

生产 trainer 如果因为：

```text
PIT gate
sample gate
math bug
artifact bug
```

崩了，evidence 反而绕过它。

### 必须怎么改

生产 evidence：

```text
authoritative path failure → FAIL
```

fallback 只能显式：

```text
--research-fallback-demo
```

并且生成的 artifact/evidence 标：

```text
NON_AUTHORITATIVE
```

不得参与 production gate。

---

## R41-274 P0 — Model hard gates 大量是直接常量 True

例如“无 score-time fit”“purge 正确”“无 random split”“无 full-sample scaler”等，不少 gate 没有执行行为证明。

### 必须怎么改

每个 gate 绑定：

```text
test_id
executable probe
negative control
expected failure
current build
```

例如：

```text
MODEL_ZERO_SCORE_TIME_FIT
```

必须通过冻结 scorer 类型 + monkey negative control，而不是字符串说明。

---

## R41-275 P0 — `generate_model_hard_gates.py` 多个基础设施异常会变“无 offender”

### 必须怎么改

统一：

```python
EvidenceProbeResult =
  PASS |
  FAIL |
  INFRASTRUCTURE_ERROR |
  NOT_APPLICABLE
```

production：

```text
INFRASTRUCTURE_ERROR → CI FAIL
```

禁止：

```python
except Exception:
    pass
```

---

## R41-276 P0 — 参数证据 completeness 漏掉“完全没有证据的 canonical”

当前 cross-product audit 默认：

```text
canonicals = evidence 中已出现的 canonicals
```

因此一个 production canonical 如果 0 evidence：

```text
它根本不进入检查集合
```

### 必须怎么改

required set 必须来自：

```text
current ProductionOperatorSurfaceSnapshot
```

调用：

```python
check_evidence_cross_product_completeness(
    points,
    required_canonicals=production_surface.canonicals,
)
```

零证据 canonical 视为缺全部 required combinations。

---

## R41-277 P1 — 参数证据 required backend cross-product 不应对所有 canonical 一刀切

当前默认要求：

```text
pandas + polars + duckdb
```

但有些复杂算子本来只支持 Pandas/Numba。

### 必须怎么改

每个 canonical 由：

```text
OperatorBackendCapabilityContract
```

派生 required combinations：

```text
required
optional
unsupported
```

Completeness 检查“该算子声称 production 可走的路径”，而不是硬塞所有 backend。

---

## R41-278 P0 — ProductionExecutionCertificate 当前不是 hard gate

`ProductionExecutionCertificate.validate()` 返回 false 是合理的。

但 `HybridExecutor` 当前明确：

```text
validation failed
→ counter + warning
→ 继续执行
```

### 必须怎么改

执行模式：

```python
CertificateEnforcementMode:
    PRODUCTION_STRICT
    RESEARCH_TELEMETRY
```

production：

```python
if not validate(...):
    raise ProductionExecutionCertificateError
```

且必须在实际 task 执行前完成。

---

## R41-279 P0 — R39 PERF-082 不应继续标 CLOSED

因为“证书存在”不等于“证书阻断错误执行”。

重新定义 DoD：

```text
wrong backend
unexpected fallback
tampered cert
missing runtime event
stale build
```

五种负控 production 均必须 hard-fail。

---

## R41-280 P1 — R39 PERF-032 `TemporaryArrayArena` 未接 hot consumer 却标 CLOSED

### 必须怎么改

如果对象存在但真实热路径没用：

```text
PARTIAL
```

只有：

```text
至少一个真实 high-allocation kernel
→ arena lease
→ benchmark allocation/GC/RSS improvement
→ parity
```

才 CLOSED。


---

# 5. Predictive Model 训练、预处理与数学正确性（R41-281～330）

## R41-281 P0 — production 不允许无 validation 时用 in-sample IC 选超参数

当前 `train_model()` 明确支持：

```text
validation_ds=None
→ predict train
→ in-sample rank IC
→ choose best hyperparams
```

### 必须怎么改

production：

```python
if validation_ds is None and len(search_space) > 1:
    raise ValidationFoldRequiredError
```

允许无 validation 的唯一生产情形：

```text
hyperparameters 已冻结
没有 selection 行为
```

research 可允许，但 artifact 标：

```text
selection_evidence = IN_SAMPLE_RESEARCH_ONLY
production_eligible = False
```

---

## R41-282 P0 — `train_plus_validation` 之后 artifact cutoff 仍写原 train_end

### 当前错误

最终 fit：

```text
train + validation
```

manifest 却：

```text
training_cutoff = original train_end
available_at = original train_end
```

### 必须怎么改

引入：

```python
EffectiveTrainingCutoff(
    latest_feature_time,
    latest_label_anchor,
    latest_label_maturity_time,
    refit_data_end,
    fit_completed_at,
)
```

最终：

```text
training_cutoff = max time of data actually consumed
available_at = max(label maturity, fit completion, publish time)
```

负控：

```text
validation_end > train_end
→ artifact 在 validation_end 前不可 resolve
```

---

## R41-283 P0 — artifact available_at 没有考虑 forward label maturity

最后训练 anchor 为 `t`、label horizon `H` 时，label 要到 `t+H` 才成熟。

### 必须怎么改

`LabelContract` 不再只存字符串规则。

增加：

```python
LabelMaturityResolver.resolve(anchor_time, calendar, horizon)
```

训练计划先算：

```text
latest_consumed_label_available_at
```

Artifact：

```text
available_at >= latest_consumed_label_available_at
```

---

## R41-284 P0 — DecisionClock 当前主要是 manifest 描述，没有成为训练 eligibility gate

### 必须怎么改

新增：

```text
TrainingTemporalLegalityPass
```

逐 feature 验证：

```text
feature availability <= decision time
label maturity <= training cutoff
universe membership knowledge <= decision time
source revision visible <= decision time
```

任何 UNKNOWN：

```text
production reject
```

---

## R41-285 P0 — 未知 feature availability 默认 decision_at，属于 fail-open

当前逻辑：

```python
feature_available_at.get(feature, decision_at)
```

### 必须怎么改

返回：

```text
Availability.UNKNOWN
```

production：

```text
UNKNOWN != AVAILABLE
```

必须显式 FieldContract / DerivedFeatureContract 提供 availability。

---

## R41-286 P0 — RichModelTiming / LabelContract 使用自由字符串表达时间规则

例如：

```text
"t+H close"
"next session VWAP"
"features through decision_time"
```

字符串无法被 planner/validator可靠执行。

### 必须怎么改

建立 typed temporal AST：

```text
SessionOpen(t)
SessionClose(t)
NextSession(t)
ShiftSessions(t, H)
BarEnd(t)
VWAPCompletion(t)
MaxAvailability(...)
```

所有时间 legality 通过 AST 解释器 + MarketCalendarSnapshot。

---

## R41-287 P0 — preprocessing 的 fit 统计没有按 pipeline 顺序拟合

声明：

```text
imputer → winsor → standardize
```

但当前三个 step 都在 raw X 上独立拟合。

### 必须怎么改

```python
work = X.copy()
for step in spec.steps:
    state = fit_step(work)
    work = transform_step(work, state)
    frozen_steps.append(state)
```

测试与 sklearn 等价 fixture 比对。

---

## R41-288 P0 — preprocessing 对 ±Inf 与全缺失列的处理不严格

### 必须怎么改

统一：

```text
PreprocessingSamplePolicy = FINITE_ONLY
```

训练前产生：

```text
all_missing_columns
nonfinite_fraction
constant_columns
near_constant_columns
```

production policy：

```text
all-missing → reject
nonfinite statistic → reject
```

禁止 `np.nan_to_num()` 把结构问题直接变 0。

---

## R41-289 P0 — adequacy 统计样本和真实 fit 样本不是同一集合

`telemetry()` 的 finite 逻辑与：

```text
imputer 后实际 X
仅按 finite y 过滤
```

不完全一致。

### 必须怎么改

建立唯一：

```python
TrainingSampleMask
```

流程：

```text
raw eligibility
→ temporal legality
→ universe eligibility
→ label maturity
→ preprocessing fit eligibility
→ feature support
→ model-specific support
```

最终：

```text
telemetry
adequacy
fit
```

全部消费同一个 mask/certificate。

---

## R41-290 P0 — `_extract_matrices()` 只按 y finite 过滤，预处理后的 X/aux 仍可能非 finite

### 必须怎么改

```python
fit_mask =
    finite(y)
    & finite_all(transformed_X)
    & finite(required_aux)
```

如果模型声明允许特殊 missing handling，则由模型 contract 单独放行。

---

## R41-291 P0 — SampleAdequacy requirement 存在但 measurement=None 时当前会跳过

例如：

```text
min_regime_obs=5000
regime_obs=None
```

当前可能通过。

### 必须怎么改

规则：

```text
requirement is not None
AND measurement is None
→ ADEQUACY_MEASUREMENT_MISSING
→ fail
```

不能把“没测量”解释成“满足”。

---

## R41-292 P0 — BaseLearner 当前没有把 regime/expert/state/missing/date coverage 测量值传入统一 adequacy

### 必须怎么改

每个 learner 实现：

```python
measure_sample_adequacy(dataset, plan) -> SampleAdequacyMeasurements
```

字段完整：

```text
raw_obs
effective_obs
unique_dates
unique_stocks
missing_fraction
date_coverage
regime_counts
expert_counts
state_transitions
cross_section_peers
```

统一 gate 消费。

---

## R41-293 P0 — `sample_contract=None` 当前直接通过

production predictive learner 不允许没有 SampleAdequacyContract。

### 必须怎么改

```text
PREDICTIVE_SUPERVISED + production
→ sample_contract mandatory
```

---

## R41-294 P0 — `_free_parameter_count()` 严重低估真实模型自由度

### 必须怎么改

Learner API：

```python
effective_parameter_count(
    n_features,
    hyperparams,
    fit_structure=None,
) -> EffectiveParameterCount
```

例如：

```text
OLS: d + 1
Regime K: K*(d+1) + gating params
MoE K: experts + gating
PCR: PCA components + regression coefficients
```

`obs_per_parameter` 使用模型自身真实值。

---

## R41-295 P0 — decay-weighted expanding preset 存在，但 trainer 一律 `weights=None`

### 必须怎么改

WalkForwardPlan 生成：

```text
WeightPlan
```

trainer：

```python
weights = weight_plan.weights(train_ds)
learner.fit(..., weights=weights)
```

每个 learner 声明：

```text
SUPPORTS_WEIGHTS
REJECTS_WEIGHTS
```

不支持时不能静默忽略。

---

## R41-296 P0 — PCR / PLS / ElasticNet 接收 weights 但未真正消费

### 必须怎么改

- PCR：weighted centering + weighted PCA/SVD + weighted regression；
- PLS：weighted NIPALS 或明确 production 不支持 weights；
- ElasticNet：weighted objective / coordinate update；
- 若暂不实现：相关 decay preset 不可绑定到该 learner。

---

## R41-297 P1 — `retrain_every_bars` 是 contract 字段但当前 split/trainer 主链没有成为实际 retrain scheduler authority

### 必须怎么改

区分：

```text
evaluation fold step
artifact retrain cadence
```

不能都由 `step_bars` 隐式代替。

新增：

```text
RetrainingSchedule
```

生成 artifact availability timeline。

---

## R41-298 P1 — `purge_policy` 字段存在，但 `purge_and_embargo()` 基本总走 label interval

### 必须怎么改

typed：

```text
PurgePolicy.LABEL_INTERVAL
PurgePolicy.FIXED_BARS
PurgePolicy.NONE_RESEARCH_ONLY
```

执行逻辑消费 spec，而不是字段只在配置中存在。

---

## R41-299 P0 — Walk-forward 使用 dataset 中“观测到的日期”作为 bars，不是 pinned 市场日历

如果缺一交易日数据：

```text
1000 observed bars
```

和：

```text
1000 exchange sessions
```

不是同一训练窗口。

### 必须怎么改

WalkForwardPlan 接：

```text
MarketCalendarSnapshot
```

窗口、purge、embargo、horizon 全按 session ordinal。

Dataset 缺 session 由 DQ/coverage 描述，不得压缩时间轴。

---

## R41-300 P0 — `nested_splits()` 需要证明 outer test 永不进入 inner search dataset

### 必须怎么改

生成：

```text
DataExposureLedger
```

逐 candidate 记录：

```text
train dates
validation dates
test dates
feature-selection exposure
hyperparam exposure
early-stopping exposure
```

Hard Gate：

```text
FINAL_TEST_EXPOSURE_TO_SEARCH == 0
```

---

## R41-301 P0 — ElasticNet 标准化空间系数被直接当原始空间系数使用

当前：

```text
Xs = (X - mx) / sx
fit coef_std
intercept = my - mx @ coef_std
predict = intercept + X @ coef_std
```

数学错误。

### 正确修改

```python
beta_raw = coef_std / sx
intercept_raw = my - mx @ beta_raw
```

artifact 存：

```text
beta_raw
intercept_raw
```

或者 predict 始终在标准化 Xs 上执行，二选一，但训练/预测必须一致。

### 必测

随机不同 scale feature：

```text
manual standardized prediction
==
artifact prediction
```

---

## R41-302 P0 — ElasticNet non-convergence 注释说 fail-closed，代码却 `pass`

### 必须怎么改

production：

```python
if not converged:
    raise ModelConvergenceError(...)
```

research 可：

```text
allow_nonconverged=True
```

但 artifact：

```text
production_eligible=False
```

---

## R41-303 P0 — ElasticNet penalty 标度需要重新对齐正式 objective

当前 coordinate update 的 `rho/denom/alpha` 是否对应：

```text
1/(2n)||y-Xb||² + alpha(...)
```

必须明确，不可只说“ElasticNet”。

### 必须怎么改

写 `ElasticNetObjectiveContract`：

```text
loss normalization
L1 scale
L2 scale
intercept penalty
weights
tol norm
```

用 sklearn ElasticNet 小样本 oracle 做 parity。

---

## R41-304 P1 — ElasticNet `n_iter` 使用循环变量 `_`，边界/零迭代语义不清

改为显式：

```python
n_iter_run
convergence_delta
objective_delta
```

进入 diagnostics。

---

## R41-305 P0 — PCR 对 `n_components` 使用 `min(requested,d)`，未完整约束 sample/effective rank

### 必须怎么改

production：

```text
requested_components > effective_rank
→ parameter point invalid
```

不要静默 clamp 后还把 requested parameter 视为同一模型。

如果允许 adaptive：

```text
effective_n_components
```

必须进入 artifact identity。

---

## R41-306 P1 — PCR SVD 需要 rank/condition diagnostics

至少记录：

```text
singular_values
effective_rank
condition_number
variance_explained
```

near-rank-deficient 按 DegeneracyPolicy 处理。

---

## R41-307 P0 — PLS 可因 NIPALS break 实际拟合少于 requested components

### 必须怎么改

production：

```text
requested != fitted
→ explicit ModelFitDegradation
```

默认 fail。

若允许 adaptive component count，必须参与 artifact identity，不得静默。

---

## R41-308 P0 — PLS `inv` 失败后静默 `pinv`，改变 estimator 语义

### 必须怎么改

`LinearSolvePolicy`：

```text
EXACT_INVERSE
QR
SVD_PSEUDOINVERSE
RIDGE_STABILIZED
```

属于模型数学 contract。

production 不得“异常后换 estimator”。

---

## R41-309 P0 — Regime gating auxiliary 需要 finite mask

`np.quantile(gate)` 遇 NaN/Inf 会破坏边界。

### 必须怎么改

```text
required_aux_fields
+
AuxSamplePolicy(FINITE_ONLY)
```

纳入 TrainingSampleMask。

---

## R41-310 P0 — Regime/MoE 只判断 `std == 0`，near-zero variance 仍可能数值爆炸

复用全局：

```text
DegeneracyPolicy
```

判断：

```text
absolute variance floor
relative variance floor
condition number
```

---

## R41-311 P1 — Regime quantile boundaries 遇大量 ties 可能生成重复边界/空 regime

### 必须怎么改

fit 后验证：

```text
strictly increasing effective boundaries
nonempty regimes
support floors
```

重复 quantile：

```text
fail or explicit reduced-regime policy
```

不能让 `np.digitize` 偶然决定。

---

## R41-312 P0 — MoE `explicit_fallback="single_best"` 改变模型结构，不能只放 metadata

Fallback 结构必须进入：

```text
ModelArtifactIdentity
ModelSemanticContract
cache key
lineage
```

否则同一 model_name/hyperparams 可对应 MoE 或 pooled OLS。

---

## R41-313 P1 — MoE/Regime soft gating 的 `1/(dist+1e-9)` 需要正式数值合同

`1e-9` 是模型定义的一部分。

声明：

```text
gating_distance
temperature/epsilon
normalization
overflow policy
```

不要藏 kernel magic constant。

---

## R41-314 P0 — 所有 learner 的 output shape / feature count 必须在 fit 与 predict 双向验证

artifact 保存：

```text
n_input_features
ordered feature IDs
output dimension
```

predict shape mismatch hard-fail。

---

## R41-315 P0 — 训练和预测 feature schema 不能只用一个自由字符串 hash

需要：

```python
FeatureSchemaCertificate(
    ordered_features,
    semantic_digests,
    dtype,
    unit,
    price_basis,
    availability,
    source_vintage,
)
```

predict exact verify。

---

## R41-316 P0 — `predict_panel()` 当前按 `ds.feature_cols` 顺序直接取矩阵，无 artifact schema 对齐

必须：

```text
artifact ordered schema
vs
dataset schema
```

严格比较。

禁止：

```text
PE,PB,ROE
```

训练后传：

```text
PB,PE,ROE
```

仍计算。

---

## R41-317 P0 — PanelDataset 没验证 `stock_col` 存在

`__post_init__` 增加：

```text
date_col present
stock_col present
label_col present if required
```

---

## R41-318 P0 — production 下 feature_cols 为空时自动把其它列当特征

这是训练泄漏入口。

### 必须怎么改

production：

```text
feature_cols mandatory explicit
```

research 才允许 inference，且打印 inferred feature list。

---

## R41-319 P0 — PanelDataset 未验证 `(date, stock)` 唯一

### 必须怎么改

默认：

```text
duplicate(date, SecurityMasterId)
→ DuplicateTrainingObservationError
```

若合法多记录必须先由 DataAccess 用明确 aggregation contract 合并。

---

## R41-320 P0 — `filter_stocks()` static `.isin()` 可制造 survivorship bias

长期训练必须使用：

```text
(date, security_id) PIT membership
```

不能把今天的股票列表应用到过去。

---

## R41-321 P1 — training coverage denominator 使用整个窗口 unique stocks 不准确

应该按每个 date：

```text
eligible universe size
observed feature size
mature label size
fit-eligible size
```

再统计 mean/median/P5/min。

---

## R41-322 P0 — label coverage 当前只看 `notna()`，Inf 仍可能算有标签

统一 finite-only label policy。

---

## R41-323 P0 — 训练 label 必须绑定 ReturnSemantic / PriceBasis

例如：

```text
VWAP→VWAP
close→close
raw return
total return
```

不能只靠 `label_name` 字符串。

---

## R41-324 P0 — 模型训练数据必须绑定 exact source snapshot / universe snapshot / calendar snapshot

新增：

```text
TrainingDatasetCertificate
```

至少：

```text
source snapshot digest
revision/vintage
universe membership digest
calendar digest
feature schema digest
label contract digest
row eligibility digest
```

---

## R41-325 P0 — 训练集 feature revision 改变时旧 artifact 不能继续宣称同一数据身份

DataAccess `DataChangeSet` → ModelImpact：

```text
which artifacts affected
earliest training fold affected
retrain required?
historical scores invalidated?
```

---

## R41-326 P1 — random_seed 不足以保证 reproducibility

还需记录：

```text
RNG algorithm
NumPy version
solver version
BLAS config
thread count
determinism level
```

---

## R41-327 P0 — model learner 必须声明 NumericDeterminismLevel

```text
BITWISE
TOLERANCE
NONDETERMINISTIC_RESEARCH_ONLY
```

production artifact 保存并在 replay 验证。

---

## R41-328 P1 — predictive training 需要独立 ResourceLease

训练不应与日频 factor 落值共享一个无区分 compute lane。

新增：

```text
ModelTrainingLease
ModelScoringLease
ArtifactIOLease
```

训练 OOM 不应拖垮生产因子服务。

---

## R41-329 P0 — training cancellation/deadline 必须贯穿 preprocessing/solver/artifact commit

不能只在训练开始前检查 token。

长 SVD / grid search / fold loop 需要 safe cancellation checkpoints。

---

## R41-330 P1 — 大型 pooled panel 不应强制完整 Pandas/NumPy 常驻

长期优化：

```text
Arrow/Polars training block
memmap
streaming standardization statistics
block SVD / randomized SVD
```

但任何 streaming 训练必须有 monolithic parity evidence。

---

# 6. Model Artifact、Registry、DSL 与发布生命周期（R41-331～370）

## R41-331 P0 — “Frozen” artifact 只是浅冻结，内部 dict/list/ndarray 仍可修改

当前：

```text
ModelArtifactManifest(frozen=True) + mutable hyperparameters dict
FrozenModel(frozen=True) + mutable params/metadata dict
FrozenPreprocessing + mutable _steps list
```

### 风险

```text
先生成 lineage/cache key
→ 后原地修改 coef / preprocessing / hyperparameters
→ 同 identity 不同预测
```

### 必须怎么改

发布前 deep-freeze：

```text
dict → MappingProxy / immutable typed object
list → tuple
ndarray → readonly array / immutable bytes + shape/dtype
```

`FrozenPreprocessing.steps` 返回不可变对象，禁止暴露内部 list。

Hard Gate：

```text
ARTIFACT_POST_FREEZE_MUTATION_IMPOSSIBLE == PASS
```

---

## R41-332 P0 — Artifact lineage hash 没有绑定 `available_at` / `training_cutoff`

这两个字段直接决定历史是否合法，却不在 lineage hash payload 中。

### 必须怎么改

建立唯一：

```python
ModelArtifactIdentity
```

包含：

```text
model semantic contract
training range
effective fit cutoff
label maturity cutoff
available_at
published_at
feature schema
source snapshot
universe snapshot
preprocessing
hyperparameters
solver/build
numeric determinism
fallback/degradation
```

cache/lineage/catalog 都投影自它。

---

## R41-333 P0 — Artifact lineage 使用 `json.dumps(default=str)`

未知对象不能通过 string/repr 偷偷进入生产 identity。

使用统一 typed canonical serializer。

---

## R41-334 P0 — `cache_key()` 不充分绑定 fit code/build identity

当前部分依赖：

```text
frozen.metadata["code_hash"]
```

但 learner 并不保证存在。

### 必须怎么改

Artifact 必须绑定：

```text
BuildManifest digest
learner implementation digest
preprocessing implementation digest
solver identity
```

且 mandatory。

---

## R41-335 P0 — Artifact load 丢弃保存的 lineage hash，不验证篡改

当前反序列化会忽略 `lineage_hash`。

### 必须怎么改

load：

```text
parse envelope
→ schema validate
→ checksum verify
→ lineage recompute
→ compare
→ implementation compatibility
→ only then construct artifact
```

---

## R41-336 P0 — Artifact JSON 没有 schema_version

以后字段变化会导致：

```text
旧 artifact 被新代码误读
```

### 必须怎么改

定义：

```text
ModelArtifactSchemaVersion
```

支持：

```text
current
known old → explicit migration
future → reject
```

---

## R41-337 P0 — Artifact save 不是 atomic/durable commit

当前 direct `open(path, "w")`。

### 必须怎么改

```text
write temp
→ flush
→ fsync file
→ validate readback/checksum
→ atomic rename
→ fsync directory
→ catalog commit
```

如果对象存储：

```text
immutable object + manifest pointer CAS
```

---

## R41-338 P0 — ArtifactStore 使用 artifact_id 拼路径，缺 machine ID validator

production artifact ID 使用严格 ASCII machine ID：

```text
[A-Za-z0-9_.:-]+
```

拒绝：

```text
../
/
NUL
homoglyph-sensitive path name
```

---

## R41-339 P0 — ArtifactStore 没有 durable catalog

当前 resolver 主要依赖进程内 registry。

### 必须怎么改

建立：

```text
ModelArtifactCatalog
```

记录：

```text
artifact_id
model namespace
version
training_cutoff
available_at
published_at
status
checksum
object path
lineage digest
access classification
promotion state
```

service restart 后从 catalog 恢复，不靠内存注册。

---

## R41-340 P0 — ArtifactResolver registry 不是线程安全事实源

彻底改为：

```text
immutable catalog snapshot
+
transactional updates
```

读路径 lock-free snapshot 或数据库查询。

---

## R41-341 P0 — ArtifactResolver 使用字符串比较时间

所有：

```text
asof
training_cutoff
available_at
published_at
```

必须 typed timezone-aware timestamp / session ordinal。

---

## R41-342 P1 — model_version 使用字符串排序

使用：

```text
SemanticVersion / packaging.version.Version
```

或内部 monotonic revision id。

---

## R41-343 P0 — process-global `_default_resolver` 不适合多租户/多项目

改 request scoped：

```python
ModelArtifactResolutionContext(
    tenant,
    project,
    market,
    namespace,
    access_policy,
    asof,
)
```

---

## R41-344 P0 — Artifact ACL / access classification 必须继承训练输入最高敏感级别

```text
public feature + restricted source
→ artifact = restricted
```

同时：

```text
artifact preview
parameters
feature names
diagnostics
model score
```

都继承权限。

---

## R41-345 P0 — Artifact encryption / key management 需要作为部署能力

若 artifact 含专有模型系数/特征定义，生产 lake 至少支持：

```text
at-rest encryption
KMS/key id
rotation
access audit
```

若当前部署明确为单机受控目录，也要写部署 contract，而不是隐含假设。

---

## R41-346 P0 — ModelArtifact.from_dict 重建 LearnerSpec 时丢 sample contract / parameter policies

反序列化后 learner governance 与训练时可能不同。

### 必须怎么改

Artifact 保存完整：

```text
ModelSpecDigest
SampleAdequacyContractDigest
ParameterSearchPolicyDigest
TrainingSpecDigest
```

load 时从 current registry 验证兼容，而不是默默使用 current defaults。

---

## R41-347 P0 — Artifact scoring 必须验证 exact ordered feature schema

见 R41-315/316；这里要求在 artifact 本身成为 mandatory gate。

---

## R41-348 P0 — Predictor runtime monkeypatch `learner.fit` 是线程不安全的

两个线程同时 score 同一 artifact：

```text
线程 A 替换 fit
线程 B 替换/恢复 fit
```

状态可能错乱。

### 必须怎么改

设计：

```python
FrozenScorer
```

它根本没有：

```text
fit()
```

训练对象与评分对象类型分离。

---

## R41-349 P0 — Predictor 使用 Python `assert` 作为安全证明

`python -O` 可移除 assert。

生产安全检查必须显式 raise typed exception。

---

## R41-350 P0 — `artifact.predict()` 可以绕过 Predictor guard

因此“Predictor 没调用 fit”不等于“所有 scoring path 无 fit”。

### 必须怎么改

唯一评分入口：

```text
FrozenModelArtifact.scorer()
```

没有 learner mutable fit surface。

---

## R41-351 P0 — `model_score()` 仍只是 Stub，未成为正式 DSL / Typed IR / PhysicalStage

当前模块自己说明 DSL compiler wiring pending。

### 必须怎么改

实现：

```text
model_score("name")
→ ModelScoreExpr
→ TypedModelScoreIR
→ ArtifactResolutionNode
→ ModelFeatureReadNode
→ FrozenScoreNode
→ ModelScoreBlock
```

IR identity 绑定 artifact resolution policy，而非编译时固定今天 artifact。

---

## R41-352 P0 — 历史回放不能在 compile 时把一个 artifact 固定给全部日期

对于：

```text
2018–2026
```

应按每个 as-of 或 artifact validity interval 选择合法 artifact。

物理优化可以：

```text
按 artifact validity intervals 分段批量 score
```

而不是逐行 resolve，也不是全历史一个模型。

---

## R41-353 P0 — `ModelScoreIR` 必须表达“未找到合法 artifact”的语义

production：

```text
NO_LEGAL_ARTIFACT
```

默认 hard-fail 或按明确 missing-score policy 产 NaN。

不能偷偷用未来 artifact / latest artifact。

---

## R41-354 P0 — ModelScore cache key 必须绑定 as-of artifact identity

缓存：

```text
model_name + date
```

不够。

至少：

```text
resolved artifact lineage
feature snapshot
universe
calendar
execution clock
```

---

## R41-355 P1 — ModelRegistry 与 LEARNER_REGISTRY 是两套事实源

当前存在：

```text
modeling.learners.base.LEARNER_REGISTRY
modeling.registry.MODEL_REGISTRY
```

### 必须怎么改

统一：

```text
ModelImplementationRegistry
→ immutable ModelRegistrySnapshot
```

一个 entry 同时包含：

```text
implementation
execution class
training spec
sample contract
search policy
feature contract
artifact schema
```

---

## R41-356 P1 — LEARNER_REGISTRY 是裸全局 mutable dict

改 bootstrap + freeze + immutable snapshot，沿用 OperatorRegistry 已建立的治理模式。

---

## R41-357 P1 — ModelRegistry.get() 返回内部 mutable dict

返回 immutable `ModelRegistryEntry`。

`all()` 也不能只浅 copy。

---

## R41-358 P1 — `register_default_learners()` 的 has→register 是 TOCTOU

并发两个线程都可能：

```text
has=False
→ one register success
→ other ValueError
```

实现 atomic：

```python
register_if_absent(...)
```

---

## R41-359 P0 — Model registry entry 必须有 explicit semantic_version

模型逻辑或默认训练策略变化时：

```text
model semantic version bump
```

不能只靠 Python source hash。

---

## R41-360 P0 — Hyperparameter Search 有两套 authority 冲突

当前：

```text
APPROVED_SEARCH_SPACES
```

包含 ElasticNet alpha/l1_ratio；

但 recommendation table 又标：

```text
searchable_by_miner=False
```

### 必须怎么改

生成唯一：

```python
ValidatedHyperparameterSearchPlan
```

来源：

```text
ModelRegistryEntry.parameter_policies
+
requested search dimensions
+
certified values/bounds
```

---

## R41-361 P0 — trainer 可直接接受任意 hyperparam_grid，绕过治理

production API 不接受 raw arbitrary grid。

改为：

```python
train_model(..., search_plan: ValidatedHyperparameterSearchPlan)
```

research helper 可接受 raw grid，但不 production eligible。

---

## R41-362 P1 — Hyperparameter selection 需要 tie-break / stability policy

若多个参数验证分数近似：

```text
选择更简单模型
```

而不是列表顺序偶然决定。

建议：

```text
score tolerance
complexity penalty
neighbor stability
```

---

## R41-363 P0 — 最终 holdout 不仅不能调超参，也不能调 feature set / label / universe / preprocessing

DataExposureLedger 必须覆盖：

```text
features
operators
model class
hyperparams
label horizon
universe policy
preprocessing
```

---

## R41-364 P1 — Model lifecycle 需要状态机

```text
CANDIDATE
VALIDATED
STAGED
PRODUCTION
DEPRECATED
RETIRED
REVOKED
```

状态变化 transactional、audited。

---

## R41-365 P0 — Production promotion 必须是显式审批动作

训练完成不能自动变 production。

Promotion 要求：

```text
current-build evidence
OOS gates
PIT/leakage negative controls
sample adequacy
artifact checksum
security/access
```

---

## R41-366 P1 — Champion / Challenger 与 rollback

保存：

```text
champion artifact
challenger artifacts
promotion time
rollback pointer
```

回滚 atomic，不重新训练。

---

## R41-367 P0 — Revocation 机制

如果发现：

```text
数据源 revision 污染
模型 bug
PIT leak
artifact corruption
```

必须能：

```text
revoke artifact
block future resolution
identify affected historical scores
```

---

## R41-368 P1 — Artifact retention / GC

不能直接删除旧模型，因为历史 replay 可能需要。

GC 必须知道：

```text
lineage references
materialized score dependencies
legal retention
```

---

## R41-369 P1 — Model lineage 需要连接 Factor lineage

如果模型特征来自 FactorEngine factors：

```text
artifact
→ feature factor versions
→ operator/field/source snapshots
```

必须可追踪。

---

## R41-370 P1 — Model Score materialization 要有独立 identity

`model_score` 落因子 lake 时：

```text
factor formula identity
+
artifact resolution timeline identity
```

都要记录。

---

# 7. Model Evaluation / OOS 统计正确性（R41-371～400）

## R41-371 P0 — turnover 当前用“当天局部 row position”当股票身份

当前：

```text
argsort(pred)
→ set(order[-k:])
```

下一天的 row 5 与上一天 row 5 被错误当同一只股票。

### 必须怎么改

输入真实：

```text
SecurityMasterId
```

top bucket 保存 security IDs。

必测：

```text
每天行顺序随机打乱
→ turnover 不变
```

---

## R41-372 P0 — `evaluate_predictions()` 接收 stock_col 但未使用

改 API：

```python
evaluate_predictions(pred, y, dates, security_ids, ...)
```

security_ids mandatory for turnover / portfolio metrics。

---

## R41-373 P0 — `rank_ic` 当前是 pooled stock-date Spearman，不是 mean daily cross-sectional RankIC

### 必须怎么改

命名拆开：

```text
mean_daily_rank_ic        ← 主要因子/模型指标
daily_rank_ic_ir
pooled_rank_correlation   ← diagnostic only
```

trainer selection 默认：

```text
mean_daily_rank_ic
```

不是 pooled。

---

## R41-374 P0 — trainer 与 evaluation ICIR ddof 不一致

定义唯一：

```python
ICMetricConvention(
    correlation="spearman",
    aggregation="daily_cross_sectional",
    std_ddof=1,
    min_pairs_per_date=...,
)
```

---

## R41-375 P0 — block-aware IC 按“剩下的有效日期序列”每 H 个分组，不是实际交易 session

改：

```text
MarketCalendarSnapshot
+
label interval
```

构造 non-overlapping evidence blocks。

---

## R41-376 P1 — overlapping forward labels 的显著性不能只靠普通 ICIR

支持至少：

```text
HAC/Newey-West
block bootstrap
effective independent block count
```

报告时注明。

---

## R41-377 P0 — top/bottom selection tie-break 不能依赖输入 row order

使用：

```text
canonical SecurityMasterId
```

作为稳定 tie-break，或明确 include-all-ties。

---

## R41-378 P1 — long-short spread 必须记录可形成组合的最小横截面支持

日期样本不足不能仅 silent skip；输出：

```text
n_valid_dates
skipped_dates_by_reason
```

---

## R41-379 P1 — coverage 应基于 eligible universe，而不是简单 finite(pred) fraction

每天：

```text
eligible
feature_available
scored
label_mature
evaluated
```

五层 coverage。

---

## R41-380 P0 — grouped metrics 的 group labels 必须 PIT

行业、市值、流动性、牛熊状态：

```text
必须使用当时可知 label
```

不能用今天行业映射或全样本分位数。

---

## R41-381 P1 — grouped labels 长度不匹配当前返回 None，过于静默

生产 evaluation 应 typed error。

---

## R41-382 P1 — year parsing 异常当前可能返回空 dict

日期 contract 错误不能“没有 year-by-year 结果”掩盖。

---

## R41-383 P1 — EvaluationReport 要区分 metric undefined 与真正 0

例如：

```text
ICIR undefined
turnover unavailable
```

不能用 0.0 混淆。

使用：

```text
MetricValue(value, status, reason, n_obs)
```

---

## R41-384 P1 — 评估需要 transaction cost / liquidity-aware diagnostic

至少支持：

```text
turnover
ADV participation proxy
spread/slippage assumption
net long-short spread
```

但属于 evaluator/model evaluation，不应污染预测模型训练目标，除非显式 optimization contract。

---

## R41-385 P1 — 需要 cross-sectional weight neutrality diagnostics

如果 score 产生严重：

```text
size
industry
beta
liquidity
```

暴露，evaluation 报告应拆出 raw vs neutralized IC。

---

## R41-386 P1 — 模型评估应按市场状态分段，但状态本身必须只由当时数据得到

避免 full-sample regime label。

---

## R41-387 P1 — OOS fold aggregation 不能让长 fold 单纯因 rows 多而支配

报告：

```text
per-date pooled
per-fold equal weight
calendar-time aggregate
```

三种口径。

---

## R41-388 P1 — 需要 model calibration / score distribution stability

至少：

```text
mean/std
tail
rank dispersion
cross-sectional entropy
score saturation
```

---

## R41-389 P1 — 需要 OOD / feature drift 评估

Artifact 保存 train distribution sketch：

```text
median/MAD or quantiles
missing rate
feature covariance summary
```

score 时计算：

```text
PSI / standardized shift / missing drift
```

但不自动用未来分布重标训练。

---

## R41-390 P0 — OOD policy 必须明确

生产遇到超阈值：

```text
WARN
ABSTAIN
DEGRADE_TO_NAN
BLOCK_MODEL
```

必须由 contract 决定，不可模型内部随意 fallback。

---

## R41-391 P1 — multiple testing / model selection bias 需要记录 search attempts

保存：

```text
number of candidates
feature candidates
hyperparam trials
model classes
```

避免最终只展示最好一个。

---

## R41-392 P1 — neighborhood_stability 当前只看 best±1，对非整数/多维参数不够

扩展：

```text
local parameter neighborhood
surface flatness
rank stability
```

---

## R41-393 P1 — 最终评估应提供 per-year / rolling OOS IC 序列，不只汇总数

用于识别：

```text
2026 失效
regime drift
```

---

## R41-394 P1 — evaluation artifact 自己也要 versioned

Metric convention 改变后旧报告不能与新报告直接混用。

---

## R41-395 P0 — final holdout 必须物理隔离

建议 dataset exposure token：

```text
TEST_LOCKED
```

训练/search API 无法读其 labels，只有 final evaluator 能解锁。

---

## R41-396 P1 — model candidate selection 不应由单一 rank IC 决定所有模型

可定义：

```text
primary metric
guardrails
```

例如：

```text
mean_daily_rank_ic
+
coverage floor
+
turnover ceiling
+
stability floor
```

---

## R41-397 P1 — final promotion 应检查 validation→test degradation

避免 validation overfit。

---

## R41-398 P1 — score monotonicity / sign convention 需要明确

模型输出：

```text
higher = more bullish
```

应进入 ModelOutputContract，避免 evaluator 误翻方向。

---

## R41-399 P1 — label horizon 与 execution horizon 要对齐

例如训练 5-day VWAP label，却用 next-day execution 解释，需要明示持有期/重叠。

---

## R41-400 P1 — Evaluation 必须有完整 reproducibility bundle

保存：

```text
artifact lineage
dataset certificate
fold plan
metric convention
calendar
universe
code/build
random seed
```


---

# 8. Service Security、Source Policy 与多租户隔离（R41-401～430）

## R41-401 P0 — `Principal.has_role()` 对未知 role 默认成 READ 等级

当前模式近似：

```python
ROLE_TIERS.get(role, 1)
```

如果代码写错：

```text
MATERALIZE
PUBLlSH
```

未知 role 可能被解释成最低权限，而不是配置错误。

### 必须怎么改

```python
if role not in ROLE_TIERS:
    raise UnknownPrivilegeError(role)
```

Principal 自身 roles 也必须全部 validate。

---

## R41-402 P0 — `scope_key()` 没包含 project

当前：

```text
identity@tenant
```

多 project 下可能共享：

```text
idempotency
cache
quota
artifact namespace
```

### 必须怎么改

```text
principal_id / tenant / project
```

全部进入 authorization scope。

---

## R41-403 P0 — API-key mapping JSON 解析失败当前静默变空 mapping

生产 startup 应 hard-fail：

```text
configured mapping exists but invalid
→ SecurityConfigurationError
```

不能退化成“没有 mapping”。

---

## R41-404 P0 — production `_authenticate` 实际要求 API key 配置，和 JWT/mTLS/Proxy 多信任源设计冲突

当前 production 如果没 API key：

```text
即使配置 JWT secret
也可能先报“production endpoints require API key”
```

### 必须怎么改

启动时建立：

```python
AuthProviderSet(
    api_key,
    jwt,
    mtls,
    trusted_proxy,
)
```

production 要求：

```text
至少一个 trusted provider ready
```

而不是硬要求 API key。

---

## R41-405 P0 — 自实现 JWT 缺 `exp/nbf/iss/aud/alg` 严格校验

### 必须怎么改

使用成熟 JWT library。

配置：

```text
allowed_algorithms
issuer
audience
clock_skew
key id / rotation
required claims
```

必须验证：

```text
exp
nbf
iat
iss
aud
sub
```

---

## R41-406 P0 — JWT header algorithm 没有正式 allowlist

防止算法混淆。只允许配置中的算法。

---

## R41-407 P1 — JWT key rotation / kid 不支持

企业级支持：

```text
active keys
grace-period old keys
rotation audit
```

---

## R41-408 P0 — 代码声明 MUTUAL_TLS，但真实 principal resolution 未看到 mTLS implementation

二选一：

```text
真正实现 mTLS provider
```

或：

```text
删掉能力宣称
```

不能 docs/API 说支持但生产没有。

---

## R41-409 P0 — reverse proxy 只靠 env bool 信任 `X-Remote-User`

### 必须怎么改

至少要求：

```text
trusted proxy network/IP
or
mTLS between proxy and FE
or
signed proxy assertion
```

外部客户端不能直连服务并伪造 header。

---

## R41-410 P0 — ClickHouse allowlist 为空时当前检查被跳过，相当于 allow-all

生产 allowlist：

```text
empty = deny
```

如果确需 allow-all：

```text
explicit ALLOW_ALL_REMOTE_SOURCES=true
```

且 production startup warning/approval。

---

## R41-411 P0 — Parquet approved_roots 为空时同样近似 allow-all

同上改 empty-deny。

---

## R41-412 P0 — composite `sources` 类型错误可能绕过递归验证

production discriminated schema：

```text
CompositeSourceConfig.sources: nonempty mapping[str, SourceConfig]
```

类型错误直接 reject。

---

## R41-413 P0 — intraday_daily / long_table 的 inner source 非 dict 时不能直接返回

必须：

```text
inner required + typed SourceConfig
```

---

## R41-414 P1 — `ALLOWED_SOURCE_TYPES_PRODUCTION` 常量与实际 validate_source 分支不一致

只保留一个 policy authority。

---

## R41-415 P0 — Source profile 内容必须 schema validate

不能：

```text
任意 dict
只取 dataset
```

至少：

```text
profile schema version
dataset
provider
market
snapshot policy
access classification
contract digest
```

---

## R41-416 P0 — source profile 不存在时 identity builder 不应退化成 hash(profile_id)

这是配置错误，必须在构建 identity 前 hard-fail。

---

## R41-417 P0 — source credentials / secrets 不能进入 source semantic hash 明文

hash 前使用 canonical secret redaction：

```text
secret value rotation不改变“数据源语义” identity
但 credential binding/security audit 可另存 secret-version id
```

---

## R41-418 P1 — Security policy reload 要有 immutable policy snapshot

在途 job 使用提交时 policy snapshot；新 job 用新 policy。

同时记录：

```text
policy version
digest
effective time
```

---

## R41-419 P0 — Artifact/Model namespace 必须与 Principal/tenant/project 对齐

一个 tenant 不得 resolve 另一 tenant 同名：

```text
predictive_pcr
```

---

## R41-420 P0 — factor lake / model artifact / spill / checkpoint ACL 必须统一

不要只保护 HTTP preview。

---

## R41-421 P1 — auth failure metric 当前逻辑需复核

认证异常发生时应该记录；不要只在得到 anonymous principal 后才 incr。

---

## R41-422 P0 — admin / publish / materialize 权限不能只靠 role tier 大小

部分权限可能不是严格线性层级。

建议：

```text
explicit capabilities set
```

例如：

```text
FACTOR_READ
FACTOR_COMPUTE
FACTOR_MATERIALIZE
MODEL_TRAIN
MODEL_PUBLISH
ARTIFACT_REVOKE
ADMIN
```

---

## R41-423 P1 — API key 比较与存储应避免明文 key mapping 长期驻留

至少：

```text
key hash
constant-time compare
key id
rotation
```

---

## R41-424 P0 — config_path 读取要继续防 TOCTOU

现有 symlink component check 是好方向，但最终应：

```text
open file descriptor safely
verify inode/device
read from fd
```

而不是 validate path 后再重新 open 路径。

---

## R41-425 P1 — config root / artifact root / lake root 需要不同 privilege scope

避免某个 config read root 权限等同于 artifact write root。

---

## R41-426 P0 — 远程 source DNS/rebinding/redirect policy

如果未来 source profile 支持 URL/object storage：

```text
scheme allowlist
host/IP policy
redirect policy
private network policy
```

必须明确。

---

## R41-427 P1 — security audit log 需要 append-only / tamper-evident

记录：

```text
principal
action
resource
decision
policy version
request id
timestamp
```

---

## R41-428 P0 — declassification 必须与 artifact/model promotion 集成

训练 restricted source 的模型，不能因为 artifact 不含原始数据就自动变 public。

---

## R41-429 P1 — secret redaction 必须覆盖 exception / trace / structured log

不能只 canonical config hash 做 redaction。

---

## R41-430 P1 — 安全配置 negative controls 纳入 CI

至少：

```text
expired JWT
wrong audience
unknown role
spoofed proxy
empty allowlist
path traversal
symlink race
cross-project artifact read
restricted model preview
```

---

# 9. Observability、异步上下文与长期服务内存（R41-431～450）

## R41-431 P0 — FastAPI async context 使用 `threading.local()` 会串请求

改：

```python
ContextVar
```

每请求：

```text
token = set(...)
try...
finally reset(token)
```

---

## R41-432 P0 — log context 没有明确 reset，线程复用可能遗留上一请求 identity

与 #431 一次修。

---

## R41-433 P1 — Histogram 当前永久 append float，长期运行无限增长

换：

```text
fixed bucket histogram
DDSketch
HDR Histogram
Prometheus client
```

不能保存全部 raw observations。

---

## R41-434 P1 — Metrics label cardinality 实际没有强制限制

建立 per-metric label schema：

```text
allowed label names
allowed/bucketed values
max cardinality
```

禁止：

```text
factor_name
artifact_id
request_id
```

作为 metrics label。

---

## R41-435 P1 — `counter_value()` 无 lock

提供 snapshot/atomic counter API。

---

## R41-436 P1 — snapshot aggregation 会重复扫描 counters

维护聚合 counter，而非 snapshot 时 O(N²-ish) 重算。

---

## R41-437 P0 — structured log 没有统一敏感字段 sanitizer

所有：

```text
config
DSN
headers
source profile
artifact metadata
```

先经过：

```text
ClassificationAwareLogSanitizer
```

---

## R41-438 P1 — `default=str` 让任意对象进入日志且可能泄漏 repr

日志只接受 typed JSON values。

未知对象：

```text
safe_type_name
```

而不是 repr。

---

## R41-439 P1 — trace span 没有 error status / exception family

span 应记录：

```text
status
error_code
cancelled
deadline
fallback
```

不记录完整 secret traceback。

---

## R41-440 P1 — tracing 要贯穿 FE→DataAccess

统一：

```text
trace_id
span_id
execution_id
job_id
```

Source scan / SQL / model score / materialize 可串起来。

---

## R41-441 P1 — resource telemetry 需要和 TTDC 统一

至少：

```text
scheduler wait
scan
conversion
compute
DQ
writer
catalog
artifact IO
```

时间分解。

---

## R41-442 P1 — Model training telemetry 需要独立 namespace

避免与 factor execution metric 混淆。

---

## R41-443 P1 — cancellation / deadline 原因要可观测

区分：

```text
user cancel
deadline
resource eviction
shutdown drain
security revoke
upstream failure
```

---

## R41-444 P1 — fallback 不能只有计数，要记录 typed reason

```text
backend unsupported
backend failed
resource pressure
certificate mismatch
```

production 不允许的 fallback 要直接 fail。

---

## R41-445 P1 — DQ metrics 要按原因分类但避免高 cardinality

固定 error family。

---

## R41-446 P1 — 日志中的 principal/project 应做最小必要披露

内部审计保留完整，普通 app log 可用 stable pseudonymous id。

---

## R41-447 P2 — 提供 `/metrics` 与 `/health/ready`

ready 必须检查：

```text
registry
parameter evidence
DataAccess capability
calendar
source policy
artifact catalog
resource coordinator
```

---

## R41-448 P0 — health 不能只返回 process alive

Production readiness 与 liveness 分离。

---

## R41-449 P1 — 服务 shutdown 时 flush telemetry 但不能阻塞无限期

有 bounded shutdown deadline。

---

## R41-450 P1 — metrics 本身不能成为新内存 owner

Metrics memory 纳入 service memory budget 或严格常数上界。


---

# 10. HTTP Execution Identity、Config 与 production semantic fencing（R41-451～475）

## R41-451 P0 — `_catalog_generations()` field registry 获取失败后用 `"unavailable"` 继续 production

### 必须怎么改

production identity 的任何 mandatory generation：

```text
field
market
calendar
backend evidence
compiler
source
```

无法解析：

```text
IdentityGenerationUnavailableError
```

不能把 `"unavailable"` 当合法 generation。

---

## R41-452 P0 — `field_contract_hash` 当前近似从 market 字符串构造，不是真实字段依赖 contract

### 必须怎么改

从 parsed/typed IR 提取实际 fields：

```text
dependency-scoped FieldContractDigest
```

`close` contract 改了才失效相关 factor，不相关字段改动不全局失效。

---

## R41-453 P0 — `source_dependency_hash` 不能只是 calendar hash

应该绑定：

```text
actual source dependencies
dataset/provider/field mappings
revision/vintage
PIT/availability
snapshot policy
```

---

## R41-454 P0 — HTTP universe membership hash 当前只 hash `sorted(universe)` 静态字符串

动态 universe 必须绑定：

```text
UniverseMembershipSnapshot / interval digest
```

历史范围不能只用当前 member list。

---

## R41-455 P0 — backend evidence generation 当前近似 hash requested backend 字符串

应绑定：

```text
BackendEvidenceManifest digest
actual implementation generation
emitter/compiler generation
numeric semantics
```

---

## R41-456 P0 — compiler build generation 不能只 hash FactorEngine package version

两个 commit 都是 0.3.1：

```text
不同实现
同 compiler_build_gen
```

必须 BuildManifest/source component digest。

---

## R41-457 P1 — `_stable_hex(*parts)` 使用 `str(p)` 不应作为 production typed identity

统一 semantic serializer。

---

## R41-458 P0 — source profile hash 使用 `default=str`

同上，改 typed schema serialization。

---

## R41-459 P0 — `model.market or "ashare"` 这类默认在 production identity 路径应清零

production 缺 market：

```text
hard fail
```

research 才可默认 A 股 compat。

---

## R41-460 P0 — DataSourceBuildContext 需要绑定 timezone/decision clock/coverage policy 的真实值

不能构造时留空，然后下游再猜。

---

## R41-461 P0 — ValidatedFactorRequest digest 必须与 actual execution identity 一致

执行前：

```text
reconstruct current identity
compare validated digest
```

任何 context drift：

```text
ValidationExecutionMismatchError
```

---

## R41-462 P1 — source policy/version reload 后 queued job 必须继续使用提交时 snapshot

已有方向继续补真实 integration test。

---

## R41-463 P0 — TypedDataSourceOptions 当前允许未知键进入 `extra`

production source config 是安全/语义边界。

### 必须怎么改

按 source type 建 discriminated union：

```text
DataAccessSourceOptions
ParquetSourceOptions
ClickHouseSourceOptions
CompositeSourceOptions
IntradaySourceOptions
```

production：

```text
extra=forbid
```

research 可有 explicit `raw_options`.

---

## R41-464 P1 — known string options 当前 `str(v)` coercion 过宽

例如 list/dict 被转成字符串。

production 使用 strict type。

---

## R41-465 P1 — max_files / port / timeout 等整数需要 range constraints

例如：

```text
max_files > 0
1 <= port <= 65535
```

---

## R41-466 P0 — start_date/end_date 需要 typed timestamp + ordering

```text
start <= end
timezone/calendar compatible
```

---

## R41-467 P0 — source type 与 options 必须 schema-compatible

parquet 不应携带 data_access-only option；ClickHouse 必需 host/db/table profile。

---

## R41-468 P1 — config schema migration 要提供完整 old→current provenance

effective config hash 绑定：

```text
original schema version
migration chain
final normalized payload
```

---

## R41-469 P0 — production config hash 与 execution identity secret redaction 要分层

不能为了不 hash password 而把真正 source semantic 信息也删掉。

分：

```text
SourceSemanticIdentity
CredentialBindingIdentity
SecuritySecretRef
```

---

## R41-470 P1 — 环境变量 override 必须进入 effective-config evidence

否则同一 YAML 在不同环境行为不同而 lineage 看不出。

---

## R41-471 P1 — Config/Profile 搜索路径不能被当前工作目录隐式改变

全部相对：

```text
known config root / package resource
```

---

## R41-472 P0 — production runtime mode 不能由多个 env/config/endpoint authority 决定

最终单一：

```text
EndpointExecutionPolicy
→ ExecutionSemanticContext.run_mode
```

下游只读，不重新猜。

---

## R41-473 P0 — parameter-domain `_is_production()` exception→False 是 fail-open

当前 direct call 没显式 run_mode 时，production policy resolver 异常可退 research。

### 必须怎么改

production-capable入口强制显式 `run_mode`。

legacy direct call：

```text
mode unresolved → UNKNOWN
```

敏感 gate 对 UNKNOWN fail。

---

## R41-474 P1 — runtime config 中 PIT default false 可以保留 research，但 production floor 必须不可覆盖

明确：

```text
effective_pit = endpoint_floor OR config_pit
```

production endpoint 永远 true。

---

## R41-475 P0 — production startup 应生成 immutable `RuntimeReadinessCertificate`

绑定：

```text
build
registry
fields
market/calendar
parameter evidence
backend evidence
DataAccess handshake
security policy
resource runtime deps
```

job admission 必须检查 certificate 当前有效。

---

# 11. Parameter Domain / Evidence Store 最新 HEAD 再整改（R41-476～495）

## R41-476 P0 — ParameterDomain store freshness 仍依赖 live `.git`

wheel/container 往往没有 `.git`。

### 必须怎么改

`_current_head()` 不再是 production authority。

改：

```text
BuildManifest.source_commit
BuildManifest.source_tree_digest
```

Evidence 绑定 BuildManifest。

---

## R41-477 P0 — installed wheel 环境中 current head 为空时 stale evidence 可能无法被拒绝

production：

```text
不能证明 freshness
→ fail
```

而不是跳过比较。

---

## R41-478 P0 — load_json 对 `passed` 使用 `bool(value)`

若恶意/错误 JSON：

```json
"passed": "false"
```

Python：

```text
bool("false") == True
```

### 必须怎么改

Evidence schema 强类型：

```text
passed must be JSON boolean
```

其他类型 reject。

---

## R41-479 P0 — CertificationKey 参数使用 `repr()` 拼 key

参数身份必须 typed canonical。

特别：

```text
numpy scalar dtype
float -0
NaN
tuple/list
enum
```

不能靠 repr。

---

## R41-480 P0 — `parameter_point` 从 JSON 反序列化缺 schema/type validation

每个参数按 Operator ParamSpec bind 后再构造 CertificationKey。

---

## R41-481 P1 — Evidence store freeze 必须真正禁止 certify_point mutation

若 `_frozen=True` 后仍能写点，应 hard-fail。

增加 mutation negative test。

---

## R41-482 P0 — `_ensure_loaded()` 已 loaded 后不重新检查文件/build变化

readiness 与 load 分开是对的，但所有 production call 必须经过 current readiness certificate。

---

## R41-483 P0 — default evidence 文件当前仍是历史 R37 store

最新 production release 必须重新生成 current-build store，不允许把 R37 文件名本身当权威。

推荐：

```text
docs/evidence/parameter_domain/<build_digest>/store.json
```

---

## R41-484 P1 — certification source/evidence_hash 不能为空对 production 证据

当前示例可见 `evidence_hash=""`。

production point 应要求：

```text
oracle id
test artifact digest
evidence hash
```

---

## R41-485 P0 — 参数证据 required canonicals 必须来自 production surface

与 R41-276 联动。

---

## R41-486 P0 — 参数点必须包含 default-bound complete call

不能只保存用户显式 kwargs。

认证与执行共享 `BoundOperatorCall`。

---

## R41-487 P0 — ParameterDomain 必须绑定 actual execution variant

不要所有都写：

```text
reference
```

如果实际走：

```text
numba
polars
sql fused
fast kernel
```

必须相应 evidence。

---

## R41-488 P0 — source_context 要成为 typed identity，而不是任意字符串

例如：

```text
memory
production
research
```

混合不同维度。

拆：

```text
run_mode
source_class
provider/snapshot context
```

---

## R41-489 P1 — dtype signature 应支持多输入，而不是单 dtype 字符串

旧 #160 继续闭环。

---

## R41-490 P1 — grain 也应是 typed GrainContract

daily/minute/event/financial，而不是自由字符串。

---

## R41-491 P0 — production parameter certificate 与 operator semantic version 必须严格一致

semantic version missing/empty 不可匹配。

---

## R41-492 P0 — certificate store 应支持 revocation

发现 oracle bug 时能撤销点并阻止新 production job。

---

## R41-493 P1 — continuous parameter domain 不应枚举无限点

使用：

```text
CertifiedRegion
+
boundary/property proof
```

但执行 membership 必须严格。

---

## R41-494 P1 — parameter certificate 查询应 O(1)/bounded，不应成为 10k factor hot bottleneck

预编译：

```text
ProductionCallCertificate
```

compile once，runtime O(1)。

---

## R41-495 P0 — current release 的 parameter evidence completeness 必须成为 main CI required check

无证据：

```text
不能发布 production wheel
```

---

# 12. DataAccess R30、ABI、Snapshot 与 FE 协同（R41-496～525）

## R41-496 P0 — `data_access.r30` package omission 修复后必须验证所有 R30 module import

包括但不限于：

```text
resolution_lease
session
calendar_snapshot
universe_snapshot
experiment_snapshot
coverage_service
partition_index
change_impact
execution_lease
lineage
policy
```

---

## R41-497 P1 — `r30.__all__` 与实际新模块存在漂移风险

例如已新增 `resolution_lease.py` 时，要么 export，要么明确 internal。

增加：

```text
physical module inventory ↔ public/internal classification
```

测试。

---

## R41-498 P0 — R30 “additive layer” 不能只存在而未接 main read path

逐对象检查：

```text
ResolutionLease
Snapshot
Coverage
PartitionMetadataIndex
R30ReadSession
```

是否被真实：

```text
data_access.get_store/read
FE DataAccessSource
PreparedBatchReadSession
```

消费。

只存在模块不算整改完成。

---

## R41-499 P0 — FE↔DA capability handshake

见 R41-265，落到 startup。

---

## R41-500 P0 — Snapshot contract version 要进入 FE source identity

DA snapshot semantics 改变时 FE cache/materialization 自动失效。

---

## R41-501 P0 — DataAccess resolution lease 必须位于 HostResourceCoordinator child lease 树

不能 DA 独立 admission。

---

## R41-502 P0 — R40 #96 cancellation 继续贯穿 DataAccess

覆盖：

```text
source discovery
LIST/HEAD
scan
DuckDB
Arrow stream
composed read
```

---

## R41-503 P0 — DataAccess deadline 同样贯穿 discovery 到 terminal consumption

旧项继续闭环并加真实 timeout integration。

---

## R41-504 P1 — R39 PERF-073 ScopedConnectionPool 实现

目标：

```text
resettable DuckDB connection pool
per-connection config fingerprint
deadline interrupt
lease ownership
no cross-job temp relation leakage
```

---

## R41-505 P0 — Connection pool 归还前必须清理 session state

包括：

```text
temp views
registered Arrow objects
PRAGMA mutations
interrupt state
transaction
```

---

## R41-506 P1 — remote metadata discovery cache 需要 snapshot-aware

不能缓存到新 revision 后仍返回旧 file list。

---

## R41-507 P0 — PartitionMetadataIndex 必须证明和实际 object store/listing 一致性

stale index 要：

```text
detect
invalidate
fallback safely
```

---

## R41-508 P0 — FE training dataset 必须从 DataAccess ExperimentDataSnapshot 构造

不要直接把任意 Pandas DataFrame 标成 production training dataset。

---

## R41-509 P0 — Model training snapshot 与 factor feature snapshot 必须相同或证明兼容

防止：

```text
factor values v1
labels v2
universe v3
```

混合。

---

## R41-510 P1 — DataAccess lineage 要能反查模型 artifact

```text
source change
→ factors
→ training dataset
→ model artifacts
→ materialized model scores
```

---

## R41-511 P1 — DataAccess API version 与 data format version 分开

Parquet schema evolution 不应等同 package version。

---

## R41-512 P0 — DataAccess wheel clean-install 必须运行真实 get_store/read path

不能只 import。

---

## R41-513 P1 — DataAccess scripts 如果不作为 package，不要文档声称 `python -m data_access.scripts...`

当前 wheel inventory doc 的模块命令要与真实 packaging 一致。

---

## R41-514 P0 — DataAccess version metadata 与 service `/health` / FE handshake 必须一致

单测试一次检查四处。

---

## R41-515 P1 — FE/DA release compatibility matrix

例如：

```text
FE 0.3.1 build X
compatible DA contract >= Y < Z
```

记录到 BuildManifest。

---

## R41-516 P0 — Snapshot fidelity UNKNOWN/FALLBACK 生产准入

继续旧 R40 语义：无法证明 snapshot 的 source 不能进入 authoritative production。

---

## R41-517 P1 — DataAccess read representation contract 与 FactorBlock 对齐

尽量：

```text
Arrow-native
```

避免 Pandas round trip。

---

## R41-518 P1 — scan cost calibration 与 ResourceCoordinator 联动

真实 P50/P95/decoded footprint 更新，不能永久使用冷启动默认。

---

## R41-519 P0 — DataAccess query cache identity 包含 snapshot/build/contract

避免 source revision 后 cache poison。

---

## R41-520 P1 — Query cache 与 FE CSE / panel cache 统一 memory inventory

旧项继续闭环。

---

## R41-521 P0 — DataAccess errors 必须 typed

至少：

```text
Resolution
Snapshot
Permission
Deadline
PIT
Schema
DQ
Resource
Backend
```

上层不能 broad except 后改空数据。

---

## R41-522 P0 — Empty result 与 resolution failure 永远分开

空市场/无数据可以 empty；找不到/无权限/超时不能 empty。

---

## R41-523 P1 — multi-source join 的 source freshness/maturity 要可解释

每一列知道来自哪个 snapshot/vintage。

---

## R41-524 P0 — 财务 revision / vintage 在训练模型时必须保留 PIT 语义

不能拿 latest restated financials 训练历史 artifact，除非明确 retrospective research。

---

## R41-525 P1 — DataAccess 需要为 Model Layer 提供 TrainingReadPlan

输入：

```text
features
label
universe
date range
decision clock
```

输出同一 pinned experiment snapshot 的 panel，避免模型层自己拼数据。

---

# 13. Runtime、并发、性能深架构与 R39 剩余闭环（R41-526～555）

## R41-526 P1 — PERF-017：native backend 从源头 emit OutputSlice

当前后处理零拷贝 slice 已有，但 native backend 仍可能先生成完整 result 再 slice。

### 改法

PhysicalPlan 将 requested output range 下推：

```text
time slice
instrument slice where legal
```

backend 只生成必要区域。

需保证 warmup/internal rows 与 external output slice 分离。

---

## R41-527 P1 — PERF-020：真正的 process worker lane

目标不是简单 `ProcessPoolExecutor.submit(fn, DataFrame)`。

设计：

```text
PlanId / SharedBufferRef / SourceSnapshotRef
→ long-lived worker
→ mmap/Arrow shared memory
→ result BufferRef
```

避免大 Pandas pickle。

---

## R41-528 P0 — process pool Future 异步 pickle/worker failure 需要真实 recovery

submit() 成功不代表任务成功。

callback/result path 识别：

```text
BrokenProcessPool
PicklingError
worker death
```

然后按 typed recovery policy。

production 不能任何失败都换 thread 后继续。

---

## R41-529 P1 — HybridExecutor 不应每次 `_ensure_pools()` 同时创建 thread+process 两套池

lazy per-kind pool，减少进程常驻内存。

---

## R41-530 P0 — HybridExecutor standalone 调用不能假设 scheduler 已预留 resource token

如果 executor 是 public/internal reusable entry，必须接受：

```text
ExecutionLease
```

没有 lease 的 production submit 拒绝。

---

## R41-531 P1 — process worker thread env 不应依赖运行期随意修改 global os.environ

进程 initializer 可以设置 worker process env；主进程 per-job 不应改。

记录实际 BLAS/OpenMP config。

---

## R41-532 P1 — PERF-073 ScopedConnectionPool

见 R41-504/505，必须有 benchmark：

```text
connection setup
PRAGMA replay
temp relation cleanup
concurrent query
```

---

## R41-533 P1 — PERF-078 NativePipelineRegion

optimizer 识别最大的同 backend region：

```text
scan
→ expressions
→ filters
→ rolling where supported
→ aggregation
```

减少 representation transition。

---

## R41-534 P1 — NativePipelineRegion 不能为了 region 大小绕过 correctness hard gate

region eligibility：

```text
semantic certified
parameter certified
PIT legal
axis legal
backend evidence current
```

先满足，再优化。

---

## R41-535 P1 — PERF-079 RollingStateBlock

多个同 input/window 的：

```text
mean
std
sum
min
max
```

共享一次 rolling state traversal。

Numba/native kernel 多输出。

---

## R41-536 P1 — RollingStateBlock 必须与各单算子 reference exact parity

包括：

```text
NaN
Inf
min_periods
ddof
partial windows
```

---

## R41-537 P1 — PERF-080 PrimitiveBlock

跨因子提取：

```text
returns
rolling mean
rolling std
rank inputs
```

形成 primitive DAG，批量物化到 FactorBlock 内部，不必永久写 lake。

---

## R41-538 P0 — Primitive CSE identity 必须绑定完整 semantic context

禁止不同：

```text
price basis
universe
calendar
source snapshot
```

共享 primitive。

---

## R41-539 P1 — TemporaryArrayArena 接真实 hot kernels

优先：

```text
rolling
regression
PCA
cross-sectional rank temporary arrays
```

观察：

```text
allocation count
GC
RSS
TTDC
```

---

## R41-540 P0 — ResourceCoordinator 训练与因子批量运行同时存在时要公平

实现 lane：

```text
LATENCY
BATCH_FACTOR
MODEL_TRAINING
BACKGROUND_COMPACTION
```

并设：

```text
minimum guaranteed
maximum burst
preemption/shrink policy
```

---

## R41-541 P1 — CPU token 与 native internal threads 联动

如果 task 占 4 tokens：

```text
DuckDB/BLAS/Polars 内层线程 ≤ 4
```

不能 scheduler 4 tokens + BLAS 16。

---

## R41-542 P0 — OOM 记忆应区分 workload semantic identity 与 machine envelope

相同 factor 在不同机器/并发环境下不能错误继承永久小 shape。

记录：

```text
host class
available memory
concurrency
shape
```

---

## R41-543 P1 — memory pressure recovery 需要慢恢复

旧 AIMD 原则继续。

---

## R41-544 P0 — writer backpressure 必须直接反馈 compute admission

队列满时降低新计算，避免 result buffers 堆内存。

---

## R41-545 P1 — compaction 是后台 lane，不能抢落值关键资源

delta compaction 有预算/暂停机制。

---

## R41-546 P1 — model artifact IO 与 factor writer 分开 queue，但共享 host storage budget

避免互相饿死。

---

## R41-547 P1 — benchmark 必须用 exact same source snapshot/build/config

否则 before/after 无意义。

---

## R41-548 P1 — Benchmark 结果记录置信区间/重复次数

至少 warmup + 3 runs，报告 median/P90。

---

## R41-549 P1 — B4 incremental 1-day 继续作为最高优先性能 workload

生产每日最常见。

目标：

```text
historical rewrite bytes = 0
full factor rescan = 0
```

---

## R41-550 P1 — Model scoring benchmark

新增：

```text
100/1000 model scores
artifact resolution timeline
batch score
historical replay
```

---

## R41-551 P1 — Training benchmark

至少：

```text
PCR/PLS/ENet
1000 days × 3000 stocks × 50 factors
```

记录：

```text
peak memory
fit time
preprocess time
artifact size
```

---

## R41-552 P0 — performance optimization 不得改变 Model/Factor identity

任何 fast kernel 替换必须：

```text
same semantic version if exactly equivalent
or
bump numeric/semantic version
```

---

## R41-553 P1 — representation-transition budget

Planner 可把：

```text
transition bytes/count
```

直接纳入 cost。

---

## R41-554 P1 — Python object overhead 纳入 large mining workload profile

10k factor 场景：

```text
PlanNode
dict
Future
context clone
```

也可能是瓶颈。

---

## R41-555 P1 — 10k factor campaign 使用 streaming result metadata

不能把全部结果/lineage/evidence Python dict 长期堆内存。

---

# 14. 生产恢复、迁移、灾备、漂移与长期运维（R41-556～585）

## R41-556 P0 — Artifact catalog / Factor catalog 都需要 crash-recovery protocol

模拟：

```text
data written
catalog not committed
catalog committed
data missing
```

恢复必须确定。

---

## R41-557 P0 — generation publication 使用 fencing token

旧 writer/重试任务不能覆盖新 generation。

---

## R41-558 P0 — service restart 后 queued/running job 状态恢复

明确：

```text
queued requeue
running → interrupted/recoverable
committing → recovery probe
```

---

## R41-559 P0 — model training job 同样需要 durable job state

不能训练半小时后服务重启，artifact 状态未知。

---

## R41-560 P1 — Artifact promotion transaction 与 factor score materialization 解耦

先 publish artifact，再由 score job消费；失败可回滚 pointer。

---

## R41-561 P0 — filesystem/object-store corruption tests

随机：

```text
truncate artifact
corrupt parquet footer
bad manifest checksum
missing delta
duplicate generation
```

必须 fail closed / recover。

---

## R41-562 P1 — chaos tests

注入：

```text
kill worker
kill service
disk full
read-only fs
network timeout
DuckDB interrupt
OOM pressure
```

检查不产生“成功但错数据”。

---

## R41-563 P0 — disk full 时 atomic commit 不能留下 current pointer 指向不完整对象

---

## R41-564 P1 — backup/restore

至少定义：

```text
factor catalog
generation manifests
model artifact catalog
security policy metadata
```

的恢复顺序。

---

## R41-565 P0 — Build rollback 与 data/artifact compatibility

旧 binary 读取新 schema 时：

```text
不兼容 → reject
```

不能猜。

---

## R41-566 P1 — API/DSL deprecation policy

旧 canonical/alias：

```text
deprecated since
remove after
migration
semantic equivalence
```

---

## R41-567 P1 — model artifact schema migration 不能原地覆盖旧 artifact

生成新 artifact object + migrated lineage。

---

## R41-568 P1 — drift detection 不等于自动 retrain+auto-promote

流程：

```text
drift signal
→ retrain candidate
→ validation/OOS
→ explicit promotion
```

---

## R41-569 P0 — 不能用未来 performance 自动回改历史 artifact selection

ArtifactResolver 只按当时 publication/as-of，不按后来知道“哪个好”。

---

## R41-570 P1 — Model performance decay monitor

滚动：

```text
IC
coverage
score dispersion
feature drift
```

触发 candidate retrain。

---

## R41-571 P1 — feature failure / source outage policy

模型缺部分 feature 时：

```text
不能自动丢列然后预测
```

除非 artifact 明确支持 missing-feature fallback。

---

## R41-572 P0 — missing-feature fallback 改变模型语义必须独立 artifact/variant

---

## R41-573 P1 — shadow scoring

新模型先：

```text
score but not publish to production signal
```

收集 parity/latency/DQ。

---

## R41-574 P1 — champion/challenger comparison 要使用同时段同 universe/snapshot

---

## R41-575 P0 — rollback 不得使用“后来训练的旧名字模型”回放更早历史

仍遵守 artifact available_at。

---

## R41-576 P1 — artifact/model audit trail

记录：

```text
created_by
reviewed_by
promoted_by
revoked_by
reason
ticket
timestamps
```

---

## R41-577 P1 — production experiment isolation

研究员的临时 artifact namespace 不得进入 production resolver。

---

## R41-578 P0 — tenant/project quotas

限制：

```text
concurrent jobs
memory
scan bytes
writer bytes
artifact count
```

防单项目拖垮宿主机。

---

## R41-579 P1 — fair scheduling

多项目并发采用 weighted fair queue，而不是先提交者长期占满。

---

## R41-580 P1 — retention policies

原始 evidence、artifacts、factor generations、logs 分开生命周期。

---

## R41-581 P0 — evidence artifact 不可在代码更新后被“就地重写成新 SHA”而无运行

证据生成必须来自真实执行结果。

---

## R41-582 P1 — release manifest

每次 production release 列：

```text
FE build
DA build
operator digest
field digest
model registry digest
evidence digest
dependency lock digest
```

---

## R41-583 P0 — release blocking policy

任何：

```text
P0 OPEN
P0 PARTIAL
production hard gate NOT_RUN
current evidence stale
```

都阻止 release。

---

## R41-584 P1 — rollback release manifest

可定位上一稳定 build 与兼容数据 schema。

---

## R41-585 P1 — 运维 runbook

至少：

```text
证据 stale
parameter store fail
calendar unavailable
artifact catalog corruption
writer stuck
memory pressure
DA source unavailable
```

有明确处理，不靠临时改 env 绕 gate。

---

# 15. R41 新增 Hard Gates

这些 gate 必须是**真实执行 gate**，不是字符串/常量。

## HG-01 Clean Wheel

```text
FE_WHEEL_HAS_MODELING
DA_WHEEL_HAS_R30
FE_DA_CLEAN_INSTALL_END_TO_END
```

## HG-02 Version / Build Truth

```text
ONE_FE_VERSION_AUTHORITY
ONE_DA_VERSION_AUTHORITY
BUILD_MANIFEST_PRESENT
DEPENDENCY_LOCK_DIGEST_PRESENT
NO_LIVE_GIT_REQUIRED_FOR_PRODUCTION
```

## HG-03 Model Training Legality

```text
PRODUCTION_HYPERPARAM_SELECTION_WITHOUT_VALIDATION == 0
TRAINING_FEATURE_AVAILABILITY_UNKNOWN == 0
IMMATURE_LABEL_USED == 0
STATIC_TODAY_UNIVERSE_USED_FOR_TRAINING == 0
```

## HG-04 Preprocessing

```text
PREPROCESS_FIT_TRANSFORM_ORDER_PARITY == PASS
ALL_MISSING_FEATURE_REJECTED
NONFINITE_PREPROCESS_STAT_REJECTED
TRAIN_SAMPLE_MASK_EQUALS_FIT_SAMPLE_MASK
```

## HG-05 Learner Math

```text
ELASTICNET_STANDARDIZED_RAW_SPACE_PARITY
ELASTICNET_NONCONVERGENCE_PRODUCTION_FAIL
PCR_REQUESTED_COMPONENTS_CONTRACT
PLS_COMPONENT_DEGRADATION_EXPLICIT
REGIME_GATE_FINITE
MOE_GATE_FINITE
```

## HG-06 Artifact Time

```text
ARTIFACT_AVAILABLE_AT >= LAST_CONSUMED_LABEL_MATURITY
ARTIFACT_TRAINING_CUTOFF >= LAST_ACTUAL_REFIT_OBSERVATION
FUTURE_ARTIFACT_HISTORY_RESOLUTION == 0
```

## HG-07 Artifact Integrity

```text
ARTIFACT_DEEP_IMMUTABLE
ARTIFACT_SCHEMA_VERSIONED
ARTIFACT_CHECKSUM_VALIDATED
ARTIFACT_LINEAGE_VALIDATED_ON_LOAD
ARTIFACT_ATOMIC_COMMIT
```

## HG-08 Feature Schema

```text
MODEL_SCORE_FEATURE_SCHEMA_EXACT
MODEL_SCORE_FEATURE_ORDER_EXACT
MODEL_SCORE_FEATURE_SEMANTICS_EXACT
```

## HG-09 DSL Model Score

```text
MODEL_SCORE_STUB_ONLY == 0
MODEL_SCORE_TYPED_IR_PRESENT
MODEL_SCORE_ASOF_ARTIFACT_RESOLUTION_PROVEN
```

## HG-10 Evaluation

```text
TURNOVER_USES_SECURITY_ID
TURNOVER_ROW_PERMUTATION_INVARIANT
PRIMARY_RANK_IC_IS_MEAN_DAILY_CS
IC_CONVENTION_SINGLE_AUTHORITY
BLOCK_AWARE_USES_MARKET_CALENDAR
```

## HG-11 Evidence Truth

```text
NO_HARDCODED_TRUE_PRODUCTION_GATE
AUTHORITATIVE_EVIDENCE_RUNNER_NO_FALLBACK
EVIDENCE_HEAD_EQUALS_BUILD_HEAD
ZERO_EVIDENCE_PRODUCTION_CANONICALS == 0
PRODUCTION_CI_NOT_RUN_COUNT == 0
```

## HG-12 Certificate Enforcement

```text
PRODUCTION_CERT_MISMATCH_EXECUTION_COUNT == 0
UNEXPECTED_PRODUCTION_FALLBACK_COUNT == 0
```

## HG-13 Security

```text
UNKNOWN_ROLE_REJECTED
EXPIRED_JWT_REJECTED
WRONG_AUDIENCE_REJECTED
UNTRUSTED_PROXY_HEADER_REJECTED
EMPTY_REMOTE_ALLOWLIST_DENIES
CROSS_PROJECT_ARTIFACT_ACCESS_DENIED
```

## HG-14 Async Observability

```text
REQUEST_CONTEXT_CROSS_TALK == 0
LOG_CONTEXT_AFTER_REQUEST == EMPTY
METRICS_MEMORY_BOUNDED
LOG_SECRET_LEAK_COUNT == 0
```

## HG-15 DataAccess

```text
DATA_ACCESS_R30_IN_WHEEL
FE_DA_CAPABILITY_HANDSHAKE
DA_CANCEL_PROPAGATION
DA_DEADLINE_PROPAGATION
DA_SNAPSHOT_IDENTITY_BOUND
```

## HG-16 R39 Deep Performance

```text
PERF017_CLOSED
PERF020_CLOSED_OR_EXPLICITLY_NON_PRODUCTION
PERF073_CLOSED
PERF078_CLOSED
PERF079_CLOSED
PERF080_CLOSED
TEMP_ARENA_REAL_HOT_CONSUMER
```

---

# 16. 必须新增的测试

## 16.1 clean wheel integration

```bash
python -m build
pip install factor_engine wheel
pip install data_access wheel
```

新 Python process：

```python
import modeling
import data_access.r30
```

再跑：

```text
DataAccess read
→ FactorEngine factor
→ model train
→ artifact save
→ restart
→ artifact resolve
→ score
```

---

## 16.2 ElasticNet 数学 oracle

随机造：

```text
feature scales = [1e-3, 1, 1e3]
```

比较：

```text
manual standardized coordinate model
artifact raw-space predict
```

必须一致。

另测：

```text
max_iter=1 forcing non-convergence
production → raise
```

---

## 16.3 artifact time leak negative control

构造：

```text
train end = Jan
validation end = Mar
H=20
```

final refit 用 validation 后：

```text
Feb asof resolve → must NOT see artifact
after last label maturity/publish → can resolve
```

---

## 16.4 DecisionClock unknown feature

训练加入：

```text
mystery_feature
```

没有 FieldAvailabilityContract：

```text
production training reject
```

---

## 16.5 preprocessing order

数据含：

```text
NaN
outlier
different scales
```

验证：

```text
imputer fit
→ transformed
→ winsor fit
→ transformed
→ scaler fit
```

与参考 pipeline 一致。

---

## 16.6 deep immutability

发布后尝试：

```python
artifact.manifest.hyperparameters["x"] = 1
artifact.frozen.params["coef"][0] = 99
artifact.preprocessing.steps.append(...)
```

全部必须失败或不影响内部对象。

---

## 16.7 artifact corruption

修改 artifact JSON 一个 coefficient：

```text
load → checksum/lineage failure
```

截断文件也必须失败。

---

## 16.8 resolver restart

```text
process A write + catalog commit
kill A
process B load catalog
historical resolve same result
```

---

## 16.9 turnover row permutation

同一日股票行随机 shuffle 100 次：

```text
turnover unchanged
```

---

## 16.10 daily RankIC objective

制造：

```text
pooled correlation 高
daily CS IC 差
```

确认 trainer 按 daily metric 选择正确 candidate。

---

## 16.11 evidence runner failure injection

让 authoritative trainer 故意抛：

```text
ModelConvergenceError
```

evidence generator：

```text
must FAIL
```

不得 fallback。

---

## 16.12 hard gate exact exception

R40/R41 negative gate 必须验证：

```python
with pytest.raises(ExpectedTypedError):
```

而不是：

```python
except Exception:
    PASS
```

---

## 16.13 security

覆盖：

```text
role typo
expired JWT
future nbf
wrong issuer
wrong audience
unsupported alg
spoof X-Remote-User
empty source allowlist
malformed source policy JSON
cross-project artifact access
```

---

## 16.14 async context cross-talk

FastAPI 并发 100 requests：

```text
different principals/request ids
```

每条 log/span 不得串。

---

## 16.15 metrics soak

模拟：

```text
1,000,000 observations
```

MetricsRegistry memory 必须 bounded。

---

## 16.16 DataAccess r30 wheel

wheel zip / clean env 确认：

```text
data_access/r30/__init__.py
resolution_lease
...
```

全部存在。

---

## 16.17 FE↔DA incompatible handshake

模拟旧 DA capability：

```text
FE production startup must reject
```

---

## 16.18 parameter evidence zero-canonical

从 evidence 删除一个 production canonical 全部 points：

```text
completeness gate must FAIL
```

---

## 16.19 production certificate

负控：

```text
tampered cert
wrong backend
missing event
unexpected fallback
```

必须阻止 `fn` 真正执行；可用 side-effect counter 证明 fn 0 calls。

---

## 16.20 model scoring concurrency

100 threads 对同一 immutable artifact predict：

```text
no mutation
same result
```

---

# 17. Static Audit

全仓扫描以下 pattern，逐项人工分类；production unsafe 计数必须归零：

```text
except Exception: pass
except Exception: return None
except Exception: fallback
assert <production safety condition>
default=str
repr( in identity
bool(json_value)
threading.local in async request context
os.environ mutation in per-job path
pd.bdate_range in production market-time code
np.sort(dataset observed dates) as calendar authority
model fit inside score path
artifact_id path concat without validation
open(path, "w") for artifact current publication
validation_ds is None + hyperparam selection
weights=None despite decay plan
std == 0 as only degeneracy test
np.quantile(aux) without finite mask
static universe .isin(stocks) in production training
```

---

# 18. 推荐整改顺序

## Phase 0 — 建立最新 HEAD closure ledger

先创建：

```text
factor_engine/docs/R41_ISSUE_CLOSURE_LEDGER.md
```

记录：

```text
execution_start_head
old R39/R40 unresolved
R41-261..585
newly discovered NEW-R41-586+
```

---

## Phase 1 — 立即修 P0 数学 / 打包

优先：

```text
R41-261
R41-262
R41-263
R41-264
R41-281
R41-282
R41-283
R41-285
R41-287
R41-291
R41-301
R41-302
R41-371
R41-373
```

先消除“算出来就是错”的问题。

---

## Phase 2 — Artifact / PIT / Model Score

```text
R41-331～370
```

目标：

```text
真正 frozen
真正 as-of
真正可恢复
真正进 DSL
```

---

## Phase 3 — Evidence Truth

```text
R41-268～280
R41-476～495
```

先让 evidence 不会自欺，再谈 READY。

---

## Phase 4 — Security / Service

```text
R41-401～475
```

---

## Phase 5 — DataAccess packaging / ABI

```text
R41-496～525
```

---

## Phase 6 — R39 深性能

```text
R41-526～555
```

正确性 hard gates 全绿之后再做。

---

## Phase 7 — 运维生产化

```text
R41-556～585
```

---

# 19. Benchmark Matrix

## Factor

```text
100 factors / 1000 / 5000 / 10000
daily
fundamental PIT
minute→daily
mixed source
```

## Model

```text
PCR
PLS
ElasticNet
Regime
MoE
```

规模：

```text
300 stocks × 3y
3000 stocks × 5y
50 features
100 features
```

记录：

```text
data build
preprocess
fit
validation
artifact serialize
score
artifact resolve
peak RSS
CPU
BLAS threads
```

## Contention

```text
factor batch + model training
factor batch + compaction
2 model trainings
4 concurrent research jobs
```

验证资源收缩。

---

# 20. 最终 Definition of Done

## Correctness

- ElasticNet raw/standardized 数学一致；
- 所有模型收敛/降级语义明确；
- preprocessing 顺序正确；
- sample adequacy 真执行；
- daily CS RankIC 口径正确；
- turnover 用真实 Security ID；
- 动态股票池无 survivorship leak。

## PIT

- feature availability 可执行；
- label maturity 可执行；
- artifact available_at 正确；
- historical score 只用当时已发布 artifact；
- calendar/session authoritative；
- training source/universe/revision pinned。

## Artifact

- deep immutable；
- schema/version/checksum/lineage；
- atomic write；
- durable catalog；
- restart 可恢复；
- ACL；
- promotion/revoke/rollback。

## Runtime

- production certificate 真阻断；
- cancellation/deadline 全链；
- one resource authority；
- no unexpected fallback；
- R39 剩余深性能项闭环。

## Security

- strict JWT claims；
- trusted proxy；
- unknown role reject；
- empty allowlist deny；
- cross-project isolation；
- logs redacted。

## Packaging

- FE wheel 含 modeling；
- DA wheel 含 r30；
- FE↔DA capability handshake；
- one version authority；
- dependency lock；
- clean-install integration。

## Evidence

- current BuildManifest；
- zero hardcoded PASS；
- zero authoritative fallback；
- zero production NOT_RUN；
- zero production canonical without required evidence；
- no live git dependency。

---

# 21. Coding AI 最终交付要求

必须输出：

```text
1. final HEAD SHA
2. BuildManifest digest
3. R41_ISSUE_CLOSURE_LEDGER.md
4. R39 unresolved closure status
5. R40 PARTIAL/OPEN closure status
6. R41-261..585 status
7. NEW-R41-586+ findings
8. exact changed files
9. exact tests + pass/fail/xfail counts
10. clean wheel results
11. FE↔DA handshake result
12. model mathematical oracle results
13. PIT/leakage negative controls
14. security negative controls
15. production hard gates
16. benchmarks before/after
17. remaining risks
```

状态只允许：

```text
FIXED
FIXED_ALREADY_WITH_CURRENT_HEAD_PROOF
NOT_APPLICABLE_WITH_PROOF
BLOCKED_BY_EXTERNAL_DEPENDENCY
```

对于 **P0**：

```text
PARTIAL
OPEN
NOT_RUN
```

均视为 release blocker。

---

# 22. 最终执行指令

把以下内容当作 Coding Agent 的最终任务：

```text
A. git fetch / read latest main
B. record actual execution-start HEAD
C. read old 260-item prompt + R39/R40 ledgers
D. revalidate every prior PARTIAL/OPEN against current code
E. map R41-261..585 to actual files/functions
F. fix confirmed P0 first
G. do not use fallback to hide authoritative-path failure
H. write exact tests and negative controls
I. run targeted tests continuously
J. run full factor_engine + dataaccess regression
K. build both wheels
L. clean-install both wheels together
M. run FE↔DA end-to-end
N. regenerate evidence from authoritative execution only
O. verify evidence current-build freshness externally
P. run production hard gates: NOT_RUN is failure
Q. run benchmark
R. final re-audit repository, add NEW-R41-586+
S. only when no P0 blocker remains, finish
```

---

# 23. 最重要的最终原则

最终要保证：

> **同一个因子或模型分数，在明确的市场、日历、动态股票池、决策时钟、PIT 数据快照、字段语义、算子语义、模型训练数据、参数、后端、数值合同、模型 Artifact 和构建环境下，只得到一个可复现、可证明、可恢复的生产结果。**

并且：

> **任何 production gate 无法证明时，结果必须是“不执行”，而不是“warning 后继续执行”。**

这两句话高于所有局部测试与历史 ledger 状态。
