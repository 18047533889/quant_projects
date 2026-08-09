# FactorEngine R16：最新代码增量全量审计与整改提示词

> **这是新的独立整改 Prompt，不是前面任何 MD 的合并版。**
>
> 前面的整改文件已经交给其他 AI 并行处理。本文件只面向当前最新代码重新验收后仍然存在的残余、新出现的二阶问题、修复引入的问题，以及此前没有真正被全库机器审计覆盖的问题。
>
> coding AI 不得把旧整改 MD 再合并进来；只需要读当前 workspace + 本文件，按当前代码实际状态处理。

## 0. 当前审计基线

- 仓库：`18047533889/quant_projects`
- 本轮人工复审基线：`main @ 1dc10425c8d170eb8c22ca76fe266085c401312e`
- coding AI 开始时必须重新读取它本地最新 HEAD；如果并行 AI 已继续推进，以本地最新代码为准逐项复核。
- 当前 `factor_engine/evidence/factor_operator_verified.json` 的 `commit_sha` 仍是 `0b659ecaecd8293fa42f4315e01cd3965a7bd513`，与本轮 main 不同，因此当前 evidence 不能直接证明最新实现。
- 本文件中的：
  - `CONFIRMED_CURRENT_DEFECT`：当前 main 源码直接确认；
  - `CURRENT_REGRESSION`：上一轮目标本应关闭，但当前 main 仍能直接确认存在；不要只改注释，必须真正闭环；
  - `MANDATORY_FULL_AUDIT`：不是断言某个具体算子一定错，而是要求对 final registry 100% 执行。任何命中继续修。

## 1. 这轮的完成标准

必须最终证明，而不是口头声称：

```text
FINAL_REGISTRY_CANONICALS == PER_CANONICAL_AUDIT_CANONICALS
FINAL_REGISTRY_CANONICALS == CONTRACT_MATRIX_CANONICALS
FINAL_REGISTRY_CANONICALS == ROLE_CLASSIFICATION_CANONICALS

UNRESOLVED_P0 == 0
AUDIT_ERROR == 0
REQUIRED_DETECTOR_NOT_RUN == 0
UNDECLARED_MINEABLE_SCALAR_PARAM == 0
SIGNATURE_DEFAULT_PARAMSPEC_DEFAULT_MISMATCH == 0
SILENT_PARAM_CAST_OR_CLAMP == 0
MULTI_INPUT_WITHOUT_AXIS_CONTRACT == 0
GRAIN_TRANSFORM_WITHOUT_SESSION_CONTRACT == 0
STALE_EVIDENCE_ACCEPTED == 0
BACKEND_SEMANTIC_PARITY_FAILURE == 0
```

绝对禁止：
- 抽样 60/200/300 个 canonical 后声称全库已审；
- detector 没跑却以空列表表示 PASS；
- 捕获异常后 `continue`；
- 通过 waiver/ignore/allowlist 隐藏数学问题；
- 用 dropna/reindex/clamp/zero-fill 把数据问题变成“可运行”；
- 用旧 evidence 自动认证新 implementation；
- 把 pending/planning manifest 当 runtime mining manifest。


### R16-001｜P0｜CONFIRMED_CURRENT_DEFECT｜R15 master coverage 只审前 200 个 canonical

**位置**：`factor_engine/scripts/audit_r15_master.py::_ashare_coverage`

**当前问题**

coverage audit 仍对排序后的 canonical 做固定切片。第 201 个以后的 operator 即使完全坏掉，master report 也可以不知情。

**必须怎么改**

删除固定切片。允许 shard 并行，但最终必须做 set equality：`audited == FinalRegistrySnapshot.canonicals`。无法构造 fixture 必须记 NOT_RUN/AUDIT_ERROR。

**必须新增/修改测试**

`test_master_audit_covers_exact_final_registry`；在排序靠后位置注入故障 canonical，必须失败。

### R16-002｜P0｜CONFIRMED_CURRENT_DEFECT｜semantic duplicate scan 只覆盖前 300 个 canonical

**位置**：`factor_engine/scripts/audit_r15_master.py::_semantic_duplicate_scan`

**当前问题**

duplicate detector 仍是 sampling，不能支撑 `SEMANTIC_DUPLICATE_CANONICALS==0`。

**必须怎么改**

全 canonical 进入候选生成。可先按 arity/unit/grain/family 分桶降低 O(N²)，但每个 public canonical 必须有 duplicate-audit outcome。

**必须新增/修改测试**

制造第301个以后等价 canonical，strict audit 必须抓到。

### R16-003｜P0｜CONFIRMED_CURRENT_DEFECT｜golden/null report 是占位产物而非逐算子测试

**位置**：`factor_engine/scripts/audit_r15_master.py`

**当前问题**

`golden_null_test_report.json` 的存在并不等于 golden/null battery 真正执行。

**必须怎么改**

建立 GoldenNullRunner；每个适用 canonical 输出 `(canonical,test_id,outcome)`。NOT_RUN 不得认证。

**必须新增/修改测试**

删除一个 family 的 runner 后 release 必须失败。

### R16-004｜P0｜CONFIRMED_CURRENT_DEFECT｜dead-parameter detector 只审少量 operator

**位置**：`factor_engine/scripts/audit_all_registered_operators.py`

**当前问题**

当前 detector 只对前约60个带 searchable 参数的算子做实际 injectivity 检查，其余不是 release blocker。

**必须怎么改**

不允许抽 operator。高成本 family 可少测参数点，但每个 searchable param 都必须有 machine outcome。

**必须新增/修改测试**

第61个以后注入 dead param，strict audit 必须 fail。

### R16-005｜P0｜CONFIRMED_CURRENT_DEFECT｜dead-param runner 对多输入算子并未真正绑定完整输入

**位置**：`factor_engine/scripts/audit_all_registered_operators.py::_run_op_hash`

**当前问题**

fixture 虽可生成 x/y 等，但执行路径仍可能只传 x；binary/ternary/specialized 抛异常后被跳过。

**必须怎么改**

只通过 ResolvedSignature 构造和绑定 panel args；异常记 AUDIT_ERROR，禁止 continue。

**必须新增/修改测试**

binary/ternary/variadic/minute/fundamental fixtures 全覆盖。

### R16-006｜P0｜CONFIRMED_CURRENT_DEFECT｜行为 hash 丢失 NaN topology

**位置**：`factor_engine/scripts/audit_all_registered_operators.py`

**当前问题**

只比较 finite values 时，不同 missing mask 可能 hash 相同，参数改变 missingness 会被误判等价。

**必须怎么改**

行为 fingerprint 同时 hash shape/index/columns/value/NaN/Inf mask/coverage。

**必须新增/修改测试**

相同 finite values、不同 NaN mask 必须不同指纹。

### R16-007｜P0｜CONFIRMED_CURRENT_DEFECT｜audit exception 可被静默跳过

**位置**：`factor_engine/scripts/audit_all_registered_operators.py`

**当前问题**

完全跑不起来的 operator 可能因为 exception→None/continue 而不进入 FAIL。

**必须怎么改**

所有 exception 输出 AUDIT_ERROR + canonical + params + fixture + traceback 摘要；mineable/production target 的 AUDIT_ERROR 一律阻断。

**必须新增/修改测试**

注入必抛异常 operator，strict audit exit!=0。

### R16-008｜P0｜CONFIRMED_CURRENT_DEFECT｜strict release 没把全部关键 invariant 当 hard gate

**位置**：`factor_engine/scripts/audit_all_registered_operators.py`

**当前问题**

`UNUSED_PUBLIC`、`FACTOR_SHAPED_RESEARCH_ONLY`、`SEMANTIC_DUPLICATE`、`DEAD_SEARCHABLE_PARAMS` 等非空仍可能不阻断。

**必须怎么改**

把 release-critical invariant 明确标 `release_blocking=true`；advisory 另命名。

**必须新增/修改测试**

逐 invariant mutation test。

### R16-009｜P0｜CONFIRMED_CURRENT_DEFECT｜CERTIFIED_NOT_IN_MANIFEST 检查使用 admission=all 自证

**位置**：`factor_engine/scripts/audit_all_registered_operators.py`

**当前问题**

`admission=all` 本就接近全 registry，无法证明 eligible runtime manifest 真包含所有 certified mineable factor。

**必须怎么改**

比较 context-compatible certified mineable set 与 **eligible-only runtime manifest**。

**必须新增/修改测试**

故意从 eligible manifest 删除一个 certified factor，必须失败。

### R16-010｜P1｜CONFIRMED_CURRENT_DEFECT｜identical_contract 被误当 semantic duplicate 证明

**位置**：`factor_engine/scripts/audit_all_registered_operators.py`

**当前问题**

相同参数名/unit/grain 只表示接口相似，不能证明数学相同。

**必须怎么改**

改成 candidate-only lint；真正 duplicate 需 implementation/AST fingerprint + golden behavioral + metamorphic equivalence。

**必须新增/修改测试**

ts_mean/ts_std 不能因接口同而判 duplicate。

### R16-011｜P0｜CONFIRMED_CURRENT_DEFECT｜R15 fix report 用源码 marker + 测试文件名冒充 proof

**位置**：`factor_engine/scripts/audit_r15_fix_report.py`

**当前问题**

源码注释含 marker、某测试文件存在，并不能证明对应 pytest node 执行通过。

**必须怎么改**

finding 显式绑定 test_node_id/evidence_id/implementation_hash；report 读取真实 junit/evidence。

**必须新增/修改测试**

只保留注释 marker、删除实现，report 必须 FAIL。

### R16-012｜P1｜CONFIRMED_CURRENT_DEFECT｜R15 fix report 路径不可移植

**位置**：`factor_engine/scripts/audit_r15_fix_report.py`

**当前问题**

面向单机 workspace/taskbook 的 hard-coded 路径会导致换机器后 report 不完整。

**必须怎么改**

所有路径 repo-relative 或 CLI 显式传入；缺 source taskbook=NOT_RUN/ERROR。

**必须新增/修改测试**

临时目录运行得到同样结果。

### R16-013｜P0｜CONFIRMED_CURRENT_DEFECT｜index_* source 被错误归 relation_pit

**位置**：`factor_engine/mining/operator_catalog.py::_required_sources`

**当前问题**

当前 fallback 会把 index 相关 operator 归到 relation source，虽然 vocabulary 已存在 `index_pit`。

**必须怎么改**

holder/relation/index/event/fundamental 分开 typed SourceRequirement；优先读 metadata，禁止 broad name guess。

**必须新增/修改测试**

仅有 index_pit 时 index-member op 可满足；仅 relation_pit 时不满足。

### R16-014｜P0｜CONFIRMED_CURRENT_DEFECT｜source eligibility 只检查 source_id，不检查 required concepts/vintage

**位置**：`factor_engine/mining/operator_catalog.py`

**当前问题**

存在 `fundamental_pit` 不等于存在 depreciation/revision vintage/analyst target 等真实字段。

**必须怎么改**

SourceCapability 至少包含 source_id/concepts/grain/pit_mode/vintage_support/market；required_concepts 做 subset。

**必须新增/修改测试**

同 source_id 但缺 concept 必须 SOURCE_CONCEPT_MISSING。

### R16-015｜P0｜CONFIRMED_CURRENT_DEFECT｜多源 operator contract 覆盖极少

**位置**：`factor_engine/mining/operator_catalog.py`

**当前问题**

benchmark、daily+fundamental、minute+daily-limit、group membership、relation exposure 等仍可能被单源 fallback。

**必须怎么改**

每个 panel param 绑定 SourceConcept，required source 从 typed signature 自动合成；central list 只做 migration。

**必须新增/修改测试**

审计所有 panel_arity>1 operator 的 required source completeness。

### R16-016｜P0｜CONFIRMED_CURRENT_DEFECT｜FieldName 被当 input_semantic_types

**位置**：`factor_engine/mining/operator_catalog.py::_input_semantic_types`

**当前问题**

函数优先读 input_fields 后返回，导致字段名与 SemanticType 混在一个字段。

**必须怎么改**

拆 `input_field_concepts` 与 `input_semantic_types`，不允许互相 fallback。

**必须新增/修改测试**

manifest/admission 同时输出且类型明确。

### R16-017｜P0｜CONFIRMED_CURRENT_DEFECT｜mining_eligible 本身没有 market context

**位置**：`factor_engine/mining/operator_catalog.py::mining_eligible`

**当前问题**

市场过滤只在外层 get_mining_operators；直接调用所谓 single authority 时可能错放 A股专用 op。

**必须怎么改**

统一 `MiningContext` 对象，market/frequency/source/session/cost 全进入同一个 eligibility。

**必须新增/修改测试**

直接调用 market mismatch 必须 false。

### R16-018｜P0｜CONFIRMED_CURRENT_DEFECT｜target_frequency 不是严格枚举

**位置**：`factor_engine/mining/operator_catalog.py`

**当前问题**

未知字符串可能不 fail closed。

**必须怎么改**

TargetFrequency enum + strict binder。

**必须新增/修改测试**

`weekly/foo` 必须 ValueError。

### R16-019｜P1｜CONFIRMED_CURRENT_DEFECT｜market fallback 默认 ASHARE+US 过宽

**位置**：`factor_engine/mining/operator_catalog.py`

**当前问题**

缺 market metadata 的 specialized operator 被默认认为两市场都支持。

**必须怎么改**

metadata 增 supported_markets 或 market_semantics=agnostic；specialized 缺声明 fail closed。

**必须新增/修改测试**

A股 limit-state 不得进入 US search space。

### R16-020｜P0｜CONFIRMED_CURRENT_DEFECT｜ESTIMATOR_RESOLUTION 仍通过 coarse lane 进入 mining

**位置**：`factor_engine/cleaned_operators/base.py + mining/operator_catalog.py`

**当前问题**

COARSE_SEARCH_ROLES 含 ESTIMATOR_RESOLUTION，catalog 又把 coarse params 交给算法；bins/grid/ridge 等继续扩 AST。

**必须怎么改**

默认 mining 只搜 ECONOMIC/HORIZON/STATE_THRESHOLD/MODEL_ORDER。Estimator resolution 只走 audited preset/robustness lane。

**必须新增/修改测试**

default manifest 不含 estimator knobs。

### R16-021｜P0｜CONFIRMED_CURRENT_DEFECT｜Admission --strict 实现与帮助文案不一致

**位置**：`factor_engine/audit/operator_admission_matrix.py`

**当前问题**

strict 主要看 admission_state unresolved，并未保证所有 P0 blocker/denied/stale evidence 都非零。

**必须怎么改**

blocker 带 severity/release_blocking；strict 对任何 blocking blocker 非零。

**必须新增/修改测试**

逐 P0 blocker mutation。

### R16-022｜P1｜CONFIRMED_CURRENT_DEFECT｜Admission blocker 被 generic B02 压扁

**位置**：`factor_engine/audit/operator_admission_matrix.py`

**当前问题**

compat/diagnostic/benchmark/edge/checkpoint/full-history 等可丢失精确原因。

**必须怎么改**

blocker 独立计算，可一 canonical 多 blocker，不做早退出。

**必须新增/修改测试**

多 blocker fixture 保留完整集合。

### R16-023｜P0｜CONFIRMED_CURRENT_DEFECT｜recursive math 与 checkpoint execution 混为一谈

**位置**：`factor_engine/audit/operator_admission_matrix.py`

**当前问题**

full-history recursive op 可能被标 recursive=False。

**必须怎么改**

拆 stateful_math/recursive_math/checkpoint_supported/incremental_supported/full_history_required。

**必须新增/修改测试**

recursive no-checkpoint fixture。

### R16-024｜P0｜CONFIRMED_CURRENT_DEFECT｜runtime manifest 默认 pending 容易误消费

**位置**：`factor_engine/scripts/export_mining_manifest.py`

**当前问题**

默认 admission=pending；下游若只读 canonical 列表就可能实际挖未认证 operator。

**必须怎么改**

严格拆 `runtime_mining_manifest.eligible` 与 `operator_remediation_plan.pending` 两个 schema。

**必须新增/修改测试**

runtime loader 拒绝 pending schema。

### R16-025｜P0｜CONFIRMED_CURRENT_DEFECT｜MISSING default 被序列化成 None

**位置**：`factor_engine/scripts/export_mining_manifest.py::_searchable_schema`

**当前问题**

“无默认值”与“默认值 None”不可区分。

**必须怎么改**

JSON 使用 has_default 或 tagged sentinel。

**必须新增/修改测试**

两个 case 序列化必须不同。

### R16-026｜P0｜CONFIRMED_CURRENT_DEFECT｜manifest schema export error 被 except/pass 吞掉

**位置**：`factor_engine/scripts/export_mining_manifest.py`

**当前问题**

schema 构建失败会静默变空字典。

**必须怎么改**

mineable operator schema error 直接 abort；planning 可 AUDIT_ERROR。

**必须新增/修改测试**

故意 schema error 必须 non-zero。

### R16-027｜P0｜CONFIRMED_CURRENT_DEFECT｜registry fingerprint 只 hash key names

**位置**：`factor_engine/scripts/export_mining_manifest.py`

**当前问题**

unit/role/default/ParamSpec 值改变但 keys 不变时 fingerprint 可能不变。

**必须怎么改**

hash 规范化完整逻辑 catalog + aliases + signature + contracts + impl hash。

**必须新增/修改测试**

只改 output_unit 也必须 fingerprint change。

### R16-028｜P0｜CONFIRMED_CURRENT_DEFECT｜cold-start validation 不先校验 artifact lineage

**位置**：`factor_engine/scripts/export_mining_manifest.py`

**当前问题**

旧 manifest 可在新代码上继续 OK。

**必须怎么改**

第一阶段严格校验 commit/dirty/registry/evidence/contract fingerprints。

**必须新增/修改测试**

旧 commit manifest 在新 commit 上必须 STALE。

### R16-029｜P1｜CONFIRMED_CURRENT_DEFECT｜cold-start validation 没逐条执行真实 seed expressions

**位置**：`factor_engine/scripts/export_mining_manifest.py`

**当前问题**

当前主要证明 registry/alias 可达，不能证明每个 cold-start factor 真能 bind/source/execute。

**必须怎么改**

逐 factor_id parse→type→bind→source→smoke execute。

**必须新增/修改测试**

非法 relational params seed 必须被抓。

### R16-030｜P0｜CONFIRMED_CURRENT_DEFECT｜closure UNUSED 与新 source semantics 接口冲突

**位置**：`factor_engine/cleaned_operators/closure/closure_audit.py`

**当前问题**

`available_sources=None` 在新 mining 语义是 UNKNOWN→fail closed，但 closure 仍用它判断 public operator 是否 unused，导致正常 op 被误判。

**必须怎么改**

closure 只检查 intrinsic reachability；真实 source eligibility 必须在具名 MiningContext 下判断。

**必须新增/修改测试**

UNKNOWN source context 不得生成 UNUSED。

### R16-031｜P0｜CONFIRMED_CURRENT_DEFECT｜minute→daily EOD same_session_usable 检查方向错误

**位置**：`factor_engine/cleaned_operators/closure/closure_audit.py`

**当前问题**

EOD factor 正确状态应是 same_session_usable=False，closure 不应把它视作 grain violation。

**必须怎么改**

检查 available_at=session_close、same_session_usable=False、SessionContract 存在。

**必须新增/修改测试**

标准 EOD op PASS；盘中使用 FAIL。

### R16-032｜P0｜CONFIRMED_CURRENT_DEFECT｜grain audit 漏只声明一侧 grain 的 operator

**位置**：`factor_engine/cleaned_operators/closure/closure_audit.py`

**当前问题**

只在两侧 grain 都存在时比较会漏掉真正的缺声明。

**必须怎么改**

按 role/category/source 判断哪些 canonical 必须有完整 GrainContract；缺任一字段 FAIL。

**必须新增/修改测试**

删除 output_grain mutation 必须失败。

### R16-033｜P0｜CONFIRMED_CURRENT_DEFECT｜closure scanner 跳过 `_*.py` 私有共享内核

**位置**：`factor_engine/cleaned_operators/closure/closure_audit.py`

**当前问题**

`_rolling_fast.py/_numpy_kernels.py/_dedupe.py` 等恰好被文件名规则排除。

**必须怎么改**

递归 AST 扫描所有 production source；只排 generated/vendor/cache。

**必须新增/修改测试**

在 `_rolling_fast.py` 注入风险模式必须命中。

### R16-034｜P0｜CONFIRMED_CURRENT_DEFECT｜skipped file 不会使 canonical 进入 UNSCANNED

**位置**：`factor_engine/cleaned_operators/closure/closure_audit.py`

**当前问题**

实现文件被 skip 后，审计仍可能声称全 registry covered。

**必须怎么改**

每 canonical 绑定 implementation files；任一 required file 未扫描 => NOT_RUN。

**必须新增/修改测试**

skip 真实 canonical file，release_safe=false。

### R16-035｜P0｜CONFIRMED_CURRENT_DEFECT｜include_behavioral=False 时 NOT_RUN 被空列表伪装 PASS

**位置**：`factor_engine/cleaned_operators/closure/closure_audit.py`

**当前问题**

没有 detector execution state，关闭 detector 后 findings=[] 容易被当作通过。

**必须怎么改**

每 detector 五态 PASS/FAIL/N/A/NOT_RUN/AUDIT_ERROR；required detector NOT_RUN 阻断。

**必须新增/修改测试**

关闭 behavioral detector 的 release mode 必须失败。

### R16-036｜P0｜CONFIRMED_CURRENT_DEFECT｜panel 参数仍靠名字 whitelist

**位置**：`factor_engine/cleaned_operators/closure/closure_audit.py::_PANEL_PARAMS`

**当前问题**

新 panel 名如 group_id/high_limit/activity/f1/benchmark 很容易漏。

**必须怎么改**

删除名字猜测，只读 ResolvedSignature/panel_params typed contract。

**必须新增/修改测试**

任意新 panel 名不改 scanner 也能识别。

### R16-037｜P0｜CONFIRMED_CURRENT_DEFECT｜central declared policy 用 replace 覆盖 local contract

**位置**：`factor_engine/cleaned_operators/closure/declared_policies.py`

**当前问题**

后加载 side-registry 可以把实现 local semantics 覆盖成另一套“权威”而不报警。

**必须怎么改**

merge conflict-detect；两处非空声明不一致直接 load/release fail；freeze 后 immutable。

**必须新增/修改测试**

local FULL_WINDOW vs central MIN_SUPPORT 冲突测试。

### R16-038｜P0｜CONFIRMED_CURRENT_DEFECT｜时间几何/频谱算子被错误分到 pairwise-valid missing

**位置**：`factor_engine/cleaned_operators/closure/declared_policies.py`

**当前问题**

TE/kernel Granger/cross-spectrum/intrinsic-dimension/Lyapunov 等依赖物理 lag/frequency，不能删除缺失行再当连续时间。

**必须怎么改**

使用 PHYSICAL_AXIS_BREAK/CONTIGUOUS_REQUIRED/LAG_PRESERVING policy。

**必须新增/修改测试**

中间 NaN 不得重连 lag/frequency。

### R16-039｜P0｜CONFIRMED_CURRENT_DEFECT｜semantic audit 全 catalog 只跑少量规则

**位置**：`factor_engine/cleaned_operators/semantic_audit.py`

**当前问题**

param-role/relational/missing/group migration/native cohort/PSD/golden 等没有全 applicable canonical 执行。

**必须怎么改**

family-required-rule matrix；每 `(canonical,rule)` 恰好一个 outcome。

**必须新增/修改测试**

缺 required rule execution 必须 coverage fail。

### R16-040｜P0｜CONFIRMED_CURRENT_DEFECT｜param_role_declared 注释与实际 rule list 不一致

**位置**：`factor_engine/cleaned_operators/semantic_audit.py`

**当前问题**

注释声称生产 target 全查 ParamRole，但 rule 没真正进入全量 catalog sweep。

**必须怎么改**

由机器 required-rule matrix 决定，不靠注释。

**必须新增/修改测试**

新增 scalar 无 role 必须被 semantic audit 捕获。

### R16-041｜P0｜CONFIRMED_CURRENT_DEFECT｜semantic rule 可能同时 checked 与 N/A

**位置**：`factor_engine/cleaned_operators/semantic_audit.py::_run_rule_sweep`

**当前问题**

rule 内 note_skipped 后外层仍可能 note_ran，coverage 双计。

**必须怎么改**

规则返回结构化 outcome，不再 side-effect 计数。

**必须新增/修改测试**

每 canonical-rule key 唯一。

### R16-042｜P0｜CONFIRMED_CURRENT_DEFECT｜certification_safe 不要求 required coverage

**位置**：`factor_engine/cleaned_operators/semantic_audit.py`

**当前问题**

大量 fixture NOT_APPLICABLE/NOT_RUN 时只要没有 error finding 就可能 safe。

**必须怎么改**

认证要求 required rules 全 PASS 或 reviewed N/A；NOT_RUN/AUDIT_ERROR 绝不 safe。

**必须新增/修改测试**

全部 behavioral NOT_RUN fixture 必须 uncertified。

### R16-043｜P0｜CONFIRMED_CURRENT_DEFECT｜semantic contract_fixture 仍会猜 unary x

**位置**：`factor_engine/cleaned_operators/semantic_audit.py`

**当前问题**

没有 panel metadata 时会猜 unary，binary/variadic fixture 可能错误。

**必须怎么改**

只根据 ResolvedSignature + SemanticType 构造 fixture；无法解析= AUDIT_ERROR。

**必须新增/修改测试**

binary/ternary/optional-panel golden。

### R16-044｜P0｜CONFIRMED_CURRENT_DEFECT｜semantic evidence record 不携带完整 rule outcome

**位置**：`factor_engine/cleaned_operators/semantic_audit.py`

**当前问题**

“有 evidence record”不代表 canonical 所有 required rules 通过。

**必须怎么改**

evidence 引用 outcome-matrix hash + required coverage。

**必须新增/修改测试**

一条 required FAIL 时不得 semantic_verified。

### R16-045｜P0｜CONFIRMED_CURRENT_DEFECT｜source static audit 永远 return 0

**位置**：`factor_engine/tools/audit_operator_source.py`

**当前问题**

高危 hit 只打印，不成为 CI blocker。

**必须怎么改**

finding stable ID + severity + disposition + owning canonical；未处置 HIGH/UNKNOWN 非零退出。

**必须新增/修改测试**

插 silent clamp 高危模式，CI fail。

### R16-046｜P0｜CONFIRMED_CURRENT_DEFECT｜当前 operator evidence 明确落后于 HEAD

**位置**：`factor_engine/evidence/factor_operator_verified.json`

**当前问题**

evidence commit `0b659eca...` 与本轮 latest main `1dc10425...` 不同。

**必须怎么改**

最终 freeze 后同 commit 重生；key 绑定 implementation/contract/test/runtime hashes。

**必须新增/修改测试**

任意实现或 contract 改动后旧证据 STALE。

### R16-047｜P0｜MANDATORY_FULL_AUDIT｜load_all 每个 finalization pass 必须输出 semantic mutation diff

**位置**：`factor_engine/cleaned_operators/__init__.py`

**当前问题**

loader 有 dedupe/governance/hardening/evidence/fiscal re-register/backend bridge/closure 多轮后处理，源文件正确不代表 final registry 正确。

**必须怎么改**

每 pass before/after diff；evidence 生成后 semantic fields 不可再变；只审 FinalRegistrySnapshot。

**必须新增/修改测试**

freeze 后 semantic mutation hard fail。

### R16-048｜P1｜MANDATORY_FULL_AUDIT｜public caller 不得观察 partial registry

**位置**：`factor_engine/cleaned_operators/__init__.py`

**当前问题**

初始化重入可必要，但 public list/get 若在 `_INITIALIZING` 期间读取可能看到不完整 registry。

**必须怎么改**

区分 internal reentrant access 与 public access；public 在 freeze 前 RegistryNotReady。

**必须新增/修改测试**

并发/重入测试。

### R16-049｜P0｜CONFIRMED_CURRENT_DEFECT｜float ParamSpec 验证后未 canonicalize 类型

**位置**：`factor_engine/cleaned_operators/base.py`

**当前问题**

float spec 可接受整数但仍把原始 `1` 留到 runtime/hash，和 `1.0` 两种表示并存。

**必须怎么改**

binder 返回 canonical typed value。

**必须新增/修改测试**

`p=1` 与 `p=1.0` canonical AST/hash 一致。

### R16-050｜P0｜CONFIRMED_CURRENT_DEFECT｜active_when 显式 None 与 unbound 混淆

**位置**：`factor_engine/cleaned_operators/base.py`

**当前问题**

`bound.get(controller)` 会把显式 None 当没传，再回退 default。

**必须怎么改**

判断 `controller in bound`；None 是真实显式值。

**必须新增/修改测试**

default 非None、调用显式None dependency test。

### R16-051｜P0｜CONFIRMED_CURRENT_DEFECT｜strict_float 返回原始 raw 值

**位置**：`factor_engine/cleaned_operators/common/strict_params.py`

**当前问题**

校验后没有统一 return float。

**必须怎么改**

先严格拒 bool/string，再 `return float(value)`。

**必须新增/修改测试**

np numeric 合法、bool/string 非法。

### R16-052｜P0｜CONFIRMED_CURRENT_DEFECT｜strict_probability 可接受字符串数值

**位置**：`factor_engine/cleaned_operators/common/strict_params.py`

**当前问题**

`float('0.5')` 成功，strict gate 可把字符串洗进 kernel。

**必须怎么改**

先要求 Real 且非 bool；string 一律拒。

**必须新增/修改测试**

`'0.5'` ValueError。

### R16-053｜P1｜CONFIRMED_CURRENT_DEFECT｜strict_enum 有跨类型 equality 漏洞

**位置**：`factor_engine/cleaned_operators/common/strict_params.py`

**当前问题**

Python 中 True==1、1==1.0；choices membership 不区分类型。

**必须怎么改**

type-aware membership 或强 Enum。

**必须新增/修改测试**

choices=(1,2) 时 True 拒绝。

### R16-054｜P1｜CONFIRMED_CURRENT_DEFECT｜bool panel validator 可能把字符串列自动转数值

**位置**：`factor_engine/cleaned_operators/common/strict_params.py`

**当前问题**

dtype=float coercion 会把 `'0'/'1'` 洗成合法 condition。

**必须怎么改**

operator semantic boundary 不做字符串 coercion；转换只能发生在 SourceAdapter 并留 lineage。

**必须新增/修改测试**

object string panel 拒绝。

### R16-055｜P0｜MANDATORY_FULL_AUDIT｜SignatureDefault == ParamSpecDefault 全库 invariant

**位置**：`FinalRegistrySnapshot`

**当前问题**

最新很多 family 仍出现 Python signature 有 default、ParamSpec 无 default 或值不同。

**必须怎么改**

每 scalar param 比较 canonicalized default；无默认使用 MISSING sentinel。

**必须新增/修改测试**

全 registry mismatch=0。

### R16-056｜P0｜MANDATORY_FULL_AUDIT｜mineable scalar ParamSpec 覆盖必须100%

**位置**：`FinalRegistrySnapshot`

**当前问题**

extended/research-promoted family 仍有大量只在 kernel int/float 的 scalar。

**必须怎么改**

`scalar_params - ParamSpec.keys == ∅`；private fixed knob 不得在 logical signature。

**必须新增/修改测试**

全 registry invariant。

### R16-057｜P0｜CONFIRMED_CURRENT_DEFECT｜Polars state/event 仍用 finite&!=0 truthiness

**位置**：`factor_engine/cleaned_operators/common/polars_state_event.py`

**当前问题**

-1/.2/2 都会被当 True，与 strict tri-state 不一致。

**必须怎么改**

所有 backend 共用 ConditionBool/EventBool validation。

**必须新增/修改测试**

非法值 Pandas/Polars 同错误。

### R16-058｜P0｜CONFIRMED_CURRENT_DEFECT｜Polars state/event 仍有 int truncation/clamp

**位置**：`factor_engine/cleaned_operators/common/polars_state_event.py`

**当前问题**

`_pi/int(value)`、min_events/max_lookback 等仍自行转换。

**必须怎么改**

backend 只接收逻辑 binder 已验证参数。

**必须新增/修改测试**

5.9/True/NaN 路由前失败。

### R16-059｜P0｜CONFIRMED_CURRENT_DEFECT｜Polars 多输入 `_cols` 取交集

**位置**：`factor_engine/cleaned_operators/common/polars_state_event.py`

**当前问题**

列 mismatch 时静默缩 universe。

**必须怎么改**

无 BroadcastSpec 时 exact axis/order required。

**必须新增/修改测试**

缺股票/乱序/重复列测试。

### R16-060｜P0｜CONFIRMED_CURRENT_DEFECT｜Polars return-decomposition 忽略 price_basis gate

**位置**：`factor_engine/cleaned_operators/common/polars_state_event.py`

**当前问题**

domain gate 只存在 metadata/Pandas 路径，native Polars 可绕过。

**必须怎么改**

semantic/domain validation backend-independent。

**必须新增/修改测试**

非法 basis 两后端一致拒绝。

### R16-061｜P0｜CONFIRMED_CURRENT_DEFECT｜SQL days-since-high/low tie 仍是旧语义

**位置**：`factor_engine/backend/sql_pushdown/emitter.py`

**当前问题**

仓库最新 semantic report 仍记录 SQL first-hit，而 canonical Pandas 是 recent tie。

**必须怎么改**

SQL 改 most-recent tie；修前从 fastpath 移除。

**必须新增/修改测试**

tie-rich window SQL/Pandas golden。

### R16-062｜P0｜CONFIRMED_CURRENT_DEFECT｜SQL ts_impulse_strength 仍可能把 current volume 混进 baseline

**位置**：`factor_engine/backend/sql_pushdown/emitter.py`

**当前问题**

Pandas 使用 strictly-prior baseline，SQL 仍有 current contamination 残余。

**必须怎么改**

window frame 截到 1 PRECEDING。

**必须新增/修改测试**

只改 current volume 不得改 prior baseline。

### R16-063｜P0｜MANDATORY_FULL_AUDIT｜backend parity 必须比较错误语义

**位置**：`all pandas/polars/sql backends`

**当前问题**

只做合法 finite allclose 抓不到 axis/domain/condition/missing 差异。

**必须怎么改**

每 backend parity 同时跑 normal、NaN/Inf、invalid param/domain、axis errors；异常代码也一致。

**必须新增/修改测试**

输出 backend_semantic_parity_matrix.json。

### R16-064｜P0｜CONFIRMED_CURRENT_DEFECT｜execution_contract 仍保留多套 stateful truth

**位置**：`factor_engine/runtime/execution_contract.py`

**当前问题**

legacy `_STATEFUL_CANONICALS`、operator declarations、checkpoint registry 同时存在；还有 stale canonical 名风险。

**必须怎么改**

最终 state/history contract 由 canonical self-declaration/ResolvedExecutionContract 单一来源；legacy seed 只迁移检查。

**必须新增/修改测试**

legacy seed 每个名字必须 resolve final canonical。

### R16-065｜P1｜CONFIRMED_CURRENT_DEFECT｜legacy stateful seed 仍有退休名称残留风险

**位置**：`factor_engine/runtime/execution_contract.py`

**当前问题**

例如旧 `ts_cusum_break_score` 一类名字残留，会让 history/state 审计针对不存在 canonical。

**必须怎么改**

启动时断言所有 legacy seed 在 final registry 或有显式 migration mapping，否则失败。

**必须新增/修改测试**

stale name test。

### R16-066｜P0｜MANDATORY_FULL_AUDIT｜history requirement 不得继续猜参数名

**位置**：`runtime execution planner`

**当前问题**

window/lookback/history_days 等名字不能证明 history kind；event-count/session-count/embedding history 都不同。

**必须怎么改**

每 mineable op 暴露 `HistoryRequirementFactory(bound_params)`。

**必须新增/修改测试**

final mining candidate fallback_usage_count=0。

### R16-067｜P0｜CONFIRMED_CURRENT_DEFECT｜rolling_time_slope 仍 silent int/clamp

**位置**：`factor_engine/cleaned_operators/_rolling_fast.py::rolling_time_slope`

**当前问题**

`max(2,int(window/min_periods))` 仍在共享 kernel；而该私有文件又被旧 scanner 跳过。

**必须怎么改**

kernel 只接受 binder 已验证 int；删除 cast/clamp。

**必须新增/修改测试**

2.9/True/0/NaN 全部 binder fail。

### R16-068｜P0｜CONFIRMED_CURRENT_DEFECT｜rolling regression/beta 把 Inf 当 paired observation

**位置**：`factor_engine/cleaned_operators/_rolling_fast.py`

**当前问题**

paired cohort 仍可能用 notna，Inf 被视为 observed。

**必须怎么改**

所有 paired stats 共用 finite mask。

**必须新增/修改测试**

Inf 注入回归/β/corr test。

### R16-069｜P1｜CONFIRMED_CURRENT_DEFECT｜cumulative top/bottom-k 会 silently 降 k

**位置**：`factor_engine/cleaned_operators/_rolling_fast.py`

**当前问题**

`take=min(k,valid.size)` 使 top5 在只有2个样本时变 top2，和 rolling top-k policy 不一致。

**必须怎么改**

固定 k support，不足 NaN；如需 available-k 另 canonical。

**必须新增/修改测试**

k-1 样本不输出。

### R16-070｜P1｜CONFIRMED_CURRENT_DEFECT｜partial WMA 权重身份不清

**位置**：`factor_engine/cleaned_operators/_rolling_fast.py::rolling_linear_weighted`

**当前问题**

W=20 只有2条时使用19:20，而标准 partial WMA 常用1:2。

**必须怎么改**

production 推荐 FULL_WINDOW；若保留 partial，拆 FixedAgeWMA 与 RenormalizedPartialWMA。

**必须新增/修改测试**

warmup 长度1..W golden。

### R16-071｜P0｜CONFIRMED_CURRENT_DEFECT｜Beta wrapper 又把 support 降回2

**位置**：`factor_engine/cleaned_operators/common/statistics.py::Beta`

**当前问题**

wrapper 显式 min_periods=2 覆盖共享 rolling_beta 更严格 floor；Intercept 同类。

**必须怎么改**

support policy canonical-level 单一来源。

**必须新增/修改测试**

2/3/4 samples 不提前 production output。

### R16-072｜P1｜CONFIRMED_CURRENT_DEFECT｜Corr/Cov/Covariance min_periods=1 造成 estimator support 漂移

**位置**：`factor_engine/cleaned_operators/common/statistics.py`

**当前问题**

window固定但 N 从很小逐步变化。

**必须怎么改**

FULL_WINDOW 或 reviewed minimum coverage；support policy non-searchable。

**必须新增/修改测试**

effective-N golden。

### R16-073｜P0｜CONFIRMED_CURRENT_DEFECT｜statistics aligned pair dropna 不滤 Inf

**位置**：`factor_engine/cleaned_operators/common/statistics.py`

**当前问题**

`dropna()` 对 ±Inf 无效。

**必须怎么改**

统一 finite-pair helper。

**必须新增/修改测试**

Inf cases。

### R16-074｜P1｜CONFIRMED_CURRENT_DEFECT｜Mad docstring 与 final alias target 不一致

**位置**：`factor_engine/cleaned_operators/common/statistics.py + _dedupe.py`

**当前问题**

代码 alias 已修成标准 mean absolute deviation，但文档仍描述旧非标准映射。

**必须怎么改**

文档/catalog 从 final definition spec 自动生成。

**必须新增/修改测试**

definition_id/doc snapshot parity。

### R16-075｜P1｜CONFIRMED_CURRENT_DEFECT｜DEDUPE_ALIASES 存在 self-loop

**位置**：`factor_engine/cleaned_operators/_dedupe.py`

**当前问题**

`canonical -> canonical` 的 alias 没价值并污染 graph。

**必须怎么改**

删除 self edge；alias graph DAG 且 old!=target。

**必须新增/修改测试**

no self/no cycle invariant。

### R16-076｜P0｜MANDATORY_FULL_AUDIT｜所有 alias edge 需要行为等价证明

**位置**：`factor_engine/cleaned_operators/_dedupe.py`

**当前问题**

名字相似不代表数学等价。

**必须怎么改**

每 alias edge 绑定 equivalence test；无 proof 则 compatibility adapter/删除。

**必须新增/修改测试**

alias_equivalence_report 100% edge coverage。

### R16-077｜P1｜CONFIRMED_CURRENT_DEFECT｜cross_section_local k/ridge 仍可被 coarse mining 搜

**位置**：`factor_engine/cleaned_operators/cross_section_local.py`

**当前问题**

注释说不自由搜，但 ESTIMATOR_RESOLUTION + searchable 默认 True 仍会被当前 coarse lane 暴露。

**必须怎么改**

显式 searchable=False，并从 default mining 排 estimator role。

**必须新增/修改测试**

runtime manifest 不含 k/ridge default search。

### R16-078｜P0｜CONFIRMED_CURRENT_DEFECT｜rank-copula grid runtime 有 choices 但 metadata 无 ParamSpec

**位置**：`factor_engine/cleaned_operators/cross_section_local.py`

**当前问题**

`grid in {4,8,16}` 只存在 runtime。

**必须怎么改**

ParamSpec choices/default/ESTIMATOR_RESOLUTION/searchable=False。

**必须新增/修改测试**

compile/runtime legal grid 一致。

### R16-079｜P1｜CONFIRMED_CURRENT_DEFECT｜bias-corrected MI 的 output range/命名不清

**位置**：`factor_engine/cleaned_operators/cross_section_local.py::cs_rank_copula_mi`

**当前问题**

有限样本修正后可轻微负值，下游若按 MI>=0 会错。

**必须怎么改**

明确 `bias_corrected_mi` semantics/output_range；不要随意 clip。

**必须新增/修改测试**

independent null around zero test。

### R16-080｜P1｜CONFIRMED_CURRENT_DEFECT｜KNN tangent 零 local scale 用 EPS 造 finite denominator

**位置**：`factor_engine/cleaned_operators/cross_section_local.py`

**当前问题**

完全重合 peer cloud 本应不可识别。

**必须怎么改**

scale<=relative tolerance => NaN。

**必须新增/修改测试**

identical peer cloud。

### R16-081｜P0｜CONFIRMED_CURRENT_DEFECT｜group spectrum 隐藏60历史状态未进入 HistoryRequirement

**位置**：`factor_engine/cleaned_operators/group_spectrum.py`

**当前问题**

breadth_history/deque 影响今天输出，incremental planner 未必知道要60历史。

**必须怎么改**

声明 HistoryRequirement 或 checkpoint breadth state。

**必须新增/修改测试**

full vs chunked 多切点 parity。

### R16-082｜P0｜CONFIRMED_CURRENT_DEFECT｜group breadth deque 按 valid observation 而非物理交易日推进

**位置**：`factor_engine/cleaned_operators/group_spectrum.py`

**当前问题**

group 缺失很久再出现可继续用很久以前 breadth。

**必须怎么改**

存 date + max_gap/physical trailing days；absence 推进时间或 reset。

**必须新增/修改测试**

group 消失100日再出现。

### R16-083｜P1｜CONFIRMED_CURRENT_DEFECT｜group spectrum 只拒 raw bitwise duplicate feature

**位置**：`factor_engine/cleaned_operators/group_spectrum.py`

**当前问题**

f2=2*f1+c 标准化后完全重复仍进入 SVD。

**必须怎么改**

标准化后做 affine duplicate/rank/condition gate；AST 也做 redundancy。

**必须新增/修改测试**

affine-copy fixture。

### R16-084｜P0｜CONFIRMED_CURRENT_DEFECT｜group_schema_version scalar 无法表达历史中途换分类体系

**位置**：`factor_engine/cleaned_operators/group_spectrum.py`

**当前问题**

一个全样本字符串无法表示 PIT taxonomy migration。

**必须怎么改**

schema version 作为 date-level SourceContext；变更时 segment/reset state。

**必须新增/修改测试**

mid-history migration fixture。

### R16-085｜P1｜CONFIRMED_CURRENT_DEFECT｜DMD 零 amplitude 被 EPS 赋假能量

**位置**：`factor_engine/cleaned_operators/dmd.py`

**当前问题**

`log(|b|²+EPS)` 让 b=0 mode 有非零能量。

**必须怎么改**

b==0 -> -inf log energy。

**必须新增/修改测试**

zero amplitude mode fixture。

### R16-086｜P1｜CONFIRMED_CURRENT_DEFECT｜DMD λ=0 被 EPS 变 arbitrary finite growth

**位置**：`factor_engine/cleaned_operators/dmd.py`

**当前问题**

`log(max(|λ|,EPS))` 信号由程序常数决定。

**必须怎么改**

λ=0 NaN 或理论定义的 versioned cap。

**必须新增/修改测试**

zero eigenvalue fixture。

### R16-087｜P1｜CONFIRMED_CURRENT_DEFECT｜DMD pinv rcond 未版本化

**位置**：`factor_engine/cleaned_operators/dmd.py`

**当前问题**

`np.linalg.pinv` 默认 rcond 可能随库版本改变有效秩。

**必须怎么改**

NumericalPolicy 显式 rcond，并进 definition/evidence hash。

**必须新增/修改测试**

near-rank-deficient golden。

### R16-088｜P1｜CONFIRMED_CURRENT_DEFECT｜DMD imaginary-mode threshold 是绝对隐藏常数

**位置**：`factor_engine/cleaned_operators/dmd.py`

**当前问题**

`abs(Im λ)>=1e-9` 无相对 scale 语义。

**必须怎么改**

relative tolerance + versioned policy。

**必须新增/修改测试**

threshold-near eigenvalue test。

### R16-089｜P0｜CONFIRMED_CURRENT_DEFECT｜turnover_survival 整族 ParamSpec 仍不完整

**位置**：`factor_engine/cleaned_operators/turnover_survival.py`

**当前问题**

window/band/q 等仍 `_run_all` 中 int/float/max。

**必须怎么改**

全 scalar ParamSpec/role/default/relations；kernel不自行修参数。

**必须新增/修改测试**

scalar contract coverage。

### R16-090｜P0｜CONFIRMED_CURRENT_DEFECT｜turnover_survival 单输出也重复计算整套统计且 cost=1

**位置**：`factor_engine/cleaned_operators/turnover_survival.py`

**当前问题**

一个 canonical 调用 `_run_all()` 会同时算多项昂贵结果，真实成本明显高于 metadata。

**必须怎么改**

共享 multi-output cache/CSE 或拆 kernel，按真实复杂度标 cost。

**必须新增/修改测试**

profile + cost budget test。

### R16-091｜P1｜CONFIRMED_CURRENT_DEFECT｜turnover chip entropy 文档仍与实现 reference 不一致

**位置**：`factor_engine/cleaned_operators/turnover_survival.py`

**当前问题**

doc 仍可描述相对 current price，但 kernel 已使用 RP。

**必须怎么改**

统一 definition spec 自动生成 docs。

**必须新增/修改测试**

doc/definition snapshot。

### R16-092｜P0｜CONFIRMED_CURRENT_DEFECT｜vol-scaled chip entropy 复用 raw log-price edges

**位置**：`factor_engine/cleaned_operators/turnover_survival.py`

**当前问题**

标准化 z 后仍用 ±0.02/.05/.10/.20 raw log band，sigma-unit 语义错误。

**必须怎么改**

standardized z 使用独立 sigma-unit grid，经校准版本化。

**必须新增/修改测试**

vol scale invariance test。

### R16-093｜P1｜CONFIRMED_CURRENT_DEFECT｜normalized chip entropy unit 仍不清

**位置**：`factor_engine/cleaned_operators/turnover_survival.py`

**当前问题**

除 log(nb) 后是 dimensionless [0,1]，不应和 raw nats entropy 同 unit。

**必须怎么改**

output_unit=dimensionless + output_range。

**必须新增/修改测试**

unit/range test。

### R16-094｜P0｜CONFIRMED_CURRENT_DEFECT｜stratified q=.5 在奇数 N 时上下组重叠

**位置**：`factor_engine/cleaned_operators/weighted_tail.py`

**当前问题**

`k=round(q*N)`；N=7,q=.5 => k=4，上下各4重叠。

**必须怎么改**

floor(q*N) + `2*k<=N` + min_stratum_count。

**必须新增/修改测试**

odd N/q=.5/tie golden。

### R16-095｜P0｜CONFIRMED_CURRENT_DEFECT｜weighted_tail ParamSpec/role 覆盖不完整

**位置**：`factor_engine/cleaned_operators/weighted_tail.py`

**当前问题**

window/q/min_periods/target/min_tail_count 等仍 runtime cast/clamp。

**必须怎么改**

补完整 spec/default/role/relations。

**必须新增/修改测试**

full scalar coverage。

### R16-096｜P1｜CONFIRMED_CURRENT_DEFECT｜weighted expected shortfall unit 应 same_as:x

**位置**：`factor_engine/cleaned_operators/weighted_tail.py`

**当前问题**

ES 是 x 的尾部均值，generic level 不正确。

**必须怎么改**

output_unit=same_as:x。

**必须新增/修改测试**

unit propagation。

### R16-097｜P1｜CONFIRMED_CURRENT_DEFECT｜weighted ES support 不考虑有效权重样本量

**位置**：`factor_engine/cleaned_operators/weighted_tail.py`

**当前问题**

一个超大权重 + 多个极小权重也能过成员数 gate。

**必须怎么改**

加入 Kish effective-N / tail mass / positive-weight member floor。

**必须新增/修改测试**

集中权重 fixture。

### R16-098｜P1｜CONFIRMED_CURRENT_DEFECT｜weighted tail sorting cost 被标太低

**位置**：`factor_engine/cleaned_operators/weighted_tail.py`

**当前问题**

每窗口 sort/weighted quantile 不是 cost:1。

**必须怎么改**

真实 W log W CostModel。

**必须新增/修改测试**

cost 随 window 单调。

### R16-099｜P1｜CONFIRMED_CURRENT_DEFECT｜ts_abs_entropy 旧 normalize 参数已成为死参数

**位置**：`factor_engine/cleaned_operators/direction_concentration.py`

**当前问题**

normalize=False 已不合法，但 public logical signature 仍保留 normalize。

**必须怎么改**

旧 canonical 只 compatibility；mining 只留 normalized/nats 明确版本。

**必须新增/修改测试**

manifest 不暴露死 normalize。

### R16-100｜P0｜CONFIRMED_CURRENT_DEFECT｜direction_concentration scalar contract 仍大量缺失

**位置**：`factor_engine/cleaned_operators/direction_concentration.py`

**当前问题**

window/min_periods/threshold/tolerance 等仍 runtime int/float/max。

**必须怎么改**

补 ParamSpec/role/default。

**必须新增/修改测试**

family scalar coverage。

### R16-101｜P1｜CONFIRMED_CURRENT_DEFECT｜direction_concentration 另造一套 bool parser

**位置**：`factor_engine/cleaned_operators/direction_concentration.py`

**当前问题**

本地 strict bool 可接受字符串 true/false，与全局 policy 不一致。

**必须怎么改**

删除 local parser，复用 single authority。

**必须新增/修改测试**

跨 family bool parity。

### R16-102｜P0｜CONFIRMED_CURRENT_DEFECT｜CPT window 没 ParamSpec

**位置**：`factor_engine/cleaned_operators/prospect_theory.py::ts_cpt_value`

**当前问题**

文档说 only window searchable，但 metadata 没 window spec。

**必须怎么改**

window HORIZON/default/min 明确。

**必须新增/修改测试**

manifest default 与 signature=60。

### R16-103｜P0｜CONFIRMED_CURRENT_DEFECT｜CPT min_coverage compile/runtime 边界不一致

**位置**：`factor_engine/cleaned_operators/prospect_theory.py`

**当前问题**

ParamSpec 允许0，runtime 要 >0。

**必须怎么改**

exclusive lower bound或 relational >0。

**必须新增/修改测试**

0 binder fail。

### R16-104｜P0｜CONFIRMED_CURRENT_DEFECT｜CPT 缺 ReturnLike SemanticType

**位置**：`factor_engine/cleaned_operators/prospect_theory.py`

**当前问题**

raw price 可以绕过类型系统进入收益分布的 prospect-theory 公式。

**必须怎么改**

input semantic=ReturnLike/unit=return。

**必须新增/修改测试**

price compile fail。

### R16-105｜P1｜CONFIRMED_CURRENT_DEFECT｜CPT hidden support policy

**位置**：`factor_engine/cleaned_operators/prospect_theory.py`

**当前问题**

`max(5,w//10)` 影响统计稳定性却不在 machine contract。

**必须怎么改**

SupportPolicy 机器化且 non-searchable。

**必须新增/修改测试**

effective-N test。

### R16-106｜P1｜CONFIRMED_CURRENT_DEFECT｜CPT cost 低估 per-window sorting

**位置**：`factor_engine/cleaned_operators/prospect_theory.py`

**当前问题**

排序/decision weighting 不是简单 rolling。

**必须怎么改**

真实 cost model。

**必须新增/修改测试**

benchmark。

### R16-107｜P0｜CONFIRMED_CURRENT_DEFECT｜robust prior window off-by-one

**位置**：`factor_engine/cleaned_operators/robust_stats.py::_rolling_prior_apply_2d`

**当前问题**

`[t-W,t-1]` 当前最多取 W-1 条。

**必须怎么改**

`start=max(0,row-window)`，统一 prior-window helper。

**必须新增/修改测试**

W=20 exactly 20 prior observations。

### R16-108｜P0｜CONFIRMED_CURRENT_DEFECT｜robust_stats 多个 scalar 无 ParamSpec

**位置**：`factor_engine/cleaned_operators/robust_stats.py`

**当前问题**

q_low/q_high/trim_ratio/center/scale/clip 等缺 machine contract。

**必须怎么改**

补 choices/default/role/relations。

**必须新增/修改测试**

family scalar coverage。

### R16-109｜P1｜CONFIRMED_CURRENT_DEFECT｜inclusive robust zscore support 仍过松

**位置**：`factor_engine/cleaned_operators/robust_stats.py`

**当前问题**

极少样本可开始输出，与 prior family support 不一致。

**必须怎么改**

加入 audited min support；或 inclusive descriptive 不进 default alpha。

**必须新增/修改测试**

warmup golden。

### R16-110｜P0｜CONFIRMED_CURRENT_DEFECT｜conditional TE min_cells_ratio 大片 dead region

**位置**：`factor_engine/cleaned_operators/conditional_dependence.py`

**当前问题**

required 已至少3*cells，ratio<=3 永远不生效。

**必须怎么改**

删除该 param，或让 ratio 直接定义 required/cells。

**必须新增/修改测试**

injectivity。

### R16-111｜P0｜CONFIRMED_CURRENT_DEFECT｜conditional TE min_transitions 缺 ParamSpec

**位置**：`factor_engine/cleaned_operators/conditional_dependence.py`

**当前问题**

runtime 参数未进入 metadata。

**必须怎么改**

support/estimator ParamSpec + default + relation。

**必须新增/修改测试**

compile/runtime parity。

### R16-112｜P0｜CONFIRMED_CURRENT_DEFECT｜MODWT level 是 coverage-only 死参数

**位置**：`factor_engine/cleaned_operators/conditional_dependence.py::ts_modwt_band_corr`

**当前问题**

j==band 时已 return；所有 level>=band coefficient 数值相同，level 只改变可用性 gate。

**必须怎么改**

删除 level 或强制 level==band。

**必须新增/修改测试**

合法参数无同输出死维度。

### R16-113｜P0｜CONFIRMED_CURRENT_DEFECT｜intraday_session grain/availability contract 缺失

**位置**：`factor_engine/cleaned_operators/intraday_session.py`

**当前问题**

实际 minute input + close-row sparse output，metadata tags 却写 daily，mining/source 可能分类错。

**必须怎么改**

选择明确的 minute→daily EOD materialization，或定义专用 minute-session-close grain。

**必须新增/修改测试**

catalog role/source/frequency。

### R16-114｜P0｜CONFIRMED_CURRENT_DEFECT｜intraday_session 未按 calendar timezone 解析 index

**位置**：`factor_engine/cleaned_operators/intraday_session.py`

**当前问题**

UTC minute 数据会按 UTC date/minute-of-day 判断 session。

**必须怎么改**

SessionContext 转 exchange wall clock 后求 trade_date/slots。

**必须新增/修改测试**

UTC A股 fixture。

### R16-115｜P0｜CONFIRMED_CURRENT_DEFECT｜intraday_session calendar grid validation 不完整

**位置**：`factor_engine/cleaned_operators/intraday_session.py`

**当前问题**

segment duration 不保证可被 bar_freq 整除，overlap 可被 set 去重隐藏。

**必须怎么改**

SessionCalendar 构造时验证 ordered/disjoint/divisible/timezone/convention。

**必须新增/修改测试**

overlap/非法bar width fixture。

### R16-116｜P0｜CONFIRMED_CURRENT_DEFECT｜intraday_session 参数关系不完整

**位置**：`factor_engine/cleaned_operators/intraday_session.py`

**当前问题**

min_history_sessions>history_days 或 n_components+1>history_days 可 compile-valid 但全NaN。

**必须怎么改**

relational constraints。

**必须新增/修改测试**

非法组合 binder fail。

### R16-117｜P0｜CURRENT_REGRESSION｜intraday_session execution history_count 仍写死20

**位置**：`factor_engine/cleaned_operators/intraday_session.py::_declare_stateful_contracts`

**当前问题**

history_days 是参数，planner contract 固定20。

**必须怎么改**

HistoryRequirementFactory(bound_params)，session_count 参数化。

**必须新增/修改测试**

5/20/40 full/chunk parity。

### R16-118｜P0｜CONFIRMED_CURRENT_DEFECT｜volume_clock 未声明 minute→daily GrainTransform

**位置**：`factor_engine/cleaned_operators/volume_clock.py`

**当前问题**

模块自称 minute→daily，但 metadata 无 grain/availability。

**必须怎么改**

补标准 EOD GrainContract/SessionContract。

**必须新增/修改测试**

必须分类 INTRADAY_EOD、require minute_bar。

### R16-119｜P0｜CONFIRMED_CURRENT_DEFECT｜volume_clock 不检查 official session completeness

**位置**：`factor_engine/cleaned_operators/volume_clock.py`

**当前问题**

若某分钟行直接缺失于 index，不会出现 NaN，partial day 仍可能过。

**必须怎么改**

用 SessionGrid observed slot coverage/full-session policy。

**必须新增/修改测试**

239/duplicate/stray second。

### R16-120｜P0｜CONFIRMED_CURRENT_DEFECT｜volume_clock 直接 index.normalize 分日

**位置**：`factor_engine/cleaned_operators/volume_clock.py`

**当前问题**

UTC/跨日市场 trade date 可能错。

**必须怎么改**

SessionContext.trade_date。

**必须新增/修改测试**

UTC A股/US。

### R16-121｜P0｜CONFIRMED_CURRENT_DEFECT｜volume_clock optional open 改变同一 canonical 定义

**位置**：`factor_engine/cleaned_operators/volume_clock.py`

**当前问题**

有 open 时用真实 session open；无 open 时用 first observed positive-activity price，可能是09:31 close。

**必须怎么改**

open required，或拆 with_session_open 与 first_observed_proxy 两 canonical。

**必须新增/修改测试**

两个定义 ID 不得相同。

### R16-122｜P1｜CONFIRMED_CURRENT_DEFECT｜volume_clock zero-activity price 用 exact float equality

**位置**：`factor_engine/cleaned_operators/volume_clock.py`

**当前问题**

经济上应比较同一个合法报价 tick，而不是机器浮点完全相等。

**必须怎么改**

使用 PriceEqualityPolicy/TickSize；无市场 metadata 时明确 stored-value exact policy。

**必须新增/修改测试**

sub-epsilon/tick boundary。

### R16-123｜P0｜CONFIRMED_CURRENT_DEFECT｜session_recovery metadata 没声明 EOD grain

**位置**：`factor_engine/cleaned_operators/session_recovery.py`

**当前问题**

doc说 minute→daily/session_close，machine metadata 没有。

**必须怎么改**

补 input minute/output daily/available_at session_close/same_session_usable false。

**必须新增/修改测试**

catalog test。

### R16-124｜P0｜CONFIRMED_CURRENT_DEFECT｜session_recovery calendar 只用于 timezone，没验完整 grid

**位置**：`factor_engine/cleaned_operators/session_recovery.py`

**当前问题**

index 直接少一行时 recovery loop 不知道缺 bar。

**必须怎么改**

official SessionGrid + physical horizon censor。

**必须新增/修改测试**

缺中间物理slot。

### R16-125｜P0｜CONFIRMED_CURRENT_DEFECT｜session_recovery event.reindex 静默修轴

**位置**：`factor_engine/cleaned_operators/session_recovery.py`

**当前问题**

event 与 x axis mismatch 被 reindex 而不是 fail。

**必须怎么改**

SameAxis exact。

**必须新增/修改测试**

date-shift/missing-symbol。

### R16-126｜P0｜CURRENT_REGRESSION｜session_recovery unknown event 仍被当无事件

**位置**：`factor_engine/cleaned_operators/session_recovery.py`

**当前问题**

event NaN 直接 continue。

**必须怎么改**

tri-state censor/fail policy，unknown != 0。

**必须新增/修改测试**

NaN event 与0 event输出不得相同。

### R16-127｜P1｜CURRENT_REGRESSION｜session_recovery 默认 min_events 仍=1

**位置**：`factor_engine/cleaned_operators/session_recovery.py`

**当前问题**

注释说单shock不稳定但 default=1。

**必须怎么改**

reviewed default>=3/5并同步 ParamSpec。

**必须新增/修改测试**

默认单shock NaN。

### R16-128｜P0｜CONFIRMED_CURRENT_DEFECT｜advanced_intraday timezone 仍猜未知=Asia/Shanghai

**位置**：`factor_engine/cleaned_operators/advanced_intraday.py::_infer_session_tz`

**当前问题**

与其他 minute family fail-closed policy 冲突。

**必须怎么改**

唯一 SessionContext resolver，未知不猜。

**必须新增/修改测试**

同 UTC input 全 intraday family一致。

### R16-129｜P0｜CONFIRMED_CURRENT_DEFECT｜Wasserstein 同数学核注册两个 public canonical

**位置**：`factor_engine/cleaned_operators/advanced_intraday.py`

**当前问题**

`intraday_wasserstein_pair_distance` 与 `baseline_scaled_wasserstein_distance` 同 `_pair_w1`。

**必须怎么改**

保留诚实 canonical，旧名 alias only。

**必须新增/修改测试**

one mining identity。

### R16-130｜P0｜CURRENT_REGRESSION｜intraday quantile PCA 可行参数仍没闭环

**位置**：`factor_engine/cleaned_operators/advanced_intraday.py`

**当前问题**

内部至少50历史 profile，但 public 仍允许 window<50；k 关系不完整。

**必须怎么改**

ParamSpec min/relations；无 guaranteed-NaN region。

**必须新增/修改测试**

parameter grid viability。

### R16-131｜P0｜CONFIRMED_CURRENT_DEFECT｜intraday quantile PCA 没完整 session gate

**位置**：`factor_engine/cleaned_operators/advanced_intraday.py`

**当前问题**

半天/缺bar仍可能形成 quantile profile。

**必须怎么改**

只接受 completed session 或显式 official-grid coverage。

**必须新增/修改测试**

partial session NaN。

### R16-132｜P1｜CONFIRMED_CURRENT_DEFECT｜PCA residual 不必要要求 rank>=k+1

**位置**：`factor_engine/cleaned_operators/advanced_intraday.py::_pca_resid_series`

**当前问题**

top-k 子空间 residual 只需 rank>=k；当前会拒绝恰好rank=k。

**必须怎么改**

score与residual分开 identifiability contract。

**必须新增/修改测试**

rank exactly k residual finite。

### R16-133｜P0｜CONFIRMED_CURRENT_DEFECT｜intraday_agg bar width 仍从 observed deltas 推断

**位置**：`factor_engine/cleaned_operators/microstructure/intraday_agg.py::_bar_width_minutes`

**当前问题**

系统性缺每另一根 bar 会被重新解释成2min数据。

**必须怎么改**

bar frequency 只能来自 DataContract/SessionCalendar。

**必须新增/修改测试**

系统性drop every other bar must fail coverage。

### R16-134｜P0｜CONFIRMED_CURRENT_DEFECT｜intraday_agg helper 直接 normalize calendar day

**位置**：`factor_engine/cleaned_operators/microstructure/intraday_agg.py::_daily_agg*`

**当前问题**

是否正确依赖 caller 是否先 session_local。

**必须怎么改**

helper 强制 SessionContext，内部统一 trade date。

**必须新增/修改测试**

UTC direct helper。

### R16-135｜P0｜CONFIRMED_CURRENT_DEFECT｜_daily_agg_two dropna 压缩物理时间轴

**位置**：`factor_engine/cleaned_operators/microstructure/intraday_agg.py`

**当前问题**

pair missing row 被删除，path/lag statistic 会跨 gap 重连。

**必须怎么改**

保留 official slot + validity mask；distribution-only 才允许 cohort drop。

**必须新增/修改测试**

gap metamorphic。

### R16-136｜P0｜CONFIRMED_CURRENT_DEFECT｜_daily_agg_three 只 drop a

**位置**：`factor_engine/cleaned_operators/microstructure/intraday_agg.py`

**当前问题**

b/c 缺失进入 kernel，nansum 易把 unknown 当0。

**必须怎么改**

MissingCohortContract ALL_REQUIRED/OPTIONAL。

**必须新增/修改测试**

b-only/c-only missing。

### R16-137｜P0｜CONFIRMED_CURRENT_DEFECT｜segment volume/amount share 用 nansum 把缺失当0

**位置**：`factor_engine/cleaned_operators/microstructure/intraday_agg.py`

**当前问题**

missing activity 与 zero activity 混淆。

**必须怎么改**

session coverage + unknown != zero。

**必须新增/修改测试**

NaN vs0 输出不同。

### R16-138｜P0｜CONFIRMED_CURRENT_DEFECT｜segment return 通过 finite 压缩跨 gap

**位置**：`factor_engine/cleaned_operators/microstructure/intraday_agg.py::_seg_return`

**当前问题**

只取有限第一/最后值，不保证官方segment边界。

**必须怎么改**

要求 first/last official slot；或另命名 observed-span return。

**必须新增/修改测试**

缺first/last/middle。

### R16-139｜P0｜CONFIRMED_CURRENT_DEFECT｜intraday segment 参数 choices 未机器声明

**位置**：`factor_engine/cleaned_operators/microstructure/intraday_agg.py`

**当前问题**

多数 op 只声明 session_tz，segment runtime dict lookup。

**必须怎么改**

segment enum ParamSpec，所有 scalar complete。

**必须新增/修改测试**

invalid segment binder fail。

### R16-140｜P0｜CONFIRMED_CURRENT_DEFECT｜Hill estimator 允许负 threshold 的经典定义域问题

**位置**：`factor_engine/cleaned_operators/extreme_tail.py::ts_hill_tail_index`

**当前问题**

经典 Hill 应在正 tail magnitude 上估，当前只拒 u==0，允许 u<0 后 log ratio。

**必须怎么改**

upper/lower 先映射严格正 tail magnitude；signed POT 另用GPD类。

**必须新增/修改测试**

全负/全正/lower mirror golden。

### R16-141｜P0｜CONFIRMED_CURRENT_DEFECT｜extreme_tail 整族 ParamSpec 覆盖不足

**位置**：`factor_engine/cleaned_operators/extreme_tail.py`

**当前问题**

window/side/tail_fraction/min_tail/q/run_length 等大量 runtime cast。

**必须怎么改**

补完整 spec/default/role/relations。

**必须新增/修改测试**

family scalar coverage。

### R16-142｜P1｜CONFIRMED_CURRENT_DEFECT｜price-level heuristic 扫未来数据决定 warning

**位置**：`factor_engine/cleaned_operators/extreme_tail.py::_reject_price_level`

**当前问题**

即使只 warning，strict warnings-as-errors 时未来数据可改变prefix调用成功与否。

**必须怎么改**

删除数值猜 SemanticType，只靠 typed FieldSpec。

**必须新增/修改测试**

future mutation 不改变 prefix execution/warning。

### R16-143｜P0｜CONFIRMED_CURRENT_DEFECT｜quantile regression beta 是 inclusive in-sample coefficient

**位置**：`factor_engine/cleaned_operators/extreme_tail.py::ts_quantile_regression_beta`

**当前问题**

窗口包含 current y，作为默认 alpha 会混入当前目标拟合。

**必须怎么改**

inclusive版 diagnostic；默认 mining 用 strictly-prior beta。

**必须新增/修改测试**

改 y_t 不影响 prior beta_t。

### R16-144｜P1｜CONFIRMED_CURRENT_DEFECT｜quantile regression extreme-q tail support 太低

**位置**：`factor_engine/cleaned_operators/extreme_tail.py`

**当前问题**

少数样本也可LP，q=.1/.9不稳定。

**必须怎么改**

`N*min(q,1-q)` minimum tail support + solver conditioning。

**必须新增/修改测试**

extreme-q floor。

### R16-145｜P0｜CURRENT_REGRESSION｜CompositionSchema 仍没有真正的 typed object

**位置**：`factor_engine/cleaned_operators/composition.py`

**当前问题**

代码仍用字段名→family map，和 doc 所说 explicit CompositionSchema 不一致。

**必须怎么改**

实现 typed schema(whole_id,part_ids,unit,closure_basis,positivity,vintage)，由AST/source注入。

**必须新增/修改测试**

schema mismatch compile fail。

### R16-146｜P0｜CURRENT_REGRESSION｜financial_statement 仍把非同一整体项目当 composition

**位置**：`factor_engine/cleaned_operators/composition.py::_COMPOSITION_SCHEMA_FIELDS`

**当前问题**

Revenue/NetIncome/Assets/Liabilities/Equity 仍同 family，但并非互斥同基准parts。

**必须怎么改**

删除 broad family，只允许真实分解：asset components/revenue mix/expense mix/holder shares等。

**必须新增/修改测试**

Revenue+Assets+Equity rejected。

### R16-147｜P0｜CONFIRMED_CURRENT_DEFECT｜宽 panel 上 DataFrame 猜 field identity 基本失效

**位置**：`factor_engine/cleaned_operators/composition.py::_part_field_name`

**当前问题**

columns通常是股票代码，因此真实 field concept 变 anonymous，schema gate绕过。

**必须怎么改**

PartId 来自 typed AST/SourceRef metadata。

**必须新增/修改测试**

wide revenue panel仍识别PartId。

### R16-148｜P1｜CONFIRMED_CURRENT_DEFECT｜unknown composition part 在无 schema 时仍 positional 放过

**位置**：`factor_engine/cleaned_operators/composition.py`

**当前问题**

default mining 可绕过 schema。

**必须怎么改**

positional compatibility只能 legacy/manual，不进default mining。

**必须新增/修改测试**

unknown parts default mining fail。

### R16-149｜P1｜CONFIRMED_CURRENT_DEFECT｜composition entropy 在严格正p上仍加EPS

**位置**：`factor_engine/cleaned_operators/composition.py`

**当前问题**

p>0 已由 contract 保证，log(p+EPS)无必要并轻微改数学定义。

**必须怎么改**

严格正p直接log(p)。

**必须新增/修改测试**

高动态范围reference parity。

### R16-150｜P0｜CONFIRMED_CURRENT_DEFECT｜GLR gap 后用短 contiguous N 改变 estimator identity

**位置**：`factor_engine/cleaned_operators/glr_change.py`

**当前问题**

window=W 但短N满足segment就输出，null penalty也随N变。

**必须怎么改**

production CONTIGUOUS_FULL_WINDOW；adaptive-N另canonical并输出effective_n。

**必须新增/修改测试**

gap后重积W前NaN。

### R16-151｜P1｜CONFIRMED_CURRENT_DEFECT｜GLR full_ss absolute EPS 非 scale-aware

**位置**：`factor_engine/cleaned_operators/glr_change.py`

**当前问题**

不同量纲可用性不同。

**必须怎么改**

relative machine-noise/scale-aware gate。

**必须新增/修改测试**

x 与1e6x可用性一致。

### R16-152｜P1｜CONFIRMED_CURRENT_DEFECT｜GLR cap/penalty hidden constants 未进入 definition hash

**位置**：`factor_engine/cleaned_operators/glr_change.py`

**当前问题**

40和penalty coefficient直接改变输出。

**必须怎么改**

EstimatorPolicy version化并进入contract/evidence hash。

**必须新增/修改测试**

改常量使旧evidence stale。

### R16-153｜P0｜CONFIRMED_CURRENT_DEFECT｜OHLC estimators gap 后短 run 仍输出

**位置**：`factor_engine/cleaned_operators/ohlc_spread.py`

**当前问题**

`trailing_contiguous_multi` 允许实际N<window。

**必须怎么改**

FULL_WINDOW 或 explicit adaptive-N variant。

**必须新增/修改测试**

gap full-window test。

### R16-154｜P1｜CONFIRMED_CURRENT_DEFECT｜EDGE 默认 min_valid_ratio=0 让2个pair代表长窗口

**位置**：`factor_engine/cleaned_operators/ohlc_spread.py`

**当前问题**

reference floor 2 pair 对production稳定性太低。

**必须怎么改**

保留 reference-exact 与 production-gated 两定义，生产设置校准coverage。

**必须新增/修改测试**

2/19 pairs production gated NaN。

### R16-155｜P0｜CONFIRMED_CURRENT_DEFECT｜Abdi logical signature 已删 correction 但底层 callable 仍公开

**位置**：`factor_engine/cleaned_operators/ohlc_spread.py`

**当前问题**

registry ABI 与 direct Python callable ABI 不一致。

**必须怎么改**

correction变private constant或compat wrapper。

**必须新增/修改测试**

inspect.signature parity。

### R16-156｜P0｜CONFIRMED_CURRENT_DEFECT｜Pastor-Stambaugh output unit 与 flow_scale 参数矛盾

**位置**：`factor_engine/cleaned_operators/ohlc_spread.py`

**当前问题**

flow_scale可变但unit永远per-million。

**必须怎么改**

最好source先normalize并移除scale；否则parameterized unit进入definition identity。

**必须新增/修改测试**

不同scale单位换算 parity。

### R16-157｜P0｜CONFIRMED_CURRENT_DEFECT｜cross_section_ext hidden rho threshold 改模型选择

**位置**：`factor_engine/cleaned_operators/cross_section_ext.py::_RHO_EPS`

**当前问题**

0.05决定升/降/双拟合路线但只藏常量。

**必须怎么改**

EstimatorPolicy version化或 non-searchable policy field。

**必须新增/修改测试**

改阈值使definition/evidence变化。

### R16-158｜P0｜CONFIRMED_CURRENT_DEFECT｜descriptive isotonic residual 仍可能普通 ALPHA terminal

**位置**：`factor_engine/cleaned_operators/cross_section_ext.py::cs_isotonic_residual`

**当前问题**

用当日y选方向/双SSE，属于 descriptive neutralization。

**必须怎么改**

role=DIAGNOSTIC/RECIPE_INTERNAL；lagged-direction版本才default mining。

**必须新增/修改测试**

manifest role test。

### R16-159｜P1｜CONFIRMED_CURRENT_DEFECT｜lagged isotonic direction 把多日全股票直接ravel

**位置**：`factor_engine/cleaned_operators/cross_section_ext.py`

**当前问题**

大截面日期权重更高且混合横截面水平与时间变化。

**必须怎么改**

逐日Spearman再跨日聚合；或诚实名为 pooled variant。

**必须新增/修改测试**

股票数变化不应改变同日关系方向。

### R16-160｜P0｜CONFIRMED_CURRENT_DEFECT｜cross_section_ext 多输入直接 np.stack 缺本地axis证明

**位置**：`factor_engine/cleaned_operators/cross_section_ext.py`

**当前问题**

若中央binder漏一次就位置错配。

**必须怎么改**

SameAxis 是 logical call boundary mandatory contract。

**必须新增/修改测试**

column permutation/missing symbol reject。

### R16-161｜P0｜CONFIRMED_CURRENT_DEFECT｜activity-clock window semantics 文档与真实实现不同

**位置**：`factor_engine/cleaned_operators/activity_clock.py`

**当前问题**

标 finite_observations，实际是最近N物理row里再删missing。

**必须怎么改**

改语义名或真正实现last-N-finite；不能混。

**必须新增/修改测试**

长gap cardinality。

### R16-162｜P0｜CONFIRMED_CURRENT_DEFECT｜activity-clock ParamSpec 只声明部分scalar

**位置**：`factor_engine/cleaned_operators/activity_clock.py`

**当前问题**

budget/max_lookback/window仍缺。

**必须怎么改**

补完整spec/relations并去kernel clamp。

**必须新增/修改测试**

scalar coverage。

### R16-163｜P1｜CONFIRMED_CURRENT_DEFECT｜activity-clock Polars backend metadata 丢逻辑contract

**位置**：`factor_engine/cleaned_operators/activity_clock.py`

**当前问题**

Polars metadata params=[]，实际delegates pandas。

**必须怎么改**

logical metadata canonical-level；backend只描述capability并标delegated。

**必须新增/修改测试**

选择backend不改变logical catalog。

### R16-164｜P0｜MANDATORY_FULL_AUDIT｜Final registry set equality

**位置**：`FinalRegistrySnapshot / 全 FactorEngine`

**当前问题**

每个 freeze 后 canonical 必须有且仅有一条 per-canonical audit record；禁止用固定数量。

**必须怎么改**

将本规则实现成机器 detector；不得依赖人工 canonical 白名单。每个适用 canonical 都必须得到明确 outcome；不适用需给 machine reason。

**必须新增/修改测试**

至少一个正例 + 一个 mutation 负例；最终 detector coverage set 与 applicable canonical set 完全相等。

### R16-165｜P0｜MANDATORY_FULL_AUDIT｜Implementation-file coverage

**位置**：`FinalRegistrySnapshot / 全 FactorEngine`

**当前问题**

每 canonical 的全部实现文件/helper dependency 必须进入 source scan；private `_*.py` 也算。

**必须怎么改**

将本规则实现成机器 detector；不得依赖人工 canonical 白名单。每个适用 canonical 都必须得到明确 outcome；不适用需给 machine reason。

**必须新增/修改测试**

至少一个正例 + 一个 mutation 负例；最终 detector coverage set 与 applicable canonical set 完全相等。

### R16-166｜P0｜MANDATORY_FULL_AUDIT｜ResolvedSignature coverage

**位置**：`FinalRegistrySnapshot / 全 FactorEngine`

**当前问题**

panel/scalar/optional/keyword-only/variadic 全 machine-resolved，不准名字猜。

**必须怎么改**

将本规则实现成机器 detector；不得依赖人工 canonical 白名单。每个适用 canonical 都必须得到明确 outcome；不适用需给 machine reason。

**必须新增/修改测试**

至少一个正例 + 一个 mutation 负例；最终 detector coverage set 与 applicable canonical set 完全相等。

### R16-167｜P0｜MANDATORY_FULL_AUDIT｜ParamSpec scalar coverage

**位置**：`FinalRegistrySnapshot / 全 FactorEngine`

**当前问题**

mineable canonical 每个 scalar 都有 dtype/default/domain/searchable/role。

**必须怎么改**

将本规则实现成机器 detector；不得依赖人工 canonical 白名单。每个适用 canonical 都必须得到明确 outcome；不适用需给 machine reason。

**必须新增/修改测试**

至少一个正例 + 一个 mutation 负例；最终 detector coverage set 与 applicable canonical set 完全相等。

### R16-168｜P0｜MANDATORY_FULL_AUDIT｜Signature default parity

**位置**：`FinalRegistrySnapshot / 全 FactorEngine`

**当前问题**

callable default、ParamSpec default、manifest default 三者一致。

**必须怎么改**

将本规则实现成机器 detector；不得依赖人工 canonical 白名单。每个适用 canonical 都必须得到明确 outcome；不适用需给 machine reason。

**必须新增/修改测试**

至少一个正例 + 一个 mutation 负例；最终 detector coverage set 与 applicable canonical set 完全相等。

### R16-169｜P0｜MANDATORY_FULL_AUDIT｜Parameter type fuzz

**位置**：`FinalRegistrySnapshot / 全 FactorEngine`

**当前问题**

对每 scalar 测 bool/int/float/string/None/NaN/Inf/np scalar。

**必须怎么改**

将本规则实现成机器 detector；不得依赖人工 canonical 白名单。每个适用 canonical 都必须得到明确 outcome；不适用需给 machine reason。

**必须新增/修改测试**

至少一个正例 + 一个 mutation 负例；最终 detector coverage set 与 applicable canonical set 完全相等。

### R16-170｜P0｜MANDATORY_FULL_AUDIT｜No silent clamp/cast

**位置**：`FinalRegistrySnapshot / 全 FactorEngine`

**当前问题**

所有 kernel 的 int/float/bool/max/min 参数修复模式逐 AST hit disposition。

**必须怎么改**

将本规则实现成机器 detector；不得依赖人工 canonical 白名单。每个适用 canonical 都必须得到明确 outcome；不适用需给 machine reason。

**必须新增/修改测试**

至少一个正例 + 一个 mutation 负例；最终 detector coverage set 与 applicable canonical set 完全相等。

### R16-171｜P0｜MANDATORY_FULL_AUDIT｜Parameter injectivity

**位置**：`FinalRegistrySnapshot / 全 FactorEngine`

**当前问题**

每 searchable param 至少两个合法值在适用fixture上产生不同定义行为。

**必须怎么改**

将本规则实现成机器 detector；不得依赖人工 canonical 白名单。每个适用 canonical 都必须得到明确 outcome；不适用需给 machine reason。

**必须新增/修改测试**

至少一个正例 + 一个 mutation 负例；最终 detector coverage set 与 applicable canonical set 完全相等。

### R16-172｜P0｜MANDATORY_FULL_AUDIT｜Relational feasibility

**位置**：`FinalRegistrySnapshot / 全 FactorEngine`

**当前问题**

fast<slow、lag<window、min_periods<=window、k<=rank、history relations全机器化。

**必须怎么改**

将本规则实现成机器 detector；不得依赖人工 canonical 白名单。每个适用 canonical 都必须得到明确 outcome；不适用需给 machine reason。

**必须新增/修改测试**

至少一个正例 + 一个 mutation 负例；最终 detector coverage set 与 applicable canonical set 完全相等。

### R16-173｜P0｜MANDATORY_FULL_AUDIT｜Inactive-param canonicalization

**位置**：`FinalRegistrySnapshot / 全 FactorEngine`

**当前问题**

inactive 参数不得进入 AST/hash/cache identity。

**必须怎么改**

将本规则实现成机器 detector；不得依赖人工 canonical 白名单。每个适用 canonical 都必须得到明确 outcome；不适用需给 machine reason。

**必须新增/修改测试**

至少一个正例 + 一个 mutation 负例；最终 detector coverage set 与 applicable canonical set 完全相等。

### R16-174｜P0｜MANDATORY_FULL_AUDIT｜SemanticType coverage

**位置**：`FinalRegistrySnapshot / 全 FactorEngine`

**当前问题**

Price/Return/Volume/Amount/PositiveLevel/EventBool/ConditionBool等全 typed。

**必须怎么改**

将本规则实现成机器 detector；不得依赖人工 canonical 白名单。每个适用 canonical 都必须得到明确 outcome；不适用需给 machine reason。

**必须新增/修改测试**

至少一个正例 + 一个 mutation 负例；最终 detector coverage set 与 applicable canonical set 完全相等。

### R16-175｜P0｜MANDATORY_FULL_AUDIT｜FieldConcept 与 SemanticType 分离

**位置**：`FinalRegistrySnapshot / 全 FactorEngine`

**当前问题**

manifest/admission 不得把字段名当语义类型。

**必须怎么改**

将本规则实现成机器 detector；不得依赖人工 canonical 白名单。每个适用 canonical 都必须得到明确 outcome；不适用需给 machine reason。

**必须新增/修改测试**

至少一个正例 + 一个 mutation 负例；最终 detector coverage set 与 applicable canonical set 完全相等。

### R16-176｜P0｜MANDATORY_FULL_AUDIT｜Unit algebra coverage

**位置**：`FinalRegistrySnapshot / 全 FactorEngine`

**当前问题**

输出单位能从输入/参数推导；不得 generic level/ratio 掩盖。

**必须怎么改**

将本规则实现成机器 detector；不得依赖人工 canonical 白名单。每个适用 canonical 都必须得到明确 outcome；不适用需给 machine reason。

**必须新增/修改测试**

至少一个正例 + 一个 mutation 负例；最终 detector coverage set 与 applicable canonical set 完全相等。

### R16-177｜P1｜MANDATORY_FULL_AUDIT｜Output range contract

**位置**：`FinalRegistrySnapshot / 全 FactorEngine`

**当前问题**

probability/correlation/normalized entropy/state等 machine range。

**必须怎么改**

将本规则实现成机器 detector；不得依赖人工 canonical 白名单。每个适用 canonical 都必须得到明确 outcome；不适用需给 machine reason。

**必须新增/修改测试**

至少一个正例 + 一个 mutation 负例；最终 detector coverage set 与 applicable canonical set 完全相等。

### R16-178｜P0｜MANDATORY_FULL_AUDIT｜Scale metamorphic

**位置**：`FinalRegistrySnapshot / 全 FactorEngine`

**当前问题**

按 contract 测 x、10x、x+c、-x。

**必须怎么改**

将本规则实现成机器 detector；不得依赖人工 canonical 白名单。每个适用 canonical 都必须得到明确 outcome；不适用需给 machine reason。

**必须新增/修改测试**

至少一个正例 + 一个 mutation 负例；最终 detector coverage set 与 applicable canonical set 完全相等。

### R16-179｜P0｜MANDATORY_FULL_AUDIT｜SameAxis exactness

**位置**：`FinalRegistrySnapshot / 全 FactorEngine`

**当前问题**

所有 multi-input 测 date shift、column permutation、missing/extra symbol、duplicate time。

**必须怎么改**

将本规则实现成机器 detector；不得依赖人工 canonical 白名单。每个适用 canonical 都必须得到明确 outcome；不适用需给 machine reason。

**必须新增/修改测试**

至少一个正例 + 一个 mutation 负例；最终 detector coverage set 与 applicable canonical set 完全相等。

### R16-180｜P0｜MANDATORY_FULL_AUDIT｜Broadcast whitelist

**位置**：`FinalRegistrySnapshot / 全 FactorEngine`

**当前问题**

只有显式 BroadcastSpec 可跨grain/axis broadcast。

**必须怎么改**

将本规则实现成机器 detector；不得依赖人工 canonical 白名单。每个适用 canonical 都必须得到明确 outcome；不适用需给 machine reason。

**必须新增/修改测试**

至少一个正例 + 一个 mutation 负例；最终 detector coverage set 与 applicable canonical set 完全相等。

### R16-181｜P0｜MANDATORY_FULL_AUDIT｜Current-row semantics

**位置**：`FinalRegistrySnapshot / 全 FactorEngine`

**当前问题**

inclusive/strict-prior/session-close 明确并行为测试。

**必须怎么改**

将本规则实现成机器 detector；不得依赖人工 canonical 白名单。每个适用 canonical 都必须得到明确 outcome；不适用需给 machine reason。

**必须新增/修改测试**

至少一个正例 + 一个 mutation 负例；最终 detector coverage set 与 applicable canonical set 完全相等。

### R16-182｜P0｜MANDATORY_FULL_AUDIT｜Future randomization

**位置**：`FinalRegistrySnapshot / 全 FactorEngine`

**当前问题**

改 t+1: 数据，output<=t 完全不变。

**必须怎么改**

将本规则实现成机器 detector；不得依赖人工 canonical 白名单。每个适用 canonical 都必须得到明确 outcome；不适用需给 machine reason。

**必须新增/修改测试**

至少一个正例 + 一个 mutation 负例；最终 detector coverage set 与 applicable canonical set 完全相等。

### R16-183｜P0｜MANDATORY_FULL_AUDIT｜Current missing

**位置**：`FinalRegistrySnapshot / 全 FactorEngine`

**当前问题**

CurrentRequired 输入t缺失 => output t缺失。

**必须怎么改**

将本规则实现成机器 detector；不得依赖人工 canonical 白名单。每个适用 canonical 都必须得到明确 outcome；不适用需给 machine reason。

**必须新增/修改测试**

至少一个正例 + 一个 mutation 负例；最终 detector coverage set 与 applicable canonical set 完全相等。

### R16-184｜P0｜MANDATORY_FULL_AUDIT｜Gap topology

**位置**：`FinalRegistrySnapshot / 全 FactorEngine`

**当前问题**

lag/path/event/embedding/spectral 不跨 unknown gap 重连。

**必须怎么改**

将本规则实现成机器 detector；不得依赖人工 canonical 白名单。每个适用 canonical 都必须得到明确 outcome；不适用需给 machine reason。

**必须新增/修改测试**

至少一个正例 + 一个 mutation 负例；最终 detector coverage set 与 applicable canonical set 完全相等。

### R16-185｜P0｜MANDATORY_FULL_AUDIT｜Inf topology

**位置**：`FinalRegistrySnapshot / 全 FactorEngine`

**当前问题**

±Inf 与 NaN/invalid 的处理全 backend 一致。

**必须怎么改**

将本规则实现成机器 detector；不得依赖人工 canonical 白名单。每个适用 canonical 都必须得到明确 outcome；不适用需给 machine reason。

**必须新增/修改测试**

至少一个正例 + 一个 mutation 负例；最终 detector coverage set 与 applicable canonical set 完全相等。

### R16-186｜P0｜MANDATORY_FULL_AUDIT｜Window cardinality

**位置**：`FinalRegistrySnapshot / 全 FactorEngine`

**当前问题**

FULL_WINDOW exactly W；prior W exactly W prior observations。

**必须怎么改**

将本规则实现成机器 detector；不得依赖人工 canonical 白名单。每个适用 canonical 都必须得到明确 outcome；不适用需给 machine reason。

**必须新增/修改测试**

至少一个正例 + 一个 mutation 负例；最终 detector coverage set 与 applicable canonical set 完全相等。

### R16-187｜P1｜MANDATORY_FULL_AUDIT｜Effective-N audit

**位置**：`FinalRegistrySnapshot / 全 FactorEngine`

**当前问题**

每统计算子可诊断 effective_n/pairs/tail/event/neighbor count。

**必须怎么改**

将本规则实现成机器 detector；不得依赖人工 canonical 白名单。每个适用 canonical 都必须得到明确 outcome；不适用需给 machine reason。

**必须新增/修改测试**

至少一个正例 + 一个 mutation 负例；最终 detector coverage set 与 applicable canonical set 完全相等。

### R16-188｜P0｜MANDATORY_FULL_AUDIT｜Tie invariance

**位置**：`FinalRegistrySnapshot / 全 FactorEngine`

**当前问题**

列顺序、等值排序不能改变结果，除非定义含稳定InstrumentId secondary key。

**必须怎么改**

将本规则实现成机器 detector；不得依赖人工 canonical 白名单。每个适用 canonical 都必须得到明确 outcome；不适用需给 machine reason。

**必须新增/修改测试**

至少一个正例 + 一个 mutation 负例；最终 detector coverage set 与 applicable canonical set 完全相等。

### R16-189｜P0｜MANDATORY_FULL_AUDIT｜PIT UniverseMask

**位置**：`FinalRegistrySnapshot / 全 FactorEngine`

**当前问题**

所有 CS/group 只在当日可知 universe/membership 上算。

**必须怎么改**

将本规则实现成机器 detector；不得依赖人工 canonical 白名单。每个适用 canonical 都必须得到明确 outcome；不适用需给 machine reason。

**必须新增/修改测试**

至少一个正例 + 一个 mutation 负例；最终 detector coverage set 与 applicable canonical set 完全相等。

### R16-190｜P0｜MANDATORY_FULL_AUDIT｜Group migration

**位置**：`FinalRegistrySnapshot / 全 FactorEngine`

**当前问题**

current-members retrospective 与 historical membership 必须不同 canonical/contract。

**必须怎么改**

将本规则实现成机器 detector；不得依赖人工 canonical 白名单。每个适用 canonical 都必须得到明确 outcome；不适用需给 machine reason。

**必须新增/修改测试**

至少一个正例 + 一个 mutation 负例；最终 detector coverage set 与 applicable canonical set 完全相等。

### R16-191｜P0｜MANDATORY_FULL_AUDIT｜Global/group state role

**位置**：`FinalRegistrySnapshot / 全 FactorEngine`

**当前问题**

广播相同值的 state 不得作为 stock-alpha terminal。

**必须怎么改**

将本规则实现成机器 detector；不得依赖人工 canonical 白名单。每个适用 canonical 都必须得到明确 outcome；不适用需给 machine reason。

**必须新增/修改测试**

至少一个正例 + 一个 mutation 负例；最终 detector coverage set 与 applicable canonical set 完全相等。

### R16-192｜P0｜MANDATORY_FULL_AUDIT｜Strict EventBool

**位置**：`FinalRegistrySnapshot / 全 FactorEngine`

**当前问题**

只允许0/1/NaN；-1/.2/2/Inf非法。

**必须怎么改**

将本规则实现成机器 detector；不得依赖人工 canonical 白名单。每个适用 canonical 都必须得到明确 outcome；不适用需给 machine reason。

**必须新增/修改测试**

至少一个正例 + 一个 mutation 负例；最终 detector coverage set 与 applicable canonical set 完全相等。

### R16-193｜P0｜MANDATORY_FULL_AUDIT｜Strict ConditionBool

**位置**：`FinalRegistrySnapshot / 全 FactorEngine`

**当前问题**

同上，Pandas/Polars/SQL一致。

**必须怎么改**

将本规则实现成机器 detector；不得依赖人工 canonical 白名单。每个适用 canonical 都必须得到明确 outcome；不适用需给 machine reason。

**必须新增/修改测试**

至少一个正例 + 一个 mutation 负例；最终 detector coverage set 与 applicable canonical set 完全相等。

### R16-194｜P0｜MANDATORY_FULL_AUDIT｜Session timezone

**位置**：`FinalRegistrySnapshot / 全 FactorEngine`

**当前问题**

minute operator 一律 SessionContext，不得自行猜 UTC->A股。

**必须怎么改**

将本规则实现成机器 detector；不得依赖人工 canonical 白名单。每个适用 canonical 都必须得到明确 outcome；不适用需给 machine reason。

**必须新增/修改测试**

至少一个正例 + 一个 mutation 负例；最终 detector coverage set 与 applicable canonical set 完全相等。

### R16-195｜P0｜MANDATORY_FULL_AUDIT｜A股 official grid

**位置**：`FinalRegistrySnapshot / 全 FactorEngine`

**当前问题**

240/239/241/duplicate/stray-second/lunch/suspension全部覆盖。

**必须怎么改**

将本规则实现成机器 detector；不得依赖人工 canonical 白名单。每个适用 canonical 都必须得到明确 outcome；不适用需给 machine reason。

**必须新增/修改测试**

至少一个正例 + 一个 mutation 负例；最终 detector coverage set 与 applicable canonical set 完全相等。

### R16-196｜P0｜MANDATORY_FULL_AUDIT｜Minute→daily availability

**位置**：`FinalRegistrySnapshot / 全 FactorEngine`

**当前问题**

EOD output session_close，盘中不得使用。

**必须怎么改**

将本规则实现成机器 detector；不得依赖人工 canonical 白名单。每个适用 canonical 都必须得到明确 outcome；不适用需给 machine reason。

**必须新增/修改测试**

至少一个正例 + 一个 mutation 负例；最终 detector coverage set 与 applicable canonical set 完全相等。

### R16-197｜P0｜MANDATORY_FULL_AUDIT｜Partial-day policy

**位置**：`FinalRegistrySnapshot / 全 FactorEngine`

**当前问题**

需要完整session的算子半天/早收盘/缺slot按calendar policy处理。

**必须怎么改**

将本规则实现成机器 detector；不得依赖人工 canonical 白名单。每个适用 canonical 都必须得到明确 outcome；不适用需给 machine reason。

**必须新增/修改测试**

至少一个正例 + 一个 mutation 负例；最终 detector coverage set 与 applicable canonical set 完全相等。

### R16-198｜P0｜MANDATORY_FULL_AUDIT｜Daily→minute PIT broadcast

**位置**：`FinalRegistrySnapshot / 全 FactorEngine`

**当前问题**

limit price/benchmark/calendar field用trade-date typed broadcast。

**必须怎么改**

将本规则实现成机器 detector；不得依赖人工 canonical 白名单。每个适用 canonical 都必须得到明确 outcome；不适用需给 machine reason。

**必须新增/修改测试**

至少一个正例 + 一个 mutation 负例；最终 detector coverage set 与 applicable canonical set 完全相等。

### R16-199｜P0｜MANDATORY_FULL_AUDIT｜Fundamental announcement/effective PIT

**位置**：`FinalRegistrySnapshot / 全 FactorEngine`

**当前问题**

late filing、不同period、公告日/生效日测试。

**必须怎么改**

将本规则实现成机器 detector；不得依赖人工 canonical 白名单。每个适用 canonical 都必须得到明确 outcome；不适用需给 machine reason。

**必须新增/修改测试**

至少一个正例 + 一个 mutation 负例；最终 detector coverage set 与 applicable canonical set 完全相等。

### R16-200｜P0｜MANDATORY_FULL_AUDIT｜Fundamental flow semantics

**位置**：`FinalRegistrySnapshot / 全 FactorEngine`

**当前问题**

single-quarter/YTD/TTM/annual由source semantic决定，不让搜索器乱切。

**必须怎么改**

将本规则实现成机器 detector；不得依赖人工 canonical 白名单。每个适用 canonical 都必须得到明确 outcome；不适用需给 machine reason。

**必须新增/修改测试**

至少一个正例 + 一个 mutation 负例；最终 detector coverage set 与 applicable canonical set 完全相等。

### R16-201｜P0｜MANDATORY_FULL_AUDIT｜Source concept capability

**位置**：`FinalRegistrySnapshot / 全 FactorEngine`

**当前问题**

source_id存在但required concept缺失必须admission失败。

**必须怎么改**

将本规则实现成机器 detector；不得依赖人工 canonical 白名单。每个适用 canonical 都必须得到明确 outcome；不适用需给 machine reason。

**必须新增/修改测试**

至少一个正例 + 一个 mutation 负例；最终 detector coverage set 与 applicable canonical set 完全相等。

### R16-202｜P0｜MANDATORY_FULL_AUDIT｜Revision vintage capability

**位置**：`FinalRegistrySnapshot / 全 FactorEngine`

**当前问题**

无历史vintage绝不允许revision factor默认挖。

**必须怎么改**

将本规则实现成机器 detector；不得依赖人工 canonical 白名单。每个适用 canonical 都必须得到明确 outcome；不适用需给 machine reason。

**必须新增/修改测试**

至少一个正例 + 一个 mutation 负例；最终 detector coverage set 与 applicable canonical set 完全相等。

### R16-203｜P0｜MANDATORY_FULL_AUDIT｜Prior vs inclusive regression

**位置**：`FinalRegistrySnapshot / 全 FactorEngine`

**当前问题**

inclusive diagnostics与strict-prior predictive不同role。

**必须怎么改**

将本规则实现成机器 detector；不得依赖人工 canonical 白名单。每个适用 canonical 都必须得到明确 outcome；不适用需给 machine reason。

**必须新增/修改测试**

至少一个正例 + 一个 mutation 负例；最终 detector coverage set 与 applicable canonical set 完全相等。

### R16-204｜P0｜MANDATORY_FULL_AUDIT｜Regression DOF

**位置**：`FinalRegistrySnapshot / 全 FactorEngine`

**当前问题**

N相对p足够；quantile/expectile另有tail support。

**必须怎么改**

将本规则实现成机器 detector；不得依赖人工 canonical 白名单。每个适用 canonical 都必须得到明确 outcome；不适用需给 machine reason。

**必须新增/修改测试**

至少一个正例 + 一个 mutation 负例；最终 detector coverage set 与 applicable canonical set 完全相等。

### R16-205｜P0｜MANDATORY_FULL_AUDIT｜Optimizer convergence

**位置**：`FinalRegistrySnapshot / 全 FactorEngine`

**当前问题**

LP/SVD/eig/GARCH/Kalman等失败/退化不能输出伪finite。

**必须怎么改**

将本规则实现成机器 detector；不得依赖人工 canonical 白名单。每个适用 canonical 都必须得到明确 outcome；不适用需给 machine reason。

**必须新增/修改测试**

至少一个正例 + 一个 mutation 负例；最终 detector coverage set 与 applicable canonical set 完全相等。

### R16-206｜P1｜MANDATORY_FULL_AUDIT｜Numerical policy hash

**位置**：`FinalRegistrySnapshot / 全 FactorEngine`

**当前问题**

rcond/tolerance/eigen-gap/ridge jitter进入definition/evidence hash。

**必须怎么改**

将本规则实现成机器 detector；不得依赖人工 canonical 白名单。每个适用 canonical 都必须得到明确 outcome；不适用需给 machine reason。

**必须新增/修改测试**

至少一个正例 + 一个 mutation 负例；最终 detector coverage set 与 applicable canonical set 完全相等。

### R16-207｜P0｜MANDATORY_FULL_AUDIT｜EPS audit

**位置**：`FinalRegistrySnapshot / 全 FactorEngine`

**当前问题**

EPS只能数值稳定，不能把0/0、zero-scale、undefined统计量变finite。

**必须怎么改**

将本规则实现成机器 detector；不得依赖人工 canonical 白名单。每个适用 canonical 都必须得到明确 outcome；不适用需给 machine reason。

**必须新增/修改测试**

至少一个正例 + 一个 mutation 负例；最终 detector coverage set 与 applicable canonical set 完全相等。

### R16-208｜P0｜MANDATORY_FULL_AUDIT｜Zero-denominator semantics

**位置**：`FinalRegistrySnapshot / 全 FactorEngine`

**当前问题**

zero variance/scale/amplitude按数学定义NaN或0，不能通用处理。

**必须怎么改**

将本规则实现成机器 detector；不得依赖人工 canonical 白名单。每个适用 canonical 都必须得到明确 outcome；不适用需给 machine reason。

**必须新增/修改测试**

至少一个正例 + 一个 mutation 负例；最终 detector coverage set 与 applicable canonical set 完全相等。

### R16-209｜P1｜MANDATORY_FULL_AUDIT｜Cost model honesty

**位置**：`FinalRegistrySnapshot / 全 FactorEngine`

**当前问题**

sort/SVD/LP/KNN O(N²)/topology不允许cost:1。

**必须怎么改**

将本规则实现成机器 detector；不得依赖人工 canonical 白名单。每个适用 canonical 都必须得到明确 outcome；不适用需给 machine reason。

**必须新增/修改测试**

至少一个正例 + 一个 mutation 负例；最终 detector coverage set 与 applicable canonical set 完全相等。

### R16-210｜P1｜MANDATORY_FULL_AUDIT｜Shared-kernel CSE

**位置**：`FinalRegistrySnapshot / 全 FactorEngine`

**当前问题**

多output昂贵kernel一次计算或真实计费。

**必须怎么改**

将本规则实现成机器 detector；不得依赖人工 canonical 白名单。每个适用 canonical 都必须得到明确 outcome；不适用需给 machine reason。

**必须新增/修改测试**

至少一个正例 + 一个 mutation 负例；最终 detector coverage set 与 applicable canonical set 完全相等。

### R16-211｜P0｜MANDATORY_FULL_AUDIT｜Native vs delegated backend

**位置**：`FinalRegistrySnapshot / 全 FactorEngine`

**当前问题**

pandas bridge 不冒充独立native backend certification。

**必须怎么改**

将本规则实现成机器 detector；不得依赖人工 canonical 白名单。每个适用 canonical 都必须得到明确 outcome；不适用需给 machine reason。

**必须新增/修改测试**

至少一个正例 + 一个 mutation 负例；最终 detector coverage set 与 applicable canonical set 完全相等。

### R16-212｜P0｜MANDATORY_FULL_AUDIT｜Backend invalid-input parity

**位置**：`FinalRegistrySnapshot / 全 FactorEngine`

**当前问题**

错误类型/错误码/axis/domain semantics也要parity。

**必须怎么改**

将本规则实现成机器 detector；不得依赖人工 canonical 白名单。每个适用 canonical 都必须得到明确 outcome；不适用需给 machine reason。

**必须新增/修改测试**

至少一个正例 + 一个 mutation 负例；最终 detector coverage set 与 applicable canonical set 完全相等。

### R16-213｜P0｜MANDATORY_FULL_AUDIT｜SQL prior/tie parity

**位置**：`FinalRegistrySnapshot / 全 FactorEngine`

**当前问题**

SQL window frame、tie rule、current contamination逐canonical验证。

**必须怎么改**

将本规则实现成机器 detector；不得依赖人工 canonical 白名单。每个适用 canonical 都必须得到明确 outcome；不适用需给 machine reason。

**必须新增/修改测试**

至少一个正例 + 一个 mutation 负例；最终 detector coverage set 与 applicable canonical set 完全相等。

### R16-214｜P0｜MANDATORY_FULL_AUDIT｜Polars missing/bool/axis parity

**位置**：`FinalRegistrySnapshot / 全 FactorEngine`

**当前问题**

不能只normal numeric allclose。

**必须怎么改**

将本规则实现成机器 detector；不得依赖人工 canonical 白名单。每个适用 canonical 都必须得到明确 outcome；不适用需给 machine reason。

**必须新增/修改测试**

至少一个正例 + 一个 mutation 负例；最终 detector coverage set 与 applicable canonical set 完全相等。

### R16-215｜P0｜MANDATORY_FULL_AUDIT｜Checkpoint parity

**位置**：`FinalRegistrySnapshot / 全 FactorEngine`

**当前问题**

stateful op 多split full-run==serialize/restore。

**必须怎么改**

将本规则实现成机器 detector；不得依赖人工 canonical 白名单。每个适用 canonical 都必须得到明确 outcome；不适用需给 machine reason。

**必须新增/修改测试**

至少一个正例 + 一个 mutation 负例；最终 detector coverage set 与 applicable canonical set 完全相等。

### R16-216｜P0｜MANDATORY_FULL_AUDIT｜Dynamic HistoryRequirement

**位置**：`FinalRegistrySnapshot / 全 FactorEngine`

**当前问题**

history从bound params生成，不靠central stale name list。

**必须怎么改**

将本规则实现成机器 detector；不得依赖人工 canonical 白名单。每个适用 canonical 都必须得到明确 outcome；不适用需给 machine reason。

**必须新增/修改测试**

至少一个正例 + 一个 mutation 负例；最终 detector coverage set 与 applicable canonical set 完全相等。

### R16-217｜P0｜MANDATORY_FULL_AUDIT｜Event/session history kind

**位置**：`FinalRegistrySnapshot / 全 FactorEngine`

**当前问题**

event-count/session-count绝不按普通row count换算。

**必须怎么改**

将本规则实现成机器 detector；不得依赖人工 canonical 白名单。每个适用 canonical 都必须得到明确 outcome；不适用需给 machine reason。

**必须新增/修改测试**

至少一个正例 + 一个 mutation 负例；最终 detector coverage set 与 applicable canonical set 完全相等。

### R16-218｜P0｜MANDATORY_FULL_AUDIT｜Full-history lane honesty

**位置**：`FinalRegistrySnapshot / 全 FactorEngine`

**当前问题**

无法checkpoint的recursive明确full-replay/high-cost，不能标stateless。

**必须怎么改**

将本规则实现成机器 detector；不得依赖人工 canonical 白名单。每个适用 canonical 都必须得到明确 outcome；不适用需给 machine reason。

**必须新增/修改测试**

至少一个正例 + 一个 mutation 负例；最终 detector coverage set 与 applicable canonical set 完全相等。

### R16-219｜P0｜MANDATORY_FULL_AUDIT｜Recursive AST source scan

**位置**：`FinalRegistrySnapshot / 全 FactorEngine`

**当前问题**

用AST而不是regex作为唯一scanner，覆盖私有helper。

**必须怎么改**

将本规则实现成机器 detector；不得依赖人工 canonical 白名单。每个适用 canonical 都必须得到明确 outcome；不适用需给 machine reason。

**必须新增/修改测试**

至少一个正例 + 一个 mutation 负例；最终 detector coverage set 与 applicable canonical set 完全相等。

### R16-220｜P0｜MANDATORY_FULL_AUDIT｜Detector five-state outcome

**位置**：`FinalRegistrySnapshot / 全 FactorEngine`

**当前问题**

PASS/FAIL/N/A/NOT_RUN/AUDIT_ERROR恰好一个。

**必须怎么改**

将本规则实现成机器 detector；不得依赖人工 canonical 白名单。每个适用 canonical 都必须得到明确 outcome；不适用需给 machine reason。

**必须新增/修改测试**

至少一个正例 + 一个 mutation 负例；最终 detector coverage set 与 applicable canonical set 完全相等。

### R16-221｜P0｜MANDATORY_FULL_AUDIT｜Required detector coverage

**位置**：`FinalRegistrySnapshot / 全 FactorEngine`

**当前问题**

required detector NOT_RUN/AUDIT_ERROR 直接阻断。

**必须怎么改**

将本规则实现成机器 detector；不得依赖人工 canonical 白名单。每个适用 canonical 都必须得到明确 outcome；不适用需给 machine reason。

**必须新增/修改测试**

至少一个正例 + 一个 mutation 负例；最终 detector coverage set 与 applicable canonical set 完全相等。

### R16-222｜P0｜MANDATORY_FULL_AUDIT｜Golden reference

**位置**：`FinalRegistrySnapshot / 全 FactorEngine`

**当前问题**

标准命名指标/统计量对论文/官方实现/独立数学reference。

**必须怎么改**

将本规则实现成机器 detector；不得依赖人工 canonical 白名单。每个适用 canonical 都必须得到明确 outcome；不适用需给 machine reason。

**必须新增/修改测试**

至少一个正例 + 一个 mutation 负例；最终 detector coverage set 与 applicable canonical set 完全相等。

### R16-223｜P0｜MANDATORY_FULL_AUDIT｜Null calibration

**位置**：`FinalRegistrySnapshot / 全 FactorEngine`

**当前问题**

MI/TE/HSIC/BDS/RQA/HVG/GLR/bicoherence/EVT等跑独立null。

**必须怎么改**

将本规则实现成机器 detector；不得依赖人工 canonical 白名单。每个适用 canonical 都必须得到明确 outcome；不适用需给 machine reason。

**必须新增/修改测试**

至少一个正例 + 一个 mutation 负例；最终 detector coverage set 与 applicable canonical set 完全相等。

### R16-224｜P1｜MANDATORY_FULL_AUDIT｜Synthetic regime battery

**位置**：`FinalRegistrySnapshot / 全 FactorEngine`

**当前问题**

linear/sine/white noise/random walk/AR1/GARCH/Markov/change point/motif/outlier/limit/suspension。

**必须怎么改**

将本规则实现成机器 detector；不得依赖人工 canonical 白名单。每个适用 canonical 都必须得到明确 outcome；不适用需给 machine reason。

**必须新增/修改测试**

至少一个正例 + 一个 mutation 负例；最终 detector coverage set 与 applicable canonical set 完全相等。

### R16-225｜P0｜MANDATORY_FULL_AUDIT｜Semantic duplicate proof

**位置**：`FinalRegistrySnapshot / 全 FactorEngine`

**当前问题**

exact/affine/rank/monotonic候选后必须行为证明，不靠名字/contract。

**必须怎么改**

将本规则实现成机器 detector；不得依赖人工 canonical 白名单。每个适用 canonical 都必须得到明确 outcome；不适用需给 machine reason。

**必须新增/修改测试**

至少一个正例 + 一个 mutation 负例；最终 detector coverage set 与 applicable canonical set 完全相等。

### R16-226｜P0｜MANDATORY_FULL_AUDIT｜Alias equivalence

**位置**：`FinalRegistrySnapshot / 全 FactorEngine`

**当前问题**

每alias edge有golden proof，self-loop/cycle为0。

**必须怎么改**

将本规则实现成机器 detector；不得依赖人工 canonical 白名单。每个适用 canonical 都必须得到明确 outcome；不适用需给 machine reason。

**必须新增/修改测试**

至少一个正例 + 一个 mutation 负例；最终 detector coverage set 与 applicable canonical set 完全相等。

### R16-227｜P0｜MANDATORY_FULL_AUDIT｜Manifest lineage

**位置**：`FinalRegistrySnapshot / 全 FactorEngine`

**当前问题**

commit/dirty/registry/evidence/implementation/contract hashes加载时校验。

**必须怎么改**

将本规则实现成机器 detector；不得依赖人工 canonical 白名单。每个适用 canonical 都必须得到明确 outcome；不适用需给 machine reason。

**必须新增/修改测试**

至少一个正例 + 一个 mutation 负例；最终 detector coverage set 与 applicable canonical set 完全相等。

### R16-228｜P0｜MANDATORY_FULL_AUDIT｜Production vs planning manifest

**位置**：`FinalRegistrySnapshot / 全 FactorEngine`

**当前问题**

eligible runtime和pending remediation严格不同schema。

**必须怎么改**

将本规则实现成机器 detector；不得依赖人工 canonical 白名单。每个适用 canonical 都必须得到明确 outcome；不适用需给 machine reason。

**必须新增/修改测试**

至少一个正例 + 一个 mutation 负例；最终 detector coverage set 与 applicable canonical set 完全相等。

### R16-229｜P0｜MANDATORY_FULL_AUDIT｜Cold-start real AST audit

**位置**：`FinalRegistrySnapshot / 全 FactorEngine`

**当前问题**

每真实seed逐 parse/type/bind/source/execute。

**必须怎么改**

将本规则实现成机器 detector；不得依赖人工 canonical 白名单。每个适用 canonical 都必须得到明确 outcome；不适用需给 machine reason。

**必须新增/修改测试**

至少一个正例 + 一个 mutation 负例；最终 detector coverage set 与 applicable canonical set 完全相等。

### R16-230｜P1｜MANDATORY_FULL_AUDIT｜LLM/operator docs freshness

**位置**：`FinalRegistrySnapshot / 全 FactorEngine`

**当前问题**

catalog/LLM prompt由FinalRegistrySnapshot生成，禁止stale doc。

**必须怎么改**

将本规则实现成机器 detector；不得依赖人工 canonical 白名单。每个适用 canonical 都必须得到明确 outcome；不适用需给 machine reason。

**必须新增/修改测试**

至少一个正例 + 一个 mutation 负例；最终 detector coverage set 与 applicable canonical set 完全相等。

### R16-231｜P0｜MANDATORY_FULL_AUDIT｜Deprecated canonical isolation

**位置**：`FinalRegistrySnapshot / 全 FactorEngine`

**当前问题**

compat alias不作为独立mining identity。

**必须怎么改**

将本规则实现成机器 detector；不得依赖人工 canonical 白名单。每个适用 canonical 都必须得到明确 outcome；不适用需给 machine reason。

**必须新增/修改测试**

至少一个正例 + 一个 mutation 负例；最终 detector coverage set 与 applicable canonical set 完全相等。

### R16-232｜P0｜MANDATORY_FULL_AUDIT｜Research/diagnostic terminal role

**位置**：`FinalRegistrySnapshot / 全 FactorEngine`

**当前问题**

supporting role不得因默认terminal逻辑被开放。

**必须怎么改**

将本规则实现成机器 detector；不得依赖人工 canonical 白名单。每个适用 canonical 都必须得到明确 outcome；不适用需给 machine reason。

**必须新增/修改测试**

至少一个正例 + 一个 mutation 负例；最终 detector coverage set 与 applicable canonical set 完全相等。

### R16-233｜P0｜MANDATORY_FULL_AUDIT｜Unsupported market hard filter

**位置**：`FinalRegistrySnapshot / 全 FactorEngine`

**当前问题**

A股/US/HK specialized operator市场不匹配不可eligible。

**必须怎么改**

将本规则实现成机器 detector；不得依赖人工 canonical 白名单。每个适用 canonical 都必须得到明确 outcome；不适用需给 machine reason。

**必须新增/修改测试**

至少一个正例 + 一个 mutation 负例；最终 detector coverage set 与 applicable canonical set 完全相等。

### R16-234｜P0｜MANDATORY_FULL_AUDIT｜UNKNOWN vs MISSING

**位置**：`FinalRegistrySnapshot / 全 FactorEngine`

**当前问题**

未知context不能误报缺数据，也不能当available。

**必须怎么改**

将本规则实现成机器 detector；不得依赖人工 canonical 白名单。每个适用 canonical 都必须得到明确 outcome；不适用需给 machine reason。

**必须新增/修改测试**

至少一个正例 + 一个 mutation 负例；最终 detector coverage set 与 applicable canonical set 完全相等。

### R16-235｜P0｜MANDATORY_FULL_AUDIT｜Stale evidence invalidation

**位置**：`FinalRegistrySnapshot / 全 FactorEngine`

**当前问题**

实现/contract/semantic policy任一变化使旧证据失效。

**必须怎么改**

将本规则实现成机器 detector；不得依赖人工 canonical 白名单。每个适用 canonical 都必须得到明确 outcome；不适用需给 machine reason。

**必须新增/修改测试**

至少一个正例 + 一个 mutation 负例；最终 detector coverage set 与 applicable canonical set 完全相等。

### R16-236｜P0｜MANDATORY_FULL_AUDIT｜Evidence same-commit release

**位置**：`FinalRegistrySnapshot / 全 FactorEngine`

**当前问题**

release evidence必须和最终code commit完全一致。

**必须怎么改**

将本规则实现成机器 detector；不得依赖人工 canonical 白名单。每个适用 canonical 都必须得到明确 outcome；不适用需给 machine reason。

**必须新增/修改测试**

至少一个正例 + 一个 mutation 负例；最终 detector coverage set 与 applicable canonical set 完全相等。

### R16-237｜P0｜MANDATORY_FULL_AUDIT｜Evidence rule-matrix binding

**位置**：`FinalRegistrySnapshot / 全 FactorEngine`

**当前问题**

evidence引用逐rule outcome hash，不只单bool。

**必须怎么改**

将本规则实现成机器 detector；不得依赖人工 canonical 白名单。每个适用 canonical 都必须得到明确 outcome；不适用需给 machine reason。

**必须新增/修改测试**

至少一个正例 + 一个 mutation 负例；最终 detector coverage set 与 applicable canonical set 完全相等。

### R16-238｜P0｜MANDATORY_FULL_AUDIT｜A股状态源规则版本

**位置**：`FinalRegistrySnapshot / 全 FactorEngine`

**当前问题**

ST/涨跌停/IPO/北交/历史规则按date-aware source contract。

**必须怎么改**

将本规则实现成机器 detector；不得依赖人工 canonical 白名单。每个适用 canonical 都必须得到明确 outcome；不适用需给 machine reason。

**必须新增/修改测试**

至少一个正例 + 一个 mutation 负例；最终 detector coverage set 与 applicable canonical set 完全相等。

### R16-239｜P0｜MANDATORY_FULL_AUDIT｜Top-K tie boundary

**位置**：`FinalRegistrySnapshot / 全 FactorEngine`

**当前问题**

tie-inclusive或InstrumentId secondary key，定义固定。

**必须怎么改**

将本规则实现成机器 detector；不得依赖人工 canonical 白名单。每个适用 canonical 都必须得到明确 outcome；不适用需给 machine reason。

**必须新增/修改测试**

至少一个正例 + 一个 mutation 负例；最终 detector coverage set 与 applicable canonical set 完全相等。

### R16-240｜P0｜MANDATORY_FULL_AUDIT｜Composition typed schema

**位置**：`FinalRegistrySnapshot / 全 FactorEngine`

**当前问题**

part-whole不能从DataFrame列名猜。

**必须怎么改**

将本规则实现成机器 detector；不得依赖人工 canonical 白名单。每个适用 canonical 都必须得到明确 outcome；不适用需给 machine reason。

**必须新增/修改测试**

至少一个正例 + 一个 mutation 负例；最终 detector coverage set 与 applicable canonical set 完全相等。

### R16-241｜P1｜MANDATORY_FULL_AUDIT｜Estimator resolution isolation

**位置**：`FinalRegistrySnapshot / 全 FactorEngine`

**当前问题**

bins/grid/projections/surrogates不作为默认经济搜索维度。

**必须怎么改**

将本规则实现成机器 detector；不得依赖人工 canonical 白名单。每个适用 canonical 都必须得到明确 outcome；不适用需给 machine reason。

**必须新增/修改测试**

至少一个正例 + 一个 mutation 负例；最终 detector coverage set 与 applicable canonical set 完全相等。

### R16-242｜P0｜MANDATORY_FULL_AUDIT｜Spectral physical frequency

**位置**：`FinalRegistrySnapshot / 全 FactorEngine`

**当前问题**

band对应physical bars，不随missing压缩。

**必须怎么改**

将本规则实现成机器 detector；不得依赖人工 canonical 白名单。每个适用 canonical 都必须得到明确 outcome；不适用需给 machine reason。

**必须新增/修改测试**

至少一个正例 + 一个 mutation 负例；最终 detector coverage set 与 applicable canonical set 完全相等。

### R16-243｜P0｜MANDATORY_FULL_AUDIT｜Embedding physical lag

**位置**：`FinalRegistrySnapshot / 全 FactorEngine`

**当前问题**

Theiler/embedding endpoint使用原physical row/time。

**必须怎么改**

将本规则实现成机器 detector；不得依赖人工 canonical 白名单。每个适用 canonical 都必须得到明确 outcome；不适用需给 machine reason。

**必须新增/修改测试**

至少一个正例 + 一个 mutation 负例；最终 detector coverage set 与 applicable canonical set 完全相等。

### R16-244｜P1｜MANDATORY_FULL_AUDIT｜Graph sample-size normalization

**位置**：`FinalRegistrySnapshot / 全 FactorEngine`

**当前问题**

RQA/HVG/topology统计不让graph N本身成为伪alpha。

**必须怎么改**

将本规则实现成机器 detector；不得依赖人工 canonical 白名单。每个适用 canonical 都必须得到明确 outcome；不适用需给 machine reason。

**必须新增/修改测试**

至少一个正例 + 一个 mutation 负例；最终 detector coverage set 与 applicable canonical set 完全相等。

### R16-245｜P0｜MANDATORY_FULL_AUDIT｜EVT tail support

**位置**：`FinalRegistrySnapshot / 全 FactorEngine`

**当前问题**

tail count/threshold stability/effective-k完整。

**必须怎么改**

将本规则实现成机器 detector；不得依赖人工 canonical 白名单。每个适用 canonical 都必须得到明确 outcome；不适用需给 machine reason。

**必须新增/修改测试**

至少一个正例 + 一个 mutation 负例；最终 detector coverage set 与 applicable canonical set 完全相等。

### R16-246｜P0｜MANDATORY_FULL_AUDIT｜DMD/SSA/SVD rank contract

**位置**：`FinalRegistrySnapshot / 全 FactorEngine`

**当前问题**

numerical rank/eigen-gap进入policy/evidence。

**必须怎么改**

将本规则实现成机器 detector；不得依赖人工 canonical 白名单。每个适用 canonical 都必须得到明确 outcome；不适用需给 machine reason。

**必须新增/修改测试**

至少一个正例 + 一个 mutation 负例；最终 detector coverage set 与 applicable canonical set 完全相等。

### R16-247｜P0｜MANDATORY_FULL_AUDIT｜Cross-sectional breadth

**位置**：`FinalRegistrySnapshot / 全 FactorEngine`

**当前问题**

CS estimator有PIT universe + minimum breadth。

**必须怎么改**

将本规则实现成机器 detector；不得依赖人工 canonical 白名单。每个适用 canonical 都必须得到明确 outcome；不适用需给 machine reason。

**必须新增/修改测试**

至少一个正例 + 一个 mutation 负例；最终 detector coverage set 与 applicable canonical set 完全相等。

### R16-248｜P0｜MANDATORY_FULL_AUDIT｜Group minimum size

**位置**：`FinalRegistrySnapshot / 全 FactorEngine`

**当前问题**

group statistic按方法有support floor。

**必须怎么改**

将本规则实现成机器 detector；不得依赖人工 canonical 白名单。每个适用 canonical 都必须得到明确 outcome；不适用需给 machine reason。

**必须新增/修改测试**

至少一个正例 + 一个 mutation 负例；最终 detector coverage set 与 applicable canonical set 完全相等。

### R16-249｜P0｜MANDATORY_FULL_AUDIT｜Benchmark ex-self

**位置**：`FinalRegistrySnapshot / 全 FactorEngine`

**当前问题**

peer/market benchmark默认排除自身或标benchmark-only。

**必须怎么改**

将本规则实现成机器 detector；不得依赖人工 canonical 白名单。每个适用 canonical 都必须得到明确 outcome；不适用需给 machine reason。

**必须新增/修改测试**

至少一个正例 + 一个 mutation 负例；最终 detector coverage set 与 applicable canonical set 完全相等。

### R16-250｜P0｜MANDATORY_FULL_AUDIT｜Benchmark SameAxis

**位置**：`FinalRegistrySnapshot / 全 FactorEngine`

**当前问题**

daily/minute benchmark不得静默reindex。

**必须怎么改**

将本规则实现成机器 detector；不得依赖人工 canonical 白名单。每个适用 canonical 都必须得到明确 outcome；不适用需给 machine reason。

**必须新增/修改测试**

至少一个正例 + 一个 mutation 负例；最终 detector coverage set 与 applicable canonical set 完全相等。

### R16-251｜P0｜MANDATORY_FULL_AUDIT｜Coverage参数非alpha

**位置**：`FinalRegistrySnapshot / 全 FactorEngine`

**当前问题**

coverage/min support只做quality gate，不作为搜索信号。

**必须怎么改**

将本规则实现成机器 detector；不得依赖人工 canonical 白名单。每个适用 canonical 都必须得到明确 outcome；不适用需给 machine reason。

**必须新增/修改测试**

至少一个正例 + 一个 mutation 负例；最终 detector coverage set 与 applicable canonical set 完全相等。

### R16-252｜P1｜MANDATORY_FULL_AUDIT｜Monotonic property tests

**位置**：`FinalRegistrySnapshot / 全 FactorEngine`

**当前问题**

理论单调算子自动 property test。

**必须怎么改**

将本规则实现成机器 detector；不得依赖人工 canonical 白名单。每个适用 canonical 都必须得到明确 outcome；不适用需给 machine reason。

**必须新增/修改测试**

至少一个正例 + 一个 mutation 负例；最终 detector coverage set 与 applicable canonical set 完全相等。

### R16-253｜P1｜MANDATORY_FULL_AUDIT｜Permutation metamorphic

**位置**：`FinalRegistrySnapshot / 全 FactorEngine`

**当前问题**

允许/不允许的时间/列permutation按contract验证。

**必须怎么改**

将本规则实现成机器 detector；不得依赖人工 canonical 白名单。每个适用 canonical 都必须得到明确 outcome；不适用需给 machine reason。

**必须新增/修改测试**

至少一个正例 + 一个 mutation 负例；最终 detector coverage set 与 applicable canonical set 完全相等。

### R16-254｜P1｜MANDATORY_FULL_AUDIT｜Serialization determinism

**位置**：`FinalRegistrySnapshot / 全 FactorEngine`

**当前问题**

同FinalSnapshot重复生成manifest/evidence字节稳定。

**必须怎么改**

将本规则实现成机器 detector；不得依赖人工 canonical 白名单。每个适用 canonical 都必须得到明确 outcome；不适用需给 machine reason。

**必须新增/修改测试**

至少一个正例 + 一个 mutation 负例；最终 detector coverage set 与 applicable canonical set 完全相等。

### R16-255｜P1｜MANDATORY_FULL_AUDIT｜Definition consistency

**位置**：`FinalRegistrySnapshot / 全 FactorEngine`

**当前问题**

docstring/metadata/catalog/LLM description/definition_id一致。

**必须怎么改**

将本规则实现成机器 detector；不得依赖人工 canonical 白名单。每个适用 canonical 都必须得到明确 outcome；不适用需给 machine reason。

**必须新增/修改测试**

至少一个正例 + 一个 mutation 负例；最终 detector coverage set 与 applicable canonical set 完全相等。

### R16-256｜P1｜MANDATORY_FULL_AUDIT｜Hidden magic constants

**位置**：`FinalRegistrySnapshot / 全 FactorEngine`

**当前问题**

影响数学定义的0.05/20/60/1e-9等进入policy或version。

**必须怎么改**

将本规则实现成机器 detector；不得依赖人工 canonical 白名单。每个适用 canonical 都必须得到明确 outcome；不适用需给 machine reason。

**必须新增/修改测试**

至少一个正例 + 一个 mutation 负例；最终 detector coverage set 与 applicable canonical set 完全相等。

### R16-257｜P0｜MANDATORY_FULL_AUDIT｜No future-dependent warnings/errors

**位置**：`FinalRegistrySnapshot / 全 FactorEngine`

**当前问题**

未来数据变化不能使prefix调用从成功变异常。

**必须怎么改**

将本规则实现成机器 detector；不得依赖人工 canonical 白名单。每个适用 canonical 都必须得到明确 outcome；不适用需给 machine reason。

**必须新增/修改测试**

至少一个正例 + 一个 mutation 负例；最终 detector coverage set 与 applicable canonical set 完全相等。

### R16-258｜P0｜MANDATORY_FULL_AUDIT｜No numeric SemanticType heuristics

**位置**：`FinalRegistrySnapshot / 全 FactorEngine`

**当前问题**

不能根据数据看起来像price/return来决定operator domain。

**必须怎么改**

将本规则实现成机器 detector；不得依赖人工 canonical 白名单。每个适用 canonical 都必须得到明确 outcome；不适用需给 machine reason。

**必须新增/修改测试**

至少一个正例 + 一个 mutation 负例；最终 detector coverage set 与 applicable canonical set 完全相等。

### R16-259｜P0｜MANDATORY_FULL_AUDIT｜Field/operator separation

**位置**：`FinalRegistrySnapshot / 全 FactorEngine`

**当前问题**

字段概念不可作为数学operator identity。

**必须怎么改**

将本规则实现成机器 detector；不得依赖人工 canonical 白名单。每个适用 canonical 都必须得到明确 outcome；不适用需给 machine reason。

**必须新增/修改测试**

至少一个正例 + 一个 mutation 负例；最终 detector coverage set 与 applicable canonical set 完全相等。

### R16-260｜P0｜MANDATORY_FULL_AUDIT｜Optional-input slot identity

**位置**：`FinalRegistrySnapshot / 全 FactorEngine`

**当前问题**

optional panel缺失不能让系数/part身份整体左移。

**必须怎么改**

将本规则实现成机器 detector；不得依赖人工 canonical 白名单。每个适用 canonical 都必须得到明确 outcome；不适用需给 machine reason。

**必须新增/修改测试**

至少一个正例 + 一个 mutation 负例；最终 detector coverage set 与 applicable canonical set 完全相等。

### R16-261｜P0｜MANDATORY_FULL_AUDIT｜Variadic arity contract

**位置**：`FinalRegistrySnapshot / 全 FactorEngine`

**当前问题**

composition/feature bundle的arity和part ids机器声明。

**必须怎么改**

将本规则实现成机器 detector；不得依赖人工 canonical 白名单。每个适用 canonical 都必须得到明确 outcome；不适用需给 machine reason。

**必须新增/修改测试**

至少一个正例 + 一个 mutation 负例；最终 detector coverage set 与 applicable canonical set 完全相等。

### R16-262｜P1｜MANDATORY_FULL_AUDIT｜Performance budget

**位置**：`FinalRegistrySnapshot / 全 FactorEngine`

**当前问题**

真实A股规模跑runtime/memory，高成本lane预算。

**必须怎么改**

将本规则实现成机器 detector；不得依赖人工 canonical 白名单。每个适用 canonical 都必须得到明确 outcome；不适用需给 machine reason。

**必须新增/修改测试**

至少一个正例 + 一个 mutation 负例；最终 detector coverage set 与 applicable canonical set 完全相等。

### R16-263｜P0｜MANDATORY_FULL_AUDIT｜Coverage report必须100% registry

**位置**：`FinalRegistrySnapshot / 全 FactorEngine`

**当前问题**

每报告同时写 registry_count/audited_count/covered set hash。

**必须怎么改**

将本规则实现成机器 detector；不得依赖人工 canonical 白名单。每个适用 canonical 都必须得到明确 outcome；不适用需给 machine reason。

**必须新增/修改测试**

至少一个正例 + 一个 mutation 负例；最终 detector coverage set 与 applicable canonical set 完全相等。

### R16-264｜P0｜MANDATORY_FULL_AUDIT｜Generated artifact stale prevention

**位置**：`FinalRegistrySnapshot / 全 FactorEngine`

**当前问题**

artifact header携snapshot fingerprints，加载时必验。

**必须怎么改**

将本规则实现成机器 detector；不得依赖人工 canonical 白名单。每个适用 canonical 都必须得到明确 outcome；不适用需给 machine reason。

**必须新增/修改测试**

至少一个正例 + 一个 mutation 负例；最终 detector coverage set 与 applicable canonical set 完全相等。

### R16-265｜P0｜MANDATORY_FULL_AUDIT｜Strict CLI exit semantics

**位置**：`FinalRegistrySnapshot / 全 FactorEngine`

**当前问题**

任何release blocker non-zero，报告脚本不能永远return0。

**必须怎么改**

将本规则实现成机器 detector；不得依赖人工 canonical 白名单。每个适用 canonical 都必须得到明确 outcome；不适用需给 machine reason。

**必须新增/修改测试**

至少一个正例 + 一个 mutation 负例；最终 detector coverage set 与 applicable canonical set 完全相等。

### R16-266｜P0｜MANDATORY_FULL_AUDIT｜Audit mutation testing

**位置**：`FinalRegistrySnapshot / 全 FactorEngine`

**当前问题**

故意破坏关键contract/implementation，确认审计真能抓而非自证。

**必须怎么改**

将本规则实现成机器 detector；不得依赖人工 canonical 白名单。每个适用 canonical 都必须得到明确 outcome；不适用需给 machine reason。

**必须新增/修改测试**

至少一个正例 + 一个 mutation 负例；最终 detector coverage set 与 applicable canonical set 完全相等。


## 3. 所有算子族必须逐 canonical 生成结论


1. **core elementwise / arithmetic / compare / logical**：列出该 family 的 final canonical set，并逐 canonical 输出 `PASS / FIXED / REMOVED / DATA_BLOCKED / N/A / AUDIT_ERROR`。不得只写 family 总结。

2. **common statistics / rolling_pack / _rolling_fast / _numpy_kernels**：列出该 family 的 final canonical set，并逐 canonical 输出 `PASS / FIXED / REMOVED / DATA_BLOCKED / N/A / AUDIT_ERROR`。不得只写 family 总结。

3. **cross_section / cross_section_ext / cross_section_local / dynamic_knn**：列出该 family 的 final canonical set，并逐 canonical 输出 `PASS / FIXED / REMOVED / DATA_BLOCKED / N/A / AUDIT_ERROR`。不得只写 family 总结。

4. **group / group_ext / group_spectrum / neutralization**：列出该 family 的 final canonical set，并逐 canonical 输出 `PASS / FIXED / REMOVED / DATA_BLOCKED / N/A / AUDIT_ERROR`。不得只写 family 总结。

5. **relation / network_metrics / mobility / exposure / graph_spectral / temporal_graph**：列出该 family 的 final canonical set，并逐 canonical 输出 `PASS / FIXED / REMOVED / DATA_BLOCKED / N/A / AUDIT_ERROR`。不得只写 family 总结。

6. **technical / EMA / Wilder / KAMA / Supertrend / PSAR / DMI / candlestick**：列出该 family 的 final canonical set，并逐 canonical 输出 `PASS / FIXED / REMOVED / DATA_BLOCKED / N/A / AUDIT_ERROR`。不得只写 family 总结。

7. **stateful / state_event / conditional / weighted_event / interval/event geometry**：列出该 family 的 final canonical set，并逐 canonical 输出 `PASS / FIXED / REMOVED / DATA_BLOCKED / N/A / AUDIT_ERROR`。不得只写 family 总结。

8. **activity_clock / threshold_cycle / state_episode / directional_change**：列出该 family 的 final canonical set，并逐 canonical 输出 `PASS / FIXED / REMOVED / DATA_BLOCKED / N/A / AUDIT_ERROR`。不得只写 family 总结。

9. **price_volume / liquidity / turnover_survival / weighted_tail / robust_tail**：列出该 family 的 final canonical set，并逐 canonical 输出 `PASS / FIXED / REMOVED / DATA_BLOCKED / N/A / AUDIT_ERROR`。不得只写 family 总结。

10. **microstructure / intraday_agg / advanced_intraday / intraday_session / session_recovery / volume_clock**：列出该 family 的 final canonical set，并逐 canonical 输出 `PASS / FIXED / REMOVED / DATA_BLOCKED / N/A / AUDIT_ERROR`。不得只写 family 总结。

11. **depth / impact_decay / spread_estimators / barrier / micro_price / imbalance**：列出该 family 的 final canonical set，并逐 canonical 输出 `PASS / FIXED / REMOVED / DATA_BLOCKED / N/A / AUDIT_ERROR`。不得只写 family 总结。

12. **fundamental transforms / flow_semantics / quality / growth / accrual / cashflow / valuation / expectation**：列出该 family 的 final canonical set，并逐 canonical 输出 `PASS / FIXED / REMOVED / DATA_BLOCKED / N/A / AUDIT_ERROR`。不得只写 family 总结。

13. **shareholder / index / A-share state machine / source transforms**：列出该 family 的 final canonical set，并逐 canonical 输出 `PASS / FIXED / REMOVED / DATA_BLOCKED / N/A / AUDIT_ERROR`。不得只写 family 总结。

14. **regression / robust regression / quantile / expectile / AR / mean-reversion**：列出该 family 的 final canonical set，并逐 canonical 输出 `PASS / FIXED / REMOVED / DATA_BLOCKED / N/A / AUDIT_ERROR`。不得只写 family 总结。

15. **GARCH / GJR / HAR / state_space / Kalman**：列出该 family 的 final canonical set，并逐 canonical 输出 `PASS / FIXED / REMOVED / DATA_BLOCKED / N/A / AUDIT_ERROR`。不得只写 family 总结。

16. **MI / CMI / TE / effective TE / HSIC / kernel Granger / BDS**：列出该 family 的 final canonical set，并逐 canonical 输出 `PASS / FIXED / REMOVED / DATA_BLOCKED / N/A / AUDIT_ERROR`。不得只写 family 总结。

17. **spectral / cross_spectrum / bicoherence / wavelet / MODWT**：列出该 family 的 final canonical set，并逐 canonical 输出 `PASS / FIXED / REMOVED / DATA_BLOCKED / N/A / AUDIT_ERROR`。不得只写 family 总结。

18. **Hankel / SSA / DMD / Matrix Profile / sequence anomaly**：列出该 family 的 final canonical set，并逐 canonical 输出 `PASS / FIXED / REMOVED / DATA_BLOCKED / N/A / AUDIT_ERROR`。不得只写 family 总结。

19. **complexity / entropy / permutation / recurrence**：列出该 family 的 final canonical set，并逐 canonical 输出 `PASS / FIXED / REMOVED / DATA_BLOCKED / N/A / AUDIT_ERROR`。不得只写 family 总结。

20. **RQA / HVG / intrinsic dimension / local Lyapunov**：列出该 family 的 final canonical set，并逐 canonical 输出 `PASS / FIXED / REMOVED / DATA_BLOCKED / N/A / AUDIT_ERROR`。不得只写 family 总结。

21. **topology / persistence / advanced topology**：列出该 family 的 final canonical set，并逐 canonical 输出 `PASS / FIXED / REMOVED / DATA_BLOCKED / N/A / AUDIT_ERROR`。不得只写 family 总结。

22. **multifractal / rough volatility / memory / fractional difference**：列出该 family 的 final canonical set，并逐 canonical 输出 `PASS / FIXED / REMOVED / DATA_BLOCKED / N/A / AUDIT_ERROR`。不得只写 family 总结。

23. **EVT / Hill / GPD / first-passage / extreme tail**：列出该 family 的 final canonical set，并逐 canonical 输出 `PASS / FIXED / REMOVED / DATA_BLOCKED / N/A / AUDIT_ERROR`。不得只写 family 总结。

24. **Markov / Kramers-Moyal / transition / survival state**：列出该 family 的 final canonical set，并逐 canonical 输出 `PASS / FIXED / REMOVED / DATA_BLOCKED / N/A / AUDIT_ERROR`。不得只写 family 总结。

25. **path signature / feature geometry / advanced structure**：列出该 family 的 final canonical set，并逐 canonical 输出 `PASS / FIXED / REMOVED / DATA_BLOCKED / N/A / AUDIT_ERROR`。不得只写 family 总结。

26. **composition / CLR / ILR / Aitchison / JS**：列出该 family 的 final canonical set，并逐 canonical 输出 `PASS / FIXED / REMOVED / DATA_BLOCKED / N/A / AUDIT_ERROR`。不得只写 family 总结。

27. **change-point / GLR / Pettitt / CUSUM**：列出该 family 的 final canonical set，并逐 canonical 输出 `PASS / FIXED / REMOVED / DATA_BLOCKED / N/A / AUDIT_ERROR`。不得只写 family 总结。

28. **OHLC spread / EDGE / Abdi-Ranaldo / Pastor-Stambaugh**：列出该 family 的 final canonical set，并逐 canonical 输出 `PASS / FIXED / REMOVED / DATA_BLOCKED / N/A / AUDIT_ERROR`。不得只写 family 总结。

29. **pandas / Polars native / Polars bridge / SQL emitter**：列出该 family 的 final canonical set，并逐 canonical 输出 `PASS / FIXED / REMOVED / DATA_BLOCKED / N/A / AUDIT_ERROR`。不得只写 family 总结。

30. **operator_surface / registry / dedupe / governance / production_hardening**：列出该 family 的 final canonical set，并逐 canonical 输出 `PASS / FIXED / REMOVED / DATA_BLOCKED / N/A / AUDIT_ERROR`。不得只写 family 总结。

31. **closure / semantic audit / edge/source/backend evidence**：列出该 family 的 final canonical set，并逐 canonical 输出 `PASS / FIXED / REMOVED / DATA_BLOCKED / N/A / AUDIT_ERROR`。不得只写 family 总结。

32. **mining catalog / admission matrix / manifests / cold-start integration**：列出该 family 的 final canonical set，并逐 canonical 输出 `PASS / FIXED / REMOVED / DATA_BLOCKED / N/A / AUDIT_ERROR`。不得只写 family 总结。

33. **execution_contract / incremental scheduler / checkpoint runtime**：列出该 family 的 final canonical set，并逐 canonical 输出 `PASS / FIXED / REMOVED / DATA_BLOCKED / N/A / AUDIT_ERROR`。不得只写 family 总结。

34. **docs / generated catalog / LLM prompt / evidence artifacts**：列出该 family 的 final canonical set，并逐 canonical 输出 `PASS / FIXED / REMOVED / DATA_BLOCKED / N/A / AUDIT_ERROR`。不得只写 family 总结。


## 4. 每个 canonical 的最终 machine audit record


```yaml
canonical:
definition_id:
implementation_files: []
aliases: []
surface:
lifecycle:
mining_role:
terminal_allowed:
supported_markets: []
panel_signature:
scalar_signature:
param_specs:
signature_default_parity:
relational_constraints:
input_field_concepts:
input_semantic_types:
input_units:
output_unit:
output_range:
input_grain:
output_grain:
available_at:
same_session_usable:
session_contract:
source_requirements:
source_concepts:
pit_mode:
window_semantics:
history_requirement:
current_row_semantics:
missing_policy:
axis_contract:
broadcast_contract:
stateful_math:
recursive_math:
checkpoint_supported:
incremental_supported:
full_history_required:
cost_model:
backend_native: []
backend_delegated: []
semantic_rule_outcomes:
golden_tests:
null_tests:
metamorphic_tests:
backend_parity:
implementation_hash:
contract_hash:
evidence_hash:
evidence_commit:
production_certified:
mining_eligible:
blockers: []
final_action:
```

**Final registry 中任何 public canonical 没有这条记录，整轮直接失败。**



## 5. Required detector matrix


最终生成 `audit_detector_matrix.json`，每个 detector 至少记录：

```text
detector_id
required_for_release
applicable_canonical_count
executed_canonical_count
pass_count
fail_count
not_applicable_count
not_run_count
audit_error_count
covered_canonical_hash
detector_implementation_hash
```

Release 条件：

```text
required detector: NOT_RUN == 0
required detector: AUDIT_ERROR == 0
required detector: executed + N/A == applicable
all P0 findings resolved
per_canonical_audit set == FinalRegistrySnapshot set
```



## 6. Evidence 与 manifest 必须在最后重生


顺序固定：

1. 完成全部实现、metadata、contract、backend、mining、audit 修复；
2. `load_all()` 完成所有 finalization；
3. `OperatorRegistry.freeze()`；
4. 生成 `FinalRegistrySnapshot` 和完整逻辑 fingerprint；
5. 运行 required semantic/golden/null/metamorphic/backend/checkpoint tests；
6. 在**同一个最终 commit**重生 primitive/operator/recipe/source/edge/backend evidence；
7. 生成 **eligible-only** runtime mining manifest；
8. 单独生成 pending remediation plan；
9. 对真实 cold-start factor expressions 逐条 parse/type/bind/source/execute；
10. 再跑 master audit，所有 required detector 100% coverage 后才允许 release-safe。



## 7. 最终交付物


- R16_FIX_REPORT.md：R16-001～R16-266 每项状态、修改位置、对应测试；

- R16_AUTO_DISCOVERED_FINDINGS.md：全 canonical 自动审计额外发现的问题，不与本文件合并，作为本轮执行报告附件；

- per_canonical_audit.jsonl；

- audit_detector_matrix.json；

- parameter_contract_audit.json；

- source_pit_capability_audit.json；

- session_grain_audit.json；

- history_checkpoint_audit.json；

- backend_semantic_parity_matrix.json；

- semantic_duplicate_report.json；

- alias_equivalence_report.json；

- runtime_mining_manifest.eligible.json；

- operator_remediation_plan.pending.json；

- cold_start_expression_audit.jsonl；

- fresh_evidence_lineage.json；

- FINAL_RELEASE_INVARIANTS.json。


## 8. Definition of Done


只有以下全部成立才能回复“这一轮整改完成”：

```text
1. final registry 已 freeze。
2. per_canonical_audit canonical set == final registry canonical set。
3. required detector 没有 sampling 冒充 full coverage。
4. NOT_RUN == 0（required detectors）。
5. AUDIT_ERROR == 0。
6. R16 CONFIRMED_CURRENT_DEFECT/CURRENT_REGRESSION 全部 fixed/removed/data-blocked。
7. mineable scalar ParamSpec coverage == 100%。
8. signature/default/ParamSpec/manifest 参数语义一致。
9. SameAxis/Missing/Window/History/Session/Grain/PIT 全为机器 contract。
10. applicable backend 的正常数值和非法输入语义 parity 均通过。
11. stateful full-run vs segmented parity 通过，或明确 full-history high-cost lane。
12. production evidence 与最终代码同 commit、同 implementation/contract hash。
13. runtime manifest 只含真正 eligible operator。
14. pending/planning operator 绝不进入 runtime manifest。
15. 真实 cold-start expressions 逐条通过完整执行链。
16. semantic duplicate/alias/dead param 没有未处理搜索重复。
17. 最终报告不存在 hard-coded empty list、placeholder、file-exists 即 PASS。
18. mutation tests 证明审计器真的能抓关键错误。
```

最后一条执行纪律：

> **不要因为 R16-001～R16-266 全部处理完就停止。处理完后必须继续跑全 final-registry machine audit；如果又发现未在本文件点名的新问题，继续修，直到 machine audit 的 applicable canonical coverage 达到 100%。**
