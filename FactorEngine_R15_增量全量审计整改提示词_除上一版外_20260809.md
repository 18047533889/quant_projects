# FactorEngine R15 增量全量审计与整改 Master Prompt
## ——只处理“上一份 2026-08-09 全量整改提示词之外”的剩余问题

> **直接用途**：把本文件完整交给另一个 coding AI。它应把本文件当作“第二份并行整改任务书”，在不覆盖第一份整改结果的前提下继续修 FactorEngine。  
> **仓库**：`18047533889/quant_projects`  
> **上一份整改提示词的代码基线**：`44b294608ad1648c0f0fa441c33e69a6d10f80d5`  
> **本次人工增量审计时最新可见 HEAD**：`9aaadb8f3a10c4fb7a3b7823617822da9d274ec1`  
> **增量 commit 数**：2。  
> **重要**：用户正在让多个 AI 并行整改。真正执行本文件时，workspace 很可能已经比上述 HEAD 更新。**每改一处之前必须重新读取当前文件，不允许拿本提示词审计时的旧文件整文件覆盖当前 workspace。**

---

# 0. 与上一份提示词的边界：绝对不要重复劳动

上一份文件已经覆盖：
- `NEW-001 ... NEW-260`
- `HIST-001 ... HIST-100`

本文件**故意不重复**那些问题。上一份已覆盖的主要类别包括 strict-int/ParamRole 旧问题、rolling 核、ConditionBool、A 股 session 基础、minute aggregation、ffill、fundamental PIT、CS/group/relation、state/history、activity clock、composition、RQA、return decomposition、turnover survival、weighted tail、BVC、quantile dynamics、legacy microstructure、price-volume/CAPM、valuation/shareholder/index，以及此前大量高级统计数学问题。

执行规则：

1. 先读取当前 workspace。
2. 对本文件每个 `R15-INC-*` 判断：`OPEN / FIXED_BY_THIS_AI / ALREADY_FIXED_BY_PARALLEL_AI / NO_LONGER_APPLICABLE`。
3. 若已经被第一份 prompt 或另一个并行 AI 修掉，不要重写、更不要回滚。
4. 只有新 R12/R15 代码重新引入同类 bug 时，才作为 `NEW_PATH_REGRESSION` 处理。
5. 最终仍要跑第一份 prompt 形成的回归测试，确保 `PREVIOUS_PROMPT_REGRESSION_COUNT = 0`。

---

# 1. 本次增量审计覆盖范围

从上一份基线到本次 HEAD，FactorEngine 相关新增/修改的关键文件包括：

- `factor_engine/mining/operator_catalog.py`（新增）
- `factor_engine/audit/operator_admission_matrix.py`（新增）
- `factor_engine/scripts/audit_all_registered_operators.py`（新增）
- `factor_engine/scripts/export_mining_manifest.py`（新增）
- `factor_engine/cleaned_operators/closure/*`（整套新增）
- `factor_engine/api/mining_integration.py`
- `factor_engine/cleaned_operators/advanced_intraday.py`
- `factor_engine/cleaned_operators/conditional_dependence.py`
- `factor_engine/cleaned_operators/cross_section_ext.py`
- `factor_engine/cleaned_operators/cross_section_local.py`
- `factor_engine/cleaned_operators/dependence_ext.py`
- `factor_engine/cleaned_operators/event_interval.py`
- `factor_engine/cleaned_operators/extreme_tail.py`
- `factor_engine/cleaned_operators/glr_change.py`
- `factor_engine/cleaned_operators/intraday_session.py`
- `factor_engine/cleaned_operators/ohlc_spread.py`
- `factor_engine/cleaned_operators/session_recovery.py`
- `factor_engine/cleaned_operators/volume_clock.py`
- `factor_engine/cleaned_operators/operator_policy.py`
- `factor_engine/cleaned_operators/common/polars_batch_mirror.py`
- `factor_engine/evidence/factor_operator_verified.json`
- 新增/修改的 R11/R12 tests。

`activity_clock.py / composition.py / liquidity_v2.py / safe_ops.py` 等虽然本次 diff 也有改动，但其核心问题多数已经在上一份 prompt 中列过。本文件只要求做**无回退验证**，不重新列旧问题。

---

# 2. 本轮整改的最高级目标

目标不是继续加零散 if，而是让 R12 新增的 mining / admission / closure / manifest 层本身成为可信机器权威：

```text
FinalRegistrySnapshot
    └── ResolvedSemanticContract per canonical
          ├── ResolvedSignature
          ├── SemanticType / Unit
          ├── AxisContract
          ├── MissingValuePolicy + TimeTopologyPolicy
          ├── CurrentRowContract
          ├── Window/HistoryContract
          ├── GrainTransform + Availability
          ├── SourceRequirement(s)
          ├── MarketCapability
          ├── MiningRole + AST positions
          ├── CostModel
          └── ExecutionContract

                ↓
AdmissionDecision(context)
                ↓
MiningCatalog / AdmissionMatrix / Manifest / ColdStart / AlphaProbe / AlphaMiner
```

不允许再出现：
- 一个模块读 `operator_policy`，另一个读 closure side registry；
- 一个脚本按 surface 猜 role；
- 一个脚本按 prefix 猜 source；
- 一个脚本把 UNKNOWN 当全部可用；
- audit invariant 写死 `[]`；
- test 只证明“有 role”，不证明 role 正确；
- manifest 没有 SHA 却被当生产 truth；
- supporting/diagnostic/denied role 被标 `terminal_allowed=True`。

---

# 3. 已确认的增量问题与整改要求

## R15-INC-001｜`assign_mining_role()` 名义 fail-closed，实际尾部分支默认 `ALPHA`  **[P0]**

**涉及文件**：`factor_engine/mining/operator_catalog.py`

**问题**：任何未命中显式规则的已注册 canonical 都会落入 `MiningRole.ALPHA`。新算子、漏标 role 的算子、全局/组广播统计都可能被静默当作个股 terminal alpha。当前测试只检查 role 非 None，反而把这种漏分类当通过。

**整改**：删除 default ALPHA；新增 `UNCLASSIFIED/UNRESOLVED`。只有显式 metadata role 或可证明的 semantic-type mapping 才能进入 ALPHA；final freeze 要求 unresolved=0。

**必须补的测试**：注册无 role 测试算子，必须 UNCLASSIFIED/失败，不能自动 ALPHA；全 canonical role_origin coverage=100%。


## R15-INC-002｜`market` 参数是 dead API  **[P0]**

**涉及文件**：`factor_engine/mining/operator_catalog.py::get_mining_operators`

**问题**：`market` 被 lower 后没有参与任何过滤，A股/美股可能得到相同 pool。

**整改**：新增 `MarketContract` 与 required market capabilities；market/source/grain 三轴分离；未知市场 fail-closed。

**必须补的测试**：A-share limit/session 算子在 US context 不得返回；US-only 反之。


## R15-INC-003｜`target_frequency='minute'` 逻辑把 minute→daily 当 minute 输出  **[P0]**

**涉及文件**：`factor_engine/mining/operator_catalog.py`

**问题**：`INTRADAY_EOD` 是 minute input / daily output，但当前频率判断方向错误。

**整改**：分别建 `input_grain` 与 `output_grain`；target_frequency 只匹配 output_grain。

**必须补的测试**：daily/minute/fundamental_period truth table；INTRADAY_EOD 只在 target=daily 且 minute+calendar capability满足时出现。


## R15-INC-004｜空 `available_sources` 被解释为“全部 source 可用”  **[P0]**

**涉及文件**：`factor_engine/mining/operator_catalog.py::_sources_available`

**问题**：UNKNOWN environment 与 empty capability 被混为 `_KNOWN_SOURCES` 全可用，产生假 eligibility。

**整改**：区分 UNKNOWN/explicit empty/explicit set；生产默认 UNKNOWN 必须 fail-closed，planning 只能标 unknown。

**必须补的测试**：omitted、empty set、full set 三种调用结果明确不同。


## R15-INC-005｜SourceRequirement 被 grain/前缀猜测  **[P0]**

**涉及文件**：`factor_engine/mining/operator_catalog.py::_required_sources`

**问题**：daily grain 不等于 daily_bar；shareholder/relation/index/event/fundamental 可能同为daily shape但依赖专用PIT源。

**整改**：每 canonical 显式 `SourceRequirement[]`，包含 source_id、PIT mode、required concepts、calendar/taxonomy/vintage；grain 不是 source proxy。

**必须补的测试**：枚举 non-daily-bar source 算子，required_sources 与 provider truth 一致。


## R15-INC-006｜Source model 无法表达多源 AND 依赖  **[P0]**

**涉及文件**：`factor_engine/mining/operator_catalog.py`

**问题**：价格+财务、minute+daily rule、relation+daily value 等需要多个 source 同时存在，当前模型通常只返回一个。

**整改**：支持 AND requirement set + alternative groups。

**必须补的测试**：缺任一 required source => blocked；补齐全部后才 eligible。


## R15-INC-007｜Minute capability 没把 SessionCalendar/Timezone/MarketRule 当 required capability  **[P0]**

**涉及文件**：`factor_engine/mining/operator_catalog.py`

**问题**：只有 minute_bar 并不足以正确计算 full-session EOD factor。

**整改**：required capabilities 增加 SessionCalendar、TimezoneContract、BarTimestampConvention、MarketRuleVersion。

**必须补的测试**：有分钟表但无calendar时所有 full-session EOD factor blocked。


## R15-INC-008｜`_searchable_params()` 可能把 panel/context 参数当 scalar 搜索维度  **[P0]**

**涉及文件**：`factor_engine/mining/operator_catalog.py`

**问题**：它直接扫 param_names，calendar、DataFrame、context object 可能漏进参数面。

**整改**：只从 `ResolvedSignature.scalar_params` 产生搜索参数；panel/context/source 参数独立。

**必须补的测试**：composition/intraday/calendar/multi-input canonical 的 searchable_params 仅含真正scalar。


## R15-INC-009｜`ParamRole` Enum 比较方式错误，policy/estimator 参数可能漏进搜索  **[P0]**

**涉及文件**：`factor_engine/mining/operator_catalog.py::_searchable_params`

**问题**：代码把 Enum `str(role).lower()` 与 value 字符串比较，通常得到 `paramrole.estimator_resolution` 等，排除条件可能失效。

**整改**：直接比较 Enum identity 或 `.value`；复用 `searchable_param_names()` 单一权威。

**必须补的测试**：每个 ParamRole 参数化测试：ESTIMATOR/NUMERICAL/POLICY永远非full-search。


## R15-INC-010｜`max_cost = int(max_cost)` 会 silent truncate  **[P1]**

**涉及文件**：`factor_engine/mining/operator_catalog.py`

**问题**：3.9 会被静默变成3，搜索配置不可精确复现。

**整改**：只接受严格整数配置；bool/float非整数拒绝。

**必须补的测试**：3.9/True/'3' 均按配置schema明确处理。


## R15-INC-011｜缺 cost contract 的算子默认 cost=1  **[P0]**

**涉及文件**：`factor_engine/mining/operator_catalog.py::cost_tier`

**问题**：O(N²) graph、SVD、LP、topology 等可被错误放cheap lane。

**整改**：所有 mineable canonical 显式 `CostModel`；缺失即blocked，不默认cheap。cost model 至少表达N/window/group/solver复杂度。

**必须补的测试**：cost_model coverage=100%；synthetic scaling验证complexity class。


## R15-INC-012｜`cost_tier()` 假设 pandas backend 一定存在  **[P1]**

**涉及文件**：`factor_engine/mining/operator_catalog.py`

**问题**：直接 `OperatorRegistry.get(canonical,'pandas_numpy')` 对Polars/SQL-only未来算子不健壮。

**整改**：cost属canonical；backend multiplier另算。

**必须补的测试**：仅Polars测试canonical也能正常生成catalog。


## R15-INC-013｜Authoring `research` tier 先于 mining role，重新制造 research 死桶  **[P0]**

**涉及文件**：`factor_engine/mining/operator_catalog.py::assign_mining_role`

**问题**：surface==research 直接RESEARCH，数学上是ALPHA/STATE的待认证目标也无法进入pending promotion。

**整改**：AuthoringTier 与 MiningRole 正交；research-tier 可 ALPHA_PENDING/STATE_PENDING，execution由certification决定。

**必须补的测试**：research-tier+explicit ALPHA fixture能出pending、未认证时不出eligible。


## R15-INC-014｜`_STATE_LITERAL_OPS` 把连续统计误分为 STATE  **[P0]**

**涉及文件**：`factor_engine/mining/operator_catalog.py`

**问题**：`state_episode_mfe/mae/efficiency/retrace_ratio`、threshold cycle period/asymmetry、candle_gap_atr 等是连续量，不是Bool/离散状态。

**整改**：由 `OutputSemanticType` 分类 ContinuousStateStatistic vs ConditionBool/DiscreteState；连续量可按明确role进入terminal或interaction。

**必须补的测试**：上述canonical建立expected-role golden。


## R15-INC-015｜所有 `event_*` 名称被按 EVENT mask 分类  **[P0]**

**涉及文件**：`factor_engine/mining/operator_catalog.py`

**问题**：`event_interval_memory/local_variation/fano_factor/fano_excess` 等连续统计会被误禁止terminal。

**整改**：区分 EventBool、MarkedEvent、ContinuousEventStatistic；prefix仅lint。

**必须补的测试**：连续 event statistic 的合法 terminal/role 与真正 EventBool 分开验证。


## R15-INC-016｜`terminal_allowed` 对 INTERNAL/DIAGNOSTIC/RESEARCH/LEGACY/DENIED 反而为 True  **[P0]**

**涉及文件**：`factor_engine/mining/operator_catalog.py`

**问题**：当前只排五种nonterminal role，其余 supporting/denied 被else当terminal。

**整改**：显式 `_TERMINAL_ROLES`；其余全false。positions与terminal做双向invariant。

**必须补的测试**：遍历全部MiningRole truth table。


## R15-INC-017｜`incremental_supported = checkpoint_supported` 错误否定stateless rolling  **[P0]**

**涉及文件**：`mining/operator_catalog.py`, `audit/operator_admission_matrix.py`

**问题**：stateless有限历史算子只需warmup即可incremental，不需要checkpoint。

**整改**：Execution model分 independent_with_warmup / checkpoint / full_history。

**必须补的测试**：ts_mean/ts_std incremental=true；KAMA无checkpoint false；EMA checkpoint true。


## R15-INC-018｜`input_semantic_types` 实际装field/param名  **[P0]**

**涉及文件**：`mining/operator_catalog.py`, `audit/operator_admission_matrix.py`

**问题**：字段名不是 PositivePrice/Return/ConditionBool/GroupId 等 semantic type。

**整改**：从ResolvedSignature读取semantic type/unit/grain/entity/price basis；field concept另列。

**必须补的测试**：semantic type列只允许枚举，不允许任意字符串。


## R15-INC-019｜`MiningOperator.blockers/recommended_action` 从未真正填充  **[P0]**

**涉及文件**：`factor_engine/mining/operator_catalog.py`

**问题**：dataclass有字段，但构造时默认空；manifest随后导出空解释。

**整改**：catalog/matrix/manifest共用单一 `AdmissionDecision`。

**必须补的测试**：任意noneligible pending/all entry都必须有blockers和required_actions。


## R15-INC-020｜Missing policy 两套权威并存  **[P0]**

**涉及文件**：`mining/operator_catalog.py`, `operator_policy.py`, `closure/missing_policy.py`

**问题**：mining读旧nan_policy，新closure又有side registry，可能漂移。

**整改**：最终只消费 immutable ResolvedSemanticContract；旧policy变compat adapter并做一致性检查。

**必须补的测试**：人工制造冲突，freeze必须失败。


## R15-INC-021｜`admission='pending'` 把弱 `pit_safe` tag 当准入证据  **[P0]**

**涉及文件**：`factor_engine/mining/operator_catalog.py`

**问题**：pit_safe声明不等于temporal/source PIT evidence。

**整改**：pending允许列入规划，但必须明确缺证 blocker；不能借一个bool tag视为ready。

**必须补的测试**：pit_safe=True但source/temporal evidence缺失的fixture mining_eligible=false。


## R15-INC-022｜`available_sources` 只记录 required∩available，无法解释环境  **[P1]**

**涉及文件**：`factor_engine/mining/operator_catalog.py`

**问题**：消费者不能分辨环境总能力与算子实际需求。

**整改**：记录 environment/required/satisfied/missing 四组。

**必须补的测试**：每entry可独立解释source pass/fail。


## R15-INC-023｜Mining entry 不记录 market/frequency/source query context  **[P1]**

**涉及文件**：`mining/operator_catalog.py`, `scripts/export_mining_manifest.py`

**问题**：离线manifest不可复现。

**整改**：header记录AdmissionQuery + repo/evidence/registry fingerprint。

**必须补的测试**：不同context生成不同manifest digest。


## R15-INC-024｜SOURCE_BLOCKED 与 MiningRole 混在一起  **[P0]**

**涉及文件**：`factor_engine/mining/operator_catalog.py`

**问题**：source blocked 算子仍可能显示正常 mineable role，消费者只看role会误判。

**整改**：Role只表示语法位置；AdmissionState单独 ELIGIBLE/PENDING/SOURCE_BLOCKED/DENIED。

**必须补的测试**：source blocked 保留role但绝不能进入eligible pool。


## R15-INC-025｜角色/source heuristic 常量存在dead code/设计漂移  **[P1]**

**涉及文件**：`factor_engine/mining/operator_catalog.py`

**问题**：例如状态suffix hint声明后不一定被稳定使用，维护者误以为有保护。

**整改**：删除未使用heuristic或只做lint，不参与生产决策。

**必须补的测试**：静态dead-code检测覆盖role/source heuristics。


## R15-INC-026｜GLOBAL/GROUP role 不能只靠tag  **[P0]**

**涉及文件**：`mining/operator_catalog.py` 与 group/global operator metadata

**问题**：很多算子只写 `global_state` tag，resolver主要读catalog role；promotion后可能落default ALPHA。

**整改**：所有factor-facing canonical显式 mining_role / OutputScope；tag仅搜索标签。

**必须补的测试**：cs_rank_copula、group spectrum、MST/coexceedance做role golden。


## R15-INC-027｜MiningRole不能再从名称前缀推断  **[P0]**

**涉及文件**：`factor_engine/mining/operator_catalog.py`

**问题**：event_/state_/fin_/intra_不是类型系统，已出现连续event statistic等误分类。

**整改**：新增 OutputSemanticType + OutputScope，role由二者映射；名称只做lint。

**必须补的测试**：rename-invariance测试：换canonical名不应改变role。


## R15-INC-028｜Admission Matrix 把空source context当source全available  **[P0]**

**涉及文件**：`factor_engine/audit/operator_admission_matrix.py`

**问题**：默认静态matrix伪造 source_available=true。

**整改**：无真实CapabilityContext时标unknown，不写true。

**必须补的测试**：默认matrix source availability不能假装全可用。


## R15-INC-029｜`default_mining_eligible` 是环境无关伪truth  **[P0]**

**涉及文件**：`factor_engine/audit/operator_admission_matrix.py`

**问题**：没有market/source/frequency上下文却称default eligible。

**整改**：拆 static_certification_eligible 与 contextual_mining_eligible(context)。

**必须补的测试**：A股/美股context得到不同结果。


## R15-INC-030｜B20 `ROLE_NOT_STOCK_ALPHA` 不应是健康STATE/EVENT的 blocker  **[P1]**

**涉及文件**：`factor_engine/audit/operator_admission_matrix.py`

**问题**：合法nonterminal role被伪装成有缺陷。

**整改**：blocker只表示需修问题；role限制单列。

**必须补的测试**：完全认证EventBool应blockers=[]、terminal=false。


## R15-INC-031｜B21/B22 GROUP/GLOBAL state 也不应是质量blocker  **[P1]**

**涉及文件**：`factor_engine/audit/operator_admission_matrix.py`

**问题**：它们是合法gate/interaction角色。

**整改**：改为AST position constraint。

**必须补的测试**：健康group/global state无质量blocker。


## R15-INC-032｜B23 HIGH_COMPUTE_COST 是lane routing不是缺陷  **[P1]**

**涉及文件**：`factor_engine/audit/operator_admission_matrix.py`

**问题**：数学/证据正确的高成本算子只需进deep lane。

**整改**：单列cost_lane；只有context资源预算不够才产生context blocker。

**必须补的测试**：已认证高成本算子direct_high_cost。


## R15-INC-033｜`recursive = stateful` 概念错误  **[P0]**

**涉及文件**：`factor_engine/audit/operator_admission_matrix.py`

**问题**：episode/session-state/stateful不等于recursive。

**整改**：从ExecutionContract.state_model读取；recursive仅真实recursive。

**必须补的测试**：episode/session state: stateful=true, recursive=false。


## R15-INC-034｜Admission Matrix 继承 stateless incremental 判定错误  **[P0]**

**涉及文件**：`factor_engine/audit/operator_admission_matrix.py`

**问题**：stateless bounded history被错误标false。

**整改**：同catalog修复execution model。

**必须补的测试**：基础rolling算子incremental=true。


## R15-INC-035｜Admission Matrix current-row semantics 被压成 True/False 字符串  **[P0]**

**涉及文件**：`factor_engine/audit/operator_admission_matrix.py`

**问题**：无法区分strict-prior fit/current query/current required/current-inclusive estimator。

**整改**：结构化 CurrentRowContract。

**必须补的测试**：prior regression/zscore/event threshold/minute EOD分别golden。


## R15-INC-036｜Admission Matrix window semantics fallback只剩bounded/full_history  **[P0]**

**涉及文件**：`factor_engine/audit/operator_admission_matrix.py`

**问题**：event/session/fiscal/finite-observation/block-censor语义丢失。

**整改**：直接引用resolved Window/HistoryContract。

**必须补的测试**：event_interval/session/fundamental matrix显示真实clock。


## R15-INC-037｜B24–B32 blocker词表没有真实detector支撑  **[P0]**

**涉及文件**：`factor_engine/audit/operator_admission_matrix.py`

**问题**：DEAD_PARAMETER/SEMANTIC_DUPLICATE/MATH/UNIT/AXIS/MISSING等看似已覆盖，实际record并未跑对应机器审计。

**整改**：每个blocker绑定detector_id、version、evidence；无detector不能声称PASS。

**必须补的测试**：故意坏算子应触发对应B-code。


## R15-INC-038｜recommended_action必须覆盖全部blockers  **[P1]**

**涉及文件**：`factor_engine/audit/operator_admission_matrix.py`

**问题**：多问题算子不能只根据first blocker给一条动作。

**整改**：生成有序 required_actions[]，按math/PIT/source→contract→evidence→performance。

**必须补的测试**：三blocker fixture三项action齐全。


## R15-INC-039｜ResearchToolRegistry被全量Admission审计跳过  **[P1]**

**涉及文件**：`factor_engine/audit/operator_admission_matrix.py`

**问题**：主DSL可不包含research tool，但全库仍需cross-registry duplicate/name collision审计。

**整改**：单独tool matrix或统一surface_kind。

**必须补的测试**：ResearchTool与主registry collision=0。


## R15-INC-040｜Admission Matrix CLI 无release失败exit code  **[P0]**

**涉及文件**：`factor_engine/audit/operator_admission_matrix.py::main`

**问题**：即使invariant坏仍可能正常写文件并0退出。

**整改**：validate_admission_matrix，任何release violation exit 1。

**必须补的测试**：注入坏record CLI红。


## R15-INC-041｜Admission artifact缺SHA/fingerprint/context  **[P0]**

**涉及文件**：`factor_engine/audit/operator_admission_matrix.py`

**问题**：旧matrix在并行AI快速提交后可被误当current truth。

**整改**：header写commit_sha、registry/evidence/contract fingerprint、query context。

**必须补的测试**：代码/evidence变更后stale artifact fail。


## R15-INC-042｜`audit_all_registered_operators.py` invariant失败也无条件exit 0  **[P0]**

**涉及文件**：`factor_engine/scripts/audit_all_registered_operators.py`

**问题**：CI可fake green。

**整改**：任何非空release invariant return 1。

**必须补的测试**：故意制造失败，CLI/CI必须红。


## R15-INC-043｜关键invariant直接硬编码空列表  **[P0]**

**涉及文件**：`factor_engine/scripts/audit_all_registered_operators.py`

**问题**：`CERTIFIED_FACTOR_NOT_IN_MINING_MANIFEST`、`SEMANTIC_DUPLICATE_CANONICALS`、`DEAD_SEARCHABLE_PARAMS` 没真实计算。

**整改**：实现set difference、semantic fingerprint graph、parameter injectivity；未实现detector时状态UNKNOWN/FAIL，不能PASS。

**必须补的测试**：已知duplicate/dead-param fixture必须被抓到。


## R15-INC-044｜A–J bucket未知情况默认落乐观ALPHA bucket  **[P0]**

**涉及文件**：`factor_engine/scripts/audit_all_registered_operators.py`

**问题**：未覆盖组合不应自动Case1。

**整改**：新增UNKNOWN bucket；release要求UNKNOWN=0。

**必须补的测试**：新role未加规则时必须失败。


## R15-INC-045｜Research candidate 被标成“ALPHA mineable”  **[P1]**

**涉及文件**：`factor_engine/scripts/audit_all_registered_operators.py`

**问题**：pending/candidate与production usable混淆。

**整改**：分 ALPHA_ELIGIBLE / ALPHA_PENDING_CERT / RESEARCH_TOOL。

**必须补的测试**：未认证算子不得出现在mineable/usable计数。


## R15-INC-046｜`mining usable total` 使用环境无关eligibility  **[P0]**

**涉及文件**：`factor_engine/scripts/audit_all_registered_operators.py`

**问题**：没有真实source/market/frequency不能称usable。

**整改**：报告static certified与per-context usable。

**必须补的测试**：分别输出A-share/US context。


## R15-INC-047｜UNCLASSIFIED只看authoring surface，漏掉role fallback误分类  **[P0]**

**涉及文件**：`factor_engine/scripts/audit_all_registered_operators.py`

**问题**：surface已分类不代表mining role正确。

**整改**：双invariant：authoring_tier_unclassified=0、mining_role_unresolved=0。

**必须补的测试**：无显式role但daily surface fixture第二项失败。


## R15-INC-048｜unused只扫research/legacy，可能漏dead public canonical  **[P1]**

**涉及文件**：`factor_engine/scripts/audit_all_registered_operators.py`

**问题**：internal/source-transform/普通surface里仍可能没有任何合法consumer。

**整改**：做DSL→canonical→role/lane→recipe/source consumer reachability graph。

**必须补的测试**：人工无consumer canonical被抓到。


## R15-INC-049｜Semantic duplicate invariant必须是真实数学/行为图  **[P0]**

**涉及文件**：`factor_engine/scripts/audit_all_registered_operators.py`

**问题**：不能继续硬编码[]或只看名字。

**整改**：canonicalized AST/ParamSchema/SemanticType + synthetic behavior hash识别候选重复。

**必须补的测试**：同核Wasserstein/alias候选必须命中。


## R15-INC-050｜Dead searchable parameter必须做动态injectivity  **[P0]**

**涉及文件**：`factor_engine/scripts/audit_all_registered_operators.py`

**问题**：metadata看不出kernel clamp/未读参数。

**整改**：每searchable scalar多点运行+静态parameter-use AST+relational grid。

**必须补的测试**：能抓MODWT level、KNN/PCA dead区。


## R15-INC-051｜`CERTIFIED_FACTOR_NOT_IN_MINING_MANIFEST` 必须真实比较当前context manifest  **[P0]**

**涉及文件**：`factor_engine/scripts/audit_all_registered_operators.py`

**问题**：不能写死[]。

**整改**：同一frozen snapshot/context生成manifest后set difference，按terminal/nonterminal合法lane比较。

**必须补的测试**：删一个eligible manifest entry invariant失败。


## R15-INC-052｜“all registered”报告scope必须包含/明确ResearchTool与supporting surface  **[P1]**

**涉及文件**：`factor_engine/scripts/audit_all_registered_operators.py`

**问题**：当前标题容易让人误以为整个FactorEngine都被审计。

**整改**：输出factor_dsl/research_tool/source_transform/internal_helper各自coverage。

**必须补的测试**：每registry set equality。


## R15-INC-053｜Manifest API与CLI默认admission不一致  **[P1]**

**涉及文件**：`factor_engine/scripts/export_mining_manifest.py`

**问题**：同名产物可能因调用路径不同而pending/all。

**整改**：统一显式默认；不同admission产物名/header区分。

**必须补的测试**：CLI与函数相同参数digest一致。


## R15-INC-054｜Mining manifest缺commit/evidence/context指纹  **[P0]**

**涉及文件**：`factor_engine/scripts/export_mining_manifest.py`

**问题**：无法判断是否当前代码和真实环境生成。

**整改**：header加入repo/evidence/registry/query fingerprint，加载时验证。

**必须补的测试**：HEAD变化后旧manifest明确stale。


## R15-INC-055｜Manifest blocker/action继承空字段  **[P0]**

**涉及文件**：`factor_engine/scripts/export_mining_manifest.py`

**问题**：上游不填但产物看似完整。

**整改**：复用AdmissionDecision；noneligible entry reason/action非空。

**必须补的测试**：随机100个noneligible entries reason完整。


## R15-INC-056｜`validate_cold_start()` 错把ALL registry canonical与任意manifest比较  **[P0]**

**涉及文件**：`factor_engine/scripts/export_mining_manifest.py`

**问题**：eligible/pending本来会排除internal/diagnostic/denied，粗集合比较语义矛盾。

**整改**：解析cold-start AST→resolve canonical→按manifest admission/role/position验证。

**必须补的测试**：eligible manifest无DENIED仍能通过；表达式引用DENIED失败。


## R15-INC-057｜Cold-start文档说public DSL allowlist，代码实际比registry canonical  **[P1]**

**涉及文件**：`factor_engine/scripts/export_mining_manifest.py`

**问题**：DSL names/aliases/canonical/internal不是同一集合。

**整改**：明确验证解析后的canonical与AST role，而非粗set。

**必须补的测试**：alias能正确resolve。


## R15-INC-058｜Cold-start独立调用可能面对partial registry  **[P0]**

**涉及文件**：`factor_engine/scripts/export_mining_manifest.py`

**问题**：未强制final load/freeze时结果依import顺序。

**整改**：入口使用FinalRegistrySnapshot。

**必须补的测试**：fresh interpreter调用与完整runtime一致。


## R15-INC-059｜Manifest未校验terminal_allowed与allowed_ast_positions一致  **[P0]**

**涉及文件**：`factor_engine/scripts/export_mining_manifest.py`

**问题**：当前catalog bug可直接写进manifest。

**整改**：terminal iff 'terminal'∈positions；supporting/denied positions=empty。

**必须补的测试**：遍历全部MiningRole。


## R15-INC-060｜Manifest未校验semantic duplicate与parameter grids  **[P0]**

**涉及文件**：`factor_engine/scripts/export_mining_manifest.py`

**问题**：reachable不等于搜索空间不重复。

**整改**：生成前消费duplicate graph与scalar param schema；旧duplicate只alias。

**必须补的测试**：active duplicate不能同时进manifest。


## R15-INC-061｜Cold-start/search manifest必须包含完整searchable scalar schema  **[P0]**

**涉及文件**：`factor_engine/scripts/export_mining_manifest.py`

**问题**：只有参数名不能防estimator knob/dead param。

**整改**：输出dtype/range/choices/default/role/search_grade/active_when/relational constraints。

**必须补的测试**：缺schema的searchable scalar阻断manifest。


# 第二部分：R12 Semantic Closure / Contract Layer 增量审计

> 本部分只记录上一份 Master Prompt 未单独覆盖、且在 R12/R15 当前代码中仍存在的新问题。不要把这些问题退化成“补注释”；必须把 machine contract、runtime enforcement、audit 和 tests 一起改完。

## R15-INC-062｜`closure_audit.py` 只扫描顶层 `cleaned_operators/*.py`，无法证明“全算子已审计” **[P0]**

**涉及文件**：`factor_engine/cleaned_operators/closure/closure_audit.py`

**问题**：源码扫描使用顶层 glob，而 FactorEngine 大量生产算子位于 `cleaned_operators/technical/`、`fundamental/`、`microstructure/`、`stateful/`、`relation/`、`cross_section/` 等子目录。当前 audit 即使全绿，也可能只是“顶层文件全绿”，不是全 registry/full source tree 全绿。

**整改**：
1. 使用 `Path.rglob("*.py")` 递归扫描整个 `cleaned_operators`；
2. 以 final registry canonical → implementation source file 的真实映射为主，不要只按文件名；
3. 对所有 backend implementation 也纳入扫描；
4. 生成 `scanned_files / skipped_files / scanned_canonicals / unscanned_canonicals` 四张表；
5. `UNSCANNED_REGISTERED_CANONICALS` 必须为 0，否则 CI hard fail。

**必须补的测试**：在嵌套目录注入一个故意违反 SameAxis/missing contract 的 dummy operator，证明 audit 能抓到。


## R15-INC-063｜Closure audit 的“跳过 helper 文件”策略会产生系统性 blind spot **[P0]**

**涉及文件**：`closure_audit.py`、所有共享 helper 模块

**问题**：很多最危险的语义错误恰恰发生在 helper（例如 axis align、rolling、session、missing、stateful helper）。如果按文件名或“非 register_operator 文件”跳过，operator 本体可以完全正确，但共享 helper 仍污染几百个算子。

**整改**：建立两类审计对象：`RegisteredOperatorImplementation` 与 `SharedSemanticHelper`。所有被生产算子 import 的 helper 都要进入 dependency graph；helper 的语义风险由调用 canonical 继承。

**测试**：修改共享 helper 的对齐语义，所有依赖 canonical 的 closure evidence 必须失效。


## R15-INC-064｜多输入算子识别仍依赖参数名启发式，不能作为 SameAxis 证明 **[P0]**

**涉及文件**：`closure_audit.py`

**问题**：参数叫 `x/y/target/source/f1/f2` 只是命名习惯，不等于它一定是 panel；反过来 `benchmark_ret/group_id/high_limit` 等也可能漏掉。启发式会既漏报又误报。

**整改**：从 `LogicalSignature / InputSemanticType / PanelParamSpec / BroadcastSpec` 获取 panel 参数集合。每个多 panel canonical 必须显式声明 `AxisRelation`: SAME_AXIS / SAME_DATE_DIFFERENT_GRAIN / DAILY_TO_MINUTE_BROADCAST / GROUP_KEY_PANEL / BENCHMARK_BROADCAST 等。

**测试**：随机重命名参数不影响审计结论。


## R15-INC-065｜`unused public operator` 的判定混淆 public、internal、source-transform、diagnostic **[P1]**

**涉及文件**：`closure_audit.py`、`mining/operator_catalog.py`

**问题**：不是 terminal alpha 不代表“unused”。SOURCE_TRANSFORM、CONDITION、GLOBAL_STATE、RECIPE_INTERNAL 都可能被搜索语法合法使用。

**整改**：unused 必须定义为：`registered && no legal consumer path && no alias/backcompat path && no recipe/internal dependency && no mining role`。必须由 dependency/grammar graph 推导，而不是 surface 名称。


## R15-INC-066｜Global/Group state 的存在不应成为 closure failure，非法 terminal 才应失败 **[P0]**

**问题**：同日全市场同值或组内同值是合法 regime state。真正的问题是被放进 stock-ranking terminal。

**整改**：把 invariant 改为：`GLOBAL_OR_GROUP_STATE_WITH_TERMINAL_POSITION == ∅`，不要要求 `GLOBAL_STATE == ∅`。


## R15-INC-067｜`same_session_usable=False` 与“字段未声明”不能用 truthiness 区分 **[P0]**

**问题**：False 本身是重要 contract；如果 audit 用 `if not metadata.same_session_usable` 判断缺失，会把“明确禁止同 session 使用”错误当成“没有声明”。

**整改**：metadata 使用三态 sentinel：UNDECLARED / TRUE / FALSE；所有 bool semantic field 同理。

**测试**：明确 False 必须通过“已声明”检查，同时进入 EOD-only gate。


## R15-INC-068｜要求每个算子硬填 MissingPolicy/WindowSemantics 会制造“假声明” **[P0]**

**问题**：为了让审计全绿而机械填枚举，比没有 contract 更危险。某些算子没有 rolling window；某些 multi-stage 算子需要组合 missing semantics，单枚举根本表达不了。

**整改**：只有语义适用时才要求声明；引入 `NOT_APPLICABLE`，并要求由实现/签名证明适用性。复杂算子允许 structured contract，不强塞一个 enum。


## R15-INC-069｜Regex source scan 无法可靠证明 PIT / SameAxis / missing topology **[P0]**

**问题**：寻找 `.reindex(`、`.dropna(`、`shift(-` 只能作为 lint hint；包装函数、别名、NumPy slice、helper 调用都会绕过。反过来合法 source-transform 也会误报。

**整改**：Regex lint 仅作为第一层。第二层必须有 AST/call-graph analyzer；第三层必须有动态 metamorphic tests（future randomization / axis permutation / gap injection）。任何单层都不能单独发 certification。


## R15-INC-070｜Closure audit 中 estimator-knob / dead-param 检查没有证明已经真正连接到最终 invariant **[P0]**

**整改**：每个 detector 必须返回 machine records，最终 invariant 从 detector 输出真实计算，禁止 placeholder empty list。增加 detector self-test：注入已知 dead param 必须让 CI 红。


## R15-INC-071｜`include_behavioral` 等审计参数若不改变输出属于 dead audit API **[P1]**

**问题**：审计工具本身也必须做 parameter injectivity，不能出现开关写在 CLI/API 但结果不变。

**整改**：给 audit API 自己加 dead-option tests；无用途参数删除。


## R15-INC-072｜Authoring surface 与 MiningRole 是两条正交轴，closure audit 仍有混用风险 **[P0]**

**整改**：最终记录至少分开：`authoring_tier`, `semantic_role`, `mining_role`, `lifecycle`, `certification`, `backend_capability`。任何一项不能从另一项推断。


## R15-INC-073｜审计器需要“变异测试 / mutation test”，否则无法证明自己真的能抓 bug **[P0]**

**整改**：CI 中自动对 fixture operators 注入：未来 shift、silent reindex、dropna reconnect、dead parameter、wrong unit、wrong terminal role、stale evidence、duplicate canonical。每一种必须让对应 detector fail。


## R15-INC-074｜Axis contract side-registry 是第二套可变真相源，存在 split-brain **[P0]**

**涉及文件**：`closure/axis_contract.py`

**问题**：运行时 metadata、registry catalog、closure side-registry 都能描述 axis semantics。加载顺序/后处理可造成不一致。

**整改**：注册阶段收集 declaration，`load_all/finalize` 后生成 immutable `ResolvedSemanticContract`；之后 runtime/audit/mining 都只读 frozen snapshot。禁止运行中随意 mutate。


## R15-INC-075｜`declare_axis_contract()` 必须拒绝未知 canonical / typo **[P0]**

**问题**：声明一个拼错的 canonical 如果只是写进 dict，不代表真实 operator 获得 contract。

**整改**：finalize 时要求 declaration key ⊆ registry canonical；orphan declaration hard fail。


## R15-INC-076｜“声明 SameAxis”不等于“runtime 真执行 SameAxis” **[P0]**

**整改**：在统一 `SeriesOperator.calculate()` wrapper 根据 structured AxisContract 自动执行检查，而不是要求每个 kernel 手写。手写检查只能 defense-in-depth。

**测试**：故意删除 kernel 内 `_align`，wrapper 仍要拒绝错位 panel。


## R15-INC-077｜`Index.equals()` 只能证明 label equality，不能证明完整 SemanticAxis **[P0]**

**问题**：相同 datetime labels 仍可能来自不同 timezone normalization、不同 calendar、不同 price basis、不同 universe/entity key。

**整改**：引入 `SemanticAxisFingerprint`：frequency / timezone / calendar / timestamp convention / entity type / universe vintage / price basis / adjustment basis。SameAxis 比较 fingerprint + labels。


## R15-INC-078｜Column equality 也不能证明 entity identity **[P0]**

**问题**：相同字符串列可以是 Symbol/Ticker/ShareholderId/IndustryId；需要 EntityType。

**整改**：PanelSchema 带 `entity_type`，多输入 contract声明允许的 entity relation。


## R15-INC-079｜Broadcast contract 需要成为 axis algebra，不应是例外 waiver **[P0]**

**整改**：`BroadcastSpec` 必须解析为输入 grain→输出 grain 的明确 mapping，包含日期映射、时区、instrument join、as-of rule、availability。禁止 `allow_panel_broadcast=True` 这类裸豁免。


## R15-INC-080｜MissingPolicy side-registry 的 broad `except Exception` 会吞掉 contract bug **[P0]**

**涉及文件**：`closure/missing_policy.py`

**问题**：如果函数内部主动 raise unknown-canonical KeyError，也可能被自己的 broad except 吞回 fallback，导致 typo/registry corruption 被隐藏。

**整改**：只捕获明确的 import/bootstrap exception；业务错误继续抛出。production 下 registry lookup error fail hard。


## R15-INC-081｜`replace=True` 允许加载后改写 missing contract，破坏 evidence 可复现性 **[P0]**

**整改**：finalize 前可声明一次；finalize 后 immutable。需要 override 时必须新 schema version + evidence invalidation，不能原地替换。


## R15-INC-082｜`PAIRWISE_VALID` 不能表达“保持物理时间轴但只在共同有限样本上估计” **[P0]**

**问题**：pairwise valid 有两类完全不同语义：普通 covariance 可对 paired observations；lag/path/embedding 若删除中间缺失则会改时间拓扑。

**整改**：把 value cohort 与 time topology 分开：`CohortPolicy` + `TimeTopologyPolicy`。例如 `PAIRWISE_VALID + PHYSICAL_AXIS_PRESERVED`。


## R15-INC-083｜`CARRY_STATE` 缺 freshness / max-gap 约束 **[P0]**

**问题**：carry 并不等于无限 carry；停牌/缺数几十天后沿用旧 state 会制造陈旧信号。

**整改**：所有 carry contract必须声明 `max_staleness` 或明确 `unbounded_by_definition`；输出可伴随 stale flag。


## R15-INC-084｜测试工具如果能 `clear_*_contracts()`，生产进程里会破坏全局语义 **[P1]**

**整改**：test reset 只存在 test-only registry instance；production global singleton 不暴露 clear API。


## R15-INC-085｜WindowSemantics side-registry 同样是可变第二真相源 **[P0]**

**涉及文件**：`closure/window_semantics.py`

**整改**：并入 frozen ResolvedSemanticContract；不再单独 mutable dict。


## R15-INC-086｜Window semantics 缺“当前行是否进入估计样本”这条正交维度 **[P0]**

**问题**：trailing window 可以 current-inclusive、strict-prior、current-as-query-only。一个 enum 很难同时表达。

**整改**：新增 `CurrentObservationRole = INCLUDED / EXCLUDED / QUERY_ONLY / EVENT_TRIGGER_ONLY`。


## R15-INC-087｜`SESSION_WINDOW` 混淆“一个 session 内的 bars”和“过去 N 个 sessions” **[P0]**

**整改**：拆为 `WITHIN_SESSION_GRID`、`TRAILING_COMPLETED_SESSIONS`、`SESSION_EVENT_CLOCK`。


## R15-INC-088｜`RECURSIVE_STATE` 不能表达 checkpoint/history/rewarm **[P0]**

**整改**：WindowSemantics 不承担 execution contract；统一引用 `ExecutionContract(state_model, chunking, checkpoint_schema, reset_policy)`。


## R15-INC-089｜MissingPolicy × WindowSemantics × CurrentRole 缺组合一致性检查 **[P0]**

**例子**：声明 `CONTIGUOUS_FULL_WINDOW` 却实现为 drop-valid；声明 `BREAK` 却 carry state；strict-prior 算子却用 current row 定 threshold。

**整改**：增加 cross-contract verifier + synthetic gap/current-row tests。


## R15-INC-090｜Closure package 应输出一个统一 `ResolvedSemanticContract`，而不是四套 API 分别查询 **[P0]**

**建议字段**：axis_relation, input_semantic_types, output_semantic_type, input/output_grain, availability, current_role, cohort_policy, time_topology, window_semantics, history_requirement, state_model, missing_policy, unit_algebra, market_support, source_requirements。


## R15-INC-091｜禁止 runtime 通过数据统计特征“猜 SemanticType” **[P0]**

**涉及文件示例**：`spectral_ext.py`、`extreme_tail.py` 等

**问题**：扫描数值路径判断“像不像价格/收益”既不可靠，又可能让 t 时点调用成功/失败取决于未来样本。

**整改**：SemanticType 必须来自 FieldSpec/IR；数值 heuristic 最多成为独立 data-quality diagnostic，且只能前缀因果地运行，不能决定 operator admission。


## R15-INC-092｜任何 whole-panel pre-scan 的 warning/error 都可能造成“调用成功与否看未来” **[P0]**

**整改**：audit 全库搜索 `np.any/np.all` 直接作用于完整 panel 后 raise/warn 的路径。domain check 必须逐 cell / 当前可见 prefix / source validation stage 完成；不能未来坏点使过去 factor 整体无法计算。


# 第三部分：本轮新增的逐算子 / 算子族数学与契约问题

## R15-INC-093｜`spectral_ext` 的 input-type heuristic 不能参与 admissibility **[P0]**

**涉及文件**：`factor_engine/cleaned_operators/spectral_ext.py`

**问题**：通过全序列均值/标准差/符号等判断“像 price level / return”不是类型系统；同一公式换样本区间可能改变合法性。

**整改**：删除对 operator admission 有影响的 numeric type inference；要求 IR 提供 `SemanticType`。若保留 heuristic，只输出 DQ warning record，不 raise、不改变历史输出。


## R15-INC-094｜Spectral family 必须把 `level / return / signed_signal` 语义做成不同输入 contract，而不是同一个 generic x **[P1]**

**整改**：明确哪些统计对 scale/translation invariant；非 invariant 算子必须限制输入类型或提供显式 transform recipe。


## R15-INC-095｜DMD 相对模态能量用 `exp(min(exponent,700))` 会扭曲 mode share **[P0]**

**涉及文件**：`factor_engine/cleaned_operators/dmd.py`

**问题**：多个 unstable mode 都被截到 700 后会变成近似相同巨大能量；真实相对排序丢失。

**整改**：所有 mode energy 在 log-domain 计算，使用 `logsumexp` 做归一化；只有最终需要绝对量时再安全 exponentiate。

**测试**：构造 growth rate 差 10/100/1000 的 modes，relative share 保持单调且不因 cap 碰撞。


## R15-INC-096｜DMD 共轭模态合并条件过弱，可能错误合并非共轭 eigenmodes **[P0]**

**问题**：只比较 `|lambda|` 与 `|arg(lambda)|` 不能证明两个复数互为共轭。

**整改**：判断 `lambda_j ≈ conj(lambda_i)`（复数距离 + tolerance），并同时比较 mode/eigenvector conjugacy；real modes 不参与配对。


## R15-INC-097｜DMD dominant frequency 在 real mode 主导时语义必须明确 **[P1]**

**问题**：零频/real eigenvalue 与周期性 mode 是不同对象。若 real mode 最大，返回 0 frequency 可能把“趋势/衰减”误读为周期极长。

**整改**：拆 `dominant_oscillatory_frequency` 与 `dominant_mode_growth_rate`；频率算子排除 near-real modes。


## R15-INC-098｜DMD 窗口 effective-rank/support 变化会让因子定义漂移 **[P1]**

**整改**：输出或 gate `effective_rank`、condition number、usable mode count；搜索时禁止同 canonical 在不同日期偷偷使用不同模型阶数而不记录。


## R15-INC-099｜Research-transform wavelet low-pass 的 output unit 不能统一写 `level` **[P1]**

**涉及文件**：`research_transform.py`

**整改**：low-pass output `same_as:x`；Mahalanobis `dimensionless`；normalized persistence/statistics按真实单位填。添加 unit algebra tests。


## R15-INC-100｜Signature Mahalanobis 把零方差维度强行 scale=1 是任意数值约定 **[P0]**

**问题**：一个历史上完全不变化的 signature dimension 不应通过 scale=1 人工获得权重。

**整改**：零方差维度从距离中剔除；若有效维度不足→NaN。或者使用明确 covariance shrinkage，但不能把 0 variance 偷换成 1。


## R15-INC-101｜Research-transform 所谓 covariance “shrinkage” 若只是固定 ridge loading，名称需诚实 **[P1]**

**整改**：要么实现 Ledoit-Wolf/OAS 等真正估计 shrinkage intensity，要么重命名 `ridge_regularized_covariance`，并把 ridge 作为固定 numerical policy。


## R15-INC-102｜Research-transform 隐藏固定 ridge 常数会改变因子 identity **[P1]**

**整改**：所有影响输出显著的 regularization 常数进入 metadata，标 `ESTIMATOR_RESOLUTION/NUMERICAL`，有版本号；不能散落 magic number。


## R15-INC-103｜Persistence birth/death dispersion 的 normalization 必须与 diagram cardinality 解耦 **[P1]**

**整改**：在不同 point-count/null paths 上做 N-sensitivity audit；必要时使用 normalized moments / per-feature averages，而不是随 persistence point 数机械变化。


## R15-INC-104｜`feature_geometry` beta-break score 的 unit 不应默认为 generic dimensionless **[P1]**

**涉及文件**：`feature_geometry.py`

**问题**：若 score 是 beta/covariance geometry 的差异，单位取决于是否先标准化/whiten。

**整改**：逐 canonical 写 unit derivation；distance on SPD after normalization 才可 dimensionless。


## R15-INC-105｜Feature-geometry 的 eigen-gap threshold 是 estimator policy，必须显式版本化 **[P1]**

**整改**：`eigengap_min` 不作为自由 alpha 参数，但必须 metadata 可见、evidence fingerprint 可见；改变阈值必须 invalidates evidence。


## R15-INC-106｜所谓 Higham nearest-SPD 若只是 eigenvalue clip + rescale，不应叫 Higham algorithm **[P1]**

**整改**：实现真正 nearest-correlation/Higham iteration，或诚实命名 `eigenvalue_clipped_spd_projection`。加入 reference numerical golden。


## R15-INC-107｜SPD projection 严重度未暴露，坏 covariance 可被“修成正常” **[P0]**

**整改**：计算 `projection_norm / original_norm`；超过阈值 fail closed，或输出 diagnostic。不能大幅修复仍照常出 factor。


## R15-INC-108｜Feature-geometry fixed feature order 需要 permutation invariance/identity contract **[P1]**

**整改**：若数学对象对 feature permutation 应 invariant，测试 permutation；若不 invariant，FeatureId 顺序必须进入 factor hash。


## R15-INC-109｜`gather_ext` group-JS divergence 在 ties 下 requested bins 与 effective bins 不一致 **[P1]**

**涉及文件**：`gather_ext.py`

**整改**：用 effective occupied bins/support 做 normalization 和 finite-sample gate；bins 属 estimator resolution。


## R15-INC-110｜Group-reference 算子必须声明 ex-self identity，而不是依靠 helper 约定 **[P0]**

**问题**：`exclude_group_from_reference`/peer reference 若没 machine contract，普通版本与 ex-self 版本容易混进同一搜索面。

**整改**：新增 `ReferenceCohortPolicy = INCLUDE_SELF / EX_SELF / EX_GROUP / BENCHMARK_EXTERNAL`。


## R15-INC-111｜Histogram-based gather operators 需要 topology/support audit **[P1]**

**整改**：离散字段 ties、极端 sparsity、不同 sample N 的 null baseline 必须测试；不足 support→NaN，不可靠 smoothing 输出稳定假值。


## R15-INC-112｜`ts_value_at_argextreme` 类算子有效样本太少时极值位置/值极不稳定 **[P1]**

**整改**：声明 min_effective_fraction/min_periods；current missing 不得回用 stale argextreme；tie policy进入 metadata。


## R15-INC-113｜Weighted-percentile 中 zero-weight observation 是否属于 support 必须统一 **[P1]**

**整改**：zero weight 不应参与 value support/tie/interpolation；negative weight invalid；effective positive-weight N 单独统计。


## R15-INC-114｜`weighted_moment_ext` 多元 ridge 在所有 regressors 常数时会退化成 intercept-only，却仍返回“多因子残差” **[P0]**

**涉及文件**：`weighted_moment_ext.py`

**整改**：检查 design numerical rank；rank < required predictors+intercept → NaN，或显式降维并返回不同 canonical，不能静默降模型。


## R15-INC-115｜Ridge λ 的有效强度随样本量/feature scale变化 **[P1]**

**整改**：先标准化 X；采用 `lambda * N` 或明确 objective convention；metadata写清。不同 window 下不应因为 N 变化机械改变 regularization strength。


## R15-INC-116｜回归型算子必须输出/内部 gate DOF、rank、condition number **[P1]**

**整改**：统一 RegressionFitDiagnostics；没有足够 DOF 或 condition number 过高→NaN。


## R15-INC-117｜EventInterval 文档开头仍残留“任意有限非零=event”语义，与 strict EventBool 实现冲突 **[P1]**

**涉及文件**：`event_interval.py`

**整改**：文档、metadata、InputSemanticType统一为 strict `{0,1,NaN}`；删除旧描述，避免 LLM/cold-start生成 -1/2 mark 输入。


## R15-INC-118｜`event_interval_memory/local_variation` 被错误声明为 `EVENT_COUNT_WINDOW` **[P0]**

**问题**：真实 kernel 用 trailing **raw row window**，只是区间由 events 定义，且还可以读 `max_pre_window_age` 个 pre-window rows。

**整改**：WindowSemantics=`TRAILING_ROWS_WITH_PREWINDOW_EVENT_CONTEXT`；HistoryFormula=`window - 1 + max_pre_window_age`（按真实 off-by-one校准）。不要声明 event-count history。


## R15-INC-119｜`event_fano_factor/excess` 被错误声明 `CONTIGUOUS_FULL_WINDOW` **[P0]**

**问题**：实现允许 unknown block 被排除，只要求至少 N 个 valid blocks，并不是整窗全连续可观测。

**整改**：声明 `BLOCKWISE_CENSORED_WINDOW`，block 包含 unknown→整 block invalid；min_valid_blocks machine field。


## R15-INC-120｜Fano 文档“至少2个block”与实现 `min_valid_blocks=5` 不一致 **[P1]**

**整改**：单一常量进入 metadata/ParamSpec/docs/tests。


## R15-INC-121｜Fano 缺编译期关系 `window >= block * min_valid_blocks` **[P0]**

**整改**：RelationalParamSpec；block是 estimator resolution/policy，不进入自由搜索。


## R15-INC-122｜Event interval pre-window age 默认等于 window，但历史规划必须消费这个默认值 **[P0]**

**整改**：HistoryRequirement 从 bound/default params 解析，不能只看参数名。


## R15-INC-123｜Intraday session 完整性使用 `set(minute_of_day)`，重复 bar 不会导致失败 **[P0]**

**涉及文件**：`intraday_session.py`

**整改**：比较有序 slot sequence / Counter；要求 observed_count == official_count 且无 duplicate timestamp/slot。


## R15-INC-124｜`_minute_of_day()` 把 `09:46:30` floor 成 `09:46`，与“stray second-level bar应失败”的注释矛盾 **[P0]**

**整改**：验证 timestamp 精确落在 calendar grid；秒/微秒必须为 convention 允许值，否则 invalid slot。


## R15-INC-125｜Intraday session 对 tz-aware index 使用 `to_numpy(datetime64[ns])` 后 timezone语义可能丢失 **[P0]**

**整改**：先用 explicit SessionCalendar timezone 做 localize/convert，再生成 trade_date/slot；禁止 numpy datetime 直接作为 timezone authority。


## R15-INC-126｜`history_days < min_history_sessions` 是 guaranteed-NaN dead region **[P0]**

**整改**：RelationalParamSpec `history_days >= min_history_sessions`。


## R15-INC-127｜PCA session `n_components` 与 history_days / canonical node dimension 缺关系约束 **[P0]**

**整改**：`n_components < min(history_days, n_nodes, effective_rank ceiling)`；search compile-time 先剪掉不可能组合。


## R15-INC-128｜`declare_stateful(history_count=20, minimum_history=5)` 与可搜索 `history_days` 不联动 **[P0]**

**问题**：用户取 history_days=60，但 execution planner 仍可能只按20 session规划。

**整改**：history_count 支持 callable(params)；两个 session operators按真实 `history_days/min_history_sessions` 解析。


## R15-INC-129｜Session-PCA rank gate `rank >= n_components+1` 需与实际投影维数数学一致 **[P1]**

**问题**：重建 top-k 子空间一般需要 rank>=k；如果额外 +1 是为了 eigengap/残差，需要明确理由和测试。

**整改**：拆 identifiability 与 eigen-gap gate，不用一个 +1 混合两件事。


## R15-INC-130｜`advanced_intraday._infer_session_tz()` 对 bare UTC 默认 Asia/Shanghai **[P0]**

**涉及文件**：`advanced_intraday.py`

**问题**：与新的 fail-closed session recovery policy 不一致；US UTC 数据会被错误映成 A 股交易日。

**整改**：统一 SessionCalendar authority；unknown UTC 必须显式 market/session_tz。


## R15-INC-131｜Naive minute index 被默认当“已是正确 session wall-clock”，仍缺 provenance **[P0]**

**整改**：naive index只有 SourceContract 声明 timezone semantics 时可用；否则生产 fail closed。


## R15-INC-132｜`intraday_wasserstein_pair_distance` 与 `baseline_scaled_wasserstein_distance` 是同一数学核的两个 public canonical **[P0]**

**整改**：保留 honest canonical `baseline_scaled_wasserstein_distance`；旧名只 alias/compatibility，不能同时进入 mining manifest/duplicate graph。


## R15-INC-133｜Intraday quantile-curve PCA 要求至少50个历史profiles，但 `window` 仍允许2..49 **[P0]**

**整改**：ParamSpec `window >= _PCA_MIN_HISTORY`；若这是固定 estimator requirement，则直接 min=50。


## R15-INC-134｜Quantile-curve PCA 的 `k` 缺上界/关系约束，存在 guaranteed-NaN 参数区 **[P0]**

**整改**：`1 <= k < min(window, profile_dim)` 并考虑 eigengap requirement。


## R15-INC-135｜PCA eigen-gap `_EIGEN_GAP_MIN=0.01` 是隐藏 estimator policy **[P1]**

**整改**：metadata可见、版本化、非搜索；改动必须使 evidence fingerprint 变化。


## R15-INC-136｜PCA sign orientation 在多个 loading 同绝对值时 tie policy不明确 **[P1]**

**整改**：定义 deterministic secondary key（最低 feature index等）并加入 permutation/reference tests。


## R15-INC-137｜Advanced intraday 多个 EOD factor 没有统一 full-session completeness gate **[P0]**

**问题**：仅按 calendar day group 可能在半日/截断日也产生分位曲线/Wasserstein等“全日”指标。

**整改**：凡 metadata `available_at=session_close` 且依赖 whole session，应通过统一 SessionGrid completeness service；半日若交易所官方 half-day需 calendar明确支持。


## R15-INC-138｜日内 kernel broad catch `ValueError/... -> NaN` 会隐藏 contract/programming bug **[P0]**

**整改**：只把明确 DataDomainError 变 NaN；参数错误、axis错误、calendar错误、代码bug必须继续抛出。


## R15-INC-139｜`session_event_recovery_score` 默认 `min_events=1` 与模块自己的统计支持声明冲突 **[P0]**

**涉及文件**：`session_recovery.py`

**整改**：默认至少3（或经模拟校准确定）；ParamSpec/default/docs一致。若单事件版本有研究价值，另建 descriptive canonical。


## R15-INC-140｜Recovery 的 unknown EventBool minute 被直接跳过，相当于“默认没有事件” **[P0]**

**整改**：若 unknown minute 位于可影响事件集合/前后 horizon，日结果 censored/NaN；不能 unknown→no-event。


## R15-INC-141｜Late-day right-censored shock 被直接删除，不是真正 survival-analysis 处理 **[P0]**

**问题**：complete-case 删除会选择性保留早盘/易观测事件并偏低恢复时间。

**整改**：二选一：
- 输出明确 `recovery_complete_case_median` 并同时有 censoring coverage gate；或
- 实现 Kaplan-Meier / restricted mean recovery time 等真正右删失 estimator。


## R15-INC-142｜Recovery 缺 effective_events / censored_events / coverage diagnostics **[P1]**

**整改**：至少内部计算并用于 gate；推荐暴露 diagnostic source-transform。


## R15-INC-143｜Recovery 也应使用官方 full-session grid，不是简单按 normalize 分日 **[P0]**

**整改**：缺一分钟、重复分钟、截断 session必须可识别。


## R15-INC-144｜Refractory 边界 `s <= suppress_until` 的定义需与“refractory N bars”严格约定 **[P1]**

**整改**：写成 exact bars excluded formula，测试 N=0/1/2；避免 off-by-one。


## R15-INC-145｜Hill estimator 允许 transformed threshold `u<0`，经典 Hill 数学域不成立 **[P0]**

**涉及文件**：`extreme_tail.py`

**问题**：只要 exceedance 与 u 同为负，`log(exc/u)` 虽数值可算，却不是经典正重尾 Hill estimator；甚至会得到负 ξ。

**整改**：Hill canonical只接受变换后 positive tail variable，要求 threshold>0 且 tail sample>0。lower signed-return tail可通过 loss magnitude `-x` 的正尾；positive-magnitude 的“lower tail”不属于 Hill，另建别的 estimator。


## R15-INC-146｜Hill 的 `side=lower` 对不同 SemanticType 不应一个公式通吃 **[P0]**

**整改**：SignedReturn 下 lower = upper on `-x`；PositiveMagnitude 下 lower tail 不使用 Hill；LossMagnitude 明确正域。类型系统编译时决定合法 side。


## R15-INC-147｜Hill metadata/docs仍存在“拒收价格水平”与“只warning”表述冲突 **[P1]**

**整改**：typed IR 是唯一 authoritative gate；doc明确 operator本身不凭数据猜类型。


## R15-INC-148｜Quantile regression beta 的最小3个pair统计支持过低 **[P1]**

**整改**：至少基于参数数/quantile tail support设置 min N；极端 q=0.05 需要远高于3。声明关系 `N * min(q,1-q) >= min_tail_support`。


## R15-INC-149｜GLR 文档称“不bounded”，实现却强制 LLR cap=40 **[P1]**

**涉及文件**：`glr_change.py`

**整改**：文档诚实写 saturated calibrated score，或拆 raw/capped 两个对象；不要一边说 unbounded 一边 saturation。


## R15-INC-150｜GLR cap 与 scan penalty 是隐藏 estimator definition，不应散落常量 **[P1]**

**整改**：做 `GLRCalibrationPreset`，版本化；固定不搜索，但进入 factor/evidence hash。


## R15-INC-151｜BIC-style `ln(m)` scan calibration 是 heuristic，应与 raw GLR identity 分离 **[P1]**

**整改**：公开名称应区分 `raw_max_glr`、`scan_penalized_glr_score`；不要让用户以为是标准原始 GLR。


## R15-INC-152｜GLR score saturation会丢失极强 break 的相对强度 **[P1]**

**整改**：如果保留 cap，明确是 robust state score；另可输出 capped_flag。搜索层不要把 saturated score当原始 likelihood strength。


## R15-INC-153｜GLR null calibration随有效 contiguous N变化，但 penalty按当前N自动变化，需做 null invariance测试 **[P1]**

**整改**：AR(1)/GARCH/heavy-tail null Monte Carlo across windows/effective N，确认均值/quantiles不会机械漂移。


## R15-INC-154｜`tail_systemic` 条件样本与未来响应样本 cohort identity 要严格绑定 **[P0]**

**涉及文件**：`tail_systemic.py`

**整改**：conditioning event at t 与 response t+h 必须同一个物理 observation pair；不能条件样本 finite、未来response缺失后偷偷改变分母。


## R15-INC-155｜Systemic peer aggregate 的 denominator 随 finite peers变化会产生 breadth factor **[P1]**

**整改**：输出/gate peer coverage；ex-self group/market response要求最小 breadth，并在跨日比较时控制 universe变化。


## R15-INC-156｜`relation_diffusion_score` 若使用 complete group graph，不应叫 relation network diffusion **[P1]**

**整改**：若没有真实 relation edges，重命名 group diffusion/peer diffusion；有真实 relation输入则显式接 adjacency/exposure panel。


## R15-INC-157｜Dependence CMI 文档仍说按 log(bins)归一，但实现使用 effective bins **[P1]**

**涉及文件**：`dependence_ext.py`

**整改**：统一 docs/metadata为 effective-support normalization。


## R15-INC-158｜CMI runtime要求 `N >= 2*bins^3`，但编译期缺 relational feasibility **[P0]**

**整改**：若 window是最大raw rows，至少声明 `window >= 2*bins^3`；若允许 missing还要更高 coverage gate。


## R15-INC-159｜CMI/HSIC 等依赖算子对 missing 的“drop aligned finite pairs”不等于可忽略时间拓扑 **[P1]**

**整改**：对于纯 iid-style dependence可用 paired cohort，但必须标 `TIME_ORDER_IRRELEVANT_WITHIN_WINDOW`；对 lagged dependence不能复用同 helper。


## R15-INC-160｜HSIC median-positive-distance bandwidth需要有效pair数量 diagnostic **[P1]**

**整改**：记录 unique support / positive-distance ratio；离散/tie严重时 NaN gate；不要只用 sqrt(N) heuristic。


## R15-INC-161｜Distance-correlation partial proxy若非标准 partial dCor，必须诚实名称/参考实现 **[P1]**

**涉及文件**：`dependence_ext.py`

**整改**：与 Székely/Rizzo 定义做 reference golden；若只是 residualized dCor proxy，名称写 proxy，不宣称标准 partial distance correlation。


## R15-INC-162｜`ts_modwt_band_corr(level > band)` 被 runtime clamp 成同一参数，但 metadata仍可搜索 **[P0]**

**涉及文件**：`conditional_dependence.py`

**问题**：`lv=min(lv,bd)` 使多个 AST 完全同输出。

**整改**：编译期 relation `level <= band`，运行时非法直接 raise；不要 silent clamp。若 level不影响输出则删除参数。


## R15-INC-163｜MODWT `level` / `band` 应归类 estimator resolution，而不是自由经济参数 **[P1]**

**整改**：给少量 reviewed wavelet bands；不让 AlphaProbe在 estimator grid上过拟合。


## R15-INC-164｜Conditional Transfer Entropy 的 `window-lag >= 3*bins^4` 只在runtime检查 **[P0]**

**整改**：RelationalParamSpec；搜索语法提前 prune。


## R15-INC-165｜CTE 的 `min_transitions` 是 estimator support policy，不是经济搜索参数 **[P1]**

**整改**：ParamRole=ESTIMATOR_RESOLUTION/POLICY，fixed reviewed default。


## R15-INC-166｜Wavelet/CTE 算子必须有外部 reference/golden，不仅 synthetic self-consistency **[P1]**

**整改**：MODWT 与 PyWavelets/reference implementation对齐；TE用小型离散过程已知方向性测试。


## R15-INC-167｜EDGE estimator “faithful reference”只证明单window核心，不等于rolling wrapper完整等价 **[P1]**

**涉及文件**：`ohlc_spread.py`

**整改**：golden覆盖完整 rolling wrapper：NaN、OHLC invalid、window edges、pair coverage、current-row semantics；不能只测 `_edge_window`。


## R15-INC-168｜EDGE `min_valid_pairs/min_valid_ratio` 属 support policy，不是 NUMERICAL **[P1]**

**整改**：新增 SUPPORT_POLICY/ESTIMATOR_SUPPORT ParamRole，避免把统计可靠性门槛误当epsilon数值稳定参数。


## R15-INC-169｜Pastor–Stambaugh `flow_scale` 是单位变换 policy，不是 NUMERICAL epsilon **[P1]**

**整改**：ParamRole=UNIT_POLICY；factor identity对合法单位归一后应 canonical equivalence，避免元/万元/百万元生成不同经济因子。


## R15-INC-170｜Spread family 每个 canonical 的 missing/window policy不能共享一个笼统模块声明 **[P1]**

**整改**：EDGE pairwise valid、Abdi completed pairs、PS regression pair cohort分别声明。


## R15-INC-171｜`cross_section_local` KNN residual默认 `k=10` 与内部 `min_neigh=20` 冲突，默认几乎全NaN **[P0]**

**涉及文件**：`cross_section_local.py`

**整改**：要么默认 `k>=20`；要么邻域选择目标从 k 变成 `max(k,min_neigh)` 并把真实neighborhood size写入identity；更推荐 ParamSpec min=20 并删除运行时隐式扩大。


## R15-INC-172｜KNN gradient 与 residual 共用 k 参数，但最小DOF必须 machine-declared **[P0]**

**整改**：Relational/support rule由 feature dimension d 动态推导，不在 kernel 内 magic `5*(d+1)`。


## R15-INC-173｜KNN `ridge` 默认可搜索会把局部模型正则强度变成过拟合维度 **[P1]**

**整改**：ridge=ESTIMATOR_RESOLUTION；使用极少 reviewed grid或固定；标准化后定义 objective。


## R15-INC-174｜Rank-copula MI 的 Miller–Madow correction 符号疑似反了 **[P0]**

**问题**：若 entropy correction采用 `H_MM=H_ML+(K-1)/(2N)`，则 `I_MM=H_X_MM+H_Y_MM-H_XY_MM`，修正项应 `(Kx + Ky - Kxy - 1)/(2N)`，当前代码加的是相反方向。

**整改**：重新推导并用独立 reference验证；不能凭注释自证。


## R15-INC-175｜Jeffreys smoothing + Miller–Madow correction 混用需要统计定义证明 **[P1]**

**整改**：二选一并明确 estimator：Bayesian-smoothed MI 或 plug-in+MM；不要把不同 bias correction叠加后仍叫标准 MI。


## R15-INC-176｜Rank-copula GLOBAL_STATE 目前只靠 tag，MiningRole层未可靠读取 **[P0]**

**整改**：metadata explicit `output_scope=GLOBAL_STATE`，final registry materialize到catalog role；禁止 tag-only语义。


## R15-INC-177｜Tangent-plane固定 `tangent_dim=2` 是 model-order policy，不能隐藏 **[P1]**

**整改**：显式 metadata固定值或参数（MODEL_ORDER coarse grid），并关系约束 `tangent_dim < feature_dim`。


## R15-INC-178｜Cross-section local scale加 `_EPS` 会把近零尺度转成超大有限值 **[P1]**

**整改**：scale <= tolerance直接NaN，epsilon只用于浮点比较，不进经济分母。


## R15-INC-179｜`group_current_members_tail_coexceedance` ParamSpec quantile=[0.5,0.99] 与 runtime `(0,1)` 不一致 **[P0]**

**涉及文件**：`cross_section_ext.py`

**整改**：统一语义。若参数表示 upper quantile level，upper应 `[0.5,0.99]`；lower不要再使用 `1-q`混淆，建议改 `tail_fraction∈(0,0.5]` + side。


## R15-INC-180｜Tail-coexceedance 实际输出是 `p_ij - p_i p_j`，不是“共同超越概率密度” **[P1]**

**整改**：重命名 `group_current_members_tail_dependence_excess`；旧名 alias。输出可正可负，不要文档写 probability。


## R15-INC-181｜Group tail / MST 都是组内广播状态，必须归 `GROUP_STATE`，不能默认 ALPHA terminal **[P0]**

**整改**：explicit output scope；只允许 gate/interaction，不能做组内股票rank terminal。


## R15-INC-182｜`group_corr_mst_length` 使用今日 membership 回看历史，应在 factor identity包含 MembershipVintagePolicy **[P1]**

**整改**：`CURRENT_MEMBERS_RETROSPECTIVE` 与 `HISTORICAL_CONTEMPORANEOUS` 是不同 canonical/semantic parameter，不靠 docstring。


## R15-INC-183｜Isotonic residual output unit必须 `same_as:y` **[P1]**

**整改**：unit algebra写清；generic `residual` 不是单位。


## R15-INC-184｜Lagged-direction isotonic把过去多日所有股票flatten成一个样本，可能被横截面breadth/重复stock-day权重支配 **[P1]**

**整改**：明确 estimator是pooled-stock-day Spearman还是daily-Spearman time average。二者不同；建议先每日rho，再对过去日期 robust aggregate，避免大截面日期权重更大。


## R15-INC-185｜`cs_knn_local_moran` tie-inclusive neighbors可能远超k，k因此不是实际邻居数 **[P1]**

**整改**：将参数定义为 kth-distance radius policy，输出/记录 effective_neighbor_count；若希望exact-k需 deterministic tie weighting而非任意截断。


## R15-INC-186｜`volume_clock` 聚合先 `concat(...).dropna()` 会删除missing bar并压缩物理session轴 **[P0]**

**涉及文件**：`volume_clock.py`

**问题**：kernel声称 NaN price/activity fail whole path，但 wrapper先dropna，使这些NaN根本看不到，直接把缺失分钟连接起来。

**整改**：禁止 dropna；按官方 SessionGrid对齐，任何 required field missing按 contract fail day，除非明确允许 paired-censor且时间位置保留。


## R15-INC-187｜Volume-clock day grouping仍直接 `index.normalize()`，未统一 SessionCalendar timezone **[P0]**

**整改**：使用统一 SessionContext；US/跨UTC日期不能错分。


## R15-INC-188｜Volume-clock `open` panel与price/activity的axis没有显式SameAxis/broadcast contract **[P0]**

**整改**：open若minute panel SameAxis；若daily session open，声明 daily→minute/session broadcast。不能 concat时自动对齐。


## R15-INC-189｜Leading zero-activity bars允许任意price而“不存在previous observable”，可能使session open→first trade路径跳变被忽略 **[P1]**

**整改**：如果有 official open，所有 leading zero-activity bar price必须与open一致或按明确 quote/no-trade policy；否则 fail closed。


## R15-INC-190｜Volume-clock要求 `Q0 points >= buckets+1` 但 buckets choices与分钟有效点数关系只在runtime **[P1]**

**整改**：daily effective-support gate并输出 coverage；不能编译时保证，但 admission evidence要用真实数据统计 default coverage。


## R15-INC-191｜Volume-clock linear interpolation本身会制造未成交activity位置的价格，roughness需区分observed vs interpolated geometry **[P1]**

**整改**：验证 buckets分辨率对结果稳定；roughness对插值节点过敏时改用 activity-weighted observed increments或固定 coarse grid。


## R15-INC-192｜Session recovery / volume clock / advanced intraday各自维护timezone/session helper，存在语义漂移 **[P0]**

**整改**：删除模块私有 timezone infer/grouping authority，统一 `runtime.SessionContext/SessionGridService`。


## R15-INC-193｜`activity_clock` current-inclusive age/value与prior版本需要在MiningRole/grammar上区分用途 **[P1]**

**问题**：current-inclusive可做state/measurement；构造predictive momentum应只用prior。不能让算法随意把current-inclusive lagged value当严格滞后。

**整改**：CurrentObservationRole写入manifest；recipe grammar针对 predictive lag只允许 PRIOR variant。


## R15-INC-194｜`safe_ops.ts_coverage_ratio` denominator固定window，warmup期会机械偏低 **[P1]**

**涉及文件**：`safe_ops.py`

**问题**：min_periods允许1时第一天 coverage=1/window，而不是“当前可观察slot coverage=1”。两种指标都合理但语义不同。

**整改**：拆 `coverage_vs_full_window` 与 `coverage_vs_observed_slots`；默认production warmup最好full-window后才输出。


## R15-INC-195｜`ts_valid_count` 的 min_periods是slot gate还是finite-value gate必须明确 **[P1]**

**整改**：不要沿用pandas `.count()`的隐含定义；Machine contract写 `minimum_window_slots` 与 `minimum_finite_observations` 两个字段。


## R15-INC-196｜`pd_ts_staleness` 存在未使用 `last_seen` ring变量，说明实现/设计已漂移 **[P2]**

**整改**：删除dead code；增加 complexity benchmark，避免每row重新scan O(NW) 若可用stateful O(N)。


## R15-INC-197｜`cs_coverage_ratio` denominator用物理列宽，会把未上市/不在universe股票当缺失 **[P1]**

**整改**：明确 physical-panel coverage 与 universe coverage不同 canonical；不要一个名字承担两种语义。


## R15-INC-198｜`group_impute_median` 仍是数据清洗，不应进入因子搜索表达式 **[P0]**

**整改**：Role=SOURCE_TRANSFORM/RECIPE_INTERNAL；只有source policy明确允许时使用。不能让AlphaProbe通过填补规则制造alpha。


## R15-INC-199｜`ts_ffill_limited(lineage=None)` 默认允许填充仍是 fail-open **[P0]**

**问题**：None被解释为“调用者断言这是price/level”，但自动挖掘算法不会真的做这个断言。

**整改**：lineage必须由SemanticType自动提供；UNKNOWN lineage默认禁止ffill。不能靠None默许。


## R15-INC-200｜FFILL lineage policy缺 `state/asof_level/slow_moving_attribute` 等合法类型 **[P1]**

**整改**：不要只price=True、其他全False；source-level FieldSpec定义可否 carry及max_gap。Fundamental as-of state与“缺失fundamental observation”也不是一回事。


## R15-INC-201｜`group_valid_count` 对 group key equality需统一missing-key与object semantics **[P1]**

**整改**：复用唯一 `GroupKey` normalization，不在不同模块各写 `g is None or pd.isna(g)`。


## R15-INC-202｜`composition.py` 把 revenue/net_income/assets/liabilities/equity 都放进同一 `financial_statement` family，经济上并不构成同一 composition **[P0]**

**问题**：同币种、同公司不等于同一个 part-whole。收入、利润、资产负债表余额之间不存在“组成同一整体并closure”的自然关系。

**整改**：CompositionSchema必须定义真实 `whole_id` + part ids，例如资产结构（cash/inventory/PPE/.../total assets）或营收分部结构；不能把所有财务科目作为一个family。


## R15-INC-203｜Unknown composition field当前默认放行，违背“explicit CompositionSchema”目标 **[P0]**

**整改**：生产模式 unknown part fail closed；只有调用者提供显式 schema object并通过unit/whole/part validation时可用。


## R15-INC-204｜Composition field name从单列DataFrame column推断，会把instrument code误当field name **[P1]**

**整改**：field identity来自PanelSchema/FieldSpec，不从DataFrame columns猜。


## R15-INC-205｜Composition zero-policy只有reject，结构性零的CoDa场景被完全排除但未提供替代路径 **[P2]**

**整改**：如果确实需要结构性零 composition，提供 source-transform multiplicative replacement并明确 sensitivity；不要在factor operator里偷偷加epsilon。


## R15-INC-206｜Composition entropy/JS 的 part-count变化会改变量纲/范围，PartSet必须进入identity **[P1]**

**整改**：可选parts(None)意味着不同AST参数组合产生不同维数；hash必须含exact ordered PartIds，normalized entropy按effective declared P而非present finite P。


## R15-INC-207｜RQA fixed epsilon按MAD×sqrt(dim)仍未校准维度/自相关下 recurrence rate可比性 **[P1]**

**涉及文件**：`recurrence_analysis.py`

**整改**：synthetic AR/sine/noise across dim/delay做null/benchmark；若跨dim不可比，将dim视MODEL_ORDER并限制比较，不宣称统一尺度。


## R15-INC-208｜RQA diagonal entropy normalization support公式需要独立reference验证 **[P1]**

**整改**：用已知 recurrence matrices直接 golden，不只从path端到端测试。


## R15-INC-209｜RQA trapping/divergence对 finite-window边界截断线段存在右删失偏差 **[P1]**

**整改**：边界触及的line标记censored；决定排除/生存校正，不把截断长度当完整line。


## R15-INC-210｜RQA minimum effective fraction=0.8是隐藏support policy **[P1]**

**整改**：metadata显式且non-search，修改需invalidate evidence。


## R15-INC-211｜Cross-sectional/global broadcast输出需要 `output_scope` 一等字段，而不是靠检测“同一行是否同值” **[P0]**

**整改**：作者声明 + audit synthetic verification双重证明；所有 group/global family迁移。


## R15-INC-212｜Group feature spectrum的breadth history以 group label为key，group taxonomy变化/label复用可能污染历史 **[P1]**

**整改**：GroupSchema/version进入key；industry classification version变化应reset或PIT mapping。


## R15-INC-213｜Group spectrum breadth gate自身是recursive state但未必进入ExecutionContract **[P1]**

**问题**：trailing deque影响当日是否输出；分块执行如果不携带breadth history会不一致。

**整改**：要么把 gate改成显式rolling panel统计（有限history），要么声明state/checkpoint。


## R15-INC-214｜Group spectrum localization eigengap阈值0.10是隐藏policy **[P1]**

**整改**：metadata版本化、non-search、reference stability tests。


## R15-INC-215｜Group spectrum的“robust zscore”对每个feature单独尺度后做SVD，得到的并非 correlation matrix spectrum，命名要精确 **[P2]**

**整改**：说明是 standardized feature matrix singular spectrum；不要在文档中泛称within-group correlation spectrum除非数学等价条件满足。


## R15-INC-216｜`cs_hartigan_dip` 输出 `sqrt(N)*dip` 后不再是原始dip statistic **[P1]**

**整改**：名称应 `cs_hartigan_dip_sqrtn_scaled` 或新增 raw dip；不能文档/用户以为输出标准dip。


## R15-INC-217｜Hartigan dip GLOBAL_STATE broadcast也必须显式 role，不靠tag/名称 **[P0]**

**整改**：output_scope=GLOBAL_STATE；terminal禁止。


## R15-INC-218｜Group Wasserstein barycenter使用current-members-retrospective membership policy应机器化 **[P1]**

**整改**：同R15-INC-182，MembershipVintagePolicy进入contract/hash。


## R15-INC-219｜Distribution-break多特征stack需要统一SameAxis wrapper，即使base wrapper已有也要在final contract证明 **[P0]**

**整改**：所有 `np.stack([f.to_numpy...])` 的多panel operator自动从logical signature生成 SameAxis check；source scanner发现裸stack且无AxisContract→fail。


## R15-INC-220｜`ts_copula_central_asymmetry` grid属于estimator resolution且需要ParamSpec/有效support关系 **[P1]**

**整改**：grid少量固定值、非自由搜索；要求 N/grid² 最小倍数；normalized output做finite-sample null calibration。


## R15-INC-221｜`MiningRole` 测试当前把“所有非 NON_TERMINAL role 都 terminal_allowed=True”钉死，反而固化错误 **[P0]**

**涉及文件**：`tests/operators/test_mining_admission.py`

**整改**：定义正向 terminal allowlist，仅 ALPHA / ALPHA_HIGH_COST / INTRADAY_EOD / FUNDAMENTAL_PIT 等真正 factor value role可terminal；INTERNAL/DIAGNOSTIC/RESEARCH/LEGACY/DENIED必须False。


## R15-INC-222｜`test_every_registered_canonical_has_a_role` 只检查 role非None，抓不到 default-ALPHA fail-open **[P0]**

**整改**：要求每个canonical role有 `role_source=explicit|verified_rule`；fallback/heuristic role单列并在production=0。


## R15-INC-223｜Mining filter tests完全没有覆盖 `market / target_frequency / available_sources` 的组合矩阵 **[P0]**

**整改**：A股/美股 × daily/minute × daily-only/minute-enabled/fundamental-enabled source capability做参数化测试。


## R15-INC-224｜测试没有检查“连续 event statistic不应被 EVENT mask role误分类” **[P0]**

**整改**：至少钉死 `event_fano_factor/event_interval_memory/event_local_variation` 等应为ALPHA/STATE_VALUE而非 EventBool mask。


## R15-INC-225｜测试没有检查 supporting roles 的 `allowed_ast_positions=()` **[P0]**

**整改**：INTERNAL/DIAGNOSTIC/RESEARCH/LEGACY/DENIED全部positions空。


## R15-INC-226｜Current factor operator evidence artifact SHA 落后于当前HEAD，必须 fail closed **[P0]**

**涉及文件**：`evidence/factor_operator_verified.json` 及 evidence loaders

**当前观察**：artifact记录 commit SHA `7293...`，当前审计HEAD为 `9aaadb8...`。

**整改**：任何 production certification artifact 的 code fingerprint/commit fingerprint与final tree不匹配 → certification false；不能“文件存在就算证据”。


## R15-INC-227｜Evidence 不能只绑定 git commit，还应绑定 operator implementation fingerprint **[P1]**

**原因**：monorepo无关文件变化不应全量失效；反过来dirty tree/生成overlay也不能被commit SHA掩盖。

**整改**：每canonical evidence记录 implementation AST hash + semantic contract hash + dependency helper hashes + runtime versions。


## R15-INC-228｜Shared helper变化必须级联失效所有依赖 operator evidence **[P0]**

**整改**：构建 implementation dependency DAG；helper fingerprint作为operator evidence fingerprint组成部分。


## R15-INC-229｜Backend evidence必须区分 native backend 与 pandas bridge/UDF wrapper **[P0]**

**涉及文件**：Polars bridges、backend_meta、admission matrix

**问题**：桥接后端“能运行”不等于 Polars-native 性能/语义独立实现。

**整改**：`backend_kind=NATIVE|BRIDGE|UDF_WRAPPER|SQL_NATIVE`；生产至少一后端可用即可，但性能路由/fastpath不能把bridge当native。


## R15-INC-230｜`polars_batch_mirror` 仍注册已被治理层移出的 research/stat tests，容易重新污染backend capability **[P1]**

**整改**：桥接注册必须先检查 final canonical lifecycle/role；ResearchToolRegistry对象不要重新以Factor Operator backend形式出现。


## R15-INC-231｜Polars bridge metadata常常丢失原canonical完整 logical signature/ParamSpec/units **[P0]**

**整改**：backend implementation不得自建简化 metadata；所有backend引用同一个 CanonicalOperatorSpec，backend只提供execution implementation。


## R15-INC-232｜同canonical不同backend的 metadata drift必须成为CI invariant **[P0]**

**字段**：param_names/defaults/types/units/semantic types/grain/missing/current/history。任何drift hard fail。


## R15-INC-233｜`api/mining_integration.py` 的传统 DSL allowlist 与新 mining catalog 是两套入口 **[P0]**

**问题**：一个formula可被DSL parser接受，但不一定在mining manifest可生成；反之mining catalog operator也可能不在目标surface parser allowlist。

**整改**：定义清晰集合关系：`MiningCallable ⊆ ProductionDSLCallable ⊆ RegisteredCanonical`，并按market/frequency/context求交集。


## R15-INC-234｜Mining API还缺 AST role-position validator **[P0]**

**整改**：parse/lower后逐节点检查 operator role 与当前位置：ConditionBool slot、gate slot、terminal slot、source-transform禁用等。不能仅检查“名字在allowlist”。


## R15-INC-235｜Production DSL validator与mining validator必须共享同一 final registry snapshot **[P0]**

**整改**：禁止parser validation前后 bootstrap不同状态；一次请求绑定同一个 RegistrySnapshotId。


## R15-INC-236｜Market-specific DSL policy与新 `get_mining_operators(market=...)` 当前没有真正接通 **[P0]**

**整改**：market capability从统一 `MarketCapabilityProfile`读取，DSL/mining/source validation共用。


## R15-INC-237｜US snapshot-only valuation语义不能只留在 data-source config notes **[P0]**

**整改**：SourceContract machine字段 `historical_availability=SNAPSHOT_ONLY`；任何历史mining query自动禁止这些field/canonical组合。


## R15-INC-238｜A股 status asof_backward 与 constituent exact join策略必须进入source PIT evidence **[P1]**

**整改**：每个 source join mode写 Availability/PIT rule；同field不同join semantics必须有不同 source fingerprint。


## R15-INC-239｜`operator_policy.py` 与新 closure MissingPolicy/WindowSemantics形成双权威 **[P0]**

**整改**：旧OperatorPolicy迁移成 ResolvedSemanticContract的view/compat adapter，不再独立推断 nan_policy/lookback。


## R15-INC-240｜`infer_operator_policy()` 名称本身说明仍在“推断”关键production contract **[P0]**

**整改**：production operator必须显式/可机械推导，无UNKNOWN。heuristic infer仅research diagnostics。


## R15-INC-241｜`NanPolicy = propagate/ignore/zero/ffill_only` 词汇过粗，无法表达现代missing semantics **[P1]**

**整改**：旧字段deprecated；映射到 CohortPolicy + TimeTopology + CurrentMissing + StateGapPolicy。


## R15-INC-242｜`Scope` 枚举与 MiningRole/output_scope 重复但不一致 **[P1]**

**整改**：统一 `ComputationScope = ELEMENTWISE/TS/CS/GROUP/GLOBAL/SESSION/FISCAL/SOURCE_TRANSFORM`；MiningRole只描述search grammar职责。


## R15-INC-243｜`RESEARCH_CORE_CANONICALS` 等手工集合继续形成隐藏治理层 **[P1]**

**整改**：final lifecycle/role从CanonicalSpec生成；legacy集合只作为迁移测试，最终删除。


## R15-INC-244｜`INTENTIONALLY_PANDAS_ONLY` 把“不移植”和“非PIT安全”混成一件事 **[P1]**

**整改**：BackendPortability与SemanticSafety正交；Pandas-only可以完全production-safe，高风险数学也可以有Polars实现。拆字段。


## R15-INC-245｜Admission matrix中的 blocker vocabulary混合“不能用”与“可以用但成本高/角色特殊” **[P0]**

**整改**：拆 `HardBlocker`, `RoutingConstraint`, `CostConstraint`, `DataDependency`, `QualityWarning`。B20/21/22/23不能与PIT/math blocker同层。


## R15-INC-246｜`recommended_action` 应根据 blocker graph生成有序动作，不是单字符串模板 **[P1]**

**整改**：输出 `actions:[{code,priority,depends_on,fix,verification}]`；多个blocker按依赖排序。


## R15-INC-247｜Admission matrix `source_available` 在无context时默认全可用，报告会给出虚假可执行感 **[P0]**

**整改**：无context→UNKNOWN，不是True；planning artifact标 `source_status=UNKNOWN`。


## R15-INC-248｜`production_eligible/default_mining_eligible` 不应成为无context的固定bool **[P0]**

**整改**：分 intrinsic eligibility 与 contextual eligibility；后者需要 market/source/frequency/universe/session context。


## R15-INC-249｜Admission record `recursive=stateful` 是错误等价 **[P1]**

**整改**：state_model显式 recursive/episode/session_state/bounded_history/stateless；不能所有stateful都标recursive。


## R15-INC-250｜Stateless finite-window operator本来就支持chunk+warmup，`incremental_supported=checkpoint` 会误报False **[P0]**

**整改**：IncrementalCapability分 `WINDOW_REPLAY`, `CHECKPOINT`, `FULL_REPLAY`, `UNSUPPORTED`。


## R15-INC-251｜Admission matrix `current_row_semantics` 不应把bool stringify成语义 **[P1]**

**整改**：用 CurrentObservationRole枚举，不是 `True/False/""`。


## R15-INC-252｜Admission matrix `window_semantics = bounded/full_history` 过度压缩 **[P0]**

**整改**：直接序列化 ResolvedSemanticContract.window/history，不自己重建一个粗糙版本。


## R15-INC-253｜B24–B32 目前只是词表，不等于有真实 detector **[P0]**

**整改**：每个blocker code必须有 detector function + evidence source + mutation test；没有 detector 的code不能出现在“已审计完”报告。


## R15-INC-254｜Admission matrix跳过ResearchToolRegistry导致“全系统算子”报告其实不全 **[P1]**

**整改**：主表可分 FactorOperator/ResearchTool/SourceTransform 三类，但coverage report必须三类全部列出；Mining manifest只筛FactorOperator。


## R15-INC-255｜Admission matrix CLI无fail threshold，生成错误报告仍返回0 **[P0]**

**整改**：`--strict`默认production严格；存在P0/invariant failure return nonzero。


## R15-INC-256｜所有审计产物必须带HEAD SHA、dirty status、registry fingerprint、evidence fingerprint **[P0]**

**整改**：防止并行AI修完后拿旧报告当新报告。


## R15-INC-257｜`audit_all_registered_operators.py` 三个关键 invariant 仍硬编码空列表 **[P0]**

**涉及项**：`CERTIFIED_FACTOR_NOT_IN_MINING_MANIFEST`, `SEMANTIC_DUPLICATE_CANONICALS`, `DEAD_SEARCHABLE_PARAMS`。

**整改**：真正执行 detector，禁止placeholder。


## R15-INC-258｜Promote/delete audit `main()` 无论失败都返回0 **[P0]**

**整改**：任何hard invariant非空→exit 1；CI必须实际调用CLI而非只import函数。


## R15-INC-259｜Promote/delete audit 的“unknown fallback”如果默认归alpha会掩盖未分类 **[P0]**

**整改**：未知/冲突role进入 `UNRESOLVED`，production hard fail；永不默认ALPHA。


## R15-INC-260｜`FACTOR_SHAPED_RESEARCH_ONLY==∅` 不能靠“把所有research改名”满足 **[P0]**

**整改**：每个研究factor必须明确：promote to high-cost / diagnostic/internal / delete / awaiting real data。分类要有数学/数据依据。


## R15-INC-261｜“unused”审计如果只看research/legacy集合会漏 daily/extended dead canonical **[P0]**

**整改**：dependency graph覆盖全部 registered FactorOperator。


## R15-INC-262｜Semantic duplicate detector必须是真实数学/行为图，不是人工pair名单 **[P0]**

**整改**：组合 static normalized AST fingerprint + randomized metamorphic equivalence + alias graph；发现高相似pair供人工确认。


## R15-INC-263｜Dead searchable param detector必须做 parameter injectivity动态测试 **[P0]**

**整改**：从合法参数grid采样，固定输入，改变单一参数；若输出/metadata/history完全不变则dead candidate。注意conditional active_when。


## R15-INC-264｜Evidence stale时不能让“之前通过测试”继续撑住 production_certified **[P0]**

**整改**：certification loader先fingerprint match再overlay；stale evidence单独状态STALE，不是PASSED。


## R15-INC-265｜当前R12计划中“六证全False待重生成”说明准入层与evidence生命周期仍未闭环 **[P0]**

**整改**：生成/修改CanonicalSpec后自动列出 invalidated evidence subset；稳定后一次性重证。不能手工记恢复顺序。


## R15-INC-266｜并行AI同时改 registry/side-registry/evidence 时需要 deterministic finalization **[P0]**

**整改**：所有源码声明加载完→single finalize→freeze→certify；任何后finalize monkey patch使snapshot dirty并拒绝certification。


## R15-INC-267｜`load_all()` 重复调用必须 idempotent，不能重复declare/alias/role mutate **[P0]**

**测试**：同进程 load_all 1/2/5次，registry snapshot hash完全相同。


## R15-INC-268｜import order必须不影响最终catalog **[P0]**

**测试**：随机module import permutations + load_all，canonical/metadata/role/hash一致。


## R15-INC-269｜Final registry必须检测 alias cycle / alias-to-alias ambiguity **[P0]**

**整改**：alias graph DAG；每alias resolve exactly one canonical；canonical自身不能被alias覆盖。


## R15-INC-270｜Canonical rename兼容不能让旧名继续作为独立 backend registration **[P0]**

**整改**：alias只存在 parser/registry alias map；backend map keyed canonical，避免一数学对象两身份。


## R15-INC-271｜Operator logical signature必须成为single authority，不能由各backend class各写param_names **[P0]**

**整改**：CanonicalOperatorSpec持signature；backend implementation只能绑定kernel，不重复声明参数ABI。


## R15-INC-272｜Default value drift跨backend/DSL/analyzer必须自动检查 **[P0]**

**测试**：inspect kernel signature、CanonicalSpec default、parser default、manifest default完全一致。


## R15-INC-273｜`**kwargs` 吞掉未知参数会让typo静默无效 **[P0]**

**整改**：central validator拒绝任何未声明kwarg；kernel `**_`仅接受runtime内部reserved args。


## R15-INC-274｜Scalar与Panel同名参数的类型重载必须禁止隐式猜测 **[P0]**

**整改**：LogicalSignature明确 `Panel[T] | Scalar[T]`；没有声明union就不接受另一类型。


## R15-INC-275｜Factor hash必须包含 semantic contract version，不仅canonical+params **[P0]**

**原因**：同名kernel修数学/改missing/session policy后，旧factor cache不能继续命中。


## R15-INC-276｜Factor hash必须包含 market/session/source semantics中会改变输出的部分 **[P0]**

**例**：A股240 bar vs US390 bar、raw vs adjusted price、current-members vs historical membership。


## R15-INC-277｜CostTier不能只从metadata tag读整数，需由benchmark/effective complexity校准 **[P1]**

**整改**：静态复杂度 + benchmark shape曲线；cost随window/N/features变化时用CostModel而非常数。


## R15-INC-278｜高成本operator需要 expression-level累计预算，而非单算子max_cost **[P1]**

**整改**：AST cost=sum/node complexity with multiplicative penalties for nested heavy ops；防止多个cost=5嵌套爆炸。


## R15-INC-279｜内存成本必须纳入CostModel，尤其O(W²)/O(N²) topology/kernel/graph **[P1]**

**整改**：估计time + memory；超budget编译时拒绝或进入batch research lane。


## R15-INC-280｜Cost/evidence应区分small-universe测试通过与全A股可运行 **[P1]**

**整改**：benchmark至少 tiny/medium/full-breadth representative sizes，记录scaling exponent；不能只“能跑”就production-friendly。


# 第四部分：跨算子族仍必须新增的系统性整改项

## R15-INC-281｜所有算子的 `OutputSemanticType` 必须一等化，不能仅有unit/category **[P0]**

**需要支持**：NumericAlpha, ConditionBool, EventBool, SignedState, ContinuousState, GlobalStateValue, GroupStateValue, PriceLevel, Return, Volatility, Probability, Count, DurationBars, Residual, SourceTransformOutput。

**原因**：MiningRole、AST positions、terminal legality都应从 output semantic type + scope 推导，不能继续靠名字。


## R15-INC-282｜InputSemanticType必须逐参数而非一个operator-level tuple **[P0]**

**例**：`close:PositivePrice`, `volume:NonNegativeActivity`, `group_id:GroupKey`, `event:EventBool`。当前“input_semantic_types tuple”无法知道谁对应谁。


## R15-INC-283｜Unit contract 与 SemanticType必须分开 **[P0]**

**问题**：两个输入都dimensionless不代表语义可互换，例如 Return、TurnoverRatio、Probability、ConditionBool。

**整改**：Unit algebra负责量纲；SemanticType负责经济/逻辑域。


## R15-INC-284｜PositivePrice/NonNegativeVolume 等domain contract必须在source/typed layer先验证，kernel只防御 **[P0]**

**整改**：避免每个operator各自扫描全panel；source validation产生DQ mask/error context。


## R15-INC-285｜`NaN`、`Inf`、`pd.NA`、`None`、`NaT` 的 MissingValue语义要统一 **[P0]**

**整改**：numeric panel统一 finite mask；object/group/event panel统一 MissingSentinel normalization；所有backend parity。


## R15-INC-286｜禁止把 `Inf` 当“不是NaN所以有效” **[P0]**

**整改**：全库 audit `.notna()/.count()` 在numeric semantic panel上的使用；需要finite-aware版本。上一版已指出部分模块，本轮要求升级为全库机器规则。


## R15-INC-287｜所有 `np.nan*` reduction都必须审计是否把missing silently ignored **[P0]**

**整改**：每canonical明确是 IGNORE_VALID / BREAK / FULL_WINDOW_REQUIRED；禁止因为用`nanmean/nansum`就默认合理。


## R15-INC-288｜所有 `.dropna()` 必须分类：cross-sectional cohort合法 vs time-axis compression非法 **[P0]**

**整改**：AST lint不能一票否决；依据 TimeTopologyPolicy。所有time geometry/lag/path/FFT/embedding中dropna默认P0。


## R15-INC-289｜所有 `.reindex()` 必须分类：explicit broadcast/source join vs silent axis repair **[P0]**

**整改**：只有 BroadcastSpec/SourceJoinContract允许reindex；ordinary factor multi-input kernel内部reindex hard fail。


## R15-INC-290｜所有 `pd.concat` 多panel路径都要检查是否发生隐式union axis **[P0]**

**整改**：concat前SameAxis，或明确join contract；不能靠concat自动对齐。


## R15-INC-291｜所有 `np.stack` / `column_stack` 多panel路径都要自动要求SameAxis **[P0]**

**整改**：source scanner发现多DataFrame `.to_numpy()`后stack而无AxisContract→blocker。


## R15-INC-292｜所有 `shift()` 必须区分 physical-row lag 与 trading-session/event lag **[P0]**

**整改**：LagSemanticType进入contract；minute跨午休/跨日、event panel/fiscal panel不能都用裸row shift。


## R15-INC-293｜所有 `rolling(window)` 必须声明 window clock **[P0]**

**取值**：TRADING_ROWS / MINUTE_SLOTS / COMPLETED_SESSIONS / FISCAL_PERIODS / EVENTS / UPDATES / ACTIVITY_CLOCK。


## R15-INC-294｜所有 `.ewm(adjust=False)` 必须自动标识recursive execution **[P0]**

**整改**：AST detector；若未ExecutionContract声明，CI fail。不能靠手工名字表。


## R15-INC-295｜所有 cumulative/expanding 算子必须审计 dataset-start dependence **[P1]**

**整改**：若数学定义确实全历史，Role=HIGH_COST/full-history且factor hash含history-start policy；若只是想要bounded，应改rolling。


## R15-INC-296｜所有 optimization/LP/SVD/eigen solver必须记录 convergence/status **[P1]**

**整改**：solver失败不能被 broad except吞成“普通NaN”而无diagnostic；evidence统计failure rate。


## R15-INC-297｜所有 iterative estimator必须声明 max_iter/tolerance 是 numerical policy **[P1]**

**整改**：不进入自由搜索，但进入version/evidence hash；convergence=false→NaN。


## R15-INC-298｜所有 random/projection/Monte Carlo型算法必须显式 deterministic seed/version或离开production factor surface **[P0]**

**整改**：若可用deterministic quadrature/fixed directions优先；seed不是alpha参数。


## R15-INC-299｜所有 tie-sensitive排序/argmin/argmax/nearest-neighbor必须有 TiePolicy **[P0]**

**取值**：AVERAGE_RANK / ALL_AT_BOUNDARY / MOST_RECENT / EARLIEST / DETERMINISTIC_SECONDARY_KEY。禁止依赖column/row arrival order。


## R15-INC-300｜所有 quantile/binning operator必须区分 requested bins 与 effective support **[P1]**

**整改**：ties导致bin collapse时normalization/support用effective bins；bins属于estimator role。


## R15-INC-301｜所有 entropy operator必须明确 log base、normalization support、单位 **[P1]**

**整改**：nats/bits/dimensionless-normalized不可混用；normalized denominator必须理论固定support或明确effective support。


## R15-INC-302｜所有 HHI/concentration operator必须消除或声明 N-dependent baseline **[P1]**

**整改**：优先 normalized excess concentration；raw HHI只保留honest raw canonical。


## R15-INC-303｜所有 probability/tail statistic必须约束输出range并做 property test **[P0]**

**例**：Probability [0,1]、Correlation [-1,1]、Entropy normalized [0,1]、count>=0。超域不能clip掩盖数学bug。


## R15-INC-304｜所有 covariance/beta/regression operator必须做 cohort-consistency **[P0]**

**整改**：mean/cov/var/residual必须同一paired cohort；不能不同pandas rolling对象各自drop missing。


## R15-INC-305｜所有 regression residual分为 in-sample descriptive 与 prior/predictive residual **[P0]**

**整改**：OutputSemanticType/MiningRole区分；default mining仅predictive/prior版本，descriptive进入DIAGNOSTIC/STATE视用途。


## R15-INC-306｜所有 benchmark/market/group aggregate必须明确 self-inclusion policy **[P0]**

**整改**：ExSelfPolicy进入canonical contract；股票对market/group factor默认优先ex-self，含self版本benchmark/diagnostic。


## R15-INC-307｜所有 group operator必须明确 membership vintage **[P0]**

**整改**：CURRENT_MEMBERS_RETROSPECTIVE / HISTORICAL_CONTEMPORANEOUS / FIXED_REFERENCE_MEMBERS 三种不能混。


## R15-INC-308｜所有 group operator必须统一 missing group key semantics **[P0]**

**整改**：None/NaN/pd.NA/NaT/empty string统一 missing；0是否missing由GroupSchema决定，不可各模块猜。


## R15-INC-309｜所有 source disclosure top-K operator必须把 K/source-coverage当source semantics **[P1]**

**整改**：top10 holders不是whole-holder universe；K进入source fingerprint，不让不同K结果混同。


## R15-INC-310｜所有 fundamental operator必须区分 flow / stock / per-share / ratio / cumulative-YTD **[P0]**

**整改**：FiscalSemanticType进入field schema；TTM/quarter/yoy transformations只能在合法类型上用。


## R15-INC-311｜所有 fiscal-period operator必须显式 adjacent-period continuity **[P0]**

**整改**：跳过季度/财年时不能把两个非相邻period直接当QoQ/YoY；PeriodId graph验证。


## R15-INC-312｜所有 expectation/analyst estimate operator必须带 TargetPeriodId identity **[P0]**

**整改**：不同target fiscal period的估计不能直接diff；same-target revision已存在逻辑，但要求全族机器化。


## R15-INC-313｜所有 event/report update-clock operator必须区分 observation date、publication date、effective date **[P0]**

**整改**：PIT availability来自明确date field，不能用TradeDate代替。


## R15-INC-314｜所有 A股涨跌停/停牌状态operator必须支持制度/板块/日期规则版本 **[P0]**

**整改**：MarketRuleVersion进入Source/Market contract；不能硬编码一个10%规则泛化所有股票日期。


## R15-INC-315｜所有 minute→daily operator必须统一 availability timestamp **[P0]**

**整改**：EOD whole-session factor = session close后可用；若用收盘集合竞价等数据，明确exact release time。不能只写daily。


## R15-INC-316｜所有 minute→daily factor必须统一 partial-session policy **[P0]**

**整改**：official full/half-day calendar；missing/duplicate/off-grid slots；停牌整日与数据缺失区分。


## R15-INC-317｜所有 minute return计算必须明确 first-bar basis **[P0]**

**整改**：within-session first return = NaN 或 open→first bar；不得引用前日close除非canonical明确overnight-inclusive。


## R15-INC-318｜所有 intraday aggregation必须明确 lunch break是否物理slot **[P0]**

**整改**：A股连续竞价240bars，不把午休120分钟当缺失bars；SessionGrid authority。


## R15-INC-319｜所有 price ratio/return operator必须检查 PriceBasis一致 **[P0]**

**整改**：raw/adjusted/continuous不能混；typed IR machine gate。不能依赖instrument column name推断。


## R15-INC-320｜所有 volume/amount/turnover输入必须有明确单位与nonnegative domain **[P0]**

**整改**：shares、currency amount、decimal turnover、percent turnover不可互换；source normalization先统一。


## R15-INC-321｜所有 scale-invariant factor参数要声明 equivalence，否则搜索制造比例重复 **[P1]**

**例**：weights只看相对比例时 `[1,2]` 与 `[2,4]`应同identity。只允许有数学证明的参数做canonicalization。


## R15-INC-322｜所有 condition/state operator必须禁止 continuous numeric自动truthiness **[P0]**

**整改**：AST type checker要求ConditionBool/EventBool；只有显式 compare/to_condition operator可从Numeric转Bool。


## R15-INC-323｜所有 state carry/episode operator必须定义 restart/reset semantics **[P0]**

**取值**：NaN break、session reset、symbol listing reset、group change reset、market rule reset。ExecutionContract必须包含。


## R15-INC-324｜所有 recursive state operator必须验证 full-run == segmented/restart **[P0]**

**整改**：有checkpoint者随机分段bit/numeric parity；无checkpoint者明确FULL_REPLAY并进入成本模型，不能声称incremental。


## R15-INC-325｜所有 bounded rolling operator必须验证 chunk+warmup == full run **[P0]**

**整改**：从HistoryRequirement自动生成warmup，随机chunk boundaries测试。


## R15-INC-326｜所有 HistoryRequirement formula必须通过动态“最小充分历史”验证 **[P0]**

**方法**：full run vs truncated prefix，逐步减少warmup，证明声明rows足够且不过度少；formula改变需evidence更新。


## R15-INC-327｜所有 role/cost/source/semantic metadata修改都必须让 manifest hash变化 **[P0]**

**整改**：防止cache/cold-start继续使用旧搜索空间。


## R15-INC-328｜所有 deleted/internalized canonical必须扫描 cold-start library/recipes/docs 中的残留引用 **[P0]**

**整改**：alias迁移或自动rewrite；不能删registry后留下大量无法解析formula。


## R15-INC-329｜所有 recommended replacement必须验证语义方向/单位/历史需求，不只是名字映射 **[P1]**

**整改**：replacement mapping有migration test：旧公式rewrite后可解析，并明确是否数值等价还是语义升级。


## R15-INC-330｜所有 FactorEngine 审计必须在 final loaded state运行，不允许基于源码静态集合的中间快照宣布完成 **[P0]**

**整改**：唯一入口 `build_final_registry_snapshot()`，完成 load/import/aliases/hardening/evidence overlay后freeze；所有audit/mining/manifest/test读取同snapshot。


# 第五部分：不要把“330条人工发现”当作结束 —— 必须执行 100% Canonical Machine Audit

## 5.1 目标

本轮完成标准不是“R15-INC-001～330都改了”，而是：

> **当前 final registry 中的每一个 canonical、每一个 backend implementation、每一个 public alias、每一个 source-transform/research-tool，都必须有一条审计记录，并且集合严格相等。人工列出的 330 条只是已知缺陷，不是审计上限。**

必须在代码中实现一个真正的全量入口，例如：

```python
snapshot = build_final_registry_snapshot(strict=True)
report = audit_every_operator(snapshot, market_profiles=["ashare", "us"])
```

并满足：

```text
SET(report.factor_operator_canonicals) == SET(snapshot.factor_operator_canonicals)
SET(report.research_tools) == SET(snapshot.research_tools)
SET(report.source_transforms) == SET(snapshot.source_transforms)
UNAUDITED_CANONICALS == []
DUPLICATE_AUDIT_ROWS == []
ORPHAN_AUDIT_ROWS == []
```

任何集合不相等，直接 exit non-zero。


## 5.2 每个 canonical 必须生成的审计字段

不要只生成 role/blocker。每一行至少包含以下字段：

### A. Identity / Registration

```text
canonical
canonical_spec_version
aliases[]
implementation_files[]
backend_implementations[]
backend_kind_by_backend
registry_snapshot_id
implementation_fingerprint
semantic_contract_fingerprint
factor_identity_fingerprint
lifecycle_status
authoring_tier
```

### B. Logical Signature

```text
panel_params[]
scalar_params[]
optional_params[]
keyword_only_params[]
defaults{}
param_specs{}
param_roles{}
active_when{}
relational_constraints[]
parameter_equivalence_classes{}
```

逐项检查：

- kernel signature == CanonicalSpec signature；
- backend signature == canonical signature；
- defaults一致；
- 没有隐藏kwargs；
- 没有未声明参数；
- 没有 declared-but-unused 参数；
- 没有 silent int/bool/string coercion；
- 没有非法 parameter region；
- 没有两个参数组合映射到同一effective parameter而未声明equivalence。

### C. Input Contract

```text
input_semantic_type_by_param{}
input_unit_by_param{}
input_grain_by_param{}
input_entity_type_by_param{}
axis_relation_by_param{}
source_requirement_by_param{}
market_support_by_param{}
price_basis_requirement_by_param{}
```

必须逐参数，不接受“input types = 一个无归属tuple”。

### D. Time / PIT Contract

```text
current_observation_role
window_clock
window_semantics
history_requirement
history_formula
availability_timestamp
prefix_causal
future_reference_detected
session_reset_policy
fiscal_period_policy
membership_vintage_policy
```

### E. Missing / Cohort / Topology

```text
missing_policy
cohort_policy
time_topology_policy
current_missing_policy
gap_reset_policy
max_staleness
min_effective_observations
min_effective_fraction
```

### F. Output Contract

```text
output_semantic_type
output_unit
output_scope
expected_range
shape_relation
terminal_legality
legal_ast_positions[]
```

### G. Execution

```text
state_model
chunking_mode
checkpoint_schema
history_replay_mode
incremental_capability
backend_capabilities
native_backend_count
bridge_backend_count
cost_model
memory_model
```

### H. Statistical / Mathematical Verification

```text
reference_implementation
synthetic_goldens[]
null_calibration_status
sample_size_sensitivity_status
tie_invariance_status
scale_invariance_status
shift_invariance_status
column_permutation_status
prefix_causality_status
chunk_equivalence_status
```

### I. Mining Admission

```text
mining_role
role_source
intrinsic_mining_eligible
contextual_eligibility_by_market{}
source_status_by_market{}
allowed_ast_positions
terminal_allowed
searchable_parameter_schema
cost_lane
hard_blockers[]
routing_constraints[]
quality_warnings[]
required_actions[]
```

### J. Resolution

```text
final_disposition = DIRECT_ALPHA | HIGH_COST_ALPHA | CONDITION | STATE |
                    EVENT_MASK | GROUP_STATE | GLOBAL_STATE | INTRADAY_EOD |
                    FUNDAMENTAL_PIT | SOURCE_TRANSFORM | RECIPE_INTERNAL |
                    DIAGNOSTIC | RESEARCH_PENDING | LEGACY_ALIAS | DELETED

verification_status = PASS | FAIL | BLOCKED_BY_DATA | BLOCKED_BY_RESEARCH
```


# 第六部分：全库静态扫描规则 —— 必须递归所有 FactorEngine Python 文件

实现 `audit_static_patterns()`，扫描整个 `factor_engine/`，至少包含以下规则。注意：静态命中是“需要语义审核”的候选，不是一律判错；但任何命中必须在最终报告中 resolved，不能无解释消失。

## 6.1 时间/PIT危险模式

扫描：

```text
shift(-
lead(
next(
bfill
interpolate using future endpoints
rolling(center=True)
[::-1] 后未来方向误用
iloc[row + ...]
array[t + ...]
future_date / next_period
```

要求每一个命中标注：SAFE / UNSAFE / NOT_FACTOR / FALSE_POSITIVE + rationale。


## 6.2 物理时间轴压缩

扫描：

```text
dropna()
arr[np.isfinite(arr)]
valid = x[finite]
compress / concatenate finite pieces
```

若后续存在 lag / FFT / embedding / recurrence / path / interval / motif / autocorr / change-point / run-length / state geometry，则默认 P0，除非明确数学上 time-order irrelevant。


## 6.3 静默 axis 修复

扫描：

```text
reindex(...)
align(..., join=...)
pd.concat([...])
join/merge
broadcast_to
set_axis
columns assignment
```

每一处必须有 Axis/Broadcast/SourceJoin contract。


## 6.4 多panel裸 NumPy转换

扫描同一函数中多个：

```text
x.to_numpy(...)
y.to_numpy(...)
f1.to_numpy(...)
np.stack / column_stack
```

没有 central SameAxis/Broadcast validation → blocker。


## 6.5 Missing被当False/0

扫描：

```text
np.nan_to_num
fillna(0)
where(condition, ..., 0)
astype(bool)
bool(value)
!= 0
== 0
notna/count on numeric data
```

根据SemanticType决定是否合法。


## 6.6 Silent parameter clamp/coercion

扫描：

```text
int(param)
max(min, int(param))
min(max, param)
np.clip(param,...)
bool(param)
str(param).lower() with fallback
kwargs.get(...)
```

若中央validator已经保证合法，kernel可以转换但不得改变值；任何 runtime clamp使不同合法AST合并都要修。


## 6.7 Hidden magic constants

扫描数值常量：

```text
0.01, 0.05, 0.1, 0.2, 0.3, 0.5,
1.345, 1.4826, 1e-3, 1e-6, 1e-12,
20, 60, 120, 240, 252, 390
```

不是说所有常数都错，而是必须分类：MATHEMATICAL_CONSTANT / MARKET_RULE / NUMERICAL_EPS / SUPPORT_POLICY / MODEL_PRESET / HIDDEN_ALPHA_KNOB。后四类必须metadata可见、版本化。


## 6.8 Solver/linear algebra

扫描：

```text
lstsq
solve
inv
pinv
svd
eig/eigh
linprog
minimize
curve_fit
```

要求 rank/condition/convergence/support 检查；禁止 failure broad-except后默默当普通NaN而不记录。


## 6.9 全局状态/组状态广播

扫描：

```text
out[t, :] = scalar
out[t, idx] = scalar
broadcast_row_stat
np.repeat(row_scalar)
```

必须 output_scope=GLOBAL/GROUP，除非 scalar实际上只写给单个selection且仍有横截面差异。


## 6.10 Recursive state候选

扫描：

```text
ewm(adjust=False)
for t in range(...): state = f(state, x[t])
last_value / last_state / accumulator
cumprod/cumsum used as persistent state
```

缺 ExecutionContract → fail。


## 6.11 Requested-vs-effective estimator dimensions

扫描：

```text
min(k, available)
min(bins, unique)
min(level, band)
max(1,...)
truncate order/rank/components
```

如果effective dimension变化，必须：非法组合raise、或effective dimension进入输出identity/diagnostic；不能悄悄缩水。


## 6.12 Wide-data whole-panel validation

扫描在kernel开始对完整DataFrame：

```text
np.any(panel < 0)
np.all(...)
series.values ... whole history
```

如果raise/warn影响过去输出，审计 future-dependent call behavior。


# 第七部分：动态 / Metamorphic Test Matrix —— 每个适用 canonical 自动生成

不要手写只覆盖几十个 operator 的测试。根据 ResolvedSemanticContract 自动决定适用测试。

## T01 Prefix Future Randomization

固定 `[0:t]`，随机改写 `t+1:`，输出 `<=t` 必须完全不变。

## T02 Append Future Rows

原始数据与尾部追加任意未来数据，原区间输出完全一致。

## T03 Column Permutation

股票列随机打乱，再恢复列顺序，输出应一致；除非 contract明确依赖 ordered entities（普通股票panel不允许）。

## T04 Multi-input Column Mismatch

只打乱一个输入列顺序：SameAxis算子必须raise，不得silent align。

## T05 Multi-input Date Shift

一个输入index整体+1交易日：必须raise/SourceJoin明确处理。

## T06 Duplicate Index

重复timestamp必须被中央axis gate拒绝。

## T07 Duplicate Columns

重复instrument必须拒绝。

## T08 Current Missing

仅当前cell设NaN，验证CurrentMissingPolicy；不得意外复用历史值。

## T09 Mid-window Gap

窗口中间插NaN，验证BREAK/PAIRWISE/CONTIGUOUS/STATE_CARRY语义。

## T10 Leading Gap

历史开头NaN，不得改变成熟期定义或错误连接。

## T11 Trailing Gap

当前前一行NaN，检查lag/embedding/session/recovery等。

## T12 Inf / -Inf

在每个numeric input注入Inf，不能被notna/count当有效。

## T13 Strict Bool Fuzz

ConditionBool/EventBool输入：`-2,-1,0,.2,1,2,NaN,Inf`；仅0/1/NaN合法。

## T14 Parameter Type Fuzz

每scalar传：bool/int/float fractional/str/None/NaN/Inf；按ParamSpec精确接受/拒绝。

## T15 Parameter Boundary

min-ε/min/min+ε/max-ε/max/max+ε；compile/runtime结果一致。

## T16 Relational Constraint Grid

fast/slow、window/lag、window/bins、k/N等所有关系边界自动生成。

## T17 Parameter Injectivity

对每searchable参数至少取3个合法值；若输出与history/contract均不变→dead candidate。

## T18 Conditional Parameter Inactivity

active_when=False时非默认值必须拒绝，不生成死AST。

## T19 Parameter Equivalence

声明equivalence的比例权重等应canonical hash相同；未声明的不能擅自合并。

## T20 Unit Scale Test

根据Unit/SemanticType自动乘常数，验证理论 scale equivariance/invariance。

## T21 Additive Shift Test

对translation-invariant统计加常数；若理论不应变则验证。

## T22 Sign Flip Test

signed-return symmetric family验证预期 odd/even性质。

## T23 Mirror Tail Test

signed series lower-tail operator与 mirrored upper-tail理论关系。

## T24 Tie Permutation

制造大量ties并随机列/行顺序，结果不应依arrival order。

## T25 Constant Input

验证定义是0还是NaN，不允许意外Inf/epsilon有限值。

## T26 Near-Constant Input

检查numerical stability和epsilon domination。

## T27 Single Observation / Tiny N

所有统计support不足应NaN/raise，不要伪精确值。

## T28 Exact Minimum Support

在刚好min support边界验证首个有效输出。

## T29 Warmup First-valid Index

机器计算理论首个有效row，与实际first finite一致。

## T30 History Sufficiency

用HistoryRequirement截取精确warmup，chunk输出与full run相同。

## T31 One-row-less History

比声明history少1，不能仍声称充分；如仍相同则检查history是否过度估计。

## T32 Random Chunk Boundaries

stateless rolling / checkpoint stateful分别验证分块执行等价。

## T33 Stateful Checkpoint Serialization

serialize→deserialize→继续运行与不中断完全一致。

## T34 Stateful Missing Reset

gap前后state按contract reset/carry。

## T35 Session Boundary

前日最后bar不能进入当日within-session return/rolling，除非明确overnight canonical。

## T36 A股240 Grid

完整240、239缺一、241重复一、off-grid second、午休、截断close、整日停牌分别测试。

## T37 US Session Grid

DST前后、390常规session、half-day、UTC storage→NY trade date。

## T38 Timezone Ambiguity

bare UTC无market/session_tz必须fail closed。

## T39 Daily→Minute Broadcast

high_limit/low_limit等必须按trade_date且instrument exact；错一天/错股票拒绝。

## T40 PriceBasis Mismatch

RAW open + adjusted close必须编译/运行拒绝。

## T41 Group Missing Key

None/np.nan/pd.NA/NaT/""一致处理。

## T42 Group Label Permutation

重命名group labels（bijection）不应改变数值。

## T43 Membership Vintage

同一历史路径但current-members vs historical-members生成不同且符合contract的结果。

## T44 Ex-self

极端单股票改变自己的值不应通过benchmark self-inclusion机械反馈到其自身reference（ex-self variant）。

## T45 Global/Group Terminal Ban

编译器尝试把GLOBAL/GROUP state作为最终rank factor，必须拒绝。

## T46 Condition Position Ban

NumericAlpha直接塞condition slot必须拒绝，除非显式compare。

## T47 Source Availability

逐market移除required source，contextual eligibility必须False/UNKNOWN，不得assume available。

## T48 Snapshot-only Historical Ban

snapshot-only source不能用于历史mining。

## T49 Evidence Staleness

改变implementation/helper/semantic contract任一hash，certification立即STALE。

## T50 Backend Metadata Parity

同canonical不同backend signature/units/semantic contract完全一致。

## T51 Backend Numeric Parity

在各backend共同合法域比较数值；注意native vs bridge分别标。

## T52 Solver Failure

构造singular/ill-conditioned/unbounded/nonconverged输入，必须有明确NaN/error/diagnostic。

## T53 Regression Rank

constant/collinear predictors不能偷偷降模型。

## T54 Quantile Tail Support

q越极端要求的N越高；不足support fail。

## T55 Histogram Effective Bins

ties导致requested bins collapse时 normalization/support正确。

## T56 Entropy Null

uniform/one-hot/known categorical distributions的范围和baseline。

## T57 MI Independence Null

独立变量MI接近0；检查bias correction方向。

## T58 Dependence Positive Control

强非线性依赖应显著高于独立null。

## T59 Spectral Sine Golden

已知frequency sine，peak/frequency/entropy读数正确。

## T60 Spectral Gap

中间NaN不能压缩成等间距序列。

## T61 DMD Known Linear System

已知eigenvalues/growth/frequency，DMD输出与理论一致。

## T62 DMD Conjugate Pair

真正共轭合并，非共轭不合并。

## T63 RQA Hand-built Recurrence Matrix

rate/diagonal entropy/trapping/divergence直接对矩阵golden。

## T64 RQA Boundary Censor

触及window边界的line处理按contract。

## T65 Event Poisson Null

Fano excess≈0、spacing CV合理；block censor策略校准。

## T66 Event Regular Process

固定间隔→低CV/Fano<1。

## T67 Event Cluster Process

burst→Fano>1。

## T68 Recovery Censoring

早盘完整shock vs晚盘right-censored shock，complete-case/KM版本语义符合定义。

## T69 Hill Pareto Golden

Pareto(alpha)已知 ξ；sample增长估计收敛。

## T70 Hill Invalid Negative Domain

负threshold/不合法SemanticType必须拒绝/NaN，不产生负Hill shape冒充合法结果。

## T71 GLR No-change Null

不同window/null AR processes false-positive baseline稳定。

## T72 GLR Known Mean Shift

方向/位置/强度单调正确；cap行为明确。

## T73 GLR Variance-only Shift

均值变化不应触发variance score，方差变化方向正确。

## T74 Isotonic Known Monotone

单调函数residual≈noise；ties顺序不影响。

## T75 KNN Default Viability

每个公开默认参数在合理A股breadth synthetic data上必须产生非全NaN结果。

## T76 KNN Tie Boundary

kth distance ties的effective neighborhood确定、列顺序不影响。

## T77 Rank-copula Known Independent/Dependent

验证MI correction、entropy normalization。

## T78 Group MST Perfect Corr

perfect corr edge length=0且合法计入MST。

## T79 Group MST Disconnected

有效边不足→NaN，不拼接。

## T80 Composition Closure Invariance

整体乘同常数CLR/ILR不变；不同whole字段必须拒绝。

## T81 Composition Part Identity

同PartSet乱序时如果按PartId匹配应等价；错配应拒绝。

## T82 Volume-clock Missing Minute

missing required bar不能被dropna压缩后仍输出。

## T83 Volume-clock Zero Activity

zero activity+unchanged price合法；zero activity+price jump按contract拒绝。

## T84 Quantile-PCA Dead Region

所有合法window/k组合必须至少存在可输出数据，不允许compile-valid guaranteed-NaN。

## T85 Global Breadth Sensitivity

global/group state对universe缺失变化有coverage gate，不把数据缺口当regime。

## T86 Fundamental Adjacent Period

缺季度时QoQ/beat streak不跨缺口。

## T87 Estimate Target Period

target period change不算same-target revision。

## T88 Market Rule Version

不同A股板块/制度日期limit operator使用正确rule profile。

## T89 Hash Semantic Version

改missing/current/session/member policy，factor hash必须改变。

## T90 Audit Mutation Suite

对以上关键bug类型自动植入fixture，证明审计器不是装饰。


# 第八部分：逐模块族覆盖清单 —— 一个都不能跳

最终报告必须按下面所有族给出 audited canonical count / pass / fail / deleted / internalized。列表只是最低范围；`load_all()` 新注册的未来模块自动追加。

```text
1. common.elementwise / scalar math
2. common.time_series / rolling / shift / cumulative / expanding
3. common.cross_sectional
4. common.group / neutralization
5. common.data_cleaning / safe_ops
6. common.statistics / regression helpers
7. price_volume core
8. price_volume technical extensions / liquidity
9. technical indicators / candlesticks / structure patterns
10. downside_risk / robust_stats / robust_tail / robust_scale
11. direction_concentration / weighted_tail / prospect_theory
12. state_event / alpha_language_state / stateful.*
13. threshold_cycle / state_episode / directional_change
14. return_decomp / activity_clock / turnover_survival
15. intraday core / intraday_session / advanced_intraday
16. volume_clock / session_recovery / intraday_activity_duration
17. microstructure core / spread / flow_impact / impact_decay / barrier
18. realized beta / higher moments / intraday volatility / jump robust
19. fundamental.transforms / quality / growth / accrual / expectation
20. valuation
21. shareholder / holder network
22. index_listing / A-share state machine / limit ops
23. relation core / relation distributions / network / spectral / temporal
24. cross_section_local / cross_section_ext / dynamic_knn
25. group_spectrum / cs_state_ops / systemic tail
26. regression_models / dynamic_regression / AR / state_space / volatility models
27. complexity / sequence anomaly / path signature
28. spectral / spectral_ext / wavelet / cross_spectrum / research_spectral
29. DMD / Hankel / SSA
30. nonlinear_dependence / dependence_ext / conditional_dependence
31. advanced_information / TE / MI / HSIC / kernel Granger
32. RQA / HVG / topology / advanced_topology
33. intrinsic_dimension / local_lyapunov / first_passage
34. multifractal / rough_vol / memory_ext
35. event_interval / marked_event / event_response / update_clock
36. distribution_break / GLR / binned_response / quantile dynamics / expectile
37. composition / feature_geometry / advanced_structure / research_transform
38. all Polars/SQL/backend mirrors and bridges
39. registry / surface / operator_spec / semantic certification
40. runtime execution/history/checkpoint/warmup
41. mining/operator_catalog / admission / manifest / cold-start
42. evidence/certification/finalization
43. ResearchToolRegistry
44. SourceTransform layer
```

对每个族必须列出：

```text
registered_canonicals
sampled_files
all_files_scanned
new_findings_count
math_defects
contract_defects
parameter_defects
role_defects
backend_defects
evidence_defects
unresolved_count
```

`all_files_scanned=False` 或 `unresolved_count>0` 不得宣布完成。


# 第九部分：这次修改的执行顺序（为避免修一层又被另一层覆盖）

## Phase 0 — 先冻结当前基线

生成：

```text
build/r15/baseline_registry_snapshot.json
build/r15/baseline_operator_fingerprints.json
build/r15/baseline_admission_matrix.json
build/r15/baseline_test_failures.json
```

记录 current HEAD / dirty status。

## Phase 1 — 修审计器本身

优先修：

- recursive source scan；
- real invariant detectors；
- nonzero exit code；
- final snapshot；
- machine role/terminal logic；
- source context；
- semantic contract统一。

**原因**：审计器不可信时，后面“全绿”没有意义。

## Phase 2 — 修 MiningRole / Manifest / Admission

先让 AlphaProbe/AlphaMiner 真正看到“正确角色、正确参数、正确source/frequency”的算子，而不是错误搜索空间。

## Phase 3 — 修 R12/R15 本轮新增算子具体bug

按 P0→P1：DMD、composition、event interval、session/intraday、KNN/copula、Hill、conditional dependence等。

## Phase 4 — 自动全registry扫描补漏

运行本文件第五～八部分的机器审计；凡新抓到而本文件未点名的问题，继续修，不得说“提示词没写所以不处理”。

## Phase 5 — 运行动态90测试矩阵

自动按contract适配，不要求每canonical都跑90项，但每canonical必须有 `applicable_tests[] / passed_tests[] / skipped_not_applicable[]`，不能空。

## Phase 6 — Backend parity / performance

native/bridge分别验证；至少一个生产后端即可，但metadata/semantic contract必须全backend一致。

## Phase 7 — 重生成 evidence

只有 tree/registry完全稳定后再做。evidence必须绑定final fingerprints。

## Phase 8 — 最终 mining manifest / cold-start catalog

最后生成，不允许中途产物继续沿用。


# 第十部分：最终必须新增/重构的核心结构

建议最终至少存在这些对象；名字可以不同，但职责不能缺：

```python
@dataclass(frozen=True)
class CanonicalOperatorSpec:
    canonical: str
    logical_signature: LogicalSignature
    semantic_contract: ResolvedSemanticContract
    lifecycle: LifecycleSpec
    mining: MiningSpec
    execution: ExecutionContract
    cost: CostModel

@dataclass(frozen=True)
class ResolvedSemanticContract:
    inputs: dict[str, InputContract]
    output: OutputContract
    current_observation_role: CurrentObservationRole
    cohort_policy: CohortPolicy
    time_topology: TimeTopologyPolicy
    window: WindowContract | None
    history: HistoryRequirement
    missing: MissingContract
    availability: AvailabilityContract
    market_support: tuple[str, ...]

@dataclass(frozen=True)
class MiningSpec:
    role: MiningRole
    allowed_ast_positions: tuple[str, ...]
    terminal_allowed: bool
    source_requirements: tuple[SourceRequirement, ...]
    parameter_search_schema: dict[str, SearchParamSpec]

@dataclass(frozen=True)
class FinalRegistrySnapshot:
    snapshot_id: str
    git_sha: str
    dirty: bool
    operators: Mapping[str, CanonicalOperatorSpec]
    aliases: Mapping[str, str]
    research_tools: ...
    source_transforms: ...
```

**禁止**继续新增第五、第六套手工 side list 作为“临时解决”。


# 第十一部分：最终硬性 Invariants —— 全部必须为零

在上一份 invariants 基础上，本轮新增以下更严格约束：

```text
UNAUDITED_REGISTERED_CANONICALS == ∅
ORPHAN_AUDIT_RECORDS == ∅
ORPHAN_SEMANTIC_DECLARATIONS == ∅
ROLE_ASSIGNED_BY_UNVERIFIED_FALLBACK == ∅
SUPPORTING_ROLE_WITH_TERMINAL_POSITION == ∅
CONTINUOUS_OUTPUT_MISCLASSIFIED_AS_EVENT_BOOL == ∅
EVENT_BOOL_WITHOUT_STRICT_BOOL_TYPE == ∅
MULTI_PANEL_WITHOUT_AXIS_CONTRACT == ∅
BROADCAST_WITHOUT_BROADCAST_SPEC == ∅
REINDEX_WITHOUT_JOIN_OR_BROADCAST_CONTRACT == ∅
TIME_TOPOLOGY_COMPRESSION_UNDECLARED == ∅
WINDOW_CLOCK_UNDECLARED_FOR_WINDOW_PARAM == ∅
CURRENT_OBSERVATION_ROLE_UNDECLARED == ∅
STATEFUL_WITHOUT_EXECUTION_CONTRACT == ∅
HISTORY_REQUIREMENT_UNRESOLVED == ∅
COMPILE_VALID_RUNTIME_GUARANTEED_NAN_PARAM_REGION == ∅
SEARCHABLE_ESTIMATOR_OR_NUMERICAL_POLICY_LEAK == ∅
DEAD_SEARCHABLE_PARAMETER == ∅
BACKEND_LOGICAL_SIGNATURE_DRIFT == ∅
BACKEND_SEMANTIC_CONTRACT_DRIFT == ∅
BRIDGE_MISLABELLED_AS_NATIVE == ∅
STALE_CERTIFICATION_ACCEPTED == ∅
EVIDENCE_WITHOUT_IMPLEMENTATION_FINGERPRINT == ∅
GLOBAL_OR_GROUP_STATE_TERMINAL == ∅
SOURCE_REQUIREMENT_ASSUMED_AVAILABLE_WITHOUT_CONTEXT == ∅
MARKET_ARGUMENT_PARSED_BUT_UNUSED == ∅
SESSION_EOD_EXPOSED_AS_MINUTE_TERMINAL == ∅
DUPLICATE_PUBLIC_MATHEMATICAL_CANONICALS == ∅
UNKNOWN_SEMANTIC_TYPE_PRODUCTION_FACTOR == ∅
UNKNOWN_OUTPUT_SCOPE_PRODUCTION_FACTOR == ∅
UNVERSIONED_HIDDEN_ESTIMATOR_POLICY == ∅
AUDIT_PLACEHOLDER_INVARIANT == ∅
AUDIT_CLI_FALSE_GREEN == ∅
```

此外：

```text
Every registered FactorOperator has exactly one FinalDisposition.
Every mineable operator has at least one legal AST position.
Every non-mineable operator has zero mining AST positions.
Every searchable scalar has dtype + domain + role + search grade.
Every multi-input operator has per-param semantic/axis contract.
Every stateful operator has reset/chunk/history semantics.
Every EOD intraday operator has SessionContext + availability semantics.
```


# 第十二部分：必须输出的最终工件

全部写入 `factor_engine/build/r15/`（或等价目录），至少：

```text
01_final_registry_snapshot.json
02_every_operator_audit.csv
03_every_operator_audit.json
04_operator_family_coverage.md
05_new_findings_discovered_by_machine_audit.md
06_semantic_contract_matrix.csv
07_parameter_contract_matrix.csv
08_axis_broadcast_contract_matrix.csv
09_history_execution_matrix.csv
10_mining_role_and_ast_matrix.csv
11_source_market_capability_matrix.csv
12_backend_native_bridge_matrix.csv
13_semantic_duplicate_graph.json
14_dead_parameter_report.json
15_static_pattern_audit.json
16_dynamic_metamorphic_results.json
17_session_intraday_results.json
18_statistical_null_goldens.json
19_evidence_fingerprint_report.json
20_final_invariants.json
21_final_mining_manifest.json
22_final_cold_start_validation.json
23_deleted_or_internalized_migrations.md
24_unresolved_items.md
25_R15_FINAL_REPORT.md
```

其中 `24_unresolved_items.md` 正常完成时必须只有：

```text
# Unresolved Items
None.
```

如果确实因数据源缺失/需要外部学术确认无法代码解决，也不能藏掉，必须写：canonical、blocker、为什么无法当前解决、是否从mining移除、未来解锁条件。


# 第十三部分：Coding AI 最终回复格式

不要只回复“已完成”“tests pass”。最终回复必须包含：

## 13.1 基线与最终覆盖

```text
baseline HEAD:
final HEAD:
registered FactorOperator count:
ResearchTool count:
SourceTransform count:
audited count:
unaudited count: 0
```

## 13.2 本提示词人工已知项

```text
R15-INC-001 ~ R15-INC-330:
fixed = X
removed = X
not_applicable_after_inspection = X
blocked_by_real_data = X
unresolved = 0  # 除非明确说明不可解决原因
```

每个 not_applicable 必须给证据，不能为了省事批量标N/A。

## 13.3 机器审计额外发现

必须单列：

```text
additional findings not listed in this prompt = N
fixed = ...
```

**如果是 0，也必须说明机器审计覆盖了哪些模块/多少canonical，而不是一句“没发现”。**

## 13.4 关键 invariants

逐条打印 PASS/FAIL + count，不能只打印总PASS。

## 13.5 Tests

```text
unit tests
metamorphic tests
session tests
backend parity
mutation tests
evidence validation
manifest/cold-start validation
```

给实际数字。

## 13.6 最终剩余不可直接挖掘对象

按 disposition 列出所有：DIAGNOSTIC / SOURCE_TRANSFORM / RESEARCH_PENDING / LEGACY_ALIAS / DELETED，以及为什么不应该直接作为alpha terminal。


# 最后强调：这份是“增量提示词”，不是上一份的替代品

1. **不要重复实现上一份 NEW-001～NEW-260 / HIST-001～HIST-100 已交给其他 AI 的整改任务。**
2. 如果当前分支里那些问题已经被另一个AI修好，保留正确实现，不要回滚。
3. 如果本文件的新问题与并行修改碰到同一文件，以“最终正确语义”为准，合并而不是覆盖。
4. 本轮重点是上一份之后新出现的 R12 admission/mining/closure 层，以及上一份未单独覆盖的长尾数学/contract问题。
5. **最重要：不要因为这份人工只写到 R15-INC-330 就停止。最终必须运行全registry机器审计，把所有 canonical 都检查一遍，并把额外发现全部解决。**
6. 不允许以“这个算子是 research/experimental”作为跳过数学正确性审计的理由；可以最终不进default mining，但数学、PIT、axis、missing、参数语义仍必须正确或删除。
7. 不允许为了让 invariant 变绿而硬编码空列表、改测试期待错误行为、扩大try/except、把问题算子全部粗暴标internal/research。
8. 优先修根因和统一contract，禁止再堆新的手工canonical名单。

