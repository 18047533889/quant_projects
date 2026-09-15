# 持续复查优先队列

## P0-HASH-CODEC — PARTIAL：v2 codec 与 FeatureSetArtifact 已修，其他消费者待迁移

2026-09-12 heartbeat 更新：已实现递归类型化 v2 codec，并让新 FeatureSetArtifact 写入显式 codec；旧无版本摘要仍按 v1 读取，不改历史key。113项限定回归稳定通过。详见 HASH_CODEC_PHASE1.md。下列碰撞仍适用于未迁移的旧 content_hash 消费者，不得全局关单。

位置：quant_platform/app/contracts/_contenthash.py。

实际执行得到：
- canonical_str({'a':'b,c=d'}) == canonical_str({'a':'b','c':'d'}) == '{a=b,c=d}'，两者 content_hash 相同。
- ['a,b','c'] 与 ['a','b,c'] 编码均为 '[a,b,c]'，hash 相同。
- int 1 与 str '1' 编码均为 '1'，hash 相同。

根因：只给最外层字段做长度前缀，递归值没有类型边界与子项长度。影响 FeatureSet schema/semantic hashes、Snapshot manifest、任务幂等键/恢复、事件、报告等平台本地派生身份。不能因为三种反例之外测试通过就关单。

下一轮要求：
1. 已完成首批codec/FeatureSetArtifact反例与v2实现；随后FeatureSetVersion内容校验、成员/嵌套metadata冻结及旧公式恢复已修，最终限定123项稳定通过（FEATURE_VERSION_INTEGRITY.md）。下一轮做FeatureSetVersion显式v2新写/旧读，并处理metadata的JSON类型保持；不可静默全局更换哈希。
2. 清点 FeatureSetVersion/FeatureSetArtifact、SnapshotManifest、report、job/outbox 和所有 decoder，逐边界实施新写 v2、历史 v1 验证/读取，避免简单替换全局 content_hash 后静默破坏恢复。
3. 旧 URI、已落库 job/outbox key 与生产资产不得自动重命名/删改；旧摘要不足以证明内容相同，不可仅凭 v1 digest 合并。
4. 小型 v1 fixture→read、v2 new→write、restart/replay/diff 反例通过后再扩大范围。正式资产迁移/发布仍需授权。
5. FeatureSetVersion supplied-hash无校验和成员可变问题已独立复现并修复；保留前置5失败与最终回归证据。v1碰撞仍可绕过v1校验，不能因此把旧摘要视为碰撞安全。

## 其他继续检查方向

- 09-14 10:11 heartbeat修复评分校验与选择重复读取不一致，前置2失败，预算搜索977项稳定通过（SUPERVISED_SCORE_SNAPSHOT.md）。仅捕获一次后同值校验/使用，不是源端原子快照；评分键/身份/repair_family前置验证仍待查。

- 09-14 09:10 heartbeat修复非法网格读取评分后才拒绝（SUPERVISED_GRID_PREFLIGHT.md），前置8失败1通过，预算搜索975项稳定通过。评分Mapping快照一致性、身份/repair_family前置校验顺序仍待查。

- 09-14 08:10 heartbeat拒绝候选集text/bytes/bytearray/Mapping静默展开，前置8失败3通过，预算搜索966项稳定通过（SUPERVISED_GRID_CARRIERS.md）。网格元素其他可转换类型及拟合前校验顺序仍待查。

- 09-14 07:10 heartbeat修复冻结参数/拟合候选与评分键接受bool（SUPERVISED_BOOLEAN_CANDIDATES.md），前置8失败1通过，预算搜索955项稳定通过。其他可转换类型、拟合网格前置校验仍待查。

- 09-14 06:10 heartbeat修复冻结参数存储读写标识混淆，前置6失败1通过，预算搜索946项稳定通过（SUPERVISED_STORE_IDENTIFIERS.md）。FrozenSupervisedParameter的value/grid类型及身份往返一致性仍待查，codec迁移等仍开放。

- 09-14 05:09 heartbeat修复effective_spec_hash非法载体写入（HYPOTHESIS_SPEC_IDENTITY.md），前置6失败1通过，预算搜索939项稳定通过；摘要真实性、监督参数接口和codec迁移仍待查。

- 09-14 04:09 heartbeat修复假设组汇总标识混淆及FDR提交计数类型，前置6失败1通过，预算搜索932项稳定通过（FDR_FAMILY_INPUTS.md）。计数相等不等于成员真实性认证；effective_spec_hash载体和监督参数等仍待查。

- 09-14 03:09 heartbeat修复假设账本executed/has_pvalue非bool扭曲计数，前置10失败5通过，预算搜索925项稳定通过（HYPOTHESIS_OUTCOME_FLAGS.md）。汇总身份、FDR提交计数类型、effective_spec_hash载体及监督参数仍待查。

- 09-14 02:07 heartbeat补齐budget_state/has_reservation标识校验（DURABLE_BUDGET_QUERY_IDENTIFIER.md）：前置6失败1通过；首轮910通过但FE报告并发变化，复测910项源码稳定通过。监督参数/假设账本其他接口仍待查。

- 09-14 01:05 heartbeat修复预算写入口int/bool标识别名，前置20失败，预算搜索903项稳定通过（DURABLE_BUDGET_IDENTIFIER.md）。budget_state/has_reservation读取及监督参数/假设账本等其他接口仍待审查，不改历史键。

- 09-14 00:05 heartbeat补齐create_campaign次数/成本类型校验，前置12失败1通过，预算搜索883项稳定通过（DURABLE_CAMPAIGN_BUDGET_TYPES.md）。campaign_id/attempt_id类型及别名冲突仍待查，不自动迁移历史非法预算。

- 09-13 23:03 heartbeat修复lease_seconds的bool/非有限数输入缺口（DURABLE_LEASE_BOUNDARIES.md）：前置3失败8通过，最终预算搜索870项稳定通过。直接create_campaign预算类型、attempt_id载体等仍待审查；不自动修改历史非法租期。

- 09-13 22:02 heartbeat修复重复预留 attempt_id 金额变化未拒绝：四状态前置4失败，预算与搜索859项稳定通过（DURABLE_RESERVATION_REPLAY.md）。lease_seconds非有限数和类型边界仍待复现检查。

- 09-13 本轮修复 DurableBudgetTracker 忽略 reserved_cost：前置8失败，修复后预算与搜索855项稳定通过（DURABLE_RESERVED_AMOUNT.md）。预留重复 attempt_id 的参数一致性、lease 数值边界仍待检查。

- 09-13 19:58 heartbeat修复SQLite reserve/settle接受bool成本，重新打开测试库账目不变，预算与搜索847项通过（DURABLE_BOOLEAN_COST.md）。DurableBudgetTracker的reserved_cost参数被忽略仍待核查。

- 09-13 18:56 heartbeat修复内存BudgetTracker写接口接受bool成本，拒绝后账目不变，预算与搜索843项通过（BUDGET_BOOLEAN_OPERATIONS.md）；持久化对应入口与其他数值类型仍需核查。

- 09-13 17:56 heartbeat修复SearchBudget次数/成本类型缺口，预算与搜索833项通过（BUDGET_LIMIT_TYPES.md）；BudgetTracker直接成本API等仍待审查。

- 09-13 15:54 heartbeat修复bool评估成本被当0/1，合法数值和None兼容，搜索794项通过（BOOLEAN_EVALUATION_COST.md）。

- 09-13 14:54 heartbeat修复SearchConfig任意对象充当执行模式，保留合法字符串规范化与生产能力门禁，搜索788项通过（EXECUTION_MODE_TYPES.md）。

- 09-13 13:54 heartbeat修复SearchConfig多精度/协议开关非bool静默解释，搜索781项通过（SEARCH_SWITCH_TYPES.md）。

- 09-13 11:53 heartbeat补齐SearchConfig窗口/并发整数及停滞阈值有限数校验，搜索765项通过（SEARCH_NUMERIC_BOUNDARIES.md）。

- 09-13 10:52 heartbeat修复plateau_detector漂移返回False后仍调用proposal，搜索749项通过（PLATEAU_CONFIG_GUARD.md）；可执行对象替换、并发瞬时修改及回调内部副作用仍需审查。

- 09-13 09:50 heartbeat补齐阶段决策request/provider返回的配置校验，防止漂移后advance/昂贵评估，747项搜索通过（STAGE_DECISION_CONFIG_GUARD.md）；其他回调及并发瞬时变更仍需检查。

- 09-13 08:49 heartbeat修复最终决策入口/请求生成/provider返回三处漂移后仍完成会话，搜索745项通过（FINAL_DECISION_CONFIG_GUARD.md）；阶段决策等仍PARTIAL。

- 09-13 07:49 heartbeat补齐评估返回后的计划校验及失败结算后漂移传播，742项搜索测试通过（EVALUATOR_CONFIG_GUARD.md）；最终/阶段决策回调、并发瞬时修改仍PARTIAL。

- 09-13 06:49 heartbeat补上验证回调返回后的配置漂移校验，防止采纳结果和启动评估，搜索741项通过（VALIDATOR_CONFIG_GUARD.md）；下方旧记录中验证返回边界已修，评估/最终决策等仍PARTIAL。

- 09-13 04:48 heartbeat修复security_classification JSON恢复后留作str的假变更，精确恢复枚举并拒绝未知值，见 SECURITY_ROUNDTRIP.md。不代表metadata通用类型恢复已解决。

- 09-13 02:47 heartbeat补齐FeatureSetVersion标量身份/引用类型校验，见 VERSION_SCALAR_TYPES.md。FE/DA另一任务现已active，继续按文件协调并标记并发测试证据。

- 09-13 01:46 heartbeat修复metadata truthiness丢失内容及非Mapping静默转换，保留None兼容，见 METADATA_CARRIER.md。

- 09-13 00:45 heartbeat修复FeatureMemberRef位置/身份/可选引用类型缺口，拒绝可变引用和非整数位置，见 MEMBER_SCALAR_TYPES.md。历史错误类型不自动转换。

- 23:45 heartbeat修复VERSION/TREATMENT/ORIENTATION被scalar category或宽松成员阈值掩盖，并统一diff/event重训摘要，见 SEMANTIC_REASON_RETRAIN.md。schema显式策略和阈值分母仍待查。

- 22:45 heartbeat已修复label/data_revision分支丢失member ledger和event计数/原因，见 GLOBAL_MEMBER_LEDGER.md；下列旧轮次中的此项待查现已闭合，其他待查项不变。

- 21:46 heartbeat修复插入/重排后同身份成员变化漏记；diff边界拒绝重复身份而非字典覆盖，见 REORDERED_MEMBER_DIFF.md。阈值分母、全局label分支丢失member ledger及宽松策略组合仍待查。

- 20:45 heartbeat修复raw引用遮蔽artifact变化及raw/artifact互换漏记，见 SOURCE_PAIR_DIFF.md。比较已使用双字段；旧before/after展示串仍非无损身份，结构化导出待查。

- 19:44 heartbeat修复同时成员变化掩盖label/data_revision重训与原因记录，见 COMBINED_VERSION_CHANGES.md。其余member语义字段组合、完整变更ledger与阈值分母仍需专项复查，不视为已验证。

- 18:44 heartbeat已修复RetrainPolicy非有限阈值与非bool开关输入缺口；36项新反例、86项限定回归通过，见 RETRAIN_POLICY_BOUNDARIES.md。本轮未推进FeatureSetVersion v2迁移，该项仍是下一轮优先任务。

- SearchConfig 活跃alias mutation为PARTIAL：09-13 05:48已加每轮/提案返回执行计划校验，738项搜索回归通过（LIVE_CONFIG_GUARD.md）；评估/验证/最终决策回调内漂移与并发瞬时修改仍需审查，不宣称线程安全。
- QE executable refs 的外部 resolver真实性边界，不可把typed envelope自身当外部账本认证。
- 新 FittedState身份的历史引用影响；FP compile新语义门下的旧合法/错声明配方迁移。
- 源码稳定后联合验收，当前FE/DataAccess仍由另一任务并行修改。
