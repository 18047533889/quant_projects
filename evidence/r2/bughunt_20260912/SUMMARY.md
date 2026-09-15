# 第二轮缺陷整改与持续复查

代码都在 server-c /home/sunhaiwei/quant_projects 的 main 工作树；无新分支、worktree、commit、push、部署或正式数据删除。保留其他 AI 修改。子代理均 GPT-5.6-sol。

## 本轮已修复

- 09-14 10:11 heartbeat：训练评分捕获一次后统一校验/选择，防止动态值改变结果或NaN绕过校验，前置2失败，977项预算搜索稳定通过，见 SUPERVISED_SCORE_SNAPSHOT.md。

- 09-14 09:10 heartbeat：候选集有限/唯一/非空检查前移，非法网格不读取训练评分，前置8失败1通过，975项预算搜索稳定通过，见 SUPERVISED_GRID_PREFLIGHT.md。

- 09-14 08:10 heartbeat：候选集错误载体拒绝，防止字节码/字典键被误当候选数值，前置8失败3通过，966项预算搜索稳定通过，见 SUPERVISED_GRID_CARRIERS.md。

- 09-14 07:10 heartbeat：冻结参数候选/选中值/训练评分键拒绝bool，前置8失败1通过，955项预算搜索稳定通过，见 SUPERVISED_BOOLEAN_CANDIDATES.md。

- 09-14 06:10 heartbeat：冻结参数存储读写标识校验，前置6失败1通过，946项预算搜索稳定通过，见 SUPERVISED_STORE_IDENTIFIERS.md。

- 09-14 05:09 heartbeat：假设账本因子身份拒绝错误载体，前置6失败1通过，939项预算搜索稳定通过，见 HYPOTHESIS_SPEC_IDENTITY.md。

- 09-14 04:09 heartbeat：假设组汇总与完整性检查的输入类型校验，前置6失败1通过，932项预算搜索稳定通过，见 FDR_FAMILY_INPUTS.md。

- 09-14 03:09 heartbeat：假设账本结果标志强制bool，防止执行/p值计数失真，前置10失败5通过，925项预算搜索稳定通过，见 HYPOTHESIS_OUTCOME_FLAGS.md。

- 09-14 02:07 heartbeat：预算查询拒绝int/bool标识误命中，6项前置失败；并发FE报告变更后复测910项稳定通过，见 DURABLE_BUDGET_QUERY_IDENTIFIER.md。

- 09-14 01:05 heartbeat：预算写入口拒绝错误标识类型，防止数字/bool匹配已有字符串键，20项前置失败、903项预算搜索稳定通过，见 DURABLE_BUDGET_IDENTIFIER.md。

- 09-14 00:05 heartbeat：持久化预算创建入口拒绝非int次数与bool成本，前置12失败1通过，883项预算搜索稳定通过，见 DURABLE_CAMPAIGN_BUDGET_TYPES.md。

- 09-13 23:03 heartbeat：预留租期拒绝bool/非有限数，保留正常到期释放；前置3失败8通过，870项预算搜索稳定通过，见 DURABLE_LEASE_BOUNDARIES.md。

- 09-13 22:02 heartbeat：持久化预算重复请求金额冲突校验，4项前置失败，859项预算/搜索回归稳定通过，见 DURABLE_RESERVATION_REPLAY.md。其余开放项不因此闭合。

- 平台：批内身份冲突、语义变更被错判重放、一次性迭代器指纹丢失、日期持久化错误、非法 carrier 字段、准入 DTO 内嵌可变数据，以及 OPS08 错误语义声明。
- FA/FO：诊断/分类政策及全局目录可变、共享 config 绕过恢复计划校验、证据仓读回未验证内容和查询键。
- QE：制品时间/因子轴校验、Probe/Executable 序列化和哈希隔离、多因子逐因子执行账本绑定。
- FP：拟合状态身份绑定实际训练窗、执行时验证真实面板时间轴、配方编译绑定注册语义/stage/fit-kind。

## 实测结果

- QE + adapters：1260 passed，2 skipped。
- FA + FO 全量：2398 passed，58 warnings；最终 wrapper 以实际 JSON 为准。
- FP：417 passed，1 xfailed，4 warnings；有运行期间其他源码变化。
- 平台：499 passed，1 skipped；有运行期间其他源码变化。
- 平台最终限定合同集：138 passed，稳定 PASS。

不同范围有重叠，不相加当唯一样本数；skip/xfail 和并发源码标记不当作生产通过。详细变化、测试及兼容边界分别见 platform.md、qe.md、fp.md、fa_fo.md。

## 仍有重要问题

公共 _contenthash 的结构/类型碰撞已做首批兼容修复：新增v2 codec，FeatureSetArtifact新写v2、旧读v1，113项限定回归稳定通过（HASH_CODEC_PHASE1.md）。其他消费者尚未迁移，仍在 OPEN_FINDINGS.md 保持最高优先级，未冒充全部完成。

FP fitted-state、QE artifact kind/hash、平台 batch fingerprint 的身份规则有变化；历史资产没有自动改写，需要显式影响审计/迁移。

## 持续机制

09-13 19:58 heartbeat修复持久化布尔成本，前置4失败、预算与搜索847通过，见 DURABLE_BOOLEAN_COST.md。

09-13 18:56 heartbeat修复预算操作布尔成本，前置10失败、预算与搜索843通过，见 BUDGET_BOOLEAN_OPERATIONS.md。

09-13 17:56 heartbeat修复底层预算上限类型，前置20失败、预算与搜索833通过，见 BUDGET_LIMIT_TYPES.md。

09-13 15:54 heartbeat拒绝布尔评估成本，前置2失败、搜索794通过，见 BOOLEAN_EVALUATION_COST.md。

09-13 14:54 heartbeat修复执行模式类型缺口，前置6失败、搜索788通过，见 EXECUTION_MODE_TYPES.md。

09-13 13:54 heartbeat修复搜索开关类型缺口，前置14失败、搜索781通过，见 SEARCH_SWITCH_TYPES.md。

09-13 11:53 heartbeat修复搜索配置数值边界，前置14失败、搜索765通过，见 SEARCH_NUMERIC_BOUNDARIES.md。

09-13 10:52 heartbeat修复停滞检测漂移后的额外提案调用，前置1失败、搜索749通过，见 PLATEAU_CONFIG_GUARD.md。

09-13 09:50 heartbeat修复阶段决策漂移后仍推进候选，前置2失败、搜索747通过，见 STAGE_DECISION_CONFIG_GUARD.md。

09-13 08:49 heartbeat修复FO最终决策配置漂移未拒绝，前置3失败、搜索745通过，见 FINAL_DECISION_CONFIG_GUARD.md。

09-13 07:49 heartbeat修复评估漂移后仍解析输出的问题，保留已执行成本结算；前置1失败、搜索742通过，见 EVALUATOR_CONFIG_GUARD.md。

09-13 06:49 heartbeat修复FO验证回调配置漂移检测过晚：前置3失败、搜索741通过，见 VALIDATOR_CONFIG_GUARD.md。

09-13 05:48 heartbeat转查FO，修复迭代/提案边界配置漂移未拒绝：前置2失败、搜索目录738通过。运行期隔离仍PARTIAL，见 LIVE_CONFIG_GUARD.md。

09-13 04:48 heartbeat修复安全等级JSON往返假语义变化，前置10失败、限定47通过，合法旧hash不变，见 SECURITY_ROUNDTRIP.md。

09-13 02:47 heartbeat修复版本标量类型缺口，前置20失败、限定稳定复测58通过；首轮有FE测试并发变化不算稳定PASS，见 VERSION_SCALAR_TYPES.md。

09-13 01:46 heartbeat修复元数据载体静默转换/丢失，前置7失败、限定46通过，见 METADATA_CARRIER.md。

09-13 00:45 heartbeat收紧成员标量合同，防止可变引用和非整数位置进入身份计算；前置51失败、限定98通过，见 MEMBER_SCALAR_TYPES.md。

23:45 heartbeat修复强制语义变化漏重训及摘要不一致，前置8失败、限定50通过，见 SEMANTIC_REASON_RETRAIN.md。

22:45 heartbeat补齐label/data_revision分支成员变化记录及事件原因/计数；前置6失败、限定52通过，见 GLOBAL_MEMBER_LEDGER.md。

21:46 heartbeat修复重排/插入后成员属性变更漏记，增加重复身份拒绝：前置8失败、限定52通过，见 REORDERED_MEMBER_DIFF.md。

20:45 heartbeat修复来源双字段比较漏记（原raw or artifact），前置4失败、限定43通过。维持纯来源变化不强制重训，见 SOURCE_PAIR_DIFF.md。

19:44 heartbeat修复label/data_revision与成员同时变更时被早返回掩盖的问题：保留旧分类但完整记录版本失效原因并强制重训，限定73项通过，见 COMBINED_VERSION_CHANGES.md。

18:44 heartbeat修复RetrainPolicy的NaN/无限及错误类型输入校验：修复前26失败，修复后限定86通过，见 RETRAIN_POLICY_BOUNDARIES.md。哈希v2迁移仍未完成。

最新heartbeat还修复了FeatureSetVersion盲信外部摘要和成员/元数据可变问题。最终限定123项稳定通过；阶段性完整平台521 passed/1 skipped。保留旧哈希公式、没有改写历史版本；v2消费者迁移仍未整体完成，见 FEATURE_VERSION_INTEGRITY.md。

09-13 本轮修复持久化预算结算/释放忽略调用方预留金额，前置8失败、预算与搜索855项稳定通过。原子校验与状态规则保留，见 DURABLE_RESERVED_AMOUNT.md。

已创建本任务每小时一次的 ACTIVE heartbeat：server-c（server-c 持续缺陷复查）。每次小而有界地继续找真实缺陷、补反例、修复和验证；无新发现时保持安静，重要修复/阻断/授权问题才通知。不扩大生产权限、不自动 commit/push。OPEN_FINDINGS.md 是下轮首要入口。
