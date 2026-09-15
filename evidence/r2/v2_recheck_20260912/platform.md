# V2 Platform 差额整改证据

正式树：/home/sunhaiwei/quant_projects；基线 HEAD e94ac507d670fd1c16b1d6a63fc5d6286daa5970；main；未 commit/push。
本文件仅报告当前任务的代码和有界测试，不认证生产环境。

## 已修改

- PL-01：非 Mapping 原始记录、不可转换/溢出的数值参数域转为 CandidateNormalizationError；报告使用独立 ingestion_record_id，content_hash=None，不因错误报告再次抛错。
- PL-02：run 新增可选 evaluation_context_ref。不同数据/政策的调用方应提供新的不可变引用；工作键、批次、事件和候选字节绑定此上下文，不替代或生成因子语义身份。
- PL-02：消费记录使用独立 operational work ID，修复旧 candidate_id 主键吞掉第二个上下文的问题。恢复完整 terminal discovery 数据，保留同上下文 semantic/spec 冲突。持久化 reservation attempt 取代进程内计数作为持久化 job attempt 键；临时错误后重启可重新执行。
- PL-02：明确无证据的 terminal outcome 会记消费；无法恢复的已完成 job 不再误报无证据并永久消费，而是停止且保留重试资格。
- PL-03：显式未成熟标签映射 DATA_UNAVAILABLE，保留为可重试；字符串 false 不能隐式变 True。其他原始 JobError 分类继续保留。
- 清除 Pipeline 内重复 _library_ref 方法，并用 class AST 反例防止方法被后定义悄悄覆盖。
- OPS08：hermetic shadow replay fixture 显式 allow_research=True，符合 FP 新门控；没有改 FE 算子门或生产发布资格。
- 检验工具支持独立 --output-dir，避免覆盖旧验收目录。

## 新增反例

quant_platform/tests/test_v2_ingestion_boundaries.py 覆盖非字典坏记录、混合好坏记录、巨大数值转换、不同数据/policy intent 重评、多个 context 后重启混批去重、持久化 semantic 冲突、持久化 job 重启重试、未成熟标签等待、字符串成熟标记拒绝以及 class 方法重复。

## 实测记录

- platform_prefix：7 failed，记录了修复前坏记录与重复方法反例。
- platform_delta：128 passed，1 PostgreSQL skipped，源码稳定。
- platform_full：474 passed，1 failed，1 skipped。唯一失败为 shadow fixture 未显式 research opt-in，已修。
- platform_delta2：66 passed，但运行期间 QE 集成测试文件变化，保留 SOURCE_CHANGED_DURING_RUN。
- platform_delta3：55 passed，源码稳定。
- platform_final：479 passed，1 skipped；运行期间独立 FE 任务修改源码，保留 SOURCE_CHANGED_DURING_RUN。此轮在最后 maturity 分类两项改动之前。
- platform_pending_final：57 passed，源码稳定；含最后 maturity 分类及此前所有平台 delta。
- canonical_sources：1 passed；现存顶层重名/模块包碰撞扫描通过。类内新增断言仅限定 Pipeline，不冒充全仓 class 图审计。

每项 JSON 与 log 同名，保留确切 command、时间、源哈希、变更清单和日志哈希。PostgreSQL live suite 被明确排除；另一个 live fencing 用例因无授权 disposable DSN 跳过。

## 迁移与剩余边界

- evaluation_context_ref 是生产者提供的引用，不是已配置的可信 resolver。必须绑定真实数据、policy、样本和其他评价输入；重用错误引用仍可能错误重放。此任务未编造跨包内容权威。
- 历史 consumed_manifests 只有 hash、没有保存 discovery 的记录无法补回未记录的语义。保留旧记录；需要审计后用新的显式上下文合法重评，未删除/改写历史生产状态。
- 默认 in-memory 流仍只是研究路径；已有 SQLite durable generation/CAS/lease 测试不等于真实 COS/消息总线/PostgreSQL 原子生产闭环。
- 生产者正确传入新 intent、trusted evidence resolver、真实模型消费与旧 worker 运行身份仍需联合验收。
- 因其他 FE 任务仍在修改正式树，不能把分时通过的各包测试合成同一冻结代码版本的全栈 PASS。
