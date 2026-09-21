# 冻结后的 TEST 与全样本描述报告

入口位于 `research_final_report`，是**评估授权端**接口，不由候选搜索调用。
搜索完成后先 `freeze_selection(raw_batch, result, dataset_identity=...)`；
冻结包含原始/输出数组与有效掩码、轴、选中计划、时间分割及搜索配置的身份。
冻结不读取标签。后续缓冲区、计划或配置改变，会在打开测试标签前被拒绝。

## 一次性读取与持久化

`evaluate_frozen(frozen, broker)` 必须获得既有 `TestAuthorityBroker`，
并且 broker 使用 `SQLiteCampaignStore` 的持久封存权限。
没有持久权限、候选集/数据集/分割/报告配置不匹配都会拒绝。
broker 的 purpose 必须为 `research_final_report`。

授权端的 opaque store 提供对齐的完整 LabelBundle，用于用户明确要求的两个报告。
数据读取仍应通过 DataAccess；不把标签回传给搜索，不提供重新挑选候选的回调。
授权端应在独立任务/进程中运行；此 Python 接口不是对恶意同进程代码的安全沙箱。

持久数据库按**数据集身份 + 分割身份**封存：更换 campaign 名称或 purpose
不能重新开放相同测试范围。配置、候选集同样绑定。
完成后再次调用只返回已保存 JSON，不重新读取标签或计算新结果。
错误发生在读取后，同一次测试已被消耗，不能伪装为基础设施错误重试。

必须复用平台指定的持久数据库和真实数据集身份；不得为了重跑而新建数据库、
删除状态、伪造数据集别名或变更分割名称。本接口不声称能阻止管理员越权绕过治理。

## 输出口径

- `test.role = held_out_final`：仅冻结分割中的 TEST，最终样本外描述。
- `full_sample.role = descriptive_only`：所有输入日期，含训练与验证；
  不能当成独立样本外证据，也不能据此再次选择当前测试范围的候选。
- 每个因子同时报告 RAW 与最终选中值，不对失败候选做 TEST 排名。
- RankIC、未年化 RankICIR、成本后 Sharpe、最大回撤、平均全名义换手、
  最差三分块 Sharpe、20 层平均收益与有效日期数均使用 QE 指标权威实现。
  公式、五分位组合及成本口径见 [联合选优公式](JOINT_SELECTION.md)。
- 每段报告独立从现金开始并计入建仓；不是从训练段无成本继承组合。
- 缺失净值不能填零以制造回撤：回撤为未知时显示 `null` 和原因。
  Sharpe 可以基于已观测收益计算，但同时显示组合有效日期数。
- 单期非重叠日收益才计算组合风险；其他标签仍报告 IC，组合风险标为不可用。
- 20 层平均收益至少需要 20 个完整分层日期，每层至少 10 个有效标签；
  报告实际完整日期数，不把不足样本的图画成可靠诊断。

搜索结果对象保持 `test_evaluated=False`，表示搜索当时未用 TEST；
独立保存的最终报告才带 `test_evaluated=True`。二者不能混为同一个阶段。

## 接线示意（在授权端，而非搜索端执行）

```python
broker = TestAuthorityBroker(
    dataset_identity=frozen.dataset_identity,
    provider_identity=trusted_provider_identity,
    search_session_id=completed_search_identity,
    split_id=frozen.split_identity,
    campaign_store=existing_durable_campaign_store,
    campaign_id=existing_campaign_id,
    candidate_set_hash=frozen.selection_hash,
    profile_hash=frozen.profile_hash,
    purpose="research_final_report",
)
broker.attach_store_ref(authoritative_label_store_ref)
report = evaluate_frozen(frozen, broker)
```

以上依赖必须来自平台既有授权配置，不能任意创建新的数据集或状态库绕过限制。
报告不会因为 TEST 表现差就修改冻结因子；无生产准入、收益保证或无 bug 保证。

## 当前验证边界

使用带随机日收益的合成数据验证真实 broker/SQLite 跨实例持久行为，不模拟其权限判定。
覆盖成功报告、缓存复用、冻结输出篡改、成本配置更改、无持久权限和读取后失败。
真实 COS TEST 还没有打开：自动化整改尚未全部收尾，必须保留最终独立检验机会。

2026-09-22 验证：优化库与预处理库 **1,724 passed、1 xfailed、16 warnings**，
123.66 秒。XFAIL 为既有离线 HP 滤波的前缀不变性。
平台根测试仍有 14 项既有收集错误，56.47 秒；具体文件清单见
[既有问题记录](METHOD_AUDIT_20260922.md)，没有计入通过或关闭检查。
