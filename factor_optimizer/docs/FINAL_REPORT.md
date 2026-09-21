# 冻结后的 TEST 与全样本描述报告

入口位于 `research_final_report`，是**评估授权端**接口，不由候选搜索调用。
搜索完成后先 `freeze_selection(raw_batch, result, dataset_identity=...)`；
冻结包含原始/输出数组与有效掩码、轴、选中计划、时间分割及搜索配置的身份。
冻结不读取标签。后续缓冲区、计划或配置改变，会在打开测试标签前被拒绝。

## 后续落值失败不改写已冻结方案

通过 VALIDATION 的方案在应用到完整输入时，如果 TEST 区间的暴露出现晚到、
重复键等执行异常，返回 `status=materialization_failed` 和
`materialization_error`。保留原先的 plan_identity、selected_family、
TRAIN 增益和 VALIDATION 置信下界；此前已成功生成的验证前缀保持不变，
从 test_start 起的输出为 NaN、validity=False。

这不是切换为 RAW，更不是重新在 TEST 挑选方案。调用方必须显示失败，
不能把缺失值填零或将该因子当成完整可用结果。修正输入后应使用同一个
冻结 plan 重放，不因未来数据故障重新搜索。最终报告的冻结与身份核验
都会拒绝未解决的 materialization_failed，避免在已知无法生成输出时消耗
一次性的 TEST 标签访问机会。成功落值路径不改变既有结果。

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
- 每段 dates 与 RAW/selected 各自的 series 一一对齐，包含 rank_ic、
  net_return、turnover、nav、drawdown；不可用元素使用 JSON null。
- 缺失净值不能填零以制造回撤：回撤为未知时显示 `null` 和原因。
  Sharpe 可以基于已观测收益计算，但同时显示组合有效日期数。
- 单期非重叠日收益才计算组合风险；其他标签仍报告 IC，组合风险标为不可用。
- 20 层平均收益至少需要 20 个完整分层日期，每层至少 10 个有效标签；
  报告实际完整日期数，不把不足样本的图画成可靠诊断。

### 净值、回撤与未知持仓

已知收益区间的初始资本为 1，使用 QE 净值与回撤权威实现：

$$V_{-1}=1,\qquad V_t=V_{t-1}(1+r_t),$$
$$D_t=\frac{V_t}{\max(1,V_0,\ldots,V_t)}-1,\qquad MDD=-\min_t D_t.$$

series.drawdown 为非正数，汇总 max_drawdown 为正损失幅度。
无资本追加时，已观测到收益 -100% 后净值保持 0；低于 -100% 的收益
需要其他资本合约，本报告不画相应净值。估值缺失后，即使后续单日收益
再次可观测，累计净值与回撤仍未知，不能拼接已观测日冒充完整路径；
确知资本耗尽时按 QE 的零资本吸收规则处理。

信号不可用不等于明确空仓。当前或上一日目标持仓未知时，当日换手
不可确定，series.turnover 为 null，整段平均换手也为 null，并附
turnover_unavailable_reason。成本率非零时，相应成本后收益同样未知；
成本率为零时不因交易成本未知而删除原本可观测的当日毛收益。
相反，只有价格估值缺失、但信号持仓均已知，不会抹掉已知换手。

本次报告版本为 research-final.v2-curves，并纳入冻结 profile_hash。
旧版已消耗的 TEST 不能借升级版本重开；保留原来持久化的报告，
需要读取旧报告时由授权端读取原有存档，不创建新数据库或数据集别名。

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

曲线补充（2026-09-22）：报告专项 9 passed；优化库＋预处理库
1790 passed、1 xfailed、16 warnings（137.40 秒）。
先复现缺少曲线字段、信号缺失仍显示确定换手两个问题，再修复。
手算收益 +10%、-10% 对应净值 1.1、0.99 与回撤 0、-0.1；
后续估值缺失不允许净值曲线假装恢复。持仓未知的再交易成本显示缺失。
平台根测试仍有 14 项收集错误（56.80 秒），具体模块见方法审查记录。

真实 COS 因子 weekly_db5616cd85a477b3 经 DataAccess 读取，使用同一冻结
输出的 267 日 TRAIN 段核验曲线长度、严格 JSON、终值复利与最大回撤。
这是 training_diagnostic_only，不调用真实 TEST 授权，不新建状态库。
仅保存小型核验摘要：[真实曲线核验](final_curves_train_replay_20260922.json)。

使用带随机日收益的合成数据验证真实 broker/SQLite 跨实例持久行为，不模拟其权限判定。
覆盖成功报告、缓存复用、冻结输出篡改、成本配置更改、无持久权限和读取后失败。
真实 COS TEST 还没有打开：自动化整改尚未全部收尾，必须保留最终独立检验机会。

后续落值隔离补充（2026-09-22）：晚到暴露和重复键两个仅发生于 TEST
日期的反例，旧实现都错误改变已冻结 plan_identity；修复后保留已选方案、
已验证前缀与验证统计，后续值缺失且最终报告拒绝。专项测试 20 passed；
两库全套 1770 passed、1 xfailed、16 warnings（137.77 秒）。
根测试仍为 14 项既有收集错误（57.58 秒），清单同 METHOD_AUDIT_20260922.md。

真实 COS 因子 alphasage_20260909062020_09e3f39d，经 DataAccess 读取，
500 日 × 256 股票，bootstrap_draws=99，修改前后完整搜索 95 个候选，
输出数组、候选评分记录、方案身份、状态与验证下界完全一致，均保留 RAW。
这验证成功路径未变，不代表该因子得到收益提升；未计算 TEST 指标，也
不是性能加速实验。来源及内容校验见
[真实对照证据](materialization_real_ab_20260922.json)。

2026-09-22 验证：优化库与预处理库 **1,724 passed、1 xfailed、16 warnings**，
123.66 秒。XFAIL 为既有离线 HP 滤波的前缀不变性。
平台根测试仍有 14 项既有收集错误，56.47 秒；具体文件清单见
[既有问题记录](METHOD_AUDIT_20260922.md)，没有计入通过或关闭检查。
