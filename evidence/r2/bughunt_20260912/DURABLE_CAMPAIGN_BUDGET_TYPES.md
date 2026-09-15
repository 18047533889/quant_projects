# 持久化campaign预算创建类型（2026-09-14 00:05 heartbeat）

server-c main原树，剩774 GiB，FE/DA任务未活动；未修改FE/DA、未分支/commit/push/部署/操作生产数据。保留已有差异。

create_campaign绕过SearchBudget的类型校验，max_evaluations接受bool/浮点/正无穷，NaN漏入数据库层，max_cost接受bool。小数次数可形成非整数上限，无穷次数可失去次数约束。现事务前要求次数为真正int，成本拒绝bool，保留原有正值/有限成本校验及合法重复创建语义。

13项小型临时SQLite测试覆盖新建与已有campaign的非法输入拒绝、重新打开后行不变，以及合法int次数/数值成本的重开和次数限制。冲突异常与类型异常区分：前置12失败不全是成功写入，其中部分为未提前校验而抛错类型不符。

- campaign_types_before：12 failed / 1 passed，源码稳定。日志SHA256 c8c2d867a982d47436de4e797a4f28ae2d55cffe617a02e37b37f8e2771bdef8。
- campaign_types_final：883 passed / 12 warnings，无skip；限定搜索与预算回归。
- 源码摘要前后相同：0a7fde8527e549e17e3b183def6202d6854a04f2c2ed809d123eae02ddc17e1c。
- 最终日志SHA256：1bc94802d9027fff313f46b0198916f2796074dc4af4d41c2470426649ac34fd。
- git diff --check通过，HEAD e94ac507d670fd1c16b1d6a63fc5d6286daa5970。

不自动迁移历史非法预算。campaign_id/attempt_id载体与别名冲突仍待审查；哈希codec迁移等开放项仍未完成。未宣称生产或全项目通过。
