# 重复预留请求的金额身份校验

2026-09-13 22:02 heartbeat：server-c 正式 main 原树，磁盘剩余775 GiB。FE/DA任务未活动；未改其文件。未commit/push/部署/生产数据操作，保留已有改动。

SQLiteCampaignStore.reserve 对已有 campaign_id/attempt_id 只查看状态，忽略请求的 estimated_cost 变化，可能将9单位请求按旧1单位预留成功返回。现在同一事务中核对原金额，冲突抛 CampaignStateError；相同金额重试的原有返回规则不变，终态不被重新打开。

小型临时SQLite覆盖 RESERVED/STARTED/SETTLED/RELEASED 四个状态；冲突后重新打开数据库确认状态、金额、预算不变，随后同金额重试保持兼容。

- reservation_replay_before：4 failed，源码前后稳定。日志SHA256：82fc610513f2ca70d6b2c7dcee6bea7fd72fd0e7302a83e67439f863d8bdb3dc。
- reservation_replay_final：859 passed，12 warnings，无skip；搜索与预算限定回归，非生产或全项目认证。
- 源码前后摘要一致：ebf346a6fdc8b6b5a11523d5ed2d9fc5c839649d47c1bfa1946f1f1bd70c4eec。
- 最终日志SHA256：1efbf40d948118ec62babdf4bb2dd65873dbb4cd25aa9ac487d6cdaf6ad0736a。
- git diff --check通过，HEAD e94ac507d670fd1c16b1d6a63fc5d6286daa5970。

lease_seconds非有限数和类型边界仍待复现检查；其他未闭合项不变。
