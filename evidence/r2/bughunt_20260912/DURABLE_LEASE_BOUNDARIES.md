# 预算预留租期边界（2026-09-13 23:03 heartbeat）

server-c 正式 main 原树，剩余775 GiB；FE/DA任务未活动，未修改其文件。保留已有改动，无分支/commit/push/部署/生产数据库操作。

reserve 原仅检查 lease_seconds <= 0：True 被接受为1秒，正无穷生成永不到期预留，NaN 进入SQLite后才触发 NOT NULL IntegrityError。现事务开始前拒绝bool及非有限/非正租期，非法输入不触发过期清理或预算更新。整数及小数正租期保留。

11项小型SQLite测试：9个非法值，2个合法租期；非法输入后重新打开数据库确认原账本/预留不变；固定时钟验证合法预留到期后预算重新可用。无真实等待或生产数据。

- lease_boundaries_before：3 failed / 8 passed，源码稳定。日志SHA256 69ca2c57f21a7665963d94aec861e5116a34be6c5ae02015fb8ae886667d1cc3。
- lease_boundaries_final：870 passed / 12 warnings，无skip；限定预算及搜索回归。
- 源码前后摘要一致：057aa633eeef7c942cf5ce01cf81724cf0c9522d07ae6f63ac9dc48fe6e32621。
- 最终日志SHA256：e62b64c7cdf13a0ca2ff1c0838c5cff68b57cd506034ca19e9dc4db2858ce6b4。
- git diff --check通过，HEAD e94ac507d670fd1c16b1d6a63fc5d6286daa5970。

不自动迁移历史非法租期。campaign创建直接入口的预算类型、attempt_id载体等仍需审查；哈希迁移与其他开放项不因此闭合。
