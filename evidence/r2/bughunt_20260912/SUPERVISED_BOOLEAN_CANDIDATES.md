# 冻结参数布尔候选（2026-09-14 07:10 heartbeat）

server-c正式main原树，剩774 GiB，FE/DA任务未活动；保留已有修改，无分支/commit/push/部署/生产数据操作。

FrozenSupervisedParameter及fit_supervised_parameter将bool候选转为0/1；selected value的bool通过数值成员检查，train_scores的bool键也与0/1相等，可能错误使用训练证据。现转换前拒绝grid bool、选择值bool以及评分键bool，保留既有合法数值与哈希公式。

9项小型合成测试：8项False/True在四个入口的反例，1项合法数值选择及带hash重建。无数据集或生产数据库。

- supervised_bool_before：8 failed / 1 passed，源码稳定。日志SHA256 bbc879c3e009a60af5e501b7da6a5994b4163b181aa82e52bf5e594329835fec。
- supervised_bool_final：955 passed / 12 warnings，无skip，预算搜索限定回归。
- 源码前后摘要一致：3fc273b419a9674e397cd6a4c6c6f987d319ddc8db77bcc7fa8593664cc6d9cd。
- 最终日志SHA256：334162024e505690fbad9602df2e824953eb202f01775eea11d47b5bd80a2626。
- git diff --check通过；HEAD e94ac507d670fd1c16b1d6a63fc5d6286daa5970。

不自动迁移旧布尔状态。grid其他可转换类型、空/重复/非有限网格的拟合前校验顺序仍待审查；codec迁移等其他开放项不因此闭合。
