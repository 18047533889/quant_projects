# 候选集载体静默转换（2026-09-14 08:10 heartbeat）

server-c正式main原树，剩774 GiB，FE/DA任务未活动；保留已有修改，无分支/commit/push/部署/生产数据操作。

候选集原直接tuple迭代：字符串被拆字符，bytes/bytearray变成ASCII整数，Mapping丢弃值只取键，可静默改变搜索范围。共享_candidate_grid现拒绝这些载体，冻结和拟合入口同时覆盖，保留list/tuple/迭代器及原合法值哈希。

11项小型合成测试：8项两个入口的四种错误载体反例，3项合法数值载体选择兼容性。未使用生产数据。

- grid_carrier_before：8 failed / 3 passed，源码稳定。日志SHA256 03cdea260db90a42c2b7bdfbc5b24cf648075db21ae9763bc80904e7ae915802。
- grid_carrier_final：966 passed / 12 warnings，无skip，预算搜索限定回归。
- 源码前后摘要一致：962b0b7d10cf19f085a39269eaf8a3f8e6014556db1b638b84bb90b8541fd7e3。
- 最终日志SHA256：df596698bed8688db878acab66c2fe0261821812b9f42c8dbf6ff3a500f1e77b。
- git diff --check通过；HEAD e94ac507d670fd1c16b1d6a63fc5d6286daa5970。

网格元素其他可转换类型及拟合前有效性校验顺序仍待查；不宣称所有数值类型闭合。codec迁移等其他开放项保留。
