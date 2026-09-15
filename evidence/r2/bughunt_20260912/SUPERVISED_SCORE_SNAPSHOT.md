# 训练评分校验/使用一致性（2026-09-14 10:11 heartbeat）

server-c正式main原树，剩766 GiB，FE/DA任务未活动；保留已有修改，无分支/commit/push/部署/生产数据操作。

fit_supervised_parameter原分别使用Mapping.values()校验、__getitem__选参：动态映射二次读取改变排序；不一致映射可让values返回有限数、getitem返回NaN而仍冻结结果。现捕获一次键值到普通dict，再基于同一份值检查完整性/有限性及选择。合法hash公式及tie-break不变。

2项小型合成Mapping反例：重复读取反转选择，以及values/getitem不一致绕过有限性。修复后每个键只读取一次并拒绝实际NaN。未连接外部评分服务或生产数据。

- score_snapshot_before：2 failed，源码稳定。日志SHA256 76c75b020c153383a255c239bf3f0a84d9186941062de6e2d6c286b23d90d5c0。
- score_snapshot_final：977 passed / 12 warnings，无skip，预算搜索限定回归。
- 源码前后摘要一致：d4bb2a81d9d064ccef392eb5ca8f452ea8c7d37531d0f176d42b5799d19d22a4。
- 最终日志SHA256：0bbcb3e675ddb17145cdb2474527346b00c472a2a5ca2fbe3649c161e9599ed0。
- git diff --check通过；HEAD e94ac507d670fd1c16b1d6a63fc5d6286daa5970。

这不是外部Mapping的原子跨键快照或线程安全保证；捕获期间源并发变化仍需源端契约。评分键/身份/repair_family前置验证顺序与其他数值类型仍待查，codec迁移等开放项保留。
