# 同时变更漏判修复（2026-09-12 19:44 heartbeat）

server-c /home/sunhaiwei/quant_projects main 原树修改，磁盘剩776 GiB。
FE/DA另一任务已结束，未触及其文件；保留前轮全部修改。未commit/push或改写生产资产。

## 缺陷

分类函数先返回成员数量/身份、treatment或schema变更，导致同时发生的label/data_revision变更未进入原因集合。
新增成员默认不重训，宽松schema策略也可不重训，因此版本级失效会被成员分类掩盖。
label和data_revision同时变化时也只保留一个原因。

## 修复

保留原scalar category和reason投影；公共分类入口独立补齐CHANGED_LABEL、CHANGED_DATA_REVISION。
重训决策优先处理这两个不可被成员策略豁免的原因，两快照和预分类diff调用均覆盖。
纯新增成员仍不强制重训。不更改历史版本哈希、数据库记录或任务键。

## 验证

新增10项组合测试：add/schema/treatment/metadata与label/data_revision组合、双版本变更、纯新增负对照。
combined_version_before：7 failed / 3 passed，失败证据已保留。
combined_version_final：73 passed，源码before=after，无并发变更。
源码digest：4a3512aa65609dca9f8c24d12ec3d36f0724f4b97ea62bb7c3376602b3688bb3。
限定日志SHA256：376c76308fb1c7eea65fd9131120425353ae30902403249f94b77ef953796919。

## 限制

最终平台回归 combined_version_platform_full：568 passed / 1 skipped / 3 warnings，源码before=after与限定回归一致。
全平台日志SHA256：25f8e7a487c1b5b01601b551b5e5a2173df4c57effdf68cb7e14def1a840eabc。
显式排除live PostgreSQL文件且移除QP_PG_DSN，另1项disposable fencing测试跳过，不算生产通过。
git diff --check（quant_platform及本轮证据目录）通过。

本轮仅闭合版本级label/data_revision被掩盖问题，未宣称所有组合变更均已验证。
FeatureSetVersion v2迁移仍待进行，旧公共哈希碰撞保持开放。
