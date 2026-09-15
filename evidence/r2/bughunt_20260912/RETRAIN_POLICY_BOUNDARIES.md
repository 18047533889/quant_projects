# 重训策略输入边界（2026-09-12 18:44 heartbeat）

正式 server-c main 原树直接修改；磁盘约剩776 GiB。FE/DA任务已结束，本轮未修改其文件。未提交、推送或写生产数据。

## 复现与修复

RetrainPolicy 原先允许 NaN ratio/count、无限count、非整数count和非bool开关。
NaN比较恒为false，可使阈值比较失效；字符串 "false" 为truthy，会误开启新增成员重训。
现要求开关为bool，ratio为有限int/float且在(0,1]，count为非负int（排除bool）。
合法默认值和边界值保持不变；不自动纠正旧无效配置，调用方需提供符合合同的类型。

新增 test_retrain_policy_boundaries.py：36项，包括畸形类型、非有限数和合法边界。
修复前 retrain_policy_before：26 failed / 10 passed；保留失败日志。
修复后 retrain_policy_final：86 passed，包含相邻diff、模型桥、版本完整性和codec测试。
源码before=after：63da4395d77327431de673b9991eadaa2342e49f38afc4c06d5fc9208e257e32。
日志SHA256：0d54d1a44b70de3efc6ce0b7674459323183d64b9ebe41f9f3822ac6fea23a15。

## 范围限制

最终平台回归 retrain_policy_platform_full：558 passed / 1 skipped / 3 warnings，稳定PASS，源码before=after与限定回归一致。
日志SHA256：c83b2a8997586320b87290214797be255f389edc371e636fac0f4d3a17cf9a02。
排除test_postgres_live.py并移除QP_PG_DSN；1项disposable PostgreSQL fencing测试因此跳过。
git diff --check（quant_platform与本轮证据目录）通过。

本轮只修策略输入校验；FeatureSetVersion v2持久化迁移未实施，P0旧哈希碰撞仍开放。
没有调整重训决策规则或历史版本ID。跳过的测试不当作通过，未进行生产认证。
