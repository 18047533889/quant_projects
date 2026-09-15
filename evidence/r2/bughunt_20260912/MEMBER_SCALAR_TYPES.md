# 成员标量输入边界（2026-09-13 00:45 heartbeat）

server-c正式main原树，剩776 GiB。另一FE/DA任务idle，本轮未触其文件。
保留已有修改，未commit/push/建分支/复制/改写生产记录。

## 复现与修复

FeatureMemberRef的位置仅做负数比较，接受bool、小数及NaN；引用仅检查truthiness或无检查，接受list/dict/int/bool。
这允许可变对象进入冻结DTO、破坏身份索引或在摘要生成后改变内容。
位置现在要求非负int且排除bool；必填身份字段要求非空字符串；可选字符串字段只接受str或None。
保留可选空字符串兼容，未改动security enum和metadata合同，也不重写历史坏记录。

member_scalar_before：51 failed / 1 passed，保留前置证据。
member_scalar_final：98 passed（52项新增及相邻完整性/diff/模型桥），源码before=after。
源码digest：4a1e55ca0c507cb2e9b9c884d8416c9551cd62426326ab26f7b400caac42fc0c。
限定日志SHA256：eee437f9deb2a38b42ea76c8c3f6965a1ede7d4da26b73ed7747936d83e700bf。

## 剩余

最终平台member_scalar_platform_full：649 passed / 1 skipped / 3 warnings；源码before=after与限定回归一致。
平台日志SHA256：2b8320401393ad9b0b7c718241cf3129bc01e809ea157d22bcd7ad1ca936fe1b。
排除live PostgreSQL文件并移除QP_PG_DSN；1项disposable fencing测试跳过，不作为通过或生产认证。
git diff --check（quant_platform与本轮证据目录）通过。

旧记录如使用错误类型将明确拒绝，不自动类型转换。公共v1哈希碰撞、v2迁移、阈值分母等仍开放。
