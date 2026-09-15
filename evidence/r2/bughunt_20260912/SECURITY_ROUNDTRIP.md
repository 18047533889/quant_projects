# 安全等级JSON恢复类型（2026-09-13 04:48 heartbeat）

server-c正式main原树，剩776 GiB，FE/DA任务idle。保留现有改动，不commit/push/建分支/复制或改写生产记录。

durable generation把security enum保存为value，恢复FeatureMemberRef时原来保留str。
因此相同哈希的恢复版本出现成员类型差异，diff误记CHANGED_VERSION并触发重训。
现将合法字符串精确转回SecurityClassification，拒绝未知字符串及其他错误类型；None仍兼容，不降级或默认等级。
v1哈希本来将enum编码为value，因此合法旧摘要保持不变。

security_roundtrip_before：10 failed（5个等级往返、5种错误类型），保留证据。
security_roundtrip_final：47 passed，源码before=after，无并发变化。
源码digest：eedfcaec83aee8c10b4673bfdd2bfde10e43f6f5db738c435111f4cf5ba6bbd5。
限定日志SHA256：60b3ad445ba23f6e87032ba40f64bde85192171e5aabd06c19191c2471a5e3d4。

最终平台security_roundtrip_platform_full：689 passed / 1 skipped / 3 warnings，源码before=after与限定回归一致。
平台日志SHA256：765a3b344a83003cf2368c46803b5aea69493f9bb94596ce15a9bd0aff3c7c47。
排除live PostgreSQL文件并移除QP_PG_DSN；1项disposable fencing测试跳过，不作为通过或生产认证。
git diff --check（quant_platform与本轮证据目录）通过。

本轮只修显式security字段，metadata中日期/枚举的通用类型保持及v2迁移仍未闭合。
