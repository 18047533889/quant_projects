# 元数据载体静默转换（2026-09-13 01:46 heartbeat）

server-c正式main，磁盘776 GiB；FE/DA任务无活动，未触其文件，保留现有改动。
未commit/push/建分支/复制或改写生产记录。

FeatureMemberRef原dict(metadata or {})会丢弃返回False的非空映射；也接受错误空值及键值对序列，重复键静默覆盖。
现在要求Mapping或历史兼容None；只用is None判断空值，不根据truthiness丢弃数据。
9项新测试：修复前7 failed / 2 passed（metadata_carrier_before），修复后相邻回归46 passed（metadata_carrier_final）。
限定源码before=after：086abf67c46d667b65c9370bebdc8b9d4c066060dff59c8372024daff04b71ce。
限定日志SHA256：72b5060946bbb73d33a5d78eb4b3ef70c9ee59f0653121b9e09a5d61ecfb9c97。

最终平台metadata_carrier_platform_full：658 passed / 1 skipped / 3 warnings；源码before=after与限定回归一致。
平台日志SHA256：a54724fb6ef2f5b1e0d51779d1fc7d45e44d19b3e6e8f6b7ea677123e53c5561。
排除live PostgreSQL文件并移除QP_PG_DSN；1项disposable fencing测试跳过，不作为通过或生产认证。
git diff --check（quant_platform与本轮证据目录）通过。

旧错误类型现在拒绝，不自动改写历史元数据。旧哈希碰撞、v2迁移及阈值分母等仍待处理。
