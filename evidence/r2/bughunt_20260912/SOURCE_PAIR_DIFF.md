# 来源双字段漏记（2026-09-12 20:45 heartbeat）

在server-c正式main原树修改；磁盘剩776 GiB；FE/DA另一任务无活动，未修改其文件。
保留原有改动，无分支、复制、commit/push或生产记录改写。

## 复现与修复

_member_changes原使用raw_value_ref or source_artifact_id比较来源。
raw引用非空时，artifact的替换、增加、移除均被遮蔽；raw和artifact互换为相同字符串也会被当作未变化。
现比较(raw_value_ref, source_artifact_id)二元组，不使用fallback或带@的展示字符串作为身份。
维持SOURCE_REF分类及纯来源变更不强制重训的现有规则，不改变哈希。

新增6项测试涵盖四种真实漏记、@展示歧义、完全不变负对照。
source_pair_before：4 failed / 2 passed（保留日志）。
source_pair_final：43 passed，包含来源/diff/组合版本变更/模型桥。
源码before=after：49f36c2cf74bb9addc190f8fdff6b370bc712dafe1c486b6b7a7d65cd938c1c8。
限定日志SHA256：66a980ca4520c1165cc818b5797587dfdd2665384e41a9fc293dd804a9d6a8ae。

## 边界

最终平台回归source_pair_platform_full：574 passed / 1 skipped / 3 warnings；源码before=after与限定回归一致。
平台日志SHA256：b81680d3434fc67d3b20b4ddf5e7d87e4c1fce3d61e08806c53d6c4c96558fb2。
显式排除live PostgreSQL文件并移除QP_PG_DSN；1项disposable fencing测试跳过，不视为通过或生产认证。
git diff --check（quant_platform及本轮证据目录）通过。

旧before/after展示字符串保持兼容，不能作为唯一来源身份；其结构化无损导出仍待专项处理。
同位置成员比对、组合变化ledger和阈值分母尚未全面验证；旧v1哈希碰撞与v2迁移仍开放。
