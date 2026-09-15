# 重排后成员变更漏记（2026-09-12 21:46 heartbeat）

server-c正式main原树修改；磁盘776 GiB；另一FE/DA任务无活动，未修改其文件。
保留已有改动，未建分支/复制/commit/push/修改生产资产。

## 复现与修复

_member_changes使用zip按位置对齐，新插入或重排后保留成员的处理、来源、元数据变化被跳过。
现按(feature_name, factor_definition_ref)匹配，按新成员顺序产生变更记录。
成员索引显式拒绝重复身份；之前字典覆盖会丢失歧义成员。校验在diff边界，不自动改写历史版本。
不把纯重排当成成员属性变更，不更改重训政策和哈希公式。

新增9项：插入/重排乘以处理/来源/metadata变化；old/new重复身份拒绝；纯重排负对照。
reordered_member_before：8 failed / 1 passed，保留前置日志。
reordered_member_final：52 passed；源码before=after，无并发变化。
源码digest：6b898a5cbe912e4759bad3d53de529b6193648220d420660a9632126a5e17a26。
限定日志SHA256：d2e8adc596d719614a8ebe9ea17e005c705f1a47a2ce2d217844c1b1bc77cc16。

## 剩余

最终平台reordered_member_platform_full：583 passed / 1 skipped / 3 warnings，源码before=after与限定回归一致。
平台日志SHA256：83c450d62934e992fc0285686c8747dedd98cdcde703e97f66a107718489affa。
排除live PostgreSQL文件并移除QP_PG_DSN；1项disposable fencing测试跳过，不当作通过或生产认证。
git diff --check（quant_platform与本轮证据目录）通过。

阈值分母、强制语义变更与宽松策略组合、全局label分类分支丢失member ledger仍需专项检查。
旧哈希碰撞与v2持久化迁移未闭合，本轮不宣称平台全部修完。
