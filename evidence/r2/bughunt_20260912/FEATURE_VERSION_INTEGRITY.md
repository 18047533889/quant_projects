# FeatureSetVersion 内容完整性修复（2026-09-12 heartbeat）

范围仅 quant_platform；正式 server-c main。保留前轮修改，未触 FE/DA，未 commit/push、改写资产或任务记录。

## 已复现并修复

- FeatureSetVersion 原来直接信任传入的 schema_hash/semantic_hash：任意摘要、旧摘要配新成员都能构造成功。现在按照原有公式重算并核对两个摘要；错误类型及伪造摘要明确拒绝。
- ordered_members/source_library_versions 原来保留外部 list，创建后可追加成员而摘要不变。现在验证内容类型并复制为 tuple。
- FeatureMemberRef.metadata 原来只深复制但仍允许内部修改。现在递归冻结，拒绝无法封闭的 opaque 可变对象、非字符串key及过深结构，避免绕过摘要绑定。
- metadata 导出返回独立普通字典/列表，不泄露内部可变引用；durable generation 序列化支持冻结 mapping，重启/合并和模型桥保持可用。
- 原模型桥反例现在同时验证构造边界拒绝和通过 object.__setattr__ 模拟损坏内存对象后的下游防御；没有删掉原下游校验。

## 验证

- feature_integrity_before：5 failed（真实前置反例）。
- feature_integrity_after：116 passed / 2 failed，分别为旧fixture预期内部list、旧模型桥fixture在更早构造边界已被拒绝；已按新边界更新并保留原测试目的。
- feature_integrity_final：122 passed，稳定 PASS。
- feature_integrity_platform_full：521 passed / 1 skipped，稳定 PASS；这是最后一个 opaque metadata guard 补丁之前的完整平台轮。
- feature_integrity_guard_final：123 passed，稳定 PASS，包含最终 opaque metadata 拒绝及全部相邻合同/恢复回归。
- 全量明确排除 live PostgreSQL 测试文件；另一个 fencing 测试因没有授权 disposable DSN 跳过。没有 skip 当通过。
- 最终限定源码 digest before=after：7b9636717bb3a690a9e90f1975c9570dfa1799419afc522a9a056ace18dfae5f。
- 最终限定日志 SHA-256：f9323a962ea3f6133f0f23ed20a379d9783dd54c2c4e8d898d9d94c17b4b4dfd。
- git diff --check 通过。

## 兼容与剩余

本轮故意保留 FeatureSetVersion 原 v1 哈希公式，不改旧有效版本摘要；新 codec 全面迁移仍独立进行。旧数据若摘要不符、成员类型错误或带 opaque mutable metadata，将 fail closed，不自动篡改记录以恢复。

因此公共 v1 结构碰撞仍不能通过这个校验识别；FeatureSetVersion 的 v2 新写/旧读、compute_feature_set_content_hash、SnapshotManifest、任务/事件键仍待逐项迁移。labels/data_revision 是否纳入新的语义版本以及日期/枚举元数据在JSON恢复中的类型保持，也需在v2合同中明确。当前修复是内容完整性及冻结，不是全平台哈希漏洞关单。
