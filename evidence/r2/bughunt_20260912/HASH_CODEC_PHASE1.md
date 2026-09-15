# P0-HASH-CODEC：第一批兼容修复

范围：仅 quant_platform；正式 server-c main 工作树。未 commit、push、改写旧资产/URI/任务键，未触 FE/DA。

## 已实现

- 保留旧 content_hash/canonical_str 以读取历史 v1；明确标记其结构碰撞风险，没有全局悄悄换算法。
- 新增 canonical_bytes_v2/content_hash_v2：每层类型标签和字节长度框定，映射按编码后的完整 key 排序，dataclass 带限定类型名与字段，list/tuple区分，布尔与整数区分；有独立哈希域。
- 保留枚举按其 underlying value 的既有平台语义；等价UTC时刻和正负零归一化；非有限浮点、无时区datetime、过深/循环结构及不支持类型拒绝。
- FeatureSetArtifact 新建无摘要对象默认 semantic-v2，写入 hash_codec；新制品不再受已知分隔符/类型混淆影响。
- 携带旧摘要且无 codec 的历史对象按 semantic-v1 验证；旧摘要原样保留，不升级其可信性。
- 新摘要手工重建时必须携带 hash_codec；缺失/错误/未知codec不自动回退通过。
- FeatureSetArtifact 两个序列字段在哈希前复制为 tuple，避免外部list别名使对象内容与摘要漂移。

## 测试

- hash_codec_before：4 failed，保留修复前真实制品碰撞、迁移和可变性反例。
- hash_codec_after：111 passed，1 failed。旧重建fixture只传新摘要未传codec，已按新协议携带 fs.hash_codec；没有放宽摘要校验。
- hash_codec_final：113 passed，0 skipped，稳定 PASS。包括碰撞类型、历史读取、新写/roundtrip、错codec/篡改拒绝、UTC/零归一化、非法结构拒绝，以及平台编排、FeatureSet diff/generation、model桥和daily snapshot邻近回归。
- source digest before=after：221e9ec654e19569d393dfcf47327040206fcef6a436b22bf2c53f2d68054584。
- 日志 SHA-256：ee0e070ee0aa32c516a65b17552920051c7f3e89f570586ff342536cfda69070。
- git diff --check 通过。

## 仍未闭合

FeatureSetVersion schema_hash/semantic_hash、compute_feature_set_content_hash、SnapshotManifest、job/outbox/report等其他旧消费者尚未迁移，仍使用v1；P0-HASH-CODEC仍为 PARTIAL，不是全局修复完成。

旧v1校验本身仍无法识别历史碰撞，测试明确保留这项限制；读取成功不能被当作v2重认证。下一轮优先 FeatureSetVersion 的 supplied-hash校验/成员冻结、版本化恢复，再逐消费者迁移。正式资产重算/映射/重发与任务状态迁移仍需授权，不因自动复查改写生产记录。
