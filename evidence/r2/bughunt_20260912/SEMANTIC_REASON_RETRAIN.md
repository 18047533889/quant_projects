# 强制语义原因被分类/阈值遮蔽（2026-09-12 23:45 heartbeat）

server-c正式main，磁盘剩776 GiB；FE/DA任务idle，未触其文件，保留已有修改。
未分支/复制/commit/push/改写生产记录。

## 修复

时点、处理后特征引用、可用性语义变化有CHANGED_VERSION原因但scalar category为METADATA_ONLY/EVIDENCE_ONLY，决策错误返回不重训。
处理规则或方向变化与新增成员组合时，宽松阈值也会遮蔽必须重训条件。
统一强制原因集合：VERSION/TREATMENT/ORIENTATION/LABEL/DATA_REVISION；决策和diff默认重训属性均识别。
因此event summary及to_event也不再与强制语义原因矛盾。纯来源/metadata不加入强制集合，schema的既有显式策略保持不变。
历史哈希与scalar category保持不变，未改写历史事件。

## 验证

semantic_reason_before：8 failed（6项语义字段×证据组合，2项新增与处理/方向组合），保留日志。
semantic_reason_final：50 passed，源码before=after，无并发变化。
源码digest：2be6b3f8ef8f93445634fcf0f616c34e18049e7b821e63bc9b6005ffcc71a74f。
限定日志SHA256：ff375dccb4b5e18de07fa14f6b1a04f30329ed99cb6bdd0473352603cf752d6b。

## 剩余

最终平台semantic_reason_platform_full：597 passed / 1 skipped / 3 warnings；源码before=after与限定回归一致。
平台日志SHA256：b35b90acd93ba3c1e82c141df48dfafe7f9dfe4172d4a2b2f6ee2464de5ff75a。
排除live PostgreSQL文件并移除QP_PG_DSN；1项disposable fencing测试跳过，不作为通过或生产认证。
git diff --check（quant_platform与本轮证据目录）通过。

阈值分母、schema策略的业务边界、哈希v2迁移和来源无损展示仍待处理，不宣称全部完成。
