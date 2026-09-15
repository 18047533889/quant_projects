# 预算写入口标识类型混淆（2026-09-14 01:05 heartbeat）

server-c 正式main原树，剩774 GiB，FE/DA任务未活动；保留全部已有差异。不分支/commit/push/部署或操作生产数据。

SQLite TEXT亲和转换使数字1、bool True在campaign_id/attempt_id位置匹配已有字符串"1"，导致错误类型请求可以reserve/start/release/settle已有账目。create_campaign还接受空白/bytes等非法标识。现预算写入口要求非空白str，事务前拒绝；不strip改写合法标识，不迁移历史键。

20项小型临时SQLite反例覆盖四个写操作、两个标识位置及int/bool别名；另覆盖创建非法标识。拒绝后重开数据库检查完整预留行和预算未变。

- budget_identifier_before：20 failed，源码稳定。日志SHA256 a1ecb5d7f7942acf511a05f245a647669f0c46ba6b928b8b624d8a04e6d5317d。
- budget_identifier_final：903 passed，12 warnings，无skip；预算与搜索限定回归，非全项目/生产认证。
- 源码前后摘要一致：58645373e9a23d89dd550d3d5902da4aed525dc0a75fc1aec38c1ff0cb571561。
- 最终日志SHA256：9f44d1a5ea0e001d21714c94028230944ef4ece52f858606247b4f92734f0d47。
- git diff --check通过；HEAD e94ac507d670fd1c16b1d6a63fc5d6286daa5970。

范围仅预算创建/预留/启动/释放/结算写入口。budget_state/has_reservation读取及其他监督参数/假设账本接口标识边界仍待审查，不宣称全store完成；哈希codec迁移等开放项保留。
