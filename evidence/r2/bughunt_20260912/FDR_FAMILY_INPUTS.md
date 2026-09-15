# FDR组汇总输入边界（2026-09-14 04:09 heartbeat）

server-c main原树，剩774 GiB，FE/DA任务未活动。保留已有修改，未分支/commit/push/部署/生产数据操作。

hypothesis_family_summary和依赖它的require_complete_fdr_family允许int/bool campaign_id匹配字符串键，完整性检查可使用错误标识对应的记录。提交计数True/1.0也可与整数1相等而通过。现汇总要求非空白str标识，完整性入口要求真正int计数，保留合法计数匹配及空组拒绝行为。

7项小型临时SQLite测试，6项类型反例和合法计数/不同campaign隔离检查。

- fdr_inputs_before：6 failed / 1 passed，源码稳定。日志SHA256 fc87fdf4ce24c75dbe6b5afb0a6eb04e1b714058ba1f2373e9e03832e9b49da7。
- fdr_inputs_final：932 passed / 12 warnings，无skip；预算搜索限定回归。
- 源码前后摘要一致：5bcd8d999d6e2ddb7b81fef29aecd494773a6365a78e46a33b3356ad93a30f2c。
- 最终日志SHA256：e4a819f30d85415e00a4254c25b9cfbbe799063859d583dfdb11e69f33f05b57。
- git diff --check通过；HEAD e94ac507d670fd1c16b1d6a63fc5d6286daa5970。

计数相等本身并不证明p值成员真实/完整；本轮仅修复类型边界，不宣称FDR统计认证完成。effective_spec_hash载体、监督参数接口及哈希codec迁移仍待查，不改历史数据。
