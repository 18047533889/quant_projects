# 冻结参数存储标识隔离（2026-09-14 06:10 heartbeat）

server-c main原树，剩774 GiB，FE/DA任务未活动，保留已有修改。无分支/commit/push/部署/生产数据操作。

freeze_supervised_parameter的campaign_id及supervised_parameter查询标识未校验，int/bool经SQLite转换命中字符串"1"记录。现冻结前验证campaign_id，读取前验证campaign_id/parent_factor_id/repair_family为非空白字符串，不自动转换或重命名历史键。

7项小型临时SQLite测试：6项错误标识拒绝后重开恢复原状态，1项合法重复冻结与跨campaign/parent/family隔离。

- supervised_ids_before：6 failed / 1 passed，源码稳定；日志SHA256 575dacfc099a54ce52c68882fa25987241e5048fcfa4da8433f6512483fbc959。
- supervised_ids_final：946 passed / 12 warnings，无skip，预算搜索限定回归。
- 源码前后摘要一致：4199f543f6ce02ee4b1a21d7a71611fe3c28815e646023e98917244d85a54f52。
- 最终日志SHA256：839223c2e48f20c1df7aaa3dc33ca1cd41070dbd7491356dbfb2e5dc87efa317。
- git diff --check通过；HEAD e94ac507d670fd1c16b1d6a63fc5d6286daa5970。

仅接口标识隔离，不宣称冻结参数所有合同闭合。FrozenSupervisedParameter的value/grid类型及身份往返一致性仍待专项复查；平台codec迁移等开放项保留。
