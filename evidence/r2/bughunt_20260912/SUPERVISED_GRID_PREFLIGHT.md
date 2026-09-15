# 候选集前置校验（2026-09-14 09:10 heartbeat）

server-c正式main原树，剩773 GiB，FE/DA任务未活动，保留已有修改。无分支/commit/push/部署或生产数据操作。

fit_supervised_parameter原在读取评分/执行选择后才经FrozenSupervisedParameter校验候选集，空/重复/非有限网格仍触发评分Mapping读取。现将原有有限/唯一/非空检查统一移入共享_candidate_grid，在冻结及拟合入口转换后立即执行。保留合法数值、hash公式与平分选择规则。

9项合成测试：两种修复类型下4种非法网格均在评分访问前拒绝，另测合法网格正常读取和tie-break。不涉及外部评分服务或生产数据。

- grid_preflight_before：8 failed / 1 passed，源码稳定。日志SHA256 3f467e11ca9a25992eb6ff11161c1ddfae89cc899e086872787d1c21a8fd884f。
- grid_preflight_final：975 passed / 12 warnings，无skip，预算搜索限定回归。
- 源码前后摘要一致：5b9b52fcb6764a1d7e0eccdcfab444445af3d353293b70bcaa1b433d73761a19。
- 最终日志SHA256：6bf86d7bd09d097762f75cd02de54d04e5f037cfecd84da1b57f6dc982d8199c。
- git diff --check通过；HEAD e94ac507d670fd1c16b1d6a63fc5d6286daa5970。

评分Mapping快照一致性、身份/repair_family在评分前的校验顺序、其他可转换数值仍待查。codec迁移等开放项保留，非生产或全项目认证。
