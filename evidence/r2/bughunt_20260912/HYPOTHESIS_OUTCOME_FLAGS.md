# 假设账本结果标志类型（2026-09-14 03:09 heartbeat）

server-c main正式原树，剩774 GiB，FE/DA任务未活动；保留已有改动，未分支/commit/push/部署/生产数据操作。

record_hypothesis_attempt未要求executed/has_pvalue为bool，int转换使2、-1等直接成为账本计数，0.5或字符串"0"被截断/转换，扭曲执行数及p值数量。现写库前要求两个标志都为bool；保留有p值必须已执行的约束，以及合法重复写入行为。

15项临时SQLite测试覆盖两字段的数字/字符串/None拒绝及三个合法状态的计数和幂等重放。非法输入后重新打开确认汇总不变，无生产数据。

- hypothesis_flags_before：10 failed / 5 passed，源码稳定。日志SHA256 84afb19e02b00327036c7ae6def564f0afc52cf4154e6a246cf5bc2f068996f5。
- hypothesis_flags_final：925 passed / 12 warnings，无skip；预算搜索限定回归。
- 源码前后摘要相同：6340244bf5680ecdd0b2f4556164b05d4b6421dfda7be84fcdffbcd532bb35f1。
- 最终日志SHA256：5f21cbb0e7b2597696003eff1d560a4cb717000ab55452da2f0c3499fe6c4719。
- git diff --check通过，HEAD e94ac507d670fd1c16b1d6a63fc5d6286daa5970。

不自动修订历史账目。hypothesis_family_summary身份边界、require_complete_fdr_family计数类型、effective_spec_hash载体及监督参数接口仍待审查；并不宣称统计方法或整个项目已闭合。
