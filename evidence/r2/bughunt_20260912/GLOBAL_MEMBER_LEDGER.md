# 标签/数据版本分支的成员审计遗漏（2026-09-12 22:45 heartbeat）

正式server-c main原树，磁盘剩776 GiB，FE/DA另一任务无活动且本轮未触其文件。
保留全部已有修改；不创建分支、复制、commit/push或改写生产数据。

## 复现与修复

label/data_revision分类分支遗漏changed_members参数，已计算的来源、metadata、timing变化被丢弃。
结果是重训仍触发，但diff/event的changed数量错误为0，相关原因代码缺失。
两个分支补传已有changes，复用原装配逻辑；保持category、reason、重训规则、历史哈希不变。

6项新反例（两种全局变化乘三种成员变化），global_ledger_before全部失败。
global_ledger_final：52 passed，源码before=after，无并发变更。
源码digest：66372938d5239517d2af0ec130aa7e31e19d60b86647d4344ba57bb6ce5e964d。
限定日志SHA256：d2e8adc596d719614a8ebe9ea17e005c705f1a47a2ce2d217844c1b1bc77cc16。

## 限制

最终平台global_ledger_platform_full：589 passed / 1 skipped / 3 warnings，源码before=after与限定回归一致。
平台日志SHA256：d6b2f1922746a1d9604636c655c6dda18472903910fef9ef187286ec5834ae5d。
排除live PostgreSQL文件并移除QP_PG_DSN；1项disposable fencing测试跳过，不作为通过或生产认证。
git diff --check（quant_platform与本轮证据目录）通过。

仅闭合label/data_revision分支成员ledger丢失，不宣称所有标志/计数语义均已验证。
阈值分母、宽松策略组合、来源无损展示、旧哈希碰撞与v2迁移仍待继续。
