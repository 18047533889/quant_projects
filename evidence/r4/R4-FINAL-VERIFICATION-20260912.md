# R4 最终验证（server-c，2026-09-12）

本轮问题、修复位置与前后失败证据索引见
[R4-BUG-AUDIT-20260912.md](R4-BUG-AUDIT-20260912.md)。
源码仍在 `/home/sunhaiwei/quant_projects` 的 main 工作树，HEAD 为
`e94ac507d670fd1c16b1d6a63fc5d6286daa5970`；没有 commit/push/新分支/整库副本。
保留前轮和其他 AI 的工作树改动；没有修改本轮协作之外的 QE/FP/FAFO/平台代码。

## 最终通过结果

| 回归组 | 项目环境 Pandas 2.3.3 | 系统环境 Pandas 3.0.5 | 本目录日志前缀 |
| --- | --- | --- | --- |
| 完整 backend_parity | 1034 passed / 340 skipped | 1026 passed / 348 skipped | `final-backend-epoch2-` |
| 默认调用、manifest、DSL、资源/租约/路由 | 261 passed | 261 passed | `final-serial-runtime-` |
| DA 读取、SQL、predicate、缓存、manifest、租约 | 353 passed | 353 passed | `final-serial-da-` |
| snapshot + R30 参数契约 | 19 passed | 19 passed | `final-snapshot-r30-` |
| 独立 R40 算子规格 | 6 passed | 6 passed | `final-r40-` |
| 相邻成本路由及旧 baseline 测试 | 16 passed | 本组未重复运行 | `final-routing-adjacent-epoch2-project` |

每条完整命令、返回码、测试进程族峰值和时间均保留在对应 `*-watchdog.json`。
最终各组返回码均为 0，看门狗未终止。跨后端项目/系统组采样峰值分别为
709,459,968 / 663,810,048 字节；执行组 1,323,737,088 / 1,251,885,056 字节；
DA 组 1,480,699,904 / 1,407,971,328 字节。
这是有界合成测试的采样结果，不是生产峰值、内核硬内存上界或 80% RSS 验收。
跳过项保留原始原因，不计入通过；不同测试组也不等于全 canonical 数学认证。

## 失败记录没有丢弃

- 第一轮 Pandas 3 完整门禁有 3 个时间索引单位比较失败（us 对 s）。仅在新测试比较边界无损归一化；时间值、行序、重复掩码和严格数值/真实后端断言保留，新增 +1 秒错位负控。最后整组双环境重新通过。
- 两套资源测试并行争用同一服务器互斥锁，各出现一次锁等待失败，见 `frozen-runtime-*.log`。没有放宽/关闭共享锁；改为串行运行完整组后均 261 passed。
- 相邻旧 baseline 测试假定未跟踪的 benchmark 文件存在，产生 2 个失败。改成明确区分 missing/seed-stale、隔离缓存以及完整 provenance 的临时测试夹具；没有添加伪造实测 benchmark。最终相邻组 16 passed。
- 中间实现和测试开发过程中的其他失败也保留在 R4 日志中；最终文件名不是旧失败日志的覆盖别名。

## 最后源码一致性

`final-source-epoch2-before.sha256` 与 `final-source-epoch2-after.sha256` 完全相同，
覆盖 FE/DA 和所选测试目录共 5,288 个 Python 文件（不含另一个任务的 docs/reports 产物）。
二者 SHA-256 均为：
`5f3057947e1643a3ad6a883ff0bb109103d5c8d221ad83b5500c39482e8807c8`。
最后一次检查在串行执行/DA 回归结束后进行；`git diff --check` 通过。
运行时发现快照身份另见 `final-runtime-snapshot-summary.json`。

## 覆盖边界与交接

1,756 个 canonical 已逐项进入结构/声明台账，429 条通过本轮契约发现检查，
1,327 条仍缺少权威声明；没有把它们改成虚假的可用/已认证标签。
数值修复、28–132 倍小面板 Pandas 回退性能代价、未认证极值组合及后续验收
详见问题索引和 `ROLLING-STATISTICS-NUMERICAL-AUDIT-20260912.md`。
没有真实数据十万因子、GPU/QE、COS 写入或生产部署验收，不能声明全库零 bug 或性能最优。
最终磁盘约剩 776 GiB；R4 必要证据约 5 MB，自建临时补丁已经清理。
