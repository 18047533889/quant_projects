# H01–H36 主干合并补充审计实施记录（历史进度）

当前结果与未认证边界以同目录 `H-SUPPLEMENTAL-CLOSURE-20260910.md` 为准；以下保留早期失败与实施过程，不作为最终状态。

正式目录：server-c /home/sunhaiwei/quant_projects。
开始基线：main / c5e880db91f672d101676f92f7901361ea48522d，工作树干净。
本轮仅直接修改该工作树；无分支、worktree、整库复制、commit、push、生产发布或依赖安装。

## 已取得的有限测试证据

- H11：有限正数校准、128 个形状键、每键最近 256 样本、真正 median/MAD、完整键序列持久化、EMA 保留、4 MiB 文件读写上界和原子替换。H11-calibration-tests.log：15 passed（含现有路径测试）。
- H12/H13/H23/H36：H-validation-tests.log：69 passed（新增负控、旧数学审计、既有 DataAccess 路径）。后来身份模块的并行深化会改变实现身份，最终需重跑，不能直接当最终证书。
- H22：真实 registry prefix 与真实 stateful 执行已接入；参考函数另命名 REFERENCE_SELF_CHECK。初跑 12 个实际 causal winner 有限通过；ts_rank_corr 无 production winner，明确 BLOCKED。stream 初跑发现 ADX 成熟期错配；ewm cov/corr 无 production winner，明确 BLOCKED。
- H22 新发现：ADX 续算原为 min_periods=1，当前获胜实现为每级 min_periods=window。已对齐成熟支持、独立 high/low/close 时钟、零方向支持缺失；checkpoint semantic_version 4.0，旧状态不得冒充等价。H22-recursive-parity.log：11 passed。
- H22-tests.log：25 passed / 1 failed；失败为旧子进程 import cleaned_operators，已改 canonical import，等待重跑。此失败记录保留，不改写成通过。
- H-root-combined-tests.log 某次遇到并行中的 registry helper 递归身份错误而在导入时失败，已反馈身份负责人；这不是已完成测试覆盖。

## 所有任务责任与状态

| 任务 | 责任范围 | 当前状态 |
|---|---|---|
| H01–H05 | registry / cleaned_bridge | 已首轮修复；复核追加 global helpers、包装实例、exception table、catalog 类型恢复/不可逆只读，继续实施 |
| H06–H08 | cross_sectional / component_score | 子代理实施与测试 |
| H09 | SQL ddof | 子代理实施与真实 DuckDB 测试 |
| H10 | 依赖最低版本声明 | 待完成；不能假称最低环境已运行 |
| H11 | scan_cost | 已修并取得有限测试，待最终联跑 |
| H12/H13 | math_certificate / audit scripts | 已修声明与执行、全列全轴比较，待最终联跑 |
| H14–H17 | SQL context / gate / fiscal / cache | 首轮 context/gate/cache 已修；严格 fiscal 继续 |
| H18–H21 | regression / sequence / candle | 子代理实施 |
| H22/H23 | 真实验证 / oracle 隔离 | 已接入；真实缺失实现保留 BLOCKED，待最终负控/联跑 |
| H24 | SQL joint DAG | 首轮真实 DuckDB 运行；补持久测试与性能证据 |
| H25/H26 | 残留与路径门禁 | 待逐 hunk 裁决及实现 |
| H27/H28 | intraday identity / finite / empty | 首轮专项 6 passed、既有 75 passed，待最终联跑 |
| H29 | linear grid / demand stats / broker | 网格已线性；按需统计与 broker 租约仍在实施，不宣称关闭 |
| H30 | physical-clock paired returns | 首轮已修，待扩展边界 |
| H31 | HAR feasible domain | 子代理实施 |
| H32/H33 | paired R² / ex-self / axes | 首轮已修，待最终联跑 |
| H34/H35 | SQL candidate rank / min_periods | 子代理实施；不提升 production 安全标签 |
| H36 | LazyColumnBundle | physical namespace、请求持有引用、context/alias 验证，批准摘要仍严格 eager |

## 性能与转换边界

本轮合成测试只证明指定数学反例、缓存 payload 上限、有限数据下真实后端及网格字节增长。
H29 网格按 4/8/16 天 × 3 bar × 2 列，grid buffer 应为 192/384/768 字节，不再为天数平方。
H36 每次请求先持有缓存命中，再插入/淘汰，别名只在返回边界转换；rename 不复制值缓冲。
SQL CSE query 字节、编译计数、EXPLAIN 与执行时间分别记录，不能推出固定倍数生产提速。
没有启动真实 A 股/COS 输出、十万生产因子任务或最低版本安装；默认 auto/80% 策略未改。
原 D02 不因 H36 或清理而关闭；approved content digest 仍走既有严格 eager 保护。

本记录不是全算子/全参数/全后端零缺陷证书。完成时按实际终态更新，不给未覆盖项自动加 PASS。
