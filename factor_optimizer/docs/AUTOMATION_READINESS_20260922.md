# 自动优化完成度与剩余真实接线

本清单区分代码能力、合成验收和真实运行证据，不是“全部完成”声明。
正式路径 /home/sunhaiwei/quant_projects；下文已更新时序表示分支的真实复验结果。

后续补充：[成交量因子谱系审查](VOLUME_LINEAGE_AUDIT_20260922.md) 已识别第三只
真实 COS 因子的声明处理步骤。COS 示例资产筛选现复用优化器的
warmup/purge 后 TRAIN 索引；此前使用前 300 日覆盖率，而实际 TRAIN 为 267 日，
未读取 TEST，但不应将此前资产筛选描述为严格使用同一份 TRAIN。
最新复验见 [资产筛选边界](COS_TRAIN_UNIVERSE_20260922.md)。

## 已有真实证据

新增 [逐因子候选预算隔离](BATCH_CANDIDATE_BUDGET_20260922.md)：TRAIN 诊断
扩展超预算时保留该因子 RAW，其他因子继续，不截断候选或扩大预算。

参数试算修复与默认跳过语义见
[FactorEngine 参数检查](https://github.com/HKUST-QUANT-SOCIETY/factor_engine/blob/main/docs/PARAMETER_EFFECTIVENESS_PROBES.md)。
真实窗口 20/21 对照见 parameter_probe_real_20260922.json；仅确认输出有变化，不表示收益更优。

最新冷启动提速见 [初始化证据复用 A/B](OVERLAY_SNAPSHOT_PERFORMANCE.md)：
固定真实样本的四次独立进程中，候选/输出/认证标记摘要相同，计时中位数减少约 14.4%。

新批量入口支持指定清单与受限对象大小，已有三因子 64 资产回放及
195 个方法案例，见 [COS 清单采样](COS_MANIFEST_SAMPLING.md)。默认示例仍兼容旧清单。
后续新增 [低覆盖隔离](COS_COVERAGE_ISOLATION.md)，避免个别低覆盖因子使合格因子无法执行。
低覆盖来源进一步追查见 [来源对照](SPARSE_SOURCE_INVESTIGATION.md)：当前注册算子一致，
旧 COS 输出与当前输入存在可计算性差异，历史原因尚未锁定，不能用补值掩盖。
这批样本受覆盖限制，不能替代 256 资产的二十层验收；未知复合谱系仍为缺口。

- DataAccess 读取绑定清单与因子 SHA256，排除质量阻断项；公开批量示例
  传递源声明谱系，当前固定样本为两个因子、500 日、256 股票。
- 基础缩尾和排名、重复 CS-rank 抑制、TRAIN 单指标及联合退化保护。
- TRAIN 诊断与成本后组合口径一致；训练后只验证一个冻结赢家，失败保留 RAW。
- 20 层形状、分层衰减、27 个准入平滑参数及其他数值方法已有逐案例回放。
- 当前每因子 65 案例：57 执行、8 需要额外输入或属于控制操作。
  两个清单绑定 COS 因子合计 114 执行、16 未执行、0 执行失败，
  见 temporal_real_20260922.json。
  “执行”不等于通过验证，更不等于未来收益保证。
- FP/FE 结果对齐和 RankIC 系数路径已有完整优化 A/B，输出和候选记录逐项等值。

证据分别见 COS_DIAGNOSTICS_20260922.md、method_family_recheck_20260922.json、
BASELINE_PIPELINE.md、DIAGNOSIS_AND_SELECTION.md，以及 FP/QE 的性能核验文档。

## 原 11 个未执行案例的处理状态

| 分支 | 案例数 | 实际缺口 | 下一步验收要求 |
|---|---:|---|---|
| INDUSTRY/SIZE/STYLE_NEUTRALIZATION | 3 | 声明的外部暴露与可信历史可用时间 | 暴露对齐、可用时间、真实中性化及收益退化对照 |
| WINDOW_REFINEMENT/OPERATOR_SWAP/LOW_DOF_INTERACTION | 3 | 需要 FE DSL 重编译，不是单列值变换 | 绑定原 DSL、语义约束、因果重算、候选去重及验证 |
| REPRESENTATION_RANK 的 ts（average/min） | 2 | 已补齐窗口并执行真实回放 | 过去窗口、并列值、缺失、自动搜索与 CS 去重均有测试 |
| REPRESENTATION_ZSCORE 的 ts | 1 | 已补齐窗口并执行真实回放 | 历史样本标准差、零方差缺失、截断与极值处理有测试 |
| ABANDON | 1 | 停止／放弃控制，不是数值变换 | 验证控制流程，不伪装成产生新因子 |
| MISSINGNESS_FRESHNESS 的 drop | 1 | 行选择，不符合保持行对齐的 Series 接口 | 保留行选择和覆盖约束契约，不静默删行以美化评分 |

上述 ts 分支现在要求显式窗口，默认自动搜索 5、10、20。
窗口进入计划身份，已有 CS-rank 不再误排除 TS-rank。
公式、兼容性和限制见 [时序表示说明](TEMPORAL_REPRESENTATION.md)。
真实回放能执行不等于候选能过经济准入：低覆盖、缺失指标及退化仍会被拒绝。

## 历史暴露边界

已核验的历史行业与估值 UpdateTime 有后补记录，见 BASELINE_PIPELINE.md。
这既不能证明信息当时不可用，也不能证明当时已可用。
不得把 TradeDate 直接改名为 available_time 当作 PIT 证据。
基础流程目前明确记录 neutralization_missing_exposures；这不是完成中性化。

## TEST 接线边界

最终报告已具备冻结身份、持久一次性读取、逐日曲线以及 TEST/full_sample
角色隔离的合成验收，真实 TRAIN 曲线也已验证；真实 TEST 仍未打开。

本次只读核查在 factor_optimizer、factor_assets、quant_platform 的指定目录
以及总仓库深度 3 的 sqlite/db 文件名范围内，未发现可确认的真实
TestAuthorityBroker 状态库配置。发现的 data/alphaprobe/hypothesis.sqlite3
不是据此就能认定为 TEST 状态库；jobs/e2e_f_noise_failure_gc.py 的
campaign.sqlite3 属于合成 E2E 工作目录创建，不是正式评估接线证据。
这只是限定范围内的核查，不声称全服务器没有其他配置。

继续真实最终评估前，应明确唯一持久状态库、真实数据集身份与授权标签读取端。
不得临时换库、改数据集别名或修改测试分割来重开已经消耗的 TEST。
当前真实数据是否存在历史暴露证据、TEST 正式配置位置已向用户询问；
在等待信息时，窗口分支等纯代码缺口仍可继续推进。

## 不能据此宣称

不能宣称所有方法真实验收完毕、真实 TEST 已通过、稳定盈利、绝对无 bug，
或已证明全局最快。平台根测试仍有 14 项收集错误，模块清单见
METHOD_AUDIT_20260922.md，未掩盖或关闭检查。
