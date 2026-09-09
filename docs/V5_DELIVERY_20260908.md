# V5 整改与验收交付

验收日期：2026-09-08。范围为用户提供 V5 文件中的 D01–D28、N01–N14、T01–T32 和 §9.5 四条闭环，不重新执行 V3 的 177 项历史清单。

规格 SHA256：`19e2962841652e5d1b287b2edff855ac9a42be4d72377595f0905c782f9bd0f2`。

## 位置与证据边界

修改位于 server-c 的 `/home/sunhaiwei/quant_projects_v3_remediation`，分支 `codex/v3-remediation-20260906`，HEAD `2db45f4e446309003381a51a5e03071d31859502`。修改尚未提交。保留原有 V3/V5 未提交内容；没有覆盖已独立推进 V6 的 `/home/sunhaiwei/quant_projects` 源码。

本报告的路径默认相对于上述远端整改目录。精确源文件 SHA256、Git blob、实际 import 路径、完整未提交状态和当前 diff 由 `scripts/v5_source_evidence.py` 保存到 `evidence/v5/final_source/`。当前 diff 包含继承的 V3 工作，不能当成纯 V5 补丁；本轮开始的 diff/status 同时保留。未跟踪源文件另有哈希索引，不会假称其已包含在 git diff 中。

只验收研究代码、合成数据上的真实库调用及隔离数据库行为。没有 push、生产发布、生产数据库迁移、真实数据删除或交易。`NET_ASSUMED` 不等于实际账户可执行认证。

状态沿用规格限定词。`ALREADY_FIXED` 表示最终 checkout 已有实现并有通过证据，可能是本轮修复，也可能是复用已有修复；不表示这些代码在本轮开始前就已全部完成。外部数据验收与本地实现分开列明。

## 最终回归

统一运行环境：Python `/home/sunhaiwei/quant_projects/.venv/bin/python`；`PYTHONPATH=factor_optimizer:factor_preprocess:.`；`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 POLARS_MAX_THREADS=2`。

| 范围 | 实际测试入口 | 结果 | 日志 |
|---|---|---|---|
| QE | `python -m pytest quant_evaluator/tests -q` | 912 passed，2 skipped | `evidence/v5/qe_final.log` |
| FO + FP | `python -m pytest factor_optimizer/tests factor_preprocess/tests -q` | 1161 passed，1 xfailed | `evidence/v5/fofp_final.log` |
| FA | `python -m pytest factor_assets/tests -q` | 1398 passed，0 skipped | `evidence/v5/factor_assets_final.log` |
| Modeling + jobs | `python -m pytest tests/modeling jobs/tests -q` | 466 passed | `evidence/v5/modeljobs_final.log` |
| Platform | `python -m pytest quant_platform/tests -q` | 456 passed，21 skipped | `evidence/v5/platform_final.log` |
| 隔离真实 PostgreSQL + fencing | 独立临时 PG 环境中的 live/fencing 测试 | 23 passed；临时服务已停止 | `evidence/v5/postgres_live_and_fencing_tests.log` |
| 实际成交及成本 | `python -m pytest vectorbt_qs/tests/test_v5_cost_schedule.py vectorbt_qs/tests/test_v5_research_trajectories.py vectorbt_qs/tests/test_v5_filled_ledger.py -q` | 19 passed | `evidence/v5/vectorbt_qs_v5_trajectory_tests.log` |
| 真实CUDA专项矩阵 | 32项支持指标实际CPU/CUDA数值对拍、5项新LO/cost严格拒绝、tiling/OOM/参数回归 | 69 passed | `evidence/v5/gpu_public_numeric_matrix_tests.log` |

跳过/限制：QE 的 2 项是仓库不存在的可选 zscore、drop-row kernel 合同检查，不是 GPU 缺失。平台默认环境的 21 项需要独立 PostgreSQL，另有上表真实 PG 运行证据。FP 的 HP 滤波是已知非因果算法，保留预期失败并限制 `OFFLINE_ONLY`，没有删测试或放开生产。完整 `vectorbt_qs/tests` 仍受既有缺失依赖 `dill`、`plotly` 和内部 `vectorbt_qs.mvp.data` 导入阻断；不宣称该整库全绿。

多包测试必须按上表分组执行。将 FO 的 `tests` 和仓库根目录的 `tests/modeling` 合成一个 pytest 进程时出现既有同名 `tests` namespace 冲突（`No module named tests.search`）；独立进程能运行各自完整测试。没有把该集合失败写成代码回归通过。

测试环境为了实际执行 FA ANN 测试，安装了 `annoy==1.17.3`、`faiss-cpu==1.15.0 --no-deps`；FA 最终没有 ANN 跳过项。没有把缺实现改成 supported 来获得通过。

## D01–D28 关闭表

下表的源码均进入精确哈希清单；测试定位详见后面的逐项 T 索引。影响类型对应文末迁移表，未执行生产迁移。

| ID | 状态 | 最终实现与证据 | 影响 |
|---|---|---|---|
| D01 | ALREADY_FIXED | `runtime/evaluator.py`、`adapters/execution_trajectory.py`：逐因子独立轨迹，拒绝单轨迹广播；T01 公共请求比较两因子/新增第三因子 | QE 数值 |
| D02 | ALREADY_FIXED | `contracts/metric_instance.py`、`api/requests.py`：canonical/default 参数/版本/场景身份；scalar、series、T×Q×F 完整回读，Q/H 不覆盖 | QE 身份 |
| D03 | ALREADY_FIXED | `runtime/gpu_executor.py`、GPU rank/correlation kernels：参数传递、raw ICIR 零方差未定义、不同换手定义分开；真实 CUDA 反例通过 | QE 数值 |
| D04 | ALREADY_FIXED | `contracts/portfolio_inputs.py`、cohort CPU/GPU：前向标签与日度持有收益分离；H1/H10/H20 入场后区间手算；共享收益计划 | QE 数值 |
| D05 | ALREADY_FIXED | `metrics/underwater.py`、`metrics/risk/drawdown_analysis.py` 共用净值/事件语义；初始亏损、两次恢复、尾段删失、缺报价不压缩时间 | QE 数值 |
| D06 | ALREADY_FIXED | `runtime/streaming_evaluator.py`、`online_moments.py`：每因子统计，不混池；稳定中心矩 v2；完整截面生成 Pearson/RankIC 的边界明确 | 增量状态 |
| D07 | ALREADY_FIXED | `metrics/probe_portfolio/_core.py`：每个实际桶人数守门，返回 counts/valid/distinct/degradation；N100/Q20/min10 反例 | QE 数值 |
| D08 | ALREADY_FIXED | `vectorbt_qs/contracts/trajectories.py`：真实成交、现金、股数、费用、滑点、分红送股及每日估值导出 LO gross/net/active/relative NAV | 成本与执行 |
| D09 | ALREADY_FIXED | FA `profiling/health_card.py`：必需门来自用途政策，空门、重复门、错因子/上下文均拒绝；14 维不覆盖门 | 评级 |
| D10 | ALREADY_FIXED | FA `profiling/policies.py`：递归不可变、内容哈希、嵌套字典不可篡改 | 评级身份 |
| D11 | ALREADY_FIXED | 正幅度回撤与 typed MetricGradeRule 的方向/单位/版本一致；风险预算 >1 阻断发布 | 评级 |
| D12 | ALREADY_FIXED | sealed QE 能力校验 + FA `adapters/quant_evaluator.py` 真实接线；`turnover_cost` v3 从声明成本贡献计算 bps，`cost_drag` 显式 /10000，预算利用率必须有明确 bps 预算 | QE 数值与评级 |
| D13 | ALREADY_FIXED | E0–E3 证据等级与效果等级分离；离散 ties、财务更新 cadence、缺失及新鲜度按用途判断 | 评级 |
| D14 | ALREADY_FIXED | 形状类型不等于等级；单调、U 型适用项分别选择，N/A 不变成 0 分 | 评级 |
| D15 | ALREADY_FIXED | FA `assembly/engine.py`：真实 micro/macro membership，和 lineage family 分离；UNKNOWN 预算共享 | 选择 |
| D16 | ALREADY_FIXED | `AssemblyPolicy.content_hash` 进入选择身份；成员相同不掩盖政策变化 | 选择身份 |
| D17 | ALREADY_FIXED | Pareto 必需目标冻结，禁止用候选指标交集改变全体比赛项目 | 选择 |
| D18 | ALREADY_FIXED | 只有实际选定代表写 `representative_of`，非整簇成员都冒充代表 | 研究归簇 |
| D19 | ALREADY_FIXED | UTC-aware as-of；拒绝未来证据；生产装配绑定 universe/snapshot/split/recipe/data-as-of | 选择 |
| D20 | ALREADY_FIXED | FA `clustering/incremental.py`：小簇合并由真实目标簇支持度决定并更新 affinity | 研究归簇 |
| D21 | ALREADY_FIXED | FA `contracts/cluster_governance.py`：真实 many-to-many SPLIT/MERGED 血缘和旧版本引用 | 研究归簇 |
| D22 | ALREADY_FIXED | VI 正确方向/尺度；相邻稳定平台；只给 modularity 不能通过，顺序不影响选择 | 研究归簇 |
| D23 | ALREADY_FIXED | FP `representation/policy.py`：方向正确的非劣关系、近零/NaN 边界、LINEAR any-of 缩放 | 表示与选择 |
| D24 | ALREADY_FIXED | FP `adapters/fe_operator.py`：规范 FE 执行、完整参数转发、重复主键拒绝、资产轴保真 | 受影响因子值 |
| D25 | ALREADY_FIXED | FO `search/paired_comparison.py`：共同 draws、严格 win/loss/tie、CI 全包含等价、带方向非劣 | 选择 |
| D26 | ALREADY_FIXED | FO typed request/scenario 转发；平台 durable state 成功、失败重试、预算/权限分离；日常固定更新不启动研究搜索 | 编排 |
| D27 | ALREADY_FIXED | QE `jobs/refs.py`、`jobs/handler.py`：引用分片≤256、有界加载与原子输出；既有 GPU tiling/OOM 保留；十万实际数值阶梯仅按实测 profile 声明 | 批处理 |
| D28 | ALREADY_FIXED | CPU/GPU 数学可定义二元 ties-aware Spearman；NaN/Inf 一致；axis/tie method 严格校验；受影响版本 v3 | QE 数值 |

其中 QE 文件默认前缀 `quant_evaluator/`，FA/FO/FP 路径分别指对应包；精确目录和文件哈希见 source manifest。

## N01–N14 新能力表

| ID | 状态 | 验收结果与边界 |
|---|---|---|
| N01 | ALREADY_FIXED | 同一公共请求同时包含 Q10/Q20、H10/H20、gross/base/stress，9 个独立实例；别名/default 去重；编译后的 ID 映射在 `requested_to_resolved_instances`，消费者不得假定请求 ID 未变化 |
| N02 | ALREADY_FIXED | LO 从实际账本给净收益、基准、active、annual TE/IR、relative DD、投资比例、容量；新指标要求明确的收益腿，不能误用总收益 |
| N03 | ALREADY_FIXED | LS signed long/short 与 bottom holding 分开，贡献/符号验收；net 子腿必须有明确 net contribution；无借券只有研究证据，不能签可执行 |
| N04 | ALREADY_FIXED | 日期/侧别/市场费表，2.5bps 佣金最低5元，卖印花税5/10bps分段，过户0.1/0.2bps分段，固定滑点只计一次；actual/365 融券与负现金融资 |
| N05 | ALREADY_FIXED | H10 普通 A 股冷启动 raw RankIC/ICIR 锚点、多维效果、独立证据等级、用途 required/optional、正幅度回撤、风险预算守门 |
| N06 | PARTIAL | 校准代码、持久不可变开发参考群、family cap、失败样本、实际 QE raw ICIR、禁 TEST/HOLDOUT、旧 raw 重评分均已通过；尚无指定的真实历史开发样本完成经验校准，合成参考群不是准入权威 |
| N07 | ALREADY_FIXED | 实际共同 draws 配对效用、经济容忍界、等价和非劣，INCONCLUSIVE 默认保留现有版本 |
| N08 | ALREADY_FIXED | 诊断路由保留 raw，初始/显式预算与 donor cap；路由身份绑定有序诊断/政策/预算。组合 recipe 深冻结、校验内容引用；真实 SearchRunner→stateful weights→A股planner→filled ledger→公共QE 验证 raw 与避雷组合结果不同；缺consumer、错误ref、未知诊断拒绝 |
| N09 | ALREADY_FIXED | micro/macro/family 分开、冻结 Pareto、政策哈希、UTC/context、实际代表；研究标签不替换生产输入 |
| N10 | ALREADY_FIXED | `clustering/refresh_service.py` 实际执行周期/触发回调并记血缘；新版本为 research candidate，真实模型替换试验另有 N11；没有安装未经授权的生产周期任务 |
| N11 | ALREADY_FIXED | `modeling/feature_set_trials.py`：ADD/REPLACE/DROP_CLUSTER 真正同 fold OOF 重训，baseline 计预算，整数/前向/不重叠 fold 校验；模型参数原子持久化、hash/readback，完整基线/阈值证据，typed bridge 校验 |
| N12 | ALREADY_FIXED | 引用式 QE jobs、有界 sink、校验内容/路径、repair corrupt bytes；现有 DB jobs/attempts/results 上原子 claim/lease/fencing，过期 worker 不能覆盖新结果；SQLite 与真实隔离 PG 已测 |
| N13 | ALREADY_FIXED | 既有 Db 上标签成熟消费队列，未成熟不计算，逐 factor×instance 幂等；full/chunk/bar/restart、RESTATED 分流、事务失败回滚、pending label GC 保护；不假称所有指标都能精确逐 Bar 更新 |
| N14 | ALREADY_FIXED | 真实四链、固定更新、局部影响查询和回滚边界；`runtime/impact_plan.py` 复用 FA invalidation authority，不重建第二套依赖系统，不删除旧制品 |

## T01–T32：逐项证据索引

以下路径均为可执行测试，不以一段范围标记替代每项验收。

| ID | 状态 | 测试文件/独立断言 |
|---|---|---|
| T01 | ALREADY_FIXED | QE `test_v5_metric_instances.py::test_t01_t02_all_h_q_cost_variants_and_third_factor_remain_independent`；adapter 按 factor 对齐 |
| T02 | ALREADY_FIXED | 同上9实例请求；`test_v5_final_semantics.py` Q5/10/20 的 T×Q×F JSON 回读 |
| T03 | ALREADY_FIXED | QE `test_v5_final_semantics.py::test_t03_constant_ic_has_undefined_raw_ir_on_real_cpu_and_cuda` |
| T04 | ALREADY_FIXED | QE `test_v3_cohort_intervals.py` 入场前跳价、H1/2/10/20手算、真实GPU同源 |
| T05 | ALREADY_FIXED | QE `test_v3_drawdown_contracts.py` 两次恢复、删失、缺报价不压缩日期、吸收全损 |
| T06 | ALREADY_FIXED | QE `test_v5_discrete_rank_semantics.py` 实际桶人数守门 |
| T07 | ALREADY_FIXED | `vectorbt_qs/tests/test_v5_filled_ledger.py` 减仓/现金/投资比例/动态分母；纯多头E2E公共指标 |
| T08 | ALREADY_FIXED | `vectorbt_qs/tests/test_v5_research_trajectories.py` 两腿加总与 bottom/signed-short 相关反号 |
| T09 | ALREADY_FIXED | FA `profiling/test_v5_multidimensional_grading.py` 的 raw .25/H10范围；QE别名到 `ic_ir` v3 |
| T10 | ALREADY_FIXED | `vectorbt_qs/tests/test_v5_cost_schedule.py` 100000元买76元/卖126元（含滑点），1000元佣金5元 |
| T11 | ALREADY_FIXED | 同文件有效日期印花税/过户费分段，买入零印花税 |
| T12 | ALREADY_FIXED | 同文件同订单多fill最低费仅一次、独立订单分别计费、已含项目不重复收费 |
| T13 | ALREADY_FIXED | 同一真实filled ledger repricing，gross不变，提高费用不提高净收益；多情景公共入口 |
| T14 | ALREADY_FIXED | research trajectory + QE adapter：无borrow evidence拒绝可执行认证；actual日历融资 |
| T15 | ALREADY_FIXED | FA `test_t15_empty_wrong_or_missing_policy_gates_fail_closed` |
| T16 | ALREADY_FIXED | FA `test_t16_positive_drawdown_magnitude_is_not_best_grade` |
| T17 | ALREADY_FIXED | FA `test_t17_t18_ties_staleness_and_wrong_shape_are_not_universal_requirements` + 财务成熟E2E |
| T18 | ALREADY_FIXED | 同上单调/U形适用性分别判定 |
| T19 | ALREADY_FIXED | FP representation tests：.02→.03不是破坏，近零/负向/NaN不靠错误比例 |
| T20 | ALREADY_FIXED | FO paired comparison tests：宽差值区间不因边际CI重叠而等价 |
| T21 | ALREADY_FIXED | FO paired comparison tests：draw/context不一致拒配，全平局不称严格100%胜率 |
| T22 | ALREADY_FIXED | FA `assembly/test_v5_assembly_delta.py` 真实micro/macro与UNKNOWN预算 |
| T23 | ALREADY_FIXED | 同文件成员相同、政策不同仍有不同selection身份 |
| T24 | ALREADY_FIXED | 同文件冻结Pareto必需目标，缺cost只拒绝该候选 |
| T25 | ALREADY_FIXED | 同文件UTC/as-of、实际代表、生产context冻结 |
| T26 | ALREADY_FIXED | FA incremental cluster tests + `_aggregate_cluster_support` 真实目标证据 |
| T27 | ALREADY_FIXED | FA `test_v5_cluster_lifecycle.py` many-to-many split/merge |
| T28 | ALREADY_FIXED | 同文件lower VI、相邻稳定平台、无稳定指标拒绝、字典顺序不影响 |
| T29 | ALREADY_FIXED | 平台 `test_v5_durable_qe_jobs.py`、`test_v5_durable_job_fencing.py`；QE `test_v5_qe_job_safety.py` |
| T30 | ALREADY_FIXED | QE `test_v5_maturity_streaming.py` 未成熟/重复/修订隔离 |
| T31 | ALREADY_FIXED | 同文件full/chunk/bar/restart；`OnlineMoments` 大均值小波动精确对拍 |
| T32 | ALREADY_FIXED | QE `test_v5_maturity_gc.py` + 平台storage/coordinator测试：保护pending/rollback/production引用，终态无引用值实际删除但保留trial |

## 四条真实端到端流程

入口 `jobs/e2e_v5_acceptance.py`，验收 `jobs/tests/test_e2e_v5_acceptance.py` 中四个 `test_v5_spec_9_5_*`。

1. 连续量价：DSL `rank(close)`→真实 FE/DA→QE 初评→高换手诊断→少量 FP/组合候选→同一真实成交账本净成本比较→paired 非劣且更便宜→实际 FA repository/lifecycle→研究冻结更新。
2. 财务事件：公告可知时间→cadence/合法缺失→实际 QE→适用性而非统一 ties/stale 拒绝→MODEL_FEATURE 用途→实际 SQLite 标签成熟队列与幂等统计。
3. 纯多头避雷：底部证据→冻结排除→实际 planner/fills/cash/valuation→公共 QE 的净收益、TE/IR、relative DD、投资比例、turnover_cost→实际订单/ADV 容量→FA研究准入，不需要假借券。
4. 库级替换：近重复/共同样本/成本→实际 ElasticNet OOF 重训→完整模型参数原子持久化/哈希回读→feature version更新→旧版本回滚→固定日常输入去重，停止重复计算。

FA末端实际经过 REGISTERED→EVALUATED→APPROVED，研究用途/簇标签有证据；没有借用这些研究测试修改生产指针。最新持久trace为 `evidence/v5/v5_spec9_5_e2e_trace_94b1e130c009ab169a6a73679d03e4f78522865fd09adba9b58b933c791f45a2.json`，文件SHA256与文件名中的64位哈希一致；持久runtime目录保留真实成熟队列SQLite与模型参数JSON。旧trace保留为历史，不冒称最新代码运行。

## 规模、GPU 与逐 Bar

十万阶梯确实逐分片调用 `QEJobHandler`→公共 `EvaluationRequest/evaluate` 并写出逐实例结果，不是仅创建任务元数据。实测 profile 为 T=4、N=20、两个指标（rank_ic/coverage）、每片最多256因子。安全补丁后的完整最终阶梯另存 `evidence/v5/final_numeric_staircase/`：1k/10k/100k 分别用时1.762/21.094/180.430秒，分别完成2000/20000/200000个因子×指标结果，均0失败；峰值RSS分别163.45/164.20/170.45MiB。100k产出782个结果文件、109537881字节。先前阶梯保留为历史记录，不覆盖其数值。

这些结果只证明这个小面板profile的有界处理，不是全历史、全指标、生产容量或盈利认证。CPU批吞吐、真实CUDA语义/能力矩阵、单因子及成熟单Bar延迟分别记录，不混为一个性能数字。

32项现有CUDA支持范围之外的指标必须严格拒绝；新增LO指标与 `turnover_cost` 目前是CPU实现，不宣称它们已有GPU数学实现。

单因子/逐Bar微基准另见 `evidence/v5/n13_latency_microbenchmark.md`：1因子、40资产、H10 RankIC-series，8次预热、80次测量；单因子公共QE（T=12）p50/p95/p99=3.888/3.992/5.724ms；单Bar成熟enqueue+drain（T=1，SQLite）=7.156/9.774/10.581ms。它是合成微基准，不是生产SLA。

## 影响查询、迁移与回滚

入口：`python scripts/v5_impact_plan.py <已有EvidenceRef字段投影.json> --change-kind <类型>`。输入必须来自调用方明确提供的已有证据集合；工具只读输出，不自动扫描或修改生产。`quant_evaluator/tests/test_v5_scoped_impact.py` 验证影响范围。

| 变化类型 | 应执行的局部动作 | 不应执行 |
|---|---|---|
| GRADING_POLICY | 用正确的已存raw指标重评分/准入 | FE重新落值 |
| COST_POLICY | 对关联成交轨迹重估成本、更新对应评价和选择 | 改信号值或任意重做全库 |
| METRIC_SEMANTICS | 只失效受影响版本评价及下游健康/选择证据；RankIC相关与turnover_cost版本可查询 | 全部V3制品失效 |
| CLUSTER_RESEARCH_LABEL | 新增研究归簇证据和血缘 | 暗改生产模型输入 |
| FACTOR_VALUE_SEMANTICS | 仅显式受影响参数/轴/数据的因子物化与依赖 | 无边界全量重算 |

增量中心矩状态升级为 `centered_moments.v2`，读取旧状态时按兼容规则提升；AS_KNOWN 保留，修订进入独立 RESTATED_RESEARCH。平台 pending label 根类型有兼容 v9 迁移代码，SQLite/隔离PG已验收，但没有替用户迁移生产库。

调用方兼容注意：旧 `build_trajectory_from_execution_plan` 现在必须提供 `valuation_price`，并委托同一真实filled-ledger authority；调用方提供的gross仅用于一致性断言，不能再当收益来源。ICIR的 `min_periods` 必须为整数且≥2；Inf排除、真正恒定序列未定义，真实微小方差不再被任意epsilon阈值丢弃。`pearson_ic_ir`、`quantile_spread`、`quantile_monotonicity` 的数值或样本证据语义也已升级到v3，只有依赖这些定义的证据需要局部更新。

回滚保留旧评价、政策、簇、模型和feature-set不可变引用。旧证据可重放，不覆盖历史。源码回退需要基于本轮前baseline及逐文件差异选择性操作，不能在这个混合未提交工作区执行整体reset。成熟队列事务保证不重复提交/累计；并发drain可重复做计算但不能重复提交，不宣称exactly-once compute。普通durable jobs另有lease/fencing，旧owner的终态写入被拒绝。

## 尚未签发的外部验收

真实历史阈值校准需要明确的开发参考群，包含失败候选、family控制和未泄漏时间边界。当前合成参考群只验证流程正确，不能替代历史经验校准。已向用户请求合适的数据引用。

真实账户佣金合同、借券可得性、实际市场冲击/容量、全历史生产profile资源、生产发布/调度和生产数据库迁移均未认证或执行。代码会保留相关拒绝/授权边界；这不是通过研究测试即可自动放行的事项。

## 本机证据索引

已独立回读核验1407个源码文件、4条trace和4个持久模型的哈希。源码清单文件SHA256：`4ce51c0a95e2e53c5f2845a6c714418236e7602d7be9db7fb5c06da838dcc969`。

- [源码、实际import与diff清单](/Users/shw/Documents/Codex/2026-09-06/base-shw-mac-ssh-qs-server/V5_验收证据/final_source/source_manifest.json)
- [真实CPU/GPU能力矩阵](/Users/shw/Documents/Codex/2026-09-06/base-shw-mac-ssh-qs-server/V5_验收证据/v5_gpu_capability_matrix.md)
- [最终1k／10k／100k数值阶梯](/Users/shw/Documents/Codex/2026-09-06/base-shw-mac-ssh-qs-server/V5_验收证据/v5_numeric_staircase.md)
- [单因子与成熟单Bar延迟](/Users/shw/Documents/Codex/2026-09-06/base-shw-mac-ssh-qs-server/V5_验收证据/n13_latency_microbenchmark.md)
- [四条真实流程trace](/Users/shw/Documents/Codex/2026-09-06/base-shw-mac-ssh-qs-server/V5_验收证据/v5_spec9_5_e2e_trace_94b1e130c009ab169a6a73679d03e4f78522865fd09adba9b58b933c791f45a2.json)

同目录保留完整测试日志、该trace引用的合成runtime数据库/模型文件，以及本轮开始和最终的diff/status。trace中的模型/数据库绝对路径仍忠实指向原远端执行位置；本机副本没有伪改这些证据引用。
