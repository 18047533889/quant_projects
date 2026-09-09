# 新49项主代码合并审计：实现、验证与剩余边界

任务书：`FactorEngine_DataAccess_主代码合并审计与统一整改.md`，2026-09-09。
唯一工作树：server-c `/home/sunhaiwei/quant_projects`，现有 main。
基线 HEAD：`6b610d0c8be7a3e6a7516ee2f4f8de6b58765f4f`。
实施期间外部将 HEAD 更新为 `53d19be4d6543caf40a820eef9b1e62045a6a92b`（V10 实现波），包含尚在验证的本轮改动；本任务未执行该提交，也未回退它。提交后仍有窄修，最终身份以具名回执源码hash为准，不能以该提交代表全部验收。
本任务未 stage/commit/push、建分支/worktree、复制代码树、部署、重启 worker 或发布生产因子。
外部提交状态与本任务权限/动作区分记录，保留其他 AI 和用户的改动。
本文的 M01 等编号属于新49项，不等于相邻 `MAIN-MERGE-REMEDIATION-20260909.md` 的旧 M01–M22。

## 状态解释

- LOCAL_FIXED：具体主代码修复有具名回归，不是所有参数/后端/真实数据认证。
- POLICY_RETAINED：审计发现政策分歧，保持既有数学对象并明确声明，未偷换估计量。
- PARTIAL：修复或保护已实现，但该工作单的完整验收尚未闭合。
- EXTERNAL_NOT_RUN：缺少批准业务 profile、发布/重启权限或真实负载证据；不记为通过。

## 49项逐项账本

| ID | 状态 | 当前实现与证据边界 |
|---|---|---|
| D01 | PARTIAL | `storage/sources/data_access_source.py` 请求级实际单位状态；eager 已归一化不再二次缩放，不改共享 lazy 标志。空筛选保留 typed Arrow schema。`test_d01_actual_unit_state.py`；真实 COS/批准 profile 与历史污染范围未认证。缓存身份增补见最终附录。 |
| D02 | PARTIAL | 批准摘要路径的 eager 安全保护保留；受管 native 优化须同时绑定 prepared/实际 ReadResult、bundle、缓存及 release。具体最后实现/未完成范围见最终附录，不把标签当原生执行证书。 |
| M01 | LOCAL_FIXED / PARTIAL | CS Huber 使用共享稳定求解器；无正尺度解的 MAD 噪声退化明确 scale_degenerate，不伪称收敛。`test_v9_m01_cs_huber_shared.py`；状态尚非完整逐因子持久拟合账本。 |
| C01 | LOCAL_FIXED | 保留另一任务的 RelationalParamSpec 显式序列化、未知类型拒绝及稳定性测试；当前 manifest 可实际导出，非远端 CI 认证。 |
| CI01 | LOCAL_FIXED / EXTERNAL_NOT_RUN | 保留另一任务的 CI 同解释器测试依赖修复、真实 synthetic 1000/1000 磁盘 oracle；仅8×32 column/scalar，远端 CI 未触发。 |
| R01 | PARTIAL | 验证现有 never-dispatched/attempt=0/NOT_STARTED 子集、原 deadline、退出证明和完整证据门；不放开任意 retry/UNKNOWN/多波旧 worker。 |
| Q01 | PARTIAL | `new-merge-operator-manifest.json` 来自当前 bootstrap，记录目录能力；不是逐算子独立数学、所有参数、真实三后端/原生证据。 |
| P01 | PARTIAL / EXTERNAL_NOT_RUN | 默认 auto/80% 与有界状态机有测试，未跑批准真实市场100K、进程族硬RSS、端到端最快后端比较；不以峰值数组节省冒充整体速度。 |
| U01 | LOCAL_FIXED | `V9-DEFAULT-CALL-AND-FAILURE-EVIDENCE.md` 有 main guard、默认调用、有限失败明细、UNAVAILABLE/截断及受限 resume 说明。业务 profile 仍须预先批准。 |
| A01 | LOCAL_FIXED | `weighted_moment_ext.py` 正权重归一/Kish、先中心化的无量纲标准矩；`test_merge_a01.py` 含1e±300与可表示1e12偏移参考。 |
| A02 | LOCAL_FIXED | `weighted_tail.py` 先剔除零质量再选尾部；分数成员数与所选尾部 Kish 双门，未移除既有支持要求。`test_merge_a02_a03.py`。 |
| A03 | LOCAL_FIXED | 加权风险族共享正质量归一，去绝对总质量阈值；各算子原缺失/负权政策保留。正质量尾部才决定值缩放。 |
| A04 | LOCAL_FIXED | `jump_robust.py` 窗内共尺度计算四阶统计；真零波动 NaN、有限负统计保留；独立公式与尺度反例 `test_merge_a04.py`。 |
| A05 | LOCAL_FIXED | `markov_dynamics.py` 稳定分位状态划分；13个已识别消费者更新语义版本，`test_markov_a05_a08.py`。 |
| A06 | LOCAL_FIXED | 精确稳定零点与零平台显式政策；不漏状态中心平衡点，解析对照保留。 |
| A07 | LOCAL_FIXED | KM 势阱由相邻 basin 边界决定，不使用远处最大脊。 |
| A08 | LOCAL_FIXED | AIS bins/history 声明与实际支持统一，联合可行域提前拒绝。 |
| A09 | LOCAL_FIXED / PARTIAL | `advanced_topology.py` Student-t 稳定尺度、固定df网格、两初值优化与score收敛门；`test_v9_a09_student_t_fit.py` 独立SciPy参考。非全拟合参数认证。 |
| A10 | LOCAL_FIXED | `robust_scale.py` 安全中点及偶数中位数，有限1e308/次正规数独立有理参考；`test_merge_a10_b12.py`。 |
| A11 | LOCAL_FIXED | `glr_change.py` 共尺度中心化，移除绝对平方和门；常数NaN、完美分割cap与penalty政策保留。`test_merge_a11.py`。 |
| A12 | LOCAL_FIXED | `downside_risk.py` 标准化满秩SVD、共同样本、window/max_lag可行关系；`test_v9_a12_a14_downside_risk.py`。 |
| A13 | POLICY_RETAINED | `distribution_break.py` 如实声明 clipped-U score，不再称返回值无偏；未改变既有非负截断，负原U反例保留。 |
| A14 | LOCAL_FIXED | surrogate seed绑定语义身份、绝对日期和证券，不再使用切片位置；Polars诚实CPU委托同核与坐标。缺日期拒绝/NaN，不伪造。 |
| A15 | LOCAL_FIXED / PARTIAL | Markov 按 selected_outputs 只保留消费者所需结果；仍有每列瞬态核工作区。小样本内存证据见下，不宣称100K最快。 |
| B01 | LOCAL_FIXED | `memory_ext.py` Geyer 相邻滞后对、正前缀与IMS累积最小；去绝对方差门，有限负估计不强剪。`test_merge_b01.py`。 |
| B02 | LOCAL_FIXED | `spectral_ext.py` 与 IR/中央签名使用立即输入声明类型；移除全历史数值猜测。生产未知/冲突拒绝；研究显式假设不覆盖已知类型。真实catalog及Polars测试见 `test_v9_b02_spectral_typed_paths.py`。 |
| B03 | POLICY_RETAINED | `directional_change.py`/`spread_estimators.py` 的原 DC panel 原子拒绝与 Roll 列原子失效政策保留，因果声明限定为已接受合法输入，不改成另一套局部屏蔽数学。 |
| B04 | LOCAL_FIXED | `tail_systemic.py` 双层窗 warmup=2*(window-1)，w20需38；`test_b04_b05_tail_history.py` 验证短warmup差异及正确重放。 |
| B05 | LOCAL_FIXED | prior 模式 min_periods<=window-1 关系与实际运行门一致，不让不可能组合全段空算。 |
| B06 | LOCAL_FIXED | `intrinsic_dimension.py` 有效Theiler与嵌入支持联合门；受限关系AST只增None身份/惰性条件，不开放任意eval调用。 |
| B07 | LOCAL_FIXED | 共中心/尺度、逐锚点有界top-k、log距离差避免平方上下溢；`test_b06_b07_intrinsic_dimension.py` 含1e±200和binary64高偏移界。 |
| B08 | LOCAL_FIXED | `rough_vol.py` 覆盖率均衡使用独立结构界，去恒真分母；`test_merge_b08_a13.py`。 |
| B09 | LOCAL_FIXED / PARTIAL | rough vol、5个intraday形状、MODWT correlation 使用无量纲一致分母；对应测试/版本更新。仅具名用例，非十个算子所有后端参数认证。 |
| B10 | PARTIAL / POLICY_RETAINED | RV 明确 `per_scale_left_complete_blocks_v1`，列出实际尺度cohort；n21端点仍为20/21。是否改共同右端cohort需明确估计量政策，未偷改/宣称关闭。 |
| B11 | POLICY_RETAINED | `intraday_vol_ext.py` 明确 Minute、rolling_cross_session/not_session_aggregate；不冒充交易日聚合，不猜市场日历。 |
| B12 | LOCAL_FIXED | 分箱 window>=bins*min_per_bin 中央关系与运行拒绝；curvature至少4 bins不放宽。`test_merge_a10_b12.py`。 |
| B13 | LOCAL_FIXED | `feature_geometry.py` 真beta在同一坐标缩放中比较；元数据/中央签名dimensionless，删过期window签名。`test_merge_b13.py`。 |
| G01 | LOCAL_FIXED | 保留另一任务删除旧覆盖helper的窄修，严格 `_table_current_snapshot_only` 为唯一权威；不整文件回退。 |
| G02 | LOCAL_FIXED | 4个递归滤波要求full history，非独立/无状态/伪checkpoint；`test_g02_g03_filter_contracts.py`。 |
| G03 | LOCAL_FIXED | Ehlers双输入平均项与状态声明同步、公式版本2；旧double-EWM近似改诚实exact CPUdelegate，不冒充native。 |
| G04 | LOCAL_FIXED | FP builtin受限注册，外部默认RESEARCH_ONLY；不因默认registry生命周期获得生产资格。 |
| G05 | LOCAL_FIXED | FP完整signature绑定含positional/defaults，再验证有限类型、范围；不能用NaN或位置参数绕过。 |
| G06 | LOCAL_FIXED / PARTIAL | 实现hash去绝对co_filename、保留常量/default/closure/helper；环境身份分开。测试支持路径变化稳定，不等于所有wheel/ABI兼容认证。 |
| G07 | LOCAL_FIXED | 递归拒绝不可认证可变捕获/默认值，执行前重算身份；不是只seal注册表。 |
| G08 | PARTIAL | FP exposure严格核对market/date/assets/classification，深冻结metadata及bytes-backed数组，请求assets预先快照。真实逐行PIT/immutable revision批准权威未接通，标量最大时间不是逐行证明。 |
| G09 | PARTIAL | DA唯一MissingReasonPlane填补白名单/age/immutable及严格Arrow传输；FP重构DA规范类型。真实DA→FE→QE逐单元生产者与含轴/快照的持久身份链未完整接通。 |
| G10 | PARTIAL | 生产recipe要求原ExecutionContext和所选backend、共享budget，不再新建research/Pandas上下文；fit权威、无FE不静默回退。上下文spy测试不是批准真实数据生产链认证。 |
| G11 | LOCAL_FIXED / PARTIAL | 保留新Pareto required_objectives及合法hash差异fixture、缺目标拒绝；只读复核74 passed。FO的trial Pareto不是FA assembly，当前无FO→FA assembler生产接线，不能称完整消费链通过；远端CI未运行。见 `new-merge-g11-review.md`。 |
| G12 | PARTIAL | 主树绑定/重复定义复核及新解释器证据；23历史refs功能语义仍未全部裁决。新进程不代表已加载旧worker，未经授权未排空/重启。 |

## 有界执行证据与性能账本

- `new-merge-root-numerical-final.log/.xml`：166 passed，2 warnings；其后增补高偏移用例/更严格spectral边界有独立最终回执，不将旧日志改写为新结果。
- `new-merge-contract-math-final.log/.xml`：224 passed，2 warnings，具名滤波/Markov/Huber/t-fit/参数关系/语义版本/DA缓存与missingness范围。
- `new-merge-fp-final.log/.xml`：406 passed，1 xfailed，4 warnings；从FP包目录执行。xfail不计通过。
- 用例范围可能重叠，不相加为算子数量或全仓覆盖率。历史失败、夹具迁移前失败均保留。
- `new-merge-operator-manifest.json` 是当前目录导出，约3.8MB；commit元数据只是dirty基线，不能单凭该字段认证实际源。源码身份见最终附录。
- A15 synthetic 360×24×8：返回数组12,994,560→1,728,000 bytes，峰RSS227,104→205,816 KiB；4.34s→4.44s，**没有速度提升证据**。仍非进程族/100K/真实数据负载。
- A14与G03 Polars路径诚实记录CPU委托；没有宣称这些核是GPU/native。G10继承上下文，不另申请80%资源池。
- 默认完整1000因子的另任务证据：`main_merge_fe_1k_actual_20260909.json`，tiny synthetic CPU 8×32、column/scalar，1000/1000、13.56s；不能外推100K/复杂模型/生产COS。
- 无批准 `FACTOR_ENGINE_V2_PROFILE` 的真实业务范围，本轮不猜表/股票池/日期/COS目标，也不发布因子或生成大型数据产物。

## 未完成项需要什么

真实读取/吞吐验收需要已批准业务profile及明确只读/非生产输出目标；旧worker生效需要排空/重启授权。B10需要确认统计cohort政策，不能由性能目标代替数学决策。G08/G09及生产持久资格链仍有实际接线工作，不能称为仅缺测试或全部完成。

所有修改保留在唯一正式主代码区。剩余磁盘约786GB（2026-09-09本轮检查），没有删除正式数据或未裁决历史代码。
