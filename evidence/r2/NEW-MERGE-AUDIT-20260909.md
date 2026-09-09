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

## 最终附录：以后置实际回执为准

1. **核心综合回归：607 passed、1 xfailed、93 warnings**。精确42文件命令、源码全清单及前后摘要：`main_merge_20260909/new49_frozen.json`，实际日志同名 `.log`。该运行无 `-k`，新增数学/桥接/关系/存储/缺失原因/跨包和11个默认资源、resume测试文件整文件执行。运行期间唯一变更为**未列入该命令**的 `factor_engine/tests/integration/test_final_closure_fe_integration2.py` 夹具，运行源码未变化；回执如实为 SOURCE_CHANGED_DURING_RUN，不能说整棵树不可变。
2. **FP全套最终：406 passed、1 xfailed、4 warnings**，`new-merge-fp-frozen.log/.xml`。`new-merge-fp-frozen-before.json` 与 `new-merge-fp-frozen-after.json` 逐字一致，含源码摘要及清单。两轮相同用例不相加为独立覆盖率。
3. 两项 xfail 各自保留原因：FE safe multi-wave worker epoch reuse 尚未实现（`test_multi_wave_epoch_reuses_pid_only_after_ownership_proof`）；FP `hp_filter` 全 lagged sample 离线目标非prefix-causal，已标 OFFLINE_ONLY、禁止生产（`test_hp_filter_is_prefix_invariant`）。都不是通过或已完成能力。
4. 保留失败历史：`new-merge-bridge-fixtures-final.log` 的37 failed/38 passed是旧多输入夹具及隐式research调用；修复后同类夹具63 passed并进入上述607范围。`main_merge_20260909/new49_final.log` 的1 failed/521 passed抓到D02撤销遗留的 `callable(store.scan)` 分支条件，已删除，非批准路径保留原lazy adapter兼容，最终607通过。未删失败条件或放松生产门。
5. 三份新增测试曾覆盖中央 `strict_fiscal_parameter_domain_certification_guard`；root复核后已删除这些no-op覆盖。`logs/v9-owned-tests-with-central-guard.log` 是66 passed/74 deselected的独立选定回归，最终607运行没有这个名称筛选并使用原中央guard。此前较小范围日志不冒充完整认证。
6. D01最终实现在 `data_access_source.py`，SHA256 `a0662e6f2ee48fcdac56fb754351d410b3a54eecbc66ea9abe92a754e394fcf9`：请求实际单位状态、空筛选typed schema、`UNIT_NORMALIZATION_ALGORITHM_ID` 纳入持久source_dependency_hash，算法身份变化只清理该source的缓存，保留snapshot。新冷暖identity测试进入607范围；未全局清空历史数据。
7. **D02未完成，且未半开默认路径**。真实 `ScanHandle.collect()` 返回的 `ReadResult.snapshot` 是 `DataSnapshot`，没有 `ResolvedSourceSnapshot.content_digest`；早期mock后置校验不能用于实际读取。这轮D02实验helpers已窄撤，`polars_lazy.py`/`scan_handle.py` 相对进入时6b610d无新增变化；本轮失败实验测试移除，不计通过。批准内容路径维持eager ReadHandle精确检查。后续需把真实DataReadIdentity或经过验证的文件集映射完整接入，而非给测试snapshot塞假摘要。
8. B02最终IR权威 SHA256 `109d79c2a2885fc02e584c4892feda05bae22af981a7a626aefdba0240af53a9`：真实catalog复权价格→returns/pct_change→ReturnDecimal已贯通；泛化数值变化/对收益再pct不冒充金融收益，明确类型不被遗留price_basis覆盖。测试不代表真实批准profile执行。
9. B12私有核也在结果数组分配前拒绝不可能参数组合，新增2个allocation-negative测试进入607范围；没有为参数错误先分配整面板。
10. G11证据的正确路径是 `new-merge-g11-pareto-contract-review.md`（表内简写链接以此为准）。74个具名合同测试范围不等于不存在的FO→FA生产assembler接线。
11. G12 root独立复查：`new-merge-duplicate-recheck.json`，1,626文件、0解析错误、0重复顶层函数/类，带实际全文件hash。先前人工抄录的无效Git对象已在 `new-merge-duplicate-binding-review.md` 纠正，没有把错误身份作为关闭证据。陈旧build目录候选仅列账，不误删正式源码/历史独有代码。
12. `new-merge-operator-manifest.json` 经当前真实bootstrap重新导出，**1,756 entries**、约3.8MB。导出基线87c69f5e，实际工作树仍dirty；metadata是目录声明，不是1,756算子的独立数学/所有后端认证。环境版本见 `new-merge-environment.json`。
13. 外部在实施期间连续产生“V10实现波/尾随”提交；本任务未执行。当前回执记录基线 `87c69f5e0dabc3d56f915097b56cf84b7a405f84`，之后的窄修留在同一main工作树；保留外部提交，不回退、不推送。
14. 本轮自建小补丁已清理，正式代码和有界日志保留。没有复制仓库/数据集，也没有删除生产数据。大规模COS/真实数据/worker重启仍未执行。

退出边界仍以49项表中的PARTIAL/EXTERNAL_NOT_RUN为准：D02工程接线、G08逐行PIT与版本批准、G09全链路逐单元原因、G10真实批准上下文执行、G11真实消费者、G12历史语义与旧worker、R01完整跨波恢复、Q01逐算子数学/后端、P01真实100K性能，以及B10统计政策均未被这批局部通过替代。

### 发布夹具最后复核与稳定身份

`main_merge_20260909/new49_publish_final.json`：发布整文件+D01实际单位及缓存身份两文件，**48 passed、4 warnings**，无筛选；源码前后摘要一致：`2ae74b5d3609bc8753dea77e5fe8946c5f4d425a04f2f4363a93819bed48cbc5`，status=PASS。
发布fixture显式声明LOCAL storage，使用真实临时Parquet及正式 `produce_coverage_receipt` / `write_staging_identity` 生成coverage/manifest/generation身份，补齐expected参数；没有手写假coverage proof或禁用发布校验。保留失败watermark不变、成功幂等与错误generation拒绝断言。数据源仅测试fake store，不是生产发布/COS/真实数据认证。
初次缺get_dataset的三失败及时间字符串格式的一失败分别保留在 `new-merge-publish-fixture-final.log`、`new-merge-publish-fixture-postfix-rerun.log`；聚焦3 passed在 `new-merge-publish-fixture-final-pass.log`，最终整文件由48项回执覆盖。不把重跑/重叠范围相加为唯一测试数。
最终本任务未commit/push；基线87c69f5e及dirty源码摘要均记录。自身临时小补丁清理完成，未删除正式数据/历史独有代码。未完成工作仍如上表，不能宣称49项全部关闭。
