# server-c 主代码合并整改：当前活跃关闭表

任务来源：用户的《server-c_主代码合并检查与统一整改.md》，M01–M22。
正式工作树：`/home/sunhaiwei/quant_projects`。
基线 HEAD：`6b610d0c8be7a3e6a7516ee2f4f8de6b58765f4f`。
本轮开始工作树干净；下述修改未提交，不能用 HEAD 代替实际修改后源码身份。
未创建分支、worktree、整库或虚拟环境副本；未 commit、push、部署或发布生产因子。
保留用户其他任务的并行修改与 23 个尚有语义待审的历史引用。

## 状态解释

- LOCAL_FIXED：本地具体缺陷修复和具名回归，不等于远端 CI 已执行或生产认证。
- PARTIAL_ISOLATED：可定位缺陷已经修复/拦截，但生产证据或端到端接线仍有缺口。
- PARTIAL_REVIEW：实际完成有界审查；未审历史范围仍保留，不猜测全部恢复。
- 各测试 JSON 的 SOURCE_CHANGED_DURING_RUN 表示运行期间全树代码变化，不能作为
  一份不可变源码快照的认证；pytest 实际输出仍保留，不改写成失败或全树通过。

## 逐项关闭表

|编号|当前状态|主实现及修复|执行证据与明确边界|
|---|---|---|---|
|M01|LOCAL_FIXED / REMOTE_CI_NOT_RUN|QE 显式纳入 probe_portfolio 子包；移除同名平铺实现，历史名称单向指向唯一 legacy 真源；CI 对 scripts 的源码枚举与实际载荷一致|最终 wheel 156/156 CI枚举、158 Python文件+1资源逐字节通过；主目录和archive-only两资产/两腿/H1/已知费用金标一致。最终hash见下；未触发远端CI安装流程|
|M02|LOCAL_FIXED|FE 现有合同序列化器明确编码 RelationalParamSpec 类型、关系与字段，未知对象仍拒绝|fe_da_final.json 包含 CI 精确 DETERMINISM 两文件及关系变更/稳定性反例|
|M03|LOCAL_FIXED|跨库 Pareto 夹具声明冻结 required_objectives，保留政策改变身份正例及缺目标负例|cross_contracts.json：integration_tests 与根目录新增检查合计 103 passed；未撤除新门槛|
|M04|LOCAL_FIXED / CI_NOT_RUN|测试依赖独立锁 requirements/ci-test.txt，同一 job 解释器安装；真实千因子运行及落盘 oracle 可执行|main_merge_fe_1k_actual_20260909.json：1000/1000，CPU、8×32、column+scalar，13.56秒；不是生产或十万全指标|
|M05|PARTIAL_ISOLATED|FA raw evidence 绑定候选、配方、状态、数据、股票池、标签和值制品；要求可信 resolver 核对而非只看 caller hash|无生产持久化 trusted-evidence resolver 被接通，缺 authority 的正式请求拒绝/等待。接口与测试 resolver 不冒充真实 QE→持久化→FO factory 完整消费者|
|M06|LOCAL_FIXED|所有候选在映射/基线选择前校验指标单位；RAW 本身不再绕过|FA 选择回归，包含基线与变体错误单位隔离；不能用高分补偿|
|M07|LOCAL_FIXED|EQ 两侧都来自 equivalence_margin；SUPERIOR 使用 minimum_gain|FA 关系区间与非对称 NI/EQ 反例；非显著不等于等价|
|M08|LOCAL_FIXED|named axes、metric axes、正权重所需指标与单位错误先隔离，再调用 strict mapper|FA 混批反例保留健康候选可处理，不用整批抛错冒充无赢家|
|M09|LOCAL_FIXED（配对边界）|original_raw 与替换/关系比较统一核对上下文及配对身份；实际QE生产者生成计划内容与样本/时间/掩码身份|公开入口参数化覆盖4个pair字段与data snapshot/universe/label。不同实际样本的候选记录不可比较，不再整批抛错；持久可信存储仍受M05边界限制|
|M10|LOCAL_FIXED|每个 replicate 内按冻结 window/scenario 权重先聚合，再对 replicate 效应求区间；具名 effect_estimand|保留“压平格子得负下界、正确聚合得正效应”的反例；压力情景硬门不被平均抵消|
|M11|PARTIAL_ISOLATED|required_dimensions 不允许 NONE 伪装合格；健康维度与 coverage 要匹配可信 evidence bundle|没有真实生产 bundle resolver 时不发正式资格；单纯 caller 自报等级不算可信来源|
|M12|PARTIAL_ISOLATED|资格回执及冻结 assertion suite 纳入 authority；数值资格与用途 admission 分离，绑定具体evidence bundle，校验时区时间、失效和撤销|25项独立authority合同反例及公开入口回归通过；用途 admission 的真实发布/撤销权威尚未接通，研究/生产能力明确分开|
|M13|LOCAL_FIXED|包括 k=1 在内先按逻辑簇聚合支持，runner-up 来自另一个簇|聚类同簇双成员不再被当作两个竞争簇；证据见 main_merge_m13_m16.json|
|M14|LOCAL_FIXED|正式逐窗口证据校验计数、有限数、区间顺序/范围与 HAC 语义；绑定真实窗口边界/样本身份去重|改名的相同真实窗口不能虚增确认数；QE→FA 适配器传递身份，不只去重显示名；clustering_final.json含真实生产者与消费者回归|
|M15|PARTIAL_ISOLATED|近似归属 RESEARCH_PROPOSED；overlay 保留资格域与来源；默认 UNVERIFIED，不能通过新成员升级未认证父版本|生产 overlay 构造明确拒绝，直到存在可信 QE 资格解析与最终发布消费者。没有声称现已提供完整生产发布功能|
|M16|LOCAL_FIXED（研究生命周期）|核对新簇/迁出/退休/全量刷新、代表成员与分区；模型变化要求 review refs|研究候选 lineage 测试；最终生产发布仍受 M15 隔离，不把研究 refresh 标为生产|
|M17|LOCAL_FIXED（已测资源范围）|GPU PnL 上传/计算 OOM 进入减半重试，F=1 明确拒绝；主机结果预算；预建因子列索引；取消全量 PnL 预上传|真实 NVIDIA L20 CUDA focused 回归含上传后 OOM、计算 OOM、tile一致性、释放与预算；不是任意规模显存认证|
|M18|PARTIAL_ISOLATED|规模 gate 区分 row/factor/GPU/realdata；CI 状态收集不代替执行证据；解析真实 FE disk-oracle 报告|实际1k因子范围通过；十万因子全指标、真实市场影子任务、手动CI保持 NOT_RUN|
|M19|LOCAL_FIXED（溯源）|账本保留原始来源/哈希及矛盾；手工 FIXED_LOCAL 不自动覆盖成当前 VERIFIED|test_main_merge_ledger.py；历史 v3 账本为历史来源，不认证当下源码；当前实际回执另行链接|
|M20|PARTIAL_REVIEW|23 refs 的1384路径实例按当前字节及迁移路径分类；冷启动缺失默认目录改为现有规范 A股/daily/core catalog，唯一包装真源|89字节相同、56历史支撑制品、138替代合同待审、1101语义待审；FE/DA有界家族82项回归通过，不等于全部旧修复恢复|
|M21|PARTIAL_ISOLATED|环境工具从仓库外在独立进程核查解释器、18个关键模块、真实加载函数及163指标注册身份|runtime_identity 回执仅证明新进程；没有旧生产worker任务接管/重启验收；未越权终止或部署|
|M22|LOCAL_FIXED（已审入口）/ 更广范围未认证|清理6库重复顶层定义、FE numba 同名module/package、失效 shadow定义；生成实际入口→函数/合同/registry/包装/消费者静态映射|AST 唯一性回归、源码/轮子小金标及真实跨库回归；静态映射137绑定+26未绑定项不当数学证书|

## 实际执行回执

主解释器均为 `/home/sunhaiwei/quant_projects/.venv/bin/python`，CPython 3.12.3。
每个具名 pytest 回执保留命令、全树源码 SHA-256 清单、前后摘要、退出码、耗时和日志哈希：

- `main_merge_20260909/qe_full.json`：1241 passed、2 skipped；实际 CUDA 可用。跳过是既有 metamorphic 文档合同，不记为通过。
- `main_merge_20260909/fo_full.json`：885 passed。
- `main_merge_20260909/fp_full.json`：377 passed、1 xfailed；预期失败不记为通过。
- `main_merge_20260909/cross_contracts.json`：103 passed。
- `main_merge_20260909/fe_da_final.json`：53 passed，源码摘要前后稳定。
- `main_merge_20260909/fe_da_m20.json`：82 passed，源码摘要前后稳定。
- `main_merge_fe_1k_actual_20260909.json`：真实执行与磁盘逐值 oracle；tiny synthetic 范围。
- `main_merge_entrypoints_20260909.json`：163项静态清单，非执行认证。
- `main_merge_runtime_identity_20260909.json`：独立新进程真实导入来源，非生产worker认证。
- `main_merge_legacy_fe_da_review.md`：保留旧失败与安全测试迁移后的范围化结果。

先前通过的全套若随后受本轮合同修改影响，必须读下方最终回归附录；
本表不自动把旧结果升格为最终源码认证。并行 FE/DA 新任务编号与本表 M01–M22 不混用。

## 历史制品、迁移和保留

涉及错误入口/单位/候选证据绑定/无效资格/窗口重复/旧合同 hash 的历史结果
应按实际命中范围失效重算并保留替代引用；本轮没有删除全部历史评价、缓存、
checkpoint、模型配方或生产数据。修改后的 FE 合同/函数身份不能沿用旧能力缓存
冒充同一语义。尚无真实生产 resolver/owner 时不猜 COS bucket 或制造资格。

## 尚未闭合的验收

本轮不能宣称“22项全部生产闭环”：生产 trusted-evidence 持久化及最终消费者、
用途发布/撤销与生产聚类发布、23 refs 完整语义裁决、旧worker纪元接管、
十万因子全指标/真实市场数据运行仍未验收。安全隔离是已实现保护，不是这些
能力已实现的代名词。远端 GitHub main 未由本任务更新。

## 最终回归附录（具体执行范围）

- `main_merge_20260909/cross_final.json`：103 passed，前后源码摘要稳定。
- `main_merge_20260909/qe_producer_final.json`：180 passed，前后源码摘要稳定。
  QE 唯一真实 joint bootstrap 生产者现在计算计划内容、实际抽样索引字节、时间
  与完整共同有效网格身份，7个新增生产者反例覆盖计划改变、重排、单独执行、
  无效列隔离；未制造本来缺失的配方/状态/生产证书。
- `main_merge_20260909/clustering_final.json`：55 passed，前后源码摘要稳定。
  覆盖 QE pairwise 与 FA 增量、资格域、生命周期公开调用；生产发布仍明确拒绝。
- `main_merge_20260909/trusted_store_final.json`：8 passed，前后源码摘要稳定。
  store 在攻击前预置并不读取攻击候选来生成响应；保留正确候选正例，以及
  samples/state/value/交换完整raw后重算caller hash、健康/coverage反例。
- 中间 `main_merge_20260909/fa_full.json` 的 27 failed / 1468 passed
  原样保留：新配对合同已改、旧夹具未完成迁移时运行。不得用此前的1495通过
  覆盖此次失败；最终完整 FA 重跑应另有具名回执。
- 临时千因子输出 `/tmp/main-merge-scale.P3UX2f` 已清理。
  删除前 `cmp` 确认正式证据目录保留的 report 与原始执行报告逐字节一致。
  清理的是可重新生成的约2MB synthetic数组与临时进度文件，不是正式数据。

### 收口重跑与最终包装

- `main_merge_20260909/fa_post_pair_isolation.json`：**1506 passed**。
  运行期间仅记录到另一个FE任务的3个源/测试文件变更；本回执保留明确路径。
- `main_merge_20260909/fo_post_pair_isolation.json`：**885 passed**，源码前后稳定。
- `main_merge_20260909/qe_final.json`：**1273 passed、2 skipped**。
  其中25项为新增根目录准入合同，QE本身1248项通过；全树有并行变化，不当作
  一个不可变提交的全仓认证。
- `main_merge_20260909/authority_rerun.json`：**77 passed**。
  含公开入口的plan/sample/time/mask及数据身份反例、正确配对真改善正例、
  冻结store对抗反例和用途准入。测试期间仅新增另一个FE任务的测试文件。
- `main_merge_20260909/authority_final.json`：保留混合收集失败。
  根tests和FO tests的Python包名冲突导致`tests.search`不可导入；改为分进程
  执行各包测试后得到上述回执，没有删除业务测试或放宽生产保护。
- `main_merge_20260909/cold_start_final.json`：**4 passed**。
  规范包配置提供src-layout测试入口；生产代码不插入sys.path、不路由旧树。
  冷启动sdist/wheel和独立安装结果见`../main_merge_m13_m16.json`，
  其中旧1495测试数已明确标为历史，非当前关闭结论。
- `main_merge_qe_wheel_final_20260909.json`：
  最终QE wheel SHA-256
  `971329a73712190f2de84d975a65d4f2b1171ff41959b6de8665126ad313a1d5`。
  最终实际CI完整性检查器本地执行156/156；逐字节检查158 Python+1资源通过。
  source和archive-only金标净收益均为[0,-0.001,0.013,0.013,0.013]，
  成交成本均为[0.0003,0.0006]。archive-only进程因Numba zip缓存限制明确关闭
  JIT；不是清洁CI的已安装wheel进程认证。
  约3MB临时build/wheel已在核对hash及保存回执后清理，可从主树重新构建。

最终代码仍在主工作树未提交；上述回执与静态入口/新进程身份都已落在该树。
本轮没有自动上传GitHub，也没有以清理名义删除历史独有修复。
