# Factor Optimizer 中文功能指南

本文解释算法、活动设计和真实运行边界。逐模块、类、函数、方法、字段与默认值请配合 [完整 API 参考](API_REFERENCE.md) 查阅。

## 1. 定位与边界

Factor Optimizer 是候选因子的实验编排与证据治理层。它负责提出合法变异、控制搜索成本、组织训练与验证、冻结赢家、一次性消费封存测试集，并把可追溯结果交给下游治理。

它不是投资组合优化器，不求解持仓权重、风险预算或组合有效前沿；不直接执行因子，不自行计算 Rank IC、收益或换手率，也不决定生产入库。Factor Engine 提供执行、算子元数据、合法性与 canonical identity；Quant Evaluator 提供指标证据；Factor Assets 执行最终 Promotion Gate。

## 2. 当前真实能力

版本 0.1.0 的机器可读状态为 research_only，production 会 fail-closed。阻塞项包括可信的 split-aware QE 生产集成、生产合法性链和 sealed-test 证据升级链尚未完成。

还应注意：

- SearchConfig 默认 screening_only 为 True；可产出筛选和诊断分数，但不能推进质量层级或冻结可选择赢家。
- SearchRunner 当前串行，max_concurrency 只能为 1。
- 默认必须提供 EvaluationProtocol 与已校验 SplitPlan。
- max_llm_calls 为 None 表示不允许 LLM 调用，并非无限额度。
- sealed test 只能在搜索结束、赢家与执行规格冻结后消费一次。

## 3. 完整流程

1. 声明父因子、假设、目标指标与允许变更的参数。
2. 建立 SearchSpace，或从结构化诊断编译有界修复路线。
3. 用变异语法和 Factor Engine validator 拒绝非法候选。
4. 在 train 构造候选，在 validation 选择候选。
5. 用便宜预筛、tiered evaluation 与 multi-fidelity 把昂贵评估留给幸存者。
6. 把成功、重复、非法、异常、预算拒绝和实际成本全部写入 TrialLedger。
7. 把方向和尺度不同的指标映射为 desirability，建立 Pareto 前沿。
8. 用稳健效用、不确定性感知或互补赢家集做选择。
9. 冻结候选身份、目标、切分、策略、快照和执行规格。
10. 通过 test authority 一次性执行 sealed test。
11. 输出 treatment 决策、证据引用与 lineage，交给下游 Promotion Gate。

缺失证据必须显式不可用，不能解释成 0、最简单或通过。

## 4. 输入与输出

### 4.1 输入

- 因子身份：factor id、mutation id、canonical hash、父子血缘。
- 变异：变异族、参数名和值、参数角色、预期作用。
- 搜索空间：连续、整数或字符串分类参数。
- 目标：ObjectiveSpec 中唯一权威的 metric name 与 maximize 或 minimize。
- 切分：train、validation、sealed test 边界以及 purge、embargo。
- 预算：提案数、完整评估数、成本单位、可选 LLM 次数。
- 评估协议：返回结构化 TrialEvaluationArtifact 的适配器或协议。
- 治理证据：稳健性、复杂度、多重性、完整性、validator 与 library snapshot。

### 4.2 输出

- Trial 与追加式 TrialLedger。
- SearchSession：最佳试验、停止原因、预算、阶段轨迹和 sealed 状态。
- 评估、统计和多重性制品。
- ParetoFrontier 或 ParetoArchive。
- 单赢家或小型互补赢家集。
- 一次性 sealed-test 结果。
- treatment 决策、诊断、修复候选与 lineage。

## 5. 变异语法

默认注册六类变异：

- parameter_tune：调整普通参数；
- window_adjust：调整观察窗口；
- operator_swap：替换允许的算子；
- decay_adjust：调整衰减；
- linear_combination：组合父表达式；
- threshold_adjust：调整阈值。

ParameterSpec 同时声明 ParameterKind 与 ParameterRole。角色守卫会拒绝当前不生效的参数；“值能解析”不等于“变异合法”。MutationValidator 返回结构化结果，非法候选应在 QE 评估前失败。真实活动还应绑定算子快照和 TrialValidatorIdentity，防止恢复后语法权威漂移。

## 6. 搜索空间与策略

### 6.1 ParameterSpace

- float：连续闭区间，可按对数尺度采样。
- int：整数闭区间；边界必须为整数，可按对数尺度采样后取整。
- choice：非空、唯一的字符串集合。

参数名必须非空，数值下界必须小于上界，对数尺度上下界都必须为正。SearchSpace 要求参数名唯一，并严格检查候选缺字段、多字段、类型和值域。

### 6.2 策略选择

| 策略 | 适合 | 限制 |
|---|---|---|
| RandomSearch | 大空间、早期探索、混合参数 | 不利用历史分数 |
| GridSearch | 低维、离散、需要完全覆盖 | 维度高时组合数大 |
| BayesianSearch | 昂贵的连续或整数小中维搜索 | 数值核拒绝 choice |
| TPESearch | 历史较多、响应不平滑的数值空间 | 数值核拒绝 choice |

GridSearch 用混合进制索引惰性解码，不物化整个笛卡尔积；max_combinations 可按种子选有界子集。策略 spec 保存类型、构造参数与种子，策略 state 另保存随机状态、历史与迭代位置；二者共同保证恢复后的提案轨迹一致。

BayesianSearch 的实际提案过程如下：

1. 前 n_initial 个点随机采样，默认 5。
2. record 把原始目标按方向转成统一的“越大越好”效用。
3. 使用 RBF 核的高斯过程拟合历史；长度尺度取已观测点两两距离的中位数，观测噪声为一个很小的正数。
4. 默认用 Expected Improvement，也可用 UCB；UCB 的默认探索系数为 2.576。
5. 每轮随机产生 5000 个数值候选，计算 acquisition，选择最大者；整数维在还原参数时遵守空间类型。

因此它不是对真实目标做梯度下降，而是在“代理均值高”和“代理不确定性大”之间取舍。空间尺度差异很大时应先做合理参数化；当前实现没有自动标准化分类距离。

TPESearch 的实际提案过程是：

1. 前 n_initial 个点随机采样，默认 5。
2. 历史分数同样先转换为统一效用，再从高到低排序。
3. 前 gamma 比例构成 good，默认 gamma 为 0.25，其余构成 bad。
4. 每个数值维分别建立高斯核密度；带宽使用基于样本标准差和样本数的规则，并设置正下限。
5. 默认从 good 密度采 24 个候选，计算各维对数密度比之和，选择 good 密度相对 bad 密度最大的点。

TPE 在此实现中是独立维度 KDE 的数值策略，不会自动学习复杂的参数交互，也不支持 choice。

### 6.3 小型示例

~~~python
from factor_optimizer.contracts.objective import ObjectiveSpec
from factor_optimizer.search.strategies import ParameterSpace, SearchSpace, RandomSearch

space = SearchSpace([
    ParameterSpace("window", "int", low=5, high=60),
    ParameterSpace("decay", "float", low=0.01, high=0.5, log_scale=True),
    ParameterSpace("winsor", "choice", choices=["mad", "quantile"]),
])
strategy = RandomSearch(
    space,
    seed=20260918,
    objective_spec=ObjectiveSpec("validation_rank_ic", "maximize"),
)
trial = strategy.propose()
~~~

trial 只是提案，仍需语法、预算、切分和评估证据。

## 7. SearchConfig、Runner 与预算

### 7.1 关键配置

| 参数 | 默认或规则 | 含义 |
|---|---|---|
| budget | 必填 | 四维预算 |
| plateau_window | 20 | 近期分数窗口 |
| plateau_threshold | 0.001 | 最小相对改进 |
| enable_multifidelity | True | 启用保真调度 |
| max_concurrency | 只能为 1 | 当前串行 |
| evaluation_cost_units | None | 单次预留成本 |
| execution_mode | research_only | production 当前拒绝 |
| require_evaluation_protocol | True | 缺协议即拒绝 |
| objective_spec | score 最大化 | 指标和方向权威 |
| tiered_evaluation | None | 可选业务漏斗 |
| candidate_strategy | None | 可选提案策略 |
| durable_campaign_store_path | None | 可选 SQLite |
| screening_only | True | 默认不能冻结赢家 |

objective_direction 只是兼容字段；若与 ObjectiveSpec 冲突，构造立即失败。

SearchBudget 默认允许 100 个提案、50 次完整评估、1000 成本单位，不授权 LLM。BudgetTracker 使用预留、开始、提交或释放协议：调度前同时预留评估名额和预计成本；成功按实际成本结算并记录超支，失败释放。attempt id、金额或状态不匹配都会失败，从而避免恢复边界上的重复收费和超卖。

下面是接线示意，不是可直接运行的完整程序；next_trial 与 evaluation_protocol 是业务方必须实现和绑定的两个端口。

~~~python
from factor_optimizer.contracts.objective import ObjectiveSpec
from factor_optimizer.contracts.search_budget import SearchBudget
from factor_optimizer.search.runner import SearchConfig, SearchRunner

config = SearchConfig(
    budget=SearchBudget(max_trials=30, max_evaluations=12, max_cost_units=120.0),
    objective_spec=ObjectiveSpec("validation_rank_ic", "maximize"),
    screening_only=True,
)
runner = SearchRunner(
    config=config,
    proposal_fn=next_trial,
    evaluation_fn=evaluation_protocol,
    strategy=strategy,
)
session = runner.run()
~~~

普通回调不能冒充 split-aware protocol；提案函数、协议与策略还必须共享候选语义和目标方向。

## 8. Tiered evaluation 与 Multi-fidelity

二者都节省成本，但职责不同。

TieredEvaluationPolicy 是业务漏斗。默认四层成本倍率为 1、5、25、100，保真阈值为 0、1、3、4；成本与保真必须严格增加，候选不能跳级。promote_after 表示需要多少次连续成功。每个 issued job 与 outcome 都绑定 candidate、recipe version、tier、attempt 和 evaluation ref，因此旧配方的晚到结果不能晋级新配方。

MultiFidelityScheduler 描述数据和计算精度：

| 档位 | 样本比例 | 指标数 | 交叉验证 | 稳健性 | 成本倍率 |
|---|---:|---:|---|---|---:|
| L0 | 5% | 1 | 否 | 否 | 1 |
| L1 | 15% | 3 | 否 | 否 | 3 |
| L2 | 50% | 5 | 否 | 否 | 10 |
| L3 | 100% | 8 | 是 | 是 | 25 |
| L4 | 100% | 12 | 是 | 是 | 50 |

这些是调度规格，不会自动让 QE 生成指标；评估器必须真正执行对应 sample fraction、metric set、交叉验证和稳健性，并返回可核验 stage profile。

PromotionCriteria 可组合最低分、同层前若干比例和相对基线改进；require_all 决定全部满足还是任一满足。所有晋级只能看 train 和 validation，sealed test 永远不得参与。

## 9. 评分、Pareto 与赢家

原始指标不能直接相加。Desirability 用预先版本化的 anchors 把每项映射到 0 到 1，并声明越高越好或越低越好。catastrophic floor 可让灾难值直接归零。anchors 不得在看完结果后调整。

ParetoPoint 保存 trial id、目标向量、方向和元数据。ParetoFrontier 只保留非支配点；ParetoArchive 提供归档。Pareto 只回答“不被谁全面击败”，不会自动给最终赢家，完整性、切分、可交易性和灾难阈值等硬门必须先执行。

主赢家选择使用稳健均衡效用：

~~~math
U = a \min_j d_j + b \left(\prod_{j=1}^{m} d_j\right)^{1/m} + c r - \lambda k
~~~

d 是各维 desirability，r 是稳健性，k 是复杂度。所有输入必须有限且在 0 到 1 内，不静默裁剪。任一维为 0 时几何均值项为 0，最差维项也惩罚它，所以高 Rank IC 不能补偿灾难性换手或容量。

效用近似并列时优先复杂度低者，再按 trial id 确定性排序。缺稳健性或复杂度映射会 fail-closed，绝不默认成 0。

select_complementary_winners 还要求 eligible 明确为 True、incremental value 达标、每 family 不超限，并用相关性去重。空赢家集是合法结果。

UncertaintyAwareWinnerSelector 用 bootstrap 样本计算置信区间、下界效用、支配概率和等价区域，避免在噪声内强行排序。相同日期、fold 或 draw 的候选应使用 compare_paired_draws，保持样本顺序并分析逐样本差。统计消费证据应覆盖所有提案，而非只数成功候选。

## 10. Plateau 停止

PlateauDetector 取最近窗口，前半形成基线，后半取当前最好值；maximize 与 minimize 通过方向统一。相对改进和可选绝对改进可要求同时或任一满足，use_median 可降低偶然尖峰影响。

- 历史少于窗口时不判平台。
- 奇数窗口把多出的观测放到后半段。
- 基线为 0 有显式分支。
- 平台只表示继续搜索的边际收益低，不代表候选通过准入。

AdaptivePlateauDetector 会按阶段和波动调阈值；MultiObjectivePlateauDetector 面向多目标。恢复时必须恢复完整分数历史。

## 11. 诊断驱动修复

DiagnosisKind 和 DiagnosisCategory 把问题标准化。DiagnosisRepairPolicy 把诊断映射到 RepairFamilyDeclaration，其中包含执行域、因果类别、参数 schema、prior、所需证据和预算。diagnosis_search_budget 给每类诊断分配有界候选槽位，禁止无限扩张。

有些诊断应修改信号表达式，如窗口、衰减、去极值和中性化；有些属于消费端，如换手抑制、持有期或组合构造。route_diagnoses 会区分 signal treatment 与 portfolio recipe。

portfolio recipe 只是把冻结配方交给显式 consumer 评估，并不让本库成为组合求解器。SearchRunner.for_diagnoses 遇到组合路线时要求 portfolio_recipe_context 提供 signal ranks 和 callable consumer，缺一即拒绝。

TreatmentDecisionPolicy 依次处理证据完整性、诊断、修复建议、预期收益、风险、接受或重试或拒绝、理由与血缘。硬性 integrity gate 不能被高分补偿。修复后必须绑定父因子、变异、证据 schema 和 digest。

## 12. Train、Validation 与 Sealed Test

train 用于拟合或构造候选，validation 用于模型选择。SplitPlan 表达时间边界、purge overlap 和 embargo。漏斗晋级、平台、Pareto 和赢家选择只能使用 search data。

冻结前应固化：

- winner trial id 与 canonical identity；
- objective spec 与方向；
- split semantics hash；
- execution plan hash；
- strategy spec 与 state；
- selected execution spec；
- library snapshot 与 validator identity；
- search-time evidence refs。

冻结后 sealed 状态不可通过普通赋值重置。TestAuthorityBroker 集中控制 test 数据，SealedTestExecutor 校验测试与搜索不相交、执行规格匹配和 reservation 有效后消费 seal。sealed test 只估计冻结决策的外样本表现，不能用于调参、晋级或重新排序。需要新 treatment 时应建立新活动、预算与 seal。

## 13. 持久化与审计

durable_campaign_store_path 可绑定 SQLite。恢复时比较 execution plan hash、策略语义、目标、预算、切分和快照；不匹配即拒绝。

TrialLedger 是追加式台账。提案抛错、返回非 Trial、重复 identity、语法非法、评估失败和预算耗尽都要记录各自状态。重复候选仍属于多重尝试边界。

恢复时还要区分“已预留未开始”和“已开始但结果未知”。前者可按租约释放或重放；后者必须按 durable attempt identity 对账，不能盲目重复执行或收费。

## 14. 常见误用

- 把它当 Optuna 包装器：错误，策略只是子系统，核心是证据、切分、预算和治理。
- 直接优化 test：错误，test 只供冻结赢家一次性评估。
- 把 screening_only 最高分当赢家：错误，筛选模式不能冻结可选赢家。
- 给 Bayesian 或 TPE 放 choice：当前拒绝；改用 Random、Grid 或先枚举分类分支。
- 缺稳健性或复杂度时填 0：错误，会偏置赢家。
- 直接比较低保真与全量分数：错误，stage profile 和证据强度不同。
- 新 recipe 复用旧 tier outcome：错误，job 与 outcome 绑定版本。
- 用高收益覆盖完整性失败：错误，integrity、split 和 capability 都是硬门。

## 15. 活动检查表

启动前：

- 确认是因子 treatment 搜索，不是组合权重优化；
- 冻结目标、方向、anchors、硬门和多重性族；
- 验证参数角色、边界、算子快照；
- 定义 train、validation、sealed test 及 purge、embargo；
- 预算覆盖提案、评估、成本和可选 LLM；
- 选择支持参数类型的策略；
- 设置持久化和恢复身份。

运行中：

- 每次评估先预留预算；
- 记录重复、异常和所有失败；
- 核对 stage profile 与实际计算；
- 只用 train 和 validation 晋级、选择；
- 不改已冻结 config、strategy spec 或 split semantics。

结束时：

- 固化完整 multiplicity artifact；
- 先执行硬门，再做 desirability 和 Pareto；
- 对接近候选使用不确定性或配对比较；
- 冻结唯一执行规格；
- 只消费一次 sealed test；
- 输出 lineage、成本、证据引用与不可用项；
- 交给 Factor Assets，不在本库宣称生产发布。

## 16. 项目协作边界

- Factor Engine：执行表达式、算子与合法性权威、canonical identity。
- Quant Evaluator：指标与统计证据。
- Factor Optimizer：treatment 提案、搜索编排、选择治理和 test 隔离。
- Factor Assets：最终准入与资产治理。
- Quant Platform：跨项目流水线编排。

遵守这一边界，Factor Optimizer 才不会被误用成计算器、回测器、组合求解器或发布器。

## 17. 实现证据与建议验证

本指南按当前实现和测试契约编写。修改活动配置或策略后，至少应运行与变更相关的测试：

- tests/search/test_strategy_validation.py 与 test_strategy_checkpoint_resume.py：策略类型、方向、状态恢复；
- tests/search/test_categorical_strategy.py：分类参数支持边界；
- tests/search/test_runner.py 与 test_budget_resume.py：runner、预算和恢复；
- tests/search/test_tiered_evaluation.py 与 test_multifidelity.py：漏斗和保真调度；
- tests/search/test_pareto.py、test_winner_selector.py、test_dlib_uncertainty_winner.py：Pareto 与赢家；
- tests/search/test_plateau.py：平台期；
- tests/search/test_split_safety.py、test_test_authority_boundary.py、test_sealed_test_hardening.py：切分与 sealed test；
- tests/test_diagnosis_repair_policy.py、test_repair_registry.py、tests/search/test_v5_diagnosis_routing.py：诊断和修复路由；
- tests/search/test_durable_campaign_store_v3.py 与相关 durable tests：持久化预留、租约和成本对账。

这些测试证明的是契约边界，不会替代真实 Factor Engine、Quant Evaluator 数据链和下游 Promotion Gate 验收。
