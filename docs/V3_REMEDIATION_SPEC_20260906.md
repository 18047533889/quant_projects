# 量化平台全量整改总任务书
## V3 · 两版合并、逐项扩写、开发 AI 单文件执行版

**交付对象**：负责真实代码库整改的开发 AI、集成协调者及独立验收者。  
**仓库**：`18047533889/quant_projects`。  
**历史审查基线**：`f9e7237fafe4112391e62c8874caae7996ce0b11`。  
**文档日期**：2026-09-06。  
**主范围**：`quant_evaluator`、`factor_optimizer`、`factor_preprocess`、`factor_assets`。  
**必要穿透范围**：`factor_engine`、`data_access`、`quant_platform`、旧平台入口、`modeling`、`lightgbm_qs`、相关 CLI/HTTP/jobs、存储与测试。不要借本任务无关重写整个 monorepo。

> 本文件是可以单独交给开发 AI 的工作规格：启动指令、全部旧问题、新增专项任务、跨库改法、测试场景、需求映射与完成标准都在同一份文件里。无需回看前两轮聊天，也无需另找 V1/V2 或外部 JSON 才能理解任务。

**本版合并口径**：保留 V2 原有 **97 项**工作单的编号、问题、影响、修改要求及验收；V1 的 **9 个 AUD 问题**逐项映射并保留全部子问题，不重复计算成新发现；为原 97 项逐一补充实施细节、追加验收和历史制品处理；再展开 **80 项 E 类专项核查／实现任务**。合计 **177 个有编号的工作单元**，不是 177 个已经在当前生产环境复现的 Bug。

**证据边界**：本版工作是完整读取并整合已提供的两版任务书、启动指令与 V2 账本后，补充可执行设计和验收要求；本轮没有重新读取当前 GitHub HEAD，没有重新运行真实仓库 pytest、行情回测或 CUDA，也没有修改远程代码。D/S/G/V 分类与 25 个隔离复现来自历史 V2 记录；新 E 项均需开发 AI 在实际 checkout 中核查，不得伪装成本轮新确认缺陷。当前代码若已改动，以真实代码与测试证据为准，已经修好的不能再次引入。

### 导航

1. [给开发 AI 的直接执行指令](#exec)
2. [证据、优先级与闭环状态](#status)
3. [统一架构与禁止重复建设的边界](#architecture)
4. [先冻结的合同与数据语义](#contracts)
5. [评价、评级与分用途硬门槛](#grading)
6. [诊断修复、顺序、中性化与模型表示](#repair)
7. [低预算搜索、去重、聚类和模型重训](#search)
8. [增量更新、资源、失败值清理和历史迁移](#runtime)
9. [实施波次与协作方式](#waves)
10. [全部 177 项快速索引](#all-index)
11. [V1/V2 合并后的 97 项详细工作单](#inherited)
12. [80 项新增专项核查与实施细目](#extension)
13. [独立金标、属性测试与端到端验收](#golden)
14. [V1 问题及原始需求的完整映射](#traceability)
15. [开发交付、来源与历史复现说明](#handoff)

---

<a id="exec"></a>
## 1. 给开发 AI 的直接执行指令

你接手的是**代码整改任务，不是再写一份建议书**。从当前真实 checkout 开始定位、建立失败测试、修改现有实现、接通调用链、生成验收证据。按本文件推进可独立完成的全部任务；遇到数据、GPU、权限缺失时记录精确阻断，继续其他不依赖部分，不因单项阻断放弃整个整改。

先读取本文件第 1–9 节，理解统一定义；随后按任务索引定位到对应详细工作单，按依赖波次实施。代码问题与验收问题同属交付范围。不要只做 P0 后宣布整体完成，也不要先实现大量新功能而保留错误数值和准入旁路。

### 1.1 启动动作

记录当前 `HEAD`、branch、工作区状态及现有未提交改动；保护用户已有工作，不执行破坏性 reset/clean。创建专用整改分支或隔离 worktree 的方式遵循仓库原规范。按各包 `pyproject.toml`、conftest、CI 配置检查实际安装方式，记录实际 import 的源码位置，不假设根目录 pytest 自动覆盖每个子包。

先盘点以下链路，形成事实覆盖表：

```text
实际 public API / CLI / HTTP / job / platform handler
  → adapter / request normalization / identity resolution
  → FE + DA 数据与因子值
  → FP recipe 编译与有效执行
  → QE 指标计划及 CPU/GPU
  → FO 搜索、预算、切分与证据
  → FA 分类、等级、准入、相似图与簇
  → feature set / modeling 消费
  → 冻结配方逐日更新
  → 失败值清理与发布/回滚
```

每条实际入口都要找到：谁计算、谁决定、谁持久化、谁验证。发现未列出的旁路、死配置、重复算子、隐式缺省、错误返回，继续追加工作单，不受 177 项上限限制；不得为了凑数把同一根因复制成多个已确认 Bug。

### 1.2 代码修改纪律

所有数值修复先有独立 expected contract，再写真实项目回归测试使其在旧实现失败；然后修改**已有权威实现**，再修所有实际调用者。不能只复制一份“正确新函数”而让旧入口继续运行错误版本。旧测试若固化错误口径，应一并修改定义、独立金标和实现版本，而不是盲目保留错误或直接删除测试。

禁止以下“完成方式”：把 `supported=False` 改成 True；删除守门；把 `UNKNOWN` 改 0；用默认高分补缺；吞掉全部异常转 fallback；测试期重调门槛；手工替适配器改参数后再宣称对拍通过；以 mock score、README 写完、声明了 Protocol 或必需测试 skip 来宣布功能完成。

V1/V2 是历史发现来源，本版给出统一执行语义。若某项历史断言在当前 HEAD 不成立，提交定位、调用链和反例测试，标为 `ALREADY_FIXED_VERIFIED` 或 `NOT_A_BUG_WITH_EVIDENCE`；不要机械为改而改。若本版示意接口与真实已有类型相近，扩展现有类型，不新建同义 authority。

### 1.3 权限边界与任务停止条件

本文件允许你设计和实现清理、发布、回滚与试验控制逻辑；**不等于授权删除用户真实生产资产、覆盖正在运行的模型、推送主分支或执行资金交易**。这些副作用依用户已有权限和批准流程处理。默认在隔离 fixture／测试快照中执行破坏性测试，生产只给 dry-run 影响计划，除非已有明确授权。

单项只有代码、对应测试、真实公开入口、版本迁移和适用环境证据齐备才可 `VERIFIED`。没有 GPU 就不能签 GPU 验收；没有行情数据也可完成合成数据的真实库链测试，但不能签实盘容量。最终必须列完整状态、剩余阻断及准确复验条件，不能以一句“全部修改完成”代替。

---

<a id="status"></a>
## 2. 证据、优先级与闭环状态

### 2.1 证据类别不随写作变成实测结论

| 类别 | 本文件意义 | 核查与关闭方式 |
|---|---|---|
| D | V2 已读源码中确认的问题分支或表达式 | 当前 HEAD 建真实回归，核实受影响入口后修复；不能假称全生产受影响 |
| S | V2 标识的定义／统计口径冲突 | 冻结规范问题，给不同变体不同 ID／版本，独立金标与下游迁移 |
| G | V2 已见边界缺口或局部风险，未穷尽上游保护 | 穿透 composition；已有不可绕过保护则证明，否则修入口 |
| V | 原用户需求的全链验收，不表示全仓没有功能 | 找到已有实现、接通、补缺、从公开路径验证 |
| E | 本版新增的专项核查与实现细目，部分是旧 V 任务的可执行拆分 | 先调查，再归类为需改、已覆盖、研究限制或阻断；不计成新确认 Bug |

原 97 项分类为 D 36、S 15、G 22、V 24；新 80 项全部 E。父任务 V 与子专项 E 可以由同一个 PR 和同一测试关闭，但必须保留映射，不重复夸大修改数。

### 2.2 风险排序

**P0**：影响金融数值真实性、PIT/收益时钟、证据身份、准入、发布或资产误删。先阻断受影响的新认证，定位是否已影响历史资产；不要无差别停止已有独立验证的无关功能。  
**P1**：功能闭环、统计稳健性、复现、模型兼容与治理。不能因为没有异常栈就降格为“仅文档问题”。  
**P2**：在既定正确语义下提高性能，不能以少算指标、改窗口或放松验证换取虚假加速。

建议任务状态：`TO_VERIFY → REPRODUCED → FIXED_LOCAL → INTEGRATION_VERIFIED → VERIFIED`。另有 `BLOCKED`、`ALREADY_FIXED_VERIFIED`、`NOT_A_BUG_WITH_EVIDENCE`、`SUPERSEDED_BY`。生产能力另记录 `RESEARCH_ONLY / CPU_CERTIFIED / GPU_CERTIFIED / RELEASE_READY` 等实际范围，任务状态不替代运行能力。

### 2.3 每项必须保存的证据字段

```yaml
issue_id: QE-01
origin: V2_D                 # 历史来源不是本轮实仓发现
current_head: null          # 实施时填写，禁止写假值
actual_entrypoints: []
root_cause: null
current_source_ranges: []
reproduction_command: null
expected_contract: null
patch_commits: []
regression_results: []
public_path_trace: null
independent_golden_ref: null
affected_artifacts: []
migration_or_no_migration_reason: null
rollback_plan: null
required_capability: []
remaining_blockers: []
status: TO_VERIFY
```

此处为**交付数据格式示意**，不是声称仓库已经有同名 schema。实施时映射到已有 ledger；只有不足时增字段，不另起一个竞争注册表。

---

<a id="architecture"></a>
## 3. 统一架构与禁止重复建设的边界

### 3.1 合并的是执行和语义，不是把四个包堆成巨型 Optimizer

| 现有模块 | 唯一业务职责 | 整改方向 |
|---|---|---|
| data_access / DA | 字段、历史可用时间、日历、证券身份、快照和规范持久化 | 数据真假和可得性在源边界证明，删除执行复用已有对象存储接口 |
| factor_engine / FE | DSL、规范 DAG、算子语义、因子与可复用变换计算 | 已有算子复用；新数学能力优先补 FE 或其明确数值后端，不让 FP 再写同义生产内核 |
| quant_evaluator / QE | 指标、统计与金融证据 | 全部相关性、IC、形状、风险和增量比较通过唯一指标定义；可执行收益消费已有执行域轨迹 |
| factor_preprocess / FP | 配方、适用性、拟合状态生命周期、模型表示和编译适配 | 不删除已有有用能力；收口数值真源、执行入口和有效参数，维护显式状态与后置性质 |
| factor_optimizer / FO | 诊断到候选、预算、搜索切分、trial/campaign、候选选择 | 不自己算 IC/Sharpe，不决定最终资产发布；能返回多个互补候选 |
| factor_assets / FA | 因子资产、分类、健康卡、准入、族/图/簇/库版本 | 评级只消费 QE 证据；聚类本身的图算法仍属于 FA，不必为“统一”错误搬到 QE/FE |
| modeling / lightgbm_qs | 固定特征版本训练/推理及模型预测制品 | 不另写预处理数学，不在验证/测试拟合状态，不从模型绩效倒灌所有因子高分 |
| Platform / jobs | 请求、任务、适配、权限、事件、存储协调 | 不另铸因子身份、不重算业务指标，不先于 FA 以单一 RankIC 过滤全部用途 |

可以使用不同 CPU/GPU kernel 实现同一算子，这是多后端，不是重复业务轮子。必须共享定义、有效参数、轴/mask/缺失语义和能力验证。研究参考实现可保留，但不能悄悄作为第二条生产路径。

### 3.2 有状态操作不要靠搬文件解决

允许 FP 管理 fit/apply 生命周期、训练边界、state ref 与 recipe；数学内核只能有明确 owner，FE 通过规范算子或受约束 state adapter 执行。对于尚未适配到 FE 的复杂拟合操作，记录临时唯一 owner 和研究限制，完成接口及 parity 后迁移；不要求一次性把所有逻辑移进 FE 后制造循环依赖。

**禁止** FE import FO/FA/modeling 来作质量决策；避免 FE↔FP 循环导入。跨包稳定合同应来自已有公开数据类型或无业务决策的薄基础类型，不复制第二组语义枚举。先用 adapter 兼容旧接口，再逐步迁移调用者，最后移除已无使用者的旧路径。

### 3.3 修复后的主链必须可追踪

```text
原始候选 → 规范身份/静态分析/字段来源 → 精确去重
 → FE + DA 落值 → QE 廉价证据 → FA 诊断与暂定健康卡
 → FO 诊断专属少量提案 → FP 配方编译 → FE 同一执行层
 → QE 共同比较窗口复评 → FA 用途门槛与增量/多样性约束
 → 冻结候选集合、方向、参数、fit-state、政策
 → 独立封存评价 → 合格库及研究簇版本
 → 冻结特征集合/模型验证/影子发布 → 正式发布指针
 → FE/DA 按固定配方增量更新 → QE 成熟新样本监测
 → 终态失败物化值按引用和保留政策清理
```

每日生产更新不再调用搜索器，不重新选择最优参数，不需要 Agent 临场解释后才能算值。

---

<a id="contracts"></a>
## 4. 先冻结的合同与数据语义

以下是要从现有类型中统一出来的**语义要求**。示意名称不代表当前仓库已经存在相应 class。允许合理沿用现有命名，禁止同义新增。

### 4.1 时间顺序、信息集和标签

至少分开 `observation_time`、原数据 `knowledge/available_time`、`signal_available_time`、真正交易 `decision_time`、`execution_time/window`、`label_start/end`、`mature_as_of`。真正的交易决策必须满足：

\[
\text{signal available} \le \text{decision} \le \text{execution}.
\]

旧 `decision_time` 若实际表示观察 bar 的时间，先调查全部消费者并迁移字段，不机械翻转比较号。不同字段可用时间可能不同，组合表达式可用时间应由依赖时钟推导，而不是只继承行情行日期。

**统一默认持有期定义建议**：以入场后完整价格区间数定义 H。在时点价格 \(P_e\) 入场，\(P_{e+H}\) 出场；若日收益定义为 \(r_t=P_t/P_{t-1}-1\)，新持仓从 \(r_{e+1}\) 开始计利，到 \(r_{e+H}\) 结束。用户的收盘后信号常对应下个可执行窗口入场，但具体 entry offset 必须由已声明执行规格决定。

因此 H=1 默认是一个入场后的收益区间，**不是同一 VWAP 价格当日进出**。同价同日无持有区间是独立退化测试场景：毛收益为零，有真实两笔成交则按费用规则扣费，没有成交则不扣。V2 对旧 H=1 分支的反例不能被解释为要求保留混乱定义。

`ForwardLabelBundle` 是用于预测检验的远期结果；`HoldingReturnPanel` 或执行价格／成交轨迹用于日度持仓核算。两者即使 shape 相同也不能互换。H 日重叠标签不直接作为每日 NAV；先按正确资本与持有区间构建组合。

### 4.2 股票轴、布局和缺失

面板必须带命名轴、规范证券 ID、时间/日历、layout、dtype、validity 和必要 missing reason。TNF 与 TFN 显式转换；数量相同不证明坐标相同。严格模式不自动排序/inner join 以掩盖不匹配；允许的显式对齐记录置换和缺失。

原始缺失、填充值、陈旧值、不可交易、尚未成熟和预算没算分别保留。诊断 RankIC 可以按共同有效标签事后统计；历史持仓目标不能由未来标签缺失决定。是否可生产由整个信息集与执行时钟判断，不是简单把所有算子 shift(1)。

### 4.3 身份与不可变对象

| 身份 | 表示什么 | 不得用什么替代 |
|---|---|---|
| Definition | 规范公式/字段/算子语义 | display name、semantic family hint |
| Recipe | 有序 DAG、有效参数、通道、状态绑定规则 | 一段“已中性化”的说明 |
| Fitted state | 训练数据范围、方法、参数、版本和状态内容 | 当前进程变量或没落盘对象 |
| Value/materialization | 定义＋配方＋数据/股票池/时钟＋有效执行实现 | 原公式 hash 或任意 UUID |
| Evaluation | 完整请求＋value/label/执行轨迹＋metric snapshot | 一个裸 IC 数字 |
| Policy/Health | 用途、阈值、证据和上下文 | 不可追溯的“永久A级” |
| Graph/Cluster/Feature/Model | 实际成员、聚合函数、版本及消费关系 | 列名不变、簇号相同 |

字节 hash 与语义 hash 分开；内容地址必须针对实际写入／执行内容。`frozen=True` 不会自动让内部 ndarray、dict 或闭包不可变，应深冻结或具备明确只读租约；不能靠重复全数组 hash 代替所有权管理而造成计算浪费。

### 4.4 统一请求、计划与证据返回

规范化后的评价请求需要绑定：因子值及标签/执行制品引用、样本范围、股票池、时钟、horizon、orientation、metric 参数、Q、费用/组合规格、split/capability、budget、backend policy 和所有版本。请求校验、身份解析与计划编译位于 backend dispatch **之前**。

规划顺序：解析同义 metric → 检验适用性/必需输入 → 建真实 artifact DAG → 验证内存与资源预算 → 选择支持的后端 → 执行共享中间对象 → 产生逐因子 typed artifacts → 下游只翻译／读取。artifact 类名、空 placeholder、UUID 或“看起来像引用”的字符串都不算已产生证据。

公开返回不因 CPU/GPU 改变含义：scalar、T×F series、Q×F profile、T×Q×F 日分层、pairwise matrix 必须分别保留。大对象可引用落盘，小对象可内嵌；统一定义必须保留 metric ID/版本、轴、有效样本单位、状态、warnings、provenance。

可选指标缺失可返回明确状态并继续独立指标；required 指标缺失阻断该用途，不应让一个可选能力缺失导致整批所有独立结果丢掉。输入身份/时间非法属于请求无效，不是可忽略 metric warning。

### 4.5 组合、净值与风险的唯一依据

快速 probe 与真实可执行 portfolio 分类型。研究多空可以诊断，但无借券、侧别成交、冲击和成本证据时不可叫实盘可执行。复用已有回测／执行域，QE 负责对其产出的轨迹计算指标，不再新造无约束回测轮子。

PortfolioSpec 必须解释资本与杠杆：例如多头 +0.5、空头 −0.5 为 gross=1；多头 +1、空头 −1 为 gross=2，不能把两种 returns 混作同一成本/风险等级。固定权重 probe、买入持有和每日再平衡各有不同成交含义。实际换手基于调仓前价格漂移后的仓位与成交变化，而不是原始rank变化代理。

NAV 从 t0 初始资本开始；回撤事件从统一 NAV 生成，保存 peak/trough/recovery/censored 时间与计数。缺报价不随意压缩日期，破产不当 NaN 从风险中消除。完成事件平均恢复时间与当前未恢复年龄是不同字段。

---

<a id="grading"></a>
## 5. 评价、评级与分用途硬门槛

### 5.1 自动分类：数据源集合、角色和经济机制分离

数据域采用可分级集合，不为每种混合建立互斥 enum：`PRICE`、`VOLUME`、`LIQUIDITY.TURNOVER`、`FUNDAMENTAL.QUALITY/GROWTH/CASHFLOW/LEVERAGE/INVESTMENT`、`VALUATION`、`EVENT`、`MICROSTRUCTURE`、`FLOW_SENTIMENT`、`RISK`、`ALTERNATIVE` 等。展示名称由集合派生。

要覆盖用户的四个金标：量价字段 only；基本面 only；量价＋财务；量价＋换手＋财务。未知字段显式保留 `UNKNOWN`。派生字段由目录与血缘解析，不重写 DSL parser；同名不同表字段不能靠裸字段名猜。

另保存角色：`alpha_source_fields`、`control_fields`、`eligibility_fields`、`weights_fields`。量价因子用市值做风险中性化，应记录控制依赖与处理，不把经济 alpha 来源改称财务。机制标签如动量、反转、价值、质量、交互带规则来源和置信程度；“用了某个算子”不自动证明经济机制。Agent 提案、人工确认、确定性目录标签、统计健康标签分别标来源。

### 5.2 14 维健康卡和两个跨维门槛

沿用既有 14 维结构，不另开一个竞争评分系统。指标的存在、可计算性与用途要求必须逐一绑定。

| 维度 | 需要覆盖的子指标/证据 | 主要错法与具体改法 |
|---|---|---|
| predictive_power | 有向 RankIC/PearsonIC、未年化 ICIR、有效日数、HAC/块区间、多期限曲面、非线性证据 | 方向只由训练确定；不能对测试绝对值再挑正负；股票格数不是独立IC日数 |
| stability | 月/季/年、rolling 分布、正向窗口比例、最差段、近期退化 | 保存真实序列，calendar与固定21/63/252期另名 |
| robustness | train-valid差值、适用retention、参数平台、子样本、负控、选择稳定性 | 近零分母/变号另状态；相同raw均值不是稳健证据 |
| turnover | rank proxy、分位成员、目标权重、实际成交、换手集中度 | 四种分别命名；不能拿proxy扣成本 |
| capacity | 目标资本、ADV/成交占比、可成交/借券、冲击敏感度 | 无数据则无容量证据，不以低换手自动算容量 |
| cost_drag | gross/net、费用分项、成本压力、盈亏平衡成本 | 固定成交轨迹下成本增高净收益不得提高；市场规则按生效日期 |
| drawdown | 最大回撤、最长/平均水下、峰谷恢复、当前删失 | 初始NAV和破产状态必须进入，缺日期不消失 |
| tail_risk | ES/CVaR、下行偏差、偏度峰度、最差自然月/滚动期 | 风险符号、分位插值、分母与样本充分性明确 |
| data_coverage | observed/joint/tradable/filled覆盖、distinct、ties、有效截面、universe churn | 预声明分母；不能靠填充与事后删股制造高覆盖 |
| freshness | 公告可用时间、观测年龄、陈旧阈值、更新频率、标签成熟 | 每日相同值不必然是新观测，事件年龄另保存 |
| complexity | DAG节点/深度/回看/字段依赖、时延、算存与恢复成本 | 真实 telemetry，不是“已用GPU”布尔值 |
| economic_sense | 机制假设、支持和反对证据、标签来源 | 不把LLM主观分当盈利统计硬门 |
| shape_quality | 10/20层、类型、U/倒U、尾部断崖、镜像不对称、跨窗稳定 | 类型先行；单调与U不互相扣分；模板分不是成功概率 |
| regime_sensitivity | 行业/size/beta/流动性等信号与组合暴露，状态切片表现 | 事前可知状态与事后压力解释分开 |

跨维门槛一：研究完整性，包括 PIT、标签成熟、快照/轴/数值/单位/收益口径和封存纪律。  
跨维门槛二：相对已存在因子库／模型／组合的增量价值，包括替代、互补和净效用；不能被展示总分遮蔽。

### 5.3 每个指标保存原值、等级和证据，不只字母

建议结构：`value + units + definition/version + horizon + oriented_sign + sample_counts + effect_grade + evidence_status + confidence_interval + evidence_sufficiency + policy_ref + context_ref`。没有证据不产生S/B/D；性能好但证据薄可以有点估计，但不获正式准入。

保留 `S+ → S → A+ → A → B+ → B → C → D`。以下是前版记录的**旧 CN 日频 H10 政策锚点**，用于兼容审查，**不是本版自动启用的行业门槛**：

| 等级 | RankIC 下限 | 旧 ICIR 表下限 | 验证保留率下限 |
|---|---:|---:|---:|
| S+ | 0.050 | 1.50 | 0.90 |
| S | 0.040 | 1.10 | 0.80 |
| A+ | 0.030 | 0.85 | 0.70 |
| A | 0.022 | 0.65 | 0.60 |
| B+ | 0.017 | 0.45 | 0.50 |
| B | 0.012 | 0.30 | 0.40 |
| C | 0.005 | 0.15 | 0.25 |
| D | 低于上一项 | 低于上一项 | 低于上一项 |

改法：先核对 ICIR 是否年化、频率和重叠标签处理，单位不明不启用；使用历史 development walk-forward 校准绝对锚点与最低线。cohort percentile另字段，新增大量差因子不能抬高旧因子的绝对等级。锚点与政策深冻结，任何变动新 hash/version。

用户历史中的 `Rank SE / Rank SIR` 不应凭名称自动映射；建立 terminology 表，显示原始名称、规范 metric ID、公式、单位及 unresolved 状态。核对现有库和实际需求后一次解决，不添加含义模糊的新同义指标。

### 5.4 缺失与质量差分开，所有分数不抵消硬失败

原 `.4×min + .6×geomean` 可作为版本化聚合形式，但必须先确定适用指标和required集合。必需证据缺失由证据门阻断；不适用项不入分母；预算未跑、实现不支持、样本不足、原始计算失败、非法证据分别保留。已测为零与没算不是同一件事。

选用哪些指标进入维度也必须避免同源重复，例如 RankIC、其alias和rolling均值并非三个独立预测优点。`economic_sense`不能因为没有LLM主观评分而让所有因子数学分数归零；应明确证据条件与适用性。

健康卡只能由验证过子证据的权威builder产生；输入的factor/value/evaluation/split/policy必须相容。序列化读入时重新核对summary与实际gate，拒绝true summary嫁接false gates。required gates必须集合完整且无重复，禁止 `all([])`。

### 5.5 入库分用途，门槛以政策明确实施

| 目标池 | 必须通过 | 不能机械要求 |
|---|---|---|
| 研究档案 | 身份/来源可追溯，说明状态与失败原因 | 不要求所有因子有Alpha；保留公式不等于可以生产 |
| 独立信号候选 | 完整性、样本外预测或形状映射证据、成本/风险符合用途、非冗余 | 原始曲线必须天然单调 |
| 模型特征候选 | 数据合法、固定表示、有限预算OOS增量/稳定非线性证据 | 独立LS Sharpe必须很高或所有单变量IC过同一线 |
| 实盘发布 | 冻结全部输入/配方、可信独立验证、可执行性、成本与容量、更新重放、发布回滚 | 研究卡总分高就直接入实盘 |

旧建议“信号预测 B+、稳定 B、数据质量 A、成本/换手 B；预声明覆盖95%；方向一致窗口70%；每母因子1–3个版本”等全部保留为**待校准起点**，不得把它们当普适私募标准。覆盖分母、最低有效时长、窗口数量及证据可信度同样要写政策；三个强相关窗口不等于三个独立证明。

评价净效用时先声明风险/成本约束，再选择少数主目标。不得同时在很多grade表、horizon、Q、方向和模型上挑test最容易过的路径。

---

<a id="repair"></a>
## 6. 诊断修复、顺序、中性化与模型表示

### 6.1 修复意图先有证据，再扩少量候选

`RepairIntent`语义至少包含：原factor/value/evaluation refs、诊断类型及置信、最主要问题、允许family、参数域、preserved properties、预算、复评profile与停止条件。FO生成；FP校验与编译；FE执行；QE复评；FA按用途决策。不要把维修写成一个直接读测试收益、改值并给分的黑盒函数。

数据错、未来信息、证据身份不符、错误单位，先修完整性并停止效用优化。对不可修复的无信息或超复杂因子允许 `ABANDON`，不必为了让每个因子变好而耗尽预算。

### 6.2 U／倒U与拆分

U型是形状，不天然是坏因子。令 \(u=CSRank(x)\)，固定中心baseline可用 \(|u-0.5|\) 或 \((u-0.5)^2\)，倒U反向；仅rank或zscore的单调操作不会改变真实U关系为单调alpha。

偏斜形状允许少量训练内中心 \(c\) 与左右 \(\max(c-u,0)\)、\(\max(u-c,0)\) 子通道。用收益拟合center/knots属于监督拟合，必须有train-only state、搜索计数和独立验证。原始、左支、右支、重组都重新落值复评，单个子因子A级不能继承成组合A级。全部保留父子血缘及同campaign尝试统计。

为树/神经模型保留原值通道是合法方案，但需要实际增量证据，不以“以后模型也许会用”无限收垃圾特征。形状稳定性与模板拟合分不是盈利概率；对称U的纯左右不对称度应为零。

### 6.3 10层与20层、尾部衰减

默认廉价阶段10层，疑似形状或入围因子20层；系统具备两种真实能力，不只增加一个配置项。Q按实际distinct、每桶有效人数、ties和分布决定。旧preferred20、fallback10/5、每层100只仅为原政策示例，不对稀疏/行业小截面硬套。离散值无法形成真实20层时返回实际层数或不适用。

保留逐日T×Q×F收益、人数、有效mask和长期profile；正向因子尾部可比较顶层与前面几层均值，例如 \(R_{20}-mean(R_{17:19})\)。一张平均图不足以下结论，先查尾部样本、风险、极值和交易限制，再比较饱和／有限hinge等少量方案。禁止按test收益删除最后一层股票或无限搜cutoff。

### 6.4 中性化的效果、性质与非劣性

提供raw、industry、industry+size等少量预声明候选；不假定加中性化总能提高Sharpe。OLS/WLS、ridge、Huber有不同数学性质，只有对应条件下的精确投影才能承诺 \(X^TWr\approx0\)。暴露数据必须当时可见，输入列由schema明确，不读取所有非key列。

用户“效果差不多就优先中性”的想法落实为成对样本外非劣性：预声明效用指标、容忍区间和最小风险改善，比较 \(\Delta=utility_{neutral}-utility_{raw}\) 的不确定性，而非临时算95%保留率。raw近零时比值不适用。strict-risk用途风险违规不可被高预测分覆盖；model-feature用途可保留raw与neutral双通道，但单独固定引用。

特征中性≠持仓中性。neutralize后再rank、clip、跨期平滑可能改变性质；最后重新验证暴露或选满足目标的顺序。组合风险仍需由实际权重/成交轨迹评价。

### 6.5 因果滤波与缺失通道

首轮只用不平滑和少量EWMA等已有因果算子，必要时再启用单侧Kalman/IIR。EWMA递推 \(s_t=(1-\alpha)s_{t-1}+\alpha x_t^{available}\)，半衰参数 \(\alpha=1-\exp(-\ln2/h)\)。定义是按交易时点、真实观测还是事件推进；重复ffill财报不一定是新事件。

生产信号禁止使用双向filtfilt、centered窗口、后向填充、全样本HP/小波或Kalman smoother冒充因果输出。离线研究可保留但显式不可生产。未来标签生成允许明确的未来shift，不得被全仓“禁止shift(-1)”误删；它应与factor执行域隔离。

验证前缀不变：在同一信息集语义下，修改未来输入不改变此前输出；并验证full/chunk/daily/checkpoint恢复一致。辅助missing/freshness channel是分支或注释，不能替换主信号继续流经其他变换。

### 6.6 操作顺序由实际DAG和后置性质认证

默认骨架：`PIT/合法样本 → 有界缺失处理 → 异常值 → 可选形状 → 因果平滑 → 可选中性化 → 模型表示`。不是允许任意全排列，也不是只有这一个序列永远合法。

允许例外必须匹配真实canonical步骤、参数、前置条件和最终性质。RAW只表示零变换；贴RAW标签不能让任何乱序图获得认证。history记录“曾经出现过”，OutputProperties表示“当前还满足”；两者分离。

对已经满足的最终zscore可避免重复，但 `zscore → nonlinear → zscore` 不盲删；不同半衰EMA不去重；内嵌子表达式rank不能把整因子标成ranked。所有等价重写记录law ID及适用前提，不能以粗 `(semantic_id, stage)` 全链删除。

### 6.7 树、神经、线性的输入表示

树模型不为了“排量纲”强制套zscore；验证具体算法的缺失支持、类别与幅度信息需求。跨所有样本同一个单调变换与每个日期独立截面rank不是同一信息操作。神经/线性通常需要更明确的缩放和mask契约，但应按具体consumer profile而非个人偏好硬套。

跨时间拟合scale、winsor阈值、PCA、knots或组合权重只使用训练边界；每个fold重新建立可得state。截面当日变换可用当时已知股票池，不因此盲加一日滞后。模型推理永远读取冻结feature order与state，不能对新测试批次重新fit。

---

<a id="search"></a>
## 7. 低预算搜索、去重、聚类和模型重训

### 7.1 预算分阶段，不让所有候选算全部指标

| 阶段 | 工作 | 真实产物与晋级要求 |
|---|---|---|
| S0 | 静态DSL、字段、单位、时钟、精确身份去重 | 合法定义与来源；不落值即可拒绝明确无效 |
| S1 | 批量coverage、RankIC、10层、变化代理与基础风险 | 廉价证据与诊断；给U/事件/互补留有界支路，不因低单变量IC全部删 |
| S2 | 2–4个最相关修复优先，少量参数平台探索 | 有效recipe/参数到FE；仅相关families激活；8–12新增候选为默认上限起点 |
| S3 | raw与少量入围候选共同完整验证 | 20层、日PnL/风险/成本、稳定性、增量与统计；低档分数不混作正式证据 |
| S4 | 硬门→少数主目标Pareto→等价简单优先→多样性 | 冻结候选集合及全部规格，默认每母因子0–3个有用途证据版本 |
| S5 | 一次逻辑sealed评价、发布资格与后续模型验证 | 不将test结果回流调参；不是挑通过的子集后再假称预先选择 |

与仓库legacy L0–L4/Stage0–5统一时，从一个执行计划派生对应视图，不能保留两张互不驱动的表。每stage存实际profile hash、样本、metric、预算、effective backend和晋级理由。

所有提案、无效公式、有效不同参数、Agent建议、失败执行、缓存命中和技术重试都记录，但分别统计proposal数、unique spec数、execution数、实际资源及统计假设族。缓存命中不等于从未尝试；技术网络重试不必伪装新科学假设。跨session与alias不能重置campaign或test访问。

### 7.2 筛选不是单冠军，也不是全收

先必须满足用途资格，再Pareto；目标不宜包含几十个重复衍生统计，否则几乎所有候选都非支配。unknown核心目标不可用±inf凑齐。设预声明等价区，等价优先简单、低成本、低状态维护负担；参数平台而非尖峰更值得保留。

可以保留高预测、低换手、风格更纯等少数不同优势版本，但每个都有独立有效证据和增量用途。全不合格返回空集是正确结果。对子因子、多个horizon、簇聚合和模型add/drop的选择也计试验族。

### 7.3 聚类前去重的四层流程

精确canonical定义／recipe重复可复用；值等价必须同数据、mask和语义；极高相关如|ρ|≥.99/.995是近重复复核起点而不是等价证明；|ρ|约.95可同族，.7不能天然当“重复删除线”。所有阈值版本化并历史校准。

同时看值/残差值相关、净PnL相关和IC序列相关；逐日截面相关与展平T×N另名。负相关记录方向；无共同样本、低相关、未测、近似和已认证分别表达。ANN负责召回，QE提供正式pairwise证据；ANN未召回不得声明已证明不重复。

### 7.4 簇动态更新与模型输入双轨

研究可增量分配到明确簇、AMBIGUOUS、PENDING或PROVISIONAL_NEW；不为避免小簇改投最大簇。最终affinity必须对应最终目标，多个代表/支持邻居比单一近邻更可靠。全量刷新由固定节奏和漂移触发，月度局部/季度全局只能作为运维起点，不是必然最优频率。

Leiden连通性不等于全簇高相关，额外检验簇内质量。GraphArtifact不能直接当协方差矩阵；协方差需单独估计和适用校验。索引build随版本变化，不在新因子×簇内重复重建。

线上模型固定 `FeatureSetVersion`。仅说明/分类更新通常不重训；代表因子、成员聚合、权重、PCA、recipe/state或列集合变动，即使列数和名字不变也改变输入函数，必须兼容检查与模型再验证，按政策重训/影子发布。研究增量加入簇不能悄悄改线上均值列。

---

<a id="runtime"></a>
## 8. 增量更新、资源、失败值清理和历史迁移

### 8.1 真实批量复用

FE复用字段加载与共同DAG；QE共享mask-aware rank、IC、分层、PnL、NAV与episodes。不同mask的因子不能无条件共用label rank。输入身份一次规范化计算，避免每metric遍历全张量hash。按F分批优先；截面rank不能简单切碎N轴，T分块须有数学归并或状态合同。

显存估算包含输入、rank/index、mask、标签、回归、排序工作区、PnL和暂存；例如2520×5000×128×4字节约6.45GB只是输入维度算术，绝非完整显存需求或实测。设备strict模式不得静默fallback；auto降级记录实际实现。OOM可缩F批，不得暗改Q/H/必需metric换速度。

基准拆分冷/热缓存、I/O、编译、转换、传输、kernel、同步、序列化、落盘和hash，总计可对账。没有实际CUDA设备，只能记录模拟协议测试，不能签硬件性能或parity。

### 8.2 固定自动更新线

固定ExecutionSpec＋DA可用watermark触发，FE读取分instrument/recipe隔离state，输出新generation；写完整manifest校验后指针最后切换。重启、同日重复job、晚到修订、资产新增/退出都有确定状态。有限lookback精确推导重算范围，无限记忆EMA从可信checkpoint重放或采用明确误差政策，不偷偷截断。

更新过程不重新search、不重新fit跨时间模型表示、不读取未声明实时源。发布模型与特征需要匹配版本。故障保持旧完整版本和风险状态，不半发布。

### 8.3 失败值删除的完整协议

用户要求“失败尝试的values最终删除，公式/操作保留”，实现如下：

```text
研究任务进入有明确原因的终态
 → 检查保留策略/诊断TTL与生产、回滚、读任务、待重试root引用
 → 遍历共享DAG/子因子/模型依赖，形成live集合
 → 生成逐对象版本GC dry-run与预计回收
 → tombstone/epoch屏障，防止扫描后新增引用竞态
 → 既有权限批准后由DA删除物化对象/暂存/cache
 → 读回/对象版本核验、receipt与provider保留状态
 → 完整保留formula/recipe/state定义/metric证据/trial ledger/失败原因
```

不是删数据库行就完成；不是按目录宽泛rm；对象存储版本/保留锁导致尚未物理回收须如实记。删除values不删除trial，也不取消尝试次数。只因数据尚未成熟/预算不足/实现bug而未成功的候选不等于永久无用，待重试资格须保留。

### 8.4 修对代码以后，旧结果也要正确处理

每个修复给 `affected_artifacts`：评价算法变更通常可复用未变因子值，重算指标/健康卡/准入；参数桥变更可能需要重算value及所有后代；仅政策阈值调整通常不必重落所有因子值，但要新policy下重新评级。沿引用传播到图/簇/feature/model。

旧对象保持只读，标 INVALID/SUPERSEDED/REQUIRES_REEVALUATION，生成新证据，不原地改旧A级变成新A级。已经看过的test可故障重放但不是新鲜独立holdout。生产切换与删除依原授权；迁移有dry-run、幂等执行和回滚。

---

<a id="waves"></a>
## 9. 实施波次与协作方式

| 波次 | 先交付什么 | 工作单主题 | 验收后才能做什么 |
|---|---|---|---|
| W0 | actual HEAD、入口/文件/能力盘点、旧缺陷金标、影响边界 | V-01、QA/QAT、全部P0定位 | 不假设文档已代表当前代码 |
| W1 | 时间、PIT、轴、缺失、typed身份、参数绑定 | QE-29/30/31、FP-01/02/03、PL-05、DTA | 对齐合法后才比较金融指标 |
| W2 | 正确持有区间、资本/费用、NAV/回撤/episodes | QE-09/11–26相关、DTA-03/05、QAT-02 | GPU必须对独立正确金标，不只是复制CPU |
| W3 | QE真实DAG、逐因子制品、统一CPU/GPU请求与输出 | QE-01–08/27/28/32–35、STA | 所有required metric从public入口可用 |
| W4 | FP执行收口、DAG性质、配方/状态/顺序、FE绑定 | FP-04–12、RCP | 维修提案确实变成可重现数值 |
| W5 | 标签、14维评级、完整证据、用途准入 | FA-01–07/11、V-02/03/05/06/08、STA | 不再有旧gate旁路和伪高分 |
| W6 | 少候选搜索、多保真、预算、恢复、封存 | FO、V-09–12/16/22、SRH | 真实非mock跨库搜索，允许多互补赢家 |
| W7 | 分层去重、ANN、动态簇、聚合与模型输入 | FA-08–10、V-14/15/17、SIM、MOD | 研究版本与线上输入隔离，增量价值成立 |
| W8 | 重试/事务/发布、更新、GC及旧结果迁移 | PL、V-18–21/24、OPS | 幂等、引用保护、影子/回滚与清理回执 |
| W9 | 全链回归、真实数据/硬件、能力认证与交付 | V-23/24、QAT-01–08、FO-03 | 仅有真实证据的范围获认证，剩余明确阻断 |

可并行分工，但每个共享合同／registry／policy snapshot由一个协调者整合。建议按数据执行、QE数值、FP配方、FO搜索、FA资产、平台模型、独立QA分工；不是要求必须有多个Agent工具。单一开发AI也按这些工作包顺序执行。

每个PR大小应便于独立测试和回滚，先接口兼容再迁移消费者再删旧代码；一个PR可关闭多个同根因工作单，不能为编号数量制造177个新模块。独立验收者检查“是否真经过public path”和“旧错误是否仍可从旁路触达”。

### 9.1 每波必须给协调者的结果

提供已改函数与实际调用者、对应问题编号、真实测试与日志、schema/数值版本变化、有效能力矩阵、历史制品影响、性能变化及未解决阻断。暂时无数据/GPU的任务保留可执行复验命令；不得写成“理论上通过”。

### 9.2 同一改动不要无限复造证据

修复后需要独立golden、公开路径和端到端三层证明，但相同可信artifact可被多个指标/任务引用，不必每项重跑全部大数据。根据受影响依赖选择最小充分回归集，末波统一全链验证。预算有限优先确定性合成反例；昂贵多horizon、bootstrap、模型增量只给已通过前层的候选。

---

<a id="all-index"></a>
## 10. 全部177项快速索引

编号总数是整改/核查工作单元数，不是新增Bug数。原97项的详细正文全部保留在第11节；新80项全部为E类，其中一些细化原V任务，可共享同一实现和验收，不重复计算修复功劳。标题可点击直达本文件工作单。

| 组别 | 范围 | 数量 | 证据性质 |
|---|---|---:|---|
| QE | 评价、统计、CPU/GPU、组合与时间 | 35 | 历史D/S/G/V |
| FP | 配方、参数、算子复用与最终性质 | 12 | 历史D/S/G/V |
| FA | 健康卡、评级、准入与聚类资产 | 11 | 历史D/S/G/V |
| FO | 搜索、证据适配、切分与状态 | 5 | 历史D/S/G/V |
| PL | 平台任务、发布、重试和模型快照 | 8 | 历史D/S/G/V |
| QA | 历史测试路径与真实能力矩阵 | 2 | 历史D/S/G/V |
| V | 原始需求端到端验收 | 24 | 历史D/S/G/V |
| DTA | 数据、PIT、日历与字段合同专项 | 12 | E：待实仓核查 |
| STA | 统计、选择偏差与评级专项 | 12 | E：待实仓核查 |
| RCP | 配方、形状、因果变换与表示专项 | 10 | E：待实仓核查 |
| SRH | 小预算搜索、候选集与封存专项 | 10 | E：待实仓核查 |
| SIM | 相似图、增量归簇与版本专项 | 10 | E：待实仓核查 |
| MOD | 模型输入、OOS增量与更新专项 | 8 | E：待实仓核查 |
| OPS | 运行、资源、发布、清理与重放专项 | 10 | E：待实仓核查 |
| QAT | 独立金标、属性、设备与关闭证据专项 | 8 | E：待实仓核查 |

### 10.1 原97项索引

| ID | 优先级 | 类别 | 责任边界 | 问题与任务 |
|---|---|---|---|---|
| [QE-01](#issue-qe-01) | P0 | D | QE | 组合绩效被错误归因：不同因子收到同一个指标 |
| [QE-02](#issue-qe-02) | P0 | D | QE | 公共接口把序列和分层向量静默压成均值 |
| [QE-03](#issue-qe-03) | P0 | D | QE | 形式上存在的依赖被当成已经构造的证据 |
| [QE-04](#issue-qe-04) | P0 | D | QE | GPU 分支绕过统一校验并遗漏 validity |
| [QE-05](#issue-qe-05) | P1 | D | QE | CPU/GPU 输出合同、metric ID 与 artifact kind 不一致 |
| [QE-06](#issue-qe-06) | P0 | S | QE | GPU 的ICIR、样本门槛和分层数脱离指标政策 |
| [QE-07](#issue-qe-07) | P1 | S | QE | GPU 把两类不同换手指标计算成同一数值 |
| [QE-08](#issue-qe-08) | P1 | S | QE | GPU 分层单调性与CPU回答的不是同一问题 |
| [QE-09](#issue-qe-09) | P0 | D | QE | GPU cohort 把远期标签当成日度持仓收益 |
| [QE-10](#issue-qe-10) | P2 | G | QE | GPU 为每个组合指标重复计算同一组合PnL |
| [QE-11](#issue-qe-11) | P0 | D | QE | 回撤计算漏掉初始净值1，第一笔亏损消失 |
| [QE-12](#issue-qe-12) | P0 | D | QE | 破产状态被NaN屏蔽，最大回撤可能反而为0 |
| [QE-13](#issue-qe-13) | P1 | D | QE | 空Sharpe返回三元回撤结构；空回撤缺少正确保护 |
| [QE-14](#issue-qe-14) | P1 | D | QE | 水下期计算删除缺失日期，压缩了真实时间轴 |
| [QE-15](#issue-qe-15) | P1 | D | QE | 恢复时间扫描跨越多个回撤事件，并混同未恢复样本 |
| [QE-16](#issue-qe-16) | P1 | S | QE | 偏度注释宣称bias=False，实际计算未做有限样本校正 |
| [QE-17](#issue-qe-17) | P1 | S | QE | Sortino与Calmar使用了未充分区分的统计变体 |
| [QE-18](#issue-qe-18) | P1 | S | QE | 最差月／季实际是滚动21／63期，而不是自然月／季 |
| [QE-19](#issue-qe-19) | P1 | D | QE | 3D因子调用时不支持文档承诺的2D公共validity mask |
| [QE-20](#issue-qe-20) | P1 | D | QE | 通用多空分桶可在平局因子上同时买卖同一组 |
| [QE-21](#issue-qe-21) | P0 | S | QE | drop缺失收益在形成桶之前排除未来不可得标签 |
| [QE-22](#issue-qe-22) | P1 | D | QE | 成本函数未拒绝负换手，能产生“交易返利” |
| [QE-23](#issue-qe-23) | P0 | D | QE | cohort在下一日VWAP入场，却赚到了入场前的涨跌 |
| [QE-24](#issue-qe-24) | P1 | S | QE | 同日进出只扣单边成本，末尾未成熟cohort被强平 |
| [QE-25](#issue-qe-25) | P0 | S | QE | probe组合与“可执行实盘PnL”的声明不匹配 |
| [QE-26](#issue-qe-26) | P1 | G | QE | 日度风险链存在多个不相容的缺失/零填充默认 |
| [QE-27](#issue-qe-27) | P1 | D | QE | 指标版本与观测数量在公共证据里被错误简化 |
| [QE-28](#issue-qe-28) | P1 | G | QE | 评估DAG未充分复用，批处理和预算在公共入口失效 |
| [QE-29](#issue-qe-29) | P0 | G | QE | 低层评估与标签合同缺少完整坐标绑定 |
| [QE-30](#issue-qe-30) | P0 | S | QE | decision_time与signal_available_time的含义和不等式需要纠正 |
| [QE-31](#issue-qe-31) | P1 | D | QE | 标签快照可被外部修改，chunk与cache又遗漏关键字段 |
| [QE-32](#issue-qe-32) | P1 | S | QE | 预测期限衰减指标实际测的是IC序列自相关 |
| [QE-33](#issue-qe-33) | P1 | D | QE | 形状稳定性输出Fisher-z而非声明的相关性尺度 |
| [QE-34](#issue-qe-34) | P1 | S | QE | 形状confidence与asymmetry指标名超出实际计算含义 |
| [QE-35](#issue-qe-35) | P1 | G | QE | 自适应分层合同存在边界洞，人数不能代替实际桶可用性 |
| [FP-01](#issue-fp-01) | P0 | D | FP/FE | 整数证券代码经FE适配后变成全空列 |
| [FP-02](#issue-fp-02) | P0 | D | FP/FE | 重复日期证券记录被first静默吞掉 |
| [FP-03](#issue-fp-03) | P0 | D | FP/FE | FP→FE参数映射遗漏，实际变换不等于记录配方 |
| [FP-04](#issue-fp-04) | P0 | G | FP/FE | 存在绕过FE的公开执行口，fallback吞掉所有加载错误 |
| [FP-05](#issue-fp-05) | P2 | G | FP/FE | FE桥仍是固定CPU长宽表转换，不是GPU批量执行链 |
| [FP-06](#issue-fp-06) | P1 | G | FP/FE | 暴露矩阵输入靠“所有非键列”猜测，缺少显式schema与轴 |
| [FP-07](#issue-fp-07) | P0 | D | FP/FE | 变换去重只看semantic_id与stage，可能删除必要操作 |
| [FP-08](#issue-fp-08) | P1 | G | FP/FE | 已有处理签名只是出现记录，不是最终输出性质 |
| [FP-09](#issue-fp-09) | P0 | D | FP | 合法顺序认证可被一个RAW标签绕过 |
| [FP-10](#issue-fp-10) | P1 | G | FP | 变换注册默认可生产，参数域和生产状态只做局部检查 |
| [FP-11](#issue-fp-11) | P1 | D | FP | 声明不可变的配方及语义标识仍可被修改 |
| [FP-12](#issue-fp-12) | P1 | D | FP | 实现哈希fallback使用生成器repr，不是字节码内容 |
| [FA-01](#issue-fa-01) | P0 | D | FA | 健康卡空完整性门槛得到通过 |
| [FA-02](#issue-fa-02) | P0 | D | FA | 健康卡可以混装其他因子的评级，并接受自报准入结论 |
| [FA-03](#issue-fa-03) | P1 | D | FA | 健康政策frozen但规则字典可原地变更 |
| [FA-04](#issue-fa-04) | P1 | S | FA | 缺一项证据就把整个维度打零，缺失与低质量混同 |
| [FA-05](#issue-fa-05) | P1 | S | FA | 形状质量不能把高单调性与高U型评分同时当共同优点 |
| [FA-06](#issue-fa-06) | P1 | G | FA/DA/FE | 字段类别、机制标签和控制字段角色尚需精确绑定 |
| [FA-07](#issue-fa-07) | P0 | D | FA | 旧PromotionGate可接受裸数值和默认成熟，并软化已有硬失败 |
| [FA-08](#issue-fa-08) | P0 | D | FA | MERGE_NEAREST把因子投到最大簇却沿用别簇高相似度 |
| [FA-09](#issue-fa-09) | P1 | S | FA | 低相似度、未测量与ANN近似不能用同一种证据状态 |
| [FA-10](#issue-fa-10) | P2 | G | FA | 增量ANN在新因子×簇循环重建索引，归属只看单个最近成员 |
| [FA-11](#issue-fa-11) | P1 | D | FA/FO | 旧Pareto实现允许缺失/NaN目标悄悄进入非支配前沿 |
| [FO-01](#issue-fo-01) | P0 | D | FO | Optimizer的QE适配器对多因子批返回空metrics |
| [FO-02](#issue-fo-02) | P0 | G | FO | QE适配器没有传递split、backend、参数和完整证据身份 |
| [FO-03](#issue-fo-03) | P0 | G | FO | Optimizer明确research_only，不能把已声明的阻断项当完成 |
| [FO-04](#issue-fo-04) | P1 | G | FO | 多保真阶段表与搜索执行需要穿透，串行限制是真实能力边界 |
| [FO-05](#issue-fo-05) | P1 | G | FO | 搜索配置可变与checkpoint恢复需要绑定实际执行状态 |
| [PL-01](#issue-pl-01) | P1 | D | Platform | 无效候选本应被报告，却因空hash触发报告对象异常 |
| [PL-02](#issue-pl-02) | P0 | D | Platform | 消费去重早于执行成功，失败候选可永久失去重试 |
| [PL-03](#issue-pl-03) | P1 | D | Platform | 平台把所有评估异常重新标成CAPABILITY，丢失重试分类 |
| [PL-04](#issue-pl-04) | P1 | G | Platform | 平台仍用RankIC是否存在先否决，绕过分用途准入政策 |
| [PL-05](#issue-pl-05) | P0 | G | Platform/FA/FE | 平台把semantic_family_hint或spec hash当factor_definition_ref |
| [PL-06](#issue-pl-06) | P1 | G | Platform/modeling | 特征快照只含本轮批准项，版本又依赖内存计数 |
| [PL-07](#issue-pl-07) | P0 | G | Platform/DA | 候选ArtifactRef的hash、大小、媒体类型和真实字节未对齐 |
| [PL-08](#issue-pl-08) | P1 | G | Platform/DA/FA | 平台骨架的事件、登记和发布不是可证明的原子事务 |
| [QA-01](#issue-qa-01) | P1 | D | QA | 对拍测试没有覆盖真实参数桥，且同名测试被覆盖 |
| [QA-02](#issue-qa-02) | P1 | G | QA | 已有文档和“已注册”不能替代可执行能力表 |
| [V-01](#issue-v-01) | P0 | V | 总协调/QA | 全量调用图与唯一权威清点，追踪旁路而非只改点名文件 |
| [V-02](#issue-v-02) | P1 | V | FA/DA/FE | 自动标签必须真的在每次候选进入时生成，并可重评更新 |
| [V-03](#issue-v-03) | P1 | V | FA/QE | 建立逐指标→逐维度→用途准入的完整政策，而非一个总分 |
| [V-04](#issue-v-04) | P1 | V | FO/FE/QE/FA | U/倒U与尾部修复需要低自由度实现、父子重评和完整血缘 |
| [V-05](#issue-v-05) | P1 | V | FO/QE/FA | 廉价初筛不能提前杀掉非线性、稀疏事件或互补特征 |
| [V-06](#issue-v-06) | P1 | V | FP/FE/QE/FO | 中性化是受约束的候选选择，必须验证最终暴露而非名字 |
| [V-07](#issue-v-07) | P0 | V | FE/FP/QA | 所有可生产滤波通过前缀不变与增量状态回放 |
| [V-08](#issue-v-08) | P1 | V | FP/modeling/FE | 树模型与神经网络的输入表示有独立契约，不统一强制zscore |
| [V-09](#issue-v-09) | P1 | V | FO | 诊断驱动小预算搜索，而非穷举所有预处理排列 |
| [V-10](#issue-v-10) | P1 | V | FO/QE | 多保真初筛与精筛用相同身份、明确样本，不能拿低精度分数冒充最终证据 |
| [V-11](#issue-v-11) | P0 | V | FO/QE/FA | 训练、验证、封存测试和跨session试验次数必须联合治理 |
| [V-12](#issue-v-12) | P0 | V | DA/QE/FO | Purging、embargo和PIT按真实标签区间验证，避免前视与过度清洗 |
| [V-13](#issue-v-13) | P1 | V | QE/回测执行域/DA | 实盘指标必须带可执行性与成本规格，研究probe另设身份 |
| [V-14](#issue-v-14) | P1 | V | FA/QE/FE | 相似度去重是分层流程，不能一把0.7/0.95阈值删除所有高相关因子 |
| [V-15](#issue-v-15) | P1 | V | FA/modeling | 新增簇、增量归属和全量重聚类有明确版本与触发 |
| [V-16](#issue-v-16) | P1 | V | FO/FA/QE | 筛选允许少量互补版本，而不是单冠军或全收 |
| [V-17](#issue-v-17) | P1 | V | FA/QE/modeling | 父因子拆分和簇聚合都生成新资产，必须再次评价 |
| [V-18](#issue-v-18) | P1 | V | FE/DA/QE/FA/FP/modeling | 定义、配方、落值、评价、簇与模型身份串成可追溯链 |
| [V-19](#issue-v-19) | P1 | V | FE/DA/FP/FA | 每日因子更新执行冻结配方，不重新跑优化器 |
| [V-20](#issue-v-20) | P0 | V | FA/DA/Platform | 失败values清理应真实执行且不破坏生产、共享DAG或统计历史 |
| [V-21](#issue-v-21) | P1 | V | FA/QE/Platform | 上线后诊断与漂移监测不重用测试集调参 |
| [V-22](#issue-v-22) | P1 | V | FO/FA/Platform | Agent仅作有界提案和解释，不承担不可验证的硬门 |
| [V-23](#issue-v-23) | P0 | V | QA/全模块 | 真实端到端与硬件基准必须覆盖公共入口，而不止内核mock |
| [V-24](#issue-v-24) | P1 | V | QA/总协调 | CI和最终验收追踪所有发现、迁移以及既有证据失效范围 |

### 10.2 新80项专项索引

| ID | 优先级 | 类别 | 责任边界 | 专项任务 |
|---|---|---|---|---|
| [DTA-01](#issue-dta-01) | P0 | E | DA/FE | 证券永久身份、更名和跨市场同名不能共用时序状态 |
| [DTA-02](#issue-dta-02) | P0 | E | DA/FE | 财报必须按双时态可见版本连接，不能按报告期连接 |
| [DTA-03](#issue-dta-03) | P0 | E | DA/FE/QE | 拆分、分红、复权和停牌复牌不应制造伪收益 |
| [DTA-04](#issue-dta-04) | P0 | E | DA/FA/QE | 股票池与退市样本覆盖必须是历史可投资集合 |
| [DTA-05](#issue-dta-05) | P0 | E | DA/执行域/QE | 可交易性要区分买入、卖出、做空和回补 |
| [DTA-06](#issue-dta-06) | P1 | E | DA/FE/FA | 派生字段的来源分类与单位代数必须一致 |
| [DTA-07](#issue-dta-07) | P1 | E | DA/FE/QE | 跨市场时区、交易日和汇率对齐不允许日期字符串碰撞 |
| [DTA-08](#issue-dta-08) | P1 | E | DA/FE/FP/QE | 缺失原因必须从原始字段一直传到模型与清理任务 |
| [DTA-09](#issue-dta-09) | P0 | E | DA/FE/FP | 所有as-of和多表join必须验证基数及方向 |
| [DTA-10](#issue-dta-10) | P1 | E | DA/FE/Platform | 晚到数据watermark与局部重算范围需要显式依赖 |
| [DTA-11](#issue-dta-11) | P0 | E | DA/QE/FO | 多期限标签成熟度与共同比较样本必须同时保留 |
| [DTA-12](#issue-dta-12) | P1 | E | FE/FP/QE/DA | 面板布局、dtype与只读所有权必须在跨库边界锁定 |
| [STA-01](#issue-sta-01) | P0 | E | QE | RankIC按每个因子的共同有效集合重新确定秩 |
| [STA-02](#issue-sta-02) | P1 | E | QE/FA | HAC和有效时间样本数不得由股票格数替代 |
| [STA-03](#issue-sta-03) | P1 | E | QE/FO | 块bootstrap必须保留时间依赖和候选间的配对关系 |
| [STA-04](#issue-sta-04) | P0 | E | FO/QE/FA | 搜索尝试账本与FDR假设族不能通过改名重置 |
| [STA-05](#issue-sta-05) | P1 | E | QE/FO | DSR的样本、Sharpe尺度和有效试验数要可审计 |
| [STA-06](#issue-sta-06) | P1 | E | QE/FO | PBO或多切分验证只能用于声明的研究目的 |
| [STA-07](#issue-sta-07) | P1 | E | QE/FA/FO | 训练验证衰减要区分过拟合、市场状态与近零分母 |
| [STA-08](#issue-sta-08) | P1 | E | FA/QE | 评级锚点校准和显示分位数必须分开版本 |
| [STA-09](#issue-sta-09) | P1 | E | QE/FA/FO | 市场状态切片要标明事前可知还是事后压力复盘 |
| [STA-10](#issue-sta-10) | P1 | E | QE/FO | 期限峰值、符号变化与滤波半衰期不能混作一个数字 |
| [STA-11](#issue-sta-11) | P1 | E | QE/FA | 尾部指标需要明确损失符号、加权分位与样本不足 |
| [STA-12](#issue-sta-12) | P1 | E | FA/QE | 多维评级不重复计分，诊断结论必须给出证据原因 |
| [RCP-01](#issue-rcp-01) | P0 | E | FP/FE/modeling | 缺失指标、新鲜度等辅助通道必须分支输出而非替换主信号 |
| [RCP-02](#issue-rcp-02) | P1 | E | FE/FP | z-score、rank的幂等和零方差处理必须带适用前提 |
| [RCP-03](#issue-rcp-03) | P1 | E | FE/FP/FO | 去极值政策需区分截面winsor、时间拟合阈值和合法经济极值 |
| [RCP-04](#issue-rcp-04) | P1 | E | FE/FP/QE | 中性化求解器必须声明设计矩阵、权重与精确性 |
| [RCP-05](#issue-rcp-05) | P1 | E | FP/FE/QE | 最终中性约束与rank、clip、平滑的顺序要以输出性质验收 |
| [RCP-06](#issue-rcp-06) | P1 | E | FE/FP | 因果平滑状态要区别重复事件值、休市和新观测 |
| [RCP-07](#issue-rcp-07) | P1 | E | FO/FP/FE/QE | 偏斜U型的中心拟合与左右分支需要低自由度监督契约 |
| [RCP-08](#issue-rcp-08) | P1 | E | FO/QE/FE | 尾部修复不能凭单张20层平均图选择任意cutoff |
| [RCP-09](#issue-rcp-09) | P1 | E | FA/FP/FE | 原始、中性、平滑、模型表示版本不互相覆盖 |
| [RCP-10](#issue-rcp-10) | P1 | E | FE/FP/FA | 可读DSL与执行计划必须能无损往返并显示实际参数 |
| [SRH-01](#issue-srh-01) | P1 | E | FO/FE/FP | 参数变异需要证明改变有效执行，而非只改变描述 |
| [SRH-02](#issue-srh-02) | P1 | E | FO/FA/QE | 诊断冲突与多问题并存时先修完整性再修效用 |
| [SRH-03](#issue-srh-03) | P1 | E | FO/FA | Pareto需要容差、未知状态与可比较目标集合 |
| [SRH-04](#issue-srh-04) | P1 | E | FO/FA/QE/modeling | 多赢家集合选择应验证边际贡献与母因子上限 |
| [SRH-05](#issue-srh-05) | P1 | E | FO/QE | 分阶段晋级必须防止短窗幸存偏差与样本不公平 |
| [SRH-06](#issue-srh-06) | P1 | E | FO/Platform | 预算预留、取消与重试要有持久化资源账本 |
| [SRH-07](#issue-srh-07) | P0 | E | FO/QE/Platform | 封存测试访问要绑定候选集合且不能在日志中泄露 |
| [SRH-08](#issue-srh-08) | P1 | E | FO | 候选生成器与异步执行顺序必须可复现 |
| [SRH-09](#issue-srh-09) | P1 | E | FO/QE | 早停需考虑评价噪声、成本和基线而非裸分数平台 |
| [SRH-10](#issue-srh-10) | P1 | E | FO/FA | 失败修复、参数平台与以后复用应沉淀为有版本知识 |
| [SIM-01](#issue-sim-01) | P1 | E | FA/QE | 相似度类型与逐日聚合方法必须写进fingerprint规格 |
| [SIM-02](#issue-sim-02) | P1 | E | FA/QE | 共同覆盖不足时不能证明不重复或高相似 |
| [SIM-03](#issue-sim-03) | P1 | E | FA/QE/FO | 极高相关候选也要检查尾部、mask和参数平台差异 |
| [SIM-04](#issue-sim-04) | P1 | E | FA/QE/modeling | 相似图不等于协方差矩阵，缺边补零需禁止隐式使用 |
| [SIM-05](#issue-sim-05) | P1 | E | FA/QE | Leiden分簇结果还要检查经济同质性与方向一致 |
| [SIM-06](#issue-sim-06) | P1 | E | FA/DA | 未测边、测得低边与认证图内容必须分开 |
| [SIM-07](#issue-sim-07) | P2 | E | FA/QE | ANN索引的召回质量、版本和增量维护需要验收 |
| [SIM-08](#issue-sim-08) | P1 | E | FA | 增量归簇要比较多个支持邻居和歧义间隔 |
| [SIM-09](#issue-sim-09) | P1 | E | FA/modeling | 簇split/merge要保留逻辑身份和旧模型引用 |
| [SIM-10](#issue-sim-10) | P1 | E | FA/FE/FP/modeling | 簇压缩成模型特征时，聚合函数与拟合窗口须固定 |
| [MOD-01](#issue-mod-01) | P0 | E | modeling/FP/FA | 模型每个训练fold都要重新拟合特征选择和预处理状态 |
| [MOD-02](#issue-mod-02) | P1 | E | modeling/QE/FA | 低单因子IC特征必须通过有限预算的增量检验 |
| [MOD-03](#issue-mod-03) | P1 | E | FO/FP/modeling/FA | 中性化的非劣性容忍区要与消费者目标一致 |
| [MOD-04](#issue-mod-04) | P1 | E | FP/modeling | 神经网络、树与线性模型各自处理缺失和尺度 |
| [MOD-05](#issue-mod-05) | P0 | E | modeling/FA/Platform | 模型特征顺序、列内定义与数据集指纹必须严格校验 |
| [MOD-06](#issue-mod-06) | P1 | E | FA/modeling/Platform | 新增研究因子不能自动改变线上簇列或模型输入 |
| [MOD-07](#issue-mod-07) | P1 | E | modeling/QE/FA | 模型预测评价、组合评价与特征归因要分开 |
| [MOD-08](#issue-mod-08) | P1 | E | FA/modeling/Platform | 漂移监测、影子运行和回滚不能触发无界自动优化 |
| [OPS-01](#issue-ops-01) | P0 | E | Platform/DA/FA | 相同工作意图的并发写入必须原子幂等 |
| [OPS-02](#issue-ops-02) | P1 | E | FO/QE/FE/Platform | 资源调度要包含共享数据、显存工作区和背压 |
| [OPS-03](#issue-ops-03) | P1 | E | QE/FE/Platform | OOM和后端降级必须保持语义且不能伪报GPU完成 |
| [OPS-04](#issue-ops-04) | P2 | E | QA/QE/FE | 性能基准必须覆盖端到端与设备同步边界 |
| [OPS-05](#issue-ops-05) | P0 | E | FA/DA/Platform | 失败物化值GC需要根集合、租约和回滚保留窗口 |
| [OPS-06](#issue-ops-06) | P0 | E | FA/DA/Platform | 扫描后新增引用的GC竞态必须可防护 |
| [OPS-07](#issue-ops-07) | P1 | E | DA/FA | 删除回执要覆盖对象版本、缓存和实际空间回收 |
| [OPS-08](#issue-ops-08) | P0 | E | FA/QE/FE/Platform | 算法修复后要按依赖失效历史评级、簇和模型 |
| [OPS-09](#issue-ops-09) | P1 | E | FE/FP/DA/FA | 离线回放必须在无Agent、无实时API下重现生产特征 |
| [OPS-10](#issue-ops-10) | P1 | E | Platform/FO/FA | 错误状态、等待成熟、维修失败与淘汰必须有不同动作 |
| [QAT-01](#issue-qat-01) | P1 | E | QA/全模块 | 属性测试需要覆盖因子、证券、时间轴与掩码变换 |
| [QAT-02](#issue-qat-02) | P0 | E | QA/QE/FE | 金标必须独立于待测实现，CPU/GPU不能一起算错 |
| [QAT-03](#issue-qat-03) | P1 | E | QA/QE/FE | 没有真实GPU时只能验证协议，不能签发硬件parity |
| [QAT-04](#issue-qat-04) | P1 | E | QA/FA/FO/QE | 变异测试证明守门真正保护，而不是写了断言名 |
| [QAT-05](#issue-qat-05) | P1 | E | QA/全模块 | 序列化与内容身份fuzz要覆盖错误值和跨对象嫁接 |
| [QAT-06](#issue-qat-06) | P1 | E | QA/DA/FE | 数据差分与增量回放要包含晚到、修订、停牌和断点 |
| [QAT-07](#issue-qat-07) | P1 | E | QA/全模块 | 独立wheel与monorepo必须使用同一有效实现 |
| [QAT-08](#issue-qat-08) | P1 | E | 总协调/QA | 任务关闭必须同时覆盖需求、入口和历史结果迁移 |

---

<a id="inherited"></a>
## 11. V1/V2合并后的97项详细工作单

以下保留V2每一项原详细工作单的完整正文，并在其后增加V3实施细节、补充测试和历史制品处置。工作单内“已阅源码”“隔离复现”是**历史V2来源**，本轮并未再次实仓运行；当前HEAD变更需要重新核查。V1 AUD编号与全部子问题的对应关系见第14节。每项历史表述与新统一合同的关系，按第13.1节的测试前提澄清执行。

### 11.1 QE：评价、统计、CPU/GPU、组合与时间

<a id="issue-qe-01"></a>
#### QE-01 · 组合绩效被错误归因：不同因子收到同一个指标

**优先级**：P0　**证据分类**：源码确认缺陷　**Owner**：QE

**已阅源码**：[`quant_evaluator/runtime/evaluator.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/quant_evaluator/runtime/evaluator.py)。

**函数/边界**：`_make_panel_wrapper / _to_per_factor_array / evaluate`。

**现象与证据**：通用 Sharpe/Sortino/win_rate 包装器把 LabelBundle 的 (T,N) 股票远期收益面板当作 returns；随后对结果整体 nanmean。long_short_returns 的 (T,F) 结果也被跨因子平均，最终同一标量复制给全部 factor_id。

**影响**：评价与排序失去因子区分度；正反因子可能同时得到 0 的多空收益；股票收益统计被误称为因子组合绩效。

**修改要求**：将收益来源限定为逐因子的 ProbePortfolioArtifact / ExecutablePortfolioArtifact；显式区分 T,N 与 T,F。各因子独立输出，批级市场统计只能放入 context，不得复制为因子表现。

**验收**：
1. 构造 f 与 -f，净成本为0时多空序列相反且分别保存。
2. 加入第三个无关因子不改变前两因子指标。
3. Sharpe必须来自各自日度组合PnL而非标签面板。

**依赖**：按波次/主合同安排，无额外单项依赖。 **需求映射**：R04, R07。 **隔离复现索引**：PR-09（非全仓测试）。

**V3实施展开——具体怎么改**

先列出所有 returns 参数的实际调用者，给股票标签、逐因子诊断收益、可执行组合收益分别设明确类型。删除跨 F 的隐式汇总；批级行情统计放入独立 context。公开返回按 factor_id 对齐，adapter 只做结构翻译。金融组合函数不能仅凭数组 shape 接受收益来源。

**V3追加回归——防止只修表面**

两因子正反向、不同缺失模式、F=1/F=3、调换因子顺序分别测；在相同因子值下换一个无关同行因子不得影响原指标。

**V3历史对象与兼容性处置**

所有由该包装器生成的因子组合绩效、对应等级与择优结果进入影响清单；不是改完函数后沿用旧分数。

**关联专项**：[MOD-07](#issue-mod-07)。这些专项可在同一根因PR落实。

---

<a id="issue-qe-02"></a>
#### QE-02 · 公共接口把序列和分层向量静默压成均值

**优先级**：P0　**证据分类**：源码确认缺陷　**Owner**：QE

**已阅源码**：[`quant_evaluator/runtime/evaluator.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/quant_evaluator/runtime/evaluator.py)。

**函数/边界**：`_to_per_factor_array / EvaluationBundle assembly`。

**现象与证据**：任意二维且末轴为 F 的结果被 nanmean(axis=0)。因此 (T,F) 日度证据和 (Q,F) 分层向量都可丢失，返回结构没有同步保存原始 artifact。

**影响**：U型、尾部、衰减、回撤和时间稳定性无法从结果可靠还原；同名 metric 的输出类型与注册元数据不一致。

**修改要求**：按 MetricSpec.output_type 构造 scalar/series/vector/matrix artifact；聚合标量有独立 ID，保留轴、有效性与源引用，禁止凭 ndim 猜语义。

**验收**：
1. rank_ic_series 保留 T×F。
2. quantile_returns_full 保留 Q×F 或约定的 T×Q×F。
3. 序列往返序列化不丢轴。

**依赖**：按波次/主合同安排，无额外单项依赖。 **需求映射**：R03, R04。 **隔离复现索引**：PR-10（非全仓测试）。

**V3实施展开——具体怎么改**

先读取 MetricSpec 声明的轴及 artifact kind，再选择序列化方法。保留 daily IC、T×Q×F 分层和 Q×F 汇总三个不同对象；想要均值时请求单独的 scalar metric。小数据可内嵌，大数据保存 typed ref，引用仍包含轴与内容哈希。

**V3追加回归——防止只修表面**

对同均值但形状不同的两个分层曲线，输出必须不同且可还原；报告只消费已存对象，不补算丢失的证据。

**V3历史对象与兼容性处置**

向量被压缩的历史证据不能凭标量恢复，应重新计算并升级 artifact schema。

---

<a id="issue-qe-03"></a>
#### QE-03 · 形式上存在的依赖被当成已经构造的证据

**优先级**：P0　**证据分类**：源码确认缺陷　**Owner**：QE

**已阅源码**：[`quant_evaluator/runtime/evaluator.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/quant_evaluator/runtime/evaluator.py)；[`quant_evaluator/registry/metrics.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/quant_evaluator/registry/metrics.py)。

**函数/边界**：`_validate_metric_requirements / _compute_metric`。

**现象与证据**：available.update(_FACADE_ARTIFACT_TYPES) 把 artifact 类加入可用项；实际 binder 只提供 batch、label、metadata 和已计算名字，公共 facade 仅给IC与旧面板做包装，未为 qr 等构造 QuantileReturnArtifact。

**影响**：形状指标虽然已注册，公开评估路径仍可能 unresolved required parameter；“功能已实现”的清单不可用。

**修改要求**：在现有 planner 中建立真实 artifact builder 与依赖拓扑；builder 返回绑定请求上下文的对象，require 验证其内容及身份。缺输入返回明确状态，不以类对象或空对象充数。

**验收**：
1. 从公开 evaluate 请求 quantile_monotonicity/u_shape_score 等真实注册项。
2. 逐项验证构建→缓存→输出→adapter→health-card全链。
3. 依赖缺失必须明确不可计算。

**依赖**：按波次/主合同安排，无额外单项依赖。 **需求映射**：R03, R04。

**V3实施展开——具体怎么改**

在已有依赖图加入 builder 注册与输入合同检查，不创建平行 planner。把 requirements 区分为原始面板、派生序列、组合和外部风险暴露；只在 builder 实际产出且验证上下文后标满足。IC 类包装器与形状类统一走计划。

**V3追加回归——防止只修表面**

缺暴露时只让依赖暴露的可选任务返回缺失；必需任务阻断，独立 coverage 仍可计算；相同依赖同时被请求只构造一次。

**V3历史对象与兼容性处置**

此前仅注册、未能执行的能力保持未验收；用新的真实公开路径测试更新能力矩阵。

---

<a id="issue-qe-04"></a>
#### QE-04 · GPU 分支绕过统一校验并遗漏 validity

**优先级**：P0　**证据分类**：源码确认缺陷　**Owner**：QE

**已阅源码**：[`quant_evaluator/runtime/evaluator.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/quant_evaluator/runtime/evaluator.py)；[`quant_evaluator/contracts/label_bundle.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/quant_evaluator/contracts/label_bundle.py)。

**函数/边界**：`evaluate CUDA early return`。

**现象与证据**：CUDA 分支在 sealed split overlap 与资产轴检查之前返回；stage_factors/stage_labels 只接收 values，没有 factor/label validity、时钟与切分上下文。

**影响**：同一个请求在CPU被拒绝、GPU可能继续；无效但有限的值被计入覆盖、IC和组合收益。

**修改要求**：所有后端先执行同一个请求校验器，再传完整规范化数据与mask；时序/成熟度/split/identity校验不得由后端选择决定。

**验收**：
1. 同请求CPU/GPU在非法split上同拒。
2. 把masked格改成极端有限值，所有结果不变。
3. 错轴错时钟在staging前拒绝。

**依赖**：按波次/主合同安排，无额外单项依赖。 **需求映射**：R08, R15。

**V3实施展开——具体怎么改**

将 validation 从 CPU 分支移到 dispatch 前；传入一个不可变 ValidatedEvaluationRequest 等价合同。设备会话必须保留因子和标签各自 mask 以及归一后的联合 mask；不允许 backend 自行重建更宽松的有效集合。

**V3追加回归——防止只修表面**

用 masked 但有限的极大值、错序证券、未成熟标签、非法 split 同时请求 CPU/CUDA；两端均在搬运或执行之前作一致裁决。

**V3历史对象与兼容性处置**

以往 GPU 缺掩码或绕过校验产生的评价需按请求范围隔离，不能只重算数值不重做资格。

**关联专项**：[DTA-12](#issue-dta-12)、[STA-01](#issue-sta-01)、[OPS-03](#issue-ops-03)、[QAT-03](#issue-qat-03)。这些专项可在同一根因PR落实。

---

<a id="issue-qe-05"></a>
#### QE-05 · CPU/GPU 输出合同、metric ID 与 artifact kind 不一致

**优先级**：P1　**证据分类**：源码确认缺陷　**Owner**：QE

**已阅源码**：[`quant_evaluator/runtime/evaluator.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/quant_evaluator/runtime/evaluator.py)；[`quant_evaluator/runtime/gpu_executor.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/quant_evaluator/runtime/gpu_executor.py)。

**函数/边界**：`evaluate / GPUExecutor.run`。

**现象与证据**：CPU返回 EvaluationBundle，GPU返回 BatchEvaluationBundle；GPU把 quantile_returns_full 写入 quantile_returns，把spread放vector；旧适配器消费metric_values等字段。

**影响**：切换后端导致下游不可用或错误解释；库级指标版本和状态没有统一。

**修改要求**：保留内部高效columnar布局，但公共返回同一版本证据合同或显式适配；canonical IDs、输出类型、坐标、版本、split与warnings等价。

**验收**：
1. 同批所有支持指标CPU/GPU schema快照一致。
2. 别名和规范名解析同一ID。
3. 不支持指标返回一致明确状态。

**依赖**：QE-02, QE-04。 **需求映射**：R08, R15。

**V3实施展开——具体怎么改**

内部 BatchEvaluationBundle 可以保留列式结构，外部必须有一个稳定的 typed evidence envelope。canonical metric ID 在计划层解析，kernel 不改名；序列/向量/scalar 的布局和缺失状态由同一 spec 控制。

**V3追加回归——防止只修表面**

CPU/GPU 的 JSON/schema 快照、单位、有效样本、别名和 factor 顺序一致；只允许数值在声明容差内不同。

**V3历史对象与兼容性处置**

新增兼容适配和版本迁移，旧下游不能因升级收到不同对象后继续凭 getattr 默认值运行。

**关联专项**：[OPS-03](#issue-ops-03)。这些专项可在同一根因PR落实。

---

<a id="issue-qe-06"></a>
#### QE-06 · GPU 的ICIR、样本门槛和分层数脱离指标政策

**优先级**：P0　**证据分类**：定义／口径冲突　**Owner**：QE

**已阅源码**：[`quant_evaluator/runtime/gpu_executor.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/quant_evaluator/runtime/gpu_executor.py)；[`quant_evaluator/metrics/ic_summary.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/quant_evaluator/metrics/ic_summary.py)；[`quant_evaluator/metrics/registry_adapters.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/quant_evaluator/metrics/registry_adapters.py)。

**函数/边界**：`GPUExecutor.run IC / quantile branches`。

**现象与证据**：GPU硬编码 min_obs=20、Q=10；ICIR除以max(sd,1e-12)，CPU遇到近零标准差返回NaN，CPU一些公开分层默认Q=5。

**影响**：零方差IC产生极高有限IR；同一metric在不同后端得到不同等级，10/20层请求不能可靠表达。

**修改要求**：统一编译MetricSpec、参数及min-observations策略到执行计划；GPU只实现同一数学合同，不自定默认值；零方差使用一致状态。

**验收**：
1. 恒定IC/单样本/11-19有效资产/5-10-20分层对拍。
2. 两端数值、mask、状态均一致。

**依赖**：QE-04。 **需求映射**：R03, R04, R08。 **隔离复现索引**：PR-24（非全仓测试）。

**V3实施展开——具体怎么改**

从注册表编译 effective parameters，将 min_assets、min_periods、ddof、zero_variance、Q 与 rank method 全量传给设备计划；执行器删除对应硬编码。零方差不通过给分母加 epsilon 伪造稳定收益。

**V3追加回归——防止只修表面**

分别测 IC=常数正值、常数零、只有一个有效日、10/19/20只有效股票；所有数值边界采用该指标明确的参考政策。

**V3历史对象与兼容性处置**

ICIR 尺度和 Q 改变需升级实现/政策版本；旧等级不可在没有单位核对的情况下重新解释。

---

<a id="issue-qe-07"></a>
#### QE-07 · GPU 把两类不同换手指标计算成同一数值

**优先级**：P1　**证据分类**：定义／口径冲突　**Owner**：QE

**已阅源码**：[`quant_evaluator/runtime/gpu_executor.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/quant_evaluator/runtime/gpu_executor.py)；[`quant_evaluator/metrics/registry_adapters.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/quant_evaluator/metrics/registry_adapters.py)。

**函数/边界**：`GPUExecutor.run turnover branch`。

**现象与证据**：GPU turnover 与 factor_turnover_rate 都输出同一个 batched_turnover 的均值；CPU前者是秩代理权重换手，后者是顶部成员换手。

**影响**：优化器无法判断信号平滑是否改变真正关心的交易层换手；评级阈值失去单位基础。

**修改要求**：拆清 signal_rank_turnover、quantile_membership_turnover、portfolio_weight_turnover、executed_turnover；别名只绑定同义项，各自注册一个权威实现。

**验收**：
1. 构造顶部成员不变但权重变化样本，两个指标应不同。
2. 构造证券进入退出与mask变化，换手计入规则一致。

**依赖**：按波次/主合同安排，无额外单项依赖。 **需求映射**：R06, R08。

**V3实施展开——具体怎么改**

给四种换手建不同 definition_id；旧 alias 仅绑定完全同义定义。信号秩变化、分位成员变更与实际成交换手从各自权威中间对象计算，不能把 proxy 指标送入成本扣减函数。

**V3追加回归——防止只修表面**

前十大成员不变但内部次序改变、同成员价格漂移、退出股票池、只剩一侧持仓，分别核对四个指标。

**V3历史对象与兼容性处置**

识别过去错误绑定到 GPU turnover 的字段；成本模型引用必须同步迁移。

---

<a id="issue-qe-08"></a>
#### QE-08 · GPU 分层单调性与CPU回答的不是同一问题

**优先级**：P1　**证据分类**：定义／口径冲突　**Owner**：QE

**已阅源码**：[`quant_evaluator/runtime/gpu_executor.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/quant_evaluator/runtime/gpu_executor.py)；[`quant_evaluator/registry/metrics.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/quant_evaluator/registry/metrics.py)。

**函数/边界**：`quantile_monotonicity`。

**现象与证据**：GPU统计逐日相邻层差>0的比例并跨时间平均；CPU注册语义基于时间平均分层剖面；NaN差在 >0 中还可能被计为False。

**影响**：短期嘈杂但长期稳定的因子会得到不一致形状等级；缺失分层被误作不单调。

**修改要求**：保留两种有意义统计但使用不同metric IDs；选择一个正式形状质量主指标，明确有效比较分母。

**验收**：
1. 均值单调但逐日波动的合成数据能区分两指标。
2. 插入缺失层不应人为降低有效观测上的比例。

**依赖**：按波次/主合同安排，无额外单项依赖。 **需求映射**：R04, R08。

**V3实施展开——具体怎么改**

同时保留“长期均值曲线的单调性”与“每日曲线单调比例”时，用独立 ID 和分母。无有效相邻对时返回证据不足；NaN 比较不得当 false 纳入分母。方向由冻结 orientation 处理。

**V3追加回归——防止只修表面**

构造均值单调、日度一半反向的面板；两个指标应按各自定义不同且 CPU/GPU 同义一致。

**V3历史对象与兼容性处置**

等级只引用预声明主指标，不能因为新增另一变体就自动双重计分。

---

<a id="issue-qe-09"></a>
#### QE-09 · GPU cohort 把远期标签当成日度持仓收益

**优先级**：P0　**证据分类**：源码确认缺陷　**Owner**：QE

**已阅源码**：[`quant_evaluator/runtime/gpu_executor.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/quant_evaluator/runtime/gpu_executor.py)；[`quant_evaluator/kernels/gpu/portfolio.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/quant_evaluator/kernels/gpu/portfolio.py)；[`quant_evaluator/contracts/label_bundle.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/quant_evaluator/contracts/label_bundle.py)。

**函数/边界**：`GPUExecutor.run probe_ls_*`。

**现象与证据**：GPUExecutor把staged labels直接传给compute_cohort_pnl_batch_gpu的daily_returns参数，并固定holding=20；LabelBundle是显式前向标签，kernel则要求 P_t/P_(t-1)-1。

**影响**：收益时序和期限错位，可能重复复利重叠标签；Sharpe与回撤不具可解释性。

**修改要求**：引入独立的HoldingReturnPanel/价格执行输入，类型层禁止LabelBundle替代；holding、entry/exit、成本与可交易数据来自已冻结PortfolioSpec。

**验收**：
1. H=1/5/10/20请求都从独立日收益累计。
2. 将多期标签作为日收益传入必须拒绝。
3. CPU/GPU共用相同组合规格。

**依赖**：QE-04, QE-23。 **需求映射**：R04, R07, R15。

**V3实施展开——具体怎么改**

设备会话分别 stage ForwardLabel 与 HoldingReturnPanel，并按不可混用的类型/来源引用读取。组合 builder 需要 PortfolioSpec 和明确真实日收益；缺此输入就阻断该组合证据，不用已有 labels 凑数。

**V3追加回归——防止只修表面**

H10 预测标签绝不能通过名为 daily_returns 的形参进入组合路径；H=1/5/20与独立价格账本对齐。

**V3历史对象与兼容性处置**

将受影响 probe 指标及据其选择的候选标记需重评；封存样本的故障重放另记审计用途。

**关联专项**：[DTA-11](#issue-dta-11)。这些专项可在同一根因PR落实。

---

<a id="issue-qe-10"></a>
#### QE-10 · GPU 为每个组合指标重复计算同一组合PnL

**优先级**：P2　**证据分类**：已见接口缺口／局部风险　**Owner**：QE

**已阅源码**：[`quant_evaluator/runtime/gpu_executor.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/quant_evaluator/runtime/gpu_executor.py)。

**函数/边界**：`GPUExecutor.run portfolio loop`。

**现象与证据**：每个probe_ls_sharpe/calmar/max_drawdown等分支重新调用cohort kernel和portfolio metrics，而非缓存共享结果。

**影响**：因子批量越大，重复大张量计算和分配越严重；有GPU不等于低成本。

**修改要求**：编译DAG一次构建PnL/净值/回撤，后续指标引用；按portfolio spec和输入hash缓存，不跨mask/label复用。

**验收**：
1. 同时请求5个组合指标，PnL builder调用次数=1。
2. 输出与逐项请求一致。
3. 记录真实峰值显存与耗时。

**依赖**：QE-09。 **需求映射**：R08。

**V3实施展开——具体怎么改**

把组合输出放入 request-local artifact store，以 (value_ref, execution_spec, missingness, clock) 为键。Sharpe/Calmar/DD/turnover 复用同一份 PnL 和 NAV；请求结束按引用释放设备缓存。

**V3追加回归——防止只修表面**

spy 证明同一组合规格只构造一次；第二个成本规格必须产生第二份对象，不能误复用第一份净收益。

**V3历史对象与兼容性处置**

性能改造不得改变组合语义版本；若浮点归并方式变化，记录实际实现和容差。

---

<a id="issue-qe-11"></a>
#### QE-11 · 回撤计算漏掉初始净值1，第一笔亏损消失

**优先级**：P0　**证据分类**：源码确认缺陷　**Owner**：QE

**已阅源码**：[`quant_evaluator/metrics/portfolio_stats.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/quant_evaluator/metrics/portfolio_stats.py)；[`quant_evaluator/metrics/underwater.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/quant_evaluator/metrics/underwater.py)。

**函数/边界**：`compute_maximum_drawdown / _drawdown_curve`。

**现象与证据**：cumprod(1+r)后直接maximum.accumulate，未把初始NAV=1纳入历史峰值。[-0.1,0]得到最大回撤0。

**影响**：回撤门槛被系统性高估，首日亏损和前段水下期漏算。

**修改要求**：统一一个NAV/episode权威，明确初始时点及峰值索引；初始净值参与running peak。所有风险、图表、GPU内核调用同一语义。

**验收**：
1. [-.1,0]最大回撤=.1且两期未恢复。
2. [.1,-.1]符合手算。
3. 空/单日/多因子索引一致。

**依赖**：按波次/主合同安排，无额外单项依赖。 **需求映射**：R07。 **隔离复现索引**：PR-04（非全仓测试）。

**V3实施展开——具体怎么改**

建立包含 t0 初始资本的 NAV artifact；running peak 从 t0 开始，收益数组与 NAV 数组的长度差显式记录。峰值/谷值返回真实时间标识，不用固定0掩盖未定义索引。

**V3追加回归——防止只修表面**

[-0.1,0] 对应最大回撤0.1；初始峰为 t0。再测先涨后跌、多个相等峰、空输入、末日恢复。

**V3历史对象与兼容性处置**

重算依赖旧初始峰语义的回撤、Calmar、水下卡和报告。

**关联专项**：[QAT-02](#issue-qat-02)。这些专项可在同一根因PR落实。

---

<a id="issue-qe-12"></a>
#### QE-12 · 破产状态被NaN屏蔽，最大回撤可能反而为0

**优先级**：P0　**证据分类**：源码确认缺陷　**Owner**：QE

**已阅源码**：[`quant_evaluator/metrics/portfolio_stats.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/quant_evaluator/metrics/portfolio_stats.py)；[`quant_evaluator/metrics/underwater.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/quant_evaluator/metrics/underwater.py)。

**函数/边界**：`wealth<=0 wipeout paths`。

**现象与证据**：非正净值之后dd被设NaN并在nanmin中排除；[.1,-1]可能回报最大回撤0。wealth_curve、aligned_wealth_curve、risk各自采用不同终止规则。

**影响**：灾难事件可以变成低风险或缺失项，污染准入与模型训练证据。

**修改要求**：保留防止负净值假恢复的保护，但额外返回破产/违约/不可解释损失状态及截至违约损失；净值归零在约定资本账户下应有100%回撤记录，负净值杠杆情形单独定义。

**验收**：
1. [.1,-1]不能得到风险通过。
2. [-1]显式破产。
3. [-1.2,-2]不能产生假恢复。
4. 所有派生指标传播该状态。

**依赖**：QE-11。 **需求映射**：R03, R07。 **隔离复现索引**：PR-05（非全仓测试）。

**V3实施展开——具体怎么改**

在 NAV 处理中显式定义 ACTIVE/DEFAULTED/INVALID_VALUATION 状态；归零记录损失事件，禁止用 NaN 删除灾难。负净值若不在允许资本合同内必须拒绝，不能继续 cumprod 后假恢复。

**V3追加回归——防止只修表面**

[0.1,-1]损失事件不可被其他高分盖过；<-100%样本按杠杆合同拒绝或有明确负资本处理。

**V3历史对象与兼容性处置**

只对受影响回撤链做可追溯失效，不能把破产当普通缺失项重评分。

**关联专项**：[STA-11](#issue-sta-11)。这些专项可在同一根因PR落实。

---

<a id="issue-qe-13"></a>
#### QE-13 · 空Sharpe返回三元回撤结构；空回撤缺少正确保护

**优先级**：P1　**证据分类**：源码确认缺陷　**Owner**：QE

**已阅源码**：[`quant_evaluator/metrics/portfolio_stats.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/quant_evaluator/metrics/portfolio_stats.py)。

**函数/边界**：`compute_sharpe_ratio / compute_maximum_drawdown`。

**现象与证据**：Sharpe的T==0分支返回(max_dd,empty_curve,peak)，不是scalar/(F,)；真正drawdown空输入后进入nanmin。

**影响**：边界批次会崩溃、schema漂移或被当成有效证据。

**修改要求**：统一空输入合同；保留维度正确的NaN/INSUFFICIENT_DATA，不把其他指标的返回结构复制过来。

**验收**：
1. T=0分别测试1D/2D以及F=0合法性。
2. 函数、公开API、适配器都不误判COMPUTED。

**依赖**：按波次/主合同安排，无额外单项依赖。 **需求映射**：R03。 **隔离复现索引**：PR-06（非全仓测试）。

**V3实施展开——具体怎么改**

写一个集中 shape/min-sample 边界验证器，在具体 metric 内返回该 spec 合法的空形状和状态。禁止复制其他指标返回分支；API 和 kernel 的异常映射一致。

**V3追加回归——防止只修表面**

T=0/F=0/N=0、空1D/2D、全NaN、全常数分别覆盖；serialize/deserialize 不把 tuple 误读为三个因子。

**V3历史对象与兼容性处置**

空证据保持不可准入，无需给不存在样本造迁移数值。

---

<a id="issue-qe-14"></a>
#### QE-14 · 水下期计算删除缺失日期，压缩了真实时间轴

**优先级**：P1　**证据分类**：源码确认缺陷　**Owner**：QE

**已阅源码**：[`quant_evaluator/metrics/underwater.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/quant_evaluator/metrics/underwater.py)。

**函数/边界**：`_as_1d / all duration and rolling consumers`。

**现象与证据**：_as_1d先过滤所有非finite值，后续把剩余位置计为交易日并切滚动窗口。

**影响**：缺报价/停牌/数据缺口会缩短水下长度；滚动252观测不是连续252交易日。

**修改要求**：输入携带时间轴与missing-reason；统计有效样本和时间跨度分离；必要时拒绝不完整路径，禁止无说明压缩日历。

**验收**：
1. 同一条路径插入未报价日期，不能无提示缩短日历水下期。
2. calendar与valid-observation两口径分开。

**依赖**：QE-11。 **需求映射**：R07, R15。 **隔离复现索引**：PR-11（非全仓测试）。

**V3实施展开——具体怎么改**

水下函数不再先 drop 时间点。使用带日期的 NAV 和 valuation mask 构建 episodes；交易日跨度、自然日跨度、有效估值观测数分别保存。未知净值区间不得悄悄插值为已知路径。

**V3追加回归——防止只修表面**

插入估值缺口不能缩短已知时间跨度；无法确认的恢复只能给 interval/unknown，不伪报已完成。

**V3历史对象与兼容性处置**

历史仅存有效观测序列但丢日期的证据需要重建时间轴后重评。

---

<a id="issue-qe-15"></a>
#### QE-15 · 恢复时间扫描跨越多个回撤事件，并混同未恢复样本

**优先级**：P1　**证据分类**：源码确认缺陷　**Owner**：QE

**已阅源码**：[`quant_evaluator/metrics/underwater.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/quant_evaluator/metrics/underwater.py)。

**函数/边界**：`compute_time_to_recovery`。

**现象与证据**：从当前episode起点向整个剩余lookback找最低dd，并未先确定该episode的恢复边界；未恢复且没有已完成样本时把ongoing_len作为恢复时间返回。

**影响**：可能跳过已恢复事件，低估或高估恢复周期；“未恢复”被当作已经测得的恢复时长。

**修改要求**：一次扫描构建peak/start/trough/recovered_at/end/censored episodes；均值只对明确定义样本计算，同时报告当前删失长度与完成比例。

**验收**：
1. 两个独立先浅后深回撤都被保留。
2. 全未恢复输出censored非已完成均值。
3. 零回撤/末日恢复/平台期分别测试。

**依赖**：QE-11, QE-14。 **需求映射**：R07。

**V3实施展开——具体怎么改**

先一次扫描分割回撤事件，再在每个事件内部找 trough。completed duration 和 censored age 是不同字段；均值必须声明仅完成事件还是含删失估计，默认不把观察年龄当恢复时长。

**V3追加回归——防止只修表面**

两段回撤中后一段更深不影响前一段恢复值；末日恢复、长期未恢复、水平峰平台分别测试。

**V3历史对象与兼容性处置**

更换 episode 定义后，所有 recovery 聚合和图表标注同步升级。

---

<a id="issue-qe-16"></a>
#### QE-16 · 偏度注释宣称bias=False，实际计算未做有限样本校正

**优先级**：P1　**证据分类**：定义／口径冲突　**Owner**：QE

**已阅源码**：[`quant_evaluator/metrics/underwater.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/quant_evaluator/metrics/underwater.py)。

**函数/边界**：`compute_return_skew`。

**现象与证据**：代码返回mean(((r-mean)/std(ddof=0))**3)，是未校正g1；说明却写bias=False。

**影响**：小样本尾部评级与SciPy参照不一致，阈值混用。

**修改要求**：选择并注册skew_biased/skew_adjusted之一或两者；按版本迁移，使用统一统计内核和min_n边界。

**验收**：
1. 不对称小样本对照scipy.stats.skew(bias=True/False)。
2. n<3、常数、NaN策略一致。

**依赖**：按波次/主合同安排，无额外单项依赖。 **需求映射**：R03, R07。 **隔离复现索引**：PR-12（非全仓测试）。

**V3实施展开——具体怎么改**

先确定保留 biased 与 adjusted 哪些变体；公式、min_n、缺失分母、单位分别入 spec，执行全部从一个统计函数分发。注释修改不能代替数值核验。

**V3追加回归——防止只修表面**

不对称三点/四点样本独立手算，再与固定版本参考库核对；无方差及不足3点按定义返回状态。

**V3历史对象与兼容性处置**

既有阈值按对应 skew 变体重新校准，不直接把旧值套入新表。

**关联专项**：[STA-11](#issue-sta-11)。这些专项可在同一根因PR落实。

---

<a id="issue-qe-17"></a>
#### QE-17 · Sortino与Calmar使用了未充分区分的统计变体

**优先级**：P1　**证据分类**：定义／口径冲突　**Owner**：QE

**已阅源码**：[`quant_evaluator/metrics/portfolio_stats.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/quant_evaluator/metrics/portfolio_stats.py)；[`quant_evaluator/metrics/underwater.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/quant_evaluator/metrics/underwater.py)。

**函数/边界**：`compute_sortino_ratio / compute_calmar_ratio / compute_downside_deviation`。

**现象与证据**：Sortino的downside RMS只对负收益样本除以n_negative；Calmar分子是算术均值×年化频数，而不是复合年化收益。二者可定义为变体，但不能和另一标准口径混用。

**影响**：评级、报表和文献比较可能使用不同分母/单位；这不是凭函数名即可认定正确或错误的问题。

**修改要求**：冻结metric definition_id、目标收益、分母和annualization规则；需要时为变体新建明确名称，历史指标不静默变更。

**验收**：
1. 含正收益和负收益混合序列手算两种downside定义。
2. 复利明显偏离算术年化样本分别验证。

**依赖**：按波次/主合同安排，无额外单项依赖。 **需求映射**：R03, R07。

**V3实施展开——具体怎么改**

把 Sortino 的 MAR、负偏差分母和年化方法写成参数；Calmar 分清 CAGR 与算术年化变体。无下行样本的处理必须可见，不能返回虚构超高有限分数。

**V3追加回归——防止只修表面**

同样负收益但加入更多正收益，检验全样本半偏差与条件负样本RMS确实不同；CAGR用独立NAV计算。

**V3历史对象与兼容性处置**

原 ID 若改变统计问题应新版本/新ID，旧报表说明保留，不覆盖旧结果。

**关联专项**：[STA-05](#issue-sta-05)、[STA-11](#issue-sta-11)。这些专项可在同一根因PR落实。

---

<a id="issue-qe-18"></a>
#### QE-18 · 最差月／季实际是滚动21／63期，而不是自然月／季

**优先级**：P1　**证据分类**：定义／口径冲突　**Owner**：QE

**已阅源码**：[`quant_evaluator/metrics/underwater.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/quant_evaluator/metrics/underwater.py)；[`quant_evaluator/registry/metrics.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/quant_evaluator/registry/metrics.py)。

**函数/边界**：`compute_worst_period_return / time aggregation metric IDs`。

**现象与证据**：无日期输入，仅用21/63/252固定长度滑窗却以month/quarter/year展示。

**影响**：无法与自然月风险或跨市场节假日日历一致比较。

**修改要求**：拆成worst_rolling_21d与worst_calendar_month；自然时间指标消费DA CalendarRef和真实时间轴，保存有效样本及不完整期标识。

**验收**：
1. 跨春节/不同月份天数的日期样本得出不同但各自正确的结果。
2. 不得靠截短日历伪造完整年。

**依赖**：按波次/主合同安排，无额外单项依赖。 **需求映射**：R03, R07, R15。

**V3实施展开——具体怎么改**

自然月使用实际日历键分组；rolling_21_periods用连续交易时点滑窗。窗口成熟度和部分月资格分开，不得看结果后决定删不删首尾月。

**V3追加回归——防止只修表面**

不同起始日期但同长度数据，rolling统计一致、自然月分组可能不同；部分月必须带标记。

**V3历史对象与兼容性处置**

报告标题与metric IDs同步改名；持久化日历引用，跨市场不硬套252日。

**关联专项**：[DTA-07](#issue-dta-07)。这些专项可在同一根因PR落实。

---

<a id="issue-qe-19"></a>
#### QE-19 · 3D因子调用时不支持文档承诺的2D公共validity mask

**优先级**：P1　**证据分类**：源码确认缺陷　**Owner**：QE

**已阅源码**：[`quant_evaluator/metrics/portfolio_stats.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/quant_evaluator/metrics/portfolio_stats.py)。

**函数/边界**：`compute_long_short_returns 3D branch`。

**现象与证据**：对validity_mask无条件使用[:,:,f]；传(T,N)共享mask会IndexError。

**影响**：多因子批处理在合法文档输入上失败。

**修改要求**：入口一次性验证并显式broadcast到(T,N,F)，或收紧合同；禁止无检查广播错轴。

**验收**：
1. 2D公共mask与等价3Dmask数值一致。
2. 错误shape和非bool mask有明确错误。

**依赖**：按波次/主合同安排，无额外单项依赖。 **需求映射**：R08, R15。 **隔离复现索引**：PR-07（非全仓测试）。

**V3实施展开——具体怎么改**

在请求规范化处把共享mask按命名轴展开，拒绝shape巧合但轴语义错误的输入。用广播视图不复制大块F数据，但下游不能修改共享buffer。

**V3追加回归——防止只修表面**

2D与显式3D masks逐因子完全一致；数值0/1 mask是否允许由合同决定，不能静默接受任意浮点。

**V3历史对象与兼容性处置**

规范化结果进入请求hash与设备staging，原始形状可作审计字段。

---

<a id="issue-qe-20"></a>
#### QE-20 · 通用多空分桶可在平局因子上同时买卖同一组

**优先级**：P1　**证据分类**：源码确认缺陷　**Owner**：QE

**已阅源码**：[`quant_evaluator/metrics/portfolio_stats.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/quant_evaluator/metrics/portfolio_stats.py)。

**函数/边界**：`compute_long_short_returns quantile cutoff`。

**现象与证据**：独立np.quantile后用>=和<=，常数/严重ties情况下long与short可重合；没有统一的disjoint与阈值顺序检查。

**影响**：不可识别的因子被报告成零收益的有效多空组合；与QE canonical quantile kernel语义重复。

**修改要求**：复用统一QuantileTiePolicy，明确离散因子可用层数；多空集合必须不交，退化情况INSUFFICIENT_DATA；检查short<long且区间合法。

**验收**：
1. 常数、二元、部分ties、反转阈值。
2. 公开API与CPU/GPU/probe同桶。

**依赖**：按波次/主合同安排，无额外单项依赖。 **需求映射**：R04, R08。 **隔离复现索引**：PR-08（非全仓测试）。

**V3实施展开——具体怎么改**

组合与分层统一使用已有 QuantileTiePolicy；定义离散因子的可行层数及是否允许部分桶空。long/short 目标集合先验证 disjoint，然后执行侧别可交易过滤并再次检查人数。

**V3追加回归——防止只修表面**

常数因子不产生两侧同一证券；二元因子按两层/不适用处理；两阈值错序必须在计算前拒绝。

**V3历史对象与兼容性处置**

退化历史组合不能继续冒充有效零收益策略，应带失效原因。

---

<a id="issue-qe-21"></a>
#### QE-21 · drop缺失收益在形成桶之前排除未来不可得标签

**优先级**：P0　**证据分类**：定义／口径冲突　**Owner**：QE

**已阅源码**：[`quant_evaluator/metrics/portfolio_stats.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/quant_evaluator/metrics/portfolio_stats.py)。

**函数/边界**：`compute_long_short_returns missing_return_policy=drop`。

**现象与证据**：先把future return非finite资产从finite_mask剔除，再计算分位门槛，组合成员因此依赖未来收益可用性。

**影响**：若用于可交易组合，相当于利用未来缺失信息重选股票；与合法的事后诊断pairwise deletion混淆。

**修改要求**：交易成员由决策时可见的universe/factor mask决定；持仓后缺价格单独估值/延迟退出/失效规则。诊断统计可joint-mask但必须显式标为诊断，不能宣称可执行收益。

**验收**：
1. 只改变未来某只股票标签是否缺失，不得改变当时的持仓目标。
2. 诊断与执行模式返回不同类型。

**依赖**：按波次/主合同安排，无额外单项依赖。 **需求映射**：R07, R15。

**V3实施展开——具体怎么改**

分开 signal_eligibility 与 ex_post_label_validity。交易目标只读决策可得字段；评价统计可以联合mask但须标诊断。持仓形成后缺价通过估值政策处理，不能回头重新选股票。

**V3追加回归——防止只修表面**

仅改变未来标签缺失位置，历史目标持仓及分桶必须不变；诊断统计的有效样本数可以随之变化并披露。

**V3历史对象与兼容性处置**

此类旧策略收益应按可交易集合重放，不能只改缺失率指标。

**关联专项**：[DTA-04](#issue-dta-04)。这些专项可在同一根因PR落实。

---

<a id="issue-qe-22"></a>
#### QE-22 · 成本函数未拒绝负换手，能产生“交易返利”

**优先级**：P1　**证据分类**：源码确认缺陷　**Owner**：QE

**已阅源码**：[`quant_evaluator/metrics/portfolio_stats.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/quant_evaluator/metrics/portfolio_stats.py)。

**函数/边界**：`apply_long_short_costs`。

**现象与证据**：只校验数组shape和cost_rate，负或非finite long_turnover/short_turnover可直接进入收益公式。

**影响**：坏输入可提高net收益并通过成本门槛。

**修改要求**：校验turnover语义与finite/nonnegative，允许未知成本时返回UNKNOWN而非0；严格注明单边/双边名义额和费用单位。

**验收**：
1. 负换手/NaN换手/错bps单位拒绝。
2. 两侧费用手算。
3. gross与net差额可对账。

**依赖**：按波次/主合同安排，无额外单项依赖。 **需求映射**：R06, R07。 **隔离复现索引**：PR-22（非全仓测试）。

**V3实施展开——具体怎么改**

费用计算消费有类型 NotionalTurnover 与 CostSpec；先校验非负有限值和单边定义，再扣除。市场合法返佣只能由独立 fee component 表达，绝不是允许负换手。

**V3追加回归——防止只修表面**

负/NaN/Inf换手拒绝；在同一固定成交轨迹下调高费用，净PnL不得提高。

**V3历史对象与兼容性处置**

历史出现成本为负的样本隔离；不得一律取abs以掩盖上游错误。

---

<a id="issue-qe-23"></a>
#### QE-23 · cohort在下一日VWAP入场，却赚到了入场前的涨跌

**优先级**：P0　**证据分类**：源码确认缺陷　**Owner**：QE

**已阅源码**：[`quant_evaluator/metrics/probe_portfolio/_core.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/quant_evaluator/metrics/probe_portfolio/_core.py)；[`quant_evaluator/kernels/gpu/portfolio.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/quant_evaluator/kernels/gpu/portfolio.py)。

**函数/边界**：`compute_cohort_pnl / compute_cohort_pnl_batch_gpu`。

**现象与证据**：声明daily_returns[t]=P_t/P_(t-1)-1、entry=s+1；实际循环从t=entry起乘新仓位。因而新仓享有P_s→P_(s+1)的入场前收益。

**影响**：前视/时间错位可系统性高估PnL；holding=1且同一VWAP进出仍可能产生收益。

**修改要求**：以真实entry/exit quote定义持有区间，收益区间只能在入场后；统一H是跨价区间数还是日期数，并改CPU/GPU、标签生成与spec。禁止只改注释掩盖错位。

**验收**：
1. 只有入场前价格跳变、入场后不变时毛PnL=0。
2. H=1/2/20手算价格路径。
3. CPU/GPU和label持有期限一致。

**依赖**：按波次/主合同安排，无额外单项依赖。 **需求映射**：R07, R15。 **隔离复现索引**：PR-23（非全仓测试）。

**V3实施展开——具体怎么改**

冻结“收益区间数量”的H定义：在价格P_e成交，首个完整持仓区间应从P_e到下个可估值价格；入场前区间归旧仓。CPU循环、GPU cumsum边界和标签builder同时使用同一调度函数。

**V3追加回归——防止只修表面**

价格100→110→110，在110入场且110退出，毛PnL=0；H=1定义为一个入场后区间，不能又把它解释同价同日进出。

**V3历史对象与兼容性处置**

保留原始出错回测作审计；不覆盖旧test再宣称全新样本外通过。

**关联专项**：[DTA-03](#issue-dta-03)、[QAT-02](#issue-qat-02)。这些专项可在同一根因PR落实。

---

<a id="issue-qe-24"></a>
#### QE-24 · 同日进出只扣单边成本，末尾未成熟cohort被强平

**优先级**：P1　**证据分类**：定义／口径冲突　**Owner**：QE

**已阅源码**：[`quant_evaluator/metrics/probe_portfolio/_core.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/quant_evaluator/metrics/probe_portfolio/_core.py)；[`quant_evaluator/kernels/gpu/portfolio.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/quant_evaluator/kernels/gpu/portfolio.py)。

**函数/边界**：`same_day / exit_t=min(...,T-1) / exit_cost`。

**现象与证据**：CPU same_day分支跳过exit fee；GPU holding=1不算exit fee；未到期cohort在数据末尾截断退出。

**影响**：短期限成本和测试期尾部收益依赖样本截断；换个评估结束日结果可能不可比。

**修改要求**：明确no-trade同日政策或两笔实际成交收费；未成熟尾部与可选清算情景分开，sealed评价使用一致成熟样本。

**验收**：
1. 固定价格只交易成本时两侧对账。
2. 同日no-trade为0。
3. 延长数据不改变已完成cohort结果。

**依赖**：QE-23。 **需求映射**：R07, R15。

**V3实施展开——具体怎么改**

把scheduled_exit、observed_exit、terminal_valuation与forced_liquidation区分。样本截断不自动生成成交；单日真实买卖有两笔手续费，无交易决策则两笔都没有。

**V3追加回归——防止只修表面**

延长数据不会改变已完成cohort的价格/费用；对仍未退出的cohort只更新估值与成熟状态。

**V3历史对象与兼容性处置**

旧尾部强平收益与完整持有收益不可混作同一评价版本。

---

<a id="issue-qe-25"></a>
#### QE-25 · probe组合与“可执行实盘PnL”的声明不匹配

**优先级**：P0　**证据分类**：定义／口径冲突　**Owner**：QE

**已阅源码**：[`quant_evaluator/metrics/probe_portfolio/_core.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/quant_evaluator/metrics/probe_portfolio/_core.py)；[`quant_evaluator/kernels/gpu/portfolio.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/quant_evaluator/kernels/gpu/portfolio.py)。

**函数/边界**：`cohort weights / tradability / netting`。

**现象与证据**：持有期每日复用初始权重，未计维持等权所需的再平衡；只要一侧还有股票就运行；桶人数在交易过滤后未重新守门；多cohort成本按gross逐个累加。

**影响**：可能从多空变为单边；固定权重代理与买入持有、净额成交不能等同。不能用它直接认证实盘容量或净风险。

**修改要求**：保留快速研究probe但显式probe-only；正式执行评估复用已有回测/执行域：股数与现金漂移、净额调仓、侧别可成交、借券、停牌估值，禁止再造无约束PnL轮子。

**验收**：
1. 一侧不可成交时按政策拒绝/降级而非假中性。
2. 买持与每日再平衡手算差异。
3. 跨cohort相抵时净成交成本正确。

**依赖**：QE-23, QE-22。 **需求映射**：R06, R07, R15。

**V3实施展开——具体怎么改**

查找已有回测/执行库而不是另写一个交易引擎。快速probe保留固定假设和标签，正式路径消费股数/现金/成交/借券记录；日间权重漂移与再平衡成交不可同时忽略。

**V3追加回归——防止只修表面**

对冲后净成交与各cohort毛成交对账；买持和每日等权再平衡产生不同成本；一侧无券时不能仍标market-neutral。

**V3历史对象与兼容性处置**

probe artifact禁止被生产容量门当实盘证据；并非删除probe，而是限制用途。

**关联专项**：[DTA-03](#issue-dta-03)、[DTA-05](#issue-dta-05)。这些专项可在同一根因PR落实。

---

<a id="issue-qe-26"></a>
#### QE-26 · 日度风险链存在多个不相容的缺失/零填充默认

**优先级**：P1　**证据分类**：已见接口缺口／局部风险　**Owner**：QE

**已阅源码**：[`quant_evaluator/metrics/portfolio_stats.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/quant_evaluator/metrics/portfolio_stats.py)；[`quant_evaluator/metrics/underwater.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/quant_evaluator/metrics/underwater.py)；[`quant_evaluator/metrics/probe_portfolio/_core.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/quant_evaluator/metrics/probe_portfolio/_core.py)；[`quant_evaluator/kernels/gpu/portfolio.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/quant_evaluator/kernels/gpu/portfolio.py)。

**函数/边界**：`return missing policies`。

**现象与证据**：旧portfolio/cohort常把missing return当0；underwater删除日期；aligned_wealth保留gap。单个默认有说明，但组合使用时证据可不一致。

**影响**：同一因子的收益、回撤、水下期、coverage可基于不同样本；停牌/退市损失被隐藏。

**修改要求**：用一个PortfolioMissingnessPolicy决定各类missing-reason和估值状态，所有派生风险引用同一NAV artifact；研究默认不能自动升级为生产。

**验收**：
1. 持仓缺报价、退市、无仓位、数据故障分别测试。
2. 不同风险指标共享同一有效路径和coverage。

**依赖**：QE-14, QE-25。 **需求映射**：R07, R15。

**V3实施展开——具体怎么改**

形成一个估值政策对象：无持仓、休市、停牌、行情丢包、终止上市分别决策。NAV与所有派生指标引用相同policy hash和已决估值事件；不能每个指标自行fill/drop。

**V3追加回归——防止只修表面**

同一估值路径的Sharpe、回撤、尾部样本覆盖可对账；行情故障不被写成真实零收益。

**V3历史对象与兼容性处置**

全链统一缺失政策后，旧默认为zero_fill的风险卡须注明旧口径。

**关联专项**：[DTA-04](#issue-dta-04)、[DTA-08](#issue-dta-08)。这些专项可在同一根因PR落实。

---

<a id="issue-qe-27"></a>
#### QE-27 · 指标版本与观测数量在公共证据里被错误简化

**优先级**：P1　**证据分类**：源码确认缺陷　**Owner**：QE

**已阅源码**：[`quant_evaluator/runtime/evaluator.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/quant_evaluator/runtime/evaluator.py)；[`quant_evaluator/registry/metrics.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/quant_evaluator/registry/metrics.py)。

**函数/边界**：`_per_factor_observation_counts / metric_versions`。

**现象与证据**：大量非IC指标把finite标量计为1个观测；metric_versions写死0.1，而注册表已有1.0.0；观测计数还重复计算IC。

**影响**：置信度、样本门槛和版本血缘误导；生产无法可靠复现当时指标定义。

**修改要求**：每个metric artifact携带n_time、n_asset、n_joint、n_episode等有类型计数；版本/实现hash来自有效注册快照，计数来自共享中间证据。

**验收**：
1. 同值不同有效样本数不得得到相同证据计数。
2. 版本变化必须传播到健康卡和缓存。
3. mean_ic与rank_ic计数基准有明确名称。

**依赖**：按波次/主合同安排，无额外单项依赖。 **需求映射**：R03, R14。

**V3实施展开——具体怎么改**

从实际sealed registry读取metric version/implementation hash，kernel或builder返回count artifact。禁止通过最终标量是否finite推断n=1；计数单位必须区分时间、证券、联合格、episode。

**V3追加回归——防止只修表面**

同一数值但20/200个有效IC日的证据不可相同；改metric实现版本使健康卡引用与cache key都变化。

**V3历史对象与兼容性处置**

证据schema增加计数单位后，旧不明count应标未知而非重解释。

**关联专项**：[STA-02](#issue-sta-02)、[STA-05](#issue-sta-05)。这些专项可在同一根因PR落实。

---

<a id="issue-qe-28"></a>
#### QE-28 · 评估DAG未充分复用，批处理和预算在公共入口失效

**优先级**：P1　**证据分类**：已见接口缺口／局部风险　**Owner**：QE

**已阅源码**：[`quant_evaluator/runtime/evaluator.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/quant_evaluator/runtime/evaluator.py)。

**函数/边界**：`evaluate IC wrappers / _cache_identity / runtime.evaluate`。

**现象与证据**：每个IC派生wrapper重算daily IC，obs计数又算一次；逐metric hash全量数组；facade强制use_chunking=False；request cost_budget只入metadata未施加到默认runtime。

**影响**：批量因子耗时和内存远超必要；设置预算不代表执行真正受限。

**修改要求**：请求级只计算一次输入身份；共享mask-aware rank/IC/quantile/PnL；把预算传入planner与执行器，支持按F批次和有数学归并合同的T流式，记录actual资源。

**验收**：
1. 多个IC族请求IC builder一次。
2. 预算0/极小预算确实阻断。
3. chunk与full parity。
4. hash时间和数据传输计入telemetry。

**依赖**：QE-02, QE-03。 **需求映射**：R08, R09。

**V3实施展开——具体怎么改**

请求规划阶段估算共享子图与真实资源预算，输入buffer hash一次计算。按F批执行共享日历和标签；耗时计数包含hash/I/O及失败。不允许代价参数只保存在metadata。

**V3追加回归——防止只修表面**

零预算请求应按政策不执行；3个IC派生项只跑一次IC；各分块与全量结果一致且缓存命中不会丢状态。

**V3历史对象与兼容性处置**

相同数学定义可以复用已验证中间artifact；前提是完整上下文一致。

**关联专项**：[STA-01](#issue-sta-01)、[SRH-05](#issue-srh-05)、[OPS-02](#issue-ops-02)。这些专项可在同一根因PR落实。

---

<a id="issue-qe-29"></a>
#### QE-29 · 低层评估与标签合同缺少完整坐标绑定

**优先级**：P0　**证据分类**：已见接口缺口／局部风险　**Owner**：QE

**已阅源码**：[`quant_evaluator/runtime/evaluator.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/quant_evaluator/runtime/evaluator.py)；[`quant_evaluator/contracts/label_bundle.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/quant_evaluator/contracts/label_bundle.py)。

**函数/边界**：`Evaluator._validate_inputs / LabelBundle`。

**现象与证据**：低层只比时间长度；LabelBundle无显式asset-axis字段，公共入口主要验证资产数量，不能仅凭shape证明证券顺序相同。

**影响**：标签资产顺序颠倒仍可形状合法；相同长度的不同日期也可能进入旁路。

**修改要求**：统一AxisRef或明确共享有hash的轴合同；入参完整性验证一次后所有路径消费validated request；禁止自动按位置猜测对齐，显式重排必须记录。

**验收**：
1. 同shape但调换asset IDs或time IDs被拒。
2. 合法显式align与手工golden一致。
3. 低层/高层/CPU/GPU守门相同。

**依赖**：按波次/主合同安排，无额外单项依赖。 **需求映射**：R15。

**V3实施展开——具体怎么改**

给所有面板绑定date/instrument AxisRef；严格校验唯一性、排序和同一坐标集合。显式align返回置换记录，不能为了通过shape检查自动inner join而静默丢资产。

**V3追加回归——防止只修表面**

同shape错证券、相同日期不同顺序、重复证券和重复时点均测试；正确显式重排不改变金融结果。

**V3历史对象与兼容性处置**

补全不能恢复的历史轴时记录数据不充分，不能猜证券顺序。

**关联专项**：[DTA-01](#issue-dta-01)、[DTA-09](#issue-dta-09)、[DTA-12](#issue-dta-12)。这些专项可在同一根因PR落实。

---

<a id="issue-qe-30"></a>
#### QE-30 · decision_time与signal_available_time的含义和不等式需要纠正

**优先级**：P0　**证据分类**：定义／口径冲突　**Owner**：QE

**已阅源码**：[`quant_evaluator/contracts/label_bundle.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/quant_evaluator/contracts/label_bundle.py)。

**函数/边界**：`LabelBundle.__post_init__ causal chain`。

**现象与证据**：代码要求decision_time<=signal_available_time。若decision表示实际作出交易决定的时刻，则应先有信号可用；若它表示数据观察bar时间，字段命名和消费语义就不一致。

**影响**：系统可能允许“决定时尚不可见的数据”，或误拒合法滞后决策；不能在未经语义确认时机械翻转比较符。

**修改要求**：分别定义observation_time、knowledge_time、signal_available、decision、execution和label interval；实际决策要求signal_available<=decision<=execution。使用bar timestamp的旧接口显式迁移映射。

**验收**：
1. 收盘后可用信号只能次日指定窗口成交。
2. 公告晚于decision拒绝。
3. 同日合法截面处理不额外盲shift。
4. 时区/交易日跨界测试。

**依赖**：按波次/主合同安排，无额外单项依赖。 **需求映射**：R15。

**V3实施展开——具体怎么改**

先调查旧decision字段的实际使用者，确认是观察bar还是交易决策，再迁移字段；实际决策链signal_available<=decision<=execution。输入数据发布时间、入库时间和修订可见性分别保留。

**V3追加回归——防止只修表面**

当天收盘后才可见信号不能成交于此前全天VWAP；晚到公告不得参与此前决策；合法过去lookback不误拦。

**V3历史对象与兼容性处置**

时钟语义是schema升级，必须迁移FO split、DA label与FE availability，不能只反转不等号。

**关联专项**：[DTA-02](#issue-dta-02)、[DTA-07](#issue-dta-07)。这些专项可在同一根因PR落实。

---

<a id="issue-qe-31"></a>
#### QE-31 · 标签快照可被外部修改，chunk与cache又遗漏关键字段

**优先级**：P1　**证据分类**：源码确认缺陷　**Owner**：QE

**已阅源码**：[`quant_evaluator/contracts/label_bundle.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/quant_evaluator/contracts/label_bundle.py)；[`quant_evaluator/runtime/evaluator.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/quant_evaluator/runtime/evaluator.py)。

**函数/边界**：`LabelBundle / _extract_chunk_labels / _cache_identity`。

**现象与证据**：frozen LabelBundle仍保留可写ndarray和dict；chunk不传signal_available_time与price_convention使其回默认；cache label identity也未显式包括这两项。

**影响**：同名证据或缓存可能对应不同可用时刻/收益口径；运行中改buffer导致不可复现。

**修改要求**：冻结或租约保护buffer与深冻结metadata；完整复制/切片合同字段；缓存键包含整个规范label身份，不散列手写少数字段。

**验收**：
1. 外部原数组和metadata修改不影响已封存证据。
2. chunk往返保留clock/basis。
3. 只变basis或available time必须失效缓存。

**依赖**：QE-29, QE-30。 **需求映射**：R14, R15。

**V3实施展开——具体怎么改**

以规范label内容身份统一slice/serialize/cache，避免每个函数手抄字段。采用只读buffer或所有权租约，深冻结metadata；COW切片不得把可写view泄给调用方。

**V3追加回归——防止只修表面**

单改available time/basis/universe使身份改变；chunk重新合并保留完整时间链；外部原数组修改不污染已冻结请求。

**V3历史对象与兼容性处置**

旧cache键覆盖不足的缓存需要有界失效，不得升级程序后复用。

**关联专项**：[DTA-02](#issue-dta-02)、[DTA-12](#issue-dta-12)。这些专项可在同一根因PR落实。

---

<a id="issue-qe-32"></a>
#### QE-32 · 预测期限衰减指标实际测的是IC序列自相关

**优先级**：P1　**证据分类**：定义／口径冲突　**Owner**：QE

**已阅源码**：[`quant_evaluator/registry/metrics.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/quant_evaluator/registry/metrics.py)；[`quant_evaluator/metrics/ic_summary.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/quant_evaluator/metrics/ic_summary.py)。

**函数/边界**：`rank_ic_decay_h01_h05_h10_h20`。

**现象与证据**：该ID注册为IC序列在1/5/10/20滞后上的自相关均值，不是因子对四种前向收益期限的预测IC。

**影响**：优化器可能用错误的“半衰/期限”信息选择窗口或滤波强度。

**修改要求**：将IC serial autocorrelation与predictive IC across horizons分为不同metric IDs；后者消费同决策轴的多LabelBundles并绑定各自成熟样本。

**验收**：
1. 构造序列稳定但只预测短期的因子，两个指标不能混同。
2. H1/H5/H10/H20曲面保留向量而非仅均值。

**依赖**：按波次/主合同安排，无额外单项依赖。 **需求映射**：R03, R06, R10。

**V3实施展开——具体怎么改**

将IC序列ACF、信号自身rank持久性、预测horizon曲面三个概念分别命名。horizon曲面从多份label输入计算，所有目标有自己的maturity与共同比较样本说明。

**V3追加回归——防止只修表面**

序列IC稳定但预测只在H1有效的样本，ACF与预测衰减输出必须不同；保留整条曲线与误差。

**V3历史对象与兼容性处置**

依靠旧错误衰减选择的滤波/窗口需要再评价，不能仅更换图名。

**关联专项**：[DTA-11](#issue-dta-11)、[STA-07](#issue-sta-07)、[STA-10](#issue-sta-10)。这些专项可在同一根因PR落实。

---

<a id="issue-qe-33"></a>
#### QE-33 · 形状稳定性输出Fisher-z而非声明的相关性尺度

**优先级**：P1　**证据分类**：源码确认缺陷　**Owner**：QE

**已阅源码**：[`quant_evaluator/metrics/shape_evidence.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/quant_evaluator/metrics/shape_evidence.py)。

**函数/边界**：`compute_shape_stability / compute_shape_regime_stability`。

**现象与证据**：返回mean(arctanh(r))未tanh逆变换；一致剖面输出约10.7082而非文档的1。

**影响**：评级阈值及稳定性解释错误。

**修改要求**：确定输出尺度；如是相关性使用tanh(mean(arctanh(r)))并版本迁移；固定训练参考或leave-one-window参考避免自包含比较偏乐观。

**验收**：
1. 重复剖面相关性尺度为1附近。
2. 反向剖面负值。
3. 单窗口INSUFFICIENT_DATA。
4. registry单位与等级阈值一致。

**依赖**：按波次/主合同安排，无额外单项依赖。 **需求映射**：R03, R04。 **隔离复现索引**：PR-13（非全仓测试）。

**V3实施展开——具体怎么改**

输出范围由metric spec固定；采用相关性尺度时最后做tanh，不直接把Fisher-z送进0..1阈值。参考剖面取训练冻结或留一窗，是否纳入自身必须明示。

**V3追加回归——防止只修表面**

重复剖面、方向相反、单窗口、零方差剖面与稀疏缺失分别测试；范围断言和显示单位一致。

**V3历史对象与兼容性处置**

更新所有消费该阈值的diagnosis规则与health policy。

---

<a id="issue-qe-34"></a>
#### QE-34 · 形状confidence与asymmetry指标名超出实际计算含义

**优先级**：P1　**证据分类**：定义／口径冲突　**Owner**：QE

**已阅源码**：[`quant_evaluator/metrics/shape_evidence.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/quant_evaluator/metrics/shape_evidence.py)。

**函数/边界**：`compute_shape_bootstrap_confidence / compute_left_right_asymmetry`。

**现象与证据**：bootstrap计算窗口抽样剖面秩相似度>=.5的频率，不是U型成立概率；asymmetry公式(top-mid)-(mid-bottom)对对称U也可大于0，衡量的包含曲率而非纯左右不对称。

**影响**：Agent/规则会把描述性统计当成概率证据，或把对称U误判为偏斜。

**修改要求**：拆开形状类型检验、模板拟合度、bootstrap稳定性及左右mirror contrast；保留原统计需改名、说明，重叠窗口block resampling；统一ties与有效mask。

**验收**：
1. 对称U的左右不对称度=0。
2. U/倒U/单调/噪声/偏斜U golden。
3. bootstrap只对其定义事件标confidence。

**依赖**：按波次/主合同安排，无额外单项依赖。 **需求映射**：R04, R17。 **隔离复现索引**：PR-14（非全仓测试）。

**V3实施展开——具体怎么改**

纯左右不对称使用镜像分层对比，不将曲率当偏斜；bootstrap需先声明事件及block方案。U型显著性、模板R²和重采样稳定率单独存储，Agent不得叫作成功概率。

**V3追加回归——防止只修表面**

对称U、偏斜U和对称倒U分别验证mirror指标；换中心拟合必须按训练fold重新进行。

**V3历史对象与兼容性处置**

旧confidence仅可保留原描述，不能无说明升级为推断概率。

**关联专项**：[STA-03](#issue-sta-03)。这些专项可在同一根因PR落实。

---

<a id="issue-qe-35"></a>
#### QE-35 · 自适应分层合同存在边界洞，人数不能代替实际桶可用性

**优先级**：P1　**证据分类**：已见接口缺口／局部风险　**Owner**：QE

**已阅源码**：[`quant_evaluator/contracts/adaptive_bins_policy.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/quant_evaluator/contracts/adaptive_bins_policy.py)。

**函数/边界**：`resolve_bin_count / AdaptiveBinsPolicy`。

**现象与证据**：部分入口按int截断计数并丢负数；无可行层数时min(fallback_bins)对空fallback可能报错；总人数门槛无法单独保证ties后的每层有效人数。

**影响**：离散、稀疏因子可能得到不可用20层；一个异常日期可拖低全窗层数且缺少明确coverage解释。

**修改要求**：严格整数非bool和非负验证；preferred/fallback有序合法；用实际每桶effective count和distinct levels决定是否降级/日期不适用，记录固定比较政策。

**验收**：
1. 空fallback、零资产、负数、float/bool计数拒绝或明确状态。
2. 2000只二元因子不能强行20层。
3. 1个缺口日期不隐式丢弃。

**依赖**：按波次/主合同安排，无额外单项依赖。 **需求映射**：R04, R08。

**V3实施展开——具体怎么改**

策略配置严格拒绝bool/非整数计数、重复或无序备选Q；对每个因子每日期检查distinct与实际bucket有效人数。冻结比较Q或明确逐日降级规则，不能跨Q无说明聚合曲线。

**V3追加回归——防止只修表面**

空fallback可选择preferred或返回不足，不能min(empty)崩溃；2000只二元值不能生成20个真实有辨识度层。

**V3历史对象与兼容性处置**

保存actual Q与降级原因，旧只记preferred=20的图须重新验证。

**关联专项**：[RCP-08](#issue-rcp-08)。这些专项可在同一根因PR落实。

---

### 11.2 FP：配方、参数、算子复用与最终性质

<a id="issue-fp-01"></a>
#### FP-01 · 整数证券代码经FE适配后变成全空列

**优先级**：P0　**证据分类**：源码确认缺陷　**Owner**：FP/FE

**已阅源码**：[`factor_preprocess/factor_preprocess/adapters/fe_operator.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/factor_preprocess/factor_preprocess/adapters/fe_operator.py)。

**函数/边界**：`_long_to_wide`。

**现象与证据**：列轴先astype(str)，pivot columns仍是原int类型，随后reindex字符串列导致全部NaN。

**影响**：正常因子变成无效数据；自动筛选误判覆盖和预测性。

**修改要求**：由DA规范证券键，输入和轴同时无损规范化；禁止随意str合并不同身份；所有列对齐采用显式AxisRef。

**验收**：
1. int/string/带前导零/混合类型证券键。
2. 不能悄悄生成全NaN。
3. 规范化前后identity一一对应。

**依赖**：按波次/主合同安排，无额外单项依赖。 **需求映射**：R14, R15。 **隔离复现索引**：PR-01（非全仓测试）。

**V3实施展开——具体怎么改**

规范ID由DA提供；adapter仅执行无损映射。数字1、字符串1、前导零代码及不同市场代码是否同一证券由catalog决定，不准随意str等同；pivot前先验证规范ID唯一。

**V3追加回归——防止只修表面**

对字符串/整数/带市场前缀ID运行真实get_execution；输入有限值数量在有效定义下不应无故归零。

**V3历史对象与兼容性处置**

之前整批全NaN的误淘汰候选需要可控重评，不能永久seen-skip。

**关联专项**：[DTA-01](#issue-dta-01)。这些专项可在同一根因PR落实。

---

<a id="issue-fp-02"></a>
#### FP-02 · 重复日期证券记录被first静默吞掉

**优先级**：P0　**证据分类**：源码确认缺陷　**Owner**：FP/FE

**已阅源码**：[`factor_preprocess/factor_preprocess/adapters/fe_operator.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/factor_preprocess/factor_preprocess/adapters/fe_operator.py)。

**函数/边界**：`_long_to_wide pivot_table`。

**现象与证据**：pivot_table(aggfunc="first")对同(date,asset)多值不拒绝。

**影响**：输入顺序可决定因子值；上游重复join被掩盖。

**修改要求**：默认验证主键唯一；确切相同重复可按显式去重政策处理并记录，冲突重复必须阻断。

**验收**：
1. 同键1与9换行序不得产生两个不同且都COMPUTED的结果。
2. 冲突报带键诊断。
3. 全NaN行也不被悄悄删除。

**依赖**：按波次/主合同安排，无额外单项依赖。 **需求映射**：R15。 **隔离复现索引**：PR-02（非全仓测试）。

**V3实施展开——具体怎么改**

长表入口先检查(date,instrument)主键；仅完全相同重复可走显式去重策略，冲突返回样本键与来源引用。避免pivot_table隐式聚合，同时保存全NaN键的轴存在性。

**V3追加回归——防止只修表面**

互换冲突行順序都应拒绝；重复等值行需记录dedup_count；没有冲突的行乱序不改结果。

**V3历史对象与兼容性处置**

修复上游join后重算受影响落值；绝不能用排序固定first掩盖冲突。

**关联专项**：[DTA-09](#issue-dta-09)。这些专项可在同一根因PR落实。

---

<a id="issue-fp-03"></a>
#### FP-03 · FP→FE参数映射遗漏，实际变换不等于记录配方

**优先级**：P0　**证据分类**：源码确认缺陷　**Owner**：FP/FE

**已阅源码**：[`factor_preprocess/factor_preprocess/adapters/fe_operator.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/factor_preprocess/factor_preprocess/adapters/fe_operator.py)。

**函数/边界**：`FeOperatorExecutor.__call__`。

**现象与证据**：只转发手写scalar_kw；FP forward_fill的max_lag没有映射到FE max_periods，rank的method/pct等可被忽略；除values外位置参数也未正常绑定。

**影响**：两种不同配方落成同一值，甚至填充长度失控；参数寻优等同“假搜索”。

**修改要求**：按canonical operator metadata与显式版本化映射绑定完整signature；未知参数拒绝，绝不静默丢弃；存requested/effective params一致性证据。

**验收**：
1. 通过registry.get_execution("forward_fill")(max_lag=1/5)验证差异，而非手动改FE参数名。
2. rank pct=False/tie method参数可追踪。
3. 多余参数拒绝。

**依赖**：按波次/主合同安排，无额外单项依赖。 **需求映射**：R06, R09, R14。 **隔离复现索引**：PR-03（非全仓测试）。

**V3实施展开——具体怎么改**

以inspect.signature.bind或等价完整合同绑定args/kwargs，再按版本化映射转换到FE名称；先apply_defaults再对比requested/effective参数，未知参数拒绝。所有被搜索参数必须能证明到达有效执行分支。

**V3追加回归——防止只修表面**

用FP参数名而不是FE名测试ffill max_lag=1/5；非默认rank method、pct、window和位置参数均测试。

**V3历史对象与兼容性处置**

新增effective_params_hash并使参数映射版本进入recipe/value identity。

**关联专项**：[RCP-03](#issue-rcp-03)、[RCP-10](#issue-rcp-10)、[SRH-01](#issue-srh-01)、[QAT-04](#issue-qat-04)。这些专项可在同一根因PR落实。

---

<a id="issue-fp-04"></a>
#### FP-04 · 存在绕过FE的公开执行口，fallback吞掉所有加载错误

**优先级**：P0　**证据分类**：已见接口缺口／局部风险　**Owner**：FP/FE

**已阅源码**：[`factor_preprocess/factor_preprocess/adapters/fe_operator.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/factor_preprocess/factor_preprocess/adapters/fe_operator.py)；[`factor_preprocess/factor_preprocess/registry/transforms.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/factor_preprocess/factor_preprocess/registry/transforms.py)。

**函数/边界**：`get_function / get_execution / get_fe_executor`。

**现象与证据**：get_function直接返回FP native；get_execution尝试FE但多层except Exception回退，未区分依赖缺失与已安装实现错误，也未重绑实际执行身份。

**影响**：同一recipe不同环境可能执行两套数学；自称FE已复用但主链仍走native。

**修改要求**：生产只暴露一个受约束executor，reference函数显式命名research-only；fallback必须受policy允许、记录原因与有效implementation hash，严格模式拒绝。

**验收**：
1. 禁用FE算子/制造加载异常严格模式均失败。
2. 研究fallback明确显示。
3. 主链不得调用get_function绕行。
4. 输出记录effective executor。

**依赖**：按波次/主合同安排，无额外单项依赖。 **需求映射**：R08, R14, R15。

**V3实施展开——具体怎么改**

列出get_function、直接import native、factory fallback全部调用者。生产组合根只接收FE可信executor；研究reference可保留，但明确拒绝生产且记录来源。只对声明的依赖缺失允许研究fallback。

**V3追加回归——防止只修表面**

人为让已安装FE加载抛逻辑错误，不得悄悄转native；strict模式缺FE拒绝，研究fallback清楚记录实现。

**V3历史对象与兼容性处置**

迁移所有使用旧绕行入口的modeling/lightgbm_qs/jobs，不只改registry。

**关联专项**：[QAT-07](#issue-qat-07)。这些专项可在同一根因PR落实。

---

<a id="issue-fp-05"></a>
#### FP-05 · FE桥仍是固定CPU长宽表转换，不是GPU批量执行链

**优先级**：P2　**证据分类**：已见接口缺口／局部风险　**Owner**：FP/FE

**已阅源码**：[`factor_preprocess/factor_preprocess/adapters/fe_operator.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/factor_preprocess/factor_preprocess/adapters/fe_operator.py)。

**函数/边界**：`_ensure / _stack_back`。

**现象与证据**：_ensure固定pandas_numpy；_stack_back按证券和日期建立Python字典再逐行回写，每步重复pivot/stack。

**影响**：预处理批量成本和数据搬运抵消FE/GPU收益。

**修改要求**：先对齐一次统一面板，整条recipe编译到同一FE执行会话；仅在边界做长宽转换；GPU/CPU能力从算子注册表选择，记录fallback及传输成本。

**验收**：
1. 连续3个变换不重复pivot3次。
2. F批次内存受预算限制。
3. 输出重排正确。
4. 端到端基准含转换和I/O。

**依赖**：FP-01, FP-03, FP-04。 **需求映射**：R08, R14。

**V3实施展开——具体怎么改**

统一长表到面板转换只在pipeline边界一次执行；整条recipe下推一个FE会话，保持设备驻留并复用中间值。禁止每个step新建DataFrame循环返回。

**V3追加回归——防止只修表面**

3-step配方pivot次数固定为边界次数；证券乱序与NaN回写无错位；端到端计入转换耗时。

**V3历史对象与兼容性处置**

在正确性修复之后优化；不要以不同mask或低精度改变结果来获得速度。

**关联专项**：[OPS-02](#issue-ops-02)。这些专项可在同一根因PR落实。

---

<a id="issue-fp-06"></a>
#### FP-06 · 暴露矩阵输入靠“所有非键列”猜测，缺少显式schema与轴

**优先级**：P1　**证据分类**：已见接口缺口／局部风险　**Owner**：FP/FE

**已阅源码**：[`factor_preprocess/factor_preprocess/adapters/fe_operator.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/factor_preprocess/factor_preprocess/adapters/fe_operator.py)。

**函数/边界**：`FeOperatorExecutor exposures branch`。

**现象与证据**：exposure frame除time/asset外的每列均转为因子面板，各自pivot；没有在这个边界强制exposure IDs/顺序/dtype/PIT/mask与signal轴一致。

**影响**：元数据列可误当风险因子，暴露错配或未来信息混入；是否被上游拦截需要查调用者。

**修改要求**：消费DA ExposureBundle，列由ExposureSpec列举；显式对齐并检查knowledge_time、秩亏、权重与缺失策略；不靠列名排除猜测。

**验收**：
1. 额外metadata列不得进入回归。
2. 行业和市值列调序不改身份。
3. 错资产错时点阻断。
4. 单资产/共线/缺失暴露有确定政策。

**依赖**：按波次/主合同安排，无额外单项依赖。 **需求映射**：R05, R15。

**V3实施展开——具体怎么改**

ExposureSpec列举规范exposure IDs、类型、PIT引用、权重、截距、缺失与秩亏策略；不把所有非键列自动当因子。显式检查估计所需资产数和可辨识性。

**V3追加回归——防止只修表面**

多加一个字符串描述列不进入回归；风险列置换在规范化后不改变结果；缺一行业暴露有政策化状态。

**V3历史对象与兼容性处置**

风险模型或列集合变动必须成为新的exposure snapshot和recipe身份。

**关联专项**：[RCP-04](#issue-rcp-04)。这些专项可在同一根因PR落实。

---

<a id="issue-fp-07"></a>
#### FP-07 · 变换去重只看semantic_id与stage，可能删除必要操作

**优先级**：P0　**证据分类**：源码确认缺陷　**Owner**：FP/FE

**已阅源码**：[`factor_preprocess/factor_preprocess/contracts/treatment_lineage.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/factor_preprocess/factor_preprocess/contracts/treatment_lineage.py)。

**函数/边界**：`TransformLineage.dedupe`。

**现象与证据**：忽略parameters、输入节点和中间操作，只保留相同(semantic_id,stage)第一次出现。

**影响**：不同EWMA参数或被非线性操作隔开的标准化可被错误折叠；记录配方和最终执行不一致。

**修改要求**：不对任意处理历史去重；仅对同输入/参数/轴/mask/权重/state的局部等价表达式做经过证明的优化；使用FE canonical DAG。

**验收**：
1. EWMA h3和h10都保留。
2. zscore→nonlinear→zscore不得删后者。
3. 同语义不同输入节点不得合并。

**依赖**：按波次/主合同安排，无额外单项依赖。 **需求映射**：R13, R14。 **隔离复现索引**：PR-17（非全仓测试）。

**V3实施展开——具体怎么改**

处理历史只做append，不自动去重。DAG优化器仅删除相同输入下可证明的幂等或共同子表达式；同算子不同参数、mask、权重或fit-state都不同。日志记录rewrite law和前后hash。

**V3追加回归——防止只修表面**

h3→h10滤波不删；zscore→abs→zscore保留；同义alias同输入且满足前提才折叠。

**V3历史对象与兼容性处置**

旧被误折叠配方及下游value需要新版本，保留旧定义供审计。

**关联专项**：[RCP-02](#issue-rcp-02)。这些专项可在同一根因PR落实。

---

<a id="issue-fp-08"></a>
#### FP-08 · 已有处理签名只是出现记录，不是最终输出性质

**优先级**：P1　**证据分类**：已见接口缺口／局部风险　**Owner**：FP/FE

**已阅源码**：[`factor_preprocess/factor_preprocess/contracts/treatment_lineage.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/factor_preprocess/factor_preprocess/contracts/treatment_lineage.py)；[`factor_engine/api/static_analysis.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/factor_engine/api/static_analysis.py)。

**函数/边界**：`ExistingTreatmentSignature / FE semantic map`。

**现象与证据**：签名仅有winsor/industry/size/rank/smoothing等布尔项，缺zscore最终状态；FE和FP维护两份名称映射；不能表达中间操作破坏已满足约束。

**影响**：可能重复标准化、错误跳过必要中性化、误判“已完成预处理”。

**修改要求**：共享operator语义元数据，并维护output properties：缩放轴、当前mask/权重、对何暴露正交、最后变换/拟合状态；history和postcondition分离。

**验收**：
1. neutralize→rank不再保证线性中性。
2. rank内嵌子表达式不代表根已rank。
3. 未知算子产生unknown-property而非untreated。

**依赖**：FP-07。 **需求映射**：R05, R13, R14。

**V3实施展开——具体怎么改**

以OutputProperties记录每个DAG节点当前性质：缩放轴、单位、mask、风险暴露正交约束、是否rank、可用时间。每个operator声明保持/破坏/产生哪些性质；history仅是解释信息。

**V3追加回归——防止只修表面**

在子表达式出现rank不能把整公式标成ranked；neutralize后rank或clip必须使strict-neutral性质失效或复验。

**V3历史对象与兼容性处置**

将粗布尔签名仅作为旧展示字段，不再用作自动跳过处理的依据。

**关联专项**：[DTA-06](#issue-dta-06)、[DTA-08](#issue-dta-08)、[RCP-01](#issue-rcp-01)、[RCP-02](#issue-rcp-02)、[RCP-05](#issue-rcp-05)。这些专项可在同一根因PR落实。

---

<a id="issue-fp-09"></a>
#### FP-09 · 合法顺序认证可被一个RAW标签绕过

**优先级**：P0　**证据分类**：源码确认缺陷　**Owner**：FP

**已阅源码**：[`factor_preprocess/factor_preprocess/grammar/search_grammar.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/factor_preprocess/factor_preprocess/grammar/search_grammar.py)。

**函数/边界**：`certify_production_presets / _certified_recipe_for`。

**现象与证据**：任意known-stage序列若携带白名单recipe tag，函数直接返回certified_exception，没有检查实际序列是否匹配该recipe；RAW也在白名单。

**影响**：非法重排仍能拿到认证；经济意义与时序约束失守。

**修改要求**：标签只能描述；认证绑定确切步骤DAG/参数/前置条件/输出性质与content hash。命名例外必须完整匹配，不匹配立即失败。

**验收**：
1. 故意E→A→D并加RAW或SMOOTH_NEUTRALIZE_RANK标签仍拒绝。
2. 合法例外匹配参数和后置暴露条件后通过。

**依赖**：按波次/主合同安排，无额外单项依赖。 **需求映射**：R13, R15。 **隔离复现索引**：PR-19（非全仓测试）。

**V3实施展开——具体怎么改**

白名单recipe是模板定义，不是免检通行证。匹配具体canonical DAG结构、参数范围、前置和后置性质；RAW只匹配无变换图，不能以tags覆盖真实步骤。

**V3追加回归——防止只修表面**

给E→A→D加RAW标签仍拒；合法例外不仅匹配名字，还须验证最后风险暴露与数据时钟。

**V3历史对象与兼容性处置**

作废仅靠tag签发的旧认证，并按真实配方重新认证。

**关联专项**：[RCP-05](#issue-rcp-05)。这些专项可在同一根因PR落实。

---

<a id="issue-fp-10"></a>
#### FP-10 · 变换注册默认可生产，参数域和生产状态只做局部检查

**优先级**：P1　**证据分类**：已见接口缺口／局部风险　**Owner**：FP

**已阅源码**：[`factor_preprocess/factor_preprocess/registry/transforms.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/factor_preprocess/factor_preprocess/registry/transforms.py)。

**函数/边界**：`register / bind_parameters / validate_production`。

**现象与证据**：缺省admission变PRODUCTION且causal_safe默认True；bind_partial只验证参数名；validate_production只检查admission与causal_safe，未在这里验证拟合边界、深层能力和实际参数域。

**影响**：新增实现可能靠声明标签而非数值/时序验收进入生产；应穿透编译入口确认更强守门。

**修改要求**：新增变换默认UNKNOWN/RESEARCH；生产能力需测试制品签发；在实际调用时校验参数schema、input requirements、fitted state、DecisionClock，而非仅注册时。

**验收**：
1. 未提供causality证据的新算子不得生产。
2. 非法半衰期/空暴露/测试拟合状态拒绝。
3. 额外kwargs不能绕域约束。

**依赖**：按波次/主合同安排，无额外单项依赖。 **需求映射**：R14, R15, R17。

**V3实施展开——具体怎么改**

register只证明目录收录，不证明production认证。生产验收绑定operator版本、数值政策、backend、因果测试与fit/apply合同；每次执行仍检查effective参数域和state来源。

**V3追加回归——防止只修表面**

默认缺causality证据不可生产；半衰期<=0、测试fit state、缺required exposure必须在调用边界拦截。

**V3历史对象与兼容性处置**

已验证算子不能无理由降级，未验证项按证据标RESEARCH/BLOCKED。

---

<a id="issue-fp-11"></a>
#### FP-11 · 声明不可变的配方及语义标识仍可被修改

**优先级**：P1　**证据分类**：源码确认缺陷　**Owner**：FP

**已阅源码**：[`factor_preprocess/factor_preprocess/contracts/treatment_lineage.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/factor_preprocess/factor_preprocess/contracts/treatment_lineage.py)；[`factor_preprocess/factor_preprocess/registry/policies.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/factor_preprocess/factor_preprocess/registry/policies.py)。

**函数/边界**：`TransformSemanticID / PolicyPreset / registry.policies.TransformStep`。

**现象与证据**：TransformSemanticID.value可赋值；registry.policies中的TransformStep/PolicyPreset不是frozen，tuple steps内仍有可变step。lineage builder还把已冻结smoothing params替换成dict。

**影响**：缓存键、集合成员及recipe identity可在运行中漂移。

**修改要求**：使用真不可变内容寻址值对象；freeze所有嵌套容器；派生新配方而非原地改状态，hash语义分离research mutable config与frozen execution plan。

**验收**：
1. 修改semantic_id.value/step.name/嵌套参数必须失败或不影响快照。
2. 序列化重建hash一致。

**依赖**：按波次/主合同安排，无额外单项依赖。 **需求映射**：R14, R15。

**V3实施展开——具体怎么改**

ResearchConfig与CompiledRecipe分开，后者深冻结并内容寻址。TransformSemanticID改为不可变值类型；frozen实例内不能再被builder替换成可变dict。

**V3追加回归——防止只修表面**

修改嵌套set/list/array、step.name、semantic.value均不得改变旧身份；两个快照无共享可变别名。

**V3历史对象与兼容性处置**

旧mutable配置只可重新编译产生新身份，不能悄悄延续旧生产ID。

---

<a id="issue-fp-12"></a>
#### FP-12 · 实现哈希fallback使用生成器repr，不是字节码内容

**优先级**：P1　**证据分类**：源码确认缺陷　**Owner**：FP

**已阅源码**：[`factor_preprocess/factor_preprocess/registry/transforms.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/factor_preprocess/factor_preprocess/registry/transforms.py)。

**函数/边界**：`_source_of`。

**现象与证据**：inspect.getsource失败后返回str(dis.get_instructions(func))，得到生成器对象表示而不是指令序列。

**影响**：同函数不同进程身份不稳定，或不同函数未按内容区分，影响缓存与复现。

**修改要求**：对可支持函数序列化实际code/常量/闭包/依赖版本；不支持时明确不可内容认证，禁止生成看似可信hash。

**验收**：
1. 动态函数在不同进程相同语义hash相同，常数变化hash改变。
2. builtin/闭包按能力拒绝或准确记录。

**依赖**：按波次/主合同安排，无额外单项依赖。 **需求映射**：R14。 **隔离复现索引**：PR-20（非全仓测试）。

**V3实施展开——具体怎么改**

源码不可得时明确支持的指纹范围；对code bytes、常量、闭包以及关键依赖版本规范序列化。无法稳定认证的动态对象返回不可缓存/不可生产，而不是任意repr hash。

**V3追加回归——防止只修表面**

新进程同代码同依赖hash稳定；仅修改常数或捕获参数hash改变；捕获可变外部状态拒绝认证。

**V3历史对象与兼容性处置**

用旧repr指纹生成的缓存和认证需重新签发。

**关联专项**：[RCP-10](#issue-rcp-10)。这些专项可在同一根因PR落实。

---

### 11.3 FA：健康卡、评级、准入与聚类资产

<a id="issue-fa-01"></a>
#### FA-01 · 健康卡空完整性门槛得到通过

**优先级**：P0　**证据分类**：源码确认缺陷　**Owner**：FA

**已阅源码**：[`factor_assets/profiling/health_card.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/factor_assets/profiling/health_card.py)。

**函数/边界**：`_integrity_results / build_health_card`。

**现象与证据**：默认integrity_gates=()，序列直接返回，all(g.passed is True for g in gates)对空序列为True。

**影响**：没有PIT/标签成熟/快照/单位等证据仍可能判admissible。

**修改要求**：按用途政策验证必需gate集合精确覆盖且无重复；每个gate绑定可解析的可信证据，空/缺失/unknown均不能通过。

**验收**：
1. 14维都高但gates为空不通过。
2. 缺1项/重复1项不通过。
3. 伪造字符串evidence_ref不构成证明。

**依赖**：按波次/主合同安排，无额外单项依赖。 **需求映射**：R03, R15。 **隔离复现索引**：PR-15（非全仓测试）。

**V3实施展开——具体怎么改**

按用途从冻结policy取得required gates，与输入集合做覆盖校验；缺项/重复/未知/无可解析证据逐项报错。derived summary在验证所有子证据身份后计算，不能all(empty)。

**V3追加回归——防止只修表面**

空门、只一个真门、重复门、全真但不同快照引用全部不得通过；全部合格作为正控。

**V3历史对象与兼容性处置**

可能经该入口误批准的资产纳入再认证，不自动删除生产值。

**关联专项**：[SRH-02](#issue-srh-02)、[QAT-02](#issue-qat-02)、[QAT-04](#issue-qat-04)。这些专项可在同一根因PR落实。

---

<a id="issue-fa-02"></a>
#### FA-02 · 健康卡可以混装其他因子的评级，并接受自报准入结论

**优先级**：P0　**证据分类**：源码确认缺陷　**Owner**：FA

**已阅源码**：[`factor_assets/profiling/health_card.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/factor_assets/profiling/health_card.py)；[`factor_assets/profiling/metric_grading.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/factor_assets/profiling/metric_grading.py)；[`factor_assets/profiling/dimensions.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/factor_assets/profiling/dimensions.py)。

**函数/边界**：`FactorHealthCardArtifact.__post_init__ / DimensionGradeArtifact`。

**现象与证据**：健康卡检查维度数量与顺序，但未逐项绑定factor/evaluation/policy；AdmissionSummary由调用者直接提供且不与实际gate和grades复算交叉验证。

**影响**：有失败gate但summary.admissible=True的对象可能被构造；旧因子/旧数据上的高分被嫁接。

**修改要求**：构造权威builder统一核验child refs和context hash；从原始证据派生grade与summary，反序列化必须复算一致性；来源验证不等于仅检查hash长度。

**验收**：
1. 其他因子/不同split/不同policy维度混入拒绝。
2. false gate+true summary拒绝。
3. 未知字母等级不能进入排名。

**依赖**：FA-01, QE-27。 **需求映射**：R03, R14, R15。

**V3实施展开——具体怎么改**

构造健康卡时核验每个dimension的factor/value/evaluation/context/policy；deserializer重算summary并比对，不接受自报admissible。MetricGrade也必须绑定可信原始证据。

**V3追加回归——防止只修表面**

借用另一因子的A评级、换horizon或旧policy、false gate+true summary全部拒绝。

**V3历史对象与兼容性处置**

健康卡结构升级时保留旧卡，按新可信证据重建。

**关联专项**：[QAT-05](#issue-qat-05)。这些专项可在同一根因PR落实。

---

<a id="issue-fa-03"></a>
#### FA-03 · 健康政策frozen但规则字典可原地变更

**优先级**：P1　**证据分类**：源码确认缺陷　**Owner**：FA

**已阅源码**：[`factor_assets/profiling/policies.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/factor_assets/profiling/policies.py)。

**函数/边界**：`FactorHealthPolicy.__post_init__ / policy mappings`。

**现象与证据**：已读实现用dict(...)保存metric_grade_rules和dimension_rules；frozen dataclass不阻止嵌套dict修改。

**影响**：同一policy_id/version对应两套门槛，历史等级不可追溯；现有评分器和优化器可漂移。

**修改要求**：将整个政策深冻结并内容寻址；任何变动生成新policy version/hash，已封存评估引用旧快照。

**验收**：
1. 外部/内部字典修改不能改变已创建政策。
2. 只变阈值会生成新hash并重新评估受影响资产。

**依赖**：按波次/主合同安排，无额外单项依赖。 **需求映射**：R03, R14。

**V3实施展开——具体怎么改**

政策所有字典、锚点、阈值、方向和适用条件深冻结并做content hash；版本号不是唯一身份。变更阈值必须新policy与影响评估，不允许在对象内部替换rule。

**V3追加回归——防止只修表面**

同版本内容冲突注册失败；旧卡继续解析旧快照；外部原始dict变动不改policy。

**V3历史对象与兼容性处置**

不要因policy变化必然重算昂贵因子值；可复用有效metric，重做评级/准入。

**关联专项**：[STA-08](#issue-sta-08)。这些专项可在同一根因PR落实。

---

<a id="issue-fa-04"></a>
#### FA-04 · 缺一项证据就把整个维度打零，缺失与低质量混同

**优先级**：P1　**证据分类**：定义／口径冲突　**Owner**：FA

**已阅源码**：[`factor_assets/profiling/dimensions.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/factor_assets/profiling/dimensions.py)；[`factor_assets/profiling/policies.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/factor_assets/profiling/policies.py)。

**函数/边界**：`dimension aggregation`。

**现象与证据**：部分缺失替0且使用.4*min+.6*geomean，任何一个0可使维度为0；全部缺失又给None；聚合实现还同时在DimensionRule.score_from与dimensions出现。

**影响**：系统把尚未算、算不了和确实很差混同，误导维修与准入。

**修改要求**：必需证据缺失直接证据门失败；optional不适用不进分母；NOT_RUN/UNSUPPORTED/INSUFFICIENT保留原因。只保留一份维度聚合与映射政策。

**验收**：
1. .9/.9/missing不得被描述为已测D级性能。
2. 必需缺失不能通过。
3. optional不适用不扣成0。

**依赖**：按波次/主合同安排，无额外单项依赖。 **需求映射**：R03, R08。 **隔离复现索引**：PR-16（非全仓测试）。

**V3实施展开——具体怎么改**

将metric state与性能grade正交建模。required缺失先门控，optional不适用从聚合集合排除，budget skipped保留原因。同一维度去掉重复衍生统计的隐含重复权重。

**V3追加回归——防止只修表面**

相同已测值但optional未跑不能冒充已测D；全缺与部分缺都能解释；经济灾难不能当optional缺失排除。

**V3历史对象与兼容性处置**

聚合定义改变需policy版本升级，既有metric数值可在语义匹配时复用。

**关联专项**：[STA-12](#issue-sta-12)。这些专项可在同一根因PR落实。

---

<a id="issue-fa-05"></a>
#### FA-05 · 形状质量不能把高单调性与高U型评分同时当共同优点

**优先级**：P1　**证据分类**：定义／口径冲突　**Owner**：FA

**已阅源码**：[`factor_assets/profiling/dimensions.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/factor_assets/profiling/dimensions.py)；[`quant_evaluator/metrics/shape_evidence.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/quant_evaluator/metrics/shape_evidence.py)；[`factor_assets/profiling/policies.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/factor_assets/profiling/policies.py)。

**函数/边界**：`shape_quality policy and diagnosis`。

**现象与证据**：现有形状质量设计需复核所配指标是否把monotonicity/U/tail共同求低值；U型本来应走不同形状分支。此项是明确的政策冲突，具体所有入口需AI继续定位。

**影响**：不同机制互相扣分，真正有用的非线性信号在修复前被淘汰。

**修改要求**：先识别MONOTONIC/U/INVERTED_U/TAIL/DISCRETE/UNKNOWN；再按该分支评价证据、稳定性和可交易映射，不用统一形状总尺。

**验收**：
1. 稳定U不因低单调性拒绝。
2. 单调因子不因u_score低拒绝。
3. 噪声型不能靠任选最好分支过关。

**依赖**：QE-03, QE-33, QE-34。 **需求映射**：R04, R11。

**V3实施展开——具体怎么改**

先用训练证据冻结shape family，再选择该family的quality rule；兼容多种形状但不在test上挑给分最高的分支。类型不清时保留UNKNOWN和低预算诊断路径。

**V3追加回归——防止只修表面**

单调不因u_score=0扣分，U不因rankIC近零直接reject；纯噪声不能反复选形状直到过门。

**V3历史对象与兼容性处置**

旧统一形状评分失效，重新根据有类型证据评级。

---

<a id="issue-fa-06"></a>
#### FA-06 · 字段类别、机制标签和控制字段角色尚需精确绑定

**优先级**：P1　**证据分类**：已见接口缺口／局部风险　**Owner**：FA/DA/FE

**已阅源码**：[`factor_assets/profiling/taxonomy.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/factor_assets/profiling/taxonomy.py)；[`factor_engine/api/static_analysis.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/factor_engine/api/static_analysis.py)。

**函数/边界**：`classify_factor_taxonomy / field_taxonomy integration`。

**现象与证据**：现有规则引擎支持多域集合，正确方向应保留；但允许built-in裸字段名map，调用方才可传DA taxonomy。必须确认生产不以同名字段猜表/市场/角色。

**影响**：量价因子用市值中性后可能被误解释为混合alpha；复合字段/未知字段被错误打标签。

**修改要求**：生产要求DA规范字段ID、派生字段血缘、signal/control角色；来源标签确定性，经济机制带rule/evidence/confidence并区分Agent建议与确认事实。

**验收**：
1. 同名不同表字段不合并。
2. 量价+换手+财务多标签完整。
3. 控制市值不改原始alpha机制。
4. UNKNOWN不静默丢掉。

**依赖**：按波次/主合同安排，无额外单项依赖。 **需求映射**：R01, R02。

**V3实施展开——具体怎么改**

生产classifier消费FE静态字段使用+DA taxonomy，不重写parser。派生字段展开叶子来源但保留中间经济角色；alpha/control/eligibility权重字段分别存。机制标签标rule-based hypothesis而非统计结论。

**V3追加回归——防止只修表面**

volume/free_float推导换手时标签完整；市值只用于neutralization不改alpha来源；未知字段保持unknown。

**V3历史对象与兼容性处置**

来源分类可独立于昂贵落值更新，但改变字段语义或依赖必须使相关身份失效。

**关联专项**：[DTA-06](#issue-dta-06)。这些专项可在同一根因PR落实。

---

<a id="issue-fa-07"></a>
#### FA-07 · 旧PromotionGate可接受裸数值和默认成熟，并软化已有硬失败

**优先级**：P0　**证据分类**：源码确认缺陷　**Owner**：FA

**已阅源码**：[`factor_assets/library/promotion_gate.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/factor_assets/library/promotion_gate.py)。

**函数/边界**：`CandidateEvaluationRef.from_dict / PromotionGate.evaluate`。

**现象与证据**：label_maturity默认True，evidence_ref等可缺；bool("false")=True；soft-duplicate分支可在处理已有reject_codes之前返回REVIEW。

**影响**：旧入口绕过健康卡和严格证据；硬失败的裁决优先级被破坏。

**修改要求**：全部入口适配到统一AdmissionAuthority；布尔严格反序列化；REJECT>REVIEW>APPROVE；policy按用途绑定，不以absIC和.7相关性独立决定入库。

**验收**：
1. "false"/1/空证据不自动成熟。
2. 错误basis+soft duplicate仍REJECT。
3. API/CLI/平台入口同证据同裁决。

**依赖**：FA-01, FA-02。 **需求映射**：R03, R11, R15。 **隔离复现索引**：PR-21（非全仓测试）。

**V3实施展开——具体怎么改**

所有旧gate、CLI与平台适配到唯一用途准入入口；先完整性失败再质量失败最后软审查。strict bool反序列化；bare IC只能用于研究显示，不能形成可信AdmissionVerdict。

**V3追加回归——防止只修表面**

wrong basis+soft duplicate仍REJECT；string false不能成熟；没有证据的高IC无法晋级。

**V3历史对象与兼容性处置**

建立旧批准对象重评队列及无损状态迁移，不直接把历史APPROVED当新认证。

**关联专项**：[QAT-05](#issue-qat-05)。这些专项可在同一根因PR落实。

---

<a id="issue-fa-08"></a>
#### FA-08 · MERGE_NEAREST把因子投到最大簇却沿用别簇高相似度

**优先级**：P0　**证据分类**：源码确认缺陷　**Owner**：FA

**已阅源码**：[`factor_assets/clustering/incremental.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/factor_assets/clustering/incremental.py)。

**函数/边界**：`_small_cluster_merge_policy_apply / incremental_assign`。

**现象与证据**：小簇被重定向到largest membership而非最近，后续top_sim和assignment affinity仍可能用原目标高分，runner gap被清空。

**影响**：新因子可进入低相似大簇并附错误高分，污染模型特征。

**修改要求**：目标选择必须基于实际合格affinity；换目标重新绑定该目标证据、gap与多代表支持；无目标就保留新小族或PENDING，不强塞。

**验收**：
1. 与A=.9与大B=.1时不能投B并记录.9。
2. 所有输出affinity对应最终cluster。
3. 小簇策略全量/增量复用。

**依赖**：按波次/主合同安排，无额外单项依赖。 **需求映射**：R12。 **隔离复现索引**：PR-18（非全仓测试）。

**V3实施展开——具体怎么改**

簇大小策略只可在合格目标中选择；改投后重新计算最终目标的支持邻居、亲和度和次优gap。若没有合格目标，应PROVISIONAL或PENDING，保留小簇不伪造置信。

**V3追加回归——防止只修表面**

A=.9、B=.1时不能投B带.9；同一最终簇的affinity与证据邻居必须一致。

**V3历史对象与兼容性处置**

用错误分配构建的后续overlay、代表和模型特征需要有范围重建。

**关联专项**：[SIM-05](#issue-sim-05)、[SIM-08](#issue-sim-08)。这些专项可在同一根因PR落实。

---

<a id="issue-fa-09"></a>
#### FA-09 · 低相似度、未测量与ANN近似不能用同一种证据状态

**优先级**：P1　**证据分类**：定义／口径冲突　**Owner**：FA

**已阅源码**：[`factor_assets/clustering/incremental.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/factor_assets/clustering/incremental.py)。

**函数/边界**：`min_measure_floor / cosine affinity`。

**现象与证据**：已计算s<.25被排除且按not measured表达；嵌入cosine用于归簇，不自动等于因子值相关或组合PnL相关。

**影响**：系统把真低相似候选当未知，或把近似召回当最终高相关证明。

**修改要求**：MEASURED_LOW/UNMEASURED/APPROXIMATE/CERTIFIED分开；ANN召回后QE按指定窗口、mask、方向计算正式pairwise证据；记录未召回区域的覆盖界限。

**验收**：
1. 真rho=0与没测区分。
2. 反向重复factor用abs/sign规则明确。
3. 召回不全不能声称全库无重复。

**依赖**：按波次/主合同安排，无额外单项依赖。 **需求映射**：R12, R15。

**V3实施展开——具体怎么改**

相似证据保留value、measured_status、approximation_kind、n_overlap与sample_ref；负相关保留符号，不能任意转成正边而丢方向。ANN未召回不写rho=0。

**V3追加回归——防止只修表面**

真实0、未测、无共同样本和近似0四种状态不同；threshold以下的已测结果可解释。

**V3历史对象与兼容性处置**

迁移UNKNOWN_AFFINITY语义时保留原值来源，不凭旧unknown推断不相似。

**关联专项**：[SIM-01](#issue-sim-01)、[SIM-02](#issue-sim-02)、[SIM-04](#issue-sim-04)、[SIM-05](#issue-sim-05)、[SIM-06](#issue-sim-06)。这些专项可在同一根因PR落实。

---

<a id="issue-fa-10"></a>
#### FA-10 · 增量ANN在新因子×簇循环重建索引，归属只看单个最近成员

**优先级**：P2　**证据分类**：已见接口缺口／局部风险　**Owner**：FA

**已阅源码**：[`factor_assets/clustering/incremental.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/factor_assets/clustering/incremental.py)。

**函数/边界**：`_select_ann / incremental_assign`。

**现象与证据**：_select_ann每次构建Annoy索引；每簇只保留cands[0]代表亲和度。

**影响**：增量开销可能接近反复重建；大簇容易因单一相似成员吸纳异质因子。

**修改要求**：每版本持久化全局或分片ANN索引，新因子批量查询；归属使用medoid+top-k有效支持与簇内质量，明确孤点和歧义区。

**验收**：
1. 一批新因子索引build次数不随每个factor×cluster增长。
2. 仅一个近邻不应覆盖整个簇质量要求。
3. 稳定新族可暂存。

**依赖**：按波次/主合同安排，无额外单项依赖。 **需求映射**：R08, R12。

**V3实施展开——具体怎么改**

索引属于library/cluster-set版本，可被多批查询复用；加入新数据使用明确rebuild或delta index。归属不只看一个最相似成员，至少检查代表和支持量。

**V3追加回归——防止只修表面**

百个新因子共享同版本索引而非百次build；一个bridge因子不会把两大族随意吞并。

**V3历史对象与兼容性处置**

性能测量包括索引构建与查询，不能只报告query时间。

**关联专项**：[SIM-07](#issue-sim-07)、[SIM-08](#issue-sim-08)。这些专项可在同一根因PR落实。

---

<a id="issue-fa-11"></a>
#### FA-11 · 旧Pareto实现允许缺失/NaN目标悄悄进入非支配前沿

**优先级**：P1　**证据分类**：源码确认缺陷　**Owner**：FA/FO

**已阅源码**：[`factor_assets/optimizer/pareto.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/factor_assets/optimizer/pareto.py)。

**函数/边界**：`ParetoPoint.dominates / ParetoOptimizer.compute_frontier`。

**现象与证据**：缺目标取-inf；没有统一拒绝NaN与目标维度缺失；NaN比较为False可使坏候选留在前沿。另FA与FO均有Pareto实现。

**影响**：未评估候选可当非支配优胜者；不同入口选择不同。

**修改要求**：主权威选定现有一个正确实现，另一个仅适配；先required-objective完整性/finite验证再Pareto；unknown不等于最差可排序数值。

**验收**：
1. NaN/缺一核心目标/方向不一致候选不进入合格前沿。
2. 两入口相同输入相同集合。
3. 不因维度顺序改变。

**依赖**：按波次/主合同安排，无额外单项依赖。 **需求映射**：R03, R11。

**V3实施展开——具体怎么改**

目标集合、方向、状态与单位先验证，再做epsilon或精确Pareto；不可计算目标不映射成-inf混在已评对象中。复用现有唯一selector，FA治理只消费其可审计结果。

**V3追加回归——防止只修表面**

NaN目标、缺核心目标、同值候选和次序置换各测；近等价候选走简单度/成本规则而非浮点随机胜出。

**V3历史对象与兼容性处置**

旧被无效目标吸入的前沿需重算，保留选择版本与理由。

**关联专项**：[SRH-03](#issue-srh-03)。这些专项可在同一根因PR落实。

---

### 11.4 FO：搜索、证据适配、切分与状态

<a id="issue-fo-01"></a>
#### FO-01 · Optimizer的QE适配器对多因子批返回空metrics

**优先级**：P0　**证据分类**：源码确认缺陷　**Owner**：FO

**已阅源码**：[`factor_optimizer/factor_optimizer/adapters/quant_evaluator.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/factor_optimizer/factor_optimizer/adapters/quant_evaluator.py)。

**函数/边界**：`ConcreteQEAdapter.evaluate`。

**现象与证据**：从bundle.metric_values抽取flat metrics，而多因子CPU结果主要在grouped_metrics；groups虽存证据却不随返回值暴露；返回标量也丢valid/count/warnings。

**影响**：用户期待的批量评估不能正确驱动搜索；坏证据可被裸数值消费。

**修改要求**：适配器返回按factor_id绑定的typed evidence refs或完整列式合同；保留status、单位、版本、统计样本、split/context。

**验收**：
1. 一次3因子与逐个调用最终metric和状态一致。
2. invalid不能通过plain(value)变成有效。
3. 不得跨factor串位。

**依赖**：QE-01, QE-02, QE-05。 **需求映射**：R08, R09。

**V3实施展开——具体怎么改**

搜索调用以batch_id、candidate_id、factor/value_ref建立显式映射；适配器返回每因子完整证据，不把空flat metrics当计算失败或零。单因子便捷接口只是批接口视图。

**V3追加回归——防止只修表面**

一次3候选与逐个3次结果、状态和选择集合相同；返回顺序不同也按ID对齐，不按zip猜。

**V3历史对象与兼容性处置**

错误batch结果对应trial应可重评，不能被记为永久无效因子。

---

<a id="issue-fo-02"></a>
#### FO-02 · QE适配器没有传递split、backend、参数和完整证据身份

**优先级**：P0　**证据分类**：已见接口缺口／局部风险　**Owner**：FO

**已阅源码**：[`factor_optimizer/factor_optimizer/adapters/quant_evaluator.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/factor_optimizer/factor_optimizer/adapters/quant_evaluator.py)。

**函数/边界**：`QuantEvaluatorAdapter protocol / ConcreteQEAdapter`。

**现象与证据**：协议仅batch/labels/metrics/context，具体调用未传split_ref/backend/gpu_policy；存储只保留部分bundle字段并生成新UUID。

**影响**：优化器的sealed/tier/预算意图可能没有抵达真实QE执行链；证据缺上下文。

**修改要求**：统一EvaluationRequest从FO到QE，完整保留不可伪造的数据权限与所有执行参数；UUID可作运行ID但不可代替内容身份，存完整证据引用。

**验收**：
1. FO请求非法split在QE真实入口拒绝。
2. GPU参数和20层配置到达执行计划。
3. evidence_store roundtrip保留所有required字段。

**依赖**：QE-04, QE-27, QE-31。 **需求映射**：R08, R09, R10, R15。

**V3实施展开——具体怎么改**

协议扩为完整EvaluationRequest或类型化ref，不继续新增散乱kwargs。传递split capability、数据快照、horizon、Q、预算和backend政策；存证据原始内容身份和独立run ID。

**V3追加回归——防止只修表面**

从FO发20层CUDA请求查看最终effective plan；恶意换split/ref在QE拒绝；持久化再取保留全部字段。

**V3历史对象与兼容性处置**

全部adapter调用方一起迁移，mock-only路径不进入生产。

**关联专项**：[SRH-07](#issue-srh-07)。这些专项可在同一根因PR落实。

---

<a id="issue-fo-03"></a>
#### FO-03 · Optimizer明确research_only，不能把已声明的阻断项当完成

**优先级**：P0　**证据分类**：已见接口缺口／局部风险　**Owner**：FO

**已阅源码**：[`factor_optimizer/factor_optimizer/capabilities.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/factor_optimizer/factor_optimizer/capabilities.py)。

**函数/边界**：`PRODUCTION_CAPABILITY`。

**现象与证据**：supported=False，列出trusted split QE integration、production legality chain、sealed evidence escalation三项缺口。

**影响**：当前不得把优化结果直接认证成自动生产资产；该保护是正确的，不是应该删掉的bug。

**修改要求**：依次补真实适配、可信隔离、评价完整性、资产门和增量回放，证据验收后签发能力版本；不可仅改supported=True。

**验收**：
1. 未完成任何阻断仍不能PRODUCTION。
2. mock证据永不能晋级。
3. 所有required integration真实跑通才可更新capability。

**依赖**：FO-02, FA-07, FP-04。 **需求映射**：R09, R14, R15。

**V3实施展开——具体怎么改**

把生产阻断拆成可验收capability证据，而非代码常量待办；分别验证可信数据权限、完整评价、合法配方、资产准入和增量执行。只对已通过范围签发证据。

**V3追加回归——防止只修表面**

任一必需capability缺失时production请求拒绝；只通过CPU不自动认证GPU；研究搜索仍可在显式模式工作。

**V3历史对象与兼容性处置**

保持保护开关，不能以实现新文档或增加测试数量作为解除依据。

---

<a id="issue-fo-04"></a>
#### FO-04 · 多保真阶段表与搜索执行需要穿透，串行限制是真实能力边界

**优先级**：P1　**证据分类**：已见接口缺口／局部风险　**Owner**：FO

**已阅源码**：[`factor_optimizer/factor_optimizer/search/runner.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/factor_optimizer/factor_optimizer/search/runner.py)；[`factor_optimizer/factor_optimizer/search/multifidelity.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/factor_optimizer/factor_optimizer/search/multifidelity.py)。

**函数/边界**：`SearchConfig / SearchRunner / StageProfileRegistry`。

**现象与证据**：已读runner使用legacy FidelityTier/MultiFidelityScheduler；另有Stage0-5命名profile。仅有两个结构不能证明已统一，max_concurrency>1当前明确拒绝。

**影响**：任务表上的GPU初筛/形状诊断/封存流程可能停留在metadata；不能假称并行已支持。

**修改要求**：从唯一执行计划导出阶段、筛选条件和预算；每个stage记录实际执行的profile/config；先实现F批量和确定性调度，再在具备预算原子性后扩并发。

**验收**：
1. 执行trace能证明每个profile真实运行。
2. 低保真分数不能与高保真直接争winner。
3. 串行与批量结果及trial计数可复现。

**依赖**：按波次/主合同安排，无额外单项依赖。 **需求映射**：R08, R09。

**V3实施展开——具体怎么改**

统一Stage与FidelityTier的映射，真实执行trace包含profile hash、input slice、metric set、资源和晋级理由。控制平面可串行，单次计算可F批量；不要为并行改松原子性。

**V3追加回归——防止只修表面**

禁用某stage builder应导致该stage真实失败；低保真不直接争冠军；取消/预算不足保留已完成证据。

**V3历史对象与兼容性处置**

废弃不再驱动执行的stage表或转成纯派生视图，避免双配置。

**关联专项**：[SRH-06](#issue-srh-06)。这些专项可在同一根因PR落实。

---

<a id="issue-fo-05"></a>
#### FO-05 · 搜索配置可变与checkpoint恢复需要绑定实际执行状态

**优先级**：P1　**证据分类**：已见接口缺口／局部风险　**Owner**：FO

**已阅源码**：[`factor_optimizer/factor_optimizer/search/runner.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/factor_optimizer/factor_optimizer/search/runner.py)。

**函数/边界**：`SearchConfig / SearchSession serialization`。

**现象与证据**：配置是mutable dataclass，objective/spec仅构造时对齐；策略spec反序列化不等于恢复策略内部随机状态/搜索历史。后续恢复路径需继续读取和测试，不能据片段断言已失败。

**影响**：中途改配置或重启可改变候选顺序、预算、已消费test状态。

**修改要求**：冻结execution plan；checkpoint包括strategy state/RNG/ledger/budget reservations/data capabilities；恢复时校验hash，拒绝状态不齐的“继续”。

**验收**：
1. 断电后下一候选和无中断运行一致。
2. 配置改变需要新session且继承campaign统计。
3. 已消费test跨进程不能重新开放。

**依赖**：按波次/主合同安排，无额外单项依赖。 **需求映射**：R09, R10, R14。

**V3实施展开——具体怎么改**

运行前compile frozen search plan；checkpoint原子保存candidate generator/RNG、ledger offset、预算reservation、incumbent集合、split授权状态。恢复核验全部hash，不齐进入恢复阻断。

**V3追加回归——防止只修表面**

同seed中断前后下一候选和无中断一致；启动另一个session仍继承campaign尝试和test消费信息。

**V3历史对象与兼容性处置**

旧checkpoint缺必要字段不能声称精确续跑，可显式新研究run且保留统计关联。

**关联专项**：[STA-04](#issue-sta-04)、[SRH-06](#issue-srh-06)、[SRH-08](#issue-srh-08)。这些专项可在同一根因PR落实。

---

### 11.5 PL：平台任务、发布、重试和模型快照

<a id="issue-pl-01"></a>
#### PL-01 · 无效候选本应被报告，却因空hash触发报告对象异常

**优先级**：P1　**证据分类**：源码确认缺陷　**Owner**：Platform

**已阅源码**：[`quant_platform/app/orchestrator.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/quant_platform/app/orchestrator.py)。

**函数/边界**：`normalization_failures → with_item → PipelineStateItem`。

**现象与证据**：normalization失败项用content_hash=""，PipelineStateItem构造器拒绝空hash；with_item会再次抛错。

**影响**：一条坏输入可中断整批处理；num_normalization_failed也未见对应增量路径。

**修改要求**：把“未能生成身份”作为合法失败状态，使用独立ingestion_record_id及optional factor identity；错误报告不得依赖尚未生成的成功字段；计数和明细对账。

**验收**：
1. 坏记录与好记录混合时好记录正常处理。
2. 失败可追溯且计数一致。
3. 不填假hash来通过校验。

**依赖**：按波次/主合同安排，无额外单项依赖。 **需求映射**：R09, R15。 **隔离复现索引**：PR-25（非全仓测试）。

**V3实施展开——具体怎么改**

摄取失败对象使用ingestion_id定位，factor/hash在未解析前为None而不是假字符串；失败报告构造器必须支持该状态。计数应由events/states派生或严格同事务更新。

**V3追加回归——防止只修表面**

坏+好混合输入好项继续，失败项不触发二次异常；失败明细数=失败总数。

**V3历史对象与兼容性处置**

不要用全零hash填洞；历史失踪失败记录在可恢复范围补审计。

---

<a id="issue-pl-02"></a>
#### PL-02 · 消费去重早于执行成功，失败候选可永久失去重试

**优先级**：P0　**证据分类**：源码确认缺陷　**Owner**：Platform

**已阅源码**：[`quant_platform/app/orchestrator.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/quant_platform/app/orchestrator.py)。

**函数/边界**：`Pipeline.run consumed tracking / processed_fingerprints`。

**现象与证据**：在评估成功前append/record_consumed；评估失败continue，批结束仍记录fingerprint已处理；后续相同批被short-circuit。

**影响**：瞬时网络/资源/依赖错误后再提交变“重复”，没有得到有效结果但再也不处理。

**修改要求**：区分DISCOVERED/CLAIMED/RETRYABLE/TERMINAL/SUCCEEDED；幂等键绑定工作意图而非失败即完成；持久化可恢复任务和租约，只有terminal outcome才定去重语义。

**验收**：
1. 第一次临时失败第二次成功。
2. 中间崩溃恢复不丢候选不重复发成功事件。
3. 同一公式新数据/policy允许合法重评。

**依赖**：按波次/主合同安排，无额外单项依赖。 **需求映射**：R09, R14, R16。

**V3实施展开——具体怎么改**

身份去重、任务幂等和终态成功三个概念分开。登记DISCOVERED不消耗执行资格；成功后terminal结果可幂等读取；retryable有lease与attempt，不因batch fingerprint终止。

**V3追加回归——防止只修表面**

首次OOM/网络失败后可续算；进程重启两worker不重复提交最终成功；同定义新数据可合法重评。

**V3历史对象与兼容性处置**

需修复历史已consume但未terminal的记录，dry-run列出后按已有权限恢复。

**关联专项**：[SRH-10](#issue-srh-10)、[OPS-01](#issue-ops-01)、[OPS-10](#issue-ops-10)。这些专项可在同一根因PR落实。

---

<a id="issue-pl-03"></a>
#### PL-03 · 平台把所有评估异常重新标成CAPABILITY，丢失重试分类

**优先级**：P1　**证据分类**：源码确认缺陷　**Owner**：Platform

**已阅源码**：[`quant_platform/app/orchestrator.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/quant_platform/app/orchestrator.py)。

**函数/边界**：`_make_candidate_handler`。

**现象与证据**：捕获所有Exception统一raise JobError(ErrorClass.CAPABILITY,...)，覆盖原错误类型和retryability。

**影响**：数据暂缺、预算、合法性和代码错误无法正确分类与重试，干扰失败清理。

**修改要求**：保留域类型化错误，显式映射retryable/data_pending/invalid_evidence/budget/cancelled/bug；未知错误保留cause并终止不无限重试。

**验收**：
1. 原retryable错误不降成永久capability。
2. 非法DSL不反复重试。
3. label_not_mature进入等待而非垃圾删除。

**依赖**：PL-02。 **需求映射**：R09, R16。

**V3实施展开——具体怎么改**

由平台维护显式域错误映射表，保留原cause和retryability。LABEL_NOT_MATURE进入等待队列、非法DSL终态失败、资源不足可降批/重试、未知bug不可无限重试。

**V3追加回归——防止只修表面**

验证每类error跨QE→FO→job仍保留类型、原始上下文和下一动作，清理不删除待成熟数据。

**V3历史对象与兼容性处置**

旧CAPABILITY记录按可证据识别重新分类，无法确定不编造原因。

**关联专项**：[OPS-10](#issue-ops-10)。这些专项可在同一根因PR落实。

---

<a id="issue-pl-04"></a>
#### PL-04 · 平台仍用RankIC是否存在先否决，绕过分用途准入政策

**优先级**：P1　**证据分类**：已见接口缺口／局部风险　**Owner**：Platform

**已阅源码**：[`quant_platform/app/orchestrator.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/quant_platform/app/orchestrator.py)。

**函数/边界**：`Pipeline.run before AdmissionAuthority.decide`。

**现象与证据**：在调用FA admission之前检查evaluation.rank_ic is None并reject；低单变量/非线性特征的替代证据政策没有机会决定。

**影响**：“平台不做域判断”的边界没有彻底实现。

**修改要求**：平台只检查证据引用与schema合法；哪种指标必需由FA用途政策决定。保留无证据拒绝，不把无RankIC等同无证据。

**验收**：
1. 有效非线性证据、RankIC不适用时到达模型特征准入门。
2. 真正缺required证据仍fail-closed。

**依赖**：FA-07。 **需求映射**：R11。

**V3实施展开——具体怎么改**

平台检查身份和证据对象可解析性；决定哪个metric必需归FA用途policy。删除提前以rank_ic=None判无证据的业务短路，但保留真正missing required evidence拒绝。

**V3追加回归——防止只修表面**

仅有合法非线性/事件证据的模型特征可到达FA；真无证据仍拒，不是取消门槛。

**V3历史对象与兼容性处置**

对曾被平台提前拒绝的类型允许按新用途重新审计。

---

<a id="issue-pl-05"></a>
#### PL-05 · 平台把semantic_family_hint或spec hash当factor_definition_ref

**优先级**：P0　**证据分类**：已见接口缺口／局部风险　**Owner**：Platform/FA/FE

**已阅源码**：[`quant_platform/app/orchestrator.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/quant_platform/app/orchestrator.py)。

**函数/边界**：`AdmissionRequest assembly / _snapshot_feature_set`。

**现象与证据**：AdmissionRequest.factor_definition_ref来源semantic_family_hint；FeatureMemberRef.factor_definition_ref=ch（spec hash）。

**影响**：身份域混用导致血缘查不到、去重错误、证据绑定无法证明。

**修改要求**：FE铸造definition identity，FA/Platform只携带强类型ref；spec bytes hash、semantic group、value ID、recipe ID分开，禁止凭名称/提示造身份。

**验收**：
1. 错误ref类型在边界拒绝。
2. 同一个定义不同落值和配方能区分。
3. 从模型feature追到原始DSL及数据快照。

**依赖**：按波次/主合同安排，无额外单项依赖。 **需求映射**：R01, R12, R14, R15。

**V3实施展开——具体怎么改**

在DTO中区分DefinitionRef、RecipeRef、ValueRef、EvaluationRef与SourceSpecRef；semantic hint只做展示。翻译器有严格映射表，找不到真实ref就停止，而不是从hash拼一个。

**V3追加回归——防止只修表面**

从模型列反查到同一FE定义；仅改展示名不改定义ID，改recipe使value身份改变。

**V3历史对象与兼容性处置**

旧模糊ref按可验证目录迁移；不可解析对象保持隔离。

**关联专项**：[MOD-05](#issue-mod-05)、[QAT-05](#issue-qat-05)。这些专项可在同一根因PR落实。

---

<a id="issue-pl-06"></a>
#### PL-06 · 特征快照只含本轮批准项，版本又依赖内存计数

**优先级**：P1　**证据分类**：已见接口缺口／局部风险　**Owner**：Platform/modeling

**已阅源码**：[`quant_platform/app/orchestrator.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/quant_platform/app/orchestrator.py)。

**函数/边界**：`_snapshot_feature_set / __init__`。

**现象与证据**：ordered_members仅来自report.approved_content_hashes；上轮成员未合并；revision=len(in-memory snapshots)+1。若作为生产累计特征库使用会丢旧成员并在重启重用版本号。

**影响**：研究批次快照与生产feature set被混同；模型输入无意换列或版本冲突。

**修改要求**：明确ROUND_RESULT与PUBLISHED_FEATURE_SET两类；正式集合从FA已发布快照和diff构建，持久化内容版本，recipe/weights/成员变动触发model再验证。

**验收**：
1. 第二批新增不偷偷删第一批。
2. 显式删除才产生remove diff。
3. 重启版本不回v1。
4. 列名未变但聚合权重变化也触发检查。

**依赖**：PL-05。 **需求映射**：R12, R14。

**V3实施展开——具体怎么改**

ROUND_RESULT与已发布feature集合分开类型；正式集合由FA snapshot加显式diff构造。版本由内容hash/持久化序列管理，重启不能v1回绕；列内聚合函数也是schema内容。

**V3追加回归——防止只修表面**

第二批新增不丢第一批，显式remove才删除；列名相同权重改变触发模型兼容验收。

**V3历史对象与兼容性处置**

旧round快照不得被误当生产完整集合，逐版本核对来源。

**关联专项**：[SIM-09](#issue-sim-09)、[MOD-05](#issue-mod-05)、[MOD-06](#issue-mod-06)。这些专项可在同一根因PR落实。

---

<a id="issue-pl-07"></a>
#### PL-07 · 候选ArtifactRef的hash、大小、媒体类型和真实字节未对齐

**优先级**：P0　**证据分类**：已见接口缺口／局部风险　**Owner**：Platform/DA

**已阅源码**：[`quant_platform/app/orchestrator.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/quant_platform/app/orchestrator.py)。

**函数/边界**：`_build_candidate_artifact`。

**现象与证据**：content_hash使用factor spec sha；size_bytes来自candidate_id:hash字符串；media_type标json、storage_uri是mem路径，本函数未发布对应内容。

**影响**：若当生产artifact，引用不能证明字节内容存在且可校验；hash身份不再是内容哈希。

**修改要求**：研究内存ref显式不可生产；正式对象经DA/publisher序列化→写入→校验→原子manifest→登记ref，hash和size都针对实际字节；语义hash另字段。

**验收**：
1. fetch(ref)字节sha/size/media_type逐项匹配。
2. 不存在payload不能登记PRODUCTION_READY。
3. 改一字节验证失败。

**依赖**：PL-05。 **需求映射**：R14, R15, R16。

**V3实施展开——具体怎么改**

使用已有DA发布接口：规范序列化bytes→写对象→校验size/hash→提交manifest→登记可解析ref。内存研究对象可用但不能PRODUCTION_READY；语义hash与字节hash独立。

**V3追加回归——防止只修表面**

上传截断、一字节修改、缺manifest、hash不符都不能晋级；读回字节与ref一致。

**V3历史对象与兼容性处置**

旧mem URI或假size对象列为未持久化，不凭元数据补认证。

**关联专项**：[OPS-05](#issue-ops-05)。这些专项可在同一根因PR落实。

---

<a id="issue-pl-08"></a>
#### PL-08 · 平台骨架的事件、登记和发布不是可证明的原子事务

**优先级**：P1　**证据分类**：已见接口缺口／局部风险　**Owner**：Platform/DA/FA

**已阅源码**：[`quant_platform/app/orchestrator.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/quant_platform/app/orchestrator.py)。

**函数/边界**：`Pipeline.run publish/register/snapshot`。

**现象与证据**：已读Pipeline默认in-memory outbox，批准事件在artifact登记前发；registry、consumed store与snapshot由分散调用完成。仓库可能另有持久化设施，必须查实际composition而非重复造一套。

**影响**：进程崩溃可有批准事件无实体、实体无快照或误消费；现有骨架不可直接当正式发布完成。

**修改要求**：复用已有outbox/DB/DA generation publisher；定义原子状态转移和恢复协议，发布指针最后CAS翻转；重放幂等但不吞retry。

**验收**：
1. 在每一步注入崩溃后均可恢复。
2. 消费者不看到半发布版本。
3. rollback回到同一血缘链完整版本。

**依赖**：PL-02, PL-07。 **需求映射**：R14, R16。

**V3实施展开——具体怎么改**

复用已有transactional outbox和generation publisher，界定单库事务和对象存储两阶段的恢复点。批准决策、注册对象和发布指针分别有状态，发布指针最后CAS更新。

**V3追加回归——防止只修表面**

每个write/publish边界崩溃注入恢复；消费者不看到半代数据；重放不重复发实际副作用。

**V3历史对象与兼容性处置**

沿同一血缘回滚，不跨到不相关旧版本；生产执行须既有批准流程。

**关联专项**：[MOD-08](#issue-mod-08)、[OPS-01](#issue-ops-01)、[OPS-06](#issue-ops-06)。这些专项可在同一根因PR落实。

---

### 11.6 QA：历史测试路径与真实能力矩阵

<a id="issue-qa-01"></a>
#### QA-01 · 对拍测试没有覆盖真实参数桥，且同名测试被覆盖

**优先级**：P1　**证据分类**：源码确认缺陷　**Owner**：QA

**已阅源码**：[`factor_preprocess/tests/test_fe_operator_parity.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/factor_preprocess/tests/test_fe_operator_parity.py)；[`factor_preprocess/factor_preprocess/adapters/fe_operator.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/factor_preprocess/factor_preprocess/adapters/fe_operator.py)。

**函数/边界**：`test_fe_operator_parity.py`。

**现象与证据**：测试仅用字符串证券ID；ffill手动传FE max_periods而非FP max_lag；同名test_cs_demean_fe_backed_matches_fp_kernel定义两次；FE缺失时跳过。

**影响**：绿灯可能只是默认输入或参考内核对拍，不能证明生产公开路径正确。

**修改要求**：增加真实registry/recipe/API全链测试，不手动修正待测输入；静态检测同作用域重复函数；required FE/GPU流水线的skip计BLOCKED而非PASS。

**验收**：
1. 测试覆盖FP-01/02/03真实调用。
2. 测试collector与文件定义数对账。
3. 生产认证报告明确依赖缺失和skip。

**依赖**：按波次/主合同安排，无额外单项依赖。 **需求映射**：R08, R14, R15。

**V3实施展开——具体怎么改**

测试真实用户参数和公开入口，不手动替adapter改名称；AST扫描检测同作用域同名test覆盖。required集成缺依赖必须记BLOCKED，不与纯研究可选skip混算。

**V3追加回归——防止只修表面**

整数ID、max_lag非默认、未知kwargs、全NaN、排序变换都走recipe→registry→FE；检查collector实际收集的nodeid。

**V3历史对象与兼容性处置**

为旧“passed”重新标覆盖范围；不声称所有历史测试失效，只隔离没有覆盖合同的声明。

**关联专项**：[QAT-01](#issue-qat-01)、[QAT-04](#issue-qat-04)、[QAT-07](#issue-qat-07)。这些专项可在同一根因PR落实。

---

<a id="issue-qa-02"></a>
#### QA-02 · 已有文档和“已注册”不能替代可执行能力表

**优先级**：P1　**证据分类**：已见接口缺口／局部风险　**Owner**：QA

**已阅源码**：[`quant_evaluator/registry/metrics.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/quant_evaluator/registry/metrics.py)；[`factor_optimizer/factor_optimizer/capabilities.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/factor_optimizer/factor_optimizer/capabilities.py)；[`quant_evaluator/runtime/evaluator.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/quant_evaluator/runtime/evaluator.py)；[`factor_preprocess/tests/test_fe_operator_parity.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/factor_preprocess/tests/test_fe_operator_parity.py)。

**函数/边界**：`capability reporting / docs generation`。

**现象与证据**：源码中存在新形状/风险metric与显式research-only标志；仅README指标数量或mock单测不能说明入口可运行及可生产。

**影响**：开发AI可能不断补声明、目录和测试数量，却没修真正链路。

**修改要求**：从真实注册表和运行验收生成能力矩阵：declared/kernel/public_cpu/public_gpu/adapter/profile/admission/production分别标状态；unknown和not-run不显示已完成。

**验收**：
1. 已注册但不可公开执行的shape metric不能标DONE。
2. GPU未跑不得标parity pass。
3. 缺真实资产证据的capability保持blocked。

**依赖**：按波次/主合同安排，无额外单项依赖。 **需求映射**：R03, R08, R14。

**V3实施展开——具体怎么改**

自动生成按能力层级的矩阵：声明、kernel、公开CPU/GPU、adapter、profile、gate、更新与发布。每格链接测试运行及数据版本；没有证据为NOT_RUN/UNSUPPORTED。

**V3追加回归——防止只修表面**

注册一个故意不可执行metric后能力表不能显示全绿；mock跑通也不能更新真实GPU格。

**V3历史对象与兼容性处置**

README指标数量和能力状态由矩阵派生，禁止手工维护相互矛盾的数字。

**关联专项**：[OPS-04](#issue-ops-04)、[QAT-08](#issue-qat-08)。这些专项可在同一根因PR落实。

---

### 11.7 V：原始需求端到端验收

<a id="issue-v-01"></a>
#### V-01 · 全量调用图与唯一权威清点，追踪旁路而非只改点名文件

**优先级**：P0　**证据分类**：需求验收／待穿透核查　**Owner**：总协调/QA

**定位规则**：尚未对全仓证明缺失；先从相关公开入口找到现有实现，证实已覆盖后直接复用。

**函数/边界**：`必须从真实公开入口反查已有实现及调用方，不允许凭本任务标题直接新建同名系统`。

**现象与证据**：本次为关键路径源码审查；未逐文件阅读所有FE、DA、平台和modeling实现。

**影响**：这是原始需求的完整性验收，不是已证明全仓缺失的bug；没有运行证据不能标完成。

**修改要求**：扫描四库、FE/DA adapters、quant_platform/research_platform、modeling/lightgbm_qs及CLI/HTTP/jobs；建立public entry→adapter→executor→evaluator→gate→store调用图。核查新旧同名类、影子入口、import别名、独立wheel与monorepo差异。

**验收**：
1. 每个生产入口必须归到唯一权威。
2. 所有直接计算IC/中性化/落值/入库/清理的旁路有明确处置。
3. 扫描未完成记BLOCKED，不声称all issues fixed。

**依赖**：按波次/主合同安排，无额外单项依赖。 **需求映射**：R14, R15。

**V3实施展开——具体怎么改**

以实际生产composition root为根列调用图；对四库和被引用FE/DA/modeling路径逐函数登记读取覆盖、执行入口、替代实现和测试。扫描命中不是Bug，必须定位真实调用。

**V3追加回归——防止只修表面**

inventory里未读文件、动态import与旧entry point均有状态；审阅范围可机器统计。

**V3历史对象与兼容性处置**

保留未查范围，不因97项处理结束声称全仓无问题。

**关联专项**：[QAT-07](#issue-qat-07)、[QAT-08](#issue-qat-08)。这些专项可在同一根因PR落实。

---

<a id="issue-v-02"></a>
#### V-02 · 自动标签必须真的在每次候选进入时生成，并可重评更新

**优先级**：P1　**证据分类**：需求验收／待穿透核查　**Owner**：FA/DA/FE

**定位规则**：尚未对全仓证明缺失；先从相关公开入口找到现有实现，证实已覆盖后直接复用。

**函数/边界**：`必须从真实公开入口反查已有实现及调用方，不允许凭本任务标题直接新建同名系统`。

**现象与证据**：已存在taxonomy/profile/health能力，尚需证明真实候选管线、字段目录和新数据评估均接入。

**影响**：这是原始需求的完整性验收，不是已证明全仓缺失的bug；没有运行证据不能标完成。

**修改要求**：字段分类来自canonical field catalog及derived lineage；source/control分离，层级可细分。机制标签保存来源、规则版本、置信度；统计标签绑定evaluation context；不要求LLM在每因子每次运行。

**验收**：
1. 量价only/财务only/量价财务/量价换手财务/未知字段均有golden。
2. 移除一个来源标签同步变化。
3. 重复运行同输入标签hash一致。

**依赖**：FA-06。 **需求映射**：R01, R02。

**V3实施展开——具体怎么改**

候选进入后静态分类立即生成；健康标签仅在评价证据到达后生成。二者版本独立，自动任务可幂等重放；纯说明修改不触发全量落值。

**V3追加回归——防止只修表面**

raw/treated/子因子均有独立来源与控制角色；同DSL同目录版本标签完全一致。

**V3历史对象与兼容性处置**

旧无来源标签可补元数据，不篡改历史统计评价。

---

<a id="issue-v-03"></a>
#### V-03 · 建立逐指标→逐维度→用途准入的完整政策，而非一个总分

**优先级**：P1　**证据分类**：需求验收／待穿透核查　**Owner**：FA/QE

**定位规则**：尚未对全仓证明缺失；先从相关公开入口找到现有实现，证实已覆盖后直接复用。

**函数/边界**：`必须从真实公开入口反查已有实现及调用方，不允许凭本任务标题直接新建同名系统`。

**现象与证据**：所有指标需要有定义、方向、适用条件、量纲、有效样本和证据状态；现有14维可沿用扩子维。

**影响**：这是原始需求的完整性验收，不是已证明全仓缺失的bug；没有运行证据不能标完成。

**修改要求**：为alpha_signal/model_feature/production分别冻结required与optional指标集合、raw/annualized口径、effect grade与evidence grade。证据完整性是硬门，展示总分无准入权。根据历史时间前推校准门槛；现有IC .02/.022/.7等不是论文证明的普适私募标准。

**验收**：
1. 同metrics不同用途按预声明policy不同决策。
2. 未算/不适用/不足/失败分开。
3. 高总分不能覆盖PIT失败。
4. 阈值边界和单位换算测试。

**依赖**：QE-27, FA-01, FA-04。 **需求映射**：R03。

**V3实施展开——具体怎么改**

建立metric→dimension→purpose三层绑定表，全部required/optional/不可适用条件机器校验；为每种形状和用途预声明门槛，效果和证据强度分开。

**V3追加回归——防止只修表面**

高总分+一个完整性失败必须拒绝；可选指标不足不显示为差性能。

**V3历史对象与兼容性处置**

threshold只在历史开发数据校准，sealed结果不参与微调。

**关联专项**：[STA-08](#issue-sta-08)、[STA-12](#issue-sta-12)。这些专项可在同一根因PR落实。

---

<a id="issue-v-04"></a>
#### V-04 · U/倒U与尾部修复需要低自由度实现、父子重评和完整血缘

**优先级**：P1　**证据分类**：需求验收／待穿透核查　**Owner**：FO/FE/QE/FA

**定位规则**：尚未对全仓证明缺失；先从相关公开入口找到现有实现，证实已覆盖后直接复用。

**函数/边界**：`必须从真实公开入口反查已有实现及调用方，不允许凭本任务标题直接新建同名系统`。

**现象与证据**：已有形状检测与repair family声明，不等于实际DSL候选可落值、可复评。

**影响**：这是原始需求的完整性验收，不是已证明全仓缺失的bug；没有运行证据不能标完成。

**修改要求**：优先复用rank、abs、subtract、square、clip/hinge组合；c=.5作无拟合baseline，拟合center/knots仅训练内。单调映射不能修复真实U；左右branch作为同族子因子，父/子/重组各自重评；避免挑测试集最优分支。

**验收**：
1. 稳定U通过对称映射提高方向性而非回看test。
2. 非对称U可保留左右不同通道。
3. 没有合成表达能力时列FE算子补充表及parity，不在FP写第二套。

**依赖**：QE-03, QE-34, FP-07。 **需求映射**：R04, R11, R14。

**V3实施展开——具体怎么改**

诊断产生RepairIntent，FO仅提案，FP编译，FE执行，QE复评，FA保存parent-child。对称/偏斜U与tail clipping分不同有界搜索族，父子共享campaign。

**V3追加回归——防止只修表面**

保存原始曲线、拟合中心、最终DSL及左右通道；每个子因子和重组都有独立评价。

**V3历史对象与兼容性处置**

同一变换不能同时作为全新原创因子规避试验次数。

**关联专项**：[RCP-07](#issue-rcp-07)、[RCP-08](#issue-rcp-08)。这些专项可在同一根因PR落实。

---

<a id="issue-v-05"></a>
#### V-05 · 廉价初筛不能提前杀掉非线性、稀疏事件或互补特征

**优先级**：P1　**证据分类**：需求验收／待穿透核查　**Owner**：FO/QE/FA

**定位规则**：尚未对全仓证明缺失；先从相关公开入口找到现有实现，证实已覆盖后直接复用。

**函数/边界**：`必须从真实公开入口反查已有实现及调用方，不允许凭本任务标题直接新建同名系统`。

**现象与证据**：原始RankIC低可能是U型或对已有模型具有交互价值，也可能只是噪声；两者不能一视同仁。

**影响**：这是原始需求的完整性验收，不是已证明全仓缺失的bug；没有运行证据不能标完成。

**修改要求**：廉价阶段保留10层profile、离散/稀疏分支与事件覆盖条件；高成本模型增量只给幸存者。基于predeclared universe报告coverage，不能为过门槛临时改股票池。

**验收**：
1. 低线性IC但稳定U样本进入形状支路。
2. 纯噪声不能因支路多获无穷尝试。
3. 事件型按已声明适用空间评估。

**依赖**：QE-03, FA-05, PL-04。 **需求映射**：R03, R04, R08, R11。

**V3实施展开——具体怎么改**

初筛给单调alpha、稳定非线性、稀疏事件和增量特征四条有界支路；可修复证据不等于无限试。首次低IC只触发支路选择，不即删raw研究档案。

**V3追加回归——防止只修表面**

纯噪声不会靠任意翻转/分段获无限尝试；事件稀疏覆盖有已声明分母。

**V3历史对象与兼容性处置**

阈值与支路在搜索前锁定。

**关联专项**：[MOD-02](#issue-mod-02)。这些专项可在同一根因PR落实。

---

<a id="issue-v-06"></a>
#### V-06 · 中性化是受约束的候选选择，必须验证最终暴露而非名字

**优先级**：P1　**证据分类**：需求验收／待穿透核查　**Owner**：FP/FE/QE/FO

**定位规则**：尚未对全仓证明缺失；先从相关公开入口找到现有实现，证实已覆盖后直接复用。

**函数/边界**：`必须从真实公开入口反查已有实现及调用方，不允许凭本任务标题直接新建同名系统`。

**现象与证据**：用户倾向效果差不多时中性化；不能假定中性化必提升Sharpe或抵御所有小盘风险。

**影响**：这是原始需求的完整性验收，不是已证明全仓缺失的bug；没有运行证据不能标完成。

**修改要求**：对raw、industry、industry+size等少量预声明候选作成对样本外非劣性比较；收益差值、置信区间和风险改善共同决策。OLS/WLS/ridge/HUBER可不同，不能对ridge宣称OLS精确正交；输出scale/rank/clip后重验postcondition。模型和组合暴露分开。

**验收**：
1. 同mask/权重OLS残差满足对应正交条件。
2. rank后的暴露重新计算。
3. 效果显著变差不默认硬套中性化。
4. 不同数据可用时间不可混用。

**依赖**：FP-06, FP-08。 **需求映射**：R05, R13。

**V3实施展开——具体怎么改**

写paired evaluation API，以同窗、同样本和同资源预算比较raw及风险受控版本；输出差值CI、实际暴露变化与成本变化。neutrality是带mask/权重的postcondition。

**V3追加回归——防止只修表面**

固定data，改变仅风险控制字段的available time，违规样本要阻断；近效用优先中性仍需非劣证据。

**V3历史对象与兼容性处置**

raw和neutralized可以共存，但有明确用途和族上限。

**关联专项**：[STA-03](#issue-sta-03)、[RCP-04](#issue-rcp-04)、[RCP-05](#issue-rcp-05)、[MOD-03](#issue-mod-03)。这些专项可在同一根因PR落实。

---

<a id="issue-v-07"></a>
#### V-07 · 所有可生产滤波通过前缀不变与增量状态回放

**优先级**：P0　**证据分类**：需求验收／待穿透核查　**Owner**：FE/FP/QA

**定位规则**：尚未对全仓证明缺失；先从相关公开入口找到现有实现，证实已覆盖后直接复用。

**函数/边界**：`必须从真实公开入口反查已有实现及调用方，不允许凭本任务标题直接新建同名系统`。

**现象与证据**：不能只把causal_safe=True写在元数据上；当前合法时间点和warm-up状态必须验证。

**影响**：这是原始需求的完整性验收，不是已证明全仓缺失的bug；没有运行证据不能标完成。

**修改要求**：优先raw+少量EWMA半衰期，必要再启用单侧Kalman/IIR；禁用filtfilt/centered roll/backfill的生产信号路径。label生成允许明确未来shift，但不得污染factor变换。状态携带序列末时点、warmup、缺失/陈旧更新、资产生命周期。

**验收**：
1. 修改未来全部价格/公告不改变历史输出。
2. full vs逐日incremental vs分块一致。
3. 重启checkpoint回放一致。
4. 新证券无其他证券状态污染。

**依赖**：QE-30, FP-03, FP-04。 **需求映射**：R06, R14, R15。

**V3实施展开——具体怎么改**

state key绑定recipe/version/instrument/timewatermark；全量和增量使用同一核心递推。事件重复值的更新政策与市场休市不同，checkpoint须包括这些参数。

**V3追加回归——防止只修表面**

future poison、随机chunk切分、重启、资产新增、短缺失和长陈旧分别回放。

**V3历史对象与兼容性处置**

有状态算子未通过增量测试不能进每日生产更新。

**关联专项**：[DTA-10](#issue-dta-10)、[RCP-06](#issue-rcp-06)、[QAT-06](#issue-qat-06)。这些专项可在同一根因PR落实。

---

<a id="issue-v-08"></a>
#### V-08 · 树模型与神经网络的输入表示有独立契约，不统一强制zscore

**优先级**：P1　**证据分类**：需求验收／待穿透核查　**Owner**：FP/modeling/FE

**定位规则**：尚未对全仓证明缺失；先从相关公开入口找到现有实现，证实已覆盖后直接复用。

**函数/边界**：`必须从真实公开入口反查已有实现及调用方，不允许凭本任务标题直接新建同名系统`。

**现象与证据**：树、NN、线性模型对尺度和缺失需求不同；截面逐日rank不是全数据上同一单调变换。

**影响**：这是原始需求的完整性验收，不是已证明全仓缺失的bug；没有运行证据不能标完成。

**修改要求**：提供raw-clean/tree/neural/linear profiles，必要保留raw+neutralized多通道；任何跨时间fit标准化仅训练内，按fold重建。cs zscore与ts zscore分开；重复处理判断来自最终DAG性质；模型消费前校验schema/分布/缺失。

**验收**：
1. 训练/验证标准化状态隔离。
2. 新输入不重新fit测试分布。
3. tree无需为统一量纲无条件丢幅度。
4. 已有有效最终zscore不重复。

**依赖**：FP-07, FP-08, FP-11。 **需求映射**：R11, R13, R14。

**V3实施展开——具体怎么改**

按consumer profile编译表示：线性、树、神经、原始通道。state ref必须落在训练边界；模型adapter显式读feature order和mask，不能自动fit新批。

**V3追加回归——防止只修表面**

同一OOF样本全部预处理state来自它之前的训练fold；缺失indicator不覆盖信号channel。

**V3历史对象与兼容性处置**

消费profile变化是新特征版本而非展示选项。

**关联专项**：[RCP-01](#issue-rcp-01)、[MOD-01](#issue-mod-01)、[MOD-03](#issue-mod-03)、[MOD-04](#issue-mod-04)。这些专项可在同一根因PR落实。

---

<a id="issue-v-09"></a>
#### V-09 · 诊断驱动小预算搜索，而非穷举所有预处理排列

**优先级**：P1　**证据分类**：需求验收／待穿透核查　**Owner**：FO

**定位规则**：尚未对全仓证明缺失；先从相关公开入口找到现有实现，证实已覆盖后直接复用。

**函数/边界**：`必须从真实公开入口反查已有实现及调用方，不允许凭本任务标题直接新建同名系统`。

**现象与证据**：搜索空间必须节省算力，并维持每个trial都能复现。

**影响**：这是原始需求的完整性验收，不是已证明全仓缺失的bug；没有运行证据不能标完成。

**修改要求**：raw永远作为基线；每个诊断只激活必要repair family，首轮建议最多8–12个新候选、每family2–3个取点（可版本化调整）。连续参数先取预声明邻域，找稳定平台而非尖峰；OOM/超时/语法失败都入账但不伪造指标。

**验收**：
1. 高换手只打开因果平滑/窗口微调而非全部family。
2. budget含失败与重复检查成本。
3. 多进程预算reservation原子。
4. 无改善早停不影响sealed纪律。

**依赖**：FO-01, FO-04。 **需求映射**：R08, R09。

**V3实施展开——具体怎么改**

每family激活条件、参数邻域、成本和候选上限写policy；先raw与最相关2–4候选，再按证据小范围扩展，8–12只是起始上限而非必须跑满。

**V3追加回归——防止只修表面**

同一候选变异后canonical不变则不重复贵计算，但proposal台账保留。

**V3历史对象与兼容性处置**

资源次数和统计假设次数分开，避免缓存命中改写多重检验含义。

**关联专项**：[STA-10](#issue-sta-10)、[RCP-03](#issue-rcp-03)、[SRH-01](#issue-srh-01)、[SRH-02](#issue-srh-02)、[SRH-06](#issue-srh-06)、[SRH-08](#issue-srh-08)、[SRH-09](#issue-srh-09)。这些专项可在同一根因PR落实。

---

<a id="issue-v-10"></a>
#### V-10 · 多保真初筛与精筛用相同身份、明确样本，不能拿低精度分数冒充最终证据

**优先级**：P1　**证据分类**：需求验收／待穿透核查　**Owner**：FO/QE

**定位规则**：尚未对全仓证明缺失；先从相关公开入口找到现有实现，证实已覆盖后直接复用。

**函数/边界**：`必须从真实公开入口反查已有实现及调用方，不允许凭本任务标题直接新建同名系统`。

**现象与证据**：小时间窗或子股票池可以省钱，但会改变分层/IC和风格；低保真比较不是免费无偏估计。

**影响**：这是原始需求的完整性验收，不是已证明全仓缺失的bug；没有运行证据不能标完成。

**修改要求**：每tier带时间、universe、horizon、sampling seed、metric profile与预算；优先抽连续时间块保留截面结构。晋级要在共同正式样本重算；早停导致缺失不等于失败绩效。

**验收**：
1. 同一候选低精度不覆盖完整证据。
2. 参数平台区在独立validation确认。
3. 声明GPU快速档实际运行路径可追踪。

**依赖**：QE-28, FO-04。 **需求映射**：R08, R09, R10。

**V3实施展开——具体怎么改**

stage记录样本时间块、universe、label成熟度、metric profile与sampling seed；最后比较必须在同一确认样本和相同资源目标上进行。

**V3追加回归——防止只修表面**

低保真效果更高不能直接淘汰已全量验证候选；同一个value的多stage证据独立可查。

**V3历史对象与兼容性处置**

低档schema不能覆盖高档artifact。

**关联专项**：[SRH-05](#issue-srh-05)。这些专项可在同一根因PR落实。

---

<a id="issue-v-11"></a>
#### V-11 · 训练、验证、封存测试和跨session试验次数必须联合治理

**优先级**：P0　**证据分类**：需求验收／待穿透核查　**Owner**：FO/QE/FA

**定位规则**：尚未对全仓证明缺失；先从相关公开入口找到现有实现，证实已覆盖后直接复用。

**函数/边界**：`必须从真实公开入口反查已有实现及调用方，不允许凭本任务标题直接新建同名系统`。

**现象与证据**：封存不能只存在单session内存布尔值；改因子名字或开启新session也不能反复偷看同一test。

**影响**：这是原始需求的完整性验收，不是已证明全仓缺失的bug；没有运行证据不能标完成。

**修改要求**：先在train/validation选择并冻结候选集合、方向和recipe，再一次性消费test；测试失败不能同窗重调。campaign级计数继承母因子全部参数、预处理、分支、Agent尝试与失败；DSR/PBO/FDR按实际假设族与相依结构使用，不机械把某一p值当盈利保证。

**验收**：
1. 跨进程/换session/改alias不能重置test权限和trial ledger。
2. 多个候选封存的是集合，不是事后挑赢家。
3. 删除values不删试验统计。

**依赖**：FO-02, FO-03, FO-05。 **需求映射**：R09, R10, R11, R16。

**V3实施展开——具体怎么改**

将test scope与campaign/数据版本绑定，先锁候选集合再执行；跨session复用结果必须读已存证据而不是新一次权限。真实执行失败和已经泄露结果需不同恢复策略。

**V3追加回归——防止只修表面**

换alias、开新session、恢复旧checkpoint均不能重新解封同一test；多个冻结候选按预声明集合裁决。

**V3历史对象与兼容性处置**

代码修复后的旧test只可故障重放，不能改标fresh。

**关联专项**：[STA-02](#issue-sta-02)、[STA-04](#issue-sta-04)、[STA-05](#issue-sta-05)、[STA-06](#issue-sta-06)、[SRH-07](#issue-srh-07)、[MOD-01](#issue-mod-01)。这些专项可在同一根因PR落实。

---

<a id="issue-v-12"></a>
#### V-12 · Purging、embargo和PIT按真实标签区间验证，避免前视与过度清洗

**优先级**：P0　**证据分类**：需求验收／待穿透核查　**Owner**：DA/QE/FO

**定位规则**：尚未对全仓证明缺失；先从相关公开入口找到现有实现，证实已覆盖后直接复用。

**函数/边界**：`必须从真实公开入口反查已有实现及调用方，不允许凭本任务标题直接新建同名系统`。

**现象与证据**：需要区分合法读取过去lookback与标签区间重叠；并非所有跨fold历史特征都属于泄漏。

**影响**：这是原始需求的完整性验收，不是已证明全仓缺失的bug；没有运行证据不能标完成。

**修改要求**：按label start/end与decision时可用数据构建purged forward splits；明确holdout边界、embargo和成熟cutoff；基本面按knowledge/announcement time而非report period，复权、行业、指数成员及退市保留PIT。

**验收**：
1. 报表修订/晚公告/停牌/跨时区样本手算。
2. 训练标签越界拒绝。
3. 验证使用合法过去lookback仍允许。
4. 任何test拟合中心/scale/cluster selection拒绝。

**依赖**：QE-29, QE-30, QE-31。 **需求映射**：R10, R15。

**V3实施展开——具体怎么改**

切分按每个样本实际label interval和成熟时间，而非机械n行间隔；验证期允许只读过去lookback。暴露/成分/财报都应用knowledge-time join。

**V3追加回归——防止只修表面**

边界相接区间的开闭语义、不同H标签、跨休市跨度、报告修订全部手算金标。

**V3历史对象与兼容性处置**

过度purge同样算问题，需报告有效样本损失及理由。

**关联专项**：[DTA-02](#issue-dta-02)、[DTA-11](#issue-dta-11)、[STA-06](#issue-sta-06)、[STA-09](#issue-sta-09)、[MOD-01](#issue-mod-01)。这些专项可在同一根因PR落实。

---

<a id="issue-v-13"></a>
#### V-13 · 实盘指标必须带可执行性与成本规格，研究probe另设身份

**优先级**：P1　**证据分类**：需求验收／待穿透核查　**Owner**：QE/回测执行域/DA

**定位规则**：尚未对全仓证明缺失；先从相关公开入口找到现有实现，证实已覆盖后直接复用。

**函数/边界**：`必须从真实公开入口反查已有实现及调用方，不允许凭本任务标题直接新建同名系统`。

**现象与证据**：多空Sharpe不能替代实际可交易收益；A股短腿、容量和税费需要具体交易假设与数据。

**影响**：这是原始需求的完整性验收，不是已证明全仓缺失的bug；没有运行证据不能标完成。

**修改要求**：从现有回测/执行库获取daily holdings/cash/PnL/turnover artifact，再由QE算指标；记录long-only active和long-short diagnostic、gross/net、杠杆资本口径、冲击、借券、涨跌停侧别、成交延迟。缺数据则NO_CAPACITY_EVIDENCE而非0成本。

**验收**：
1. 手续费提高净收益不应提高。
2. 无借券不能称实盘可执行short。
3. 不同持有期资金重叠正确。
4. 尾部不可成交不被事后删选。

**依赖**：QE-09, QE-23, QE-24, QE-25。 **需求映射**：R06, R07。

**V3实施展开——具体怎么改**

QE指标消费执行域真实成交轨迹，不能暗造交易；规格含目标资本、净/毛敞口、借券、成本、成交窗口和缺价规则。仅probe可用时明确限制准入。

**V3追加回归——防止只修表面**

固定价格成本ledger、卖不出与借不到券、部分成交、净额cohort对冲分别验证。

**V3历史对象与兼容性处置**

实际税费规则按市场和生效日期配置，缺数据不写0。

**关联专项**：[DTA-05](#issue-dta-05)。这些专项可在同一根因PR落实。

---

<a id="issue-v-14"></a>
#### V-14 · 相似度去重是分层流程，不能一把0.7/0.95阈值删除所有高相关因子

**优先级**：P1　**证据分类**：需求验收／待穿透核查　**Owner**：FA/QE/FE

**定位规则**：尚未对全仓证明缺失；先从相关公开入口找到现有实现，证实已覆盖后直接复用。

**函数/边界**：`必须从真实公开入口反查已有实现及调用方，不允许凭本任务标题直接新建同名系统`。

**现象与证据**：公式相同、数值相同、极高相关、同族和模型冗余是不同概念。

**影响**：这是原始需求的完整性验收，不是已证明全仓缺失的bug；没有运行证据不能标完成。

**修改要求**：先canonical definition/recipe/value精确去重，再高相关候选复核、ANN召回、QE正式相关。0.99/0.995可作近重复复核起点，0.95通常同族候选，均待校准；跨窗口/sign/mask/coverage/成本/残差IC共同决定替换或保留。

**验收**：
1. f与-f识别方向别名但保留方向证据。
2. 只在部分日期高度相似不当全期等价。
3. 极高相关但成本或增量差异明确可以保留不同用途。

**依赖**：FA-07, FA-09。 **需求映射**：R11, R12。

**V3实施展开——具体怎么改**

四级身份/数值/近重复/增量筛查分别输出判断理由；相似相关阈值是候选审查条件，不是万能删除线。反向因子保存方向别名。

**V3追加回归——防止只修表面**

高corr但尾部和净成本不同能保留不同用途；无共同样本不误称低相关。

**V3历史对象与兼容性处置**

近重复淘汰只删无人引用的物化值，保留公式和族血缘。

**关联专项**：[SIM-01](#issue-sim-01)、[SIM-03](#issue-sim-03)、[SIM-04](#issue-sim-04)、[SIM-07](#issue-sim-07)。这些专项可在同一根因PR落实。

---

<a id="issue-v-15"></a>
#### V-15 · 新增簇、增量归属和全量重聚类有明确版本与触发

**优先级**：P1　**证据分类**：需求验收／待穿透核查　**Owner**：FA/modeling

**定位规则**：尚未对全仓证明缺失；先从相关公开入口找到现有实现，证实已覆盖后直接复用。

**函数/边界**：`必须从真实公开入口反查已有实现及调用方，不允许凭本任务标题直接新建同名系统`。

**现象与证据**：用户不希望每来一个因子就重训全部；也不能把所有新因子硬塞旧簇。

**影响**：这是原始需求的完整性验收，不是已证明全仓缺失的bug；没有运行证据不能标完成。

**修改要求**：研究簇允许PROVISIONAL_NEW/AMBIGUOUS/PENDING；正式ClusterSetVersion冻结。定期/漂移触发重新建图或重聚类，记录split/merge/retire血缘；触发阈值按稳定性和变化比例校准，不称每月/每季必然最优。

**验收**：
1. 新族确实可产生新簇。
2. 未知affinity不自动合并。
3. 全局refresh不原地改旧版本。
4. 产出cluster quality和稳定性检查。

**依赖**：FA-08, FA-09, FA-10。 **需求映射**：R12。

**V3实施展开——具体怎么改**

研究增量overlay不改production cluster set；新族观察区、歧义区、全局refresh入口都要可运行。刷新输出split/merge映射及质量报告。

**V3追加回归——防止只修表面**

新因子不匹配任何簇时存在合法状态；换ref旧簇仍可回放；重训只在输入函数变动时触发。

**V3历史对象与兼容性处置**

图算法参数和随机seed进入cluster hash。

**关联专项**：[SIM-09](#issue-sim-09)。这些专项可在同一根因PR落实。

---

<a id="issue-v-16"></a>
#### V-16 · 筛选允许少量互补版本，而不是单冠军或全收

**优先级**：P1　**证据分类**：需求验收／待穿透核查　**Owner**：FO/FA/QE

**定位规则**：尚未对全仓证明缺失；先从相关公开入口找到现有实现，证实已覆盖后直接复用。

**函数/边界**：`必须从真实公开入口反查已有实现及调用方，不允许凭本任务标题直接新建同名系统`。

**现象与证据**：因子最终用途可能是独立alpha、低换手替代、纯风格残差或模型互补特征。

**影响**：这是原始需求的完整性验收，不是已证明全仓缺失的bug；没有运行证据不能标完成。

**修改要求**：先hard gates再多目标Pareto和统计等价区间；等价时优先简单稳定低成本；默认母因子保留1–3个有明确差异的候选作为预算起点，不以quota强收。最终需相对当前库/模型的OOS增量价值。

**验收**：
1. 参数19/20近等价只能合并或代表。
2. raw/低换手/中性三个版本需各自用途证据。
3. 无增量的高总分不能挤满模型输入。

**依赖**：FA-11, FO-01。 **需求映射**：R09, R11。

**V3实施展开——具体怎么改**

先hard gate，再少量主要目标Pareto，再统计等价时选简单低成本，再做库/模型增量验证。多赢家输出集合与各自用途，不再硬编码single winner。

**V3追加回归——防止只修表面**

同族候选1–3是默认上限非配额；全不合格返回空集；都相同只留代表及alias。

**V3历史对象与兼容性处置**

扩展候选保留数量也计搜索/选择政策，不能事后挑测试通过者。

**关联专项**：[SRH-03](#issue-srh-03)、[SRH-04](#issue-srh-04)、[SIM-03](#issue-sim-03)、[MOD-02](#issue-mod-02)。这些专项可在同一根因PR落实。

---

<a id="issue-v-17"></a>
#### V-17 · 父因子拆分和簇聚合都生成新资产，必须再次评价

**优先级**：P1　**证据分类**：需求验收／待穿透核查　**Owner**：FA/QE/modeling

**定位规则**：尚未对全仓证明缺失；先从相关公开入口找到现有实现，证实已覆盖后直接复用。

**函数/边界**：`必须从真实公开入口反查已有实现及调用方，不允许凭本任务标题直接新建同名系统`。

**现象与证据**：左右分支、horizon组合或cluster平均可以比单个更差；不能继承子因子评级作为组合评级。

**影响**：这是原始需求的完整性验收，不是已证明全仓缺失的bug；没有运行证据不能标完成。

**修改要求**：任何有值函数变化的组合创建新definition/recipe/value/evaluation refs；评估父、子、组合及互补性；权重只在训练/验证内拟合并冻结，诊断簇不必强制压成一个均值特征。

**验收**：
1. 两个单独有效但合成抵消的因子组合被识别。
2. 簇成员变化会改变feature definition并触发模型再验证。
3. 加入新成员不悄悄改生产列。

**依赖**：QE-01, PL-06。 **需求映射**：R04, R11, R12, R14。

**V3实施展开——具体怎么改**

父子/簇组合以可计算定义建新资产；符号方向、归一、权重、缺成员规则明确。无独立重评不得继承任一成员的A级健康卡。

**V3追加回归——防止只修表面**

输入两相反信号不应合成好alpha；动态缺一个成员不准自动重归一而不改规格。

**V3历史对象与兼容性处置**

旧组合仅由成员评分推断的评价需重建。

**关联专项**：[SIM-10](#issue-sim-10)。这些专项可在同一根因PR落实。

---

<a id="issue-v-18"></a>
#### V-18 · 定义、配方、落值、评价、簇与模型身份串成可追溯链

**优先级**：P1　**证据分类**：需求验收／待穿透核查　**Owner**：FE/DA/QE/FA/FP/modeling

**定位规则**：尚未对全仓证明缺失；先从相关公开入口找到现有实现，证实已覆盖后直接复用。

**函数/边界**：`必须从真实公开入口反查已有实现及调用方，不允许凭本任务标题直接新建同名系统`。

**现象与证据**：原始DSL与后处理分开存在时，必须明确知道每一步以及对应数据。

**影响**：这是原始需求的完整性验收，不是已证明全仓缺失的bug；没有运行证据不能标完成。

**修改要求**：冻结canonical DSL、ordered typed recipe、effective operator IDs/versions、fit state、clock、universe、data snapshot、dtype/mask/ties/ddof、metric policy、cluster/model schema；一个完整FactorExecutionSpec入口用于回放，展示可读DSL+sidecar状态。

**验收**：
1. 任一实际数学参数或dataset快照变化导致正确身份变动。
2. 显示名改变不改变数学身份。
3. 从模型列可以找到原始字段和每次变换。

**依赖**：FP-03, FP-08, FP-11, PL-05, PL-07。 **需求映射**：R13, R14, R15。

**V3实施展开——具体怎么改**

FactorExecutionSpec贯穿definition→recipe→state→value→eval→policy→cluster/feature/model，公开解释页可显示完整DSL及sidecar。每一跳有可解析typed ref。

**V3追加回归——防止只修表面**

从生产模型任一列可追到字段知识时间和effective参数；展示名不参与数学身份。

**V3历史对象与兼容性处置**

不可解析的旧链保留研究档案，不伪造生产引用。

**关联专项**：[RCP-09](#issue-rcp-09)、[RCP-10](#issue-rcp-10)、[SRH-10](#issue-srh-10)、[SIM-06](#issue-sim-06)、[SIM-10](#issue-sim-10)、[MOD-05](#issue-mod-05)、[OPS-08](#issue-ops-08)、[OPS-09](#issue-ops-09)。这些专项可在同一根因PR落实。

---

<a id="issue-v-19"></a>
#### V-19 · 每日因子更新执行冻结配方，不重新跑优化器

**优先级**：P1　**证据分类**：需求验收／待穿透核查　**Owner**：FE/DA/FP/FA

**定位规则**：尚未对全仓证明缺失；先从相关公开入口找到现有实现，证实已覆盖后直接复用。

**函数/边界**：`必须从真实公开入口反查已有实现及调用方，不允许凭本任务标题直接新建同名系统`。

**现象与证据**：用户要求实盘有一条固定自动更新线；有recipe文本不代表增量服务可运行。

**影响**：这是原始需求的完整性验收，不是已证明全仓缺失的bug；没有运行证据不能标完成。

**修改要求**：执行状态以资产和recipe版本隔离，checkpoint含warmup、最后处理时点、数据watermark。延迟数据/修订触发明确recompute范围，原子写新generation；失败保留旧正式指针并告警。

**验收**：
1. 全量历史与每天增量同值。
2. 重启无重复/漏算。
3. 数据修订按政策更新并保留旧snapshot。
4. 更新任务不调用FO search。

**依赖**：PL-07, PL-08, V-07。 **需求映射**：R14, R15。

**V3实施展开——具体怎么改**

增量更新任务只执行冻结计划，使用DA watermark触发与FE state推进；晚到修订按依赖lookback求重算区间；成功generation完成后再切读指针。

**V3追加回归——防止只修表面**

按日更新、分块补算、整段重算在相同快照下等价；同日重复触发不重复写最终值。

**V3历史对象与兼容性处置**

可变线上重新择优禁止混入更新任务。

**关联专项**：[DTA-10](#issue-dta-10)、[OPS-09](#issue-ops-09)、[QAT-06](#issue-qat-06)。这些专项可在同一根因PR落实。

---

<a id="issue-v-20"></a>
#### V-20 · 失败values清理应真实执行且不破坏生产、共享DAG或统计历史

**优先级**：P0　**证据分类**：需求验收／待穿透核查　**Owner**：FA/DA/Platform

**定位规则**：尚未对全仓证明缺失；先从相关公开入口找到现有实现，证实已覆盖后直接复用。

**函数/边界**：`必须从真实公开入口反查已有实现及调用方，不允许凭本任务标题直接新建同名系统`。

**现象与证据**：公式、操作配方和trial/evidence需要保留；大块失败values和缓存不应长期占存储。

**影响**：这是原始需求的完整性验收，不是已证明全仓缺失的bug；没有运行证据不能标完成。

**修改要求**：决策终态后以引用/租约/共享节点检查生成GC plan→tombstone→删除对象/缓存→校验与receipt；可设置短诊断TTL并允许人工保留bug样本。禁止仅删数据库行或按目录宽泛rm；生产快照、父子依赖、待重试值和保留账本不可误删。

**验收**：
1. 被生产/子因子引用的失败中间节点不删。
2. 0引用失败values实际删除并回收空间。
3. 清理进程中断可重放。
4. 试验次数、formula/recipe/evidence/meta仍可查。

**依赖**：PL-02, PL-07, PL-08, V-11。 **需求映射**：R16。

**V3实施展开——具体怎么改**

GC plan由FA引用保留政策生成，DA执行对象删除。生产、回滚窗口、共享DAG和正在读取的lease都是roots。删除回执核验对象版本与cache，不止删DB行。

**V3追加回归——防止只修表面**

两worker清理同对象幂等；扫描后新建引用的race安全；失败trial记录永久保留统计身份。

**V3历史对象与兼容性处置**

生产破坏性删除必须原权限审批；默认只dry-run和隔离测试空间执行。

**关联专项**：[RCP-09](#issue-rcp-09)、[SRH-10](#issue-srh-10)、[OPS-05](#issue-ops-05)、[OPS-10](#issue-ops-10)。这些专项可在同一根因PR落实。

---

<a id="issue-v-21"></a>
#### V-21 · 上线后诊断与漂移监测不重用测试集调参

**优先级**：P1　**证据分类**：需求验收／待穿透核查　**Owner**：FA/QE/Platform

**定位规则**：尚未对全仓证明缺失；先从相关公开入口找到现有实现，证实已覆盖后直接复用。

**函数/边界**：`必须从真实公开入口反查已有实现及调用方，不允许凭本任务标题直接新建同名系统`。

**现象与证据**：应计算成熟新样本的预测退化、成本/暴露/覆盖漂移，而不是靠旧静态高分永久有效。

**影响**：这是原始需求的完整性验收，不是已证明全仓缺失的bug；没有运行证据不能标完成。

**修改要求**：健康卡随新evaluation context更新，retain历史；设警戒/隔离/重新研究状态；暂未成熟指标UNKNOWN；新研究campaign继承试验族，替换生产需独立验证与回滚计划。

**验收**：
1. 新样本不成熟不计算假0IC。
2. 出现缺数据先数据告警而非宣称策略失效。
3. 风险门违规停止晋级，非自动调参直到回测变好。

**依赖**：按波次/主合同安排，无额外单项依赖。 **需求映射**：R03, R10, R14, R15。

**V3实施展开——具体怎么改**

线上monitor只看截至当日成熟标签；先区分data故障、分布漂移、暴露偏移和预测退化，给不同动作。重研究新campaign，旧模型继续固定配方或按风险政策隔离。

**V3追加回归——防止只修表面**

数据中断时不显示IC突然归零；未成熟窗口不参与退化阈值；升级需shadow/release而非自动替换。

**V3历史对象与兼容性处置**

保留每次健康卡状态变更与可见信息范围。

**关联专项**：[STA-07](#issue-sta-07)、[STA-09](#issue-sta-09)、[MOD-08](#issue-mod-08)。这些专项可在同一根因PR落实。

---

<a id="issue-v-22"></a>
#### V-22 · Agent仅作有界提案和解释，不承担不可验证的硬门

**优先级**：P1　**证据分类**：需求验收／待穿透核查　**Owner**：FO/FA/Platform

**定位规则**：尚未对全仓证明缺失；先从相关公开入口找到现有实现，证实已覆盖后直接复用。

**函数/边界**：`必须从真实公开入口反查已有实现及调用方，不允许凭本任务标题直接新建同名系统`。

**现象与证据**：用户允许Agent辅助，但不是每步必须调用LLM，更不能用它覆盖PIT/证据/预算限制。

**影响**：这是原始需求的完整性验收，不是已证明全仓缺失的bug；没有运行证据不能标完成。

**修改要求**：确定性规则先执行；Agent从白名单repair family和参数域提JSON提案，携带证据ref和预算。代码validator拒绝不合法操作。机制解释区分建议/确认；严禁test可见、阈值自降、伪造指标、触发无授权清理。

**验收**：
1. 恶意/错误提案、未知算子、超预算、未来字段全部拒绝。
2. LLM不可用不影响确定性主链。
3. 相同最终recipe可脱离Agent复现。

**依赖**：FP-09, FP-10, V-11。 **需求映射**：R17, R14。

**V3实施展开——具体怎么改**

Agent输出结构化RepairProposal，validator独立验证字段/算子/参数/权限/预算。保存最终提案及模型/提示词版本；回放不再次调用Agent。

**V3追加回归——防止只修表面**

prompt injection要求读test、降门或生成未知算子一律不能扩大权限；LLM不可用走确定性基线。

**V3历史对象与兼容性处置**

Agent建议、经规则确认的机制和统计证据分别标来源。

**关联专项**：[STA-12](#issue-sta-12)。这些专项可在同一根因PR落实。

---

<a id="issue-v-23"></a>
#### V-23 · 真实端到端与硬件基准必须覆盖公共入口，而不止内核mock

**优先级**：P0　**证据分类**：需求验收／待穿透核查　**Owner**：QA/全模块

**定位规则**：尚未对全仓证明缺失；先从相关公开入口找到现有实现，证实已覆盖后直接复用。

**函数/边界**：`必须从真实公开入口反查已有实现及调用方，不允许凭本任务标题直接新建同名系统`。

**现象与证据**：本次只进行了隔离表达式复现；未在仓库测试环境、服务器GPU或私有行情上执行完整链。

**影响**：这是原始需求的完整性验收，不是已证明全仓缺失的bug；没有运行证据不能标完成。

**修改要求**：在AI拥有的checkout先运行小型synthetic真实FE→FP→QE→FO→FA→平台→模型输入链；GPU硬件路径单独运行并报告dtype/tolerance/status/backend。后续私有数据跑受限样本并保留snapshot。依赖不可用时输出BLOCKED与准确命令，不造通过记录。

**验收**：
1. 每个required metric都有公开API/adapter/health-card/入库验收。
2. GPU缺失不能算parity pass。
3. 耗时含I/O、传输、hash、排序、缓存和落盘。

**依赖**：QA-01, QA-02。 **需求映射**：R08, R14, R15。

**V3实施展开——具体怎么改**

端到端分synthetic真实库链、批准快照数据链、真实GPU硬件链三级；每层有trace和场景，mock只测协议。记录冷/热缓存与测量边界。

**V3追加回归——防止只修表面**

public API->adapter->gate完整覆盖；没有GPU则GPU验收BLOCKED不是skip=pass；同一数据所有布局对拍。

**V3历史对象与兼容性处置**

本轮提供的是任务规格，不能把原25个表达式重放当本条完成。

**关联专项**：[OPS-03](#issue-ops-03)、[OPS-04](#issue-ops-04)、[QAT-01](#issue-qat-01)、[QAT-02](#issue-qat-02)、[QAT-03](#issue-qat-03)。这些专项可在同一根因PR落实。

---

<a id="issue-v-24"></a>
#### V-24 · CI和最终验收追踪所有发现、迁移以及既有证据失效范围

**优先级**：P1　**证据分类**：需求验收／待穿透核查　**Owner**：QA/总协调

**定位规则**：尚未对全仓证明缺失；先从相关公开入口找到现有实现，证实已覆盖后直接复用。

**函数/边界**：`必须从真实公开入口反查已有实现及调用方，不允许凭本任务标题直接新建同名系统`。

**现象与证据**：修对源码后，旧cache、旧指标、旧health cards、旧模型输入仍可能依赖错误定义。

**影响**：这是原始需求的完整性验收，不是已证明全仓缺失的bug；没有运行证据不能标完成。

**修改要求**：任务状态固定DISCOVERED/REPRODUCED/FIXED_LOCAL/VERIFIED/BLOCKED/NOT_A_BUG；每项链接commit/test/log。提升算法语义版本并显式invalidate受影响证据，不能直接改旧artifact。旧test用于故障复核不再作为全新独立holdout。

**验收**：
1. 无改测试削弱门槛、无required skip冒充绿灯、无新重复math。
2. 所有source risk有实测结论。
3. affected artifacts清单、迁移dry-run和rollback完成。

**依赖**：V-01, V-23。 **需求映射**：R03, R14, R15, R16。

**V3实施展开——具体怎么改**

CI读取任务账本和需求矩阵，closed条目必须有可解析测试和版本证据；迁移按metric/recipe/data/policy依赖传播，保持旧制品只读。

**V3追加回归——防止只修表面**

修改kernel后失效清单覆盖下游而不无限全仓清空；不能关闭仍有required阻断的父任务。

**V3历史对象与兼容性处置**

最终列出VERIFIED、BLOCKED、NOT_A_BUG及新增发现，拒绝单一句all fixed。

**关联专项**：[OPS-08](#issue-ops-08)、[QAT-08](#issue-qat-08)。这些专项可在同一根因PR落实。

---

<a id="extension"></a>
## 12. 80项新增专项核查与实施细目

全部为E类：本轮从原需求和历史问题扩写为开发可执行的专项，并非本轮新确认源码Bug。先找真实实现与调用路径；已经覆盖就补证明并复用，未覆盖再改。下列“定位范围”是需调查的现有包/目录，不保证存在同名新API。每项依赖都是工作先后关系，不要求把所有依赖都变成新的Python import。

### 12.1 DTA：数据、PIT、日历与字段合同专项

<a id="issue-dta-01"></a>
#### DTA-01 · 证券永久身份、更名和跨市场同名不能共用时序状态

**优先级**：P0　**证据分类**：E（待实仓核查）　**Owner**：DA/FE

**现有定位范围**：`data_access`；`factor_engine`；`quant_evaluator/contracts`；`quant_platform/app/adapters`。

**需要排查的风险与为什么要改**

核查ticker、数值证券代码、交易所代码与永久instrument_id之间是否存在有损转换。即使表形状正确，重用代码也可能继承已退市证券的滤波状态。

**实施步骤**

1. 先找到DA现有证券目录，建立带effective interval的symbol→instrument映射；不是新建第二目录。
2. 所有面板轴、operator state、因子值和持仓都使用规范instrument_id；ticker仅展示。
3. 映射冲突、缺市场或无法确认永久身份时阻断相关资产，而不是自动str合并。

**必须落地的回归／验收**

1. 同ticker不同交易所不合并。
2. 旧证券退市后代码重用不得继承旧EMA状态。
3. 证券更名但永久ID未变时历史时序连续且可回放。

**依赖**：[QE-29](#issue-qe-29)、[FP-01](#issue-fp-01)。  **需求映射**：R14、R15。

**关闭方式**：若实际代码已满足，提交对应公开路径测试与证据，标为已覆盖；若需修改，提交权威实现及调用者变更、旧制品影响和测试结果；环境不足保留具体BLOCKED项，不以文档或元数据声明关闭。

---

<a id="issue-dta-02"></a>
#### DTA-02 · 财报必须按双时态可见版本连接，不能按报告期连接

**优先级**：P0　**证据分类**：E（待实仓核查）　**Owner**：DA/FE

**现有定位范围**：`data_access`；`factor_engine`；`quant_evaluator/contracts`；`quant_platform/app/adapters`。

**需要排查的风险与为什么要改**

核查报告所属期、首次公告、修订公告、供应商接收和平台ingestion时间是否被混成一个timestamp；本项不假定当前DA未实现PIT。

**实施步骤**

1. 沿现有as-of读取契约增加或确认event/effective time与knowledge/available time；记录实际数据快照。
2. 决策时仅选截至当时已可见的最新版本，同一报告后续修订只影响之后的可见查询。
3. 同值修订仍保存来源版本；所有填充以最后可见版本及TTL执行。

**必须落地的回归／验收**

1. 同一财报先发布后更正，早期决策读取旧数值。
2. 延迟供应商数据不能被回写成当时已知。
3. 复跑旧snapshot得到旧结果，新snapshot有明确差异。

**依赖**：[QE-30](#issue-qe-30)、[QE-31](#issue-qe-31)、[V-12](#issue-v-12)。  **需求映射**：R01、R14、R15。

**关闭方式**：若实际代码已满足，提交对应公开路径测试与证据，标为已覆盖；若需修改，提交权威实现及调用者变更、旧制品影响和测试结果；环境不足保留具体BLOCKED项，不以文档或元数据声明关闭。

---

<a id="issue-dta-03"></a>
#### DTA-03 · 拆分、分红、复权和停牌复牌不应制造伪收益

**优先级**：P0　**证据分类**：E（待实仓核查）　**Owner**：DA/FE/QE

**现有定位范围**：`data_access`；`factor_engine`；`quant_evaluator/contracts`；`quant_platform/app/adapters`。

**需要排查的风险与为什么要改**

核查因子价格、成交价格、持有收益、复权因子和现金分红是否使用兼容的经济口径。价格单位一致并不证明股数与现金路径一致。

**实施步骤**

1. 分别标注raw price、adjusted signal price和total-return accounting；复用DA企业行为数据。
2. 执行域按拆分调整股数，分红记现金，QE仅消费已对账收益；禁止把后复权最新全历史数值当时可见输入而不说明。
3. 变更企业行为快照后计算影响范围，不覆盖旧研究历史。

**必须落地的回归／验收**

1. 纯2拆1无经济收益却价格减半，组合毛收益应不变。
2. 除息下降加分红现金的总回报可手算。
3. 因子与label复权口径不匹配在入口拒绝。

**依赖**：[QE-23](#issue-qe-23)、[QE-25](#issue-qe-25)。  **需求映射**：R07、R14、R15。

**关闭方式**：若实际代码已满足，提交对应公开路径测试与证据，标为已覆盖；若需修改，提交权威实现及调用者变更、旧制品影响和测试结果；环境不足保留具体BLOCKED项，不以文档或元数据声明关闭。

---

<a id="issue-dta-04"></a>
#### DTA-04 · 股票池与退市样本覆盖必须是历史可投资集合

**优先级**：P0　**证据分类**：E（待实仓核查）　**Owner**：DA/FA/QE

**现有定位范围**：`data_access`；`factor_engine`；`quant_evaluator/contracts`；`quant_platform/app/adapters`。

**需要排查的风险与为什么要改**

核查历史研究是否用今天成分股或当前仍存续证券反推；覆盖率分母不能由未来成功样本决定。

**实施步骤**

1. 读取当日历史universe与筛选规则版本，保留上市、停牌、退市和资格变化的时间戳。
2. 分别计算可投资集合、信号可得集合、标签可评集合，不为满足95%目标临时删异常股票。
3. 退市后价值/结算不足时有明确未知或损失规则，不能自动从PnL中删除。

**必须落地的回归／验收**

1. 加入已退市股票不改变过去股票池的规则但改变正确覆盖统计。
2. 未来被剔除成员在此前仍应出现。
3. 同一因子不同预声明universe取得独立评价身份。

**依赖**：[QE-21](#issue-qe-21)、[QE-26](#issue-qe-26)。  **需求映射**：R03、R07、R15。

**关闭方式**：若实际代码已满足，提交对应公开路径测试与证据，标为已覆盖；若需修改，提交权威实现及调用者变更、旧制品影响和测试结果；环境不足保留具体BLOCKED项，不以文档或元数据声明关闭。

---

<a id="issue-dta-05"></a>
#### DTA-05 · 可交易性要区分买入、卖出、做空和回补

**优先级**：P0　**证据分类**：E（待实仓核查）　**Owner**：DA/执行域/QE

**现有定位范围**：`data_access`；`factor_engine`；`quant_evaluator/contracts`；`quant_platform/app/adapters`。

**需要排查的风险与为什么要改**

一个tradable布尔值未必能表达涨跌停、停牌、无借券和限额。需核查策略评价与执行事件是否按侧别处理。

**实施步骤**

1. 在现有ExecutionSpec中定义可买/可卖/可借/可回补及价格有效性，不由QE暗造执行规则。
2. 未成交订单保持实际持仓和风险，不能当作已经平仓；事后实际成交状态与决策时资格分开。
3. 规则和费用按市场及生效版本引用，不硬编码当前法律或交易制度。

**必须落地的回归／验收**

1. 无法卖出时旧持仓继续估值且风险不归零。
2. 无借券时研究short仍可诊断但不可获可执行标签。
3. 部分成交后现金、股数、目标偏差可对账。

**依赖**：[QE-25](#issue-qe-25)、[V-13](#issue-v-13)。  **需求映射**：R05、R07、R15。

**关闭方式**：若实际代码已满足，提交对应公开路径测试与证据，标为已覆盖；若需修改，提交权威实现及调用者变更、旧制品影响和测试结果；环境不足保留具体BLOCKED项，不以文档或元数据声明关闭。

---

<a id="issue-dta-06"></a>
#### DTA-06 · 派生字段的来源分类与单位代数必须一致

**优先级**：P1　**证据分类**：E（待实仓核查）　**Owner**：DA/FE/FA

**现有定位范围**：`data_access`；`factor_engine`；`quant_evaluator/contracts`；`quant_platform/app/adapters`。

**需要排查的风险与为什么要改**

换手率、估值比和收益率既可能作为供应商字段也可能在DSL中构造；仅扫字段名容易漏域或错标单位。

**实施步骤**

1. 复用FE字段及算子血缘，展开派生字段叶子来源并保留经济domain annotation。
2. 区分volume、amount、shares、market cap及货币/股数单位；自动分类不把control字段当alpha。
3. 单位检查识别不合理相加和分母近零，但不把无量纲因子默认视作已标准化。

**必须落地的回归／验收**

1. 直接turnover字段与volume/free_float在相同定义下来源标签一致。
2. 不同币种amount相加必须显式转换。
3. 未知派生字段保持unknown及待确认原因。

**依赖**：[FA-06](#issue-fa-06)、[FP-08](#issue-fp-08)。  **需求映射**：R01、R02、R13、R15。

**关闭方式**：若实际代码已满足，提交对应公开路径测试与证据，标为已覆盖；若需修改，提交权威实现及调用者变更、旧制品影响和测试结果；环境不足保留具体BLOCKED项，不以文档或元数据声明关闭。

---

<a id="issue-dta-07"></a>
#### DTA-07 · 跨市场时区、交易日和汇率对齐不允许日期字符串碰撞

**优先级**：P1　**证据分类**：E（待实仓核查）　**Owner**：DA/FE/QE

**现有定位范围**：`data_access`；`factor_engine`；`quant_evaluator/contracts`；`quant_platform/app/adapters`。

**需要排查的风险与为什么要改**

全球扩展时同一个YYYY-MM-DD可能代表不同可用信息集；本项是接口准备和核查，不要求本轮新增所有国家交易服务。

**实施步骤**

1. 时刻使用可比较带时区instant，交易session另有市场日历ID；日期标签不是实际可用时间。
2. 多市场截面明确共同决策cutoff，较晚收盘信息不得提前加入早收盘决策。
3. 本币和基准币收益分开，FX采样窗口、假日填充及价格来源写进spec。

**必须落地的回归／验收**

1. 跨夏令时和半日市的边界用批准日历fixture测试。
2. 一市场休市另一市场开市时不伪造新价格。
3. 汇率缺失不能默认0回报或用未来汇率。

**依赖**：[QE-18](#issue-qe-18)、[QE-30](#issue-qe-30)。  **需求映射**：R07、R14、R15。

**关闭方式**：若实际代码已满足，提交对应公开路径测试与证据，标为已覆盖；若需修改，提交权威实现及调用者变更、旧制品影响和测试结果；环境不足保留具体BLOCKED项，不以文档或元数据声明关闭。

---

<a id="issue-dta-08"></a>
#### DTA-08 · 缺失原因必须从原始字段一直传到模型与清理任务

**优先级**：P1　**证据分类**：E（待实仓核查）　**Owner**：DA/FE/FP/QE

**现有定位范围**：`data_access`；`factor_engine`；`quant_evaluator/contracts`；`quant_platform/app/adapters`。

**需要排查的风险与为什么要改**

NaN可能表示未上市、无覆盖、报价故障、尚未公告或公式无定义，修复动作不能一律ffill/zero。

**实施步骤**

1. 复用或扩展missing_reason枚举，区分原始缺失、算术非法、mask排除、陈旧过期和预算未运行。
2. 变换同时输出value/mask/reason，filled值保留原始缺失flag与age。
3. 下游coverage同时报告observed和usable/filled，禁止通过填充伪造原始数据质量。

**必须落地的回归／验收**

1. 同为NaN的未上市与短时行情故障走不同政策。
2. 填充后原始missing indicator仍为真。
3. 尚未成熟标签的values不得进入失败值GC。

**依赖**：[QE-26](#issue-qe-26)、[FP-08](#issue-fp-08)。  **需求映射**：R03、R06、R13、R16。

**关闭方式**：若实际代码已满足，提交对应公开路径测试与证据，标为已覆盖；若需修改，提交权威实现及调用者变更、旧制品影响和测试结果；环境不足保留具体BLOCKED项，不以文档或元数据声明关闭。

---

<a id="issue-dta-09"></a>
#### DTA-09 · 所有as-of和多表join必须验证基数及方向

**优先级**：P0　**证据分类**：E（待实仓核查）　**Owner**：DA/FE/FP

**现有定位范围**：`data_access`；`factor_engine`；`quant_evaluator/contracts`；`quant_platform/app/adapters`。

**需要排查的风险与为什么要改**

核查重复公告、非唯一行业行或跨频率连接是否产生一对多扩张；不能等到pivot first才消除。

**实施步骤**

1. 每次join声明left key/right key、允许基数、as-of方向和最大lag；记录前后行数和重复率。
2. 对同knowledge timestamp的多版本定义稳定tie-break，只按来源版本不按未来结果。
3. join失败返回有定位信息的契约错误，不自动inner join丢证券。

**必须落地的回归／验收**

1. 注入双行业记录必须阻断或按显式版本解析。
2. right表顺序置换不改结果。
3. future as-of行不能被backward查询选中。

**依赖**：[FP-02](#issue-fp-02)、[QE-29](#issue-qe-29)。  **需求映射**：R14、R15。

**关闭方式**：若实际代码已满足，提交对应公开路径测试与证据，标为已覆盖；若需修改，提交权威实现及调用者变更、旧制品影响和测试结果；环境不足保留具体BLOCKED项，不以文档或元数据声明关闭。

---

<a id="issue-dta-10"></a>
#### DTA-10 · 晚到数据watermark与局部重算范围需要显式依赖

**优先级**：P1　**证据分类**：E（待实仓核查）　**Owner**：DA/FE/Platform

**现有定位范围**：`data_access`；`factor_engine`；`quant_evaluator/contracts`；`quant_platform/app/adapters`。

**需要排查的风险与为什么要改**

晚到公告或修订历史价格可能影响滚动算子之后很多日期；只重算修订当日或无条件重算全库都不合理。

**实施步骤**

1. 使用现有FE lookback/state依赖推导受影响区间；有限窗口与无限记忆EMA分开政策。
2. 明确data watermark与event time，晚到更新生成新snapshot和重算job。
3. 无限记忆状态采用精确从最近有效checkpoint重放或经批准截断误差政策，不能无声明近似。

**必须落地的回归／验收**

1. 修订滚动窗口起点影响后续预期日期且不越界。
2. 修订发生在checkpoint前时不得从污染state继续。
3. 同快照整段重算与局部重算一致。

**依赖**：[V-07](#issue-v-07)、[V-19](#issue-v-19)。  **需求映射**：R08、R14、R15。

**关闭方式**：若实际代码已满足，提交对应公开路径测试与证据，标为已覆盖；若需修改，提交权威实现及调用者变更、旧制品影响和测试结果；环境不足保留具体BLOCKED项，不以文档或元数据声明关闭。

---

<a id="issue-dta-11"></a>
#### DTA-11 · 多期限标签成熟度与共同比较样本必须同时保留

**优先级**：P0　**证据分类**：E（待实仓核查）　**Owner**：DA/QE/FO

**现有定位范围**：`data_access`；`factor_engine`；`quant_evaluator/contracts`；`quant_platform/app/adapters`。

**需要排查的风险与为什么要改**

H1与H20在样本尾部可评日期不同；简单各算均值再比较会把期限变化与样本变化混淆。

**实施步骤**

1. 每个horizon独立存label interval、mature mask、nobs；再生成预声明intersection比较视图。
2. 完整可评样本指标与共同样本指标不同ID或context，不能互相覆盖。
3. 不为补尾部将尚未发生收益设0，也不根据结果挑共同起止日。

**必须落地的回归／验收**

1. 增加最后19天数据时H1成熟样本增加但H20未成熟仍不计。
2. 共同样本各H的日期轴完全相同。
3. forward label不能误送cohort holding收益输入。

**依赖**：[QE-09](#issue-qe-09)、[QE-32](#issue-qe-32)、[V-12](#issue-v-12)。  **需求映射**：R03、R10、R15。

**关闭方式**：若实际代码已满足，提交对应公开路径测试与证据，标为已覆盖；若需修改，提交权威实现及调用者变更、旧制品影响和测试结果；环境不足保留具体BLOCKED项，不以文档或元数据声明关闭。

---

<a id="issue-dta-12"></a>
#### DTA-12 · 面板布局、dtype与只读所有权必须在跨库边界锁定

**优先级**：P1　**证据分类**：E（待实仓核查）　**Owner**：FE/FP/QE/DA

**现有定位范围**：`data_access`；`factor_engine`；`quant_evaluator/contracts`；`quant_platform/app/adapters`。

**需要排查的风险与为什么要改**

TNF、TFN和扁平长表转换若只靠shape或隐式broadcast，可能在N=F时悄悄串轴；GPU零拷贝也有buffer生命周期风险。

**实施步骤**

1. 布局由named axes和contract标识，转换在边界有一个实现；验证所有有效mask轴。
2. 浮点精度、NaN/Inf政策与只读buffer所有权进入execution metadata；避免未同步设备buffer提前复用。
3. 独立metric不能修改共享factor或label值，必要使用COW或临时工作区。

**必须落地的回归／验收**

1. N恰等于F的对称shape也能拒绝错误layout。
2. view被外部写入不污染已冻结artifact。
3. CPU/GPU转换往返值/轴/mask均保持。

**依赖**：[QE-04](#issue-qe-04)、[QE-29](#issue-qe-29)、[QE-31](#issue-qe-31)。  **需求映射**：R08、R14、R15。

**关闭方式**：若实际代码已满足，提交对应公开路径测试与证据，标为已覆盖；若需修改，提交权威实现及调用者变更、旧制品影响和测试结果；环境不足保留具体BLOCKED项，不以文档或元数据声明关闭。

---

### 12.2 STA：统计、选择偏差与评级专项

<a id="issue-sta-01"></a>
#### STA-01 · RankIC按每个因子的共同有效集合重新确定秩

**优先级**：P0　**证据分类**：E（待实仓核查）　**Owner**：QE

**现有定位范围**：`quant_evaluator/metrics`；`quant_evaluator/registry`；`factor_assets/profiling`；`factor_optimizer/factor_optimizer/contracts`。

**需要排查的风险与为什么要改**

核查加速共享label ranks时是否忽略因子不同missing mask；共同有效样本改变时预先整截面排名不总与pairwise排名一致。

**实施步骤**

1. 先求每个(T,F)的共同有效资产集合，再对x和y在该集合上按统一ties规则排名。
2. 仅对相同mask hash的组共享label ranks，缺失模式不同不能盲复用。
3. 输出n_pairs和常数/不足状态；保留Pearson与Spearman不同路径。

**必须落地的回归／验收**

1. 两个因子相同有效值但缺失证券不同，对照独立rank金标。
2. masked极值修改结果不变。
3. ties、单一有效资产、重复秩两后端一致。

**依赖**：[QE-04](#issue-qe-04)、[QE-28](#issue-qe-28)。  **需求映射**：R03、R08、R15。

**关闭方式**：若实际代码已满足，提交对应公开路径测试与证据，标为已覆盖；若需修改，提交权威实现及调用者变更、旧制品影响和测试结果；环境不足保留具体BLOCKED项，不以文档或元数据声明关闭。

---

<a id="issue-sta-02"></a>
#### STA-02 · HAC和有效时间样本数不得由股票格数替代

**优先级**：P1　**证据分类**：E（待实仓核查）　**Owner**：QE/FA

**现有定位范围**：`quant_evaluator/metrics`；`quant_evaluator/registry`；`factor_assets/profiling`；`factor_optimizer/factor_optimizer/contracts`。

**需要排查的风险与为什么要改**

检验精度来自IC时间序列而非T×N独立样本；重叠horizon或连续持仓引入的时间相关需要明确处理。

**实施步骤**

1. 每个统计检验记录样本单位、n_time、依赖假设、HAC kernel/bandwidth与选择规则。
2. bandwidth从预声明方法或horizon/经验相关结构估计，不在最终test挑最显著配置。
3. 统计不足返回INSUFFICIENT，不能仅靠raw ICIR或√252作显著性证据。

**必须落地的回归／验收**

1. 同一个IC序列复制股票列不应让t值倍增。
2. 有自相关序列与独立序列使用不同稳健误差但同均值。
3. 带宽边界和短序列有明确状态。

**依赖**：[QE-27](#issue-qe-27)、[V-11](#issue-v-11)。  **需求映射**：R03、R10。

**关闭方式**：若实际代码已满足，提交对应公开路径测试与证据，标为已覆盖；若需修改，提交权威实现及调用者变更、旧制品影响和测试结果；环境不足保留具体BLOCKED项，不以文档或元数据声明关闭。

---

<a id="issue-sta-03"></a>
#### STA-03 · 块bootstrap必须保留时间依赖和候选间的配对关系

**优先级**：P1　**证据分类**：E（待实仓核查）　**Owner**：QE/FO

**现有定位范围**：`quant_evaluator/metrics`；`quant_evaluator/registry`；`factor_assets/profiling`；`factor_optimizer/factor_optimizer/contracts`。

**需要排查的风险与为什么要改**

单独对每个候选随意抽样，会破坏非劣性差值和共同市场冲击；对证券和时间同时iid抽样也可能回答不同问题。

**实施步骤**

1. 为每次比较生成共享time-block draws，所有候选使用相同draw索引。
2. 保持同一日期完整截面或按预声明双向聚类方案；缺失样本和block长度有状态。
3. 保存seed、块方法、重复次数和区间构造类型；早筛使用少量估计，入围才增加精度。

**必须落地的回归／验收**

1. 两个完全相同候选的成对差值bootstrap区间包含0且无伪差。
2. 改变请求顺序不改变共享draw。
3. 重叠窗口的shape bootstrap不得假称独立窗口样本。

**依赖**：[QE-34](#issue-qe-34)、[V-06](#issue-v-06)。  **需求映射**：R04、R08、R10。

**关闭方式**：若实际代码已满足，提交对应公开路径测试与证据，标为已覆盖；若需修改，提交权威实现及调用者变更、旧制品影响和测试结果；环境不足保留具体BLOCKED项，不以文档或元数据声明关闭。

---

<a id="issue-sta-04"></a>
#### STA-04 · 搜索尝试账本与FDR假设族不能通过改名重置

**优先级**：P0　**证据分类**：E（待实仓核查）　**Owner**：FO/QE/FA

**现有定位范围**：`quant_evaluator/metrics`；`quant_evaluator/registry`；`factor_assets/profiling`；`factor_optimizer/factor_optimizer/contracts`。

**需要排查的风险与为什么要改**

统计假设族应包含原始因子、变换、方向和参数比较；技术重试与新的科学假设又不能简单一概累加成同一种数。

**实施步骤**

1. 分别登记proposal_count、executed_trial_count、unique_effective_spec_count与test_access事件。
2. 定义campaign级统计族和相关结构，记录未成功计算的尝试为何无p值。
3. 使用FDR方法前检查其适用假设；可保守校正、层级检验或明确不具推断资格，不机械给q≤.05印章。

**必须落地的回归／验收**

1. 同值alias不产生新独立发现，却不删原提案记录。
2. 新session继承同campaign历史。
3. 只向校正函数提供赢家p值的请求被拒或标不合格。

**依赖**：[V-11](#issue-v-11)、[FO-05](#issue-fo-05)。  **需求映射**：R09、R10、R16。

**关闭方式**：若实际代码已满足，提交对应公开路径测试与证据，标为已覆盖；若需修改，提交权威实现及调用者变更、旧制品影响和测试结果；环境不足保留具体BLOCKED项，不以文档或元数据声明关闭。

---

<a id="issue-sta-05"></a>
#### STA-05 · DSR的样本、Sharpe尺度和有效试验数要可审计

**优先级**：P1　**证据分类**：E（待实仓核查）　**Owner**：QE/FO

**现有定位范围**：`quant_evaluator/metrics`；`quant_evaluator/registry`；`factor_assets/profiling`；`factor_optimizer/factor_optimizer/contracts`。

**需要排查的风险与为什么要改**

DSR不能从一个Sharpe裸数直接可靠生成；需要确定样本长度、收益频率、偏度峰度和选择过程。

**实施步骤**

1. 先查QE现有实现，列出全部必需输入、统计定义和缺失条件。
2. 同一期间收益尺度一致，试验数来源trial ledger，独立有效尝试数的估计方法明确。
3. 无法支持假设或样本不足时标研究辅助，不伪造默认值使其可算。

**必须落地的回归／验收**

1. 改变原始收益样本长度但Sharpe相同，证据应反映不同不确定性。
2. 年化与未年化输入不混用。
3. 缺试验账本不能获生产统计认证。

**依赖**：[QE-17](#issue-qe-17)、[QE-27](#issue-qe-27)、[V-11](#issue-v-11)。  **需求映射**：R03、R10。

**关闭方式**：若实际代码已满足，提交对应公开路径测试与证据，标为已覆盖；若需修改，提交权威实现及调用者变更、旧制品影响和测试结果；环境不足保留具体BLOCKED项，不以文档或元数据声明关闭。

---

<a id="issue-sta-06"></a>
#### STA-06 · PBO或多切分验证只能用于声明的研究目的

**优先级**：P1　**证据分类**：E（待实仓核查）　**Owner**：QE/FO

**现有定位范围**：`quant_evaluator/metrics`；`quant_evaluator/registry`；`factor_assets/profiling`；`factor_optimizer/factor_optimizer/contracts`。

**需要排查的风险与为什么要改**

某些组合式切分允许时间逆序，不能把这种研究稳定性结果当作严格前推OOS实盘证据。

**实施步骤**

1. 实现前定位现有split方案并标明prediction validation或selection diagnostics用途。
2. 候选收益矩阵必须同日期同成本规格，不能用不完整赢家结果代替全集。
3. PBO结果保留辅助标签；正式发布仍按冻结时间前推和sealed策略。

**必须落地的回归／验收**

1. 把diagnostic split引用送入production OOS gate会被拒。
2. 只剩winner收益的PBO请求返回缺必要证据。
3. 少候选和少切分时不输出夸大精度。

**依赖**：[V-11](#issue-v-11)、[V-12](#issue-v-12)。  **需求映射**：R03、R10。

**关闭方式**：若实际代码已满足，提交对应公开路径测试与证据，标为已覆盖；若需修改，提交权威实现及调用者变更、旧制品影响和测试结果；环境不足保留具体BLOCKED项，不以文档或元数据声明关闭。

---

<a id="issue-sta-07"></a>
#### STA-07 · 训练验证衰减要区分过拟合、市场状态与近零分母

**优先级**：P1　**证据分类**：E（待实仓核查）　**Owner**：QE/FA/FO

**现有定位范围**：`quant_evaluator/metrics`；`quant_evaluator/registry`；`factor_assets/profiling`；`factor_optimizer/factor_optimizer/contracts`。

**需要排查的风险与为什么要改**

train/valid比值低可能来自过拟合，也可能来自股票池变化、成本上升、状态切换或训练值接近零。

**实施步骤**

1. 同时输出有向差值、配对不确定性、有效窗口数和retention适用标志。
2. 方向反转、分母近零和符号变化不直接映射普通保留率。
3. 先排除数据/时钟/配置变化，再诊断机制退化；repair建议不能仅由比值自动触发复杂变换。

**必须落地的回归／验收**

1. train=.0001、valid=.0002不凭200%评S+。
2. train正valid负显式标sign reversal。
3. 相同模型遇到不同成本时区分gross与net退化。

**依赖**：[QE-32](#issue-qe-32)、[V-21](#issue-v-21)。  **需求映射**：R03、R09、R10。

**关闭方式**：若实际代码已满足，提交对应公开路径测试与证据，标为已覆盖；若需修改，提交权威实现及调用者变更、旧制品影响和测试结果；环境不足保留具体BLOCKED项，不以文档或元数据声明关闭。

---

<a id="issue-sta-08"></a>
#### STA-08 · 评级锚点校准和显示分位数必须分开版本

**优先级**：P1　**证据分类**：E（待实仓核查）　**Owner**：FA/QE

**现有定位范围**：`quant_evaluator/metrics`；`quant_evaluator/registry`；`factor_assets/profiling`；`factor_optimizer/factor_optimizer/contracts`。

**需要排查的风险与为什么要改**

若全库质量变差但按当期百分位分档，总有一批因子获S；绝对门槛和同类相对排序不可混同。

**实施步骤**

1. 保留绝对经济/工程硬门和相对cohort percentile两个字段，cohort只作同类比较。
2. 用历史development folds校准S+至D锚点，记录市场、horizon、成本、股票池和政策hash。
3. 上线后新增因子不改变历史同policy等级；正式重校准生成新policy并重新评级。

**必须落地的回归／验收**

1. 整个因子库变差不会仅凭百分位让更多因子过绝对门。
2. 新增一批差因子不提高旧因子绝对grade。
3. sealed数据无权改变锚点。

**依赖**：[FA-03](#issue-fa-03)、[V-03](#issue-v-03)。  **需求映射**：R03、R10。

**关闭方式**：若实际代码已满足，提交对应公开路径测试与证据，标为已覆盖；若需修改，提交权威实现及调用者变更、旧制品影响和测试结果；环境不足保留具体BLOCKED项，不以文档或元数据声明关闭。

---

<a id="issue-sta-09"></a>
#### STA-09 · 市场状态切片要标明事前可知还是事后压力复盘

**优先级**：P1　**证据分类**：E（待实仓核查）　**Owner**：QE/FA/FO

**现有定位范围**：`quant_evaluator/metrics`；`quant_evaluator/registry`；`factor_assets/profiling`；`factor_optimizer/factor_optimizer/contracts`。

**需要排查的风险与为什么要改**

利用整段未来收益标出的bull/bear状态适合事后解释，却不能直接作为当日regime切换器输入。

**实施步骤**

1. 将ex-post diagnostic regime和online causal regime分不同类型和可用时间。
2. 状态切换模型拟合/阈值只用当时训练数据，保存state/refit规则。
3. 所有conditional IC的n_window和覆盖必须披露，不从最差窗删除后再评分。

**必须落地的回归／验收**

1. 事后状态标签不能进入生产因子DAG。
2. 未来改价不改变此前online regime。
3. 极小状态样本返回不足而非高确定性好分。

**依赖**：[V-12](#issue-v-12)、[V-21](#issue-v-21)。  **需求映射**：R03、R10、R15。

**关闭方式**：若实际代码已满足，提交对应公开路径测试与证据，标为已覆盖；若需修改，提交权威实现及调用者变更、旧制品影响和测试结果；环境不足保留具体BLOCKED项，不以文档或元数据声明关闭。

---

<a id="issue-sta-10"></a>
#### STA-10 · 期限峰值、符号变化与滤波半衰期不能混作一个数字

**优先级**：P1　**证据分类**：E（待实仓核查）　**Owner**：QE/FO

**现有定位范围**：`quant_evaluator/metrics`；`quant_evaluator/registry`；`factor_assets/profiling`；`factor_optimizer/factor_optimizer/contracts`。

**需要排查的风险与为什么要改**

预测曲线可能多峰、反号或非指数衰减；简单拟合一个半衰期可能为错误滤波强度背书。

**实施步骤**

1. 分别保存signal persistence、IC serial ACF、horizon预测曲线及可拟合性。
2. 只有满足预声明形状条件才拟合半衰期，反号/多峰保留诊断和原曲线。
3. 寻找最佳horizon也计入搜索族；不能按test最优期限重新定义目标。

**必须落地的回归／验收**

1. 短期反转长期动量曲线不输出一个无说明正半衰期。
2. 噪声曲线可返回不适用。
3. 窗口修复使用natural horizon证据而非任意合成指标。

**依赖**：[QE-32](#issue-qe-32)、[V-09](#issue-v-09)。  **需求映射**：R03、R06、R09、R10。

**关闭方式**：若实际代码已满足，提交对应公开路径测试与证据，标为已覆盖；若需修改，提交权威实现及调用者变更、旧制品影响和测试结果；环境不足保留具体BLOCKED项，不以文档或元数据声明关闭。

---

<a id="issue-sta-11"></a>
#### STA-11 · 尾部指标需要明确损失符号、加权分位与样本不足

**优先级**：P1　**证据分类**：E（待实仓核查）　**Owner**：QE/FA

**现有定位范围**：`quant_evaluator/metrics`；`quant_evaluator/registry`；`factor_assets/profiling`；`factor_optimizer/factor_optimizer/contracts`。

**需要排查的风险与为什么要改**

VaR/ES符号、含等号分位边界和尾部样本太少会让风险等级在不同实现间漂移。

**实施步骤**

1. 明确以损失正数或收益负数表达，统一confidence level和quantile插值政策。
2. 极小尾部样本标低证据强度；不能仅20个日收益就把99% ES当可靠硬门。
3. 权重/交易时间异质性存在时采用已声明权重，灾难值不能因非finite被排除。

**必须落地的回归／验收**

1. 全正收益下ES语义按规范且不造负损失惊喜。
2. 分位边界ties与单个极端损失可手算。
3. CPU/GPU/报告使用同一tail artifact。

**依赖**：[QE-12](#issue-qe-12)、[QE-16](#issue-qe-16)、[QE-17](#issue-qe-17)。  **需求映射**：R03、R07。

**关闭方式**：若实际代码已满足，提交对应公开路径测试与证据，标为已覆盖；若需修改，提交权威实现及调用者变更、旧制品影响和测试结果；环境不足保留具体BLOCKED项，不以文档或元数据声明关闭。

---

<a id="issue-sta-12"></a>
#### STA-12 · 多维评级不重复计分，诊断结论必须给出证据原因

**优先级**：P1　**证据分类**：E（待实仓核查）　**Owner**：FA/QE

**现有定位范围**：`quant_evaluator/metrics`；`quant_evaluator/registry`；`factor_assets/profiling`；`factor_optimizer/factor_optimizer/contracts`。

**需要排查的风险与为什么要改**

RankIC、mean RankIC、rolling RankIC均值若同时作为独立优点，会把同一信号重复加权；诊断高分也不等于确定获利。

**实施步骤**

1. 每维建立required metric groups和冗余关系，同一证据的别名不得重复入聚合。
2. 展示value/effect grade/evidence sufficiency/CI/bottleneck分开，拒绝只输出字母。
3. 说明REJECT/REVIEW/REPAIR的触发原始metric、阈值、上下文与可修复性，Agent只转述。

**必须落地的回归／验收**

1. 同metric多alias请求不提高维度分。
2. 必需证据不足即使点估计S仍不可批准。
3. 报告里每条维修建议可追到实际证据。

**依赖**：[FA-04](#issue-fa-04)、[V-03](#issue-v-03)、[V-22](#issue-v-22)。  **需求映射**：R03、R17。

**关闭方式**：若实际代码已满足，提交对应公开路径测试与证据，标为已覆盖；若需修改，提交权威实现及调用者变更、旧制品影响和测试结果；环境不足保留具体BLOCKED项，不以文档或元数据声明关闭。

---

### 12.3 RCP：配方、形状、因果变换与表示专项

<a id="issue-rcp-01"></a>
#### RCP-01 · 缺失指标、新鲜度等辅助通道必须分支输出而非替换主信号

**优先级**：P0　**证据分类**：E（待实仓核查）　**Owner**：FP/FE/modeling

**现有定位范围**：`factor_preprocess/factor_preprocess`；`factor_engine`；`factor_optimizer/factor_optimizer/policy`。

**需要排查的风险与为什么要改**

逐步fn(value)流水线可能把missing_indicator的布尔结果当成下一步主信号；需验证FeatureBundle channel routing。

**实施步骤**

1. 每个步骤声明input/output channel与REPLACE/FORK/ANNOTATE效果。
2. missing indicator从填充前原始mask读取，freshness另开channel，主feature值不被辅助输出覆盖。
3. DAG与manifest保留通道角色，模型列名和dtype稳定。

**必须落地的回归／验收**

1. 插入missing_indicator步骤前后主feature在非缺失位置不变。
2. 填充后原始缺失标志仍可见。
3. 辅助binary通道不自动zscore/neutralize除非profile明示。

**依赖**：[FP-08](#issue-fp-08)、[V-08](#issue-v-08)。  **需求映射**：R03、R13、R14。

**关闭方式**：若实际代码已满足，提交对应公开路径测试与证据，标为已覆盖；若需修改，提交权威实现及调用者变更、旧制品影响和测试结果；环境不足保留具体BLOCKED项，不以文档或元数据声明关闭。

---

<a id="issue-rcp-02"></a>
#### RCP-02 · z-score、rank的幂等和零方差处理必须带适用前提

**优先级**：P1　**证据分类**：E（待实仓核查）　**Owner**：FE/FP

**现有定位范围**：`factor_preprocess/factor_preprocess`；`factor_engine`；`factor_optimizer/factor_optimizer/policy`。

**需要排查的风险与为什么要改**

数学上同形运算在不同ddof、weights、sample mask或非线性中间步骤下不一定可删；标准化为0也不总等于无信息。

**实施步骤**

1. FE rewrite law注明输入节点、轴、mask、权重、ddof和constant policy。
2. 根输出属性用于“是否还需标准化”；inner zscore不是最终zscore。
3. singleton与constant截面返回0或缺失按consumer policy，状态不能伪装正常有方差。

**必须落地的回归／验收**

1. zscore(zscore(x))仅在前提满足时折叠。
2. 换mask/权重不能折叠。
3. zscore→abs→zscore与删后层结果应在金标中不同。

**依赖**：[FP-07](#issue-fp-07)、[FP-08](#issue-fp-08)。  **需求映射**：R13、R14。

**关闭方式**：若实际代码已满足，提交对应公开路径测试与证据，标为已覆盖；若需修改，提交权威实现及调用者变更、旧制品影响和测试结果；环境不足保留具体BLOCKED项，不以文档或元数据声明关闭。

---

<a id="issue-rcp-03"></a>
#### RCP-03 · 去极值政策需区分截面winsor、时间拟合阈值和合法经济极值

**优先级**：P1　**证据分类**：E（待实仓核查）　**Owner**：FE/FP/FO

**现有定位范围**：`factor_preprocess/factor_preprocess`；`factor_engine`；`factor_optimizer/factor_optimizer/policy`。

**需要排查的风险与为什么要改**

极值处理可能抹掉真实事件alpha；按全样本算阈值也可能使用未来分布。

**实施步骤**

1. 为CS quantile、MAD与train-fit bounds声明轴/fit边界、阈值单位和最小样本。
2. 首轮只启用少量阈值作为诊断候选，保留raw与clipped ratio。
3. 高clip比例触发数据/分布复核，不自动多试阈值直到IC好看。

**必须落地的回归／验收**

1. 未来加入极端值不改变此前因果CS/训练冻结变换。
2. 含真实跳跃的事件因子raw基线保留。
3. 所有阈值到FE有效参数并改变recipe身份。

**依赖**：[FP-03](#issue-fp-03)、[V-09](#issue-v-09)。  **需求映射**：R04、R09、R13、R15。

**关闭方式**：若实际代码已满足，提交对应公开路径测试与证据，标为已覆盖；若需修改，提交权威实现及调用者变更、旧制品影响和测试结果；环境不足保留具体BLOCKED项，不以文档或元数据声明关闭。

---

<a id="issue-rcp-04"></a>
#### RCP-04 · 中性化求解器必须声明设计矩阵、权重与精确性

**优先级**：P1　**证据分类**：E（待实仓核查）　**Owner**：FE/FP/QE

**现有定位范围**：`factor_preprocess/factor_preprocess`；`factor_engine`；`factor_optimizer/factor_optimizer/policy`。

**需要排查的风险与为什么要改**

OLS、WLS、ridge、Huber和分行业demean不是同一数学操作；同叫neutralized会误导后置检查。

**实施步骤**

1. ExposureSpec明确截距、类别编码、行业基准、权重与最小自由度。
2. OLS/WLS用稳定QR/SVD等已支持求解器，秩亏按FAIL/REGULARIZE政策；ridge记录lambda且不宣称精确OLS正交。
3. QE核验实际残差暴露，求解失败不返回raw却仍标neutralized。

**必须落地的回归／验收**

1. XᵀWr在规定容差内，仅对承诺精确正交路径要求。
2. 共线/稀疏行业/单行业/极大市值数值稳定。
3. 缺暴露fallback有独立状态与值身份。

**依赖**：[FP-06](#issue-fp-06)、[V-06](#issue-v-06)。  **需求映射**：R05、R08、R15。

**关闭方式**：若实际代码已满足，提交对应公开路径测试与证据，标为已覆盖；若需修改，提交权威实现及调用者变更、旧制品影响和测试结果；环境不足保留具体BLOCKED项，不以文档或元数据声明关闭。

---

<a id="issue-rcp-05"></a>
#### RCP-05 · 最终中性约束与rank、clip、平滑的顺序要以输出性质验收

**优先级**：P1　**证据分类**：E（待实仓核查）　**Owner**：FP/FE/QE

**现有定位范围**：`factor_preprocess/factor_preprocess`；`factor_engine`；`factor_optimizer/factor_optimizer/policy`。

**需要排查的风险与为什么要改**

“neutralize放最后”也可能与神经输入缩放需求冲突；必须明确最终目标约束，而不是把所有链固定同一顺序。

**实施步骤**

1. 提供strict-neutral与rank-friendly两个声明不同保证的profile。
2. 中性残差可在相同mask和权重、含截距前提下做截面仿射缩放；非线性rank/clip或跨期平滑需重评暴露。
3. 需要再次投影时编译显式第二步并计成本，不允许标签说已处理就跳过。

**必须落地的回归／验收**

1. neutralize→rank后若暴露超限，strict profile拒绝或追加显式修复。
2. 仅正比例缩放在前提满足时保持中性。
3. 两profile都输出真实postcondition而非同一布尔旗。

**依赖**：[FP-08](#issue-fp-08)、[FP-09](#issue-fp-09)、[V-06](#issue-v-06)。  **需求映射**：R05、R13。

**关闭方式**：若实际代码已满足，提交对应公开路径测试与证据，标为已覆盖；若需修改，提交权威实现及调用者变更、旧制品影响和测试结果；环境不足保留具体BLOCKED项，不以文档或元数据声明关闭。

---

<a id="issue-rcp-06"></a>
#### RCP-06 · 因果平滑状态要区别重复事件值、休市和新观测

**优先级**：P1　**证据分类**：E（待实仓核查）　**Owner**：FE/FP

**现有定位范围**：`factor_preprocess/factor_preprocess`；`factor_engine`；`factor_optimizer/factor_optimizer/policy`。

**需要排查的风险与为什么要改**

forward-filled财报值每天重复输入EMA可能人为改变“事件衰减”含义；持平价格与没有新观测不应混为一类。

**实施步骤**

1. operator声明clock mode：每交易时点、每实际观测或每事件更新。
2. EMA checkpoint包含last_value、weight mass、last observation time、warmup和missing policy。
3. 对gap采用冻结/时间衰减/失效之一，规则写state hash；不要静默按0更新。

**必须落地的回归／验收**

1. 同一财报重复落点与真正新公告相同值的行为符合声明。
2. full/chunk/daily在休市和长gap一致。
3. 不同instrument状态独立且recipe变更不能复用旧state。

**依赖**：[V-07](#issue-v-07)、[DTA-08](#issue-dta-08)。  **需求映射**：R06、R14、R15。

**关闭方式**：若实际代码已满足，提交对应公开路径测试与证据，标为已覆盖；若需修改，提交权威实现及调用者变更、旧制品影响和测试结果；环境不足保留具体BLOCKED项，不以文档或元数据声明关闭。

---

<a id="issue-rcp-07"></a>
#### RCP-07 · 偏斜U型的中心拟合与左右分支需要低自由度监督契约

**优先级**：P1　**证据分类**：E（待实仓核查）　**Owner**：FO/FP/FE/QE

**现有定位范围**：`factor_preprocess/factor_preprocess`；`factor_engine`；`factor_optimizer/factor_optimizer/policy`。

**需要排查的风险与为什么要改**

从收益曲线拟合center/knots属于监督学习，会增加试验自由度；不能伪装stateless变换。

**实施步骤**

1. 保留c=.5固定baseline；拟合中心仅在train folds内，限定少量候选区间并记录拟合目标。
2. 左右hinge作为两个有明确方向的子定义；每支有覆盖、换手与风险证据。
3. 验证未优于raw或不稳定时可不修复，不强制把所有U变为单調。

**必须落地的回归／验收**

1. 改变验证/测试收益不会改变训练center。
2. 左右边一边无效时允许只留另一支但计选择成本。
3. parent/left/right/recombined各自重评且血缘完整。

**依赖**：[V-04](#issue-v-04)、[STA-03](#issue-sta-03)。  **需求映射**：R04、R09、R10、R11。

**关闭方式**：若实际代码已满足，提交对应公开路径测试与证据，标为已覆盖；若需修改，提交权威实现及调用者变更、旧制品影响和测试结果；环境不足保留具体BLOCKED项，不以文档或元数据声明关闭。

---

<a id="issue-rcp-08"></a>
#### RCP-08 · 尾部修复不能凭单张20层平均图选择任意cutoff

**优先级**：P1　**证据分类**：E（待实仓核查）　**Owner**：FO/QE/FE

**现有定位范围**：`factor_preprocess/factor_preprocess`；`factor_engine`；`factor_optimizer/factor_optimizer/policy`。

**需要排查的风险与为什么要改**

极端层低收益可能是采样噪声、风格、交易限制或数据错误；无限cutoff搜索会过拟合。

**实施步骤**

1. 先检查逐日期层人数、ties、风险暴露与可成交性，再确认异常是否跨窗重复。
2. 只允许预声明的饱和/有限hinge候选，使用共同validation样本。
3. 保存10和20层原始/修复后的对比，主目标是净效用与稳定性，不只是让曲线好看。

**必须落地的回归／验收**

1. 单窗尾部异常不自动生成大量阈值。
2. 数据错误尾部走数据修复而非公式掩盖。
3. 改变test尾部收益不能改变已冻结cutoff。

**依赖**：[QE-35](#issue-qe-35)、[V-04](#issue-v-04)。  **需求映射**：R04、R09、R10。

**关闭方式**：若实际代码已满足，提交对应公开路径测试与证据，标为已覆盖；若需修改，提交权威实现及调用者变更、旧制品影响和测试结果；环境不足保留具体BLOCKED项，不以文档或元数据声明关闭。

---

<a id="issue-rcp-09"></a>
#### RCP-09 · 原始、中性、平滑、模型表示版本不互相覆盖

**优先级**：P1　**证据分类**：E（待实仓核查）　**Owner**：FA/FP/FE

**现有定位范围**：`factor_preprocess/factor_preprocess`；`factor_engine`；`factor_optimizer/factor_optimizer/policy`。

**需要排查的风险与为什么要改**

同一因子同时需要信号研究与模型输入，不应一个原地修改的value文件被不同消费者当成不同版本。

**实施步骤**

1. 保留definition与recipe/state/value身份层次，raw、repair和representation分层。
2. 同一有效值可按完整语义共享物化对象；不同用途仅display alias不重复落值。
3. 所有更新与清理通过引用图，替换当前首选不删除仍被模型使用版本。

**必须落地的回归／验收**

1. raw与neutralized同日可同时读取且hash不同。
2. 模型使用的版本在重新搜索后仍固定。
3. 重复表示内容等价时有可审计复用而非第二份无血缘文件。

**依赖**：[V-18](#issue-v-18)、[V-20](#issue-v-20)。  **需求映射**：R11、R13、R14、R16。

**关闭方式**：若实际代码已满足，提交对应公开路径测试与证据，标为已覆盖；若需修改，提交权威实现及调用者变更、旧制品影响和测试结果；环境不足保留具体BLOCKED项，不以文档或元数据声明关闭。

---

<a id="issue-rcp-10"></a>
#### RCP-10 · 可读DSL与执行计划必须能无损往返并显示实际参数

**优先级**：P1　**证据分类**：E（待实仓核查）　**Owner**：FE/FP/FA

**现有定位范围**：`factor_preprocess/factor_preprocess`；`factor_engine`；`factor_optimizer/factor_optimizer/policy`。

**需要排查的风险与为什么要改**

自然语言“做了中性化和标准化”不足以重现；opaque callback也不能作为唯一生产定义。

**实施步骤**

1. 支持canonical DSL加显式sidecar state refs，两者共同决定执行身份。
2. render→parse→compile保持算子、参数、axis、timing、input fields及state binding。
3. 解释接口输出requested/effective参数、backend、可用能力和fallback，未知算子不自动替换近义名。

**必须落地的回归／验收**

1. 往返序列化后的value与原计划一致。
2. 改一项半衰期或暴露集产生新recipe。
3. 失效state ref或未知op编译失败，而非继续使用默认值。

**依赖**：[FP-03](#issue-fp-03)、[FP-12](#issue-fp-12)、[V-18](#issue-v-18)。  **需求映射**：R01、R13、R14。

**关闭方式**：若实际代码已满足，提交对应公开路径测试与证据，标为已覆盖；若需修改，提交权威实现及调用者变更、旧制品影响和测试结果；环境不足保留具体BLOCKED项，不以文档或元数据声明关闭。

---

### 12.4 SRH：小预算搜索、候选集与封存专项

<a id="issue-srh-01"></a>
#### SRH-01 · 参数变异需要证明改变有效执行，而非只改变描述

**优先级**：P1　**证据分类**：E（待实仓核查）　**Owner**：FO/FE/FP

**现有定位范围**：`factor_optimizer/factor_optimizer`；`factor_assets/selection`；`quant_evaluator`。

**需要排查的风险与为什么要改**

无效参数、inactive参数或canonical化后相同候选会浪费资源；也可能让搜索台账名义多样但输出完全一样。

**实施步骤**

1. 从FE参数元数据读取active role、依赖、范围和离散步长；生成后compile验证。
2. effective spec相同则复用评价并记录duplicate proposal，不重复昂贵计算。
3. 参数敏感性以合成输入金标验证，不以真实样本恰好数值相同判定所有参数无效。

**必须落地的回归／验收**

1. 切换inactive参数应拒绝或标no-op。
2. h3/h10在合成脉冲信号输出不同。
3. 搜索给出的FP参数到达FE调用trace。

**依赖**：[FP-03](#issue-fp-03)、[V-09](#issue-v-09)。  **需求映射**：R08、R09、R14。

**关闭方式**：若实际代码已满足，提交对应公开路径测试与证据，标为已覆盖；若需修改，提交权威实现及调用者变更、旧制品影响和测试结果；环境不足保留具体BLOCKED项，不以文档或元数据声明关闭。

---

<a id="issue-srh-02"></a>
#### SRH-02 · 诊断冲突与多问题并存时先修完整性再修效用

**优先级**：P1　**证据分类**：E（待实仓核查）　**Owner**：FO/FA/QE

**现有定位范围**：`factor_optimizer/factor_optimizer`；`factor_assets/selection`；`quant_evaluator`。

**需要排查的风险与为什么要改**

同一因子可能高换手、低覆盖又有size暴露；同时堆叠三类修复会失去因果归因并放大搜索。

**实施步骤**

1. 给诊断明确priority、repairability、prerequisites和互斥关系。
2. 数据时钟/错误单位/算术非法先停止效用优化；可比较raw的一项最主要可修问题先运行。
3. 每次修复后重新诊断，只有仍有证据的下一问题才激活后续family。

**必须落地的回归／验收**

1. 前视失败因子不通过平滑提高IC后被准入。
2. 高换手+tail异常可先独立少量候选而非全部排列。
3. 修复无效时有ABANDON理由且预算受限。

**依赖**：[V-09](#issue-v-09)、[FA-01](#issue-fa-01)。  **需求映射**：R03、R04、R05、R06、R09。

**关闭方式**：若实际代码已满足，提交对应公开路径测试与证据，标为已覆盖；若需修改，提交权威实现及调用者变更、旧制品影响和测试结果；环境不足保留具体BLOCKED项，不以文档或元数据声明关闭。

---

<a id="issue-srh-03"></a>
#### SRH-03 · Pareto需要容差、未知状态与可比较目标集合

**优先级**：P1　**证据分类**：E（待实仓核查）　**Owner**：FO/FA

**现有定位范围**：`factor_optimizer/factor_optimizer`；`factor_assets/selection`；`quant_evaluator`。

**需要排查的风险与为什么要改**

大量高度相关目标会使几乎所有候选非支配；浮点微小差异又会造成不稳定冠军。

**实施步骤**

1. 限定少数经济上不同的主要目标，其他指标当硬门/诊断。
2. 统一方向和单位，采用预声明epsilon或成对等价区，不允许测试后调epsilon。
3. 缺核心目标先待测/拒绝，不能用±inf填进去；相同目标再用复杂度和成本排序。

**必须落地的回归／验收**

1. 目标顺序及候选排列不改变选择集合。
2. 在容差内等价候选选稳定简单版本。
3. 增加一个重复目标不改变排序权重。

**依赖**：[FA-11](#issue-fa-11)、[V-16](#issue-v-16)。  **需求映射**：R03、R09、R11。

**关闭方式**：若实际代码已满足，提交对应公开路径测试与证据，标为已覆盖；若需修改，提交权威实现及调用者变更、旧制品影响和测试结果；环境不足保留具体BLOCKED项，不以文档或元数据声明关闭。

---

<a id="issue-srh-04"></a>
#### SRH-04 · 多赢家集合选择应验证边际贡献与母因子上限

**优先级**：P1　**证据分类**：E（待实仓核查）　**Owner**：FO/FA/QE/modeling

**现有定位范围**：`factor_optimizer/factor_optimizer`；`factor_assets/selection`；`quant_evaluator`。

**需要排查的风险与为什么要改**

多个候选各自达标不意味着一起有用；一族占满输入也会压低其他机制多样性。

**实施步骤**

1. 先在同一确认样本建立合格前沿，再与已选集合比较增量/替代价值。
2. 上限按母因子族与用途预声明，默认1–3仅为预算起点，允许0个。
3. 记录选择顺序依赖并用少量顺序稳定性或替换检查，不做无限组合搜索。

**必须落地的回归／验收**

1. 两个单独强但完全重复只能保留代表。
2. 一个低换手与一个高预测版本可因明确互补同时保留。
3. 选择空集是合法结果。

**依赖**：[V-16](#issue-v-16)、[STA-04](#issue-sta-04)。  **需求映射**：R09、R11、R12。

**关闭方式**：若实际代码已满足，提交对应公开路径测试与证据，标为已覆盖；若需修改，提交权威实现及调用者变更、旧制品影响和测试结果；环境不足保留具体BLOCKED项，不以文档或元数据声明关闭。

---

<a id="issue-srh-05"></a>
#### SRH-05 · 分阶段晋级必须防止短窗幸存偏差与样本不公平

**优先级**：P1　**证据分类**：E（待实仓核查）　**Owner**：FO/QE

**现有定位范围**：`factor_optimizer/factor_optimizer`；`factor_assets/selection`；`quant_evaluator`。

**需要排查的风险与为什么要改**

随机挑少数有利日期粗筛可能丢掉状态依赖信号；不同候选跑不同窗就比较raw分数也不公平。

**实施步骤**

1. 每stage按预声明连续块和状态覆盖取样，保存具体样本ref。
2. 晋级只分配更多资源，最终胜出必须在共同确认样本复评。
3. 建立小规模full evaluation抽检，估计粗筛误淘汰率以调整政策，不利用sealed样本回调。

**必须落地的回归／验收**

1. 相同候选不同stage不会覆盖前一份证据。
2. 低档未运行指标保持NOT_RUN。
3. 幸存者共同样本计数与目标成本一致。

**依赖**：[V-10](#issue-v-10)、[QE-28](#issue-qe-28)。  **需求映射**：R08、R09、R10。

**关闭方式**：若实际代码已满足，提交对应公开路径测试与证据，标为已覆盖；若需修改，提交权威实现及调用者变更、旧制品影响和测试结果；环境不足保留具体BLOCKED项，不以文档或元数据声明关闭。

---

<a id="issue-srh-06"></a>
#### SRH-06 · 预算预留、取消与重试要有持久化资源账本

**优先级**：P1　**证据分类**：E（待实仓核查）　**Owner**：FO/Platform

**现有定位范围**：`factor_optimizer/factor_optimizer`；`factor_assets/selection`；`quant_evaluator`。

**需要排查的风险与为什么要改**

并行试验若各自先查剩余预算再执行，可能同时超额；进程崩溃可能泄漏reservation或重复扣费。

**实施步骤**

1. 复用现有store提供reserve→start→settle/release事务和lease。
2. 区分估算成本与实耗，失败/OOM/取消也计实际资源；重试不删除原attempt。
3. 先保持单调度控制平面，具备原子预算后再开放worker并发。

**必须落地的回归／验收**

1. 两个worker竞争最后预算只一个获授权。
2. 崩溃reservation按lease恢复且已耗资源不退回。
3. 取消后不继续派生新候选或发布。

**依赖**：[FO-04](#issue-fo-04)、[FO-05](#issue-fo-05)、[V-09](#issue-v-09)。  **需求映射**：R08、R09、R16。

**关闭方式**：若实际代码已满足，提交对应公开路径测试与证据，标为已覆盖；若需修改，提交权威实现及调用者变更、旧制品影响和测试结果；环境不足保留具体BLOCKED项，不以文档或元数据声明关闭。

---

<a id="issue-srh-07"></a>
#### SRH-07 · 封存测试访问要绑定候选集合且不能在日志中泄露

**优先级**：P0　**证据分类**：E（待实仓核查）　**Owner**：FO/QE/Platform

**现有定位范围**：`factor_optimizer/factor_optimizer`；`factor_assets/selection`；`quant_evaluator`。

**需要排查的风险与为什么要改**

只限制函数调用不够，错误日志、可访问缓存和Agent上下文也可能泄露测试结果；重复执行故障又需合法恢复。

**实施步骤**

1. 冻结candidate-set hash、purpose、dataset/time scope、profile后由authority授权一次逻辑评估。
2. 已完成结果重试只读取immutable结果，不再抽选或拟合；数据未公开失败按明确状态恢复。
3. 搜索与Agent仅获test完成/资格结果所允许信息，访问原始test有独立角色权限和审计。

**必须落地的回归／验收**

1. 换名字或重启不能再次得到新test调参机会。
2. 异常trace不得把test收益面板放入Agent提案上下文。
3. 同一冻结集合的网络重试返回同一evidence ref。

**依赖**：[V-11](#issue-v-11)、[FO-02](#issue-fo-02)。  **需求映射**：R10、R14、R15、R17。

**关闭方式**：若实际代码已满足，提交对应公开路径测试与证据，标为已覆盖；若需修改，提交权威实现及调用者变更、旧制品影响和测试结果；环境不足保留具体BLOCKED项，不以文档或元数据声明关闭。

---

<a id="issue-srh-08"></a>
#### SRH-08 · 候选生成器与异步执行顺序必须可复现

**优先级**：P1　**证据分类**：E（待实仓核查）　**Owner**：FO

**现有定位范围**：`factor_optimizer/factor_optimizer`；`factor_assets/selection`；`quant_evaluator`。

**需要排查的风险与为什么要改**

仅固定随机seed不足以应对worker完成顺序、全局RNG污染和恢复时策略内部状态缺失。

**实施步骤**

1. 候选生成和试验RNG使用命名stream且记录state；预算顺序由逻辑trial index确定。
2. 异步场景明确更新incumbent的顺序政策，不能让速度优势变成未声明统计选择。
3. checkpoint保存待执行、运行中、已完成与策略模型状态；恢复不重新抽样覆盖旧候选。

**必须落地的回归／验收**

1. 固定seed下模拟不同完成顺序，按声明政策得到相同候选序列。
2. 中断再恢复与完整运行等价。
3. 一个无关随机调用不改变策略stream。

**依赖**：[FO-05](#issue-fo-05)、[V-09](#issue-v-09)。  **需求映射**：R09、R10、R14。

**关闭方式**：若实际代码已满足，提交对应公开路径测试与证据，标为已覆盖；若需修改，提交权威实现及调用者变更、旧制品影响和测试结果；环境不足保留具体BLOCKED项，不以文档或元数据声明关闭。

---

<a id="issue-srh-09"></a>
#### SRH-09 · 早停需考虑评价噪声、成本和基线而非裸分数平台

**优先级**：P1　**证据分类**：E（待实仓核查）　**Owner**：FO/QE

**现有定位范围**：`factor_optimizer/factor_optimizer`；`factor_assets/selection`；`quant_evaluator`。

**需要排查的风险与为什么要改**

噪声导致连续无改进不代表真平台；反复微小改善也可能无限消耗预算。

**实施步骤**

1. 定义minimum meaningful gain与最大探索预算，成对同窗比较raw/incumbent。
2. 非法候选和没有执行的候选不冒充差绩效，分别记attempt及stop reason。
3. 早停只终止资源分配，不能修改试验族或已封存判断。

**必须落地的回归／验收**

1. 纯噪声微小分数起伏最终受预算限制。
2. 尚有显著风险改善的候选按规则可晋级。
3. baseline无效时先完整性处理，不自动认为所有修复优秀。

**依赖**：[V-09](#issue-v-09)、[STA-03](#issue-sta-03)。  **需求映射**：R08、R09、R10。

**关闭方式**：若实际代码已满足，提交对应公开路径测试与证据，标为已覆盖；若需修改，提交权威实现及调用者变更、旧制品影响和测试结果；环境不足保留具体BLOCKED项，不以文档或元数据声明关闭。

---

<a id="issue-srh-10"></a>
#### SRH-10 · 失败修复、参数平台与以后复用应沉淀为有版本知识

**优先级**：P1　**证据分类**：E（待实仓核查）　**Owner**：FO/FA

**现有定位范围**：`factor_optimizer/factor_optimizer`；`factor_assets/selection`；`quant_evaluator`。

**需要排查的风险与为什么要改**

保留公式却丢失为何失败、适用环境和effective配方，会导致下一轮Agent不断重复浪费试验。

**实施步骤**

1. 失败元数据包括诊断、输入上下文、参数、reason、统计尝试及可重试条件。
2. 成功recipe可作为该类新因子的候选prior，但不可直接继承评价或grade。
3. 旧失败在数据/指标/政策变化后可受控重评，SeenIndex按身份与evaluation intent区分。

**必须落地的回归／验收**

1. 相同上下文失败候选可便宜跳过贵评但保留尝试记录。
2. 新数据窗口可合法重新评估。
3. 迁移旧recipe有新环境校验而非直接APPROVED。

**依赖**：[PL-02](#issue-pl-02)、[V-18](#issue-v-18)、[V-20](#issue-v-20)。  **需求映射**：R09、R14、R16、R17。

**关闭方式**：若实际代码已满足，提交对应公开路径测试与证据，标为已覆盖；若需修改，提交权威实现及调用者变更、旧制品影响和测试结果；环境不足保留具体BLOCKED项，不以文档或元数据声明关闭。

---

### 12.5 SIM：相似图、增量归簇与版本专项

<a id="issue-sim-01"></a>
#### SIM-01 · 相似度类型与逐日聚合方法必须写进fingerprint规格

**优先级**：P1　**证据分类**：E（待实仓核查）　**Owner**：FA/QE

**现有定位范围**：`factor_assets/similarity`；`factor_assets/graph`；`factor_assets/clustering`；`factor_assets/aggregation`；`quant_evaluator/metrics`。

**需要排查的风险与为什么要改**

值相关、rank相关、残差相关、IC序列和PnL相关对应不同冗余概念，不能用同一个similarity_score掩盖。

**实施步骤**

1. FingerprintSpec记录数据窗口、预处理、mask、方向、聚合方法与embedding模型版本。
2. 逐日截面相关再跨期聚合与展平相关另名，Fisher-z加权有样本依据。
3. 正式相关由QE计算，FA仅组织索引/图与阈值裁决。

**必须落地的回归／验收**

1. 相同值相关但不同净PnL的两个因子保留两份证据。
2. 交换输入次序相关对称。
3. 改变预处理spec产生新fingerprint身份。

**依赖**：[FA-09](#issue-fa-09)、[V-14](#issue-v-14)。  **需求映射**：R03、R11、R12。

**关闭方式**：若实际代码已满足，提交对应公开路径测试与证据，标为已覆盖；若需修改，提交权威实现及调用者变更、旧制品影响和测试结果；环境不足保留具体BLOCKED项，不以文档或元数据声明关闭。

---

<a id="issue-sim-02"></a>
#### SIM-02 · 共同覆盖不足时不能证明不重复或高相似

**优先级**：P1　**证据分类**：E（待实仓核查）　**Owner**：FA/QE

**现有定位范围**：`factor_assets/similarity`；`factor_assets/graph`；`factor_assets/clustering`；`factor_assets/aggregation`；`quant_evaluator/metrics`。

**需要排查的风险与为什么要改**

只有少数共同交易日或证券的相关性即使很高，也不够支持长期族归属；没有共同数据更不等于0。

**实施步骤**

1. PairwiseArtifact带n_days、每日期有效n、覆盖范围和不确定性。
2. 阈值前先检查最小有效重叠与窗口稳定性；不满足进入UNKNOWN_PENDING_EVIDENCE。
3. 不为提速填0扩充共同样本；signed和absolute相关同时可追溯。

**必须落地的回归／验收**

1. 完全不重叠两因子返回未知。
2. 十天高相关与两年高相关证据等级不同。
3. 移除无效日期不改变已声明有效样本上的计算。

**依赖**：[FA-09](#issue-fa-09)、[STA-01](#issue-sta-01)。  **需求映射**：R03、R12、R15。

**关闭方式**：若实际代码已满足，提交对应公开路径测试与证据，标为已覆盖；若需修改，提交权威实现及调用者变更、旧制品影响和测试结果；环境不足保留具体BLOCKED项，不以文档或元数据声明关闭。

---

<a id="issue-sim-03"></a>
#### SIM-03 · 极高相关候选也要检查尾部、mask和参数平台差异

**优先级**：P1　**证据分类**：E（待实仓核查）　**Owner**：FA/QE/FO

**现有定位范围**：`factor_assets/similarity`；`factor_assets/graph`；`factor_assets/clustering`；`factor_assets/aggregation`；`quant_evaluator/metrics`。

**需要排查的风险与为什么要改**

ρ≥.99只说明样本内相近，不足以宣布公式等价；小差异可能集中在重要尾部或低成本路径。

**实施步骤**

1. 近重复复核比较多窗口相关、数值误差分布、尾部排序差异、mask一致率和净成本。
2. 精确等价、方向等价、近重复和可替代四种结论分开。
3. 保留更简单/便宜或更稳定代表，并为不同用途提供有证据例外，不按相关阈值自动删公式。

**必须落地的回归／验收**

1. 全期相关高但压力期方向不同不能认定精确重复。
2. 值相同mask不同不合并成同一个value identity。
3. 更便宜实现通过相同语义证据可优先。

**依赖**：[V-14](#issue-v-14)、[V-16](#issue-v-16)。  **需求映射**：R09、R11、R12。

**关闭方式**：若实际代码已满足，提交对应公开路径测试与证据，标为已覆盖；若需修改，提交权威实现及调用者变更、旧制品影响和测试结果；环境不足保留具体BLOCKED项，不以文档或元数据声明关闭。

---

<a id="issue-sim-04"></a>
#### SIM-04 · 相似图不等于协方差矩阵，缺边补零需禁止隐式使用

**优先级**：P1　**证据分类**：E（待实仓核查）　**Owner**：FA/QE/modeling

**现有定位范围**：`factor_assets/similarity`；`factor_assets/graph`；`factor_assets/clustering`；`factor_assets/aggregation`；`quant_evaluator/metrics`。

**需要排查的风险与为什么要改**

pairwise不同样本相关矩阵不一定能直接作可逆风险矩阵；稀疏检索图更不是完整协方差估计。

**实施步骤**

1. GraphArtifact与CovarianceArtifact分不同类型，未测边有显式状态。
2. 风险模型需要协方差时调用已有QE/风险域估计方法，声明共同样本、收缩/PSD修复及误差。
3. 任何PSD投影/收缩生成新artifact，不能悄悄修改正式相似证据。

**必须落地的回归／验收**

1. 稀疏ANN图不能直接传组合风险优化器当covariance。
2. 非PSD输入在要求PSD的函数中拒绝或走显式修复。
3. 修复前后矩阵和方法可审计。

**依赖**：[FA-09](#issue-fa-09)、[V-14](#issue-v-14)。  **需求映射**：R11、R12、R15。

**关闭方式**：若实际代码已满足，提交对应公开路径测试与证据，标为已覆盖；若需修改，提交权威实现及调用者变更、旧制品影响和测试结果；环境不足保留具体BLOCKED项，不以文档或元数据声明关闭。

---

<a id="issue-sim-05"></a>
#### SIM-05 · Leiden分簇结果还要检查经济同质性与方向一致

**优先级**：P1　**证据分类**：E（待实仓核查）　**Owner**：FA/QE

**现有定位范围**：`factor_assets/similarity`；`factor_assets/graph`；`factor_assets/clustering`；`factor_assets/aggregation`；`quant_evaluator/metrics`。

**需要排查的风险与为什么要改**

图社区连通并不要求簇内任意两点相近；传递链可把不同风险机制连接在一起。

**实施步骤**

1. 保留现有算法并记录resolution/seed/边政策；额外计算medoid affinity、跨窗一致性、簇内异质性与方向冲突。
2. 过大或bridge驱动簇进入拆分研究，不机械按成员数扩大。
3. 负相关边如何映射相似族与交易方向写政策，aggregation先方向对齐再考虑权重。

**必须落地的回归／验收**

1. A-B强、B-C强而A-C弱的链图不自动宣称全部高相关。
2. 同簇相反方向信号聚合前检测。
3. 不同seed输出的质量和稳定性可比较。

**依赖**：[FA-08](#issue-fa-08)、[FA-09](#issue-fa-09)。  **需求映射**：R11、R12。

**关闭方式**：若实际代码已满足，提交对应公开路径测试与证据，标为已覆盖；若需修改，提交权威实现及调用者变更、旧制品影响和测试结果；环境不足保留具体BLOCKED项，不以文档或元数据声明关闭。

---

<a id="issue-sim-06"></a>
#### SIM-06 · 未测边、测得低边与认证图内容必须分开

**优先级**：P1　**证据分类**：E（待实仓核查）　**Owner**：FA/DA

**现有定位范围**：`factor_assets/similarity`；`factor_assets/graph`；`factor_assets/clustering`；`factor_assets/aggregation`；`quant_evaluator/metrics`。

**需要排查的风险与为什么要改**

图认证若只校验数组hash，却没有候选库版本、来源指标和缺边含义，仍可能给错误上下文的图合法印章。

**实施步骤**

1. GraphManifest绑定library snapshot、pairwise metric版本、window、universe、mask政策及edge coverage。
2. 明确低于阈值删边和根本未测两种来源，不能假设后者相关为0。
3. 认证器验证实际边数据与manifest，typed refs和图hash都解析通过才聚类。

**必须落地的回归／验收**

1. 同边数组换股票池或窗口不得沿用认证。
2. 增加未测节点需要pending标志而非默认孤立真新颖。
3. 图字节篡改或缺pairwise证据拒绝正式聚类。

**依赖**：[FA-09](#issue-fa-09)、[V-18](#issue-v-18)。  **需求映射**：R12、R14、R15。

**关闭方式**：若实际代码已满足，提交对应公开路径测试与证据，标为已覆盖；若需修改，提交权威实现及调用者变更、旧制品影响和测试结果；环境不足保留具体BLOCKED项，不以文档或元数据声明关闭。

---

<a id="issue-sim-07"></a>
#### SIM-07 · ANN索引的召回质量、版本和增量维护需要验收

**优先级**：P2　**证据分类**：E（待实仓核查）　**Owner**：FA/QE

**现有定位范围**：`factor_assets/similarity`；`factor_assets/graph`；`factor_assets/clustering`；`factor_assets/aggregation`；`quant_evaluator/metrics`。

**需要排查的风险与为什么要改**

提高ANN速度若漏掉高相似候选，会让重复因子进入库；仅有索引对象不等于可靠召回。

**实施步骤**

1. 固定版本index以embedding spec和成员集hash寻址；支持delta index或受控重建。
2. 从历史开发快照抽取审计query用exact检索核对召回，保存近重复区域recall而不只平均recall。
3. 召回不足调整预算或精确fallback；严格去重声明不得超出已测覆盖。

**必须落地的回归／验收**

1. 索引build次数与版本更新而非新因子×簇次数相关。
2. 删除/废弃因子不继续作为无标注近邻。
3. 故意降低ANN精度时能力报告显示不够认证。

**依赖**：[FA-10](#issue-fa-10)、[V-14](#issue-v-14)。  **需求映射**：R08、R12。

**关闭方式**：若实际代码已满足，提交对应公开路径测试与证据，标为已覆盖；若需修改，提交权威实现及调用者变更、旧制品影响和测试结果；环境不足保留具体BLOCKED项，不以文档或元数据声明关闭。

---

<a id="issue-sim-08"></a>
#### SIM-08 · 增量归簇要比较多个支持邻居和歧义间隔

**优先级**：P1　**证据分类**：E（待实仓核查）　**Owner**：FA

**现有定位范围**：`factor_assets/similarity`；`factor_assets/graph`；`factor_assets/clustering`；`factor_assets/aggregation`；`quant_evaluator/metrics`。

**需要排查的风险与为什么要改**

单个最近成员可能是桥接异常值；强行投大簇也会丢掉新机制。

**实施步骤**

1. 预声明medoid与top-k支持规则，要求足够有效证据后再计算cluster affinity。
2. 最优与次优差在噪声区时标AMBIGUOUS；新族与pending区有容量与复核政策。
3. 任何改投重新计算最终目标证据，保存候选簇排名和拒绝原因。

**必须落地的回归／验收**

1. 一名近邻与其余成员完全不同不直接过强同族门。
2. A/B相近时不根据列表顺序强选。
3. 新因子没有合格簇时状态合法并可后续产生新簇。

**依赖**：[FA-08](#issue-fa-08)、[FA-10](#issue-fa-10)。  **需求映射**：R11、R12。

**关闭方式**：若实际代码已满足，提交对应公开路径测试与证据，标为已覆盖；若需修改，提交权威实现及调用者变更、旧制品影响和测试结果；环境不足保留具体BLOCKED项，不以文档或元数据声明关闭。

---

<a id="issue-sim-09"></a>
#### SIM-09 · 簇split/merge要保留逻辑身份和旧模型引用

**优先级**：P1　**证据分类**：E（待实仓核查）　**Owner**：FA/modeling

**现有定位范围**：`factor_assets/similarity`；`factor_assets/graph`；`factor_assets/clustering`；`factor_assets/aggregation`；`quant_evaluator/metrics`。

**需要排查的风险与为什么要改**

全量重聚类后重新编号cluster_1可能对应完全不同成员；线上模型不能因为名字相同就继续消费。

**实施步骤**

1. 区分logical cluster ID与immutable cluster version；建立父子split/merge匹配证据。
2. 全局refresh是新研究版本，旧发布簇保持可读；模型依赖精确version。
3. 漂移触发和固定频率都版本化，刷新后先质量和compatibility gate。

**必须落地的回归／验收**

1. 簇编号置换不应构成假模型变化；真实成员/聚合变化必须触发。
2. 旧模型在refresh后仍读旧函数。
3. 拆分合并血缘可从任一新簇追溯。

**依赖**：[V-15](#issue-v-15)、[PL-06](#issue-pl-06)。  **需求映射**：R12、R14。

**关闭方式**：若实际代码已满足，提交对应公开路径测试与证据，标为已覆盖；若需修改，提交权威实现及调用者变更、旧制品影响和测试结果；环境不足保留具体BLOCKED项，不以文档或元数据声明关闭。

---

<a id="issue-sim-10"></a>
#### SIM-10 · 簇压缩成模型特征时，聚合函数与拟合窗口须固定

**优先级**：P1　**证据分类**：E（待实仓核查）　**Owner**：FA/FE/FP/modeling

**现有定位范围**：`factor_assets/similarity`；`factor_assets/graph`；`factor_assets/clustering`；`factor_assets/aggregation`；`quant_evaluator/metrics`。

**需要排查的风险与为什么要改**

簇不必压成单均值；PCA、加权平均或代表选择都有不同信息与拟合状态，成员变化不能隐形改变列。

**实施步骤**

1. 声明KEEP_MEMBERS/MEDOID/FROZEN_WEIGHTED/PCA等实际采用方案，先复用已有aggregation。
2. 有值的聚合由FE执行或复用明确权威计算，拟合权重/主成分仅用训练数据并落state。
3. 父/子/聚合都经QE独立评价，缺成员是否重归一按冻结recipe。

**必须落地的回归／验收**

1. 同列名不同成员/权重必须不同feature identity。
2. 测试数据变化不改变PCA loading。
3. 缺一个成员不能无说明动态改权重。

**依赖**：[V-17](#issue-v-17)、[V-18](#issue-v-18)。  **需求映射**：R11、R12、R14。

**关闭方式**：若实际代码已满足，提交对应公开路径测试与证据，标为已覆盖；若需修改，提交权威实现及调用者变更、旧制品影响和测试结果；环境不足保留具体BLOCKED项，不以文档或元数据声明关闭。

---

### 12.6 MOD：模型输入、OOS增量与更新专项

<a id="issue-mod-01"></a>
#### MOD-01 · 模型每个训练fold都要重新拟合特征选择和预处理状态

**优先级**：P0　**证据分类**：E（待实仓核查）　**Owner**：modeling/FP/FA

**现有定位范围**：`modeling`；`lightgbm_qs`；`factor_preprocess/factor_preprocess/representation`；`factor_assets/assembly`；`quant_platform/app/contracts`。

**需要排查的风险与为什么要改**

只按fold训练模型，却用全历史选因子、拟合scaler或聚类，仍会把验证数据用于选择。

**实施步骤**

1. 训练dataset manifest绑定该fold可得的因子库、cluster、recipe和fit-state。
2. 全数据用于非生产探索的结果标EX_POST，不送严格OOF评价。
3. 生产训练终版可用全部development样本，但独立holdout不可进入任何拟合/选择。

**必须落地的回归／验收**

1. 修改heldout标签不能改变训练fold因子选择。
2. 修改heldout特征也不改变跨时间fit-state。
3. 过去lookback只读可允许，不等于未来拟合。

**依赖**：[V-08](#issue-v-08)、[V-11](#issue-v-11)、[V-12](#issue-v-12)。  **需求映射**：R10、R11、R14、R15。

**关闭方式**：若实际代码已满足，提交对应公开路径测试与证据，标为已覆盖；若需修改，提交权威实现及调用者变更、旧制品影响和测试结果；环境不足保留具体BLOCKED项，不以文档或元数据声明关闭。

---

<a id="issue-mod-02"></a>
#### MOD-02 · 低单因子IC特征必须通过有限预算的增量检验

**优先级**：P1　**证据分类**：E（待实仓核查）　**Owner**：modeling/QE/FA

**现有定位范围**：`modeling`；`lightgbm_qs`；`factor_preprocess/factor_preprocess/representation`；`factor_assets/assembly`；`quant_platform/app/contracts`。

**需要排查的风险与为什么要改**

不能仅说树模型能利用交互就放行所有低IC特征，也不能只依赖重要性/SHAP大小当新增价值。

**实施步骤**

1. 冻结baseline特征集、训练预算和时间fold，用相同随机种子及资源做add/drop/substitute实验。
2. QE评价新增预测或净组合效用及稳定性，modeling只输出预测artifact。
3. 增量试验计入campaign，先小样本proxy再共同确认，不搜索无限模型配置。

**必须落地的回归／验收**

1. 加入纯噪声特征不能凭训练loss下降过门。
2. 稳定U或互补特征可以在独立验证证明价值。
3. 相同baseline和资源下结果可重复。

**依赖**：[V-05](#issue-v-05)、[V-16](#issue-v-16)、[STA-03](#issue-sta-03)。  **需求映射**：R09、R10、R11。

**关闭方式**：若实际代码已满足，提交对应公开路径测试与证据，标为已覆盖；若需修改，提交权威实现及调用者变更、旧制品影响和测试结果；环境不足保留具体BLOCKED项，不以文档或元数据声明关闭。

---

<a id="issue-mod-03"></a>
#### MOD-03 · 中性化的非劣性容忍区要与消费者目标一致

**优先级**：P1　**证据分类**：E（待实仓核查）　**Owner**：FO/FP/modeling/FA

**现有定位范围**：`modeling`；`lightgbm_qs`；`factor_preprocess/factor_preprocess/representation`；`factor_assets/assembly`；`quant_platform/app/contracts`。

**需要排查的风险与为什么要改**

单因子IC轻微下降并不自动意味着模型效用没损失；反之单因子Sharpe差也未必模型无用。

**实施步骤**

1. consumer profile声明比较目标：prediction loss、RankIC、净组合utility或风险预算。
2. 成对比较给delta CI、effect size与风险降低；预声明epsilon和最小风险改善条件。
3. 输出raw/neutralized分别准入，不按测试效果临时换目标。

**必须落地的回归／验收**

1. IC非劣但模型净效用明显下降时不能误判同等。
2. 近零baseline禁止仅百分比保留率。
3. 严格风险profile不允许效用高分掩盖超限。

**依赖**：[V-06](#issue-v-06)、[V-08](#issue-v-08)。  **需求映射**：R05、R11、R13。

**关闭方式**：若实际代码已满足，提交对应公开路径测试与证据，标为已覆盖；若需修改，提交权威实现及调用者变更、旧制品影响和测试结果；环境不足保留具体BLOCKED项，不以文档或元数据声明关闭。

---

<a id="issue-mod-04"></a>
#### MOD-04 · 神经网络、树与线性模型各自处理缺失和尺度

**优先级**：P1　**证据分类**：E（待实仓核查）　**Owner**：FP/modeling

**现有定位范围**：`modeling`；`lightgbm_qs`；`factor_preprocess/factor_preprocess/representation`；`factor_assets/assembly`；`quant_platform/app/contracts`。

**需要排查的风险与为什么要改**

全库统一zscore会抹掉部分幅度信息；填0可能混淆真实0与未知，辅助mask也可能被错当alpha。

**实施步骤**

1. 按模型合同配置raw-clean、rank、robust scale、train-fit zscore与missing-channel，不默认全部强加。
2. 神经输入明确填充值及mask，树模型按其实际实现能力处理NaN，类别编码仅训练拟合。
3. 保留CS与TS缩放区别，验证可用截面范围，避免用未来batch分布fit。

**必须落地的回归／验收**

1. 有真实0和缺失两证券在mask通道可区分。
2. 同日分批推理与整截面在合同要求下等价。
3. 未知类别有显式处理而非重新编码全批。

**依赖**：[V-08](#issue-v-08)、[DTA-08](#issue-dta-08)。  **需求映射**：R11、R13、R15。

**关闭方式**：若实际代码已满足，提交对应公开路径测试与证据，标为已覆盖；若需修改，提交权威实现及调用者变更、旧制品影响和测试结果；环境不足保留具体BLOCKED项，不以文档或元数据声明关闭。

---

<a id="issue-mod-05"></a>
#### MOD-05 · 模型特征顺序、列内定义与数据集指纹必须严格校验

**优先级**：P0　**证据分类**：E（待实仓核查）　**Owner**：modeling/FA/Platform

**现有定位范围**：`modeling`；`lightgbm_qs`；`factor_preprocess/factor_preprocess/representation`；`factor_assets/assembly`；`quant_platform/app/contracts`。

**需要排查的风险与为什么要改**

只有列数和名称检查不足：权重、配方、字段快照或列顺序都可能改变输入含义。

**实施步骤**

1. FeatureManifest包含ordered value/recipe/state refs、dtype、mask及consumer profile。
2. 训练与推理比较完整schema hash，缺列/多列/换顺序必须显式对齐或拒绝。
3. 不要按dict迭代顺序临时构建模型矩阵，零值补缺列不得静默发生。

**必须落地的回归／验收**

1. 相同列名但neutralization exposure变化触发拒绝。
2. 交换两列若未显式映射不能继续预测。
3. 同manifest回放训练输入字节/数值可核对。

**依赖**：[PL-05](#issue-pl-05)、[PL-06](#issue-pl-06)、[V-18](#issue-v-18)。  **需求映射**：R12、R14、R15。

**关闭方式**：若实际代码已满足，提交对应公开路径测试与证据，标为已覆盖；若需修改，提交权威实现及调用者变更、旧制品影响和测试结果；环境不足保留具体BLOCKED项，不以文档或元数据声明关闭。

---

<a id="issue-mod-06"></a>
#### MOD-06 · 新增研究因子不能自动改变线上簇列或模型输入

**优先级**：P1　**证据分类**：E（待实仓核查）　**Owner**：FA/modeling/Platform

**现有定位范围**：`modeling`；`lightgbm_qs`；`factor_preprocess/factor_preprocess/representation`；`factor_assets/assembly`；`quant_platform/app/contracts`。

**需要排查的风险与为什么要改**

用户希望动态聚类又不频繁重训，应通过研究/发布双版本实现，而不是悄悄将新成员加进旧线上列。

**实施步骤**

1. 研究membership可频繁变动，线上模型只引用冻结FeatureSetVersion。
2. 新增成员只有经aggregation复评与compatibility/model gate后进入新版本。
3. 仅标签/备注更新不强迫重训，实际输入函数变化即使维数不变仍需再验证。

**必须落地的回归／验收**

1. 研究归簇运行后线上特征值保持不变。
2. 新发布版本才改变列内计算。
3. 失败shadow版本不能修改current production pointer。

**依赖**：[SIM-09](#issue-sim-09)、[SIM-10](#issue-sim-10)、[PL-06](#issue-pl-06)。  **需求映射**：R12、R14。

**关闭方式**：若实际代码已满足，提交对应公开路径测试与证据，标为已覆盖；若需修改，提交权威实现及调用者变更、旧制品影响和测试结果；环境不足保留具体BLOCKED项，不以文档或元数据声明关闭。

---

<a id="issue-mod-07"></a>
#### MOD-07 · 模型预测评价、组合评价与特征归因要分开

**优先级**：P1　**证据分类**：E（待实仓核查）　**Owner**：modeling/QE/FA

**现有定位范围**：`modeling`；`lightgbm_qs`；`factor_preprocess/factor_preprocess/representation`；`factor_assets/assembly`；`quant_platform/app/contracts`。

**需要排查的风险与为什么要改**

一个模型的好Sharpe不能直接分配给其每个输入因子；特征importance也不能替代单因子证据或增量检验。

**实施步骤**

1. 模型预测以prediction_id关联OOS标签，QE按模型输出算预测证据。
2. 执行域把预测转持仓，QE另算组合证据；因子通过ablation/conditional研究取得增量ref。
3. 报告分开raw factor grade、model contribution和portfolio performance，避免再次跨因子复制。

**必须落地的回归／验收**

1. 替换一个无关输入不会让所有因子继承同一模型Sharpe。
2. 训练内importance与OOS增量分栏。
3. 预测顺序和标签轴错配在QE拒绝。

**依赖**：[QE-01](#issue-qe-01)、[MOD-02](#issue-mod-02)。  **需求映射**：R03、R07、R11。

**关闭方式**：若实际代码已满足，提交对应公开路径测试与证据，标为已覆盖；若需修改，提交权威实现及调用者变更、旧制品影响和测试结果；环境不足保留具体BLOCKED项，不以文档或元数据声明关闭。

---

<a id="issue-mod-08"></a>
#### MOD-08 · 漂移监测、影子运行和回滚不能触发无界自动优化

**优先级**：P1　**证据分类**：E（待实仓核查）　**Owner**：FA/modeling/Platform

**现有定位范围**：`modeling`；`lightgbm_qs`；`factor_preprocess/factor_preprocess/representation`；`factor_assets/assembly`；`quant_platform/app/contracts`。

**需要排查的风险与为什么要改**

指标下降可以是数据故障或市场变化；自动优化到历史变好会破坏封存纪律，频繁重训也有风险。

**实施步骤**

1. 监测data/feature/exposure/predictive四类漂移并分流；预测指标只使用成熟新样本。
2. 风险违规按已批准政策降级/隔离，重新研究生成新campaign而非修改旧recipe。
3. shadow评估、发布和rollback各有版本与许可，旧模型/数据依赖保留至回滚窗口结束。

**必须落地的回归／验收**

1. 未成熟标签不触发假预测下降。
2. 数据格式故障先数据告警而不自动调参。
3. 回滚恢复模型与匹配feature版本而不是仅权重文件。

**依赖**：[V-21](#issue-v-21)、[PL-08](#issue-pl-08)。  **需求映射**：R03、R10、R14、R15。

**关闭方式**：若实际代码已满足，提交对应公开路径测试与证据，标为已覆盖；若需修改，提交权威实现及调用者变更、旧制品影响和测试结果；环境不足保留具体BLOCKED项，不以文档或元数据声明关闭。

---

### 12.7 OPS：运行、资源、发布、清理与重放专项

<a id="issue-ops-01"></a>
#### OPS-01 · 相同工作意图的并发写入必须原子幂等

**优先级**：P0　**证据分类**：E（待实仓核查）　**Owner**：Platform/DA/FA

**现有定位范围**：`quant_platform/app`；`data_access`；`factor_assets/lifecycle`；`factor_assets/registry`；`quant_evaluator/runtime`；`factor_optimizer/factor_optimizer/contracts`。

**需要排查的风险与为什么要改**

相同formula不同data/policy需要合法重评，相同意图重试又不能产生重复批准。幂等键必须覆盖正确身份层。

**实施步骤**

1. 以definition/recipe/data/evaluation spec/purpose构造work intent，attempt ID单独。
2. 在现有DB事务实现claim/lease/terminal结果唯一约束；冲突返回已有结果。
3. 对象写入、证据登记与发布用可恢复协议，不能仅凭内存set防重复。

**必须落地的回归／验收**

1. 两个worker同意图只能一个terminal发布。
2. 改变数据snapshot后不被旧seen错误跳过。
3. 进程重启重放不丢失败待重试项。

**依赖**：[PL-02](#issue-pl-02)、[PL-08](#issue-pl-08)。  **需求映射**：R09、R14、R16。

**关闭方式**：若实际代码已满足，提交对应公开路径测试与证据，标为已覆盖；若需修改，提交权威实现及调用者变更、旧制品影响和测试结果；环境不足保留具体BLOCKED项，不以文档或元数据声明关闭。

---

<a id="issue-ops-02"></a>
#### OPS-02 · 资源调度要包含共享数据、显存工作区和背压

**优先级**：P1　**证据分类**：E（待实仓核查）　**Owner**：FO/QE/FE/Platform

**现有定位范围**：`quant_platform/app`；`data_access`；`factor_assets/lifecycle`；`factor_assets/registry`；`quant_evaluator/runtime`；`factor_optimizer/factor_optimizer/contracts`。

**需要排查的风险与为什么要改**

输入数组估算不足以覆盖排序、mask、回归和PnL临时张量；多个任务共享GPU还会突然OOM。

**实施步骤**

1. 实际planner估算峰值工作集而非只value bytes，按F批和合法T流式切分。
2. 按设备/进程设置reservation与并发限制，超预算排队/backpressure。
3. 共享只读标签/字段缓存有引用计数和生命周期，不能每trial重复加载全历史。

**必须落地的回归／验收**

1. 同时多个大任务不突破已声明reservation。
2. F批大小减少时金融定义不变。
3. 预算含DataFrame转换、hash与落盘峰值。

**依赖**：[QE-28](#issue-qe-28)、[FP-05](#issue-fp-05)、[SRH-06](#issue-srh-06)。  **需求映射**：R08、R09。

**关闭方式**：若实际代码已满足，提交对应公开路径测试与证据，标为已覆盖；若需修改，提交权威实现及调用者变更、旧制品影响和测试结果；环境不足保留具体BLOCKED项，不以文档或元数据声明关闭。

---

<a id="issue-ops-03"></a>
#### OPS-03 · OOM和后端降级必须保持语义且不能伪报GPU完成

**优先级**：P1　**证据分类**：E（待实仓核查）　**Owner**：QE/FE/Platform

**现有定位范围**：`quant_platform/app`；`data_access`；`factor_assets/lifecycle`；`factor_assets/registry`；`quant_evaluator/runtime`；`factor_optimizer/factor_optimizer/contracts`。

**需要排查的风险与为什么要改**

GPU故障后简单转CPU或少算几个指标可能得到不同结果，但报告仍称同一完整请求完成。

**实施步骤**

1. 严格CUDA拒绝非允许fallback；auto模式记录fallback原因、effective backend和新执行计划。
2. OOM优先减F批或合法时间块，保持相同metric、horizon、mask和数值政策。
3. 部分结果带partial状态，required指标未完成不准入，失败消耗入账。

**必须落地的回归／验收**

1. 模拟OOM后缩批结果与全批一致。
2. 不能自动把Q20改Q5来过显存限制。
3. 没有运行CUDA的结果不能标GPU_PARITY_PASS。

**依赖**：[QE-04](#issue-qe-04)、[QE-05](#issue-qe-05)、[V-23](#issue-v-23)。  **需求映射**：R08、R09、R15。

**关闭方式**：若实际代码已满足，提交对应公开路径测试与证据，标为已覆盖；若需修改，提交权威实现及调用者变更、旧制品影响和测试结果；环境不足保留具体BLOCKED项，不以文档或元数据声明关闭。

---

<a id="issue-ops-04"></a>
#### OPS-04 · 性能基准必须覆盖端到端与设备同步边界

**优先级**：P2　**证据分类**：E（待实仓核查）　**Owner**：QA/QE/FE

**现有定位范围**：`quant_platform/app`；`data_access`；`factor_assets/lifecycle`；`factor_assets/registry`；`quant_evaluator/runtime`；`factor_optimizer/factor_optimizer/contracts`。

**需要排查的风险与为什么要改**

只计kernel发射时间会低估异步GPU实际耗时；热缓存与冷启动也不能混比。

**实施步骤**

1. 记录硬件、依赖版本、T/N/F、dtype、Q、metric/profile、数据量和缓存状态。
2. 拆分I/O、compile、transfer、kernel、sync、serialize、publish与hash耗时，并给端到端总计。
3. 先验算再测速度，对每种受支持路径记录误差、缺失状态与资源峰值。

**必须落地的回归／验收**

1. 设备完成前不停止计时。
2. CPU/GPU使用同数据同指标而非不同任务。
3. 冷/热缓存各自可重跑且包含构建索引成本。

**依赖**：[V-23](#issue-v-23)、[QA-02](#issue-qa-02)。  **需求映射**：R08、R14。

**关闭方式**：若实际代码已满足，提交对应公开路径测试与证据，标为已覆盖；若需修改，提交权威实现及调用者变更、旧制品影响和测试结果；环境不足保留具体BLOCKED项，不以文档或元数据声明关闭。

---

<a id="issue-ops-05"></a>
#### OPS-05 · 失败物化值GC需要根集合、租约和回滚保留窗口

**优先级**：P0　**证据分类**：E（待实仓核查）　**Owner**：FA/DA/Platform

**现有定位范围**：`quant_platform/app`；`data_access`；`factor_assets/lifecycle`；`factor_assets/registry`；`quant_evaluator/runtime`；`factor_optimizer/factor_optimizer/contracts`。

**需要排查的风险与为什么要改**

只看trial失败或value无直接引用会误删共享DAG、模型旧版本和读任务正在使用的对象。

**实施步骤**

1. 从production/approved release、active reads、retryable jobs、保留研究样本与rollback versions收集roots。
2. 沿transitive refs标记live对象，失败values只在终态且不live时成为候选。
3. 先输出GC dry-run按对象版本的清单和预计回收，真实删除走既有审批及受限prefix。

**必须落地的回归／验收**

1. 被子因子引用的失败母中间值不删。
2. 无人引用终态失败值可回收。
3. 在读租约存在时清理延期且记录理由。

**依赖**：[V-20](#issue-v-20)、[PL-07](#issue-pl-07)。  **需求映射**：R14、R16。

**关闭方式**：若实际代码已满足，提交对应公开路径测试与证据，标为已覆盖；若需修改，提交权威实现及调用者变更、旧制品影响和测试结果；环境不足保留具体BLOCKED项，不以文档或元数据声明关闭。

---

<a id="issue-ops-06"></a>
#### OPS-06 · 扫描后新增引用的GC竞态必须可防护

**优先级**：P0　**证据分类**：E（待实仓核查）　**Owner**：FA/DA/Platform

**现有定位范围**：`quant_platform/app`；`data_access`；`factor_assets/lifecycle`；`factor_assets/registry`；`quant_evaluator/runtime`；`factor_optimizer/factor_optimizer/contracts`。

**需要排查的风险与为什么要改**

扫描时无引用不代表删除时仍无引用；另一个任务可能在两者之间把对象纳入生产版本。

**实施步骤**

1. 使用tombstone/epoch或引用事务屏障，删除前复核引用版本与lease。
2. 被标删除对象禁止新发布引用或需原子撤销tombstone；最终对象删除有幂等key。
3. 处理中断可恢复，不以目录级rm代替逐对象受保护删除。

**必须落地的回归／验收**

1. 在mark和sweep之间创建合法引用，对象不得被误删。
2. 两个GC worker重复执行同plan结果一致。
3. 删除期间发布失败不留下指向缺对象的current版本。

**依赖**：[OPS-05](#issue-ops-05)、[PL-08](#issue-pl-08)。  **需求映射**：R14、R16。

**关闭方式**：若实际代码已满足，提交对应公开路径测试与证据，标为已覆盖；若需修改，提交权威实现及调用者变更、旧制品影响和测试结果；环境不足保留具体BLOCKED项，不以文档或元数据声明关闭。

---

<a id="issue-ops-07"></a>
#### OPS-07 · 删除回执要覆盖对象版本、缓存和实际空间回收

**优先级**：P1　**证据分类**：E（待实仓核查）　**Owner**：DA/FA

**现有定位范围**：`quant_platform/app`；`data_access`；`factor_assets/lifecycle`；`factor_assets/registry`；`quant_evaluator/runtime`；`factor_optimizer/factor_optimizer/contracts`。

**需要排查的风险与为什么要改**

删除数据库行或写墓碑可能不释放对象存储旧版本；缓存仍存失败值也可能浪费空间或误命中。

**实施步骤**

1. 物化对象、multipart暂存、CPU/GPU/disk cache与历史object versions分别纳入保留政策。
2. 区分logical_deleted、physical_deleted和provider_retention_pending，不虚报全部空间已回收。
3. 回执含对象ID/version、删除时间、验证结果、保留原因；公式/recipe/trial/必要证据不删除。

**必须落地的回归／验收**

1. 读回被物理删除失败值返回不存在且不会命中陈旧cache。
2. provider保留锁存在时明确pending。
3. 重复清理不删同名不同version的生产对象。

**依赖**：[OPS-05](#issue-ops-05)、[OPS-06](#issue-ops-06)。  **需求映射**：R08、R16。

**关闭方式**：若实际代码已满足，提交对应公开路径测试与证据，标为已覆盖；若需修改，提交权威实现及调用者变更、旧制品影响和测试结果；环境不足保留具体BLOCKED项，不以文档或元数据声明关闭。

---

<a id="issue-ops-08"></a>
#### OPS-08 · 算法修复后要按依赖失效历史评级、簇和模型

**优先级**：P0　**证据分类**：E（待实仓核查）　**Owner**：FA/QE/FE/Platform

**现有定位范围**：`quant_platform/app`；`data_access`；`factor_assets/lifecycle`；`factor_assets/registry`；`quant_evaluator/runtime`；`factor_optimizer/factor_optimizer/contracts`。

**需要排查的风险与为什么要改**

代码正确后继续使用错误旧metrics会让整改没有实际效果；全库粗暴清空又浪费资源并破坏审计。

**实施步骤**

1. 以metric/recipe/implementation版本查受影响evaluation/value集合，沿血缘传播到health/admission/graph/feature/model。
2. 分INVALID、SUPERSEDED、REQUIRES_REEVALUATION和仍有效；尽量复用未变数据与正确中间值。
3. 新结果另建版本，生产指针只有验证后切换；已阅test只作故障重放不作fresh认证。

**必须落地的回归／验收**

1. 修回撤算法不必重算无关原始factor值，却必须重算相关风险卡。
2. 改参数桥需重算value及其全部下游。
3. 迁移dry-run列表与实际更新可对账。

**依赖**：[V-24](#issue-v-24)、[V-18](#issue-v-18)。  **需求映射**：R03、R08、R14、R16。

**关闭方式**：若实际代码已满足，提交对应公开路径测试与证据，标为已覆盖；若需修改，提交权威实现及调用者变更、旧制品影响和测试结果；环境不足保留具体BLOCKED项，不以文档或元数据声明关闭。

---

<a id="issue-ops-09"></a>
#### OPS-09 · 离线回放必须在无Agent、无实时API下重现生产特征

**优先级**：P1　**证据分类**：E（待实仓核查）　**Owner**：FE/FP/DA/FA

**现有定位范围**：`quant_platform/app`；`data_access`；`factor_assets/lifecycle`；`factor_assets/registry`；`quant_evaluator/runtime`；`factor_optimizer/factor_optimizer/contracts`。

**需要排查的风险与为什么要改**

DSL之外隐藏在Notebook、环境变量、在线服务或Agent临场建议的参数，会使生产结果不可重复。

**实施步骤**

1. 冻结所有必需数据snapshot、state、registry/依赖版本、numeric policy及ExecutionSpec。
2. 提供现有CLI或API的replay模式，禁止访问未声明实时源和重新搜索。
3. 环境可不相同硬件，但数值容差、状态和来源必须可比较；无法还原则明确缺哪些制品。

**必须落地的回归／验收**

1. 断开Agent和市场实时API后仍能回放批准快照。
2. 修改环境默认参数不改变已冻结计划。
3. 缺state时拒绝，不从全历史重新fit悄悄替代。

**依赖**：[V-18](#issue-v-18)、[V-19](#issue-v-19)、[RCP-10](#issue-rcp-10)。  **需求映射**：R14、R15。

**关闭方式**：若实际代码已满足，提交对应公开路径测试与证据，标为已覆盖；若需修改，提交权威实现及调用者变更、旧制品影响和测试结果；环境不足保留具体BLOCKED项，不以文档或元数据声明关闭。

---

<a id="issue-ops-10"></a>
#### OPS-10 · 错误状态、等待成熟、维修失败与淘汰必须有不同动作

**优先级**：P1　**证据分类**：E（待实仓核查）　**Owner**：Platform/FO/FA

**现有定位范围**：`quant_platform/app`；`data_access`；`factor_assets/lifecycle`；`factor_assets/registry`；`quant_evaluator/runtime`；`factor_optimizer/factor_optimizer/contracts`。

**需要排查的风险与为什么要改**

FAILED一个状态无法区分数据等待、参数非法、预算不足、不可修复噪声与实现Bug，清理和重试会冲突。

**实施步骤**

1. 统一错误reason-code与retryability/action映射；保存原cause，平台不覆盖为统一CAPABILITY。
2. WAITING_LABEL、DATA_PENDING、BUDGET_STOP、INVALID_SPEC、IMPLEMENTATION_ERROR、REJECTED_QUALITY分别定义转换。
3. 状态变更发持久化事件，报告、UI、GC和seen index消费同一状态源。

**必须落地的回归／验收**

1. 未成熟标签不会被质量判D或删值。
2. 非法DSL终态后不无限重试。
3. 实现bug修复可重新评估但统计历史仍保留。

**依赖**：[PL-02](#issue-pl-02)、[PL-03](#issue-pl-03)、[V-20](#issue-v-20)。  **需求映射**：R03、R09、R14、R16。

**关闭方式**：若实际代码已满足，提交对应公开路径测试与证据，标为已覆盖；若需修改，提交权威实现及调用者变更、旧制品影响和测试结果；环境不足保留具体BLOCKED项，不以文档或元数据声明关闭。

---

### 12.8 QAT：独立金标、属性、设备与关闭证据专项

<a id="issue-qat-01"></a>
#### QAT-01 · 属性测试需要覆盖因子、证券、时间轴与掩码变换

**优先级**：P1　**证据分类**：E（待实仓核查）　**Owner**：QA/全模块

**现有定位范围**：`integration_tests`；`quant_evaluator/tests`；`factor_preprocess/tests`；`factor_optimizer/tests`；`factor_assets/tests`；`quant_platform/tests`；`jobs`。

**需要排查的风险与为什么要改**

仅少数happy-path固定样本容易漏N=F、ties和不同缺失模式；需用可复现属性样本持续找反例。

**实施步骤**

1. 对核心metric/transform建立排列等变、masked poison不变、batch decomposition等metamorphic tests。
2. 时间序列不能随意打乱后要求不变，应测试声明可合法的分块/重放性质。
3. 失败自动保存最小输入fixture、seed、版本和期望合同。

**必须落地的回归／验收**

1. 因子和证券置换在正确坐标下结果相应置换。
2. 未来输入修改不改变历史输出。
3. 无关候选加入不改变已有候选独立指标。

**依赖**：[V-23](#issue-v-23)、[QA-01](#issue-qa-01)。  **需求映射**：R08、R14、R15。

**关闭方式**：若实际代码已满足，提交对应公开路径测试与证据，标为已覆盖；若需修改，提交权威实现及调用者变更、旧制品影响和测试结果；环境不足保留具体BLOCKED项，不以文档或元数据声明关闭。

---

<a id="issue-qat-02"></a>
#### QAT-02 · 金标必须独立于待测实现，CPU/GPU不能一起算错

**优先级**：P0　**证据分类**：E（待实仓核查）　**Owner**：QA/QE/FE

**现有定位范围**：`integration_tests`；`quant_evaluator/tests`；`factor_preprocess/tests`；`factor_optimizer/tests`；`factor_assets/tests`；`quant_platform/tests`；`jobs`。

**需要排查的风险与为什么要改**

将CPU实现复制到GPU再互相对拍，只能证明一致，不能证明时间、资本和风险定义正确。

**实施步骤**

1. 先建立手算小价格/成交/成本账本，明确每时点持仓现金和NAV。
2. 统计金标使用独立公式或有固定版本的一手参考实现，不能调用待测函数生成expected。
3. 每个已知缺陷先建立intended-contract红测试，再修改；保留旧错误复现只作为审计演示。

**必须落地的回归／验收**

1. 入场前跳涨金标PnL=0。
2. 首笔亏损回撤与破产风险显式。
3. 负换手和空gate在全部公开入口被拒。

**依赖**：[QE-11](#issue-qe-11)、[QE-23](#issue-qe-23)、[FA-01](#issue-fa-01)、[V-23](#issue-v-23)。  **需求映射**：R07、R14、R15。

**关闭方式**：若实际代码已满足，提交对应公开路径测试与证据，标为已覆盖；若需修改，提交权威实现及调用者变更、旧制品影响和测试结果；环境不足保留具体BLOCKED项，不以文档或元数据声明关闭。

---

<a id="issue-qat-03"></a>
#### QAT-03 · 没有真实GPU时只能验证协议，不能签发硬件parity

**优先级**：P1　**证据分类**：E（待实仓核查）　**Owner**：QA/QE/FE

**现有定位范围**：`integration_tests`；`quant_evaluator/tests`；`factor_preprocess/tests`；`factor_optimizer/tests`；`factor_assets/tests`；`quant_platform/tests`；`jobs`。

**需要排查的风险与为什么要改**

NumPy模拟CuPy表达式或monkeypatch设备调用不能证实GPU排序、数值稳定性、显存或异步行为。

**实施步骤**

1. 测试报告区分reference unit、GPU-contract simulation和real CUDA integration。
2. production GPU能力只由实际设备日志和对应测试产生，required缺设备=BLOCKED。
3. mock用于异常注入，但不得更新真实性能/认证格。

**必须落地的回归／验收**

1. CI没有CUDA时最终报告不显示GPU passed。
2. 真实设备记录型号/driver/runtime/dtype及峰值显存。
3. 模拟OOM测试与实际OOM基准分栏。

**依赖**：[QE-04](#issue-qe-04)、[V-23](#issue-v-23)、[OPS-04](#issue-ops-04)。  **需求映射**：R08、R14。

**关闭方式**：若实际代码已满足，提交对应公开路径测试与证据，标为已覆盖；若需修改，提交权威实现及调用者变更、旧制品影响和测试结果；环境不足保留具体BLOCKED项，不以文档或元数据声明关闭。

---

<a id="issue-qat-04"></a>
#### QAT-04 · 变异测试证明守门真正保护，而不是写了断言名

**优先级**：P1　**证据分类**：E（待实仓核查）　**Owner**：QA/FA/FO/QE

**现有定位范围**：`integration_tests`；`quant_evaluator/tests`；`factor_preprocess/tests`；`factor_optimizer/tests`；`factor_assets/tests`；`quant_platform/tests`；`jobs`。

**需要排查的风险与为什么要改**

测试名叫no_leakage或fail_closed不意味着校验生效；若移除校验测试仍过，覆盖没有价值。

**实施步骤**

1. 对空gate、wrong ref、split guard、mask staging、parameter mapping做定向mutation。
2. 删除/反转关键判断时对应测试必须失败，记录存活mutation并补测试。
3. 不要为了提高mutation百分比修改无关代码，优先金融正确性和准入边界。

**必须落地的回归／验收**

1. 把all gates检查改成True至少一个公开集成测试失败。
2. 删max_lag映射会被非默认参数测试捕获。
3. 删GPU mask传递会被poison测试捕获。

**依赖**：[QA-01](#issue-qa-01)、[FA-01](#issue-fa-01)、[FP-03](#issue-fp-03)。  **需求映射**：R03、R14、R15。

**关闭方式**：若实际代码已满足，提交对应公开路径测试与证据，标为已覆盖；若需修改，提交权威实现及调用者变更、旧制品影响和测试结果；环境不足保留具体BLOCKED项，不以文档或元数据声明关闭。

---

<a id="issue-qat-05"></a>
#### QAT-05 · 序列化与内容身份fuzz要覆盖错误值和跨对象嫁接

**优先级**：P1　**证据分类**：E（待实仓核查）　**Owner**：QA/全模块

**现有定位范围**：`integration_tests`；`quant_evaluator/tests`；`factor_preprocess/tests`；`factor_optimizer/tests`；`factor_assets/tests`；`quant_platform/tests`；`jobs`。

**需要排查的风险与为什么要改**

bool("false")、NaN、重复key和未知enum常穿过弱类型边界；hash字符串存在也不证明内容可信。

**实施步骤**

1. 对所有跨包artifact/ref JSON做严格类型fuzz、roundtrip及篡改测试。
2. 验证child context与parent一致；禁止自报admissible、任意implementation hash、旧policy冒用。
3. 数据结构解析不执行传入DSL外任意代码；诊断内容作为数据而非Agent指令。

**必须落地的回归／验收**

1. "false"、1、空string成熟度按合同拒绝。
2. 换一个factor/ref保持高分也拒绝。
3. 改一个字节或state参数使内容hash失配。

**依赖**：[FA-02](#issue-fa-02)、[FA-07](#issue-fa-07)、[PL-05](#issue-pl-05)。  **需求映射**：R14、R15、R17。

**关闭方式**：若实际代码已满足，提交对应公开路径测试与证据，标为已覆盖；若需修改，提交权威实现及调用者变更、旧制品影响和测试结果；环境不足保留具体BLOCKED项，不以文档或元数据声明关闭。

---

<a id="issue-qat-06"></a>
#### QAT-06 · 数据差分与增量回放要包含晚到、修订、停牌和断点

**优先级**：P1　**证据分类**：E（待实仓核查）　**Owner**：QA/DA/FE

**现有定位范围**：`integration_tests`；`quant_evaluator/tests`；`factor_preprocess/tests`；`factor_optimizer/tests`；`factor_assets/tests`；`quant_platform/tests`；`jobs`。

**需要排查的风险与为什么要改**

只在干净连续数据上验证incremental会漏公告修订和状态warmup；持有期跨chunk也是高风险边界。

**实施步骤**

1. 保存小型合法PIT数据包，模拟arrival与knowledge时间而非只传最终表。
2. 对同一final snapshot执行全量、每日、随机chunk、断点重启和局部重算。
3. 比较值、mask、availability、state hash、label成熟度及portfolio episodes，不只比均值。

**必须落地的回归／验收**

1. 历史修订后的局部重算等价全量。
2. H期持仓跨chunk收益无重复/遗漏。
3. 新增退市证券不会污染其他state。

**依赖**：[DTA-10](#issue-dta-10)、[V-07](#issue-v-07)、[V-19](#issue-v-19)。  **需求映射**：R07、R14、R15。

**关闭方式**：若实际代码已满足，提交对应公开路径测试与证据，标为已覆盖；若需修改，提交权威实现及调用者变更、旧制品影响和测试结果；环境不足保留具体BLOCKED项，不以文档或元数据声明关闭。

---

<a id="issue-qat-07"></a>
#### QAT-07 · 独立wheel与monorepo必须使用同一有效实现

**优先级**：P1　**证据分类**：E（待实仓核查）　**Owner**：QA/全模块

**现有定位范围**：`integration_tests`；`quant_evaluator/tests`；`factor_preprocess/tests`；`factor_optimizer/tests`；`factor_assets/tests`；`quant_platform/tests`；`jobs`。

**需要排查的风险与为什么要改**

PYTHONPATH、同名顶层包、可编辑安装和旧安装wheel可能让测试import到不是待改文件的代码。

**实施步骤**

1. 按各pyproject建立独立安装与组合安装测试矩阵，记录实际module.__file__及版本。
2. 检查新旧路径、import alias、entry points及可选extras；缺依赖不能静默fallback认证。
3. 清理同作用域重复定义与死入口前先追调用，保持必要兼容adapter。

**必须落地的回归／验收**

1. 独立FP wheel研究fallback明确，生产组合必须真实FE。
2. monorepo测试日志显示当前checkout源码路径。
3. 同名测试不会被覆盖导致collector漏项。

**依赖**：[V-01](#issue-v-01)、[QA-01](#issue-qa-01)、[FP-04](#issue-fp-04)。  **需求映射**：R14、R15。

**关闭方式**：若实际代码已满足，提交对应公开路径测试与证据，标为已覆盖；若需修改，提交权威实现及调用者变更、旧制品影响和测试结果；环境不足保留具体BLOCKED项，不以文档或元数据声明关闭。

---

<a id="issue-qat-08"></a>
#### QAT-08 · 任务关闭必须同时覆盖需求、入口和历史结果迁移

**优先级**：P1　**证据分类**：E（待实仓核查）　**Owner**：总协调/QA

**现有定位范围**：`integration_tests`；`quant_evaluator/tests`；`factor_preprocess/tests`；`factor_optimizer/tests`；`factor_assets/tests`；`quant_platform/tests`；`jobs`。

**需要排查的风险与为什么要改**

177个编号的机械勾选仍可能只改局部函数；原始需求和生产依赖完整性才是交付目标。

**实施步骤**

1. 建立issue→requirement→public entry→regression→integration→migration映射，closed缺证据自动报错。
2. 父任务只有所有必需子项verified或有可验证已覆盖结论才能关闭；blocked不当done。
3. 最终附git diff/test logs/真实能力/影响制品/回滚和新增问题，禁止输出单一句all fixed。

**必须落地的回归／验收**

1. 一个公开旁路仍绕过gate时相关需求不得完成。
2. GPU blocked时CPU可单独验证但不整体认证。
3. 所有V1 AUD和V2 IDs都有保留映射且新增E项不冒充已确认bug。

**依赖**：[V-24](#issue-v-24)、[V-01](#issue-v-01)、[QA-02](#issue-qa-02)。  **需求映射**：R03、R14、R15、R16。

**关闭方式**：若实际代码已满足，提交对应公开路径测试与证据，标为已覆盖；若需修改，提交权威实现及调用者变更、旧制品影响和测试结果；环境不足保留具体BLOCKED项，不以文档或元数据声明关闭。

---


<a id="golden"></a>
## 13. 独立金标、属性测试与端到端验收

以下 **40 组金标／反例场景是开发 AI 必须实现的验收规格，不是本轮已经执行的测试日志**。用真实仓库函数、公开入口和小型可追溯 fixture 承载；多个场景可以参数化，但不能只用复制旧算法的脚本证明旧算法“按预期运行”。每个测试保存输入、独立 expected、调用入口和算法版本。

### 13.1 避免把验收规范本身写错

本版优先级为：**明确金融/时间合同 → 独立手算金标 → 同义后端对拍 → 旧实现兼容**。兼容旧接口不是保留错误数值的理由。

H 持有期统一按第4节明确的价格区间解释。H=1为入场到下一个退出价格的一段收益，不能一概等于同日进出；旧版同日场景是识别错误时钟的反例，不是要求所有H1收益为0。只有真正相同成交价且没有持有区间的no-op，才应零毛收益。

f与-f的PnL反号测试仅在对称分桶、相同股票池、对称多空权重、无费用及相同可执行约束时成立。真实借券、单边不可成交或不对称费用下，应分别核算而非强求反号。“提高费用不提高净收益”也仅针对**同一冻结成交序列**；优化器重新改变持仓时比较的是不同策略，不是该算术性质。

OLS/WLS残差正交要求同一有效样本、相同权重、明确截距和求解政策；ridge或Huber不应被迫通过OLS精确正交测试。声明“最终中性”必须对最终输出或实际组合重新验证。

PIT/前缀不变测试针对因子和决策输入；前向收益标签依法需要未来数据。禁止把合法标签未来窗口当成因子前视，也禁止为了测试变绿对所有因子再盲加一次lag。

### 13.2 测试分层

L0为纯数学与schema；L1为同包公开API；L2为adapter与registry真实参数；L3为四库+FE/DA完整合成链；L4为设备、实际数据快照与部署边界。一个L0通过不能替代L2/L3。GPU没有设备或依赖时，CPU测试照常执行，但该L4项为BLOCKED而不是PASS。

对核心guard增加mutation test：临时移除mask、校验、费用、gate覆盖或轴比较后，相应测试必须失败。若错误实现仍能通过测试，先修测试再签能力。

### 13.3 四十组必须落实的金标／反例

#### GOLD-01 · 逐因子独立性与正反向对照

**输入／前提**：同一合法股票池、同一收益、对称且不相交的多空分桶，f及-f；无费用、无借券差异、无方向再选择。

**独立预期**：两列各自保留；在上述对称前提下PnL互为相反数。添加无关第三列、交换F顺序不改变按ID解析的原两列。

**实际调用范围**：QE公开入口→FO适配→FA证据；不要在不对称交易约束下强行要求反号。

**关联整改**：[QE-01](#issue-qe-01)、[FO-01](#issue-fo-01)、[QAT-01](#issue-qat-01)。

#### GOLD-02 · 相同均值不等于相同分层

**输入／前提**：两个Q层剖面均值相同，一个U型、一个单调型；另含T×F日IC。

**独立预期**：序列、向量和标量分别保存且形状/轴可还原；不同剖面不能被相同均值替换。

**实际调用范围**：QE请求→序列化→报告及健康卡。

**关联整改**：[QE-02](#issue-qe-02)、[QE-03](#issue-qe-03)。

#### GOLD-03 · 对称形状金标

**输入／前提**：用确定性分位u构造收益u、-u、(u-.5)^2、-(u-.5)^2；再加低噪声跨窗样本。

**独立预期**：形状类型分别为正单调、负单调、U、倒U；方向按训练冻结。U及单调的质量分支不互相扣分；对称U的mirror asymmetry为0。

**实际调用范围**：QE形状→FA类型→FO候选；确定性样本用于函数金标，不用于伪造经济显著性。

**关联整改**：[QE-33](#issue-qe-33)、[QE-34](#issue-qe-34)、[FA-05](#issue-fa-05)、[RCP-07](#issue-rcp-07)。

#### GOLD-04 · 尾部断崖与过度拟合负控

**输入／前提**：20层均值前17层递增、最后3层下降；同时放入纯噪声因子。

**独立预期**：能保留全部层值/人数并标出尾部异常；修复只在训练内确定cutoff，噪声不得无限尝试直到过关。

**实际调用范围**：QE20层→FO低自由度修复→共同validation。

**关联整改**：[V-04](#issue-v-04)、[V-05](#issue-v-05)、[RCP-08](#issue-rcp-08)。

#### GOLD-05 · 二元、常数与严重ties

**输入／前提**：2000个资产但只有2个不同因子值；常数截面；同值跨分位边界。

**独立预期**：实际桶数受distinct levels和人数约束；不按股票编号伪造20层；多空桶不交；不可计算返回原因。

**实际调用范围**：统一quantile policy、CPU/GPU及probe。

**关联整改**：[QE-20](#issue-qe-20)、[QE-35](#issue-qe-35)。

#### GOLD-06 · 2D与3D mask以及有限但无效极值

**输入／前提**：合法(T,N)共享mask与显式广播(T,N,F)；masked位置填入极大有限数。

**独立预期**：两个mask等价；改masked值不影响指标、分桶和准入；错轴、非bool掩码按合同拒绝。

**实际调用范围**：CPU/GPU dispatch前及FO返回。

**关联整改**：[QE-04](#issue-qe-04)、[QE-19](#issue-qe-19)、[DTA-12](#issue-dta-12)。

#### GOLD-07 · 形状相同但轴不同

**输入／前提**：N=F时故意交换证券轴与因子轴；日期同长度但不同；证券顺序反向。

**独立预期**：仅shape合法不能通过；显式alignment可按批准规则恢复，自动按位置猜测禁止。

**实际调用范围**：DA→FE→FP→QE所有边界。

**关联整改**：[QE-29](#issue-qe-29)、[FP-01](#issue-fp-01)、[DTA-12](#issue-dta-12)。

#### GOLD-08 · 原始证券键和同键冲突

**输入／前提**：整数键、带前导零字符串、跨交易所同ticker；同(date,instrument)值1与9。

**独立预期**：身份无损或明确拒绝；不能全NaN；冲突不取first；改变输入行顺序不改变合法输出或错误类别。

**实际调用范围**：FE适配真实registry调用，不在测试中提前修正证券键。

**关联整改**：[FP-01](#issue-fp-01)、[FP-02](#issue-fp-02)、[DTA-01](#issue-dta-01)、[DTA-09](#issue-dta-09)。

#### GOLD-09 · 财报发布、供应商晚到及修订

**输入／前提**：报告期3月31日、首次可用4月30日、修订5月15日、另一来源5月2日才接收。

**独立预期**：4月29日不可见；各决策只读取其可见版本。旧snapshot复跑旧结果；新snapshot不把修订回写为当时已知。

**实际调用范围**：DA as-of→FE→taxonomy/评估；日期只作fixture，无真实公司事实。

**关联整改**：[DTA-02](#issue-dta-02)、[V-12](#issue-v-12)。

#### GOLD-10 · 收盘后信号与入场前跳变

**输入／前提**：P0=100、P1=110、P2=110，信号在0日收盘后可用，在P1入场，在P2退出。

**独立预期**：毛持有收益为0，不获得P0→P1的10%；费用另扣。H=1按一个P1→P2价格区间解释。

**实际调用范围**：CPU/GPU cohort和真实执行适配共用手算价格账本。

**关联整改**：[QE-23](#issue-qe-23)、[QE-30](#issue-qe-30)、[QAT-02](#issue-qat-02)。

#### GOLD-11 · H1/H5/H20及末端未成熟

**输入／前提**：明确entry=e、exit=e+H；同时提供H期forward label和独立daily holding returns。

**独立预期**：只累积r[e+1:e+H+1]；类型层拒绝用H期标签替代日收益。未到exit的样本按maturity/pending处理，不默认收益0或隐式强平。

**实际调用范围**：标签→组合→sealed窗口及多期曲面。

**关联整改**：[QE-09](#issue-qe-09)、[QE-24](#issue-qe-24)、[DTA-11](#issue-dta-11)。

#### GOLD-12 · 初始净值与第一笔亏损

**输入／前提**：returns=[-.1,0]，初始NAV=1；另测[.1,-.1]。

**独立预期**：第一组最大回撤=.1、峰为t0且两期未恢复；第二组峰1.1、谷.99、回撤=.1。峰谷索引为真实时间标识。

**实际调用范围**：所有NAV/回撤/Calmar/报告入口。

**关联整改**：[QE-11](#issue-qe-11)。

#### GOLD-13 · 净值归零与杠杆负净值

**输入／前提**：[.1,-1]、[-1]、[-1.2,-2]。

**独立预期**：归零记录100%资本损失和破产状态；负净值采用单独账户语义或拒绝。不得通过NaN屏蔽灾难，也不得因负数再乘负数生成假恢复。

**实际调用范围**：NAV→risk artifacts→健康硬门→发布。

**关联整改**：[QE-12](#issue-qe-12)。

#### GOLD-14 · 缺报价不压缩时间与回撤删失

**输入／前提**：同一真实日期轴上存在缺价格；两段分别恢复的回撤和一段末尾尚未恢复。

**独立预期**：事件分段正确，不跨过前一恢复点找全局谷值；未恢复有censored标志，不能当完成恢复平均值。交易日跨度和有效观测数分开。

**实际调用范围**：underwater核心→图表→等级。

**关联整改**：[QE-14](#issue-qe-14)、[QE-15](#issue-qe-15)、[QE-26](#issue-qe-26)。

#### GOLD-15 · 空输入及样本阈值

**输入／前提**：T=0、T=1、全NaN、仅一只有效证券；注册阈值上下各一个样本。

**独立预期**：返回类型和轴稳定；不可计算的原因明确，不返回其他指标的三元组；不从NaN自动填高分。

**实际调用范围**：函数、公开evaluate、adapter三层。

**关联整改**：[QE-13](#issue-qe-13)、[QE-27](#issue-qe-27)。

#### GOLD-16 · 四种换手与价格漂移

**输入／前提**：顶层成员相同但内部rank变化；目标权重不变但持仓价格漂移；新增与退出证券。

**独立预期**：秩proxy、成员换手、目标对漂移权重的成交换手和实际成交分别定义；不可互相当成本输入。

**实际调用范围**：QE turnover→执行持仓与成本。

**关联整改**：[QE-07](#issue-qe-07)、[V-13](#issue-v-13)。

#### GOLD-17 · 费用账本和同日退化交易

**输入／前提**：固定成交序列、无价格变化、两次明确成交；负turnover；同日可选no-trade政策。

**独立预期**：单边费率按每笔真实名义额扣；两笔成交两笔费；no-trade无成交无费用。负turnover拒绝。在相同成交序列下提高费用不增加net PnL。

**实际调用范围**：CPU/GPU probe适配与执行域，不把策略改变后的路径混作该属性测试。

**关联整改**：[QE-22](#issue-qe-22)、[QE-24](#issue-qe-24)、[OPS-04](#issue-ops-04)。

#### GOLD-18 · 做空不可得、部分成交与不能卖出

**输入／前提**：仅多头可成交、空头无借券；卖单无法成交；跨cohort相抵。

**独立预期**：实际持仓和现金反映未成交；不自报美元中性；研究short不自动等于可执行。净额成交规则与费用账本一致。

**实际调用范围**：DA侧别资格→执行域→QE证据。

**关联整改**：[QE-25](#issue-qe-25)、[DTA-05](#issue-dta-05)。

#### GOLD-19 · 拆分与现金分红

**输入／前提**：2拆1、现金分红及除息价下降；用显式数量和现金小账本。

**独立预期**：拆分本身无凭空盈亏，股数与价格一致；总回报包括现金，因子adjusted price不冒充成交报价。

**实际调用范围**：DA企业行为→执行资产→QE NAV。

**关联整改**：[DTA-03](#issue-dta-03)。

#### GOLD-20 · 共同有效样本上的Spearman

**输入／前提**：两个因子有不同missing pattern；label极值在一列被mask但另一列有效；大量ties。

**独立预期**：x和y按各自共同有效集合取秩；与独立手工金标一致。只有mask相同才能共享label rank。

**实际调用范围**：NumPy reference、实际CPU/GPU内核、公开接口。

**关联整改**：[STA-01](#issue-sta-01)、[QE-28](#issue-qe-28)。

#### GOLD-21 · 零方差ICIR、HAC和样本单位

**输入／前提**：恒定正IC、恒定零IC、自相关IC；在保持IC序列不变前提下增加股票列。

**独立预期**：零方差不产生5e10级IR；稳健t统计来自IC时间序列样本，不能因股票格数增加便制造精度。

**实际调用范围**：QE统计→样本量artifact→FA evidence grade。

**关联整改**：[QE-06](#issue-qe-06)、[STA-02](#issue-sta-02)。

#### GOLD-22 · 偏度、Sortino、Calmar和自然月定义

**输入／前提**：不对称小样本；少量负收益、多量正收益；跨两个月且交易日数不等。

**独立预期**：分别验证已声明bias、downside分母、复合/算术年化及自然月/21d。不同定义另名，不强求数值相等。

**实际调用范围**：定义版本→golden→所有显示与等级消费者。

**关联整改**：[QE-16](#issue-qe-16)、[QE-17](#issue-qe-17)、[QE-18](#issue-qe-18)、[STA-11](#issue-sta-11)。

#### GOLD-23 · 配对块重采样和候选差值

**输入／前提**：两候选面对相同市场冲击的收益差；重叠窗口；固定随机种子。

**独立预期**：同一抽样索引同时应用两候选；保存block length、有效B、seed和实际事件定义；不能把稳定性频率称U型真实概率。

**实际调用范围**：QE统计证据→FO非劣性选择。

**关联整改**：[STA-03](#issue-sta-03)、[QE-34](#issue-qe-34)。

#### GOLD-24 · 衰减、保持率和事后方向

**输入／前提**：训练接近0但验证稍正；训练正验证负；IC自相关强但H20无预测；测试负相关。

**独立预期**：近零retention不强算；分开horizon predictive IC与IC序列ACF。测试不得事后取abs改方向；记录性能衰减而非重新优化test。

**实际调用范围**：QE→FA→FO，所有统计含方向和split。

**关联整改**：[QE-32](#issue-qe-32)、[STA-07](#issue-sta-07)、[STA-10](#issue-sta-10)。

#### GOLD-25 · 参数绑定必须产生真实执行差异

**输入／前提**：通过FP registry传max_lag=1/5；rank pct=False和不同ties；多余参数；位置与关键字两调用。

**独立预期**：request→canonical→effective参数完整可查；预期不同值确实不同。未知参数拒绝；测试不得手工改为FE参数名掩盖adapter。

**实际调用范围**：完整recipe→get_execution→FE→materialization。

**关联整改**：[FP-03](#issue-fp-03)、[SRH-01](#issue-srh-01)、[QA-01](#issue-qa-01)。

#### GOLD-26 · 辅助通道不会替换主因子

**输入／前提**：带NaN的主因子，先产生missing indicator和age，再填充、缩放和中性化。

**独立预期**：主value仍是数值信号，indicator为旁路通道；填充后仍保留原始缺失和age，不用indicator继续当alpha。

**实际调用范围**：FP typed DAG/FeatureBundle→model manifest。

**关联整改**：[RCP-01](#issue-rcp-01)、[DTA-08](#issue-dta-08)。

#### GOLD-27 · 局部幂等与末端性质

**输入／前提**：EMA(h3)→EMA(h10)；zscore→非线性→zscore；子表达式rank；连续同合同zscore。

**独立预期**：前两种不能删；子表达式出现rank不等于根已rank。连续缩放只在轴、mask、权重、numeric policy和输入相同且等价可证时折叠。

**实际调用范围**：FE canonical DAG→FP eligibility→DSL回显。

**关联整改**：[FP-07](#issue-fp-07)、[FP-08](#issue-fp-08)、[RCP-02](#issue-rcp-02)。

#### GOLD-28 · 中性化solver与后置变化

**输入／前提**：有截距满秩X、明确W；随后rank、clip或跨期平滑；另测ridge与秩亏。

**独立预期**：OLS/WLS按其合同满足XᵀWe≈0；ridge/Huber不假称同一正交性质。后续非线性变化重新测最终暴露；秩亏政策确定。

**实际调用范围**：FE残差→QE exposure→FO非劣性→FA标签。

**关联整改**：[FP-06](#issue-fp-06)、[V-06](#issue-v-06)、[RCP-04](#issue-rcp-04)、[RCP-05](#issue-rcp-05)。

#### GOLD-29 · RAW白名单不能代替顺序认证

**输入／前提**：非法E→A→D配方但tag=RAW；合法例外但参数不合规。

**独立预期**：两者都拒绝；只有实际DAG、parameters和pre/postconditions匹配才认证。RAW必须实际为空/no-op，不看tag。

**实际调用范围**：生产recipe编译真实入口。

**关联整改**：[FP-09](#issue-fp-09)。

#### GOLD-30 · 因果前缀、事件时钟和重启

**输入／前提**：修改t之后的价格/公告；EMA全量、分块、逐日和中断恢复；新证券。

**独立预期**：历史<=t输出不变；状态回放与声明精度一致；事件级更新和每日观察级更新按不同合同，不能靠盲shift(1)通过。

**实际调用范围**：FE/FP causal路径→生产更新；labels的合法未来窗口另测。

**关联整改**：[V-07](#issue-v-07)、[RCP-06](#issue-rcp-06)、[OPS-09](#issue-ops-09)。

#### GOLD-31 · 只读对象与可重现哈希

**输入／前提**：改变原数组、嵌套policy dict、semantic_id.value；跨进程生成动态函数/相同配方。

**独立预期**：快照不能被别名写污染；hash来自实际内容、依赖和政策。生成器repr不得进入内容身份。相同执行配置不同运行ID可共享数学身份。

**实际调用范围**：所有contract→cache→evidence store。

**关联整改**：[QE-31](#issue-qe-31)、[FP-11](#issue-fp-11)、[FP-12](#issue-fp-12)、[FA-03](#issue-fa-03)。

#### GOLD-32 · 完整性硬门与证据嫁接

**输入／前提**：14维高分但空gates；缺1门、重复1门；引用其他因子/窗口/policy的高分；False门+True summary。

**独立预期**：全部拒绝或不可晋级；显示高分不能覆盖完整性失败；反序列化必须复算引用一致性与派生summary。

**实际调用范围**：FA health builder→AdmissionAuthority→平台各入口。

**关联整改**：[FA-01](#issue-fa-01)、[FA-02](#issue-fa-02)。

#### GOLD-33 · 缺失、低分和分用途准入

**输入／前提**：高分指标+一个可选NA；一个必需未运行；低线性IC但独立OOS非线性增量；字符串false。

**独立预期**：NA不扣为已测0；必需缺证据阻断；模型特征通道可依法使用其预声明替代证据；字符串false不能解析True；REJECT优先。

**实际调用范围**：QE status→FA policy→Platform。

**关联整改**：[FA-04](#issue-fa-04)、[FA-05](#issue-fa-05)、[FA-07](#issue-fa-07)、[PL-04](#issue-pl-04)、[V-08](#issue-v-08)。

#### GOLD-34 · 搜索状态、预算与封存权限

**输入／前提**：候选语法失败/重复/OOM；预算不足；运行中断并恢复；改名字或另开session；测试访问失败重试。

**独立预期**：实际费用与统计试验计数分别对账；恢复候选/RNG/预算可复现；改名不重开holdout。基础设施故障允许对同冻结请求作可审计重试，不允许重选方案。

**实际调用范围**：FO campaign/ledger/broker→QE可信请求。

**关联整改**：[FO-05](#issue-fo-05)、[V-11](#issue-v-11)、[SRH-06](#issue-srh-06)、[SRH-07](#issue-srh-07)、[SRH-08](#issue-srh-08)。

#### GOLD-35 · NaN前沿与多个互补版本

**输入／前提**：含NaN或缺必需目标的候选；预测强、高成本与预测稍弱、低成本候选；参数19/20近等价。

**独立预期**：缺证据不进入合格Pareto；允许有用途证据的少量互补版本，不强制单冠军或quota全收；高相关近等价优先代表。

**实际调用范围**：FO选择→FA准入及family cap。

**关联整改**：[FA-11](#issue-fa-11)、[V-16](#issue-v-16)、[SRH-03](#issue-srh-03)、[SRH-04](#issue-srh-04)。

#### GOLD-36 · 近重复、未知相关与错误改投

**输入／前提**：f/-f，多窗口部分相似，ANN未召回；与小A=.9、大B=.1且小簇受限。

**独立预期**：不把未召回当不相关，0与UNKNOWN分开；不能归B并记A的.9；最终affinity引用真实目标。

**实际调用范围**：QE pairwise→FA graph/ANN/incremental。

**关联整改**：[FA-08](#issue-fa-08)、[FA-09](#issue-fa-09)、[V-14](#issue-v-14)、[SIM-02](#issue-sim-02)、[SIM-08](#issue-sim-08)。

#### GOLD-37 · 新簇与聚合版本改变

**输入／前提**：旧簇中加入新因子，新增独立家族，簇分裂/合并，原列名未变但权重变化。

**独立预期**：可形成provisional新簇；旧版本不改写；输入函数变化触发model compatibility验证。研究catalog更新本身不强制线上重训。

**实际调用范围**：FA cluster lineage→feature set→model serving spec。

**关联整改**：[V-15](#issue-v-15)、[V-17](#issue-v-17)、[PL-06](#issue-pl-06)、[SIM-09](#issue-sim-09)、[MOD-06](#issue-mod-06)。

#### GOLD-38 · 一条坏候选与重试恢复

**输入／前提**：同批包含无效DSL、合法候选及第一次暂时失败候选；执行前/登记后进程退出。

**独立预期**：坏记录可在尚无factor hash时报告；合法候选不被整批中断；失败可重试不永久被去重；错误类别保持原retryability。

**实际调用范围**：Platform真实handler→job store→domain resolver。

**关联整改**：[PL-01](#issue-pl-01)、[PL-02](#issue-pl-02)、[PL-03](#issue-pl-03)、[OPS-01](#issue-ops-01)。

#### GOLD-39 · 发布、模型列及GC竞态

**输入／前提**：对象写入后未登记即崩溃；发布CAS冲突；待删节点突然被新feature引用；试验values无人引用。

**独立预期**：读取者不见半发布；ref bytes/hash/size匹配；受保护对象不删；无引用失败值实际删除且有receipt；trial/formula/evidence完整保留。

**实际调用范围**：DA publisher→FA ref graph→Platform outbox→GC。

**关联整改**：[PL-07](#issue-pl-07)、[PL-08](#issue-pl-08)、[V-20](#issue-v-20)、[OPS-05](#issue-ops-05)、[OPS-06](#issue-ops-06)、[OPS-07](#issue-ops-07)。

#### GOLD-40 · 算法修复后的历史失效与离线复现

**输入／前提**：修正QE-01或FP参数映射后，旧evaluation→health→cluster→feature链；另有不受影响对象。

**独立预期**：只失效受影响依赖，禁止原地改旧高分；旧test重放标审计用途不变新holdout。冻结合格recipe可脱离Agent及实时外部查询重放；缺GPU仍标未验收。

**实际调用范围**：全链迁移/rollback→真实能力矩阵。

**关联整改**：[V-18](#issue-v-18)、[V-19](#issue-v-19)、[V-23](#issue-v-23)、[V-24](#issue-v-24)、[OPS-08](#issue-ops-08)、[OPS-09](#issue-ops-09)。


### 13.4 八条端到端验收链

每条链至少使用一个真正可执行的DSL或现有解析后表达式，经过规范字段目录和数据快照，不用裸随机“评估分数”代替QE。没有私有数据时可用合成数据接入现有DA测试provider；测试provider只提供数据，不能替代FE/QE的数学。

| 场景 | 必经节点 | 成功与失败都要验证 |
|---|---|---|
| E2E-A：单调量价 | DSL→字段标签→落值→Q10/IC→评级→准入 | 方向在训练冻结，所有指标按同一因子身份保存；无效PIT即使高分也拒绝 |
| E2E-B：U／倒U | 原始低线性IC→形状救援→少量映射→父子复评 | 不能在廉价层误删；测试不参与选中心；单调型与U型走不同政策 |
| E2E-C：20层尾部塌陷 | Q10触发→Q20证据→流动性/暴露检查→有限饱和或hinge | 小样本/ties降级明确；不用测试集挑尾部cutoff |
| E2E-D：风险偏移 | raw→industry/size候选→最终暴露→非劣性 | 效用近似且暴露改善可优先中性；显著损害效用不强行覆盖raw |
| E2E-E：稀疏基本面 | 公告版本→PIT→有界填充+age旁路→模型特征 | 不把ffill后的coverage称原始覆盖；晚到/修订触发范围正确的增量重放 |
| E2E-F：纯噪声和失败 | 提案→预算初筛→拒绝/待重试→GC | 不靠无限尝试制造赢家；失败物化值真实删除，formula/trial历史完整 |
| E2E-G：相似族与新增簇 | 高相关复核→多代表亲和度→新族/歧义→簇版本 | 不把候选硬塞最大簇；旧模型列不被研究归簇更新偷偷改变 |
| E2E-H：生产固定更新 | 冻结recipe/fit/checkpoint→逐日计算→成熟样本监测 | 不调用Optimizer或Agent；中断恢复和全量一致；失败不翻转正式指针 |

每条链都输出可检索trace：`request_id / parent-trial / factor-definition / recipe / value / evaluation / health-policy / verdict / library / feature-version`。身份不存在时明确失败阶段，不能填一个看似合法的hash继续。

### 13.5 如何从现有代码开始改：两个执行骨架

下面是**设计伪代码，不是声称仓库已有这些函数名**。开发AI需要先定位已有同等对象并修正它们，不照抄新建平行框架。

```python
# 公开评价入口的目标顺序（映射到已有 QE planner/runtime）
def evaluate_request(request):
    validated = validate_all_contracts(request)       # 所有backend之前
    resolved = resolve_typed_inputs(validated)       # 保留真实轴/时钟/masks
    plan = compile_from_metric_registry(resolved)    # 绑定effective参数及依赖
    reserve = reserve_actual_budget(plan)
    try:
        artifacts = execute_dependency_plan(plan)    # 每个共享节点一次
        evidence = build_typed_evidence(artifacts)   # 不把T×Q×F压成通用均值
        verify_requested_outputs(evidence, plan)
        persist_immutable_evidence(evidence)
        return evidence
    finally:
        settle_usage_and_release(reserve)           # 失败/取消也结算实际资源
```

```python
# 优化器目标顺序（复用现有 SearchRunner/TrialLedger/FA adapter）
def process_factor(definition, campaign):
    baseline = materialize_and_evaluate_raw(definition, campaign.train_scope)
    profile = request_fa_taxonomy_health_and_diagnosis(baseline)
    candidates = propose_bounded_repairs(profile, campaign.budget)
    for candidate in candidates:
        recipe = compile_and_validate_existing_recipe(candidate)
        ledger_record_before_execution(recipe, campaign)
        evaluate_on_declared_train_validation(recipe, campaign)
    finalists = select_frozen_multiobjective_set(campaign) # 可为0、1或多个
    sealed_result = consume_authorized_holdout(finalists) # 不回流再调参
    verdicts = request_fa_admission(sealed_result)
    plan_publication_or_safe_gc(verdicts)                 # 副作用按权限处理
```

实际执行中，可能无需为每个研究候选消费sealed test。只有进入预声明确认阶段的冻结候选集合才使用holdout；研究archive与生产资格分离。基础设施失败重试只能重放同一冻结请求与权限，不能借重试换新候选。

### 13.6 运行命令与日志不得编造

在实际checkout先执行只读盘点命令，例如`git rev-parse HEAD`、`git status --short`、列举各`pyproject.toml`和CI配置；安装与pytest路径据真实环境确定。对每个独立包与组合安装先做test collection，再执行对应suite。不要把本文件的示例测试ID假装成已经存在的pytest函数。

每份结果区分：定义存在、测试已写、已运行、通过/失败、被跳过、依赖阻断、真实硬件运行。附软件版本、实际import文件路径、随机种子、数据snapshot、有效backend、计时区间和峰值资源。真实GPU测试必须同步计时并记录传输成本，不能把异步提交时间或模拟array结果当CUDA基准。


---

<a id="traceability"></a>
## 14. V1问题及原始需求的完整映射

### 14.1 V1九个AUD问题全部并入，不通过更名丢掉子问题

| V1编号 | 本版工作单 | 明确保留的子问题 |
|---|---|---|
| AUD-01 | [FA-01](#issue-fa-01)、[FA-02](#issue-fa-02) | 空序列all通过；缺门/重复门；PIT等证据引用非可信证明；跨因子/快照嫁接；完整性结果须从权威builder派生。 |
| AUD-02 | [FA-08](#issue-fa-08)、[FA-09](#issue-fa-09)、[SIM-08](#issue-sim-08) | MERGE_NEAREST实际上选最大簇；改投后沿用原高affinity；runner gap丢失；全量与增量小簇政策必须同源。 |
| AUD-03 | [FO-03](#issue-fo-03)、[FO-02](#issue-fo-02)、[V-23](#issue-v-23) | research_only及三项显式阻断必须保留，补可信切分QE集成、合法性链、封存证据升级后凭验收签能力。 |
| AUD-04 | [FP-07](#issue-fp-07)、[FP-08](#issue-fp-08)、[FP-11](#issue-fp-11)、[RCP-02](#issue-rcp-02) | 参数和输入DAG未入去重；隔着变换的第二次处理可能必需；z-score最终状态缺失；OLS/dual摘要不完整；FE/FP语义映射不能分叉。 |
| AUD-05 | [QE-33](#issue-qe-33)、[QE-34](#issue-qe-34)、[STA-03](#issue-sta-03) | Fisher-z缺逆变换；自包含窗口参考偏乐观；bootstrap秩相似频率不等于U成立概率；ties与重叠窗口需规范。 |
| AUD-06 | [FA-04](#issue-fa-04)、[STA-12](#issue-sta-12) | 部分缺失代0导致维度为0；全部缺失又为None；required/optional/NA/NOT_RUN状态应分离；聚合实现收口。 |
| AUD-07 | [FA-05](#issue-fa-05)、[V-04](#issue-v-04)、[RCP-07](#issue-rcp-07) | 单调与U机制不能共同要求高分；按形状类型评价证据与可交易表达；方向性尾部指标按训练冻结方向解释。 |
| AUD-08 | [FA-07](#issue-fa-07)、[FA-02](#issue-fa-02)、[PL-04](#issue-pl-04) | 裸IC和缺省成熟；字符串false解析True；0.02及0.7旧门与新健康政策分裂；soft duplicate提前REVIEW覆盖硬失败；追全部入口。 |
| AUD-09 | [FA-09](#issue-fa-09)、[FA-10](#issue-fa-10)、[SIM-02](#issue-sim-02)、[SIM-07](#issue-sim-07)、[SIM-08](#issue-sim-08) | 嵌入cosine不等于金融相关；已测低相关不等于UNKNOWN；ANN召回不是正式证明；单个近邻不能代表全簇。 |

### 14.2 V1其余设计内容的保留位置

V1不只有9个Bug，也包含数据模型与完整流程。下表防止“只合并编号、丢掉设计要求”。

| V1内容 | 本版位置 | 保留重点 |
|---|---|---|
| 模块边界与已存在能力 | 第3节；V-01、FA-06、QA-02 | 保留已有taxonomy、14维、分级、shape、自适应Q、增量簇；不重复建设。 |
| 分类和健康证据数据模型 | 第4–5节；DTA-06、FA-06、V-02 | 来源/机制/结构/频率/处理状态，signal与control分开，等级绑定上下文。 |
| 等级、旧数值与分用途硬门 | 第5节；V-03、STA-08、STA-12 | S+到D原表、ICIR尺度、方向、retention近零、required/optional；不是私募普适阈值。 |
| U/倒U、尾部、中性化、滤波 | 第6节；RCP-03–RCP-08、V-04–V-07 | 原父因子、左右子因子和重组分别评估；预算低自由度且禁test拟合。 |
| 模型表示和顺序 | 第6节；RCP-01、RCP-02、RCP-05、MOD-03、MOD-04 | 不是所有因子强制末尾zscore；实际输出性质优先于操作出现历史。 |
| 预算与多候选筛选 | 第7节；SRH-01–SRH-10 | raw基线、8–12候选/1–3版本为可调起点，不用满配额、不全排列。 |
| 去重、聚类、刷新与重训 | 第7节；SIM-01–SIM-10、MOD-05、MOD-06 | 0.99/0.995、0.95、0.7不同用途；ANN限定；研究归簇与线上输入函数分离。 |
| 计算复用与运行证据 | 第8节；QE-28、STA-01、OPS-02–OPS-04 | 同mask才能复用label秩；显存含中间量；基准包括转换、I/O和实际同步。 |
| 生产复现、失败值清理 | 第8节；V-18–V-21、OPS-05–OPS-10 | DSL+sidecar+状态+快照；安全物理GC但trial历史和受保护依赖保留。 |
| Agent边界及实施验收 | 第1、9、13、15节；V-22、QAT-01–QAT-08 | Agent提案不能覆盖硬门；真实代码及端到端证据关闭；有限权限下发布清理。 |
| 统计研究与工具参考 | 第15.7节 | 原一手研究链接全部保留，不能被引作项目阈值或盈利保证。 |

### 14.3 原始需求R01–R17逐项对账

每条需求同时列历史工作单与本版专项；同一需求必须有最终公开路径验收，不因其某个子功能存在就整体关闭。

#### R01 · 按字段自动多级分类：量价、财务、换手/流动性及混合

**领域权威**：FA taxonomy + DA field taxonomy + FE static analysis。

**继承工作单**：[FA-06](#issue-fa-06)、[PL-05](#issue-pl-05)、[V-02](#issue-v-02)。

**新增专项**：[DTA-02](#issue-dta-02)、[DTA-06](#issue-dta-06)、[RCP-10](#issue-rcp-10)。

**相关金标场景**：GOLD-09。最终以本需求全部必需路径的有效证据判断完成，不按链接数量评分。

#### R02 · 机制、结构、频率、更新类型与标签来源/置信度

**领域权威**：FA profiling；不把Agent建议当事实。

**继承工作单**：[FA-06](#issue-fa-06)、[V-02](#issue-v-02)。

**新增专项**：[DTA-06](#issue-dta-06)。

**相关金标场景**：。最终以本需求全部必需路径的有效证据判断完成，不按链接数量评分。

#### R03 · 逐指标、逐维度等级、最低标准与证据状态

**领域权威**：QE metric authority + FA versioned policy。

**继承工作单**：[QE-02](#issue-qe-02)、[QE-03](#issue-qe-03)、[QE-06](#issue-qe-06)、[QE-12](#issue-qe-12)、[QE-13](#issue-qe-13)、[QE-16](#issue-qe-16)、[QE-17](#issue-qe-17)、[QE-18](#issue-qe-18)、[QE-27](#issue-qe-27)、[QE-32](#issue-qe-32)、[QE-33](#issue-qe-33)、[FA-01](#issue-fa-01)、[FA-02](#issue-fa-02)、[FA-03](#issue-fa-03)、[FA-04](#issue-fa-04)、[FA-07](#issue-fa-07)、[FA-11](#issue-fa-11)、[QA-02](#issue-qa-02)、[V-03](#issue-v-03)、[V-05](#issue-v-05)、[V-21](#issue-v-21)、[V-24](#issue-v-24)。

**新增专项**：[DTA-04](#issue-dta-04)、[DTA-08](#issue-dta-08)、[DTA-11](#issue-dta-11)、[STA-01](#issue-sta-01)、[STA-02](#issue-sta-02)、[STA-05](#issue-sta-05)、[STA-06](#issue-sta-06)、[STA-07](#issue-sta-07)、[STA-08](#issue-sta-08)、[STA-09](#issue-sta-09)、[STA-10](#issue-sta-10)、[STA-11](#issue-sta-11)、[STA-12](#issue-sta-12)、[RCP-01](#issue-rcp-01)、[SRH-02](#issue-srh-02)、[SRH-03](#issue-srh-03)、[SIM-01](#issue-sim-01)、[SIM-02](#issue-sim-02)、[MOD-07](#issue-mod-07)、[MOD-08](#issue-mod-08)、[OPS-08](#issue-ops-08)、[OPS-10](#issue-ops-10)、[QAT-04](#issue-qat-04)、[QAT-08](#issue-qat-08)。

**相关金标场景**：GOLD-02、GOLD-03、GOLD-04、GOLD-11、GOLD-13、GOLD-15、GOLD-20、GOLD-21、GOLD-22、GOLD-24、GOLD-26、GOLD-31、GOLD-32、GOLD-33、GOLD-35、GOLD-36、GOLD-40。最终以本需求全部必需路径的有效证据判断完成，不按链接数量评分。

#### R04 · 10/20层、U/倒U、尾部断崖、分因子及复评

**领域权威**：QE typed shape artifact + FO repairs + FE DAG。

**继承工作单**：[QE-01](#issue-qe-01)、[QE-02](#issue-qe-02)、[QE-03](#issue-qe-03)、[QE-06](#issue-qe-06)、[QE-08](#issue-qe-08)、[QE-09](#issue-qe-09)、[QE-20](#issue-qe-20)、[QE-33](#issue-qe-33)、[QE-34](#issue-qe-34)、[QE-35](#issue-qe-35)、[FA-05](#issue-fa-05)、[V-04](#issue-v-04)、[V-05](#issue-v-05)、[V-17](#issue-v-17)。

**新增专项**：[STA-03](#issue-sta-03)、[RCP-03](#issue-rcp-03)、[RCP-07](#issue-rcp-07)、[RCP-08](#issue-rcp-08)、[SRH-02](#issue-srh-02)。

**相关金标场景**：GOLD-01、GOLD-02、GOLD-03、GOLD-04、GOLD-05、GOLD-11、GOLD-21、GOLD-23、GOLD-33、GOLD-37。最终以本需求全部必需路径的有效证据判断完成，不按链接数量评分。

#### R05 · 行业/市值/风格中性化及非劣性选择

**领域权威**：FE operators/FP recipe + QE exposure evidence。

**继承工作单**：[FP-06](#issue-fp-06)、[FP-08](#issue-fp-08)、[V-06](#issue-v-06)。

**新增专项**：[DTA-05](#issue-dta-05)、[RCP-04](#issue-rcp-04)、[RCP-05](#issue-rcp-05)、[SRH-02](#issue-srh-02)、[MOD-03](#issue-mod-03)。

**相关金标场景**：GOLD-18、GOLD-27、GOLD-28。最终以本需求全部必需路径的有效证据判断完成，不按链接数量评分。

#### R06 · 换手、因果滤波、真实交易成本与缺失状态

**领域权威**：QE turnover + FE causal operators。

**继承工作单**：[QE-07](#issue-qe-07)、[QE-22](#issue-qe-22)、[QE-25](#issue-qe-25)、[QE-32](#issue-qe-32)、[FP-03](#issue-fp-03)、[V-07](#issue-v-07)、[V-13](#issue-v-13)。

**新增专项**：[DTA-08](#issue-dta-08)、[STA-10](#issue-sta-10)、[RCP-06](#issue-rcp-06)、[SRH-02](#issue-srh-02)。

**相关金标场景**：GOLD-16、GOLD-17、GOLD-18、GOLD-24、GOLD-25、GOLD-26、GOLD-30。最终以本需求全部必需路径的有效证据判断完成，不按链接数量评分。

#### R07 · 多空/多头超额、净Sharpe、回撤、水下期、尾部与容量

**领域权威**：QE consumes correctly timed portfolio artifacts。

**继承工作单**：[QE-01](#issue-qe-01)、[QE-09](#issue-qe-09)、[QE-11](#issue-qe-11)、[QE-12](#issue-qe-12)、[QE-14](#issue-qe-14)、[QE-15](#issue-qe-15)、[QE-16](#issue-qe-16)、[QE-17](#issue-qe-17)、[QE-18](#issue-qe-18)、[QE-21](#issue-qe-21)、[QE-22](#issue-qe-22)、[QE-23](#issue-qe-23)、[QE-24](#issue-qe-24)、[QE-25](#issue-qe-25)、[QE-26](#issue-qe-26)、[V-13](#issue-v-13)。

**新增专项**：[DTA-03](#issue-dta-03)、[DTA-04](#issue-dta-04)、[DTA-05](#issue-dta-05)、[DTA-07](#issue-dta-07)、[STA-11](#issue-sta-11)、[MOD-07](#issue-mod-07)、[QAT-02](#issue-qat-02)、[QAT-06](#issue-qat-06)。

**相关金标场景**：GOLD-01、GOLD-10、GOLD-11、GOLD-12、GOLD-13、GOLD-14、GOLD-16、GOLD-17、GOLD-18、GOLD-19、GOLD-22。最终以本需求全部必需路径的有效证据判断完成，不按链接数量评分。

#### R08 · GPU/批量/缓存/共享计算、显存、预算与真实速度

**领域权威**：FE/QE plans + effective backend trace。

**继承工作单**：[QE-04](#issue-qe-04)、[QE-05](#issue-qe-05)、[QE-06](#issue-qe-06)、[QE-07](#issue-qe-07)、[QE-08](#issue-qe-08)、[QE-10](#issue-qe-10)、[QE-19](#issue-qe-19)、[QE-20](#issue-qe-20)、[QE-28](#issue-qe-28)、[QE-35](#issue-qe-35)、[FP-04](#issue-fp-04)、[FP-05](#issue-fp-05)、[FA-04](#issue-fa-04)、[FA-10](#issue-fa-10)、[FO-01](#issue-fo-01)、[FO-02](#issue-fo-02)、[FO-04](#issue-fo-04)、[QA-01](#issue-qa-01)、[QA-02](#issue-qa-02)、[V-05](#issue-v-05)、[V-09](#issue-v-09)、[V-10](#issue-v-10)、[V-23](#issue-v-23)。

**新增专项**：[DTA-10](#issue-dta-10)、[DTA-12](#issue-dta-12)、[STA-01](#issue-sta-01)、[STA-03](#issue-sta-03)、[RCP-04](#issue-rcp-04)、[SRH-01](#issue-srh-01)、[SRH-05](#issue-srh-05)、[SRH-06](#issue-srh-06)、[SRH-09](#issue-srh-09)、[SIM-07](#issue-sim-07)、[OPS-02](#issue-ops-02)、[OPS-03](#issue-ops-03)、[OPS-04](#issue-ops-04)、[OPS-07](#issue-ops-07)、[OPS-08](#issue-ops-08)、[QAT-01](#issue-qat-01)、[QAT-03](#issue-qat-03)。

**相关金标场景**：GOLD-01、GOLD-04、GOLD-05、GOLD-06、GOLD-07、GOLD-16、GOLD-17、GOLD-20、GOLD-21、GOLD-23、GOLD-25、GOLD-28、GOLD-33、GOLD-34、GOLD-39、GOLD-40。最终以本需求全部必需路径的有效证据判断完成，不按链接数量评分。

#### R09 · 少参数诊断寻优、有限组合、早停、预算与试验记录

**领域权威**：FO conditional search / ledger。

**继承工作单**：[QE-28](#issue-qe-28)、[FP-03](#issue-fp-03)、[FO-01](#issue-fo-01)、[FO-02](#issue-fo-02)、[FO-03](#issue-fo-03)、[FO-04](#issue-fo-04)、[FO-05](#issue-fo-05)、[PL-01](#issue-pl-01)、[PL-02](#issue-pl-02)、[PL-03](#issue-pl-03)、[V-09](#issue-v-09)、[V-10](#issue-v-10)、[V-11](#issue-v-11)、[V-16](#issue-v-16)。

**新增专项**：[STA-04](#issue-sta-04)、[STA-07](#issue-sta-07)、[STA-10](#issue-sta-10)、[RCP-03](#issue-rcp-03)、[RCP-07](#issue-rcp-07)、[RCP-08](#issue-rcp-08)、[SRH-01](#issue-srh-01)、[SRH-02](#issue-srh-02)、[SRH-03](#issue-srh-03)、[SRH-04](#issue-srh-04)、[SRH-05](#issue-srh-05)、[SRH-06](#issue-srh-06)、[SRH-08](#issue-srh-08)、[SRH-09](#issue-srh-09)、[SRH-10](#issue-srh-10)、[SIM-03](#issue-sim-03)、[MOD-02](#issue-mod-02)、[OPS-01](#issue-ops-01)、[OPS-02](#issue-ops-02)、[OPS-03](#issue-ops-03)、[OPS-10](#issue-ops-10)。

**相关金标场景**：GOLD-01、GOLD-03、GOLD-04、GOLD-20、GOLD-24、GOLD-25、GOLD-34、GOLD-35、GOLD-38。最终以本需求全部必需路径的有效证据判断完成，不按链接数量评分。

#### R10 · 过拟合、跨窗衰减、purge/embargo、封存、多重尝试

**领域权威**：FO data authority + QE statistical evidence。

**继承工作单**：[QE-32](#issue-qe-32)、[FO-02](#issue-fo-02)、[FO-05](#issue-fo-05)、[V-10](#issue-v-10)、[V-11](#issue-v-11)、[V-12](#issue-v-12)、[V-21](#issue-v-21)。

**新增专项**：[DTA-11](#issue-dta-11)、[STA-02](#issue-sta-02)、[STA-03](#issue-sta-03)、[STA-04](#issue-sta-04)、[STA-05](#issue-sta-05)、[STA-06](#issue-sta-06)、[STA-07](#issue-sta-07)、[STA-08](#issue-sta-08)、[STA-09](#issue-sta-09)、[STA-10](#issue-sta-10)、[RCP-07](#issue-rcp-07)、[RCP-08](#issue-rcp-08)、[SRH-05](#issue-srh-05)、[SRH-07](#issue-srh-07)、[SRH-08](#issue-srh-08)、[SRH-09](#issue-srh-09)、[MOD-01](#issue-mod-01)、[MOD-02](#issue-mod-02)、[MOD-08](#issue-mod-08)。

**相关金标场景**：GOLD-03、GOLD-04、GOLD-09、GOLD-11、GOLD-21、GOLD-23、GOLD-24、GOLD-34。最终以本需求全部必需路径的有效证据判断完成，不按链接数量评分。

#### R11 · 多互补赢家、模型特征准入、增量价值而非总分

**领域权威**：FA use-case admission + FO selection + modeling evidence。

**继承工作单**：[FA-05](#issue-fa-05)、[FA-07](#issue-fa-07)、[FA-11](#issue-fa-11)、[PL-04](#issue-pl-04)、[V-04](#issue-v-04)、[V-05](#issue-v-05)、[V-08](#issue-v-08)、[V-11](#issue-v-11)、[V-14](#issue-v-14)、[V-16](#issue-v-16)、[V-17](#issue-v-17)。

**新增专项**：[RCP-07](#issue-rcp-07)、[RCP-09](#issue-rcp-09)、[SRH-03](#issue-srh-03)、[SRH-04](#issue-srh-04)、[SIM-01](#issue-sim-01)、[SIM-03](#issue-sim-03)、[SIM-04](#issue-sim-04)、[SIM-05](#issue-sim-05)、[SIM-08](#issue-sim-08)、[SIM-10](#issue-sim-10)、[MOD-01](#issue-mod-01)、[MOD-02](#issue-mod-02)、[MOD-03](#issue-mod-03)、[MOD-04](#issue-mod-04)、[MOD-07](#issue-mod-07)。

**相关金标场景**：GOLD-03、GOLD-04、GOLD-33、GOLD-34、GOLD-35、GOLD-36、GOLD-37。最终以本需求全部必需路径的有效证据判断完成，不按链接数量评分。

#### R12 · 聚类前去重、近重复、动态新簇、刷新/重训

**领域权威**：FA similarity/cluster governance + modeling schema。

**继承工作单**：[FA-08](#issue-fa-08)、[FA-09](#issue-fa-09)、[FA-10](#issue-fa-10)、[PL-05](#issue-pl-05)、[PL-06](#issue-pl-06)、[V-14](#issue-v-14)、[V-15](#issue-v-15)、[V-17](#issue-v-17)。

**新增专项**：[SRH-04](#issue-srh-04)、[SIM-01](#issue-sim-01)、[SIM-02](#issue-sim-02)、[SIM-03](#issue-sim-03)、[SIM-04](#issue-sim-04)、[SIM-05](#issue-sim-05)、[SIM-06](#issue-sim-06)、[SIM-07](#issue-sim-07)、[SIM-08](#issue-sim-08)、[SIM-09](#issue-sim-09)、[SIM-10](#issue-sim-10)、[MOD-05](#issue-mod-05)、[MOD-06](#issue-mod-06)。

**相关金标场景**：GOLD-35、GOLD-36、GOLD-37。最终以本需求全部必需路径的有效证据判断完成，不按链接数量评分。

#### R13 · 有经济意义的操作顺序、zscore/rank去重、模型表示

**领域权威**：FE DAG property analysis + FP typed recipe。

**继承工作单**：[FP-07](#issue-fp-07)、[FP-08](#issue-fp-08)、[FP-09](#issue-fp-09)、[V-06](#issue-v-06)、[V-08](#issue-v-08)、[V-18](#issue-v-18)。

**新增专项**：[DTA-06](#issue-dta-06)、[DTA-08](#issue-dta-08)、[RCP-01](#issue-rcp-01)、[RCP-02](#issue-rcp-02)、[RCP-03](#issue-rcp-03)、[RCP-05](#issue-rcp-05)、[RCP-09](#issue-rcp-09)、[RCP-10](#issue-rcp-10)、[MOD-03](#issue-mod-03)、[MOD-04](#issue-mod-04)。

**相关金标场景**：GOLD-26、GOLD-27、GOLD-28、GOLD-29、GOLD-33、GOLD-40。最终以本需求全部必需路径的有效证据判断完成，不按链接数量评分。

#### R14 · DSL可读、完整recipe、血缘、生产自动更新和模型版本

**领域权威**：FE/DA/FP execution + FA asset refs。

**继承工作单**：[QE-27](#issue-qe-27)、[QE-31](#issue-qe-31)、[FP-01](#issue-fp-01)、[FP-03](#issue-fp-03)、[FP-04](#issue-fp-04)、[FP-05](#issue-fp-05)、[FP-07](#issue-fp-07)、[FP-08](#issue-fp-08)、[FP-10](#issue-fp-10)、[FP-11](#issue-fp-11)、[FP-12](#issue-fp-12)、[FA-02](#issue-fa-02)、[FA-03](#issue-fa-03)、[FO-03](#issue-fo-03)、[FO-05](#issue-fo-05)、[PL-02](#issue-pl-02)、[PL-05](#issue-pl-05)、[PL-06](#issue-pl-06)、[PL-07](#issue-pl-07)、[PL-08](#issue-pl-08)、[QA-01](#issue-qa-01)、[QA-02](#issue-qa-02)、[V-01](#issue-v-01)、[V-04](#issue-v-04)、[V-07](#issue-v-07)、[V-08](#issue-v-08)、[V-17](#issue-v-17)、[V-18](#issue-v-18)、[V-19](#issue-v-19)、[V-21](#issue-v-21)、[V-22](#issue-v-22)、[V-23](#issue-v-23)、[V-24](#issue-v-24)。

**新增专项**：[DTA-01](#issue-dta-01)、[DTA-02](#issue-dta-02)、[DTA-03](#issue-dta-03)、[DTA-07](#issue-dta-07)、[DTA-09](#issue-dta-09)、[DTA-10](#issue-dta-10)、[DTA-12](#issue-dta-12)、[RCP-01](#issue-rcp-01)、[RCP-02](#issue-rcp-02)、[RCP-06](#issue-rcp-06)、[RCP-09](#issue-rcp-09)、[RCP-10](#issue-rcp-10)、[SRH-01](#issue-srh-01)、[SRH-07](#issue-srh-07)、[SRH-08](#issue-srh-08)、[SRH-10](#issue-srh-10)、[SIM-06](#issue-sim-06)、[SIM-09](#issue-sim-09)、[SIM-10](#issue-sim-10)、[MOD-01](#issue-mod-01)、[MOD-05](#issue-mod-05)、[MOD-06](#issue-mod-06)、[MOD-08](#issue-mod-08)、[OPS-01](#issue-ops-01)、[OPS-04](#issue-ops-04)、[OPS-05](#issue-ops-05)、[OPS-06](#issue-ops-06)、[OPS-08](#issue-ops-08)、[OPS-09](#issue-ops-09)、[OPS-10](#issue-ops-10)、[QAT-01](#issue-qat-01)、[QAT-02](#issue-qat-02)、[QAT-03](#issue-qat-03)、[QAT-04](#issue-qat-04)、[QAT-05](#issue-qat-05)、[QAT-06](#issue-qat-06)、[QAT-07](#issue-qat-07)、[QAT-08](#issue-qat-08)。

**相关金标场景**：GOLD-01、GOLD-04、GOLD-06、GOLD-07、GOLD-08、GOLD-09、GOLD-10、GOLD-15、GOLD-17、GOLD-19、GOLD-25、GOLD-26、GOLD-27、GOLD-30、GOLD-31、GOLD-32、GOLD-33、GOLD-34、GOLD-37、GOLD-38、GOLD-39、GOLD-40。最终以本需求全部必需路径的有效证据判断完成，不按链接数量评分。

#### R15 · PIT、未来函数、时钟/证券轴/数值政策一致与严格守门

**领域权威**：DA/FE/QE validated contracts across every entrypoint。

**继承工作单**：[QE-04](#issue-qe-04)、[QE-05](#issue-qe-05)、[QE-09](#issue-qe-09)、[QE-14](#issue-qe-14)、[QE-18](#issue-qe-18)、[QE-19](#issue-qe-19)、[QE-21](#issue-qe-21)、[QE-23](#issue-qe-23)、[QE-24](#issue-qe-24)、[QE-25](#issue-qe-25)、[QE-26](#issue-qe-26)、[QE-29](#issue-qe-29)、[QE-30](#issue-qe-30)、[QE-31](#issue-qe-31)、[FP-01](#issue-fp-01)、[FP-02](#issue-fp-02)、[FP-04](#issue-fp-04)、[FP-06](#issue-fp-06)、[FP-09](#issue-fp-09)、[FP-10](#issue-fp-10)、[FP-11](#issue-fp-11)、[FA-01](#issue-fa-01)、[FA-02](#issue-fa-02)、[FA-07](#issue-fa-07)、[FA-09](#issue-fa-09)、[FO-02](#issue-fo-02)、[FO-03](#issue-fo-03)、[PL-01](#issue-pl-01)、[PL-05](#issue-pl-05)、[PL-07](#issue-pl-07)、[QA-01](#issue-qa-01)、[V-01](#issue-v-01)、[V-07](#issue-v-07)、[V-12](#issue-v-12)、[V-18](#issue-v-18)、[V-19](#issue-v-19)、[V-21](#issue-v-21)、[V-23](#issue-v-23)、[V-24](#issue-v-24)。

**新增专项**：[DTA-01](#issue-dta-01)、[DTA-02](#issue-dta-02)、[DTA-03](#issue-dta-03)、[DTA-04](#issue-dta-04)、[DTA-05](#issue-dta-05)、[DTA-06](#issue-dta-06)、[DTA-07](#issue-dta-07)、[DTA-09](#issue-dta-09)、[DTA-10](#issue-dta-10)、[DTA-11](#issue-dta-11)、[DTA-12](#issue-dta-12)、[STA-01](#issue-sta-01)、[STA-09](#issue-sta-09)、[RCP-03](#issue-rcp-03)、[RCP-04](#issue-rcp-04)、[RCP-06](#issue-rcp-06)、[SRH-07](#issue-srh-07)、[SIM-02](#issue-sim-02)、[SIM-04](#issue-sim-04)、[SIM-06](#issue-sim-06)、[MOD-01](#issue-mod-01)、[MOD-04](#issue-mod-04)、[MOD-05](#issue-mod-05)、[MOD-08](#issue-mod-08)、[OPS-03](#issue-ops-03)、[OPS-09](#issue-ops-09)、[QAT-01](#issue-qat-01)、[QAT-02](#issue-qat-02)、[QAT-04](#issue-qat-04)、[QAT-05](#issue-qat-05)、[QAT-06](#issue-qat-06)、[QAT-07](#issue-qat-07)、[QAT-08](#issue-qat-08)。

**相关金标场景**：GOLD-01、GOLD-06、GOLD-07、GOLD-08、GOLD-09、GOLD-10、GOLD-11、GOLD-14、GOLD-17、GOLD-18、GOLD-19、GOLD-20、GOLD-22、GOLD-25、GOLD-28、GOLD-29、GOLD-30、GOLD-31、GOLD-32、GOLD-33、GOLD-34、GOLD-36、GOLD-38、GOLD-39、GOLD-40。最终以本需求全部必需路径的有效证据判断完成，不按链接数量评分。

#### R16 · 删除失败物化值但保留公式/配方/试验和受保护依赖

**领域权威**：FA retention authority + DA deletion executor。

**继承工作单**：[PL-02](#issue-pl-02)、[PL-03](#issue-pl-03)、[PL-07](#issue-pl-07)、[PL-08](#issue-pl-08)、[V-11](#issue-v-11)、[V-20](#issue-v-20)、[V-24](#issue-v-24)。

**新增专项**：[DTA-08](#issue-dta-08)、[STA-04](#issue-sta-04)、[RCP-09](#issue-rcp-09)、[SRH-06](#issue-srh-06)、[SRH-10](#issue-srh-10)、[OPS-01](#issue-ops-01)、[OPS-05](#issue-ops-05)、[OPS-06](#issue-ops-06)、[OPS-07](#issue-ops-07)、[OPS-08](#issue-ops-08)、[OPS-10](#issue-ops-10)、[QAT-08](#issue-qat-08)。

**相关金标场景**：GOLD-26、GOLD-34、GOLD-38、GOLD-39、GOLD-40。最终以本需求全部必需路径的有效证据判断完成，不按链接数量评分。

#### R17 · 可控Agent建议、解释与确定性验收

**领域权威**：FO proposals + strict validators。

**继承工作单**：[QE-34](#issue-qe-34)、[FP-10](#issue-fp-10)、[V-22](#issue-v-22)。

**新增专项**：[STA-12](#issue-sta-12)、[SRH-07](#issue-srh-07)、[SRH-10](#issue-srh-10)、[QAT-05](#issue-qat-05)。

**相关金标场景**：GOLD-03、GOLD-23、GOLD-34。最终以本需求全部必需路径的有效证据判断完成，不按链接数量评分。


---


<a id="handoff"></a>
## 15. 开发交付、来源与历史复现说明

### 15.1 一个PR应交付什么

每个根因PR至少含：改动范围与实际调用入口、旧错误复现或已修证明、规范合同、权威实现diff、所有adapter联动、独立金标、公开链trace、语义版本与受影响历史制品、回滚或无需迁移理由。不要让97个旧问题和80个专项变成177套新类；同一根因可以合并PR、共享证据，但每个工作单都要有关闭链接。

如果某条任务无需改代码，因为当前已有正确实现，提交真实入口测试与范围证明，不作无意义重构。如果某条尚无GPU/私有行情/执行数据，只关闭已经验证的层级，保留精确BLOCKED子项。源码和单元测试通过，不等于已经签发实盘可执行或容量等级。

### 15.2 整改工作台账必须能中断续做

每一批改动结束，把实际HEAD、完成项、失败项、待做项、改动合同、已运行命令及结果、下一批依赖保存到仓库既有任务位置。新的AI会话读取此状态继续，而不是重新生成建议或重置试验账本。RNG、budget和sealed authority另有运行级持久状态，不能用整改MD替代生产控制存储。

建议新增发现编号`NEW-001`起，包含证据类别。先确认影响再指定优先级，不能把grep命中`shift(-1)`一律宣称前视：标签生成有合法负移；不能把所有`except`一律删掉：需要保留正确分类与有条件fallback。

### 15.3 一份足够严格的完成报告

| 字段 | 必需内容 |
|---|---|
| 工作区 | 起始与结束commit、未提交改动保护、实际分支/环境 |
| 任务账本 | 177项状态及新增项；完成、已修、非Bug、阻断分别计数 |
| 需求覆盖 | R01–R17及V1 AUD子问题均有实现/证明/阻断 |
| 真实路径 | public API/CLI/HTTP/jobs→adapter→实际kernel→资产裁决的覆盖范围 |
| 测试 | 独立golden、属性测试、故障注入、同包/跨包/设备级命令与结果 |
| 数值与成本 | CPU/GPU同义参数、误差、样本状态、峰值内存、实际端到端资源 |
| 资产迁移 | 受影响evaluation/health/admission/cluster/feature/model及缓存的失效/重算计划 |
| 生产回放 | 冻结recipe增量/全量一致、晚到数据和checkpoint恢复 |
| 清理 | roots/leases验证、dry-run、测试环境物理删除receipt；生产动作须已有授权 |
| 剩余风险 | 无法验证的准确原因、需要的输入/硬件/命令与受限能力，不写模糊“后续优化” |

最终判断不是“总共修了多少行”，而是**同一输入在所有合格入口具备一致的金融含义；所有决策有正确证据；已发布资产能按冻结配方更新；失败大值能被安全回收**。

### 15.4 交给开发AI的最终指令

> 完整执行本文件，不要再次只给建议。先保护工作区并记录真实HEAD，核查全部97个历史任务与80个扩展任务；先确认现有实现，再修改权威实现及所有调用方。每项写正确合同和真实失败测试，再修代码；按W0–W9依赖逐批推进，不被单项外部依赖阻断其余工作。不能删守门、降低门槛、假造数值、用required skip/mock/GPU模拟签验收。无法证实的问题用代码与测试澄清；新问题追加台账。修复后联动旧证据失效、模型输入版本、增量重放和受引用保护的失败值清理。最终交付真实代码差异、逐项闭环证据与全部剩余阻断；远程推送、生产发布和真实资产删除严格遵循用户已有授权。

### 15.5 本版来源与合并方式

来源是本次对话已提供的V1主文档、V2主文档、V2启动指令及V2 ZIP内问题/需求/来源/隔离复现账本。本轮对它们做全文读取和程序化交叉核对，新增实现规格明确标E；没有把新的规格冒充新的源码发现。下文固定commit链接和25项隔离复现仅承接历史记录，用于当前开发定位；必须在当前checkout重新核实。

所有97项旧详细工作单的正文被保留，并追加V3实施要求。旧工作单中“已阅源码”“隔离复现”均指历史审查；缺少当前运行证据的项不得因此标VERIFIED。旧问题中的概括性测试前提，由第4节统一合同与第13.1节澄清约束；这避免把上一版含糊表达当成新的错误金标。

#### 原文件完整性记录

本表哈希针对收到的原始文件字节，仅证明合并来源一致，不是任何量化代码的安全或正确性证明。

| 原文件 | 字节数 | SHA-256 |
|---|---:|---|
| 量化平台代码审查与改造交接_20260906.md | 25196 | `5327586254e8816d47870b84a8175783baef5237f32173425b78274a001281bb` |
| 量化平台_AI整改总任务书_V2_20260906.md | 176593 | `207e030402b1004049da5779610765c5c30002695cb70365511de0ce65ffee11` |
| 给开发AI的启动指令.md | 3522 | `017da3ad7cef16c0f262a8f467ba234104e203dceabadd1ec146494a708fcaa0` |

### 15.6 历史25个隔离复现与32个源码索引

下表是V2已记录的隔离表达式/分支观察，**本轮未重新运行**。它们简化了无关包装，并未导入整个项目、未使用真实行情、未在CUDA执行；旧复现脚本退出0表示“重现旧错误”，不代表修复验收通过。本文件内给出全部观察与正确合同，开发AI无须寻找旧ZIP或额外脚本才知道要写什么回归。

| 复现 | 关联问题 | 观察 | 正确合同 |
|---|---|---|---|
| PR-01 | FP-01 | `{"input_finite":2,"output_finite":0}` | 保留两个有限值，或显式拒绝未规范化的证券键；不得静默变全NaN。 |
| PR-02 | FP-02 | `{"first_order":1.0,"reversed_order":9.0}` | 同键冲突必须拒绝，不能取first导致顺序依赖。 |
| PR-03 | FP-03 | `{"supplied":{"max_lag":1,"method":"dense","pct":false},"forwarded":{}}` | FP参数必须显式映射到对应FE参数或拒绝，不允许悄悄消失。 |
| PR-04 | QE-11 | `{"returns":[-0.1,0.0],"max_drawdown":-0.0,"drawdown_series":[0.0,0.0]}` | 最大回撤应为0.1且处于水下。 |
| PR-05 | QE-12 | `{"returns":[0.1,-1.0],"max_drawdown":-0.0,"drawdown_series":[0.0,null]}` | 破产必须报告损失与破产状态，不能最大回撤为0。 |
| PR-06 | QE-13 | `{"actual_python_type":"tuple","tuple_length":3}` | 应返回符合Sharpe scalar合同的缺失值/状态，而不是回撤三元组。 |
| PR-07 | QE-19 | `{"factor_shape":[2,3,2],"allowed_mask_shape":[2,3],"error":"IndexError"}` | 2D共享mask应与广播后的3D mask等价。 |
| PR-08 | QE-20 | `{"long_count":5,"short_count":5,"overlap_count":5}` | 退化因子不得构造重叠多空集合并称其为有效策略。 |
| PR-09 | QE-01 | `{"true_per_factor_means":[0.025,-0.025],"published_scalar_copied_to_all":[0.0]}` | 保留每因子+0.025/-0.025，不能跨F平均后复制。 |
| PR-10 | QE-02 | `{"quantile_profile":[0.01,0.02,0.0,0.03],"scalar_after_generic_adapter":[0.015]}` | 分层向量必须保留，不得通用nanmean抹掉。 |
| PR-11 | QE-14 | `{"original_periods":6,"periods_after_as_1d":4}` | 应保留日历和缺失状态；不能把观测数量当实际连续时长。 |
| PR-12 | QE-16 | `{"current_formula":1.1547005383792515,"scipy_bias_true":1.1547005383792515,"claimed_bias_false":2.0}` | 函数声明bias=False时需返回校正值，或修改名称/版本明确未校正口径。 |
| PR-13 | QE-33 | `{"reported_stability":10.708206522644142,"correlation_scale_after_inverse":0.999999999}` | 相关性尺度应在[-1,1]；原Fisher-z输出需另名另单位。 |
| PR-14 | QE-34 | `{"perfectly_symmetric_profile":[0.25,0.140625,0.0625,0.015625,0.0,0.015625,0.0625,0.140625,0.25],"reported_left_right_asymmetry":0.234375}` | 纯左右不对称指标对mirror对称U应为0。 |
| PR-15 | FA-01 | `{"gate_count":0,"hard_gates_passed":true}` | 必须覆盖全部必需门槛，空门槛不能通过。 |
| PR-16 | FA-04 | `{"desirabilities_with_missing_replaced_by_zero":[0.9,0.9,0.0],"dimension_score":0.0}` | 缺证据状态与已测低质量分开；必需缺失用证据门拒绝，不称其已测为0。 |
| PR-17 | FP-07 | `{"input_steps":[{"semantic_id":"SMOOTH:ewma","stage":"temporal","halflife":3},{"semantic_id":"SMOOTH:ewma","stage":"temporal","halflife":10}],"steps_after_dedupe":[{"semantic_id":"SMOOTH:ewma","stage":"temporal","halflife":3}]}` | 不同参数或不同输入DAG的操作不得按粗semantic key删掉。 |
| PR-18 | FA-08 | `{"target_before_merge":"A","affinity_A":0.9,"affinity_B":0.1,"target_after_merge":"B","recorded_affinity":0.9}` | 选择B必须使用B的0.1并重新过门槛，不能借用A的0.9。 |
| PR-19 | FP-09 | `{"illegal_sequence":["E_RepresentationScaling","A_Missingness","D_Neutralization"],"tag":["RAW"],"certified_exception":"RAW"}` | 实际非RAW非法步骤即使带RAW标签也必须拒绝。 |
| PR-20 | FP-12 | `{"fallback_text":"<generator object _get_instructions_bytes at 0x7fd585147a60>"}` | hash输入应为实际指令/代码内容，不是带对象身份的iterator repr。 |
| PR-21 | FA-07 | `{"serialized_label_maturity":"false","parsed_bool":true}` | 字符串false不得被解析为True；严格类型反序列化。 |
| PR-22 | QE-22 | `{"gross_return":0.0,"long_turnover":-1.0,"cost_rate":0.001,"net_return":0.001}` | 负换手应拒绝，不能提高net收益。 |
| PR-23 | QE-23 | `{"prices":[100.0,110.0,110.0],"entry_price":110.0,"exit_price":110.0,"pre_entry_return_counted":0.10000000000000009,"current_long_leg_pnl":0.050000000000000044}` | 同VWAP买卖且无持有区间，毛PnL必须为0。 |
| PR-24 | QE-06 | `{"mean_ic":0.05,"std_ic":0.0,"gpu_expression_on_cpu":50000000000.0}` | 零方差IR与CPU相同返回明确不可计算状态，不造5e10高分。 |
| PR-25 | PL-01 | `{"normalization_failure_record":{"candidate_id":"?","content_hash":"","status":"FAILED","reason":"normalization_failed"},"report_constructor_error":"content_hash must be a non-empty string"}` | 错误报告必须能表示尚未取得factor hash的输入，不得再次抛错。 |

**历史源码范围**：下面是V2来源账本所记录的阅读范围；不是当前HEAD覆盖清单。开发AI必须记录本轮实际代码范围和对应commit。固定链接用于定位，不能把整个目录都视为已读。

| 来源键 | 固定commit文件 | 历史阅读范围 |
|---|---|---|
| fp_adapter | [`factor_preprocess/factor_preprocess/adapters/fe_operator.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/factor_preprocess/factor_preprocess/adapters/fe_operator.py) | full |
| portfolio | [`quant_evaluator/metrics/portfolio_stats.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/quant_evaluator/metrics/portfolio_stats.py) | 1-580 |
| underwater | [`quant_evaluator/metrics/underwater.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/quant_evaluator/metrics/underwater.py) | 1-360 |
| qe_runtime | [`quant_evaluator/runtime/evaluator.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/quant_evaluator/runtime/evaluator.py) | 265-825；825-1205 (response tail truncated)；1200-end |
| qe_gpu | [`quant_evaluator/runtime/gpu_executor.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/quant_evaluator/runtime/gpu_executor.py) | full |
| gpu_portfolio | [`quant_evaluator/kernels/gpu/portfolio.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/quant_evaluator/kernels/gpu/portfolio.py) | full |
| cohort | [`quant_evaluator/metrics/probe_portfolio/_core.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/quant_evaluator/metrics/probe_portfolio/_core.py) | 1-310 (long response truncated)；240-460 |
| health | [`factor_assets/profiling/health_card.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/factor_assets/profiling/health_card.py) | 1-360 |
| fo_qe | [`factor_optimizer/factor_optimizer/adapters/quant_evaluator.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/factor_optimizer/factor_optimizer/adapters/quant_evaluator.py) | 1-330 |
| runner | [`factor_optimizer/factor_optimizer/search/runner.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/factor_optimizer/factor_optimizer/search/runner.py) | 1-270 |
| platform | [`quant_platform/app/orchestrator.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/quant_platform/app/orchestrator.py) | 1-330；540-1040 |
| fp_registry | [`factor_preprocess/factor_preprocess/registry/transforms.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/factor_preprocess/factor_preprocess/registry/transforms.py) | 1-270；300-800 |
| parity_tests | [`factor_preprocess/tests/test_fe_operator_parity.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/factor_preprocess/tests/test_fe_operator_parity.py) | 1-240 |
| grammar | [`factor_preprocess/factor_preprocess/grammar/search_grammar.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/factor_preprocess/factor_preprocess/grammar/search_grammar.py) | 1-420 |
| metric_grading | [`factor_assets/profiling/metric_grading.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/factor_assets/profiling/metric_grading.py) | 200-430 |
| health_policy | [`factor_assets/profiling/policies.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/factor_assets/profiling/policies.py) | 290-480 (V2)；500-780 (V1 source reading) |
| qe_registry | [`quant_evaluator/registry/metrics.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/quant_evaluator/registry/metrics.py) | 1-250 (V1)；1700-2185 (V2) |
| labels | [`quant_evaluator/contracts/label_bundle.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/quant_evaluator/contracts/label_bundle.py) | full |
| capability | [`factor_optimizer/factor_optimizer/capabilities.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/factor_optimizer/factor_optimizer/capabilities.py) | full |
| lineage | [`factor_preprocess/factor_preprocess/contracts/treatment_lineage.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/factor_preprocess/factor_preprocess/contracts/treatment_lineage.py) | 1-270 |
| shape | [`quant_evaluator/metrics/shape_evidence.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/quant_evaluator/metrics/shape_evidence.py) | 1-280；350-650 (V1 source reading) |
| dimensions | [`factor_assets/profiling/dimensions.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/factor_assets/profiling/dimensions.py) | 1-300 (V1 source reading) |
| promotion | [`factor_assets/library/promotion_gate.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/factor_assets/library/promotion_gate.py) | 1-270；440-660 (V1 source reading) |
| incremental | [`factor_assets/clustering/incremental.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/factor_assets/clustering/incremental.py) | 1-540 (V1 source reading) |
| fa_pareto | [`factor_assets/optimizer/pareto.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/factor_assets/optimizer/pareto.py) | 1-210 (V1 source reading) |
| static | [`factor_engine/api/static_analysis.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/factor_engine/api/static_analysis.py) | 1-210 (V1 source reading) |
| taxonomy | [`factor_assets/profiling/taxonomy.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/factor_assets/profiling/taxonomy.py) | 1-260 (V1 source reading) |
| multifidelity | [`factor_optimizer/factor_optimizer/search/multifidelity.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/factor_optimizer/factor_optimizer/search/multifidelity.py) | 1-245 (V1 source reading) |
| ic_summary | [`quant_evaluator/metrics/ic_summary.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/quant_evaluator/metrics/ic_summary.py) | 1-190 (V1 source reading) |
| registry_adapters | [`quant_evaluator/metrics/registry_adapters.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/quant_evaluator/metrics/registry_adapters.py) | 1-250 (V1 source reading) |
| adaptive_bins | [`quant_evaluator/contracts/adaptive_bins_policy.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/quant_evaluator/contracts/adaptive_bins_policy.py) | full (V1 source reading) |
| fp_policies | [`factor_preprocess/factor_preprocess/registry/policies.py`](https://github.com/18047533889/quant_projects/blob/f9e7237fafe4112391e62c8874caae7996ce0b11/factor_preprocess/factor_preprocess/registry/policies.py) | 1-420 (V1 source reading) |

### 15.7 两版保留的一手参考资料

以下沿用历史两版的参考索引，本版未重新联网核验。它们支持统计或工具定义，不为项目的等级阈值、相关门槛、搜索预算、聚类节奏或收益作背书。实施时如依赖当前软件API、市场制度或费用规则，应另核验实际版本与生效日。

1. Harvey, Liu & Zhu, “… and the Cross-Section of Expected Returns”, RFS 2016；NBER 工作论文：`https://www.nber.org/papers/w20592`。支持多重检验提高证据门槛，不提供通用 RankIC 等级。
2. Bailey & López de Prado, “The Deflated Sharpe Ratio”, JPM 2014，DOI `10.3905/jpm.2014.40.5.094`。支持记录搜索次数和校正选择偏差/非正态；不是生产盈利保证。
3. Bailey, Borwein, López de Prado & Zhu, “The Probability of Backtest Overfitting”, JCF 2017，DOI `10.21314/JCF.2016.322`；作者机构存档 `https://escholarship.org/uc/item/4w1110bb`。适合在样本内候选收益矩阵上追加选择稳定性研究，不替代真实时间前推验证。
4. Gu, Kelly & Xiu, “Empirical Asset Pricing via Machine Learning”, RFS 2020；`https://www.nber.org/papers/w25398`。支持考虑非线性与交互价值，而非只看单变量线性指标。
5. Feng, Giglio & Xiu, “Taming the Factor Zoo”, JF 2020，DOI `10.1111/jofi.12883`。支持检验相对既有因子的新增贡献。
6. Traag, Waltman & van Eck, “From Louvain to Leiden: guaranteeing well-connected communities”, Scientific Reports 2019，DOI `10.1038/s41598-019-41695-z`。算法连通性不等于金融信号同质性。
7. SciPy filtfilt 官方说明：`https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.filtfilt.html`。前后双向滤波不应冒充因果在线滤波。
8. scikit-learn 官方 Decision Trees / StandardScaler：`https://scikit-learn.org/stable/modules/tree.html`、`https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html`。支持按模型需求选择预处理，不机械统一归一化。

9. SciPy `scipy.stats.skew` 官方定义：`https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.skew.html`，用于核对有限样本偏差校正及NaN约定。
10. Python `dis.get_instructions` 官方文档：`https://docs.python.org/3.11/library/dis.html`，用于区分指令迭代器与可稳定序列化的实际指令内容。

### 15.8 本版交付核对

V1九个AUD及其子问题、V2全部97项详细正文、17类原始需求均已纳入。97项逐一追加改法/测试/迁移说明，新增80项E类专项、40组独立金标与8条端到端链。全部任务编号唯一，依赖只引用本文已有编号并可拓扑排序。上述是**任务书结构与内容合并校验**，不是仓库整改、行情回测或生产验收结果。

**只把本文件交给开发AI即可。最终要交付的是修改后的正确代码和逐项证据，而不是另一份更短的方案。**
