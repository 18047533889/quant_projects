# 修复方法审查与训练期优化方案（2026-09-21）

完整通过数、跳过项及平台根目录未闭合问题见 [回归验证记录](REPAIR_VALIDATION_20260921.md)。

## 1. 先区分三个问题

优化器负责提议候选、分配预算与选择；预处理负责实际变换及参数准入；evaluator 负责分层收益、形态和其他评价证据。生成一个 repair family 名称，不代表有可执行配方；有可执行配方，也不代表外样本改善。

本轮已用真实函数重现并修复：

| 问题 | 所属层 | 修复 |
|---|---|---|
| SMA 不在 causal smoothing 方法域中，候选被拒绝 | Optimizer | 增加 SMA 分支；执行委托 FP trailing_sma |
| KAMA 搜索输出 er_window/fast_span/slow_span，执行却要 period_er/period_fast/period_slow | Preprocess | 搜索空间与 registry 元数据统一使用执行参数名 |
| 缩减平滑预算却降低 IIR alpha，反而更迟钝 | Preprocess | 提高 alpha 下界而保留上界；Kalman process_noise 同样按反方向收窄 |
| 倒 U 型在声明中存在，但训练中心冻结合同不接受它 | Optimizer | 支持 INVERTED_U_REPAIR；并列时与 U 型一样优先 0.5 中心 |
| 分层成本可接受正无穷 | Optimizer | 在构造阶段拒绝非有限成本 |
| 多暴露 OLS 把第二个矩阵当成 group，产生全 NaN；min_observations 未映射 | Preprocess → FE | 暴露矩阵作为单个 exposures 元组传递，映射 min_obs 并保留 FP 默认最小样本数 |
| 公共 OLS 接口只接受暴露表，但适配器要求额外 exposure_cols | Preprocess | 对有 FP 签名的 OLS 按非身份列解析暴露；直接底层调用仍要求显式列 |
| CUDA IC 测试使用 12 只股票却期望满足默认 20 只门槛 | 测试夹具 | 正向测试改用 32 只，另保留 CPU/CUDA 对 12 只均不可用的反例；未放宽 QE 门槛 |
| decay/causal smoothing 的候选声明没有统一 FP 执行桥接 | Optimizer/Preprocess 边界 | 新增显式、版本化、仅研究用的编译适配器，先绑定 registry 参数，再委托真实 kernel |

原有形态、路由、条件搜索和 U/倒 U 端到端定向集：102 项通过。新增缺陷反例先在修改前失败，修改后复验。不能由此推断全部生产流程已无缺陷。

## 2. 本次新增执行入口的边界

[preprocessing.py](../factor_optimizer/adapters/preprocessing.py) 的 compile_smoothing_repair 接收修复族、候选参数、训练期冻结的 natural_time_scale 和 training_context_ref。

- 返回的计划提供 transform、实际 parameters、mapping_version 与 identity，便于审计。
- execute 必须显式 allow_research=True；它只委托 FP 注册执行入口，不读取收益标签、不生成评分、不发布因子。
- 它不会自动接管已有业务回调。接线时应把条件搜索候选送入该编译器，执行所得信号再交给 QE；不能继续在回调中忽略候选参数。
- 训练上下文引用是调用者提供的来源标识，不是自动验证数据权限的证书。正式训练/验证边界仍必须由 EvaluationProtocol、SplitPlan 和数据能力控制。
- 不支持的其他修复族明确拒绝，不能把窗口改写、行业中性化或 U 型处理误交给平滑编译器。
- 缺少 factor_preprocess 时需先安装匹配库，可使用本库的 factor_preprocess 可选依赖。

FP 的准入域比数学函数的理论定义域窄。例如当前 SMA window 下界为 3、EWMA halflife 下界为 3，IIR alpha 上界为 0.5。相对参数映射后越界的候选抛出 IneligibleSmoothingRepair，应计入提案/不合格记录，不计作成功评估。禁止静默裁剪，也不能用 raw 冒充候选输出。

### 2.1 版本 v1 的明确换算

令 H 是训练期确定并冻结的自然时间尺度（单位：观测条数），r 为 natural_time_scale_relative，h=Hr。不同方法的 h 只是搜索标尺，不保证等效频率响应。

| 方法 | 解析到 FP 的参数 |
|---|---|
| SMA | window=ceil(h)，min_periods=window |
| EWMA | halflife=h，min_periods=1 |
| IIR | alpha=1−2^(−1/h) |
| KAMA | period_er=ceil(h)，period_fast=2，period_slow=max(3,ceil(3h))，use_current=False |
| Kalman | measurement_noise=1，process_noise=alpha²/(1−alpha)；稳态增益与 alpha 对应，初始化阶段不等同 EWMA |

decay refinement 在 half_life_relative=True 时采用 h=H·decay；False 时 decay 是旧状态保留系数：

```math
y_t=d\,y_{t-1}+(1-d)x_{t-1}
```

零 decay 对应纯一期滞后，不应除以零；但当前 FP 对 unit gain 不准入，因此在编译阶段拒绝这一提案。以上是新适配器 v1 的明确约定，不给历史上缺少单位说明的 recipe 自动补语义。

所有这些平滑只消费 t−1 及更早值。原始 RAW 对照不能被重复滞后；候选应各自从同一份 raw 派生。不同 kernel 的缺失处理不完全一致（例如 IIR/Kalman 会重置），不要仅凭相同 h 宣称数学等价。

## 3. U 型分层到底怎么优化

U 型意味着左右两端可能同时有收益，中间弱；不能用低 Rank IC 直接判断因子无用，也不能只优化 Q高−Q低的线性多空组合。先在训练期检查分层收益形态、曲率、覆盖、各时间窗稳定性，再选择处理族。

令 r 为截面百分位秩、c 为训练期拟合中心：

```math
z_{U}=\lvert r-c\rvert^p,\qquad
z_{\mathrm{inverted\ U}}=-\lvert r-c\rvert^p
```

左右分支也可拆开验证：

```math
z_L=\max(c-r,0),\qquad z_R=\max(r-c,0)
```

中心、幂次、是否拆成两支都属于选择次数，必须纳入预算和多重检验记录。当前中心冻结函数采用有限网格、训练分数最大化；倒 U 使用独立修复族身份，不再需要冒用 U 型身份。该函数不会替你完成时间切分，不能把验证/测试分数伪装成 train_scores。

U/倒 U evaluator 的现有合成端到端测试通过；这只说明这些明确样例，没有证明用户报告的实际数据、缓存、适配器及网页展示完全无误。

## 4. 推荐的训练期寻优流程

不存在不依赖数据的“最好方法”。建议优先提高外样本可信度与净收益，而不是不断增加变体直到训练分数最高。

1. 先固定可用时间、标签区间、交易延迟、成本、资产宇宙和缺失口径。标签不能穿过折边界；按持有期设置 purge/embargo。
2. 将外层 TRAIN 再划成按时间推进的内部训练/验证折。每折只在过去拟合参数和尺度，在下一段评分；禁止随机打乱时序。
3. 留 RAW；先修数据完整性、重复处理和时钟问题，再按诊断选择一到两个修复族。高换手试短 EWMA/SMA，异常尖峰试 robust clipping，稳定 U 型试冻结中心/左右尾，而不是所有处理做笛卡尔积。
4. 先小型、预声明离散网格，确认每个候选确实执行且参数敏感。连续数值参数和足够历史再用条件 TPE；同一方法没有足够观测时，不应假装已学到最优区域。
5. 内部折统计净指标、稳定性、换手、成本、覆盖与复杂度。在相同日期、相同资产有效支持上与 RAW 做配对比较；不能让候选靠丢掉困难样本取胜。
6. TRAIN 内拟合结束后冻结中心、尺度、参数和变换顺序；外层 VALIDATION 只对有限候选做最终选择。验证后不得继续无限调参还称同一验证集为外样本。
7. 只有冻结赢家一次性进入 SEALED TEST；测试失败如实报告，不能反向改变方向、中心或窗口再重复测试。

建议预声明一个简单的跨折目标，例如：

```math
J=\mathrm{median}_{k}(g_k)-\lambda\,\mathrm{MAD}_{k}(g_k)-\eta\,C
```

g_k 是第 k 个内部验证折相对 RAW 的、已计成本的改善，C 为固定定义的复杂度；系数需在看结果前确定。这是推荐实验设计，不是本轮偷偷替换的生产评分公式。覆盖率、时钟、可交易性和灾难阈值仍是硬门，不能被 J 补偿。

分层漏斗应先检查合法性/覆盖，再廉价统计，再完整评估；不同保真度的分数不能直接混排。没有晋级、因准入域被拒绝、全 NaN warmup、执行异常、评分不足，必须分别记录，不能统一写成“方法没效果”。

## 5. 热身与历史上下文

SMA 窗口 w 至少需要 w 个过去观测；把 validation 单独截断再平滑会制造新的热身 NaN。可带入验证开始前的合法历史作为只读上下文，执行后仅对 validation 日期计分。标签不能随历史上下文泄漏到拟合过程。

新增测试检查真实 SMA 手算值、五种平滑的当前条排除与前缀不变性、输入不被覆盖、候选到执行绑定，以及不合格提案的明确拒绝。后续生产接入还需验证多资产、PIT、快照、成本与证据身份；研究适配器不改变整库 research_only 状态。

## 6. 旧脚本不是当前安全入口

jobs/optimize_factors.py 和 jobs/optimize_top_factors.py 仍有退役 get_function 调用及历史研究选择逻辑。本轮未运行或部署这些会落盘真实结果的旧任务，也没有把它们的历史输出宣称为严格外样本收益。业务方应使用受控的 split-aware 接口接线；若实际报错来自这些脚本，需要结合其输入和运行日志单独迁移。
