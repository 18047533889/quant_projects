# 优化库自动研究入口与方法公式（2026-09-22 更新）

后续已接通指定 COS 因子池的 DataAccess 有界读取、TRAIN-only 多维诊断，
以及基于真实 20 层谷底/峰顶的修复中心提案，见
[COS 与 20 层诊断报告](COS_DIAGNOSTICS_20260922.md)。
以下原 95 项网格在具备有效分层诊断时最多额外增加 2 个训练拟合候选；
基础预处理已接入批量入口（需要显式处理谱系和可选暴露），见
[基础预处理与退化保护](BASELINE_PIPELINE.md)。未知谱系不会被假定为未处理。
默认已接入 [joint.v1 联合多指标选优](JOINT_SELECTION.md)，不是只展示诊断。
COS 谱系/暴露自动加载及最终独立 TEST 报告仍未完成。

## 逐方法复核更新

本轮在原有入口上修复 KAMA 缺失窗口状态、极大有限值标准化、
候选常量日遗漏覆盖率检查，以及真实示例按行号选择训练资产的问题。
新增自动方向与平滑组合、双侧尾部搜索和逐方法数据审查工具。
下面 2026-09-21 的回放数字保留为历史证据；最新结果见
[逐方法复核报告](METHOD_AUDIT_20260922.md)。

默认候选上限由 64 改为 128，默认完整网格为 95 条（含明确不适用的候选）。
每个有足够 RAW 证据的因子逐条记录 transform、orientation、覆盖率及有效训练日数。
真实执行错误保留原因；不会跳过失败后仍声称全方法执行成功。

当允许 SIGN_ORIENTATION 和平滑/衰减族时，新增复合方案：

$$f'_{t,i}=s\,S_\theta(f)_{t,i},\qquad s\in\{-1,+1\}.$$

方向 s 与平滑参数 θ 都在 TRAIN 选择，VALIDATION 只检验一个冻结赢家。
配置 compose_smoothing_sign=False 可禁用组合；显式 families 不包含
SIGN_ORIENTATION 时不额外翻转。所有组合计入 maximum_candidates，
预算不足明确拒绝，不静默截断，也不在验证集更换方向。

尾部自动搜索现在包含 hinge 的 top/bottom，以及 saturation 的 top/bottom/both。
形态幂、标准化或排名运行成功，不等于 Rank IC 必然提升。

覆盖率同时约束有效单元与有效 IC 日期：

$$C=\min\left(\frac{N_{\text{共同有效单元}}}{N_{\text{RAW有效单元}}},
\frac{N_{\text{共同有效IC日}}}{N_{\text{RAW原始截面有效IC日}}}\right)\ge0.90.$$

RAW 原始截面单独作为日期覆盖率基准，避免候选丢掉难预测日期、
或变成常量而 IC 无定义后仍显示“100%覆盖”。TRAIN 与 VALIDATION 均检查。
validation_candidate_identity 和 validation_coverage 记录真正验证的候选；
即使最终回退 RAW，也能追踪是哪一个方案未通过。

KAMA 在 ER 窗口有 NaN/inf 时，保留已建立的历史状态，完整有限窗口恢复后再递推。
窗口内值不变（有效波动为 0）与窗口无效（缺失）不再混为一谈。
注意这仍可能延续陈旧状态；不是具有自动时效淘汰的生产信号。
ZSCORE 使用先缩放再计算均值/样本标准差，数学口径不变，
但不再把 [-1e308,0,1e308] 因方差溢出错误变成全 0。
执行映射版本提升为 smoothing-repair.v2 / value-repair.v3，旧身份不能代表新数值语义。

真实示例先对齐交易日及标签窗口，再用最终 TRAIN 的日期选择资产，
不再假设各 parquet 文件最后 500 行代表相同区间。

逐方法审查覆盖全部 27 个准入平滑参数，不再每种方法只测第一组。
自动搜索将同一平滑计划的正负方向相邻执行，共享一个落值数组，
但仍逐方向计算配对指标；只保留最近的底层数组，不缓存整个候选网格。
400×64、4 个衰减计划、5 次重复的局部 A/B 中，数组含 NaN 逐元素完全一致，
中位变换时间 0.28090s → 0.13876s（约 2.02 倍）。
这仅是落值阶段，不代表包括数据读取和 QE 评分的全流程提速。

真实审查示例统一使用 DataAccess：因子使用受授权的 read_uri，
行情使用 ashare_stock_daily_adj 的列、时间和股票过滤，并限制扫描/返回预算。
在 server-c 对已有镜像做研究回放的命令如下（不下载或更新正式数据）：

```bash
DATA_ACCESS_SKIP_COS_MIRROR=1 \
DATA_ACCESS_READ_URI_ROOTS=/home/sunhaiwei/quant_projects/weekly_backtest_output/factor_matrices_all \
ASHARE_PARQUET_ROOT=/home/sunhaiwei/cos_data \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
.venv/bin/python factor_optimizer/examples/method_audit.py
```

这次本地镜像回放不等于已完成 COS 平铺候选池读取或上游 PIT 认证。

该工具即使遇到 RAW 的 IC 无效，也会执行每种可数值运行的方法，
检查原始行对齐、资产顺序不变性和截断未来后的前缀一致性。
逐方法分数仅在 TRAIN 作诊断，另由正式自动入口验证唯一赢家，不把验证集用于挑方法。
缺暴露或 DSL 的方法明确报告所缺输入，不宣称已完成数值效果验证。

## 本轮变化

训练/验证日期不需要用户手工选择。新增库入口 `factor_optimizer.research_batch.optimize_factor_batch`：接收轴和标签时点明确的 QE FactorBatch / LabelBundle，自动分段、在训练集选参数、验证、保留 RAW 或输出冻结后的变体。旧的 jobs 筛选脚本没有冒充这个新入口。

值修复统一由 `compile_value_repair` 编译成冻结计划，数值处理交给 FP，Rank IC 交给 QE。本轮从仅平滑/衰减的执行桥扩展至 12 个数值修复族，加 RAW 控制，共 13 个 family 的值级研究路径。仍不更改 production capability，不部署或入库。

## 最小调用

```python
from factor_optimizer.research_batch import optimize_factor_batch
result = optimize_factor_batch(factor_batch, label_bundle, allow_research=True)
optimized_batch = result.optimized
for factor_id, outcome in result.factors.items():
    print(factor_id, outcome.status, outcome.selected_family, outcome.reason)
```

`factor_batch` 为 (时间, 资产, 因子)，time_axis.values 与 labels.decision_time 必须一致；资产标识及顺序也必须与 labels.asset_axis.values 完全一致。库不会猜收益标签或自动挪动错误轴。实际 server-c 示例见 `examples/real_batch_audit.py`，直接读取已有小批量数据，默认完成日期选择。

输出为每个输入因子保留一条记录：`improved`、`raw_retained`、`invalid_raw` 或 `error_raw_retained`。候选错误明确保留原因；未执行的方法不能算作执行成功。返回计划有内容身份和训练证据引用，可用同一冻结计划处理后续值。输出包含应用冻结计划后的全部因子值，但不包含测试集评分。

## 自动时间分段和选择规则

- 时间顺序前 60% 为 TRAIN，中间 20% 为 VALIDATION，后 20% 保留为 TEST。绝不随机打散时间。
- 预留前 30 条历史供平滑暖机；在 TRAIN→VALIDATION、VALIDATION→TEST 边界，剔除 label_end_time 跨界标签，并再保留 1 条决策观测的隔离。
- 剔除后默认至少 60 个训练日、30 个验证日、30 个测试日；数据不足明确拒绝，不捏造结果。实际 IC 还需至少 20 个配对有效资产。
- 固定的有限候选网格只在 TRAIN 比较。默认自然尺度 10 bars 是事前基准，不冒称由数据学习；不同平滑时间尺度由训练比较选出。
- 将训练差值按时间分成三段，各段平均差为 g₁,g₂,g₃，选择分数如下：

$$G=\operatorname{mean}(g_1,g_2,g_3)-\operatorname{std}_{ddof=0}(g_1,g_2,g_3).$$

这里差值为候选减 RAW 的逐日 Rank IC。惩罚跨时间段不稳定的提升；G 默认须超过 0.01。双方每次使用相同日期和相同有效资产，候选保留的有效单元比例默认不低于 90%。

TRAIN 只选一个赢家。将这个冻结赢家放到 VALIDATION，用共同连续时间块 bootstrap 比较与 RAW 的 IC 差值：默认 499 次、块长至少 max(5, 标签 horizon)。有效区间不足三个块就拒绝确认。95% 双侧分位数区间的下界必须大于 0 且至少 0.01 才接受。验证失败不尝试第二名，防止反复消费验证集。

TEST 标签的 values/validity 不参与候选生成、打分或选择；其时间边界用于分区。这里的“保留”不是生产 sealed-test authority 认证。该入口的目标是稳健 Rank IC 改进，不声称同时优化净收益、交易成本或组合风险；成本/多维治理继续使用已有完整证据链。跨很多因子的正式显著性声明仍需多重检验控制，不能把这些研究区间当成全局统计认证。

## 逐方法的实际公式

所有横截面运算仅使用同日可用资产，不用未来日期。参数选择仅在 TRAIN 完成。

### 方向、排名与形态

方向修复为 f′=−f（flip）或 f′=f（keep）。横截面平均秩 R 的归一坐标为：

$$r=\frac{R-1}{n-1},\quad n>1;\qquad r=0.5\text{ when }n=1.$$

并列值默认使用平均秩；横截面排名也支持注册表声明的 min tie method。缺失保持缺失。

对称 U 型修复及倒 U 型：

$$u(r)=|r-c|^p,\qquad u_{inv}(r)=-|r-c|^p.$$

默认搜索中心 c∈{0.35,0.50,0.65} 和幂 p∈{1,2}，仅依据训练证据选择并冻结。不是对每个验证日用收益重新拟合。单调幂变化不一定改变排名，因此不应宣称总能提高 Rank IC。

显式 asymmetry=True 时使用左右侧归一：

$$u(r)=\begin{cases}((c-r)/c)^p&r<c,\\((r-c)/(1-c))^p&r\ge c.\end{cases}$$

c=0 或 1 的空侧安全处理；倒 U 再整体取负。此选项可以显式编译，默认自动网格不额外增加非对称自由度。

### 平滑与衰减

平滑只用历史值并保留既有 FP 的滞后一条观测约定，不直接使用当期值代替历史状态。

| 方法 | 参数及口径 |
|---|---|
| SMA | 前 w 条历史观测均值；完整窗口后出值 |
| EWMA | 半衰期 h，α=1−2^(−1/h)，sₜ=αxₜ+(1−α)sₜ₋₁；输出使用历史状态 |
| IIR | 同一半衰期换算 α；按 FP 缺失策略递推 |
| KAMA | ER 窗口 ceil(h)，fast=2，slow=ceil(3h)，不使用当前 bar；仍须满足 FP 参数域 |
| Kalman | 测量噪声 r=1，过程噪声 q=α²/(1−α)；只是稳态增益对应，不声称有限样本初始化与 EWMA 完全相同 |
| 相对 decay | h=自然尺度×decay |
| 绝对 decay | decay 是历史状态保留系数，α=1−decay |

KAMA 的效率比与自适应增益沿用 FP：

$$ER_t=\frac{|x_t-x_{t-w}|}{\sum_{j=0}^{w-1}|x_{t-j}-x_{t-j-1}|},\quad SC_t=[ER_t(2/(fast+1)-2/(slow+1))+2/(slow+1)]^2.$$

分母为零、暖机和缺失按已有 kernel 明确规则处理，独立 oracle 已覆盖。EWMA 跳过缺失观测的行为与其他平滑方法不完全相同，不能把输出差异一概当 bug。

`compile_admissible_smoothing_grid` 对事前时间尺度 {3,5,8,10,13,20,30,60} 取 FO/FP 合法参数域交集，先排除无法执行的候选，不偷偷夹逼参数。本轮默认尺度 10 下产生 27 个合法平滑计划。零 decay 若映射 alpha=1 而不满足 FP admission，明确 ineligible，不伪装 RAW。

### 截尾、尾部、标准化及缺失

| 方法 | 定义 |
|---|---|
| ROBUST_OUTLIER | clip(x,Q_low,Q_high)，分位点按同日截面计算 |
| TAIL_SATURATION | top 截上尾、bottom 截下尾、both 截双尾；阈值为同日 Q_q / Q_(1−q) |
| TAIL_HINGE | top=max(x−h,0)，bottom=min(x−h,0) |
| ZSCORE | (x−mean)/sample_std（ddof=1），再截到 [−cap,cap] |
| ROBUST_SCALE | (x−location)/scale，location 为 mean/median；scale 为 MAD、IQR 或 sample std |
| MISSING fill | 按资产，只向前填充最多 freshness_window 条，不后向填充 |
| MISSING flag | 缺失指示通道；若全不缺失产生常量，IC 无效，不给假 0 分 |

MAD=median(|x−median(x)|)，IQR=Q₀.₇₅−Q₀.₂₅。本轮修正大量并列值导致 MAD/IQR 为 0 的非恒定截面：回退 sample std，只有真正恒定截面才返回 0。先按最大绝对值缩放后计算，避免有限大数溢出后被错误变为全 0；inf 视为缺失。

默认填充候选窗口为 1、3、5；也保留 flag 候选，不能把诊断通道与填充值混为一谈。单纯正向仿射缩放和排名通常不会提高 Rank IC，新入口在没有稳健增益时保留 RAW。

## 仍需要额外执行输入的方法

行业/市值/风格中性化需要明确对齐的外部暴露；窗口替换、算子替换、交互需要原始 DSL 和输入来源。仅给因子数值矩阵不能可靠反推出它们，新值级入口会记录具体缺项而不伪造执行。TS rank/zscore 的旧 family schema 没有窗口，当前值编译器明确拒绝该分支。ABANDON 是控制决策而非数值优化，这个入口对不可确认情况采用 RAW 保留。

## 实际数据回放

自行选取 server-c 已有四个因子，500 决策日×64 资产，2024-08-02 至 2026-08-25。资产集合只用 TRAIN 覆盖率选择；收益来自已有后复权 AdjVwap，定义为 VWAP(t+2)/VWAP(t+1)−1：t 收盘信号，下一交易日成交，再下一交易日退出。没有使用 t 日 VWAP 假设可成交。

自动切分后：TRAIN 267，VALIDATION 97，TEST 保留 100；test_evaluated=False。每个有有效 RAW 的因子记录 61 个候选，其中缺暴露/DSL 的族明确不可执行。

| 因子 | 结果 |
|---|---|
| price_smoothness | 无稳健训练提升，保留 RAW |
| volume_ratio_persistence | TRAIN 增益 0.0504098，VALIDATION 下界 −0.0291381，不确认，保留 RAW且不重试第二名 |
| ema_trend_simple | 无稳健训练提升，保留 RAW |
| cyclical_trend_confidence | 有效训练 IC 日不足，invalid_raw |

这证明能运行且会拒绝不可靠改动，不证明所有市场/因子都无 bug，也不重新认证这些历史矩阵上游构建的 PIT 合规性。没有发布因子或写回正式因子数据。

## 回归证据

合成批量测试覆盖反向因子自动翻转、U 型恢复、好因子不强制改变、无效因子隔离、测试标签/值/有效性污染不影响选择、候选/因子顺序不影响结果、长标签 purge、短数据拒绝、严格参数域。

- 本轮 Optimizer + Preprocess 最终全套：1661 passed、1 xfailed、16 warnings（101.39 秒）；比上一轮新增 53 项测试。预期失败为 HP filter 非因果性，继续禁止生产使用。
- Evaluator 全套：1270 passed、2 skipped、10 warnings（144.61 秒）；两项跳过为既有 build/lib 缺失 kernel 的检查，不是新增修复跳过。
- 真实示例完整执行成功，结果如上；没有把真实数据高分当作通过标准。
- 平台根目录默认 pytest 仍是 14 项收集错误、1 skipped、2 warnings（77.29 秒），错误文件清单与 [第一轮回归记录](REPAIR_VALIDATION_20260921.md) 相同。本轮没有改动其他 AI 正在处理的 Factor Engine 文件，不能宣称平台全绿。
- 两库 API 参考已重新生成，并通过 --check 与文档回归。

独立文档测试在各自包目录运行分别 2 passed、3 passed。曾在根目录仅用 importlib 模式混跑两份同名文档测试，出现 test_transform_catalog_matches_default_registry 的 factor_preprocess.registry 导入错误；这是该调用方式的包命名冲突，没有用这种失败结果冒充通过。此外，在补缺失填充回归的红/绿阶段临时恢复旧源码时运行过索引测试，test_api_reference_matches_source 报过期；恢复最终实现后索引检查和完整回归均重新通过。
