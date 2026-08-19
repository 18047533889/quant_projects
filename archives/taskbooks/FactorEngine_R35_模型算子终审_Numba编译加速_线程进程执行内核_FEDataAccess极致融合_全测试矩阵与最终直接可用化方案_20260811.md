# FactorEngine R35｜模型算子终审、Numba 编译加速、线程/进程执行内核、FE×DataAccess 极致融合、全测试矩阵与最终直接可用化方案

> 审计基线
>
> - Repository: `18047533889/quant_projects`
> - Branch: `main`
> - Fresh audited HEAD: `8449d9c253c55308f3e406a15739e984387e9691`
> - 日期：2026-08-11
> - R33：负责 FactorEngine × DataAccess 统一物理执行、读波、扫描复用、零拷贝、FactorBlock、COW、TimeToDurableCommit。
> - R34：负责全算子真实正确性证据、PIT、参数域、source contract、backend variant 认证。
> - R35：继续补齐 **模型/统计类算子的真实可用性、计算 kernel 的最优加速路线、Numba/线程/进程的实际落地、FE×DA 最终执行形态，以及一套足够强的全量测试体系**。

---

# 0. 先给结论

当前代码不能诚实地描述为：

```text
“所有算子都已经真实无误，可以直接生产使用，
而且 FactorEngine + DataAccess 已经把一批因子算到尽可能快。”
```

更准确的状态是：

```text
1. 基础算子和大量扩展算子已经有很丰富的 runtime、contract 和审计框架；
2. R28/R30/R31/R32 修掉了很多真实 bug；
3. 模型家族已经有明显的 PIT/fit_lag/label_horizon/solver fail-closed 改进；
4. 但多个模型模块本身仍明确是 experimental / research-only；
5. supervised panel forecast 仍明确 not_pit_certified；
6. 一些“所有 model 都有 timing contract”的门仍可通过 name/family 自动生成 default contract；
7. 大量高成本模型 kernel 仍是 Python row/column loop + 每窗重拟合；
8. Numba 已列为 optional accel 依赖，架构也考虑了 Numba nogil，但尚未形成统一的 Numba production kernel layer；
9. HybridExecutor 有线程池和进程池，但当前 AdaptiveBatchScheduler 的 root payload 不可 pickle，因此实际统一强制 thread；
10. R33 尚未完成，FE 与 DataAccess 仍存在双 planner、read wave 未真正执行、physical stage 主要是规划视图、aggregate+join 临时 Parquet 等问题；
11. 最新 HEAD 当前没有可验证的 GitHub workflow/combined-status 证明，因此历史 “1522 passed”等不能自动视为当前 main 已完整回归。
```

所以 R35 的目标不是继续“堆算子”，而是把系统收口成：

> **所有 retained operator 都在其声明角色内真实可用；所有 production-fast kernel 都有 reference parity；所有高成本模型都有明确的计算预算与加速策略；所有 batch 优化都经过全量差分测试。**

---

# 1. 最终应该形成六层 Operator Lane

不要再用“是不是 production”一个标签表达所有事情。

每个 canonical 最终进入以下唯一一类：

## Lane A｜FAST_NATIVE_ALPHA

```text
- 默认自动因子挖掘可见
- PIT certified
- 参数域 certified
- 至少一个高性能 backend
- 低/中成本
- 适合大批量搜索
```

典型：

```text
基础量价
rolling mean/std/corr
cross-section
简单 group
大量技术结构算子
```

## Lane B｜EXPENSIVE_CERTIFIED_ALPHA

```text
- 真实可生产计算
- 公式/PIT/参数域 certified
- 但成本高
- 不进入默认随机大规模搜索
- 进入 budget-aware / explicit search lane
```

典型候选：

```text
GARCH
复杂 robust regression
DMD / SSA
matrix profile
高级 nonlinear dependence
```

## Lane C｜STATE / CONDITION / EVENT

```text
- 可以在因子表达式中直接使用
- 可能不适合作为独立 alpha terminal
- 是 if / gating / regime / state condition 的合法 intermediate
```

## Lane D｜MODEL_FEATURE / MODEL_SCORE

```text
- 模型型 operator
- 必须有 ModelTimingContract
- 训练/预测边界明确
- 可能被策略显式使用
- 默认不作为普通随机 grammar 里的轻量 operator
```

## Lane E｜DIAGNOSTIC / RESEARCH

```text
- in-sample residual
- self-fit R²
- diagnostic state
- 论文研究/诊断
- 不进入默认 production mining
```

## Lane F｜DELETE / TOMBSTONE

```text
- 无数据
- 未来函数
- 语义错误
- 随机不可复现
- 完全重复
- 没有经济/统计意义且不可修
```

目标不是“所有 retained 算子都 Lane A”，而是：

> **所有 retained 算子都能在自己的 Lane 内真实直接用，不会被错误地暴露到不该出现的场景。**

---

# 2. 当前模型侧的真实状态

## 2.1 Panel Model Family 还不能算 production direct-use

当前 `cross_section/panel_model.py` 自己声明：

```text
Rolling panel model operators (P2, experimental)
```

注册时：

```text
status="experimental"
extend_research_only(...)
```

而 supervised model：

```text
pit_safe=False
tags:
    supervised_model
    not_pit_certified
```

包括：

```text
panel_rolling_pcr_forecast
panel_rolling_pls_forecast
panel_rolling_elastic_net_forecast
panel_regime_conditioned_forecast
panel_mixture_of_experts_score
```

虽然当前 kernel 已经在 `_forecast_loop` 内部增加：

```text
fit_lag=1
label_horizon maturity
```

这是正确方向，但：

```text
“实现改对了”
!=
“已经完成 production certification”
```

### R35 要求

把这批 model 分为：

```text
A. descriptive panel PCA factors
B. supervised forward-label model scores
```

分别认证，不要混在一起。

---

# 3. Panel PCA 类的进一步问题

当前 rolling PCA：

```python
for row in range(rows):
    X = rolling_window(...)
    SVD(X)
```

也就是每一天重新：

```text
切窗口
算 mean/std
填 missing
SVD
```

industry PCA 更是：

```text
row × stock
再按 group 取成员
再做 PCA
```

正确性层面已有：

```text
active coverage gate
missing current stock不污染 peers
SVD sign deterministic orientation
n_components rank cap
```

但性能层面仍很重。

## R35 优化方向

### P0：先保留 Reference Kernel

当前 Pandas/NumPy版本作为：

```text
canonical reference
```

不要为了快直接改掉。

### Fast Kernel 1：rolling covariance sufficient statistics

对固定 active set/规则：

维护：

```text
sum
sum_sq
cross_product
finite_count
```

窗口滑动：

```text
remove oldest
add newest
```

避免每次重新扫描全部 window。

然后：

```text
covariance matrix
→ eigh
```

而不是：

```text
raw X
→ full SVD
```

需要证明：

```text
eigen subspace / explained ratio / residual
≈ reference SVD
```

### Fast Kernel 2：active-set clustering

如果连续很多日：

```text
active instrument mask
```

相同，则：

```text
复用 matrix layout / allocation
```

### Fast Kernel 3：industry batch

不要：

```text
row × stock × repeated industry PCA
```

同一天同一个行业：

```text
只算一次 PCA
```

然后把 loadings回填所有成员。

这是非常明确的当前复杂度优化点。

---

# 4. PCR / PLS / ElasticNet 当前是否“处理好了”

答案：

```text
逻辑比早期正确很多，但还不能说完全 production ready。
```

当前已有：

```text
fit_lag
label_horizon maturity
feature standardization只使用训练数据
PLS多 component deflation
ElasticNet convergence fail-closed
```

但仍缺：

```text
1. independent sklearn/statsmodels oracle
2. 全参数域 parity
3. future perturbation
4. label maturity multi-horizon
5. missing-pattern hostile tests
6. condition-number tests
7. cross-thread/BLAS determinism
8. chunk/incremental semantics
9. performance budget classification
10. latest-head evidence
```

---

# 5. Supervised Model 最危险的边界：Label Contract

必须把：

```text
y
```

从“普通 panel input”升级为：

```python
LabelPanel(
    origin_time,
    horizon,
    maturity_time,
    overlapping,
    purge,
    embargo,
)
```

不能仅靠：

```text
parameter name y
+
label_horizon
```

推断。

Production model必须：

```text
training row s
只能在 decision_time >= label_maturity(s) 后使用
```

---

# 6. ModelTimingContract 必须全部 Explicit

当前 `get_model_timing_contract()`：

```text
explicit entry
否则 family/name-based default
```

因此：

```text
R28_ALL_MODEL_CANONICALS_TIMING_CONTRACTED
```

可以通过自动生成 default。

R35 production必须改成：

```text
explicit ModelTimingContract only
```

自动 default：

```text
research-only hint
```

## Gate

```text
R35_PRODUCTION_MODEL_DEFAULT_TIMING_CONTRACTS = 0
```

---

# 7. 模型名字不能决定语义

不能再因为：

```text
_forecast
_next_
innovation
pca
garch
```

就自动决定：

```text
fit_lag
forecast_horizon
label_horizon
```

名字可以帮助 audit discovery，
不能是 production truth。

---

# 8. In-Sample Model Operators 保留，但要明确 Diagnostic

当前 dynamic regression 已经做了正确区分：

```text
fit_lag=0
→ in-sample descriptive
→ diagnostic_only

fit_lag=1
→ prior / forecast_error
→ causal candidate
```

这个方向应该推广到全部 model families。

例如：

```text
ts_ar_fitted_value
ts_ar_in_sample_resid
```

不能因为没用未来数据就自动成为默认 alpha search terminal。

---

# 9. Model Direct-Use 最终分类

## 可以争取进 Lane A/B 的

```text
prior-window rolling regression
prior AR
Kalman causal filter state
HAR prior forecast
GARCH prior-parameterized shock / next forecast
PCA residual/commonality（严格 rolling prior）
```

## 更适合 Lane D

```text
PCR/PLS/ElasticNet forward-label model
regime model
Mixture-of-Experts
dynamic KNN
kernel Granger
复杂 supervised panel score
```

## Lane E

```text
in-sample fitted value
in-sample residual
self-fit R²
diagnostic coefficient
research transforms
```

---

# 10. Kalman 是最值得 Numba 化的一类

当前 Kalman：

```text
for t in range(n)
```

并且：

```text
for col in range(cols)
```

这是 Numba 非常适合的形态：

```text
纯数值
固定 dtype
递归状态
无 Pandas object
无 SciPy optimizer
```

## 推荐

```python
@numba.njit(cache=True, nogil=True)
def kalman_level_kernel(...)

@numba.njit(cache=True, nogil=True)
def kalman_trend_kernel(...)

@numba.njit(cache=True, nogil=True)
def kalman_beta_kernel(...)
```

外层输入：

```text
contiguous float64 ndarray
```

输出：

```text
float64 ndarray
```

### 并行方式

第一阶段：

```text
Numba kernel parallel=False
FE scheduler负责外层 factor/block thread parallelism
```

避免：

```text
FE threads × Numba prange
```

双层 oversubscription。

如果 benchmark证明：

```text
单个超大 Kalman matrix
```

更适合：

```text
Numba prange over columns
```

则该 task申请：

```text
backend_threads > 1
```

并由 ResourceBroker独占对应 CPU token。

---

# 11. Stateful / CTA Rule Operators 也很适合 Numba

包括：

```text
state_latch
state_hold
state_slew_limit
state_deadband
event_refractory
directional change
episode MFE/MAE
threshold cycles
PSAR
Supertrend
KAMA
Wilder recurrence
```

这些都是：

```text
state_t = f(state_{t-1}, x_t)
```

Numba收益通常明显。

---

# 12. Numba 不应该用在哪些地方

不要“看到 Python就 Numba”。

## 不优先

### SVD / Eigh / LSTSQ

```text
NumPy已经调用 BLAS/LAPACK
```

Numba本身不会神奇地把 SVD变快很多。

应该优化：

```text
调用次数
矩阵规模
sufficient statistics
batch reuse
```

### SciPy minimize

Numba不能直接把：

```text
scipy.optimize.minimize
```

整体 JIT。

可以 JIT：

```text
objective recurrence
```

但 optimizer本身仍有 Python/SciPy调用成本。

### linprog / HiGHS

同理。

### DataAccess IO

应交给：

```text
DuckDB
Arrow
Polars
```

不是 Numba。

### 简单 vectorized NumPy/Polars

已经 SIMD/native，
Numba未必更快。

---

# 13. Numba 当前是什么状态

当前项目：

```text
pyproject optional accel:
    bottleneck
    numba
```

调度设计也认识：

```text
Numba nogil
```

但：

```text
Numba不是 full/core production dependency
```

而我本轮抽查的：

```text
Kalman
rolling regression
PCA/PCR/PLS/ElasticNet
GARCH/HAR
```

主要执行仍是：

```text
NumPy/SciPy + Python loops
```

因此 R35 不要写“支持 Numba 已完成”。

应该写：

```text
Numba capability scaffold exists;
systematic certified Numba kernel layer pending.
```

---

# 14. 建立 NumbaKernelRegistry

不要散落：

```python
try:
    import numba
```

每个文件自己处理。

统一：

```python
NumbaKernelSpec(
    canonical_family,
    kernel_name,
    semantic_version,
    supported_dtypes,
    supported_param_domain,
    nogil,
    parallel,
    cache,
)
```

并有：

```text
reference kernel
numba kernel
parity evidence
benchmark evidence
```

---

# 15. Numba Production Admission

Numba不是“编译成功就 production safe”。

必须验证：

```text
NaN
Inf
dtype
overflow
fastmath
parallel reduction
platform
CPU architecture
```

尤其禁止默认：

```python
fastmath=True
```

因为会改变：

```text
NaN
Inf
associativity
```

FactorEngine需要严格语义时，
默认：

```text
fastmath=False
```

---

# 16. Numba 冷启动成本

JIT第一次运行会慢。

production service：

```text
startup warm compile
```

或：

```text
cache=True
```

并把 compile latency单独统计。

Benchmark区分：

```text
cold
warm
```

---

# 17. Bottleneck 也值得评估

`bottleneck` 已是 accel optional依赖。

适合：

```text
moving mean
moving std
moving min/max
nan reductions
```

但必须先验证：

```text
min_periods
NaN policy
ddof
window endpoint
```

与 canonical一致。

不能只因为 bottleneck快就替换。

---

# 18. 多线程到底要不要

要，但只应该用于：

```text
releases-GIL / native compute
```

最适合：

```text
DuckDB
Polars
NumPy BLAS/LAPACK
Numba nogil
Arrow IO
compression/write
```

当前 HybridExecutor 对这个方向的设计是合理的。

---

# 19. 当前线程模型的现实问题

虽然 HybridExecutor会区分：

```text
pandas_numpy -> process
native -> thread
```

但当前 AdaptiveBatchScheduler明确：

```text
root payload + shared ctx/cache不可 pickle
```

所以：

```text
统一 prefer="thread"
```

也就是说：

> **你现在“有 ProcessPoolExecutor”，但主 batch root执行还没有真正吃到进程池。**

---

# 20. 多进程是不是必须

不是第一优先级。

最佳路线：

```text
先把热点 Python loop
→ Numba/native/vectorized
```

这样大多数任务释放 GIL，
线程就够。

多进程只保留给：

```text
无法 Numba
无法 Polars/DuckDB
真正 Python-heavy
计算足够大
值得付序列化成本
```

---

# 21. 如果要真正用 Process Pool，必须改 Execution Payload

不要把：

```text
live ctx
DataSource object
large DataFrame
closure
```

pickle给 worker。

构造：

```python
ExecutionCapsule(
    plan_ir,
    source_snapshot_id,
    buffer_refs,
    operator_contract_version,
    execution_options,
)
```

worker：

```text
自己初始化 runtime
通过 BufferRef / shared memory / Arrow IPC
读取数据
```

返回：

```text
ResultRef
```

而不是完整大 DataFrame。

---

# 22. Process Worker 不应该复制 DataAccess Cache

如果每个进程都有：

```text
独立几GB panel cache
```

会直接爆内存。

Process lane需要：

```text
immutable shared Arrow buffers
memory-mapped Arrow
shared memory
或 worker-local bounded cache
```

---

# 23. 最重要的并行原则：只有一层拥有 CPU

禁止：

```text
16 FE workers
×
16 MKL threads
×
8 Polars threads
```

正确：

```text
ResourceBroker
→ task拿 N CPU tokens
→ inner library threads=N
```

其他 task不能再抢同一 CPU预算。

---

# 24. threadpoolctl 应真正成为生产能力

项目 performance extra已经包括：

```text
threadpoolctl
```

执行每个 task时：

```python
with threadpool_limits(limits=backend_threads):
    ...
```

而不是只靠环境变量。

因为：

```text
NumPy/MKL/OpenBLAS
```

可能已经在 process import后初始化。

---

# 25. 不建议现在优先上 GPU

当前主要瓶颈更可能是：

```text
重复 scan
重复 materialization
Python rolling refit
重复 SVD/OLS
FE/DA转换
调度 overhead
```

不是简单：

```text
矩阵乘法算力不够
```

GPU会引入：

```text
host-device transfer
CUDA memory
新的 backend parity
新的 deployment complexity
```

只有 benchmark证明：

```text
大型 matrix factor block
```

占主要时间时再考虑。

---

# 26. 也暂时不建议先写 C++/Rust Extension

优先级：

```text
1. DuckDB/Polars native
2. NumPy BLAS
3. Numba
4. algorithmic reuse
5. only then C++/Rust/PyO3
```

如果 Numba仍不够，再把最热点 kernel写 native extension。

---

# 27. Rolling OLS 可以比当前快很多

当前 dynamic regression：

```text
每 row
重新切 window
重新 build design
重新 lstsq
```

对固定小维度 features：

维护：

```text
X'X
X'y
y'y
count
```

滑动窗口：

```text
减掉旧 row outer product
加上新 row outer product
```

然后求解：

```text
small p×p system
```

复杂度从：

```text
O(T * window * p²)
```

降到接近：

```text
O(T * p² + T * solve(p))
```

p通常 <= 5，
收益会非常明显。

---

# 28. Rolling Ridge 同样可以 Sufficient-Stats 化

```text
(X'X + λI)β = X'y
```

直接使用 rolling Gram matrix。

---

# 29. Rolling R² / Residual Std 同样复用统计量

维护：

```text
sum_y
sum_y2
X'X
X'y
```

避免重新计算。

---

# 30. Huber / Quantile / Expectile 不完全能用简单 sufficient stats

但可以：

```text
用 OLS/ridge warm start
Numba IRLS inner loop
批量小矩阵 solve
```

真 quantile LP：

```text
高成本 Lane B
```

不要默认大规模随机挖。

---

# 31. ElasticNet 最适合 Numba

当前 coordinate descent：

```text
for iter
    for feature
        X @ beta
```

可以进一步：

```text
维护 residual
更新一个 beta_j 时 O(n)
```

避免每个 j都重新：

```text
X @ beta
```

再 JIT：

```text
Numba
```

会比当前 Python版本快很多。

---

# 32. PLS 适合小矩阵 Numba Kernel

自定义 NIPALS：

```text
loop component
matrix-vector
deflation
```

可以 JIT。

但一定与：

```text
sklearn PLSRegression
```

做 independent oracle。

---

# 33. GARCH 是“算法复杂度”优先于多线程

当前最重的地方之一：

```text
for row
    取前缀/窗口
    scipy.optimize.minimize
    objective里再递归 h_t
```

这不是简单开 8个线程就能解决。

---

# 34. GARCH 第一阶段 Fast Kernel

保持 canonical语义不变：

```text
1. JIT likelihood recurrence
2. JIT variance recursion
3. 预分配 scratch buffer
4. 避免每次创建 h array，可滚动 scalar h
5. fixed initial parameters保持 reference一致
```

SciPy optimizer仍存在，
但 objective调用会轻很多。

---

# 35. GARCH 第二阶段：专门 Expensive Lane

GARCH不应默认跑在：

```text
10000 random formulas
```

它应该：

```text
BudgetClass = VERY_HIGH
```

mining planner可以：

```text
限制同时候选数量
限制参数组合
先做廉价 proxy筛选
```

---

# 36. 不要为了快改变 GARCH 的统计定义

例如：

```text
隔5天才重估参数
warm-start上一窗口
```

可能非常快，
但会改变 canonical语义或 optimization path。

如果想要：

```text
rolling-refit-every-k
```

应该新 operator/新 parameter semantic version，
不能静默替代严格每窗 refit。

---

# 37. HAR-RV 相对更适合 Rolling OLS Fast Path

HAR本质上：

```text
小维度线性模型
```

可以进入：

```text
rolling Gram matrix engine
```

---

# 38. Kalman 不需要多进程

Numba单次编译后：

```text
整个 T×N panel
```

可非常快。

优先：

```text
Numba + thread/block
```

不是 process pickle。

---

# 39. DMD / SSA / Hankel / Spectral

主瓶颈通常：

```text
SVD/eigendecomposition
```

Numba帮助有限。

优化：

```text
减少重复矩阵构造
复用 Hankel embedding
多输出共享一次分解
batch同 window共享
```

---

# 40. Model Family Multi-Output

例如同一次 GARCH fit可以同时给：

```text
persistence
shock
next vol forecast
vol surprise
```

如果语义完全来自同一 fit window：

```text
一次 fit
→ 多 output
```

不要4个 canonical独立重复 fit。

同理：

```text
PCA loading
resid
commonality
explained ratio
```

可以一次 decomposition供多个 outputs。

---

# 41. Multi-Output Kernel 是模型侧非常大的性能机会

建立：

```python
FamilyKernelResult
```

例如：

```text
GARCHFitBlock
PCABlock
RollingOLSBlock
KalmanStateBlock
```

外部 canonical仍独立，
内部 planner自动共享。

---

# 42. Model CSE 不能只看完整 AST 相同

即使：

```text
GARCH_persistence(x,120)
GARCH_shock(x,120)
```

AST不同，
但共享：

```text
same fit state
```

需要：

```text
semantic sub-kernel dependency
```

层级的 CSE。

---

# 43. 建立 ModelIntermediate DAG

例：

```text
returns
  ↓
GARCHFit(window=120)
  ├─ persistence
  ├─ conditional_variance
  ├─ standardized_shock
  └─ next_vol
```

PCA：

```text
rolling standardized matrix
  ↓
PCA decomposition
  ├─ loading
  ├─ score
  ├─ resid
  └─ commonality
```

---

# 44. Model Intermediate 也需要 Identity

key至少：

```text
input semantic identity
window
model params
fit cutoff
source snapshot
universe
```

不能跨不同 universe共享 PCA。

---

# 45. FactorEngine × DataAccess 当前还没有真正完全融合

DataAccess已经有很好的基础：

```text
DataRequest
CompiledDataRequest
ReadPlan
PreparedRead
physical plan
```

而且 DataRequest文档本身就写：

```text
FactorEngine Analyzer 一次性提交逻辑数据需求
```

这说明正确方向非常明确。

但 FactorEngine当前又有：

```text
BatchDataRequest
```

形成重复 planning authority。

---

# 46. 当前 BatchDataRequest 还不是真正 Multi-Source Batch Planner

当前实现最终：

```text
取 engine.data_source
union所有 fields
创建一个 SourceScanGroup(group_id=0)
```

没有真实按：

```text
daily price
fundamental
holder
event
index
minute
```

分 source scope。

因此 R33必须落实。

---

# 47. 当前 ReadWave 也还没有成为真正执行读波

当前 `build_waves_from_dag`：

```text
SOURCE_SCAN columns = task.inputs
```

但 source scan inputs本来是 predecessor，
不是字段。

对 root/CSE：

```text
columns=(task.op,)
```

甚至可能把 operator name当 column。

而且：

```text
time_range=None
```

所以当前 read wave不能被称为生产级真实 IO planner。

---

# 48. Physical Stage DAG 当前也主要还是规划视图

`physical_lowerer.py` 自己明确：

```text
current production execution
仍是 whole-root backend.execute(root_plan)
```

而 scheduler `_dispatch()`：

```text
CSE_SHARED 真执行
ROOT 真执行
其他 task type -> None
```

因此：

```text
SOURCE_SCAN
ROLLING_SHARED
GROUP
CROSS_SECTION
STATEFUL
OPERATOR
```

当前还没有逐 stage真正执行。

---

# 49. 这也是为什么现在“多线程/多进程”还没有发挥到最优

因为真实执行单位仍是：

```text
whole factor root
```

而不是：

```text
shared physical stages / blocks
```

最优并行应该对：

```text
真正独立且有足够粒度的 physical stage
```

调度。

---

# 50. DataAccess Aggregate+Join 当前还有一次明显 Hot-Path 临时文件

当前 composite physical plan：

```text
minute aggregate
→ Arrow
→ tempfile parquet
→ DuckDB joined SQL
```

这是非常明确的额外：

```text
serialization
disk IO
filesystem metadata
```

R33应去掉。

优先：

```text
DuckDB registered Arrow relation
或
single SQL/CTE
```

---

# 51. FE×DA 最终唯一正确分工

```text
FactorEngine:
    formula
    canonical semantics
    operator DAG
    CSE
    model intermediate
    final factor identity

DataAccess:
    field resolution
    dataset resolution
    physical object scope
    source snapshot
    PIT
    universe
    temporal join
    unit normalization
    minute aggregation
    ScanCost

Unified Physical Planner:
    backend boundaries
    materialization boundaries
    BufferRef
    read waves
    memory
    spill
    write
```

---

# 52. FE 不应该再重新实现 DataAccess Source Planner

删除/降级：

```text
parallel source planning logic
```

变成：

```text
FE PlanDependencyManifest
→ compile to DataAccess DataRequest
```

---

# 53. PreparedRead 应升级成 PreparedBatchReadTemplate

当前 PreparedRead里包含：

```text
resource_reservation
```

对于整批 future waves，
prepare时就占资源可能太早。

拆成：

```text
PreparedBatchReadTemplate
    immutable semantic/physical scope
    no live lease

ExecutionLease
    scheduler JIT admission时拿
```

---

# 54. ResourceBroker 与 DataAccess Governor 必须只有一个 Host Authority

不能：

```text
FE认为有8 CPU
DA也认为有8 CPU
```

然后实际16。

设计：

```text
HostGlobalResourceBroker
├─ FE ComputeLease
├─ DA ScanLease
├─ WriteLease
└─ SpillLease
```

---

# 55. 最快路径不应该优先“多进程”

最终优先级：

```text
0. 少读数据
1. 少做重复计算
2. 不转换格式
3. Native fusion
4. Algorithmic reuse
5. Numba/custom kernel
6. Threads for nogil/native
7. Processes only for residual GIL-bound work
```

---

# 56. 速度优化的第一原则：先降 Work，不是先加 Workers

如果当前做：

```text
1000份重复 rolling fit
```

开32进程只是：

```text
更快地做1000份重复工作
```

真正优化是：

```text
1份 shared intermediate
→ 1000 outputs
```

---

# 57. R35 性能路线图

## Tier 0｜IO

```text
same physical data read once
projection pushdown
row-group prune
snapshot pin
```

## Tier 1｜Representation

```text
Arrow/DuckDB/Polars保持列式
避免 early Pandas
```

## Tier 2｜Expression Fusion

```text
DuckDB SQL
Polars lazy
multi-root
```

## Tier 3｜Shared Numeric Kernels

```text
rolling sufficient stats
PCA/GARCH/Kalman family kernels
```

## Tier 4｜Compiled Loops

```text
Numba
```

## Tier 5｜Parallelism

```text
threads
process residual lane
```

## Tier 6｜Storage

```text
FactorBlock
COW
atomic commit
```

---

# 58. 是否需要更多测试

答案：

> **需要，而且测试应该成为下一阶段的主工程之一。**

当前历史 suite很多，
但还不足以证明：

```text
all operators are actually correct and directly usable
```

原因不是“测试数量少”，而是很多测试维度还没有形成系统覆盖。

---

# 59. 不要再只统计 pytest 数量

```text
1522 passed
```

信息量很有限。

最终应该看：

```text
canonical × required evidence dimension
```

---

# 60. 建立 Test Obligation Matrix

每个 operator自动得到：

```python
TestObligation(
    execution,
    semantic_golden,
    parameter_domain,
    edge,
    temporal,
    source_pit,
    backend_parity,
    batch_parity,
    chunk,
    incremental,
    concurrency,
    performance,
)
```

由 RiskProfile决定哪些 required。

---

# 61. Model Family 必须额外 Required Tests

所有 model-like：

```text
MODEL_ORACLE
FIT_CUTOFF
FUTURE_PERTURBATION
LABEL_MATURITY
SCALER_HISTORY_ONLY
HYPERPARAM_HISTORY_ONLY
CONDITION_NUMBER
NON_CONVERGENCE
MISSING_PATTERN
PARAM_DOMAIN
DETERMINISM
```

---

# 62. PCA 专项测试

至少：

```text
sklearn PCA oracle
manual 2-feature fixture
sign orientation
rank cap
n_components boundaries
missing active-set
one suspended stock
all-equal stocks
universe reorder
universe membership change
batch multi-output parity
```

---

# 63. PCR 专项

```text
sklearn PCA + LinearRegression oracle
future perturbation
label_horizon H=1/5/20
feature scaling only training window
n_components
rank deficiency
```

---

# 64. PLS 专项

```text
sklearn PLSRegression oracle
1/2/3 components
deflation of new sample
degenerate component
missing features
```

---

# 65. ElasticNet 专项

```text
sklearn ElasticNet oracle
alpha
l1_ratio
max_iter/tol
nonconvergence
zero-variance feature
collinearity
```

---

# 66. Kalman 专项

```text
manual recurrence oracle
missing leading rows
missing middle gap
long gap covariance growth
q=0
small r
invalid q/r
chunk checkpoint parity
Numba/reference parity
```

---

# 67. GARCH 专项

```text
independent QML reference
known simulated GARCH path
stationarity boundary
optimizer failure
current-shock leakage
forecast timing
NaN gaps
extreme return
Numba likelihood/reference
```

---

# 68. HAR 专项

```text
manual lag feature construction
rolling OLS oracle
next variance vs next volatility naming
fit t-1
```

---

# 69. Dynamic Regression 专项

```text
statsmodels / numpy oracle
fit_lag 0 vs 1
R² no-intercept
adjusted R²
ridge intercept penalty
Huber convergence
true quantile vs expectile
```

---

# 70. Model Input Permutation Tests

对 panel model：

随机：

```text
column reorder
```

如果语义基于 instrument identity：

```text
reorder后映射回来结果一致
```

抓 positional pairing bug。

---

# 71. Universe Perturbation Tests

PCA/cross-sectional model：

```text
add/remove one stock
```

影响允许变化，
但必须符合：

```text
as-of universe contract
```

不能使用未来 universe membership。

---

# 72. Future Data Poison Test

把：

```text
t+1以后数据
```

改成极端：

```text
1e100
NaN
random
```

输出：

```text
<=t
```

不能变化。

这个测试比普通 prefix更强。

---

# 73. Label Poison Test

对 supervised model：

未来 labels随机改，
验证：

```text
尚未 mature 的 label
不能影响当前 fit
```

---

# 74. Hyperparameter Leakage Test

如果未来支持：

```text
CV / auto-alpha / auto-components
```

必须保证：

```text
hyperparameter selection cutoff
```

也是 t-1。

---

# 75. Backend Test 必须覆盖参数域，不是 operator名

例如：

```text
ts_mean:
window 2/5/20/252
min_periods
NaN
```

不是只测默认一组。

---

# 76. Numba Test Matrix

每个 Numba kernel：

```text
reference parity
NaN mask parity
Inf behavior
dtype
cold compile
warm execution
parallel=False
parallel=True（如支持）
CPU architecture
thread count
```

---

# 77. Thread/Process Equivalence Test

同 batch：

```text
serial
thread 1
thread 4
process 2
process 4
hybrid
```

全部：

```text
result equal
```

---

# 78. Oversubscription Test

测试：

```text
FE=8
BLAS=8
```

故意 bad config，
确保 broker能限制或拒绝。

---

# 79. Process Serialization Test

如果引入 ExecutionCapsule：

```text
pickle/unpickle
worker execute
same output
```

并验证：

```text
source snapshot
factor identity
```

没有丢。

---

# 80. Process Crash Test

worker kill后：

```text
lease release
retry policy
pool recover
no duplicate write
```

---

# 81. FE×DA Integration Golden

建立 tiny真实数据仓：

```text
daily price
minute
fundamental PIT
holder
index
event
```

执行：

```text
DataAccess DataRequest
→ FE batch
→ materialize
→ DataAccess readback
```

---

# 82. Mixed 100-Factor Golden

真实混合：

```text
40 price-volume
10 cross-section
10 fundamental
10 event/index/holder
10 minute-derived
10 stateful
10 model/high-cost
```

对比：

```text
independent reference
```

---

# 83. 1000-Factor Batch Differential

```text
run_one × 1000
vs
run_many × 1000
```

必须相同。

这是最终批量优化最重要的 correctness test之一。

---

# 84. Test Result 应该直接生成“能不能用”的机器结论

不要让人读100个 csv。

每 canonical：

```text
DIRECT_USE_CERTIFIED
EXPENSIVE_CERTIFIED
INTERMEDIATE_CERTIFIED
DIAGNOSTIC_ONLY
PENDING
FAILED
```

---

# 85. 测试失败后自动生成 Remediation Queue

例如：

```text
canonical = ts_xxx
failed = backend_parity
backend = duckdb
params = window=252
```

自动：

```text
remove that backend/domain from production
```

而不是整个 engine停摆。

---

# 86. Current HEAD 的测试证据问题

本轮 GitHub没有返回：

```text
8449d9... combined status
8449d9... workflow run
```

因此：

```text
历史 1522 passed
历史 950 passed
```

不能自动作为当前 HEAD证明。

下一阶段必须：

```text
current HEAD full release run
```

---

# 87. R35 最终系统目标

```text
Operator
    ↓
Typed Contract
    ↓
Reference Correctness
    ↓
Parameter Domain
    ↓
PIT/Source/Model Timing
    ↓
Fast Kernel Certification
    ↓
Batch/Optimizer Differential
    ↓
FE×DA Unified Physical Execution
    ↓
Resource-Aware Parallelism
    ↓
Atomic Materialization
```


# 88. 本轮新确认的模型侧具体问题

下面不是泛化“应该多测试”，而是当前 HEAD 的具体可整改项。

---

# 89. R35-P0-M01｜GARCH Timing Contract 与 Forecast Kernel 不一致

当前 `ModelTimingContract` 对：

```text
ts_garch_next_vol_forecast
ts_garch_persistence
ts_gjr_garch_vol_forecast
```

声明：

```text
fit_cutoff_offset = 1
```

语义是：

```text
参数拟合截止 t-1
```

但当前 `_garch_path`：

```python
fit_seg = seg[:-1] if stat == "shock" else seg
```

因此：

```text
shock -> 参数 fit 到 t-1
forecast -> 参数 fit 包含 t
persistence -> 参数 fit 包含 t
```

`next_vol_forecast` 使用当前 `r_t` 重新估计参数后预测 t+1，
本身可以是完全因果的定义：

```text
decision after r_t observed
fit through t
forecast t+1
```

但它不是 contract 写的：

```text
fit through t-1
```

## 必须二选一

### 方案 A

保持当前 kernel：

```text
fit_cutoff_offset = 0
forecast_horizon = 1
```

并明确：

```text
requires completed current bar
```

### 方案 B

保持 contract：

```text
fit_cutoff_offset = 1
```

kernel改为：

```text
params fit on seg[:-1]
h_t由 prior params得到
h_{t+1}用 r_t update
```

从因子研究角度，我更建议 **B**：

```text
参数严格 prior
当前 shock只用于一步状态更新
```

更稳定、也更容易解释 walk-forward。

但必须 benchmark/semantic-version决定，不能静默改。

---

# 90. R35-P0-M02｜GARCH Persistence 的 Timing 也需统一

当前 persistence：

```text
_fit_garch(seg)
```

包含当前 t。

contract：

```text
fit t-1
```

同样不一致。

如果它被解释为：

```text
“截至当前时刻重新估计的 persistence”
```

contract应 fit=0。

如果被解释为：

```text
“当前时点可用的 prior persistence state”
```

kernel应 fit t-1。

---

# 91. R35-P0-M03｜GJR Forecast 同类 Timing Mismatch

`ts_gjr_garch_vol_forecast`：

```text
contract fit t-1
kernel fit current seg
```

与 GARCH一起统一。

---

# 92. R35-P0-M04｜HAR Next Forecast 的 Fit Cutoff 也要重新定义

HAR next forecast当前：

```text
features at s
target RV_{s+1}
```

训练样本可一直用到：

```text
target RV_t
```

然后用：

```text
X_t
```

预测：

```text
RV_{t+1}
```

如果 decision是在 t收盘后：

```text
RV_t 已知
```

那么这种 fit-through-t 定义是因果的。

但 ModelTimingContract当前统一写：

```text
fit_cutoff_offset=1
```

因此 contract没有准确描述 kernel实际可见信息。

## 要求

把 timing contract从一个整数：

```text
fit_cutoff_offset
```

升级为：

```python
FeatureLabelTiming(
    feature_origin_offset,
    label_origin_offset,
    label_maturity_offset,
    fit_latest_mature_label_offset,
    score_feature_offset,
)
```

否则 forward-label模型很难仅用一个 fit_cutoff准确表达。

---

# 93. R35-P0-M05｜Panel Forecast 允许 0 Feature 的签名缺口

当前：

```python
x1=None
x2=None
x3=None
x4=None
```

都是默认允许。

但 `_forecast_loop`：

```python
np.column_stack([f[:, col] for f in feats])
```

如果：

```text
0 feature
```

会进入非法路径。

## 修复

Typed Signature：

```text
min_feature_panels = 1
max_feature_panels = 4
```

compile阶段拒绝：

```text
no feature
```

而不是 runtime NumPy异常。

---

# 94. R35-P0-M06｜Regime / MoE 的 market_state 实际必需，但函数默认 None

目前 public callable形态允许：

```text
market_state=None
```

kernel马上：

```text
market_state.to_numpy(...)
```

因此缺少 required-panel contract。

修复：

```text
market_state: RequiredPanelParam
```

---

# 95. R35-P0-M07｜PCA `min_history` 语义过松/未显式

当前 `_pca_svd` 的 active coverage：

```text
>= max(2, ceil(current_available_window * 0.7))
```

而 rolling window早期：

```text
当前 available rows远小于 requested window
```

因此一个：

```text
window=120
```

的 PCA 在非常早的日期也可能开始给值。

这未必是数学错误，但它必须是显式 contract：

```text
min_history
min_coverage
```

不能由内部 hardcode：

```text
2
0.7
```

偷偷决定。

建议默认：

```text
min_history >= max(n_components + safety_margin, explicit user/default floor)
```

并在 metadata/ParamSpec公开。

---

# 96. R35-P0-M08｜PCA Sign Tie 仍可能依赖 Instrument Column Order

当前 sign orientation：

```text
找到最大 abs loading
如果它 <0 就整体翻转
tie -> first index
```

若两个 loadings绝对值完全相同：

```text
first index
```

取决于 panel column order。

因此：

```text
columns reorder
```

可能导致整个 eigenvector sign翻转。

虽然 PCA sign本来没有物理唯一性，
但因子值必须有稳定定义。

## 修复

tie-break用：

```text
stable instrument identity
```

而不是 column position。

或者定义：

```text
lexicographic deterministic loading orientation
```

---

# 97. R35-P0-M09｜PCA Loading Description 已与实现漂移

当前描述仍写：

```text
跨窗 sign 对齐
```

但实现已经改成：

```text
stateless per-window deterministic sign
```

这不是小文案问题。

如果 semantic docs进入：

```text
LLM mining
operator catalog
evidence
```

错误描述会直接影响因子生成。

修正 description并加入：

```text
description-to-semantic-spec consistency test
```

---

# 98. R35-P0-M10｜Panel Model 文件头的 Label 说明与当前实现也已漂移

文件顶部仍强调：

```text
label panel需要 caller预先 lag/embargo
```

而后面 `_forecast_loop` 已经实现：

```text
label_horizon maturity
```

这会让调用方不知道：

```text
是否应该二次 lag
```

必须统一。

---

# 99. R35-P0-M11｜Polars `ts_variance_ratio_slope` Current-NaN 语义疑似与 Pandas Reference 分叉

Pandas reference：

```text
trailing_contiguous_finite
current row NaN
=> empty
=> output NaN
```

当前 Polars callback：

```text
如果最后行 NaN
向前找最后 finite
再用此前连续块计算
```

相当于：

```text
current missing
→ stale backoff
```

这对 factor current-slot语义非常危险。

## 修复

Polars必须：

```text
if last input is non-finite:
    return NaN
```

再比较 contiguous suffix。

---

# 100. R35-P0-M12｜Polars Pairwise Rolling 用 `is_not_null()` 而不是 `is_finite()`

当前：

```python
both = y[c].is_not_null() & x[c].is_not_null()
```

这并不排除：

```text
NaN
+Inf
-Inf
```

而 Pandas reference大量使用：

```text
np.isfinite
```

因此：

```text
ts_market_liquidity_beta
ts_industry_liquidity_beta
```

等可能在 hostile input发生 backend divergence。

## 修复

明确 canonical policy：

```text
finite-only
```

然后 Polars：

```text
is_not_null & is_finite
```

或按 contract精确实现。

---

# 101. R35-P0-M13｜Model Timing Dict 存在重复 Key Block 风险

当前 `MODEL_TIMING_CONTRACTS` 中可以看到部分 family key在大字典里重复出现。

Python dict literal：

```text
后面的 key silently overwrite前面的
```

现在即使值相同，
以后一个 reviewer改前面、另一个 reviewer改后面：

```text
前面改动无效
```

非常危险。

## Gate

AST检查：

```text
duplicate dict literal key = FAIL
```

所有 canonical timing contract：

```text
exactly one authored record
```

---

# 102. R35-P0-M14｜用数据分布“猜是不是价格”不应是 Production Unit Authority

GARCH当前同时有：

```text
input_units={"x":"return"}
```

又有：

```text
sd(level)/sd(diff)>5
```

的 runtime heuristic来判断是否误传价格。

heuristic会：

```text
false positive
false negative
gap compression
```

Production应该：

```text
Typed Field/IR Unit Contract
```

直接阻止：

```text
price level -> return-only model
```

数值 heuristic只能：

```text
research warning / defense-in-depth
```

不能作为 semantic authority。

---

# 103. R35-P0-M15｜Strict Production Loader 与“模型 direct-use”目标需要重新对齐

当前 `RESEARCH_LOAD_MODULES` 把整批：

```text
ts_model.*
cross_section.panel_model
dmd
research_spectral
```

按 module级排除。

但同一 module内部又可能存在：

```text
少数 reviewed daily canonicals
```

如果未来要让其中一部分真正 production direct-use，
module级“整包 research”会很别扭。

## 建议

拆：

```text
production_model/
research_model/
```

或者：

```text
per-canonical registration manifest
```

不要由 module所在目录决定最终 authoring/admission。

---

# 104. 模型侧必须建立 `ModelOperatorContract`

建议：

```python
@dataclass(frozen=True)
class ModelOperatorContract:
    canonical: str
    role: str

    feature_params: tuple[str, ...]
    required_feature_count_min: int
    required_feature_count_max: int

    label_param: str | None
    label_horizon_param: str | None

    fit_cutoff_rule: str
    label_maturity_rule: str
    scaler_cutoff_rule: str
    hyperparam_cutoff_rule: str

    refit_policy: str
    missing_policy: str
    convergence_policy: str
    min_train_obs: int

    stateful: bool
    checkpoint_supported: bool

    cost_class: str
```

---

# 105. 模型侧不要全部作为普通 Primitive

模型 operator的核心语义不是：

```text
f(x)
```

而是：

```text
fit(history)
then score(current)
```

所以 IR需要显式：

```text
MODEL_FIT
MODEL_SCORE
```

或者复合 node。

这样 planner才知道：

```text
哪些 intermediate可共享
哪些需要 prior history
哪些不能 CSE across different cutoff
```

---

# 106. Model Fit State 可以跨 Outputs 共享

例如：

```text
GARCH params
PCA decomposition
rolling OLS beta
Kalman state
```

都应该成为：

```text
first-class shared intermediate
```

而不是隐藏在 operator Python函数里面。

---

# 107. 建立 ModelKernelGraph

例如：

```text
ReturnPanel
   ↓
GARCHFitState(window=120, cutoff=t-1)
   ├─ persistence
   ├─ h_t
   ├─ shock_z
   └─ next_vol
```

PCA：

```text
ReturnCrossSection
   ↓
RollingStandardization
   ↓
PCAState
   ├─ loading
   ├─ residual
   ├─ commonality
   └─ explained
```

---

# 108. ModelKernelGraph 与 R33 Unified QueryGraph 合并

R33不应只共享普通 expression subtree，
还要共享：

```text
semantic model intermediate
```

这是模型侧大幅提速的关键。

---

# 109. 编译/执行 Backend 最终分五类

```text
SQL_NATIVE
POLARS_NATIVE
NUMPY_BLAS
NUMBA_KERNEL
PYTHON_SPECIALIZED
```

不要把所有非 SQL都粗暴叫：

```text
pandas_numpy
```

否则 scheduler无法准确判断：

```text
GIL
threads
cost
memory
```

---

# 110. 每个 BackendKind 声明真实 GIL 行为

```python
ExecutionTraits(
    releases_gil,
    internal_threads,
    parallelizable_dimension,
    picklable_payload,
    memory_locality,
)
```

---

# 111. Numba 优先改造顺序

建议第一批：

```text
1. Kalman level/trend/beta
2. stateful rule language
3. PSAR / Supertrend / KAMA
4. directional-change / episode
5. AR recurrence / custom rolling loops
6. ElasticNet coordinate descent
7. PLS NIPALS
8. GARCH likelihood + variance recursion
9. complex custom rolling callbacks
```

---

# 112. 不要先 Numba PCA SVD

PCA最大问题是：

```text
每窗口重建 + 每窗口分解
```

先：

```text
rolling sufficient stats / decomposition reuse
```

再考虑外围 loop JIT。

---

# 113. Rolling Regression 建立统一 FastLinearWindowEngine

接口：

```python
fit_ols(...)
fit_ridge(...)
resid(...)
r2(...)
beta(...)
```

内部共享：

```text
rolling X'X
rolling X'y
rolling y'y
```

并同时服务：

```text
dynamic regression
HAR
liquidity beta
CAPM-like operators
```

---

# 114. FastLinearWindowEngine 必须处理 Missing Pattern

不能简单减旧/加新：

```text
如果不同 row的 feature missing pattern不同
```

需要：

```text
joint-valid observation accounting
```

或者按：

```text
valid-mask state
```

维护 sufficient stats。

---

# 115. Robust/Quantile 独立 Expensive Kernels

```text
Huber
Quantile LP
Expectile
```

不要强行塞进普通 rolling OLS fast lane。

---

# 116. Quantile LP 默认不进入大规模 Grammar

SciPy HiGHS每窗口 solve：

```text
非常贵
```

Lane B：

```text
explicit budget
```

---

# 117. Matrix Profile / RQA / HSIC / Kernel Granger 等同样 Budget-Aware

用户想扩大搜索空间没问题，
但：

```text
搜索空间大
```

不等于：

```text
每一轮都随机调用最贵 operator
```

给每 operator：

```text
estimated_cost_class
```

mining engine使用：

```text
cost-aware grammar
```

---

# 118. Mining 先便宜后昂贵

建议：

```text
Stage 1:
    cheap broad exploration

Stage 2:
    promising formulas
    allow expensive operators

Stage 3:
    high-value candidates
    model/advanced operators
```

能大幅提高实际吞吐。

---

# 119. 线程策略

## Thread Lane

```text
DuckDB
Polars
Arrow
NumPy BLAS
Numba nogil
```

## Process Lane

仅：

```text
remaining pure Python GIL-bound
```

---

# 120. 当前 Scheduler 要真正用 Process 的前提

现在 root：

```text
ctx不可 pickle
```

所以强制 thread。

如果 R35实现 Numba/native后：

```text
绝大多数 production hot path不需要 process
```

因此可以把“process-safe root”降为第二优先级。

---

# 121. 如果仍需要 Process，按 Block 而不是 Factor 传

坏：

```text
一个 factor一个 process task
大 DataFrame pickle
```

好：

```text
一个 FactorBlock / ModelBlock
共享 input BufferRef
worker-local compute
ResultRef
```

---

# 122. BufferRef 应支持 Shared Memory / mmap

R33已有 BufferRef概念。

扩展：

```text
SharedArrowBufferRef
MemmapNumpyRef
DuckDBRelationRef
PolarsLazyRef
```

Process worker只传：

```text
small descriptor
```

---

# 123. 不要多个进程同时各自打开同一大 Parquet

如果所有 worker：

```text
各读一遍
```

I/O会被放大。

正确：

```text
DA读取/缓存一次
→ shared buffer
→ process consumers
```

或者由 DuckDB本身并行读。

---

# 124. DataAccess 与 Compute 并行用 Pipeline，不是“所有东西同时开线程”

理想：

```text
Read Wave N+1
     ||
Compute Wave N
     ||
Write Wave N-1
```

资源 broker分别给：

```text
IO tokens
CPU tokens
write tokens
```

---

# 125. DataAccess 应尽量输出 Backend-Native Representation

如果下一段：

```text
DuckDB SQL
```

返回：

```text
DuckDB relation
```

不要先：

```text
Arrow -> Pandas -> DuckDB
```

如果下一段 Polars：

```text
Polars LazyFrame
```

如果 Numba：

```text
contiguous NumPy block
```

只在边界转换一次。

---

# 126. Model Kernel 的输入最好是 ndarray Block

不要在每 row：

```text
Series
DataFrame
```

对象层操作。

compile阶段：

```text
panel -> contiguous float64 block
```

然后整个 kernel一次处理。

---

# 127. ndarray Layout 也要规划

TS per-stock loop：

```text
asset-major contiguous
```

可能更好。

Cross-section per-date：

```text
date-major contiguous
```

可能更好。

Representation planner选择：

```text
transpose once
```

而不是每 operator transpose。

---

# 128. NumPy Memory Allocation 也是热点

模型 loops中大量：

```text
np.full
np.column_stack
np.concatenate
temporary design
```

改：

```text
scratch pool
preallocated buffers
views
```

---

# 129. `np.linalg.svd` / `lstsq` 的 BLAS Threads 需要 Broker 控制

小矩阵：

```text
多线程 BLAS反而慢
```

大矩阵：

```text
可适当多线程
```

cost model可按：

```text
matrix dimensions
```

选择 backend_threads。

---

# 130. 小模型大量并行时，BLAS=1

例如：

```text
1000个 5×5 / 120×5 solve
```

通常更适合：

```text
outer parallel
BLAS threads=1
```

---

# 131. 大 Cross-Section PCA 可允许 BLAS 多线程

例如：

```text
5000 stocks × 120 days
```

单个 SVD可能值得：

```text
BLAS threads=N
```

但 broker需要独占N token。

---

# 132. NUMA / CPU Affinity 暂不做第一阶段

先 profile。

只有：

```text
多 socket大内存机器
```

且 profiler显示 remote memory显著，
再做：

```text
NUMA-aware pinning
```

---

# 133. Full Fast Path 优先级建议

对大多数因子：

```text
DuckDB SQL / Polars Lazy
```

对 custom recurrence：

```text
Numba
```

对 matrix linear algebra：

```text
NumPy/LAPACK
```

对极少 residual Python：

```text
ProcessPool
```

---

# 134. 建立 `KernelBenchmarkRegistry`

每个 family保存：

```text
rows
assets
window
params
backend
cold_ms
warm_ms
memory
conversion_ms
parity
```

router不凭“估计 speedup=2.0”长期写死。

---

# 135. 自动选择 Numba / NumPy / Polars / DuckDB

例如：

```text
rows small
→ reference/native direct

rows large + stateful
→ Numba

SQL-emittable
→ DuckDB

cross-section expr-native
→ Polars/DuckDB

small linear rolling
→ FastLinearWindowEngine
```

---

# 136. Fast Kernel 失败自动回退必须分两类

### Capability mismatch

计划阶段就知道：

```text
不要选
```

### Unexpected runtime failure

production：

```text
记录
quarantine fast variant
reference fallback
```

前提：

```text
reference本身 certified
```

---

# 137. 不能 silently fallback 后还说“最快”

每次 batch报告：

```text
native_hit_rate
numba_hit_rate
reference_fallback_rate
conversion_bytes
```

---

# 138. 最终性能 KPI

不要只：

```text
factors/sec
```

完整：

```text
compile_ms
DA_plan_ms
scan_ms
bytes_read
PIT_join_ms
conversion_ms
kernel_ms
scheduler_ms
DQ_ms
write_ms
commit_ms
TimeToDurableCommit
peak_rss
CPU_utilization
```

---

# 139. 模型侧性能 KPI

额外：

```text
fit_count
shared_fit_count
SVD_count
lstsq_count
optimizer_call_count
iterations
nonconvergence_rate
```

这样可以证明：

```text
family multi-output
```

真的减少了重复 fit。

---

# 140. R35 测试体系：从“测试文件”升级为“实验矩阵”

建议每个 canonical自动组合：

```text
fixture
×
parameter case
×
backend
×
execution variant
×
batch mode
×
thread mode
```

不需要笛卡尔全爆炸，
由 RiskProfile裁剪。

---

# 141. Test Fixture Layers

## F0 tiny hand

```text
手算
```

## F1 normal synthetic

```text
普通市场形态
```

## F2 hostile numeric

```text
NaN/Inf/tie/outlier
```

## F3 PIT source

```text
真实时间字段
```

## F4 batch mixed

```text
多 operator
```

## F5 large perf

```text
性能
```

---

# 142. Model Oracle Library

tests-only optional dependencies可考虑：

```text
scikit-learn
statsmodels
arch（若你决定加入）
```

目的：

```text
独立 reference
```

不要进入 production dependency。

---

# 143. Test Oracle Version Pinning

external oracle：

```text
version
parameters
solver
tolerance
```

固定。

---

# 144. Golden 不应要求每个高级统计量“完全相同算法”

如果定义允许多个合法 estimator，
必须先明确：

```text
canonical定义的是哪一个 estimator
```

然后测试它。

不能用“论文名字一样”就比较。

---

# 145. 统计高级算子的 Synthetic Recovery Test

例如：

```text
AR(1) simulated data
→ AR coeff接近真实 phi

GARCH simulated
→ persistence合理恢复

PCA latent factor
→ first component recovery

Kalman simulated state
→ filtered state tracking
```

这是比只看无异常更有价值的 test。

---

# 146. Economic Sanity Tests

不是用收益判断算子正确，
而是检查定义级 invariant：

```text
volatility >= 0
probability ∈ [0,1]
R² upper bound
weight sum
rank range
entropy bounds
```

---

# 147. Monotonicity Tests

适合：

```text
winsorize
clip
rank
distance
threshold strength
```

---

# 148. Scale Invariance / Equivariance

例如：

```text
zscore(a*x+b) invariant
return scale
beta unit transformation
```

按 operator定义测试。

---

# 149. Permutation Invariance

对：

```text
cross-section
panel model
group
```

instrument column reorder后：

```text
按 symbol对齐结果一致
```

---

# 150. Time Translation Invariance

不依赖 calendar absolute date的纯 TS：

```text
平移 index日期
```

值应一致。

calendar/event类 N/A。

---

# 151. Causality Metamorphic Test

统一：

```text
future random perturbation
prefix output unchanged
```

这是全 TS/model production的重要 universal test。

---

# 152. Data Revision Test

PIT sources：

```text
revision after decision
```

过去输出不变。

---

# 153. Backend Metamorphic Test

不同 backend：

```text
不仅值一致
还要 invariant一致
```

---

# 154. Optimizer Mutation Test

故意关闭/开启：

```text
CSE
fusion
pushdown
window share
```

结果一致。

---

# 155. Numba Mutation Test

Numba fast kernel故意注入：

```text
off-by-one
skip missing
state reset
```

测试必须杀掉。

---

# 156. 全算子测试 Dashboard

建议生成：

```text
R35_OPERATOR_READINESS.parquet
```

每行：

```text
canonical
lane
risk
semantic
pit
params
backend
batch
stateful
performance
final_status
```

---

# 157. `DIRECT_USE_CERTIFIED` 只能由 Dashboard 自动算

例如：

```text
semantic PASS
pit PASS
param PASS
required backends PASS
role legal
```

才：

```text
DIRECT_USE_CERTIFIED
```

---

# 158. 模型测试 Dashboard 额外字段

```text
model_family
timing_explicit
future_perturb
label_maturity
oracle
convergence
min_train
cost_class
fast_kernel
```

---

# 159. 当前所有算子“逐个人工读一遍”不是最优最终方法

人工审计必要，
但 1400+ canonical长期维护不可能只靠人工。

最优方式：

```text
人工定义 contract
+
自动生成测试 obligation
+
独立 oracle/property/fuzz
+
每 commit影响重跑
```

---

# 160. 只有这样才能回答“以后新算子是不是也没问题”

因为：

```text
新增 operator
```

如果缺：

```text
typed contract
tests
evidence
```

CI直接拒绝。

---

# 161. R35 P0 Hard Gates

```text
R35_MODEL_TIMING_EXPLICIT_FOR_PRODUCTION
R35_ZERO_DUPLICATE_MODEL_TIMING_KEYS
R35_GARCH_TIMING_IMPL_CONTRACT_MATCH
R35_HAR_TIMING_IMPL_CONTRACT_MATCH

R35_PANEL_MODEL_REQUIRED_FEATURES_EXPLICIT
R35_MARKET_STATE_REQUIRED_CONTRACT
R35_PCA_MIN_HISTORY_EXPLICIT
R35_PCA_PERMUTATION_STABLE
R35_MODEL_DOC_SEMANTIC_MATCH

R35_POLARS_VARIANCE_RATIO_CURRENT_NAN_PARITY
R35_POLARS_PAIRWISE_FINITE_PARITY

R35_NUMBA_KERNELS_REFERENCE_CERTIFIED
R35_ZERO_UNCERTIFIED_FAST_KERNEL_PRODUCTION

R35_THREAD_PROCESS_REFERENCE_PARITY
R35_NO_UNCONTROLLED_NESTED_THREADS

R35_FE_DA_SINGLE_SOURCE_PLANNER
R35_REAL_EXECUTABLE_READ_WAVES
R35_REAL_EXECUTABLE_PHYSICAL_STAGES
R35_ZERO_HOTPATH_TEMP_PARQUET

R35_ALL_DIRECT_USE_OPERATORS_TEST_OBLIGATIONS_PASS
R35_CURRENT_HEAD_FULL_SUITE_PASS
```

---

# 162. R35 Performance Gates

```text
R35_KALMAN_NUMBA_SPEEDUP_MEASURED
R35_STATEFUL_NUMBA_SPEEDUP_MEASURED
R35_LINEAR_ROLLING_FAST_ENGINE_SPEEDUP
R35_MODEL_FAMILY_SHARED_FIT_EFFECTIVE

R35_SMALL_BATCH_NO_REGRESSION
R35_100_FACTOR_TTDC
R35_1000_FACTOR_TTDC
R35_MIXED_SOURCE_TTDC

R35_SCAN_REUSE_RATE
R35_NATIVE_HIT_RATE
R35_CONVERSION_BYTES
R35_PEAK_RSS
```

---

# 163. 不要给 Speedup 写死一个数字

验收应该：

```text
fast variant >= reference performance
```

并设置：

```text
meaningful speedup threshold
```

按 family benchmark校准。

如果 Numba：

```text
只快 3%
```

但增加很多维护成本，
不一定值得 production。

---

# 164. 最重要的 Benchmark Workloads

## B1 Small Cheap

```text
20 factors
100 factors
```

验证 scheduler overhead。

## B2 Daily 1000

```text
price-volume
A-share
5 years
```

## B3 Mixed 1000

```text
price
fundamental
event
holder
cross-section
stateful
```

## B4 Model 100

```text
AR
rolling regression
PCA
Kalman
HAR
```

## B5 Expensive Model 20

```text
GARCH
quantile LP
DMD
matrix profile
```

## B6 Minute→Daily 500

## B7 Incremental one-day 1000

## B8 Historical correction

## B9 Materialize/readback

---

# 165. Model Benchmark 需要单独统计 `fits/sec`

例如 rolling GARCH：

```text
rows × assets
```

不能简单用：

```text
factors/sec
```

看不出重复 fit。

---

# 166. 线程 / 进程 Benchmark Matrix

```text
serial
threads 2/4/8
process 2/4
hybrid
```

同时限制：

```text
BLAS threads
```

---

# 167. Numba Benchmark Matrix

```text
reference Python
Numba cold
Numba warm
Numba warm + thread blocks
```

---

# 168. DataAccess Benchmark

```text
same source repeated factors
multi-source
PIT joins
remote files
minute aggregation
```

---

# 169. 端到端最终只看 TTDC

```text
request received
→ generation committed
```

期间任何：

```text
wrong result
```

该 run不进入性能排名。

---

# 170. 建议 R35 分六个实施 Workstream

## W1 Model Correctness

```text
timing
label
oracle
missing
convergence
```

## W2 Kernel Acceleration

```text
Numba
rolling stats
family multi-output
```

## W3 Execution Runtime

```text
threads
process
nested thread control
```

## W4 FE×DA

```text
R33 completion
```

## W5 Test Factory

```text
obligation matrix
property/fuzz
```

## W6 Benchmark / Release

```text
current-head evidence
TTDC
```

---

# 171. 推荐实施优先级

第一阶段：

```text
修模型具体语义 bug
+
建立 current-head test matrix
```

第二阶段：

```text
R33 FE×DA
```

第三阶段：

```text
Numba Kalman/stateful
Rolling Linear Engine
Model Family Multi-Output
```

第四阶段：

```text
process residual lane
更高级硬件优化
```

---

# 172. 为什么不是先做多进程

因为当前最大的收益来自：

```text
重复工作消除
```

而不是：

```text
把重复工作分给更多 CPU
```

---

# 173. 为什么 Numba 值得做，但不是万能药

Numba对：

```text
纯 Python numeric recurrence
```

非常合适。

对：

```text
数据库
SVD
SciPy optimizer
```

不是主解。

所以 R35的正确做法是：

```text
按 KernelShape自动选择 acceleration strategy
```

---

# 174. KernelShape

建议：

```python
KernelShape(
    elementwise,
    rolling_reduction,
    recursive_state,
    small_linear_model,
    matrix_decomposition,
    iterative_optimization,
    cross_section,
    session_aggregation,
)
```

---

# 175. Acceleration Strategy Mapping

```text
elementwise
→ DuckDB/Polars

rolling_reduction
→ native window / sufficient stats / bottleneck

recursive_state
→ Numba

small_linear_model
→ rolling Gram + NumPy/Numba

matrix_decomposition
→ LAPACK + reuse

iterative_optimization
→ compiled objective + budget lane

cross_section
→ vector matrix / SQL/Polars

session_aggregation
→ DataAccess pushdown
```

---

# 176. 最终 Direct-Use 的定义

一个 retained operator“可以直接用”，不是：

```text
OperatorRegistry.get()不为 None
```

而是：

```text
1. role明确
2. input contract明确
3. parameter domain明确
4. formula/reference正确
5. PIT正确
6. missing/edge正确
7. source正确
8. state正确
9. 默认搜索暴露合理
10. 至少一条 certified execution path
```

---

# 177. 对模型的 Direct-Use 定义更严格

再加：

```text
11. explicit ModelTimingContract
12. label maturity
13. scaler cutoff
14. hyperparam cutoff
15. convergence
16. min train observations
17. walk-forward oracle
```

---

# 178. 最终模型不一定全部进入 Daily Mining

这不是失败。

例如：

```text
GARCH
quantile LP
matrix profile
```

正确状态可以是：

```text
EXPENSIVE_CERTIFIED_ALPHA
```

“能直接用”：

```text
是
```

“默认随机搜”：

```text
否
```

---

# 179. 最终测试体系必须可以自动判定

目标输出：

```text
1436 canonicals
1436 assigned lanes
1436 test obligations resolved

N direct-use certified
N expensive certified
N intermediate certified
N diagnostic
N research
N deleted
0 unclassified
```

---

# 180. 当前最需要避免的误区

不要继续追求：

```text
所有算子标签 production
```

而是追求：

```text
所有有价值算子都有准确且可执行的角色
```

---

# 181. R35 当前 HEAD 新问题优先修复清单

```text
P0-M01 GARCH forecast timing
P0-M02 GARCH persistence timing
P0-M03 GJR timing
P0-M04 HAR fit cutoff semantics
P0-M05 zero-feature panel forecast
P0-M06 market_state required
P0-M07 PCA min history
P0-M08 PCA tie sign identity
P0-M09 PCA description drift
P0-M10 label doc drift
P0-M11 Polars VR current-NaN
P0-M12 Polars pairwise finite mask
P0-M13 duplicate timing keys
P0-M14 input unit heuristic authority
P0-M15 mixed research/production model loader architecture
```

这些要在大范围性能改造前先收口。

---

# 182. 执行 AI 的完整整改顺序

以下顺序不要打乱。

---

# Phase A｜先修模型 Truth

## A1

清理：

```text
MODEL_TIMING_CONTRACTS duplicate keys
```

## A2

禁止 production：

```text
default generated timing contract
```

## A3

逐模型 family写 explicit contract。

## A4

修：

```text
GARCH/HAR timing contract vs implementation
```

## A5

修：

```text
panel required features
market_state required
PCA min_history
PCA sign orientation
docs semantic drift
```

## A6

修 Polars model parity：

```text
current NaN
is_finite
Inf
```

---

# Phase B｜模型 Oracle

为：

```text
PCA
PCR
PLS
ElasticNet
OLS/Ridge
Huber
Quantile
Expectile
AR
Kalman
GARCH/GJR
HAR
```

建立 independent oracle。

---

# Phase C｜全模型 Causality

统一生成：

```text
future perturbation
label poison
feature poison
multi-horizon maturity
```

---

# Phase D｜全模型 Parameter Domain

测试：

```text
window
components
order
alpha
l1_ratio
q
r
noise
regimes
experts
label horizon
```

---

# Phase E｜全模型 Readiness Lane

每个 canonical决策：

```text
FAST_NATIVE_ALPHA
EXPENSIVE_CERTIFIED_ALPHA
STATE_CONDITION_EVENT
MODEL_FEATURE_SCORE
DIAGNOSTIC_RESEARCH
DELETE
```

不得 unclassified。

---

# Phase F｜Numba Layer

先：

```text
Kalman
stateful recurrence
technical recurrence
ElasticNet
PLS
GARCH recurrence
```

每个：

```text
reference parity
benchmark
```

---

# Phase G｜Rolling Linear Engine

统一改：

```text
OLS
Ridge
HAR
CAPM-like
liquidity beta
```

用 rolling sufficient statistics。

---

# Phase H｜Model Family Intermediate CSE

```text
PCAState
GARCHFitState
KalmanState
RollingLinearState
```

成为 physical intermediate。

---

# Phase I｜完成 R33 FE×DA

必须完成：

```text
single DA planning authority
real multi-source DataRequest
real executable SourceScan
real ReadWave
real stage execution
zero-copy relation/lazy/buffer
host-global broker
zero temp parquet hot path
```

---

# Phase J｜Runtime Parallelism

完成：

```text
ExecutionTraits
threadpoolctl
Numba nogil lane
optional Process ExecutionCapsule
```

---

# Phase K｜Test Factory

自动生成：

```text
canonical × obligations
```

---

# Phase L｜Release Benchmark

current HEAD：

```text
correctness
+
TTDC
```

---

# 183. 建议新增代码模块

```text
factor_engine/cleaned_operators/model_contract.py
factor_engine/cleaned_operators/model_evidence.py

factor_engine/backend/numba_kernels/
factor_engine/backend/numba_kernel_registry.py

factor_engine/backend/fast_linear_window.py
factor_engine/backend/model_family_kernels.py

factor_engine/runtime/execution_traits.py
factor_engine/runtime/execution_capsule.py

factor_engine/tests/factory/operator_obligations.py
factor_engine/tests/factory/model_obligations.py
```

注意：

```text
不要重复现有 contract事实
```

这些模块要消费 R34统一 CanonicalContract，
不是另起一套冲突 metadata。

---

# 184. 建议 Evidence 目录

```text
factor_engine/docs/evidence/r35/
```

至少：

```text
R35_HEAD.json

R35_MODEL_CANONICAL_INVENTORY.csv
R35_MODEL_TIMING_CONTRACTS.csv
R35_MODEL_ORACLE_RESULTS.csv
R35_MODEL_FUTURE_PERTURBATION.csv
R35_MODEL_PARAMETER_DOMAIN.csv
R35_MODEL_CONVERGENCE.csv

R35_NUMBA_KERNEL_PARITY.csv
R35_NUMBA_KERNEL_BENCHMARK.csv

R35_THREAD_PROCESS_PARITY.csv
R35_THREADING_BENCHMARK.csv

R35_FE_DA_EXECUTION_PATH.json
R35_FE_DA_SCAN_REUSE.csv

R35_OPERATOR_READINESS.csv
R35_OPERATOR_TEST_OBLIGATIONS.csv

R35_BATCH_REFERENCE_PARITY.csv
R35_PERFORMANCE_BENCHMARK.csv

R35_FINAL_ACCEPTANCE_REPORT.md
```

---

# 185. 模型 Inventory 必须完整

```text
canonical
model_family
lane
production_target
explicit_timing
label_based
stateful
cost_class
reference_backend
fast_backend
```

最终：

```text
model canonical unclassified = 0
```

---

# 186. Test Obligation 自动生成规则示例

## Pure elementwise

```text
semantic
edge
param
backend
batch
```

## Rolling TS

加：

```text
future perturb
chunk
incremental
```

## Source-dependent

加：

```text
source PIT
revision
```

## Model

加：

```text
timing
label maturity
oracle
convergence
```

## Stateful model

再加：

```text
checkpoint
```

---

# 187. 允许 N/A，但必须说明

例如：

```text
elementwise source PIT = N/A
```

合法。

不允许：

```text
没测试
→ N/A
```

---

# 188. 全量 Test Result Schema

```text
canonical
test_dimension
case_id
backend
execution_variant
param_case
fixture
status
error
runtime_ms
artifact_hash
commit_sha
```

---

# 189. 测试结果用 Parquet，不要只 CSV

全量 case可能非常多。

```text
Parquet = machine truth
CSV = summary export
```

---

# 190. Release Gate 直接查询 Parquet Ledger

不要再：

```text
从文件名猜测试是否存在
```

---

# 191. 推荐的 Nightly 规模

可以从：

```text
1436 canonicals
```

自动生成几万甚至十几万 case。

不需要每个 PR全部跑。

---

# 192. PR Test Selection

ChangeImpactDAG：

```text
改 operator
→ operator tests

改 rolling core
→ all dependents

改 DataAccess PIT
→ all source-dependent

改 backend emitter
→ affected backend families
```

---

# 193. Release Test

release必须：

```text
full matrix
```

不能只 change impact。

---

# 194. 测试随机化

property/fuzz：

```text
固定 seed corpus
+
每晚新的 deterministic date-seed
```

失败 case固化。

---

# 195. 性能 Test 不得污染 Correctness Test

分：

```text
correctness suite
performance suite
```

性能超时不能让 correctness被 skip。

---

# 196. 生产前必须真的跑的 Full Suite

建议最终提供：

```bash
python -m scripts.generate_operator_test_obligations
pytest -m "not perf and not slow"

python -m scripts.run_semantic_goldens
python -m scripts.run_parameter_domain_matrix
python -m scripts.run_future_perturbation_matrix
python -m scripts.run_source_pit_matrix
python -m scripts.run_model_matrix
python -m scripts.run_backend_parity_matrix
python -m scripts.run_stateful_incremental_matrix
python -m scripts.run_batch_reference_matrix

pytest -m perf
pytest -m slow

python -m scripts.verify_r33
python -m scripts.verify_r34
python -m scripts.verify_r35
```

---

# 197. 当前不能说“测试都做全了”

R35 final report必须非常诚实。

在完成上述 matrix前：

```text
TEST COVERAGE STATUS = INCOMPLETE
```

不是因为当前测试少，
而是：

```text
测试证明的维度还没有覆盖到用户要求的最终强度。
```

---

# 198. 是否需要再人工审每个 operator

需要一次最终 inventory review，
但后续不能依赖纯人工。

推荐：

```text
人工 review：
    semantic contract
    role
    expected behavior

自动：
    exhaustive execution
    property
    oracle
    fuzz
    backend
    batch
```

---

# 199. 全算子“真实直接用”最终 Hard Gate

```text
R35_ZERO_UNCLASSIFIED_CANONICALS

R35_ALL_RETAINED_HAVE_ROLE
R35_ALL_DIRECT_USE_HAVE_TYPED_SIGNATURE
R35_ALL_DIRECT_USE_HAVE_SEMANTIC_EVIDENCE
R35_ALL_DIRECT_USE_HAVE_PARAM_EVIDENCE
R35_ALL_DIRECT_USE_HAVE_PIT_EVIDENCE
R35_ALL_DIRECT_USE_HAVE_EXECUTION_PATH

R35_ALL_MODEL_DIRECT_USE_EXPLICIT_TIMING
R35_ALL_MODEL_DIRECT_USE_ORACLE
R35_ALL_MODEL_DIRECT_USE_FUTURE_PERTURBATION

R35_ZERO_KNOWN_BACKEND_PARITY_DRIFT

R35_RUN_MANY_EQUALS_RUN_ONE
R35_OPTIMIZED_EQUALS_REFERENCE

R35_FE_DA_E2E_GOLDEN
R35_MATERIALIZED_READBACK_PARITY
```

---

# 200. Speed Hard Gate 不能独立于 Correctness

逻辑：

```text
if correctness != PASS:
    speed benchmark = INVALID
```

---

# 201. 推荐最终 Route Policy

```python
def choose_execution(region):
    eligible = certification.filter(region)

    if not eligible:
        fail_closed()

    return min(
        eligible,
        key=predicted_time_to_durable_commit
    )
```

不是：

```text
choose fastest
then hope it is correct
```

---

# 202. 未来新增 Operator 的开发流程

```text
1. semantic spec
2. typed signature
3. role/lane
4. reference kernel
5. oracle/property tests
6. PIT
7. param domain
8. optional fast backend
9. parity
10. benchmark
11. production admission
```

缺任一步：

```text
not direct-use
```

---

# 203. 未来新增 Numba Kernel 的开发流程

```text
1. preserve reference
2. port numeric loop
3. fastmath=False
4. nogil
5. cache
6. parity hostile fixtures
7. benchmark cold/warm
8. execution traits
9. admission
```

---

# 204. 未来新增 Model 的开发流程

额外：

```text
timing
label
refit
scaler
hyperparam
convergence
```

---

# 205. 三个非常重要的“不要做”

## 不要 1

```text
为了“全部 direct-use”
把 research model强行 production。
```

## 不要 2

```text
为了“最快”
给所有 Python loop开 multiprocessing。
```

## 不要 3

```text
为了“有 Numba”
把本来 native SQL/Polars/BLAS的东西搬回 Numba。
```

---

# 206. 对你的实际目标最合理的最终配置

## 高频普通因子

```text
DuckDB / Polars
```

## 自定义时序递归

```text
Numba
```

## 线性滚动模型

```text
FastLinearWindowEngine
```

## 大矩阵分解

```text
NumPy/LAPACK + shared decomposition
```

## 高成本优化模型

```text
Budget Lane
```

## Data IO/PIT

```text
DataAccess
```

## 写入

```text
FactorBlock + atomic generation
```

---

# 207. 预期最大的性能收益来源排序

在不看真实 profiler前，我建议优先级：

```text
1. R33去重复 scan/materialization/conversion
2. Model family共享 fit/decomposition
3. rolling linear sufficient stats
4. Numba递归/loop kernel
5. batch native fusion
6. representation/layout
7. thread tuning
8. process residual lane
9. low-level native extension/GPU
```

---

# 208. 为什么这个顺序最合理

前四项减少：

```text
实际工作量
```

后几项只是提高：

```text
同样工作量的执行速度
```

减少工作量通常收益更大。

---

# 209. 最终完成状态应该长这样

```text
Correctness:
    100% required obligations resolved

Model:
    100% production model explicit timing
    100% required oracle/future perturbation

Execution:
    0 uncertified fast path
    0 unintended Pandas fallback

Batch:
    run_many == run_one
    optimized == reference

FE×DA:
    one planning authority
    one source read per compatible scope
    zero unnecessary materialization

Performance:
    calibrated automatic router
    TTDC benchmark current HEAD

Release:
    current HEAD full suite green
```

---

# 210. 最终报告不要写“全部算子 production”

应该写：

```text
All retained operators have a reviewed and executable role.
All operators exposed for direct production use are fully certified.
```

这比：

```text
all production
```

更准确。

---

# 211. R35 Definition of Done

只有下面全部完成，才算 R35结束：

```text
[ ] 当前 HEAD模型 inventory完整
[ ] 所有 production model explicit timing
[ ] GARCH/HAR contract一致
[ ] panel model required inputs contract化
[ ] PCA min-history/sign/docs修复
[ ] Polars model parity bug修复

[ ] Model oracle全量
[ ] Future perturbation全量
[ ] Model parameter-domain全量
[ ] Convergence全量

[ ] Numba第一批 kernel完成并认证
[ ] rolling linear fast engine完成
[ ] family multi-output完成

[ ] R33 FE×DA执行层完成
[ ] process/thread真实策略完成
[ ] nested thread统一治理

[ ] operator obligation matrix完成
[ ] latest-head full suite完成
[ ] 100/1000/mixed/model benchmarks完成

[ ] R33 PASS
[ ] R34 PASS
[ ] R35 PASS
```

---

# 212. 最终验收问答

### “模型算子都处理好了吗？”

当前：

```text
没有全部处理到 production direct-use。
```

R35完成后：

```text
每个模型要么 certified direct-use，
要么明确 expensive/model/research lane。
```

### “FactorEngine 和 DataAccess 配合好了吗？”

当前：

```text
接口方向正确，但物理执行融合没有完成。
```

R33完成后才算真正融合。

### “Numba有用吗？”

```text
非常有用，但只对适合的 loop/recurrence kernel。
```

### “现在支持了吗？”

```text
optional dependency / architecture aware，
但还不是系统性 production Numba layer。
```

### “多线程需要吗？”

```text
需要，native/nogil主力。
```

### “多进程需要吗？”

```text
只做 residual GIL-bound lane，不是主路线。
```

### “测试都做全了吗？”

```text
没有。
历史测试很多，但不足以证明你要求的最终强度。
```

### “是不是还应该补大量测试？”

```text
是。
但重点不是无脑增加测试数量，
而是建立 operator obligation matrix，
自动证明每个 operator需要证明的所有维度。
```

---

# 213. 最终执行指令

执行整改 AI 应直接以本文件作为任务说明：

```text
先修具体模型/Backend语义问题；
再建立全模型 oracle + causality + parameter test；
再完成 Numba / rolling fast engine / model shared intermediates；
同步完成 R33 FE×DA；
最后用全量 obligation matrix + current-head benchmark决定真正 readiness。
```

不要在：

```text
“代码都写了”
```

的时候停。

必须停在：

```text
“当前 HEAD 的证据已经证明它们正确，而且 fastest eligible route 的 benchmark 也真实通过”
```

这里。
