# 日分位收益：请求内复用、口径及内存边界

## 解决的问题与模块

同一 CPU 请求中的 daily/full/spread/profile/daily-monotonicity 指标原先会
重复分箱和聚合。新增 `runtime/daily_quantile_cache.py`，统一缓存不可变
`DailyQuantileReturnArtifact`；`registry_adapters.py` 负责 typed 输入验证和
指标投影，`runtime/evaluator.py` 只负责请求绑定。不创建跨请求全局数据缓存。

同一输入及同一 Q/min_assets 的小型回归请求同时评估六类指标，实测
kernel build 从 7 次减少为 1 次。**调用次数不等于端到端速度倍数**；
全市场批量 A/B 和最终源码实数/CUDA资格需独立测量，不能据此写“7倍更快”。

## 数学定义不变

每个日期和因子，先用有效且有限的因子值按 QE 原有线性分位边界分箱。
至少需要 Q 个有限因子值。边界相等值仍使用 `max` tie policy
（`searchsorted(..., side="right")`），不是随机拆分同值股票。
因子 mask 决定成员资格；label mask 和有限 label 决定可聚合的收益样本。
对每个桶：

$$c_{tqf}=\sum_i 1[i\in q_{tf},\ \mathrm{label}_{ti}\text{有效}],\quad
R_{tqf}=\frac{\sum_{i\in q_{tf},\ \mathrm{label}_{ti}\text{有效}} y_{ti}}{c_{tqf}}.$$

当 $c_{tqf}<\mathrm{min\_assets}$ 时，$R_{tqf}=\mathrm{NaN}$，但保留真实 count。
完整分位均值按**各桶自己的有效日期**聚合；spread 则按共同有效日期：

$$S_f=\frac{\sum_{t\in D_f}(R_{t,Q-1,f}-R_{t,0,f})}{|D_f|},\quad
D_f=\{t:\ R_{t,Q-1,f}-R_{t,0,f}\text{有限}\}.$$

spread 的有效日期数须达到 `min_periods`。不能用“分别取两个桶的均值后相减”
代替，因为缺失日期集合可能不同。缓存中不含 min_periods：该参数控制投影，
不改变每日分箱。profile/window 的缓存还绑定实际输入及聚合参数。

## 参数与使用

direct daily/full/spread/daily-monotonicity 的公开默认仍是 Q=5/min_assets=10。
`quantile_builder_parameters` 供消费 QuantileReturnArtifact 的 profile 指标使用；
不能把它静默改成所有 direct 指标的默认值。显式 metric 参数优先，
不同 Q/min_assets 保持独立结果，不受请求中的指标顺序影响。

```python
from quant_evaluator.runtime.evaluator import evaluate
bundle = evaluate(batch, labels, backend="cpu", metrics=(
    "quantile_returns_daily", "quantile_returns_full", "quantile_spread",
    "quantile_monotonicity", "daily_quantile_monotonicity_series",
    "daily_quantile_monotonicity_rate",
))
daily = bundle.artifacts["quantile_returns_daily"].values  # T,Q,F
full = bundle.artifacts["quantile_returns_full"].values    # Q,F
```

此优化是 CPU 请求内复用；不是新 GPU 后端，也不自行重新认证 public `auto`。
CPU 内核原有 `use_numba=True/False` 显式选项保持，安装了 Numba 不等于
已证明最快。外层 `backend="auto"/"cpu"/"cuda"` 的选择资格另行管理。

## 输入隔离与稳定身份

内部 cache 强持有确切 FactorBatch/LabelBundle 对象，并绑定值、mask、轴、
factor IDs、label 内容身份、Q/min_assets/tie policy。`value_hash=None` 不能
被视为两份因子值相同；同轴但值或 mask 不同的输入不能复用。
typed 消费者检查形状、轴、来源绑定、producer 版本、tie policy、有效 mask。
禁止通过 `metric_parameters` 注入 `daily_quantile_artifact` 或私有绑定参数。

内部 process-local binding **不是持久内容证书**。普通 public builder 默认不带
内存对象 ID；public 输出剥离内部绑定，保持可序列化、稳定的正式证据。
剥离后的 public artifact 不能冒充同请求内部预计算结果回注；这是内部优化接口，
不是跨进程 artifact 准入 API。不会为每个指标重新哈希 GB 级输入数组。

## 内存与验证范围

默认 LRU 最多 64 entries、256 MiB 保留数组预算；计入 values/counts/mask、
valid-day provenance 数组以及非根输入的值/mask/轴数组，共享非根数组保守重复计数。
根输入本就由请求持有，不重复收费。entry 上限也限制小面板的 Python 元数据开销。
这不是 whole-process RSS 上限，也不是构建内核的临时数组峰值证书。

超预算 entry 仍返回正确计算结果但不保留，LRU 淘汰不改变指标定义。
零字节预算禁用保留；非法预算或分箱参数必须拒绝，warm hit 也不能绕过校验。
已覆盖请求内一次构建、跨请求隔离、参数分键、指标排列、输入串用、typed 反例、
有效 mask/producer/tie、LRU 命中刷新、字节/entry 上限和零预算重算。
2026-10-03 当前源码的三份新增 cache 测试独立复跑 **45 passed**；连同 CPU
benchmark 的边界 smoke 共 **54 passed**。广泛指标/auto/分位边界回归为
**1624 passed、24 skipped、1 deselected**（149.54 秒）；排除了性能报告测试。
这些统计存在重叠，不能相加。skipped 涉及需要专用 exposure/generalization/
portfolio/calendar 输入的指标，不代表那些指标已经验证。此结果不是性能证书。

尚需全市场多因子 A/B、当前源码实数 source/CUDA 资格；不能宣称全部指标无 bug/最快。
