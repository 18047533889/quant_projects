# Source 批量 Rank IC 汇总：口径、稳定性与 GPU 内核

## 支持范围与调用

Source 批量入口新增支持 rank_ic_positive_ratio、recent_3m_rank_ic、
rolling_rank_ic_ir 三个标量指标。CPU 与显式 cuda_strict 均可使用；
GPU 同批请求复用同一个每日 Spearman 结果及有限日期计数，不重新读因子、
不重复计算同一门槛的 Spearman。回归测试通过实际 CUDA 运算计数验证。

```python
result = evaluate_factor_source_batch(
    source, labels,
    metrics=("rank_ic_positive_ratio", "recent_3m_rank_ic", "rolling_rank_ic_ir"),
    backend="cuda_strict", max_tile_size=16,
)
```

调用者负责关闭 source；tile cap 不是固定执行宽度，资源与 source 合同仍需
满足。Source 入口没有 metric_parameters，本组三项使用默认门槛和窗口。
低层滚动函数及通用 evaluate 的相应参数通道可以选择 window/min_periods；
正整数、排除 Python/NumPy 布尔值，允许 NumPy 整数；min_periods > window
合法但无合格窗口，返回 NaN，不偷偷降低门槛。

新增实现集合 SOURCE_SUMMARY_METRICS 与 SOURCE_AUTO_METRICS 分离。
显式 GPU 能计算不代表 auto 已获得该组合的当前真实性能资格；本组三项
没有新的真实 COS 资格报告。默认自动选择的使用与证据边界见
SOURCE_DEFAULT_PROFILE_PROVIDER_20261004.md，不能沿用旧静态路由冒充资格。

## 数学口径

令 c[t,f] 为每日成对有限资产样本上的平均并列秩 Spearman IC；默认至少
20 个有效资产。令 D[f] 为全部原始时间位置中有限 c 的数量。

```math
D_f=\sum_t\mathbf{1}\{c_{t,f}\text{ finite}\},\qquad
P_f=\frac{\sum_t\mathbf{1}\{c_{t,f}\text{ finite and }c_{t,f}>0\}}{D_f}.
```

rank_ic_positive_ratio 为 P，D < 20 时 NaN；零 IC 不算正值。
recent_3m_rank_ic 是最后 63 个原始时间位置内有限 IC 的均值，另外要求
全历史 D >= 20。不是最后 63 个有限值，不按自然月重分组，尾部全缺失时
仍为 NaN。名称中的“三个月”仅是 63 位置的近似。

对截至 t 的最近 60 个原始位置，跳过非有限 IC，记有效集 W[t,f]：

```math
n_{t,f}=|W_{t,f}|,\quad
\mu_{t,f}=\frac{1}{n_{t,f}}\sum_{u\in W_{t,f}}c_{u,f},\quad
s_{t,f}=\sqrt{\frac{\sum_{u\in W_{t,f}}(c_{u,f}-\mu_{t,f})^2}{n_{t,f}-1}},
\quad I_{t,f}=\frac{\mu_{t,f}}{s_{t,f}}.
```

只有 n >= 20 且 s > 1e-12 的窗口保留 I。rolling_rank_ic_ir 是所有合格
I 的均值，没有合格窗口则 NaN；采用样本标准差 ddof=1，不年化。
三项 observation_counts 均记录全历史 D，包括结果为 NaN 的因子，
不能把计数改成标量输出的 0/1，也不能把缺失日期作为有效观察。

## 稳定性与内存

旧前缀平方和减 n*mean^2 会在常量、近常量与较长历史上灾难性抵消，
出现约七千万的虚假 IR，或把真实近常量方差变成零。CPU 改为逐窗中心化，
按因子和时间分块；65,536 是目标块元素数，不是全部临时数组的字节上限。
多个数组会放大临时空间，超长窗口仍至少需要一个窗口的工作空间。
GPU 使用独立 rolling_ic_statistics.py 原生 CUDA RawKernel：每线程处理
一个时间/因子窗口，两遍计算均值与中心化平方和，int64 地址与参数，
只保存 O(TF) 输出，没有 T×window×F 窗口立方；非连续输入明确规整为
float64。缓存编译对象，但首次编译成本不包含在热运行微基准中。

Root 独立三次微基准：T=2586/F=48/window=60、3.5% NaN、CPU 单线程，
GPU 预热并同步；稳定 CPU 中位 43.942 ms，融合 CUDA 中位 0.308 ms。
不含主机传输、首次编译、COS 读取和 Spearman，不是端到端资格或普遍最快
证明。旧错误前缀公式更快，不能作为正确性 oracle；稳定 CuPy 窗口重算
版本曾退化至约 50 ms，融合内核用于消除该退化。

## 同轮修复：Rank IC 自相关 decay

这是 evaluator 的 rank_ic_decay，不是 optimizer 的因子 decay refinement，
也不是需要多 horizon 标签的 ic_decay。本轮未将它新增到 source 指标目录；
这里修复其原有 CPU/GPU 计算路径。
其公开函数默认 horizons=(1,5,10,20)，按原始位置取滞后，不插补 NaN。
每个滞后只用左右共同有限的样本，至少两个配对且两侧方差严格正；
结果是有定义滞后相关性的均值，同时全历史有限 IC 数须 >=20。
全部滞后无定义时返回 NaN，不把常量相关性写成 0 或 1。

旧 n*sum(x*y)-sum(x)*sum(y) 形式会灾难性抵消，CPU/GPU 在近常量
100k 位置上出现小于 -1 的相关性。简单“减总体均值”仍会把非二进制
常量 0.1/0.3/0.996908 的均值末位误差当成方差，返回假相关性 1。
现分别用左右第一个真实有限配对值作独立锚点，先取差，再计算均值与
中心化协方差/平方和，不靠结果裁剪掩盖错误。

```math
u_j=x_j-a,\quad v_j=y_j-b,\qquad
\rho_h=\frac{\sum_j(u_j-\bar u)(v_j-\bar v)}
{\sqrt{\sum_j(u_j-\bar u)^2\sum_j(v_j-\bar v)^2}}.
```

左右必须独立锚定：只用左锚点会把右侧常量减成非精确常量，仍有虚假
方差。持久反例是 130 位置、前 30 个左值 0.1+k*1e-8、中间缺失、末
30 个右值 0.3，horizon=100；正确结果 NaN，单锚点版却返回 0。
GPU 回归以原始配对值上的 SciPy pearsonr 作独立 oracle，并明确常量无定义；
不复制生产锚点算法来证明生产算法正确。

## 独立复跑与证据边界

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q \
  quant_evaluator/tests/test_predictive_numeric_boundaries_oct04.py \
  quant_evaluator/tests/test_ic_decay_horizons.py \
  quant_evaluator/tests/test_source_spearman_summary_batch_oct04.py \
  quant_evaluator/tests/test_gpu_metric_expansion.py \
  quant_evaluator/tests/test_rank_ic_coefficient_parity.py \
  quant_evaluator/tests/test_source_metric_catalog_oct04.py \
  quant_evaluator/tests/test_gpu_rolling_ic_statistics_oct04.py \
  quant_evaluator/tests/test_gpu_rank_ic_decay_stability_oct04.py
```

Root 在 server-c 实际复跑：73 passed、7 个空窗口/空均值警告，2.96 秒，
无跳过、退出码 0，包括真实 CUDA 内核执行。没有因此证明所有指标无 bug、
所有参数最快，或本组三项已经获得真实 COS source auto 性能资格。
