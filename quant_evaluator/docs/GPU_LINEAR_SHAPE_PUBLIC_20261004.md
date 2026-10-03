
## 计算口径与公式

分层按因子从低到高排列，索引为 $q=0,\ldots,Q-1$。桶边界只由
当日有效且有限的因子值决定，**未来标签缺失不能反向改变桶边界**。
分桶后才剔除无效/非有限标签，以可用标签数检验每桶 `min_assets`。
该规则也见 `quantile_assignments_reuse_20261003.md`。

令 $S_{t,q,f}$ 为因子 $f$ 在日期 $t$ 的桶 $q$ 内可用标签集合：

$$R_{t,q,f}=\frac{1}{|S_{t,q,f}|}\sum_{i\in S_{t,q,f}}y_{t,i}.$$

若集合规模不足，日桶收益为 NaN。令 $D_{q,f}$ 为有有限日桶收益的
日期集合，profile 为

$$r_{q,f}=\frac{1}{|D_{q,f}|}\sum_{t\in D_{q,f}}R_{t,q,f}.$$

各桶至少 20 个有效日期才得到有限 profile。以下省略因子索引：

| 指标 | 公式 | 所需有限数据 |
| --- | --- | --- |
| `quantile_curvature` | $\frac{1}{|I|}\sum_{q\in I}(r_{q+1}-2r_q+r_{q-1})$ | $Q\ge3$；$I$ 为三个邻点均有限的内点；空集为 NaN |
| `quantile_tail_asymmetry` | $r_{Q-1}-2r_{\lfloor Q/2\rfloor}+r_0$ | $Q\ge3$；三个指定桶均有限 |
| `quantile_adjacent_spread` | $\frac{1}{|P|}\sum_{q\in P}|r_{q+1}-r_q|$ | $P$ 为两个邻点均有限的边；空集为 NaN |
| `quantile_extreme_cliff` | $\frac{(r_{Q-1}-r_{Q-2})+(r_1-r_0)}{2}$ | $Q\ge2$；两端两桶均有限 |
| `top_quantile_cliff` | $r_{Q-1}-r_{Q-2}$ | $Q\ge2$；顶部两桶均有限 |
| `bottom_quantile_cliff` | $r_1-r_0$ | $Q\ge2$；底部两桶均有限 |

除 adjacent spread 外均保留符号；extreme cliff 是两端差值的有符号
均值，不是绝对值、最大值或多空总收益。偶数 Q 的中桶索引为 Q//2。
typed observation count 沿用 CPU 的 `finite_metric_value`：有限结果为
1，否则 0，**不是**有效日期数/资产数/有效邻边数。

## 使用与实现边界

公开 `evaluate` 和 `evaluate_many` 可显式选择 `backend="cuda_strict"`，
并传 `quantile_builder_parameters={"n_quantiles": 5, "min_assets": 10}`。
默认 Q=5、每桶 min_assets=10。CUDA 只支持全局分层选项 Q/min_assets；
显式 `window_size` 会拒绝（即使传默认值 20），不会忽略参数。
六项 compute_fn 没有 Q/min_assets 逐指标参数，不能放进
`metric_parameters`。已有 quantile_returns_full 等指标仍使用其自己的
逐指标参数，不因全局 profile 选项改口径。

`runtime/gpu_quantile_shape_adapter.py` 集中维护六个公开 ID 与原生设备
kernel 的映射。GPUExecutor 在每个 run/tile/label 内复用日桶收益和
原始时序均值；门控 profile 缓存键包括 `(Q,min_assets,min_periods)`，
不跨请求、标签或 tile 复用。

六项结果和 0/1 有效计数打包为 float64 `(2M,F_tile)`，每个 tile 的
shape 结果组只做一次设备到主机传输；主机端计数恢复为 int64。
混合指标计划仍可能有其他传输，不宣称整个计划仅一次传输。
打包前按新增临时空间做 admission；这不是整个进程 RSS 上限。

普通数值走设备运算，危险溢出/抵消/subnormal 输入走设备内精确二进制
修复；有限标量错误/准入标志的同步不等同于 CPU 数值 fallback。
低层六个 kernel 默认仍返回 NumPy；`return_device=True` 供执行器使用。

## 验证与 auto

公开接线本身不构成最快后端的资格证据。`auto` 不会因为新增显式 CUDA
支持就扩展证书。需要当前源码、当前参数和真实批量计划的数值、计数、
内存与端到端 CPU/CUDA ABBA 验证后，才能自动选择合格赢家。
每次修改 QE Python 源码后，旧全源码 digest 对应的性能资格不能直接沿用。

2026-10-04 低层 CPU/GPU 极端数值回归实跑 21 passed、1 skipped；
skip 是本机只有一个 CUDA 设备，无法验证跨第二设备情形。公开路径
新测试首轮发现独立 oracle 错用未来标签有效性决定桶边界；以源码、
既有计算口径文档、信息时序及两个手算 mask 分离反例修正。
公开路径完整重跑 **20 passed / 1.52 秒**（session84267，exit 0），
包含 CPU/CUDA/独立公式对照、Q2/Q3、tile1/2/3、多标签及指标逆序、
shared factor upload、执行器 shape 组一次 D2H spy、mixed Q3/Q2 缓存
隔离、低层 device-return、打包值/计数 dtype、负例与手算 mask。
手算 fixture 曾因未传必须的显式时间戳失败，补齐后才计作通过。
另 seeded/temporal/public numeric/turnover GPU 回归 20 passed、1 skipped；
rank/boundary/benchmark validator CPU 回归 92 passed / 1.69 秒。
这些是有界正确性证据，不是全股票多年计划速度证书，也不证明所有
输入、硬件及指标组合均无 bug。
新增流式 source CUDA 测试实跑复现 `shape_kernel_backend` metadata 缺失，
在 `run_source_tiled` 补真实 dispatch/backend/no-fallback 证据后，与公开
入口联合重跑 **22 passed / 2.93 秒**（session27060，exit 0）。source
测试检查固定 tile2+1、因子及标签 validity、六项独立公式/CPU/CUDA
一致、0/1 counts、未知指标在 source read 前拒绝。真实 COS 全股票
多年 F48 shape ABBA 与当前源码 auto qualification 仍需继续完成。
