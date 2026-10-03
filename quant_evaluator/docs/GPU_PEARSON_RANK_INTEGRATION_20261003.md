# GPU Pearson / 排名集成与使用（2026-10-03）

## 默认路由与明确选项

已有批量调用选择 `backend="auto"` 时，仍按已测量的形状、指标组合、
dtype、设备资格和当前内存门槛确定路由；未认证的区域保留 CPU 路径。
本轮没有扩大 auto 的认证区域。进入 CUDA 的普通 Pearson f32/f64 分支
自动使用新的融合归约，无需另一个开关；Spearman 的 bounded 排名归约保持原路径。

```python
from quant_evaluator import evaluate, evaluate_factor_source_batch
from quant_evaluator.contracts.backend_policy import GPUExecutionPolicy, PrecisionPolicy

policy = GPUExecutionPolicy(
    precision_policy=PrecisionPolicy.GPU_FP64,
    max_vram_fraction=0.75,
    oom_retile=True,
)
# batch 是 FactorBatch，labels 是与其轴和时间口径匹配的 LabelBundle。
bundle = evaluate(
    batch, labels,
    metrics=("pearson_ic_series", "pearson_ic", "pearson_ic_std", "pearson_ic_ir"),
    backend="auto", gpu_policy=policy,
)
# 要求必须 CUDA 可将 backend 改为 cuda_strict；不具资格会报错而非假冒 GPU。
# source 是已验证的 FactorTileSource；调用者负责 source.close()。
columnar = evaluate_factor_source_batch(
    source, labels, metrics=("rank_ic", "quantile_spread", "factor_turnover_rate"),
    backend="auto", max_tile_size=16, gpu_policy=policy,
)
```

`max_tile_size` 是 source 读取上限，不保证实际 GPU tile 等于该值，
也不是 materialized evaluate 的 GPU tile 参数。现有 GPUExecutionPolicy
还没有显式 factor tile 上限字段；不可把上面的 source 参数套到 materialized API。
源接口目前提供 auto/cpu/cuda_strict；两接口返回类型及支持指标集合不同。

## 实现与数值语义

Pearson 模块在 pairwise finite 的资产集合 P 上进行两遍设备归约：
先统计有效数量及均值，再直接累计中心化平方和、乘积和，
不构造整个面板的 dx/dy/product 临时数组。

$$
n=|P|,\quad \bar x=\frac{\sum_{i\in P}x_i}{n},\quad
\bar y=\frac{\sum_{i\in P}y_i}{n}
$$

$$
\operatorname{IC}=
\frac{\sum_{i\in P}(x_i-\bar x)(y_i-\bar y)}
{\sqrt{\sum_{i\in P}(x_i-\bar x)^2\sum_{i\in P}(y_i-\bar y)^2}}
$$

有效观测不足、常数及不安全数值仍按原有判据处理，极端值 repair 没有被删除。
融合模块支持负 stride、非连续 view、f32/f64 混合以及 bool finite mask；
其它 dtype 保留既有路线。超出 CUDA 行索引范围明确拒绝。

排名模块仍使用稳定排序与 average ties。无效值的 sentinel 在排序内核中
已产生 NaN，scatter 是完整置换，因此删除末尾重复掩码复制；
distinct 边界数组只在调用方请求 distinct 时生成。

## 实测与未完成项

L20、512 天 × 1000 股票、32/48 因子、f64 的最终源 Pearson A/B：
核心调用中位耗时约快 5.50/5.84 倍，最大差异约 5.55e-17。
48 因子 f32 排名 A/B：rank-only 约快 1.92%，带 distinct 约快 0.77%；
CuPy pool 增长未减少。这些均不是 COS 整批或全部指标的端到端加速承诺。

第二轮 QE 全库为 5054 passed、26 skipped、53 warnings，208.49 秒。
之后修正的预检测试单独与加载边界/旧预检联合运行为 28 passed，0.66 秒；
这些测试范围分别记录，不把跳过项当成已验证。
新融合内核针对性测试 13 passed、既有极值/偏移/repair 针对性测试 14 passed；
公开 CPU/CUDA 32/48 因子链路含缺失和 min_assets=20 的 19/20 边界验证。

研究加载器显式因子上限扩到 48；单对象 128 MiB、总对象 2048 MiB 上限
和 load_real_batch 默认总预算 128 MiB 均未扩大。
`python -m quant_evaluator.scripts.preflight_materialized_f48_auto`
仅读取已绑定 manifest，不读取因子或执行评价；失败输出脱敏 JSON 并返回非零。
它使用保守内存估计，预期 shape 不是已经读取对象后的轴认证。
本轮 COS CLI 默认不可启动，改用已安装 CLI 的单进程只读试验仍未取得对象元数据；
没有获取当前 48 对象实际总字节、没有绕过授权/HEAD/hash/预算门槛。
所以本轮不能提供新内核的真实 COS 全流程重测结果，也未把 F48 materialized auto
从 CPU 擅自改为 GPU。

FE 原生 EWM 的 4 个独立 Decimal 极值反例仍失败，不属于本轮已修复内容；
全部 FE 原生复用、所有指标最快及无 bug 的总目标仍未完成。

复现与完整范围见
[排序 A/B](benchmarks/gpu_rank_scatter_cleanup_ab_20261003.md)、
[Pearson A/B](benchmarks/pearson_centered_fused_ab_20261003.md)及对应 JSON。
