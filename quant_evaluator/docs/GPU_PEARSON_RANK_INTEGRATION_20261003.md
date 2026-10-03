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
也不是 materialized evaluate 的 GPU tile 参数。
`GPUExecutionPolicy(max_factor_tile_size=8)` 可为所有 GPU 路径指定因子分块上限，
省略或 `None` 保持原有候选。该字段只接受正的内置整数（拒绝 bool），
是上限而非精确宽度或速度保证；内存准入及 OOM 缩块仍生效。
CPU source 不受 GPU 专用上限约束，仍按自身 `max_tile_size` 读取。
静态 materialized auto 对自定义上限回退 CPU，因为旧测速证据未认证该政策；
实测校准可继续运行，缓存身份包含上限字段，不能借用默认政策的校准结果。
source auto 在上限不小于已认证宽度时保留该宽度，低于它则明确回退 CPU。
显式 `cuda_strict` 也受上限约束，执行回执记录请求上限及实际 GPU 宽度。
源接口目前提供 auto/cpu/cuda_strict；两接口返回类型及支持指标集合不同。
完整选项见 [SOURCE_BATCH_OPTIONS.md](SOURCE_BATCH_OPTIONS.md)。

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
早先默认 CLI `clean-cos-ro` 不存在，直接试用 `coscli` 也未取得元数据。
随后按 DataAccess 研究运行手册使用服务器已有的 `admin-cos` 网关，
仅设置当前测试进程环境（不修改凭据或持久配置），已成功读取并校验固定 manifest。
固定 manifest SHA256 为
`b2cf8709e68d0d2b3168fcf3a4ccbb207b4be0f1be510e77df01b9ddb42e6864`。
按现有 source 4096 MiB 总预算选出的 48 个不同内容哈希对象共
3,620,857,809 字节（3453.1191 MiB），最大单对象 105,227,272 字节。
本次诊断只读清单，未读取因子对象；因此证明的是网关可用及清单字节预算，
不是因子轴、ETag 或评估结果已通过新一轮验证。
整批 materialized 预检的 2048 MiB 门槛确实不足以装下该批对象；
保守主存估计加 24 GiB 余量也高于本次约 37 GiB 可用主存。
未降低这些保护，也未扩大加载默认预算；真实全量测试优先走现有有界 source 路线。
截至本节记录，新内核真实 COS 全流程重测仍待运行；未把 F48 materialized auto
从 CPU 擅自改为 GPU。

网关使用示例（仅当前命令环境）：

```bash
ASHARE_PARQUET_ROOT=/home/sunhaiwei/cos_data \
DATA_ACCESS_COS_CLI=admin-cos \
DATA_ACCESS_COS_CACHE_ROOT=/home/sunhaiwei/.cache/quant-dataaccess/research \
.venv/bin/python -m quant_evaluator.scripts.preflight_materialized_f48_auto
```

该 materialized 预检现在会因明确的总对象预算不足而非认证失败返回非零。
不要为解决默认 CLI 名称问题读取凭据、切换角色或绕过 DataAccess。

FE 原生 EWM 的 4 个独立 Decimal 极值反例仍失败，不属于本轮已修复内容；
全部 FE 原生复用、所有指标最快及无 bug 的总目标仍未完成。

复现与完整范围见
[排序 A/B](benchmarks/gpu_rank_scatter_cleanup_ab_20261003.md)、
[Pearson A/B](benchmarks/pearson_centered_fused_ab_20261003.md)及对应 JSON。

## 后续分块控制与运行证据

新增 policy cap、source 路由/回执、OOM 防御限制及测速预检失败回执后，
完整 `quant_evaluator/tests` 为 **5075 passed、26 skipped、53 warnings，230.78 秒**。
定向分块/OOM/初始准入回执测试另为 35 passed；source benchmark harness
与失败回执测试联合为 73 passed。真实 L20 的
`_device_admission(GPUExecutionPolicy(max_factor_tile_size=2))`
也返回通过，未将自定义 cap 错误放入共享设备拒绝条件。
校准身份测试验证 cap None/1/2 生成不同 key，不能复用旧缓存。

本次真实 F48 混合三指标测试首次完成因子扫描，但在读取标签前被
DataAccess 默认镜像 `/home/shw/...` 的目录权限拒绝，评估没有开始。
现有正式镜像是 `/home/sunhaiwei/cos_data`，项目目录的 A 股数据路径
已符号链接到该镜像；测试进程可使用已有 `ASHARE_PARQUET_ROOT` 选项，
无需改凭据、修改 DataAccess 或创建另一个数据副本。

后续启动时主存余量下降，32 GiB source 准入门槛拒绝执行；
可审查的 [失败回执](benchmarks/real_cos_f48_source_resource_rejection_20261003.json)
标明 `preflight_rejected`、`evaluation_started=false`、`factor_objects_read=0`，
该回执只描述最后一次启动，不否认首次尝试曾扫描因子对象。
它没有耗时/结果数据，绝不能作为新内核端到端加速证据或 auto 认证输入。
上述失败为历史尝试；随后资源恢复并使用正式镜像后，真实 F48 当前后端
对照已成功完成，见下节。失败回执继续保留，不用于认证输入。
所有指标最快、全部真实场景无 bug、FE 极值递推及完整 FE 原生复用仍未闭合。

## 当前真实 F48 验证更新

[当前混合三指标对照与普通 auto](REAL_COS_F48_CURRENT_BACKENDS_20261003.md)
记录了 2586 天 × 5461 股票 × 48 因子的有界 COS 评估。
CPU-first 当前 CPU 为 233.8842 秒，CUDA 为 54.9888 秒，144 个标量
以及有效掩码、观测数量对照通过；后续普通 auto 真实选择 CUDA、宽度 2。
后续反序测试可能与另一窗口的准备读取重叠，其耗时不作独立稳定基准。
这是当前后端对照，不是新旧内核端到端提速；不能据此扩大其他形状、
指标组合或块宽的 auto 认证范围。全指标最优仍需继续验证。
