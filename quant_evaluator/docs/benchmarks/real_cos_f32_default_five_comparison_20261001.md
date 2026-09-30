# F32 五指标：接入前 CPU / CUDA 对比

本次测试使用 COS 中的 32 个真实因子，通过公开批量评估入口计算
`rank_ic, pearson_ic, ic_ir, quantile_spread, coverage`。
输入为 `(2586, 5461, 32)`、`float64`；指标参数沿用默认值。
测试在 server-c 的正式工作树执行，没有复制数据集。

## 六轮结果

我们按 CPU、CUDA、auto、auto、CUDA、CPU 的顺序运行独立子进程。
每个子进程执行一次冷运行和一次热运行，420 秒上限覆盖整个子进程。

| 轮次 | 请求后端 | 实际后端 | 冷运行秒 | 热运行秒 |
| --- | --- | --- | ---: | ---: |
| 1 | cpu | cpu | 120.0668 | 120.3957 |
| 2 | cuda_strict | cuda | 19.2490 | 17.9236 |
| 3 | auto | cpu | 119.8601 | 118.0567 |
| 4 | auto | cpu | 119.3846 | 117.3564 |
| 5 | cuda_strict | cuda | 19.2685 | 17.9389 |
| 6 | cpu | cpu | 119.6572 | 118.3749 |

显式 CPU 热运行中位数为 119.3853 秒，显式 CUDA 为 17.9312 秒，
本次请求约快 6.66 倍。两个 CUDA 进程峰值显存均为 8,054,511,104 字节
（约 7.50 GiB）。CPU 峰值 RSS 约 11.69 GiB，CUDA 约 7.84 GiB。

## 一致性与边界

六轮完整比较均通过：描述字段、数值、finite/valid mask、计数、
provenance 和 metric_values。数值容差为 `rtol=1e-8, atol=1e-10`。
我们没有据此宣称 CPU/CUDA 浮点结果逐位相等。

两轮 auto 在 v27 下返回 `metric_set_not_certified`，因此选择 CPU。
这份记录证明显式 CUDA 在该请求上的性能优势；接入后的 auto 行为需要
另一份实测记录。其他形状、非默认参数、不同指标组合仍需各自验证。

原始记录：[real_cos_f32_default_five_ab_20261001.json](real_cos_f32_default_five_ab_20261001.json)。
配置哈希：`1ae0d37cc7db4505aadd24a0f7db5452d7407b81029dafa94897b331b6d6bef8`。
公开数据清单 SHA256：`b2cf8709e68d0d2b3168fcf3a4ccbb207b4be0f1be510e77df01b9ddb42e6864`。
我们保留原始对比记录，后续路由修改不会覆盖本次证据。
