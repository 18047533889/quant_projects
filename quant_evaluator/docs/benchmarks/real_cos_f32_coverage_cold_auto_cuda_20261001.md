# F32 coverage：auto 首次 CUDA 路由验证

原始回执：[JSON](real_cos_f32_coverage_cold_auto_cuda_20261001.json)。
我们在 server-c NVIDIA L20 上读取 COS 的 32 个真实因子，面板为
`(2586, 5461, 32)`，float64。数据加载耗时 123.074 秒，不计入下表。
清单 SHA-256：`b2cf8709e68d0d2b3168fcf3a4ccbb207b4be0f1be510e77df01b9ddb42e6864`。

## 六轮公开评估

每轮启动隔离进程并调用三次公开评估，包含诊断和配置身份计算。

| 轮次 | 请求后端 | 冷运行秒 | 两次暖运行中位数秒 | 三次实际路由 |
|---|---|---:|---:|---|
| 1 | cpu | 15.816 | 6.638 | CPU / CPU / CPU |
| 2 | cuda_strict | 15.394 | 5.475 | CUDA / CUDA / CUDA |
| 3 | auto | 15.294 | 5.305 | CUDA / CUDA / CUDA |
| 4 | auto | 15.393 | 5.411 | CUDA / CUDA / CUDA |
| 5 | cuda_strict | 15.567 | 5.439 | CUDA / CUDA / CUDA |
| 6 | cpu | 15.864 | 6.571 | CPU / CPU / CPU |

18 次数值对照、重复数值对照及重复一致性检查通过；配置 hash 相同。
我们检查测量前后的 HEAD 和五个相关源码 SHA-256，均未改变。
HEAD 为 `dc70ad6489358d742f48e83f2859af86902cfe37`，路由改动当时未提交。
evaluator SHA-256 为 `fa616afedba6b769c782f34ad9915e9a2163f4e03235c1c35630a22bf247cbfe`。
CUDA 峰值显存为 5,083,974,144 字节，因子 tile 为 32，单 tile。
QE 全量回归：4593 passed、26 skipped。跳过项仍需专用输入或对应内核。

## 路由与使用

我们移除这个精确 coverage 请求的冷身份缓存 CPU 降级。默认 `auto`
从首次调用就执行 CUDA 准入检查；精度、设备和显存拒绝规则保持不变。
调用 `evaluate(batch, labels, metrics=("coverage",))` 使用默认 auto；
你也可传 `backend="cpu"` 或 `backend="cuda_strict"`。
缓存复用继续减少身份计算成本，缓存冷热不再决定这条静态路由。

本次 auto 暖运行比 CPU 中位数低约 18%–20%，与显式 CUDA 同一耗时范围。
冷运行的差距较小。我们保留整次请求的计时，未只比较 GPU 内核。
这些结果不证明其他指标、形状、设备或资源竞争情况下的最优后端。

旧策略与测量数值保存在
[前一轮设备掩码报告](real_cos_f32_coverage_device_mask_three_repeats_20261001.md)。
复现使用同一报告的命令，将输出路径改成新的文件名，避免覆盖原始回执。
