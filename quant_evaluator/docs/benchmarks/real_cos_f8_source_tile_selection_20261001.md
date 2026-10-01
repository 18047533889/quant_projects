# F8 COS source：tile=2 与 tile=8 的端到端实测

## 测量范围

我们通过公共 `evaluate_factor_source_batch` 接口评估 8 个已有 COS 因子。
请求形状为 `(2586, 5461, 8)`，因子与标签存储类型为 `float64`，指标为
`rank_ic` 和 `rank_ic_series`。设备为 NVIDIA L20，使用默认 `GPUExecutionPolicy`。
COS adapter 使用 DataAccess 绑定 manifest、文件内容与轴身份，并开启有界预取。

Manifest SHA-256：
`b2cf8709e68d0d2b3168fcf3a4ccbb207b4be0f1be510e77df01b9ddb42e6864`。
两组 fresh tile=2 请求的语义指纹均为
`14ab6a8e65afa45ed996d2f9d7aac07388c14944d9a2f6556d28ab3c8882746c`。
该指纹包括标签内容、因子顺序、完整轴、source snapshot、类型及指标顺序，
不包括 tile cap；同一请求可以更换执行分块并保持语义身份。

## 结果

我们分别按 CPU→CUDA 和 CUDA→CPU 的顺序执行完整请求。
表中秒数包括公共 source API 的读取、面板构建、计算和返回结果。
计时前的 manifest 检查、公共轴求交与标签准备不在此秒数内。

| tile | 执行顺序 | CPU 秒数 | CUDA 秒数 |
|---|---|---:|---:|
| 2 | CPU→CUDA | 25.222 | 9.169 |
| 2 | CUDA→CPU | 24.081 | 9.135 |
| 8 | CPU→CUDA | 28.258 | 12.450 |
| 8 | CUDA→CPU | 30.358 | 11.377 |

CUDA 两轮中位数：tile=2 为 9.152 秒，tile=8 为 11.914 秒。
在本请求上，tile=2 比 tile=8 降低约 23.18% 的端到端耗时。
CUDA 计算之外的 source 读取及构建占据多数时间：tile=2 的 CUDA 读取计时
为 8.188、8.161 秒，tile=8 为 11.468、10.400 秒。
我们据此选择 tile=2；增加 tile 宽度并不能保证端到端更快。

每组比较覆盖 8 个 scalar 值及 `2586×8` 个 series 值，共 20,696 个值。
两轮同宽度 CPU/CUDA 输出哈希、有限值掩码和观测计数一致；CPU 与 CUDA
按 `rtol=1e-8, atol=1e-10` 比较通过，序列最大绝对误差为 `2.22e-16`。
不同 CUDA tile 宽度的 scalar 均值存在末位归约差异，因此 auto 验证必须
匹配所选 tile=2 的结果哈希；不要求它与 tile=8 的 scalar 哈希相同。

## 内存与执行边界

我们对两种宽度都设置 8 GiB source 预算、2 个预取 worker、512 MiB 预取预算，
保留 32 GiB 可用 RAM、缓存磁盘余量及 CUDA 有效显存门槛检查。
tile=8 的 source 逻辑峰值估算为 5,733,820,640 字节，默认 4 GiB 预算会拒绝该请求。
估算覆盖面板副本、validity owner 与声明的转换缓冲，不代表整个进程 RSS 上限。
`peak_process_rss_kib` 是进程至该时点的累计峰值，不能据此比较单个 backend 的独立峰值。
这四份 A/B 报告未保存 GPU pool 峰值；下面的 auto 验收补充了显存计数。

本轮路由只覆盖精确形状、上述指标集合、float64 输入/标签及请求 cap=8，
有效 CUDA tile 为 2，理由为 `bounded_f8_rank_pair_gpu_cap8_tile2`。
真实公共 auto 请求已通过验收；我们保留原 cap=2 的独立认证记录。
其他形状、指标、类型和 GPU 策略需要各自证据，PIT 与生产准入不在本次认证范围。

## Auto 验收与调用选项

公共 auto 验收耗时 9.089 秒，实际 backend 为 CUDA，有效 tile=2。
GPU pool 峰值为 2,586,991,616 字节（约 2.41 GiB），OOM 重试为 0。
同次显式 CUDA tile=8 的 GPU pool 峰值为 8,174,620,672 字节（约 7.61 GiB）。
Auto 输出与 fresh tile=2 参考的值及计数哈希相同，共核验 20,696 个值；
与显式 tile=8 比较时，series 完全一致，scalar 最大绝对差为 1.04e-17。
两次请求的 source 语义指纹相同。验收期间其他 AI 更新了仓库 HEAD，
四轮 A/B 的 HEAD 记录只描述那四轮测量；auto 验收独立保存运行记录。

你可以在公共接口选择 backend="auto"、"cpu" 或 "cuda_strict"。
max_tile_size 是上限；auto 在已认证的请求范围内可选更小的实际块宽。
此 F8 cap=8 请求需在 CosFactorTileSource 中将 max_source_memory_bytes 设置为 8 * 1024**3，
并保留 GPUExecutionPolicy 与 source adapter 的资源检查。默认 4 GiB
source 预算不够容纳 cap=8 的保守估算，不能靠忽略门槛运行。
当前认证不等于任意因子数、指标组合或设备下都最快。

## 原始记录与源码

- [auto cap=8、有效 tile=2 验收](real_cos_f8_source_auto_cap8_tile2_20261001.json)
- [fresh tile=2 CPU-first](real_cos_f8_source_rank_pair_tile2_fresh_cpu_first_20261001.json)
- [fresh tile=2 CUDA-first](real_cos_f8_source_rank_pair_tile2_fresh_cuda_first_20261001.json)
- [tile=8 CPU-first](real_cos_f8_source_rank_pair_tile8_cpu_first_20261001.json)
- [tile=8 CUDA-first](real_cos_f8_source_rank_pair_tile8_cuda_first_20261001.json)

测量时 HEAD 为 `c08c82fb55fb56013984401ec1efacafc17024c6`，数值运行时代码保持不变。
tile=8 harness SHA-256 为 `0ec16c10e625a1225c09816976dfd8651cefd782c35cba06f8dc09342dc86374`；
fresh tile=2 harness SHA-256 为 `2a641909b874135731adbb8e70920a50aae967b2f5c7d4caad1ba43e860bf3f3`。
两组之间的 harness 修改仅增加运行记录中的 `source_request_fingerprint` 字段。
