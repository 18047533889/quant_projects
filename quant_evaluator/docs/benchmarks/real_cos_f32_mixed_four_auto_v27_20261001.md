# F32 四指标 auto v27：真实数据接入验证

我们在 server-c 的主工作树提交 `a6d948762` 后，重新运行公开批量评估
入口的六轮 CPU/CUDA/auto 对照。两轮 auto 的四项指标均使用 CUDA，
选择原因为 `certified_batch_real_cos_f32_coverage_mixed_four`。

输入来自既有 COS 因子落值：`float64`、2586 个评估日期、5461 只股票、
32 个因子，日期范围 2016-01-04 至 2026-08-25。我们使用 NVIDIA L20。
源数据带有 research lineage，并缺少上游 PIT/可投资性认证；我们没有
用此次性能实验宣称生产时点合规。

## 六轮结果

每个独立子进程运行两次。热运行不含父进程的 COS 加载时间，包含
公开评估调用开销。共享服务器上还有其他任务。

| 轮次 | 请求后端 | 冷运行 s | 热运行 s | 实际后端 |
| --- | --- | ---: | ---: | --- |
| 1 | cpu | 147.411 | 144.766 | cpu |
| 2 | cuda_strict | 19.531 | 18.453 | cuda |
| 3 | auto | 19.663 | 18.376 | cuda |
| 4 | auto | 19.614 | 18.354 | cuda |
| 5 | cuda_strict | 19.551 | 18.541 | cuda |
| 6 | cpu | 148.384 | 143.448 | cpu |

CPU、strict CUDA、auto 的热运行中位数分别为 144.107、18.497、
18.365 秒。此次 auto 相对 CPU 约快 7.85 倍。我们没有把 auto 比显式
CUDA 少约 0.13 秒解释为策略本身的收益，这个差额包含计时波动。

我们核验六轮状态、唯一 config hash 和报告中的完整对照项，
`parity_pass=true`。数值比较采用 `rtol=1e-8`、`atol=1e-10`；轴、
finite/valid mask、观测计数及 provenance 检查均通过。两轮 auto 与
两轮显式 CUDA 的 artifact SHA 相同；CPU 与 CUDA SHA 不同。

CUDA 峰值显存为 8,054,511,104 B，约 7.50 GiB。v27 要求该组合有
12 GiB 有效空闲显存，默认 `max_vram_fraction=0.75` 下需要至少
16 GiB 设备空闲显存。新组合只匹配上述精确 F32 形状、默认参数、
无额外上下文和单 L20；其他条件需独立测试或使用已有显式校准选项。

## 用法与证据

你可以调用 `evaluate(fb, lb, metrics=("rank_ic", "quantile_spread",
"factor_turnover_rate", "coverage"), backend="auto")`，然后检查
`bundle.metadata["execution_receipt"]`。省略 backend 与 auto 使用同一
选择流程；此次六轮实跑明确传入 auto。

- [改后六轮原始报告](real_cos_f32_mixed_four_auto_v27_ab_20261001.json)
- [接入前对照说明](real_cos_f32_coverage_comparison_20261001.md)

本报告证明四指标组合的接入；coverage 单项的改后回放还需独立执行。
