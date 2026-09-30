# F32 coverage 与四指标批量评估：auto 接入前的真实数据对照

我们在 server-c 上用 COS 中已有的 32 个因子运行六轮对照。输入为
`float64`，形状为 `(2586, 5461, 32)`，日期覆盖 2016-01-04 至
2026-08-25。交易日历包含 2588 个 session；评估面板包含 2586 行。
我们使用 NVIDIA L20，并保留源报告中的 research lineage、上游 PIT
和可投资性证明限制。此次测试不能替代生产因子的时点合规审查。

## 测试方法

每个独立子进程评估两次，依次选择 CPU、strict CUDA、auto、auto、
strict CUDA、CPU。下表取同一路线两轮热运行耗时的中位数，包含请求
评估开销，排除父进程的 COS 加载时间。共享服务器上还存在其他任务，
这不是隔离机器的性能上限。`--timeout-s 420` 限制整个子进程，包含
该进程的两次评估。

| 请求 | CPU 热运行 | CUDA 热运行 | auto 热运行 | CPU/CUDA |
| --- | ---: | ---: | ---: | ---: |
| coverage | 16.782 s | 14.734 s | 16.863 s | 1.14 倍 |
| rank_ic + quantile_spread + factor_turnover_rate + coverage | 145.202 s | 18.441 s | 144.535 s | 7.87 倍 |

这两份报告中的 auto 均选择 CPU。coverage 的原因是
`metric_not_certified`；四指标请求的原因是 `metric_set_not_certified`。
读者不能用这些报告证明修改后的 auto 已经选择 CUDA。接入策略后还需
运行 public API 对照，并检查 `backend_used` 和各指标的 backend。

## 正确性与内存

我们核验了六轮状态及与首轮 CPU 的对照：数值按 `rtol=1e-8`、
`atol=1e-10` 比较，轴描述、finite/valid mask、观测计数和 provenance
检查均通过。CPU 与 CUDA 的 artifact SHA 不同，因此这里证明的是
报告列出的数值和语义一致性，不是逐位相同。

CUDA 的峰值显存：coverage 为 1,256,872,448 B；四指标为
8,054,511,104 B。四指标 CPU 子进程峰值 RSS 约 11.71 GiB，CUDA
子进程约 7.87 GiB。显存准入还需考虑用户的 `max_vram_fraction`、
设备空闲显存及其他任务；峰值测量值不能直接充当无余量的准入阈值。

## 原始证据

- [coverage 六轮报告](real_cos_f32_coverage_ab_20261001.json)
- [四指标六轮报告](real_cos_f32_mixed_three_coverage_ab_20261001.json)

我们尚未用这两份报告证明其他形状、因子数量、dtype、非默认指标参数
或其他 GPU 型号的速度优势。当前选择证据限定在上面列出的请求。
