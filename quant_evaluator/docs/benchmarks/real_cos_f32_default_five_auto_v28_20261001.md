# F32 五指标 auto v28：真实数据接入验证

我们在 server-c 正式主干提交 `dbbe2983c` 后，重跑公开 `evaluate`
入口的 CPU/CUDA/auto 六轮对照。两轮 auto 的五项指标均使用 CUDA，
选择原因为 `certified_batch_real_cos_f32_default_five`。

输入来自既有 COS 因子落值：2586 个交易日、5461 只股票、32 个因子，
float64，日期范围 2016-01-04 至 2026-08-25，设备 NVIDIA L20。
研究数据带有缺失值；本次实验不包含上游 PIT 或可投资性认证。

## 六轮公开入口结果

每个独立子进程运行两次。以下计时包含公开评估调用，排除父进程
COS 加载时间。我们在共享服务器上测试，没有同时运行自己的其他性能基准。

| 轮次 | 请求后端 | 实际后端 | 冷运行秒 | 热运行秒 |
| --- | --- | --- | ---: | ---: |
| 1 | cpu | cpu | 119.5621 | 117.8363 |
| 2 | cuda_strict | cuda | 19.4018 | 18.2900 |
| 3 | auto | cuda | 19.3812 | 18.1477 |
| 4 | auto | cuda | 19.4239 | 18.0463 |
| 5 | cuda_strict | cuda | 19.6721 | 18.3221 |
| 6 | cpu | cpu | 123.4173 | 118.0259 |

CPU、显式 CUDA、auto 的热运行中位数为 117.9311、18.3061、
18.0970 秒；本次 auto 相对 CPU 约快 6.52 倍。auto 与显式 CUDA 的
计时差包含运行波动，不能解释为自动选择器额外加速。

我们核验六轮状态、唯一 config hash、每项指标的描述字段、数值、
finite/valid mask、计数、provenance 和 metric_values，完整比较通过。
数值容差为 `rtol=1e-8, atol=1e-10`。两轮 auto 和两轮显式 CUDA
artifact SHA 相同；CPU 与 CUDA SHA 不同，浮点结果并非逐位相等。
CUDA 峰值显存均为 8,054,511,104 字节，约 7.50 GiB；每轮使用
8 因子的 tile，处理 4 个 tile。全量回归：4475 passed、26 skipped。

## 默认调用与手动选择

```python
from quant_evaluator.runtime.evaluator import evaluate
metrics = ("rank_ic", "pearson_ic", "ic_ir", "quantile_spread", "coverage")
bundle = evaluate(fb, lb, metrics=metrics)  # backend=None 使用同一 auto 选择流程
print(bundle.metadata["execution_receipt"])
reference = evaluate(fb, lb, metrics=metrics, backend="cpu")
gpu = evaluate(fb, lb, metrics=metrics, backend="cuda_strict")
```

本次六轮实测显式传入 `backend="auto"`。你还可指定 `cuda` 或 `gpu`
别名；显式 GPU 不会静默回退 CPU。新路由要求精确 `(2586,5461,32)`
形状、双 float64 输入、上述完整指标集合、默认参数和单 L20。
有效空闲显存门槛为 12 GiB；默认 `max_vram_fraction=0.75` 下需要
至少 16 GiB 设备空闲显存。未测请求不能借用这份精确请求证据。

原始记录：[接入后六轮 JSON](real_cos_f32_default_five_auto_v28_ab_20261001.json)。
对照：[接入前说明](real_cos_f32_default_five_comparison_20261001.md)。
