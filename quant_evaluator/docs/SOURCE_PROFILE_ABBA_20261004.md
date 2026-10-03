# 独立参考约束下的批量后端交叉测速

`scripts.source_profile_abba.produce_source_route_profile_abba` 固定运行
CPU → CUDA → CUDA → CPU，再生成两份相反顺序的类型化资格记录。
实际调用默认复用 `benchmark_real_cos_source_batch.run_backend`；使用默认
live observer 捕获实际 source、标签、请求、源码、运行时及设备上下文。
四次运行必须使用完全相同的上下文，且各次前后上下文都不能漂移。

调用者必须显式提供 `oracle` 函数。函数收到 `bundle`、`run_receipt`、
`context`、`backend`、`run_index`，返回独立参考的 `BatchEvaluationBundle`
（包含相同 factor_ids 和 source_request_fingerprint），或者逐指标映射：

```python
{metric_id: {"values": float_array, "observation_counts": integer_array}}
```

每个指标校验完整输出形状与覆盖数、有限/NaN/正负 Inf 类别、有效观测计数，
并使用资格上下文中的固定绝对误差门槛。布尔“正确”声明不能替代数组参考；
CPU/CUDA 彼此一致也不是独立参考。回调本身是调用者可信边界，不是签名证明。
Pearson 四指标可用 `source_pearson_oracle.reference_source_pearson_chain`
在新的有界顺序 source 上事先计算参考，放在性能计时之外。

数值通过后，还要验证各后端实际执行的完整分块范围、零 OOM、相同后端两次
分块配置一致，以及两种运行顺序均有同一严格耗时赢家。CPU 和 GPU 可以有
不同的已测分块宽度，不强迫它们共用一个宽度；不得把未测宽度冒充已测配置。
不稳定赢家或执行偏离会拒绝生成正式资格，而不是给出乐观默认值。

返回 `result.records` 可供同进程实际请求使用：

```python
evaluate_factor_source_batch(
    fresh_source, labels, metrics=metrics, backend="auto",
    max_tile_size=requested_cap, gpu_policy=policy,
    source_qualification=result.records,
)
```

真正是否应用须检查结果的 source_qualification_applied/status/winner，而非只看
backend_used。未初始化运行时、上下文改变或资源门禁不满足时会保守拒绝资格。
本模块不绕过真实 COS 总体内存门禁；实测入口必须每次 source/运行前检查资源。

## 全市场真实 F48 入口与默认 auto 验证

`scripts.benchmark_real_cos_profile_abba` 默认只做资源预检，不读取因子或启动 GPU。
明确运行时使用如下命令（正式服务器工作树及已有授权 DataAccess 环境）：

```sh
.venv/bin/python -m quant_evaluator.scripts.benchmark_real_cos_profile_abba \
  --run \
  --axis-index quant_evaluator/docs/benchmarks/real_cos_f48_axis_index_20261004.json \
  --output quant_evaluator/docs/benchmarks/real_cos_f48_profile_abba_20261004.json
```

固定覆盖 2586 个交易日、5461 只股票、48 个因子，以及 Pearson IC 均值、序列、
样本标准差和 IR。独立参考先在宽度 1 的新 source 上计算；CPU/CUDA 初始化不计入
ABBA 的 API 耗时。每次实际调用之前重新检查至少 32 GiB 可用内存及磁盘/显存门禁。
v2 报告在四次 ABBA 后，再做第五次显式资格 auto 和第六次不传资格的默认 auto；
第六次必须命中同进程资格缓存、复用已测赢家及分块宽度，并再次通过独立数值参考。
四次 ABBA 才用于比较性能，第五/六次用于验证路由，不充当新的性能资格。

报告载入示例：

```python
from quant_evaluator.scripts.source_profile_report_reader import load_source_profile_report
report = load_source_profile_report(report_path)
bundle = evaluate_factor_source_batch(
    fresh_source, labels, metrics=metrics, backend="auto",
    max_tile_size=requested_cap, gpu_policy=policy,
    source_qualification=report.records,
)
```

载入只验证报告结构及自洽性，不证明当前源码、数据或硬件仍相同。冷进程须显式准备
运行时，API 会重新校验 live context；资格缓存是进程内、最多 32 项、有效期 1 小时，
不跨进程自动恢复。相同进程后续省略资格可命中缓存；请求/数据/配置变化会失效。
此入口只覆盖四个 Pearson 指标，不证明所有指标或任何新工作负载的最快后端。
