# F48 Pearson 链默认 auto 验收（2026-10-03）

## 默认调用与适用范围

```python
from quant_evaluator import evaluate_factor_source_batch

result = evaluate_factor_source_batch(
    source, labels,
    metrics=("pearson_ic", "pearson_ic_series", "pearson_ic_std", "pearson_ic_ir"),
    backend="auto",
)
```

这里 source 是已验证的有界 FactorTileSource，声明 max_tile_size=16；
公共 API 的默认 max_tile_size=None 使用该声明上限，并非所有 source 都自动设为 16。
本轮请求为 2586 天 × 5461 股票 × 48 因子、因子和标签均 float64。
默认策略在已认证 L20、有效显存至少 14 GiB、内存准入通过时选择 CUDA，
有效块宽 4；源内存预算为 4096 MiB，预取为两个 worker / 512 MiB。
显式 API max_tile_size=4 另由正反顺序 strict CPU/CUDA 对照支持。
其他形状、指标组合、精度、requested cap 或设备不能凭此扩大认证。
GPUExecutionPolicy.max_factor_tile_size 是 GPU 专用限制；低于认证宽度会回退 CPU。
调用者仍负责关闭 source。评估是研究用途，不代表 PIT 或生产因子认证。

## 实际验收

默认 auto 的新回执：
[完整通过报告](benchmarks/real_cos_f48_pearson_chain_default_auto_20261003.json)。

- status=complete，route_pass / coverage_pass / reference_comparison / source_provenance_verification 全通过。
- 实际 CUDA；source 声明上限 16，预算准入上限 5，实际宽度 4；完整 12 块覆盖全部 48 因子。
- API 耗时 60.69646 秒，OOM 重试 0，峰值显存 1,186,262,528 字节。
- 共 124,272 个值与 strict 参考的哈希逐项一致，包括 124,128 个序列位置；
  Pearson 序列有 123,529 个有限值，不将缺失位置算作有效观测。
  各指标观测数量哈希也一致。
- 169 文件的声明 Python 来源范围在运行前后不变，摘要为
  2fb9cc39c921395a56df08b0d155b6d3f5499ce2c083d673250c826e3a43b3ba。
  不覆盖外部 DataAccess/FO、已导入内存代码和原生库的完整依赖闭包。
  验收后注册表补入该回执引用会改变源码摘要；不声称当前树与历史摘要相同。

两个独立顺序的 width-4 strict 对照分别为 CPU 82.03586 / CUDA 57.89339 秒，
以及 CPU 87.75727 / CUDA 58.62351 秒；两轮数值、掩码与观测数量通过。
这是当前后端对照，不是新旧内核端到端加速，也不保证所有服务器负载下同样耗时。

[CPU-first](benchmarks/real_cos_f48_pearson_chain_width4_cpu_first_20261003.json)
与 [CUDA-first](benchmarks/real_cos_f48_pearson_chain_width4_cuda_first_20261003.json)
具有相同历史来源摘要。校验器同时检查文件大小、类型、预检门槛、来源/请求哈希、
块覆盖、caller 资源设置、有限耗时、CUDA 较快、掩码/计数及跨报告结果哈希。

## 修复过的验收反例

[第一次默认调用失败回执](benchmarks/real_cos_f48_pearson_chain_default_auto_range_container_failure_20261003.json)
保留不变。该次 route、结果及计数均匹配，但校验脚本把运行时 list-of-tuples
与预期 list-of-lists 直接比较，误判 coverage=False，按设计以非零退出。
已将块范围规范化为 tuple-of-tuples，正向测试使用真实运行时元组形式，
错范围仍会被拒绝；不是通过篡改失败 JSON 或移除失败保护获得通过。
修复后重新进行了真实调用，上述新报告才通过。

准备前与标签物化后各检查至少 32 GiB 可用 RAM、至少 5 GiB 缓存磁盘余量。
预检失败输出有界拒绝回执并禁止开始评估。对路由、覆盖、数值、来源变化的
负向测试均检查非零退出；不降低准入门槛。

## 当前瓶颈与后续

本轮 source-read 等待 59.29295 秒，约为 API 耗时的 97.7%。
并发任务的 bound_factor_read 累计时间不能直接加到总 wall time。
应进一步分解 HEAD、下载、内容摘要校验、Arrow 读取与队列等待；
不能直接认定其中哪一步占主要成本，或跳过内容/ETag 校验。
在同一 512 MiB 预取预算内提升并发需要先证明每个 worker 的峰值，
不为了性能直接放宽保护。全量指标/所有请求最优仍未证明。

## 复现

```bash
ASHARE_PARQUET_ROOT=/home/sunhaiwei/cos_data \
DATA_ACCESS_COS_CLI=admin-cos \
DATA_ACCESS_COS_CACHE_ROOT=/home/sunhaiwei/.cache/quant-dataaccess/research \
.venv/bin/python -m quant_evaluator.scripts.benchmark_f48_pearson_default_auto \
  --references \
    quant_evaluator/docs/benchmarks/real_cos_f48_pearson_chain_width4_cpu_first_20261003.json \
    quant_evaluator/docs/benchmarks/real_cos_f48_pearson_chain_width4_cuda_first_20261003.json \
  --output /tmp/f48-pearson-default-auto-reproduction.json
```

复现输出选择新文件，勿覆盖已有证据；source 扫描/标签准备与 close 不计入 API 秒数。
