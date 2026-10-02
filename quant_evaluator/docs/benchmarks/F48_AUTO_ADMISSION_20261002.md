# F48 真实源批评估：auto 验证与准入边界

## 当前状态

维护者已保存 2,586 日×5,461 股票×48 因子的 CPU/严格 CUDA 源端 A/B。
两份记录采用相反运行顺序，比较 RankIC、分位收益差和因子换手率。
维护者在候选验证通过后，将正式选择器中的该条证据注册为
`measured_source_ab`。当前版本 `source_routes_20261002_v7` 另含请求上限 16、执行宽度 2 的条目。

历史 `--auto-f48-references` 命令用于独立基准进程内的候选注入，要求注册表
仍处于 pending 状态。注册后请用 `--verify-auto` 验证普通 auto；候选工具会
拒绝已准入记录，防止维护者把注入测试误记为正式路径验证。

维护者已完成候选验证，记录为
`f48_pending_auto_verification_20261002_resume.json`。该次运行覆盖 48 个因子、
144 个因子指标结果，auto 候选耗时 57.128 秒，显式 CUDA 耗时 53.863 秒。
两次执行与历史参考的结果比较通过，源文件运行前后摘要一致。候选执行使用
24 个二因子 tile，显存峰值约 2.41 GiB，OOM 重试为零。
历史两种运行顺序的 CPU 耗时为 220.312 / 231.666 秒，CUDA 耗时为
57.264 / 58.449 秒；这些历史记录不证明本次代码的 CPU 性能。
维护者随后使用 `--verify-auto` 完成正式注册表下的普通 auto 验证，记录为
[f48_ordinary_auto_cpu_cuda_ab_20261002.json](f48_ordinary_auto_cpu_cuda_ab_20261002.json)。
本轮 CPU 为 227.497 秒，显式 CUDA 为 60.825 秒，普通 auto 为 58.012 秒；
CPU/auto 比值约 3.92。三个请求的源请求指纹相同，48 因子、144 个指标结果的
CPU/CUDA 及 CPU/auto 比较均通过，有效值掩码和观测数一致。
RankIC、分位收益差、因子换手率的最大绝对差分别为
`1.08e-16`、`2.93e-18`、`3.89e-16`。普通 auto 返回
`bounded_f48_mixed_three_gpu_tile2`，显存峰值 2,586,991,616 字节，OOM 重试为零。
记录中的源文件摘要在本次运行前后保持一致。该次运行顺序为 CPU、CUDA、auto，
不能据此声称 auto 比显式 CUDA 内核更快；两者使用相同的 GPU 计算路径。

## 精确请求与资源门槛

验证仅接受上述形状、float64、二因子 tile 和以下指标顺序：
`rank_ic,quantile_spread,factor_turnover_rate`。工作进程使用 COS 源适配器，
auto 预取、两个预取对象、两个预取 worker、512 MiB 预取预算。
单对象上限为 128 MiB，总对象上限为 4,096 MiB，逻辑源内存上限为 4 GiB。

执行前至少需要 32 GiB 可用 RAM、5 GiB 缓存盘空闲空间，以及满足 CUDA
硬件检查的 14 GiB 有效显存预算。脚本在各次执行前重新检查资源。
这些值是准入门槛和逻辑预算，不能代替进程 RSS 监测。资源不足时停止运行，
不要降低检查值、删除其他人的缓存或把测试迁到另一个代码副本。

在正式 server-c 主路径、资源满足门槛后，你可运行：

```bash
ASHARE_PARQUET_ROOT=/home/sunhaiwei/cos_data \
DATA_ACCESS_SKIP_COS_MIRROR=1 \
DATA_ACCESS_COS_CLI=/usr/local/bin/admin-cos \
DATA_ACCESS_COS_CACHE_ROOT=/home/sunhaiwei/.cache/quant-dataaccess/research \
.venv/bin/python -m quant_evaluator.scripts.benchmark_real_cos_source_batch \
  --factors 48 --tile-size 2 --days 2586 --assets 5461 \
  --metrics rank_ic,quantile_spread,factor_turnover_rate \
  --source-adapter cos --cos-prefetch auto --cos-prefetch-workers 2 \
  --max-prefetch-memory-mib 512 --max-source-memory-mib 4096 \
  --max-object-mib 128 --max-total-mib 4096 \
  --verify-auto \
  --output quant_evaluator/docs/benchmarks/f48_ordinary_auto_cpu_cuda_ab_UNIQUE.json
```

先将 `UNIQUE` 换成新的运行标识，避免覆盖已有证据。这条命令读取现有授权
COS 数据，不发布生产因子。

## 公开调用与选项

调用方可用已有的、绑定元数据的因子源及 `LabelBundle` 请求批评估：

```python
from quant_evaluator.api.factor_source import evaluate_factor_source_batch

result = evaluate_factor_source_batch(
    source, labels,
    metrics=("rank_ic", "quantile_spread", "factor_turnover_rate"),
    backend="auto", max_tile_size=2,
)
print(result.metadata["backend_used"])
print(result.metadata["auto_backend_reason"])
```

`backend` 默认为 `auto`；传 `cpu` 可强制 CPU，传 `cuda_strict` 可强制 CUDA。
调用方负责关闭 `source`。可通过 `gpu_policy=GPUExecutionPolicy(...)` 设置
显存比例、结果内存预算与 OOM 重分块；改变精度策略会超出已认证的 auto 范围。

本条注册只覆盖 2,586 日、5,461 股票、48 因子、float64 因子和标签，以及上述
三指标集合。集合内重排指标不会改变选择，重复或增减指标会超出范围。
本条 cap2 的有效请求 tile 上限须为 2；新增 cap16 条目要求上限恰为 16，
两者均执行二因子 tile。`max_tile_size` 未传时采用源声明的上限。
F47、F49、F63、F64、F65 等形状没有从这两条证据获得准入。
不满足范围或 GPU 资源门槛时，普通 auto 回退 CPU，并在 metadata 中记录原因。
该范围的速度结论不等于任意数据、任意指标的全局最快。

## 记录可信度

新记录包含声明范围内 QE Python 源文件的运行前后摘要，范围涵盖计算模块和
列出的基准辅助脚本。实现限制文件数量和读取字节数，拒绝路径越界与符号链接。
源文件、声明范围或受检查的版本变化时，成功状态会失效。
这项检查不覆盖已导入的内存代码、DataAccess/Factor Optimizer 的传递依赖、
GPU 驱动或硬件等价性。历史记录缺少的 caller settings 和代码摘要保持未知，
维护者不能通过补写字段或比较数据摘要来追认它们。

## 内存允许批宽与默认调用

COS 适配器保留你声明的 `max_tile_size`，并提供
`admitted_max_tile_size`，表示逻辑源预算允许的最大批宽。
构造时至少需容纳一个因子的组装与预取；读取时再检查本次批宽，
超预算会在因子读取和组装之前报错。构造成功不保证你能读取声明上限的整批。

公开评估 API 会限制 CPU 和显式 CUDA 的执行批宽，并在 metadata 中返回
`admitted_source_tile_size` 和 `effective_max_tile_size`。
auto 仍按原始请求上限选择已验证路径。如果内存允许宽度小于该路径的已验证
GPU 宽度，auto 退回 CPU，原因是
`source_memory_budget_outside_certified_tile`。
调用方需给自定义组装回调声明额外临时内存；估算不覆盖进程 RSS 的全部开销。

维护者已完成声明上限 16、公开调用省略 `max_tile_size` 的 F48 验证。
正式条目为 `real_cos_f48_mixed_three_cap16_tile2`，auto 原因是
`bounded_f48_mixed_three_gpu_cap16_tile2`。4 GiB 源预算允许批宽 5，
选择器仍采用经过测量的执行宽度 2；你不能将内存允许宽度当作性能最优宽度。
此前 cap2 记录的源摘要不证明本轮内存逻辑的代码等价性。

## 宽度 2/4 的追加对照

维护者按 2、4、4、2 顺序运行四次独立的真实 F48 GPU 请求。
记录见 `f48_gpu_width2_4_abba_20261002.json`。宽度 2 的耗时为
61.083 / 58.398 秒，宽度 4 为 60.257 / 58.800 秒；
中位数为 59.740 / 59.529 秒，差约 0.35%，样本耗时范围重叠。
结果比较通过，但这次对照不足以证明宽度 4 稳定胜出；正式路径继续采用已验证宽度 2。

## 正式默认 cap16 路径的使用与记录

你可省略公开 API 的批宽参数，前提是源声明上限为 16，且满足上文的精确形状、
float64 和三指标集合。COS 源工厂的声明上限默认值为 16。

```python
try:
    result = evaluate_factor_source_batch(
        source, labels,
        metrics=("rank_ic", "quantile_spread", "factor_turnover_rate"),
        backend="auto",
    )
    print(result.metadata["auto_backend_reason"])
    print(result.metadata["admitted_source_tile_size"])  # 本次资源配置为 5
    print(result.metadata["effective_max_tile_size"])   # 已验证执行宽度为 2
finally:
    source.close()
```

仍可通过 `backend="cpu"` 或 `backend="cuda_strict"` 指定后端，
并用 `max_tile_size` 限制批宽。auto 对其他请求上限保持现有准入边界；
cap16 证据没有覆盖 cap8、cap15、cap17 或任意因子数量。

维护者保留首轮失败记录 `f48_cap16_default_candidate_20261002.json`：
其数值与读取比较通过，但 API 顶层 metadata 漏报内存允许批宽，
导致路线检查失败。修复后，候选重跑记录
`f48_cap16_default_candidate_retry_20261002.json` 通过。
随后，维护者运行正式选择器、无候选注入的默认调用，记录为
`f48_cap16_ordinary_default_auto_20261002.json`：

- 默认 auto 为 56.159 秒，显式 CUDA tile2 为 54.847 秒。
- 两个请求都读取 24 个二因子 tile，覆盖 48 因子和 144 个标量结果。
- 历史参考与本轮直接对照的数值、有效值掩码和观测数检查通过。
- 两次 GPU 峰值为 2,586,991,616 字节，OOM 重试为零。
- 声明范围内源摘要前后相同：
  `f3dbf3243f54954161097677ad1406b258a221d32cdda8bf205ef6bc41357870`。

这次测试没有重跑 CPU；维护者不以历史 CPU 耗时宣称本轮 CPU/GPU 加速比。
追踪到的传输字节不覆盖内核内部自行发起的传输；GPU 换手率路径的隐式
主机往返仍待单独修复和计量。这项准入也没有证明所有后端或全部指标的最优性能。

复验默认正式路径时使用新输出文件名，并沿用上文授权的数据环境：

```bash
.venv/bin/python -m quant_evaluator.scripts.benchmark_f48_cap16_auto \
  --ordinary \
  --references quant_evaluator/docs/benchmarks/real_cos_f48_source_cpu_first_telemetry_20261002.json \
               quant_evaluator/docs/benchmarks/real_cos_f48_source_cuda_first_telemetry_20261002.json \
  --output quant_evaluator/docs/benchmarks/f48_cap16_ordinary_UNIQUE.json
```
