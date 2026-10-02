# F48 真实源批评估：auto 验证与准入边界

## 当前状态

维护者已保存 2,586 日×5,461 股票×48 因子的 CPU/严格 CUDA 源端 A/B。
两份记录采用相反运行顺序，比较 RankIC、分位收益差和因子换手率。
生产选择器仍将该条证据标为 `measured_source_ab_pending_auto`，不会凭这两份
显式后端记录启用默认 GPU 路径。

`--auto-f48-references` 只在独立基准进程内注入待验证的 auto 候选。
候选状态为 `benchmark_only_pending_auto`；生产证据注册表保持原样。
维护者需检查实际 auto 输出、显式 CUDA 输出、相反顺序源 A/B 的输出摘要、
有限值掩码及观测计数，再决定生产准入。你不能把注入候选的成功记录称为
普通默认 auto 已经验证。

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
  --auto-f48-references \
  quant_evaluator/docs/benchmarks/real_cos_f48_source_cpu_first_telemetry_20261002.json \
  quant_evaluator/docs/benchmarks/real_cos_f48_source_cuda_first_telemetry_20261002.json \
  --output quant_evaluator/docs/benchmarks/f48_pending_auto_verification_UNIQUE.json
```

先将 `UNIQUE` 换成新的运行标识，避免覆盖已有证据。这条命令读取现有授权
COS 数据，不发布生产因子。

## 记录可信度

新记录包含声明范围内 QE Python 源文件的运行前后摘要，范围涵盖计算模块和
列出的基准辅助脚本。实现限制文件数量和读取字节数，拒绝路径越界与符号链接。
源文件、声明范围或受检查的版本变化时，成功状态会失效。
这项检查不覆盖已导入的内存代码、DataAccess/Factor Optimizer 的传递依赖、
GPU 驱动或硬件等价性。历史记录缺少的 caller settings 和代码摘要保持未知，
维护者不能通过补写字段或比较数据摘要来追认它们。
