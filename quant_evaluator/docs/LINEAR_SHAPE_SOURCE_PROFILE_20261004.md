# 六形态指标真实 COS 批量性能资格

## 范围与使用

`scripts/benchmark_real_cos_linear_shape_profile_abba.py` 只测固定的
六形态计划：curvature、tail asymmetry、adjacent spread、extreme cliff、
top cliff、bottom cliff。完整公式见 `GPU_LINEAR_SHAPE_PUBLIC_20261004.md`。
固定默认 Q=5、每桶最少 10 个标签、每桶 profile 最少 20 个有效日期。
20 是六项指标各自 `MetricSpec.min_periods` 的固定门槛，不是这里可传入的
调用参数。公开 `evaluate/evaluate_many` 的 CUDA profile 路径支持
`quantile_builder_parameters` 中的 `n_quantiles`（Q）/`min_assets`；显式
`window_size` 不支持。
本 source F48 测速工具不传这些 overrides，其资格仅证明默认 5/10/20
请求，不能据此宣称非默认分桶参数也已完成最快后端校准。

目标请求为真实 COS 的 `(T,N,F)=(2586,5461,48)`，输入轴索引与原先
Pearson 大批量验证共用。48 个因子不是只测试两个因子。此计划计时包括
公开 source API 的读取、分桶、日桶收益、时序均值、形态计算和最终落值，
不能用单个 `(Q,F)` 小 kernel 的提速替代端到端证据。

独立 oracle 在计时前读取全部 F48 tiles，之后还有 CPU/CUDA 小请求预热。
因此现有 DataAccess/COS 本地对象缓存可能已经温热。本工具验证的是
**温热来源缓存条件下的完整 API wall time**，不声称冷 COS 首次下载时延。
四个 ABBA arm 都重新构建相同 source，预取策略和预算一致；不为制造
冷缓存而删除正式缓存或复制数据。冷来源性能需要另外设计并明确标记
场景，不能把这份资格泛化为任意 I/O 状态或所有指标的最快后端。

默认调用仅检查资源，不读取 COS、不创建 source、不做 GPU warmup：

```bash
cd /home/sunhaiwei/quant_projects
.venv/bin/python -m quant_evaluator.scripts.benchmark_real_cos_linear_shape_profile_abba
```

只有发布和资源协调完毕、其他性能任务结束后，才启动显式实测：

```bash
.venv/bin/python -m quant_evaluator.scripts.benchmark_real_cos_linear_shape_profile_abba \
  --run \
  --axis-index quant_evaluator/docs/benchmarks/real_cos_f48_axis_index_20261004.json \
  --output quant_evaluator/docs/benchmarks/real_cos_f48_linear_shape_profile_abba_20261004.json
```

输出和对应 `.progress.json` 路径都必须是新路径。不要覆盖他人的证据，
也不要仅因观察超时重启仍活着的进程。失败进度不是性能资格。

## 校验及确定性 auto

独立 reference 位于 `scripts/source_linear_shape_oracle.py`：因子有效性
决定分位边界；标签有效性只用于桶内收益；独立排序、线性 percentile、
tie-max、有限桶均值、时序门控和六公式，不调用生产分桶/profile/shape
数值函数。危险尺度和 shape 线性公式采用二进制有理数；curvature 必须
对所有有效 stencil 累加后再除个数、最后只舍入一次。真实 ±Inf 保留在
结果中，typed count 仍为 `int(isfinite(value))`，不能改记成 NaN。

CPU/CUDA/CUDA/CPU 四次 ABBA 分别与 reference 比较每项全部 F48 数值，
以及 shape、有限/NaN/±Inf 掩码、0/1 counts、完整覆盖和容差。资格记录
还约束当前源码、依赖、输入、参数、线程、设备、实际 tile 与零 OOM。
之后再次实跑显式记录 auto，再实跑不传记录的进程内 default auto，确认
后者 cache hit。新进程 provider 的验证是另一个步骤，不由进程内命中推断。

严格 reader 以独立 schema `real_cos_profile_abba_f48_linear_shape.v1` 解析，
不可冒充旧 Pearson 报告；字段、六项顺序、F48 形状和默认 auto 验证固定。
只有当前上下文验证通过的完整报告才能复用；修改 Python 源码后旧资格
不能继续冒充 current source，也不能据此扩大所有指标/所有数据的 auto 范围。

## 内存与现有证据

参考计算默认一次只保留一个 factor tile（约 108 MiB 的 T×N float64
因子值，不含 source/解码临时空间）；不构建全 F48 的 5.4 GB 因子 cube。
六结果与计数共 `96*F` 字节，首读前检查输出预算；行视图必须在下次读
tile 前释放。这个预算不是整个进程 RSS 上限。DataAccess source 采用
现有 COS 读法与有界预取，禁止复制仓库、数据集或临时堆积大型产物。

2026-10-04 最新独立 oracle、CLI 安全编排与六形态严格报告 reader 联合
测试实跑 37 passed / 2.62 秒（终端 session 63869，退出码 0），包含极端
数值、Inf mask、弱引用 tile 生命周期、默认 dry-run、资源拒绝、输出不覆盖
及真实 typed-record/report roundtrip；CLI 的数据读取与计时执行仍使用测试替身。
这些不是性能测量。公开与 source CUDA 接线已通过 22 项实测并双仓发布；
独立发布读回证据保存在 `benchmarks/linear_shape_publication_verified_20261004.json`。
**真实 F48 六形态 ABBA 尚未运行，最快后端与性能资格仍未确认。**

## 下一轮性能定位：设备标量同步

只读源码审查发现 `kernels/gpu/shape_linear_numeric.py` 的每指标精确修复
路径分别读取 guard error flag 和 risk count；风险列存在时还读取精确修复
error flag。六项指标即使没有风险列，也会在最终结果一次打包回传前执行
12 个显式 `.item()` 标量读取；风险路径最多有 18 个显式读取，此外
`nonzero` 等操作可能另有同步。这里不把标量读取误称为全输入 CPU fallback。

这是待测的延迟来源，不是已经证明的端到端瓶颈。下一步先完成固定源码的
真实 F48 ABBA 基线，再单独测 guard 元数据合并读取是否减少同步并带来
稳定收益。任何改动都必须保留错误 fail-closed、危险尺度精确修复、工作
空间 admission 和数值/counts parity；不得通过去掉校验获得速度。修改
Python 源码后重新生成性能资格，不能沿用旧源码的 auto 记录。

## 进度证据的并发写入保护

测速 CLI 现使用独立模块 `scripts/profile_progress_writer.py`：在 ABBA
开始前排他创建进度文件，随后仅通过已持有的文件描述符更新；即使路径
后来被替换，也不会改写另一个任务的文件。完整 JSON 序列化和 1 MiB
预算检查发生在文件内容修改前，结束及异常路径均关闭句柄。进度始终
标记 `qualification_available=false`，不能当作完整性能资格。

晚创建进度文件的集成反例在旧实现上实际失败（session 99285，退出码 1），
修复后联合独立 oracle、CLI、严格 reader、writer、运行时输出绑定及
live guards 实跑 73 passed / 1.11 秒（session 14637，退出码 0）。包括
五个 writer 专项测试及真实进度 JSON 检查，仍不是 COS 端到端性能证据。

随后新增 Ctrl-C 反例：旧实现中断后进度仍为 `running`（session 47478，
退出码 1）；修复后保留 `KeyboardInterrupt` 抛出，同时记录失败类型并
关闭句柄。最新同范围联合实跑 74 passed / 1.13 秒（session 79426，退出码 0）。
