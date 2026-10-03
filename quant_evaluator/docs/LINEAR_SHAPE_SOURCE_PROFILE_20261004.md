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

默认调用仅检查资源和已配置 CLI 是否可执行；不执行 CLI、不读取 COS、
不创建 source、不做 GPU warmup：

```bash
cd /home/sunhaiwei/quant_projects
ASHARE_PARQUET_ROOT=/home/sunhaiwei/cos_data \
DATA_ACCESS_COS_CLI=/usr/local/bin/admin-cos \
DATA_ACCESS_COS_CACHE_ROOT=/home/sunhaiwei/.cache/quant-dataaccess/research \
.venv/bin/python -m quant_evaluator.scripts.benchmark_real_cos_linear_shape_profile_abba
```

这些是 server-c 已有且此前 F48 已使用的 research 配置，不是新增权限。
不得猜凭据、改 COS 目标、自动替换为底层 `coscli` 或绕过研究数据授权。
其他环境必须使用各自已经配置并授权的入口，不能照抄管理员身份或新增
权限。此测试不支持 strict/production，也不发布生产因子。

只有发布和资源协调完毕、其他性能任务结束后，才启动显式实测：

```bash
ASHARE_PARQUET_ROOT=/home/sunhaiwei/cos_data \
DATA_ACCESS_COS_CLI=/usr/local/bin/admin-cos \
DATA_ACCESS_COS_CACHE_ROOT=/home/sunhaiwei/.cache/quant-dataaccess/research \
.venv/bin/python -m quant_evaluator.scripts.benchmark_real_cos_linear_shape_profile_abba \
  --run \
  --axis-index quant_evaluator/docs/benchmarks/real_cos_f48_axis_index_20261004.json \
  --output quant_evaluator/docs/benchmarks/real_cos_f48_linear_shape_profile_abba_shared_20261004_r2.json
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
当时真实 F48 六形态 ABBA 尚未运行；后续成功结果与适用范围见下文。

## 下一轮性能定位：设备标量同步

旧实现每指标分别读取 guard error flag 和 risk count；风险列存在时还读取
精确修复 error flag，六项指标合计 12 个显式 `.item()`，风险路径最多 18 个。
2026-10-04 修复只删除从不被 guard 内核写入的 flag 读取；保留紧随其后
的 risk count 同步和精确修复错误检查，分配、内核参数与 workspace 预算不变。
现在六指标安全路径为 6 个显式读取，风险路径最多 12 个；此外
`nonzero` 等操作可能另有同步。这里不把标量读取误称为全输入 CPU fallback。

同步次数反例先失败后通过；修后独立运行真实 CUDA/source、Fraction 极值
及 CPU 边界联合测试 25 passed、1 skipped（仅单 GPU 无法测试跨设备）。
这证明本轮测试范围的正确性和显式同步减少，不是端到端性能收益。
旧真实 F48 基线已完成；新源码需重新做批量 A/B，再判断是否有
稳定收益。任何改动都必须保留错误 fail-closed、危险尺度精确修复、工作
空间 admission 和数值/counts parity；不得通过去掉校验获得速度。修改
Python 源码后重新生成性能资格，不能沿用旧源码的 auto 记录。

guard-only 微型真实 GPU A/B 见 `benchmarks/shape_guard_sync_micro_ab_20261004.json`：
Q=5、F=5，六项各重复 100 次，三组交错顺序中新实现两组更快、一组更慢。
它不计算完整指标或读取 COS，不构成稳定加速证据；保留混合结果供后续批量验证。

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

## 首次真实启动的诊断与研究入口预检

首次真实启动已实际终止失败（session 57673，PID 1139631，退出码 1），
尚未进入 ABBA。默认 `clean-cos-ro` 不在当前 SSH PATH；DataAccess 原本
把缺少命令归为未知对象元数据并拒绝读取。这里不放宽拒绝语义，而用
`scripts/research_cos_cli_preflight.py` 在任何 COS 调用前检查现有配置入口。
缺少或不可执行时给出不含路径/凭据的明确错误，不猜替代 CLI；成功后
仍返回原有资源 preflight 字段，严格报告 schema 不变。入口存在不等于
授权通过，后续完整授权、ETag、内容摘要和预算检查继续由 DataAccess 执行。

安全诊断见 `benchmarks/linear_shape_cli_gateway_diagnostic_20261004.json`；
已有授权入口的 metadata 探测及有界 manifest/轴索引验证已通过，未构建
因子 cube、未开始计时。新 gate 与原有范围联合实跑 79 passed / 0.99 秒
（session 9508，退出码 0）。失败资源侧车保留，不能作为性能资格。

## 真实 F48 六形态 ABBA 完成（2026-10-04）

权威报告为 `benchmarks/real_cos_f48_linear_shape_profile_abba_shared_20261004_r2.json`，
严格 reader 已独立读取通过，状态为 `complete`。输入规模为 2586 个交易日、
5461 个资产、48 个因子；只覆盖本文六项形状指标，不是全部 QE 指标。
CPU 两轮完整 API 耗时为 119.579、126.254 秒；CUDA 为 69.193、64.373 秒。
四轮均通过独立 oracle，未发生 OOM，实际 tile 宽度均为 5。
显式资格与同进程默认 `auto` 均选择 CUDA，输出通过独立参考校验，后者命中缓存。
这是共享服务器、已预读 COS 缓存条件下的有限样本，不证明其他请求都最快。
源码依赖改变后必须重新验证资格。此前 `real_cos_f48_provider_verification_20261004.json`
仅是 Pearson 四指标的新进程证据，不能替代六形态的新进程 provider 验证。

已确认成功路径没有把 `.progress.json` 从 `running` 更新为 `complete`；
r2 进度含四轮记录但状态仍为 `running`。最终报告、严格 reader 和退出码
是该历史运行完成依据，不要据旧进度状态重启。本轮已修复成功路径终态，
旧 r2 文件保留原始证据，不回写历史状态。如果最终报告保存成功但诊断
终态写入失败，异常继续传播，却不把已完成的报告/计算误标为失败。
成功路径与模拟诊断 I/O 失败路径均先观察到状态反例，再修复通过。
修后 CLI/writer、同步错误检测、实际 CUDA/source、Fraction 极值与 CPU
边界联合独立实跑 48 passed、1 skipped（单 GPU 无法测试异设备），diff 检查通过。
这不是新源码的真实 COS 性能资格；提交后仍须重新生成并验证性能报告。
资源侧车的 `validated_runs` 是资源快照，不是通过 oracle 的指标运行次数。
