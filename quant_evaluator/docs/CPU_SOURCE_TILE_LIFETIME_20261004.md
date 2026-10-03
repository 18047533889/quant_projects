# CPU source 分块生命周期修复（2026-10-04）

## 问题与修复

`evaluate_factor_source_batch(..., backend="cpu")` 原来在下一次读取的右侧表达式求值时，仍保留上一块 `tile`。即使 source 严格遵守每次读取的宽度上限，也会同时持有旧输入和正在分配的新输入。

在每块的指标值与计数已经复制到最终 columnar 输出后，明确释放 tile、EvaluationBundle、最后一个 artifact 与 values，并清空 grouped metrics 局部引用。不会改变指标公式、读取范围、最终输出、结果预算、错误传播、auto 资格检查或调用方的 source.close 责任。

## 行为证据

小型独立分配 source：T=8、N=48、F=5，分块宽度2，最后一块宽度1。修复前弱引用诊断为下一次读取前仍存活的 tile 数 `[0, 1, 1]`，整批参考值和计数相等。

新增测试对 tile、FactorBatch、values、validity 同时持有弱引用，直接在下一次分配前检查全部已释放，不依赖手动垃圾回收。测试包括标量、序列、混合以及 rank/coverage/quantile 多指标请求，并核验全量参考值与计数、完整读取范围、最后一块退出后的释放。

相同测试命令修复前4失败，修复后4通过。相关公共批量入口、shape、输出身份、live guards、default cache、provider 联合测试：68通过、18 deselected、4个 All-NaN slice 警告，3.44秒。`-k "not cuda"` 仅按测试名过滤，部分公共 auto 测试仍会执行小型 GPU 路径，不能称为纯CPU测试。

第二组验证（生命周期、CPU OOM 回执、source 内存预算、catalog、profile API、两类报告 reader、依赖身份）：89通过，1个已有 fork deprecation 警告，3.60秒。独立 Luna 只读审查检查了结果复制时机、series-only 局部变量、空 metrics 的前置拒绝和异常退出路径，未发现阻塞问题；未独立运行测试，也未覆盖所有 catalog 指标。

## 证据边界

这证明正常完成每块计算时输入生命周期的边界，不是进程 RSS 测量或端到端性能 A/B；不宣称所有输入都无 bug、全链最快或不会 OOM。底层 source 自己持有缓存、allocator 保留内存、最终输出容量和临时内核内存仍须分别预算。

真实 COS 的 F48、全股票、多年 CPU/CUDA ABBA 由独立窗口协调运行；本修复后必须重新捕获严格源码身份，旧资格不能自动继承。
