# 默认 source auto：跨进程候选与当前依赖校验

## 如何接入已测后端

程序初始化时显式配置可信报告位置，后续请求仍使用普通 `backend="auto"`，
不必每次传入 `source_qualification`。配置本身不读文件、不预热、不测速：

```python
from quant_evaluator.scripts.source_profile_report_reader import load_source_profile_report
from quant_evaluator.runtime.source_qualification_provider import (
    FileSourceQualificationProvider, configure_source_qualification_provider,
)
report = load_source_profile_report(report_path)
fingerprint = report.records[0].context.request_content_sha256
configure_source_qualification_provider(
    FileSourceQualificationProvider({fingerprint: report_path})
)
bundle = evaluate_factor_source_batch(source, labels, metrics=metrics,
    backend="auto", max_tile_size=requested_cap, gpu_policy=policy)
```

默认路由顺序：显式资格记录优先；否则查当前进程资格缓存；缓存缺失时才查询已配置
报告候选。首次报告候选通过当前上下文校验后进入缓存，后续同请求不重复读报告。
报告不是签名证明；配置是调用者的可信边界。它不会自行扫描目录、读取环境变量、
执行 pilot、benchmark 或冷启动 CUDA。未配置时保持原有保守 auto。

## 判定规则与状态

CPU/CUDA 两个相反测试顺序必须都有同一个严格耗时赢家，且完整数值、计数、
NaN/Inf 掩码、覆盖范围、分块配置和零 OOM 校验均通过。只能使用已测宽度，不能
把未测宽度或其他请求的局部快内核结果当作整个请求的最快配置。

检查 `bundle.metadata`：`source_qualification_applied` 才说明实际应用；
`source_qualification_origin` 区分 `explicit_receipts`、`process_cache`、
`report_candidate`、`none`。`source_qualification_provider_status` 区分
`candidate_validated`、`candidate_rejected`、`candidate_guard_error`、
`candidate_execution_deviated`、`provider_error`、未配置或未查询。
`source_qualification_cache_status` 在首次报告准入时仍为 `supplied`，不是缓存命中；
以后同进程才为 `cache_hit`。`source_qualification_candidate_id` 不包含配置路径。

## 安全与覆盖边界

最多显式配置 32 个精确请求映射，每份报告只读 1 MiB 加一个超限哨兵字节。
报告必须是普通文件；FIFO、目录等拒绝，错误状态不暴露路径或原始异常。
fork 子进程会清除 provider 配置；每个新进程须由应用显式重新配置。CUDA 运行时
仍须显式准备；冷进程缺少当前 CUDA 身份时不能通过性能资格。

当前文件 provider 使用严格 F48 Pearson 四指标报告读取器，不覆盖任意指标集合、
形状或不同因子集合。其他请求保守回退；普通非 source batch API 暂不使用此 provider。

运行时身份 v3 纳入已加载 DuckDB、PyArrow、Polars 的版本；未加载时不主动导入。
source 资格的执行摘要严格覆盖已加载本地 QE、DataAccess、FactorOptimizer、
FactorPreprocess 的 Python 文件树，采用跨组件总文件数/字节预算，缺失或扫描失败
拒绝资格，变化时淘汰旧缓存。它是磁盘源内容与包根身份，不是完整已加载字节码或
外部二进制依赖闭包证明；不把未执行的 FE 整树冒充当前 F48 执行依赖。
