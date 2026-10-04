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

`evaluate_factor_source_batch` 的默认 `source_auto_policy="qualified_only"`
只允许通过当前上下文验证的 CPU/CUDA 赢家决定 auto 路由。没有当前资格时
使用 CPU，不再让历史静态形状记录独自授权默认 GPU。这个安全回退不是
“CPU 已测得最快”的声明：应用仍须提供匹配当前源码的批量性能报告。

确需复现历史静态路由时，必须显式传 `source_auto_policy="legacy_measured"`：

```python
bundle = evaluate_factor_source_batch(
    source, labels, metrics=metrics, backend="auto",
    source_auto_policy="legacy_measured",  # 显式历史研究选项，非当前性能资格
)
```

历史选项仅在没有显式资格、缓存真正缺失且 provider 未配置或没有精确请求
候选时开放；拒绝的候选、坏缓存、provider 读取/解析错误、守卫错误和显式
坏资格即使开启此选项也不能再授权历史 GPU。严格缓存查找区分真正的缺失
与已淘汰的坏记录，保留拒绝原因，防止后续缺失覆盖它。已有 getter 调用者
的非严格默认行为保持兼容。有效当前资格仍优先于历史选项。

两个选项只接受精确内置字符串；错误值在 source 元数据/因子读取前拒绝。
`backend="cpu"` 与显式 `backend="cuda_strict"` 不依赖历史授权。
回执与 metadata 记录 `source_auto_policy`；历史兼容的资格状态名称中
`*_legacy_fallback` 字面后缀不代表静态路由获准，应同时检查实际 backend、
auto 原因与资格 applied 字段。历史路由仍主要绑定形状、指标、dtype 和 tile，
不具有实时资格的完整源码/设备/请求绑定。

研究基准 CLI `python -m quant_evaluator.scripts.benchmark_real_cos_source_batch`
提供同名语义的 `--source-auto-policy qualified_only|legacy_measured`，默认也是
qualified_only。既有 `--verify-auto` 等历史研究开关不会暗中开启历史路由；
要复现旧门槛必须明确选择 legacy_measured。函数 `run_backend` 转发选项并
核对实际输出 metadata 的 policy，再记录到回执。新六指标 provider 验证
工具不提供历史模式，必须验证当前资格而非旧路由。

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

## 输出契约与资格撤销

每次 source-batch 执行（含显式 CPU/CUDA 与未取得资格的 auto）都先校验最终
bundle，再读取执行元数据或交给性能资格校验。因子 ID 及顺序、标签 ID、指标
分组与键、标量 `(F,)` / 序列 `(T,F)` 浮点数组、逐指标 `(F,)` 整数计数
必须符合请求；负计数、超出 int64 范围的无符号计数、意外向量指标等拒绝。
这些是输出结构错误，抛出 `InvalidContractError` 前淘汰对应性能资格缓存。

结构合法但数值或计数摘要与资格报告不同，则保留已有语义：撤销本次性能资格、
淘汰缓存，但返回结构合法的结果。摘要漂移本身不证明算法算错；因子顺序变化
不是这一类可返回漂移。调用者应同时检查资格状态，不只检查 `backend_used`。

## 安全与覆盖边界

最多显式配置 32 个精确请求映射，每份报告只读 1 MiB 加一个超限哨兵字节。
报告必须是普通文件；FIFO、目录等拒绝，错误状态不暴露路径或原始异常。
fork 子进程会清除 provider 配置；每个新进程须由应用显式重新配置。CUDA 运行时
仍须显式准备；冷进程缺少当前 CUDA 身份时不能通过性能资格。

当前严格读取器支持两类固定 F48 报告：原 Pearson 四指标报告，以及六项线性
分层形状指标报告（`quantile_curvature`、`quantile_tail_asymmetry`、
`quantile_adjacent_spread`、`quantile_extreme_cliff`、`top_quantile_cliff`、
`bottom_quantile_cliff`）。后者要求四次 ABBA、显式资格 auto 和缓存命中默认 auto
两次验证全部完成，并绑定各后端自己的值、有限性和计数摘要。
这不覆盖任意指标集合、形状或不同因子集合；普通 `evaluate` / `evaluate_many`
暂不使用此 provider。旧 Pearson 的 provider 验证文件不能证明新六项指标的跨进程接入。

### 默认行为不是自动启动发现

截至本次只读审查，生产应用未配置此 provider，FE reporting adapter 使用通用
`evaluate`，未发现生产路径调用 source-batch API。未配置的新进程会报告
`provider_not_configured`。因此“普通 auto 不必逐次传资格”仅在该进程先显式初始化
provider、且请求匹配已测报告时成立；不能将同进程的 `cache_hit` 宣称为重启后天然
可用，或将 API 支持宣称为生产部署已接通。本轮没有部署或发布生产因子。

应用需在实际执行评估的进程中提供受管理的指纹到报告路径映射，不能让普通请求
任意选择服务器文件。报告解析不等于签名认证，仍要保留 live source/runtime/device、
请求、宽度、输出身份验证。缓存未命中时 provider 不会自行测速或发现最快后端。
实际集成应采用 source-batch 入口并保留准确的 qualification 状态与保守回退；
仅在服务启动处配置 provider，不会改变仍使用通用 evaluate 的调用链。

运行时身份 v3 纳入已加载 DuckDB、PyArrow、Polars 的版本；未加载时不主动导入。
source 资格的执行摘要严格覆盖已加载本地 QE、DataAccess、FactorOptimizer、
FactorPreprocess 的 Python 文件树，采用跨组件总文件数/字节预算，缺失或扫描失败
拒绝资格，变化时淘汰旧缓存。它是磁盘源内容与包根身份，不是完整已加载字节码或
外部二进制依赖闭包证明；不把未执行的 FE 整树冒充当前 F48 执行依赖。
