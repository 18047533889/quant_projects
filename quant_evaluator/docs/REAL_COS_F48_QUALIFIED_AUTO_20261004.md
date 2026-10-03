# F48 全市场多年 source auto 实测（2026-10-04）

## 已完成的实际验证

正式证据：`benchmarks/real_cos_f48_profile_abba_20261004.json`。
严格 typed report reader 已接受 v2 complete 报告。

请求为 2586 个交易日 × 5461 个资产 × 48 个真实 COS 因子。
指标为 pearson_ic、pearson_ic_series、pearson_ic_std、pearson_ic_ir。
请求分块上限 16，CPU/CUDA 实际有效宽度均为 5，不能视为宽度 16 的测速。

| 顺序 | CPU 秒 | CUDA 秒 | CUDA 耗时减少 |
| --- | ---: | ---: | ---: |
| CPU → CUDA | 107.758550162 | 69.416602137 | 35.58% |
| CUDA → CPU | 91.634351645 | 75.335984958 | 17.79% |

计时范围是整个 evaluate_factor_source_batch 调用，包含 source 读取和计算，
不包含独立 oracle、运行时预热及调用前后上下文捕获；不是单个 GPU 内核测速。
两种顺序同一严格赢家 CUDA。四轮均通过独立 SciPy/Decimal Pearson 链的全输出、
计数和 NaN/Inf 掩码校验，零 OOM 重试。

第五轮显式 supplied qualification 与第六轮省略 source_qualification 的默认 auto
均实际选择 CUDA，并通过独立 oracle。第六轮 cache_status=cache_hit，
qualification_applied=true，status=qualified_current_source。

## 新进程文件 provider 实测

证据：`benchmarks/real_cos_f48_provider_verification_20261004.json`。
新进程显式准备运行时并配置精确请求报告映射，清空资格缓存，不传 source_qualification。
实际 origin=report_candidate，provider_status=candidate_validated，cache_status=supplied，
qualification_applied=true，status=qualified_current_source，backend_used=cuda。
实际有效宽度 5，零 OOM，API 耗时 63.643510069 秒；四指标完整独立 oracle 均通过。
这是新进程真实接入证据，不是第三个 ABBA arm，也不是“冷启动 CUDA 自动初始化”的承诺。
执行代码由 stdin 运行，复用既有 helpers，未新增 Python 源文件或改变被测源码摘要。

## 边界

该证据只证明精确 F48 数据/指标集合、已测宽度、策略、源码与运行时上下文的准入。
不证明所有指标、任意因子集合或其他宽度的最快后端，不证明程序绝无 bug。
每个新进程须显式配置可信报告 provider，并准备所需 CUDA 运行时；过期上下文拒绝。
`.progress.json` 只是四轮诊断，不是正式资格，也不能从 running 字段推断进程仍活着。
