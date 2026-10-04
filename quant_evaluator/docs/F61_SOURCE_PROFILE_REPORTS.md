# F61 全市场批量评估：资格报告与默认 auto

## 当前实现与验收边界

已实现 F61 的闭合报告协议、严格读取器和文件 provider 接入测试。它们是取得当前性能资格所需的基础设施，不等于已经取得真实 F61 性能资格。单元测试中的合成报告不可以作为生产路由证据。

真实运行仍需在固定源码和环境下完成完整 ABBA、逐指标独立 oracle、资格 auto、默认缓存命中 auto，以及新进程 provider 验证。真实 F61 全量结果取得前，不能声称 24 指标默认 GPU 已验收或全指标最快。

## 两个独立报告域

- `real_cos_profile_abba_f61_all15.v1`：历史 15 指标的固定顺序，见 `F61_ALL15_METRICS`。
- `real_cos_profile_abba_f61_all24.v1`：固定 24 指标，见 `F61_ALL24_METRICS`；前 15 项与历史顺序一致，随后为六个分层形状指标和三个 RankIC 摘要。

这两个常量来自 `runtime/source_profile_report_schema.py`，不是从 frozenset 迭代顺序推导。两域均仅允许形状 `(2586, 5461, 61)` 和请求 tile 上限 16；不授权任意因子数、任意指标子集、其他市场形状、派生来源或多 horizon 调度。15 指标报告不能冒充 24 指标报告。

这里的 2586 日是标签对齐后的评估轴，不是原始因子轴。现有 `docs/benchmarks/real_cos_f61_axis_index_20260929.json` 的原始轴为 `(2588, 5461, 61)`；注册的 t+1 到 t+2 收益标签可能裁掉末尾两日。两个入口共享 `scripts/source_f61_axis_admission.py`：加载标签前仅允许 2586–2588 日，股票和因子数仍须精确匹配；加载标签后必须严格等于 `(2586, 5461, 61)`。原始 `source_rows` 身份与全因子面板轴摘要保持不变，不裁写轴索引、不扩展资格协议。索引文件本地校验和通过不代表其 COS 内容已在本轮重新验收。

独立 oracle 名称固定为 `independent all-source metric chain`。实际计算口径参见 [指标参考](METRIC_REFERENCE.md)；报告协议不修改公式，也不认证报告生产者。

## 怎样确定获胜后端

测速顺序为 CPU、CUDA、CUDA、CPU，组成两个执行次序相反的配对。计时是完整 `evaluate_factor_source_batch` API 调用，不是孤立内核时间。

设 CPU 的两次完整耗时为 $t_{C,1},t_{C,2}$，GPU 的两次耗时为 $t_{G,1},t_{G,2}$：

$$\bar t_C=\frac{t_{C,1}+t_{C,2}}2,\qquad\bar t_G=\frac{t_{G,1}+t_{G,2}}2.$$

两个配对必须分别得出同一个非平局胜者，均值比较也必须与该胜者一致；执行次序改变胜者、耗时非法、OOM 重试或正确性证据缺失，均不授予资格。实现采用稳定的正数均值计算以避免极大计时溢出。

因此“最快”指这个精确请求与已测 CPU／CUDA 配置中通过全部门槛的稳定胜者，不是未经测量的全局最优。随机配对测试是稳定性补充，不能自行代替资格协议。

## 覆盖数、误差与输出身份

`rank_ic_series` 和 `pearson_ic_series` 的覆盖为 $2586\times61=157746$；其余 source 指标的覆盖为 61。覆盖指完整输出位置数，包含 NaN 位置，不是有效日数或观测样本数。每个输出还另有 observation counts 身份。

新协议的逐指标最大绝对误差门槛固定为 $0\le e_m\le10^{-10}$。报告不能自行缩小覆盖范围或放宽容差，即使 context、profile 和 oracle 在内部相互一致，也会被拒绝。

显式资格 auto 和随后默认 auto 的回执必须同时满足：

- `backend_requested="auto"`、`source_auto_policy="qualified_only"`；实际后端与资格胜者一致。
- `metric_outputs` 是固定指标顺序的完整六字段 `MetricOutputReceipt` 列表：metric_id、values_sha256、finite_mask_sha256、observation_counts_sha256、coverage_expected、coverage_observed。
- 上述列表逐项等于选中后端的 profile 输出；原 `values_sha256` 映射也必须与选中后端一致。CPU 和 GPU 在容差内可以有不同数值摘要，不要求跨后端逐位相同。
- auto oracle 的 run index 为 4，默认 auto 的 run index 为 5 且 cache_status 为 cache_hit；四个原始测速 oracle 也必须各自完整通过。

JSON 对象的键插入顺序不是指标顺序；有序指标元组和 metric_outputs 列表才是顺序契约。

## 已有真实报告时如何接入

下面只展示接线。source、label_bundle 和 gpu_policy 必须由实际任务准备，并与报告测量的来源、标签、完整轴、dtype、GPU 策略及环境完全一致。不要把测试 fixture 或旧报告替换成“当前资格”。新进程的 CUDA 显式准备也必须遵循真实验收流程，本片段不代替它。

```python
from quant_evaluator.api.factor_source import evaluate_factor_source_batch
from quant_evaluator.runtime.source_profile_report_schema import F61_ALL24_METRICS
from quant_evaluator.runtime.source_qualification_provider import (
    FileSourceQualificationProvider, configure_source_qualification_provider,
)
from quant_evaluator.scripts.source_profile_report_reader import load_source_profile_report

# report_path 是实际生成并审查的完整 F61 all24 报告候选，不是合成示例。
# 读取仅校验结构；当前 source/runtime/device/request 复核后才可能获路由资格。
report = load_source_profile_report(report_path)
request_fp = report.records[0].context.request_content_sha256
configure_source_qualification_provider(
    FileSourceQualificationProvider({request_fp: report_path})
)
bundle = evaluate_factor_source_batch(
    source, label_bundle, metrics=F61_ALL24_METRICS,
    backend="auto", max_tile_size=16, gpu_policy=gpu_policy,
    source_auto_policy="qualified_only",
)
```

这里不显式传 source_qualification，由普通 auto 查找 provider 候选，再绑定当前环境。provider 配置上限为 32 个精确请求指纹；不自动扫描文件，不隐式校准。fork 后 provider／资格缓存清空，各评估子进程需显式配置。

该 source API 返回 `BatchEvaluationBundle`：标量在 scalar_metrics，两个序列在 series_metrics，计数在 observation_counts，路由证据在 metadata。它不同于公开 `evaluate()` 的 EvaluationBundle；后者的序列通常在 artifacts[id].values，不能混用容器接口。

## 选项和失败策略

可顺序复用同一个 source，不必为第二次评估清空读取历史或新建 source 来迁就资格检查。资格执行检查在本次计算前记录不可变历史前缀，并要求：

$$H_{\mathrm{after}}=H_{\mathrm{before}}\mathbin{\Vert}R_{\mathrm{profile}},$$

其中 $\Vert$ 表示按顺序拼接，$R_{\mathrm{profile}}$ 是所选后端本次应执行的 tile 区间。历史前缀必须保持不变，每个区间必须由两个整数构成并满足 $0\le start<end\le F$，本次新增区间和本次输出的 tile 数必须精确匹配 profile。历史被删除、修改，新增区间缺失或多出，都撤销资格；不能拿上次日志冒充这次执行。该检查不保证同一 source 的并发使用线程安全；共享 source 并发追加也不能绕过回执核验。

小规模 CUDA 复用回归入口：

```bash
QE_RUN_SOURCE_PROFILE_CUDA=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m pytest -q \
  quant_evaluator/tests/test_source_profile_reused_source_oct04.py
```

这里的 CUDA 用例为 65 日、64 资产、5 因子的单个 RankIC 指标：实际 GPU 输出对照 SciPy，并验证同一 source 三次 auto 和后两次缓存命中。上下文及 CPU/CUDA 耗时是测试 fixture，不是实际性能测量；该回归不能替代 F61 全量股票、多年历史、完整指标域的 COS 正确性与测速资格。

backend 支持 auto、cpu、cuda_strict。auto 默认 qualified_only：缺少当前资格时采用 CPU；错误或过期证据也不能恢复历史静态 GPU 授权。legacy_measured 仅是显式历史兼容选项，不属于新 F61 报告的合格 auto 回执。

max_tile_size 控制来源读取上限，gpu_policy.max_factor_tile_size 控制 GPU tile 上限；实际 tile 必须满足来源、主机内存和显存准入。改变策略、指标顺序、标签、来源内容、源码、包、线程或设备上下文后，旧报告不再自动具有当前资格。

准入上限不等于实际执行宽度：例如准入上限为 5，而稳定获胜的 GPU 配置每块执行 4 个因子。已获资格的 auto 必须沿用实测宽度 4，不能把它改写成上限 5，也不能修改原 context 来迁就回执。`source_f61_auto_verification.verify_source_profile_auto` 对实际执行做独立 oracle 对照，再核验选中 profile 的完整输出、分块计划和配置身份。底层回执构造器的 `qualified_execution_cap` 仅用于已获资格的 qualified_only auto；默认显式后端的原准入校验不变。

## 真实测速入口

在 server-c 正式总仓库目录执行；以下入口已具备安全与执行编排测试，不代表已有真实 F61 性能资格。

新 SSH 会话不继承其他窗口的 research 环境。当前 server-c 已配置并通过实机预检的入口如下；这些不是凭据或新的 COS 授权目标，不要自行改成其他 bucket：

```bash
export ASHARE_PARQUET_ROOT=/home/sunhaiwei/cos_data
export DATA_ACCESS_COS_CLI=/usr/local/bin/admin-cos
export DATA_ACCESS_COS_CACHE_ROOT=/home/sunhaiwei/.cache/quant-dataaccess/research
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
```

此配置只在当前终端会话生效，不修改用户的全局环境。环境缺失导致 `configured_cli_resolvable=false` 时，先核对已有配置，不猜测凭据或把默认 CLI 不可解析误判为 COS 服务不可达。线程配置也是测速环境身份的一部分；实际执行必须与报告保持一致。

```bash
.venv/bin/python -m quant_evaluator.scripts.benchmark_real_cos_f61_profile_abba --metric-set all24
```

不传 `--run` 只检查现有 COS CLI 和资源余量并输出 `preflight_only`，不会读取因子数据或生成性能资格。`--metric-set all15` 是另一个独立域；默认域为 all24。

真正运行必须显式提供现有且核验过的轴索引文件和新的输出路径：

```bash
.venv/bin/python -m quant_evaluator.scripts.benchmark_real_cos_f61_profile_abba \
  --run --metric-set all24 \
  --axis-index /path/to/verified-axis-index.json \
  --output /path/to/new-f61-all24-report.json
```

两条 `/path/to/` 路径是占位说明，必须替换，不能猜测或新造来源身份。该入口属于重型全市场任务，启动前应与其他窗口协调资源和源码冻结；既有输出及其 `.progress.json` 不允许覆盖。每个执行臂重新做准入检查，最终完整报告必须先通过严格读取器再落盘。该入口只完成当前进程内的 ABBA、资格 auto 和默认缓存 auto；新进程 provider 以及随机交叉次序的实测仍需单独验收。

文件 provider 的 candidate_found 只表示读取到未验证候选，不表示候选已与当前 source 匹配。损坏报告返回 report_invalid，不是 candidate_not_found；即使显式选择 legacy_measured，也不能把损坏候选当纯缺失而回退授权。

旧 F48 合法报告的读取协议保留；能读取历史证据，不代表它适用于修改后的源码。没有真实 F61 验收报告时，本指南不给出可用于生产授权的示例摘要或模拟耗时。

## 新进程普通 auto 的 provider 验证

`verify_real_cos_f61_provider` 是独立验证入口，默认只做 preflight，不读取报告、因子或标签，不初始化 CUDA，也不写结果：

```bash
.venv/bin/python -m quant_evaluator.scripts.verify_real_cos_f61_provider
```

在源码、环境和资源已协调的全新进程中，可用真实完整 F61 报告运行：

```bash
.venv/bin/python -m quant_evaluator.scripts.verify_real_cos_f61_provider \
  --run --axis-index /path/to/verified-axis-index.json \
  --profile-report /path/to/verified-f61-report.json \
  --output /path/to/new-f61-provider-receipt.json
```

路径都是占位说明。报告必须为完整 all15 或 all24 固定域，与当前 manifest 和实际轴一致；该入口不生成或修补 ABBA 资格。进程已有 provider 或已验证的来源缓存时会拒绝执行，不会自动清空他人的状态。CPU 胜者也需准备报告所绑定的 CUDA 环境身份。入口先独立计算 oracle，再做两次不显式传 `source_qualification` 的普通 `auto`：首次要求文件候选经过当前资格校验，第二次要求进程缓存命中；最终核对胜者、执行配置、完整逐指标输出和 oracle。

结果属于独立回执族 `real_cos_f61_fresh_provider.v1`，物理执行 `run_index` 为 0/1；复用 auto 检查协议的 `auto_protocol_run_index` 为 4/5。二者同时记录，不代表额外执行了第 5/6 次 API 调用，也不改变原 ABBA 报告。输出使用独占新建，不覆盖既有文件；结束时只移除自己仍持有的 provider，不清除其他调用者后来安装的 provider。

这里的报告和结果均为未签名、调用者信任的证据，不认证来源或生产者。模拟边界测试只证明编排契约；只有实际 COS 全批量、当前环境下的完整运行才能提供相应真实验收证据。未取得该证据前，不把命令示例或合成测试结果用于生产路由授权。
