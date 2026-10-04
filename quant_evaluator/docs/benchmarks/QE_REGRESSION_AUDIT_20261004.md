# QE 完整回归审查记录（2026-10-04）

最新独立完整回归：6643 passed, 26 skipped, 81 warnings，223.49 秒，退出码 0。命令为 `.venv/bin/python -m pytest -q quant_evaluator/tests --tb=short`，OPENBLAS_NUM_THREADS=1、OMP_NUM_THREADS=1。测试期间 QE 源码与测试冻结；仅出版已有工作树内容，未新增源修改。下文 27 个失败及“尚未闭合”列表是此前排查过程的历史记录，当前剩余事项以文末为准。

这不是所有输入永无 bug 的证明，也不是 COS/F61 的性能资格证明。

## 验收状态

目标仍未完成；不能宣称全部指标无 bug、默认 GPU 最快或取得当前真实 F61 性能资格。正式代码树为 server-c /home/sunhaiwei/quant_projects，直接修改，未创建分支或副本。子代理仅 GPT-6-luna、medium；禁止使用任何重置卡。

## 完整回归与失败复测

正式树命令：
```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q quant_evaluator/tests
```

终态：27 failed, 6595 passed, 25 skipped, 81 warnings，228.30 秒。随后同范围 --lf --tb=short 复测：26 failed, 1 passed，3.22 秒。后者不能替代完整回归；一次复测通过也不证明不稳定项已修复。

下列为完整回归全部失败项（路径均相对 quant_evaluator/tests），不省略非本轮引入的失败。数学端点契约与文档的五项已经通过后续定向复测；其他项尚待定位，不能一概视为过时测试。

- `test_calibration_source_modes.py::test_modes_share_identity_without_false_source_drift`
- `test_f48_cap16_benchmark.py::test_default_tile_flag_is_forwarded_and_reported`
- `test_gpu_factor_tile_policy_cap.py::test_source_auto_cap_below_certified_width_falls_back_without_shrinking_cpu`
- `test_gpu_factor_tile_policy_cap.py::test_source_gpu_routes_apply_policy_cap_but_cpu_route_does_not`
- `test_gpu_factor_tile_policy_cap.py::test_source_auto_keeps_certified_width_when_policy_cap_is_larger`
- `test_gpu_temporal_profile_reuse_audit_oct04.py::test_quantile_key_isolates_n_quantiles_and_min_assets`
- `test_metric_reference_document.py::test_generated_reference_is_current`
- `test_metric_reference_document.py::test_published_math_has_no_known_markdown_or_macro_hazards`
- `test_public_auto_calibration.py::test_real_public_cuda_calibration_and_cache_receipts`
- `test_rank_ic_coefficient_parity.py::test_rank_ic_is_exact_scipy_coefficient_for_ties_and_distinct_values[float32-2]`
- `test_rank_ic_coefficient_parity.py::test_rank_ic_is_exact_scipy_coefficient_for_ties_and_distinct_values[float64-2]`
- `test_rank_ic_coefficient_parity.py::test_rank_ic_is_exact_scipy_coefficient_for_ties_and_distinct_values[int64-2]`
- `test_source_benchmark_observer_oct04.py::test_observer_order_timer_and_close[None]`
- `test_source_benchmark_observer_oct04.py::test_auto_verification_forwards_exact_supplied_qualification`
- `test_source_execution_receipt_oct03.py::test_cuda_actual_width_may_be_below_request_cap`
- `test_source_linear_shape_batch_oct04.py::test_unqualified_auto_keeps_new_shape_request_on_cpu`
- `test_v3_public_artifacts.py::test_daily_quantile_public_cpu_gpu_keeps_all_axes_counts_and_masks`
- `test_v3_public_artifacts.py::test_quantile_consumers_share_raw_builder_across_profile_policies`
- `test_v5_final_semantics.py::test_daily_quantile_axes_survive_cpu_cuda_and_json_roundtrip[5]`
- `test_v5_final_semantics.py::test_daily_quantile_axes_survive_cpu_cuda_and_json_roundtrip[10]`
- `test_v5_final_semantics.py::test_daily_quantile_axes_survive_cpu_cuda_and_json_roundtrip[20]`
- `test_v5_gpu_capability_matrix.py::test_all_19_factor_label_metrics_execute_real_cpu_and_strict_cuda`
- `test_v5_gpu_capability_matrix.py::test_supported_matrix_is_exactly_33_without_metadata_only_entries`
- `test_v5_metric_instances.py::test_public_q10_q20_h10_h20_coexist_and_roundtrip[cpu]`
- `test_v5_metric_instances.py::test_t01_t02_all_h_q_cost_variants_and_third_factor_remain_independent`
- `test_v8_numeric_goldens.py::test_quantile_returns_bounded_workspace_matches_full[20]`
- `test_v8_numeric_goldens.py::test_quantile_returns_bounded_workspace_matches_full[40]`

## 本轮确认的修复与证据

- 完全仿射 Pearson/Spearman 输入此前会制造浮点离散度与约 10^16 的虚假 ICIR。先复现失败，再通过精确二进制有理数证明端点；不采用小标准差阈值归零。普通样本保留原结果，不放宽普通 SciPy/NumPy 的逐位相等断言。
- 公共批量 CPU/CUDA、独立 oracle、显式 exact/numba/polars/gpu 端点及通常输入对照共 105 passed，24 条空样本警告，6.53 秒。这不是全部指标、所有输入的正确性证明。
- 文档公式持久源 docs/formulas_a.json 已更新，用既有生成器重建 METRIC_REFERENCE.md，覆盖 164 个注册指标、5989 行。文档七项检查 7 passed，0.35 秒。
- test_rank_ic_coefficient_parity.py 仅对独立证明同秩/互补秩的样本采用精确 ±1，普通样本仍严格等于 SciPy。16 passed，0.17 秒。
- GPU 稀有候选精确证明按最多 64 行分块，仅处理共同有效样本；常规输入不复制全因子立方体到主机。仍新增候选发现同步点，不能声称零开销。
- provider 原子安装/释放与 F61 编排检查 45 passed，1 条 fork 弃用警告，1.42 秒。尚未出版且尚未实际 F61 provider 验收；依赖 producer 先出版。
- FA shape 迭代器有界读取修复已双推并核验代码树；相关 84 passed。optimizer 有界 bootstrap 微基准工具已双推，10 passed；其加速不代表整个 optimizer 流程加速。
- 多期限、多分桶指标实例的内部参数泄漏已用共享 contracts/metric_parameter_policy.py 修复；默认推导不再序列化内部制品，早期与执行前参数检查统一拒绝外部注入。新注入测试实际 3 failed/1 passed 后，同测试及既有 v5 final semantics/metric instances 共 15 passed，1.67 秒。更大范围回归仍需继续，不据此宣称所有多期限路径已验收。

## GPU 普通输入开销 A/B

固定合成随机且有 10% 因子 NaN 的输入，T=128、N=5461、F=4；同一个进程 GPU Pearson 内核，预热后 3 轮 baseline/certified/certified/baseline。baseline 仅在进程内禁用新增证明，不修改磁盘代码。两臂输出与计数逐位相等，候选端点行数为 0。

| dtype | baseline 中位秒 | certified 中位秒 | 额外中位秒 |
|---|---:|---:|---:|
| float32 | 0.0003084755153 | 0.0003992929996 | 0.0000908174843 |
| float64 | 0.0006112770061 | 0.0006960089959 | 0.0000847319898 |

仅为预上传数据的 GPU 内核含 guard 时间，不含主机上传、输出下载或全 API；不是 COS 真实数据、整体 all24、F61 或资格测速。一次小范围测量不能推断全局最快。需进一步减少同步成本，并在源码稳定后重做端到端实测。

## 尚未闭合

1. 多期限/指标实例 CPU 的已复现内部参数泄漏有定向修复与 15 项通过证据；仍需最终稳定源码完整回归及真实组合场景验收。
2. GPU quantile 小工作区 q20/q40 失败，需判明固定修复工作区与分块预算契约，不可单纯放大测试预算。
3. source auto 同一来源重复评估累积读取日志误撤销资格，另有历史回放/擦除可能误授资格；需按本次追加范围与未变历史前缀核验，不能清空历史掩盖问题。
4. 其余失败需逐项区分当前代码缺陷、陈旧 fixture、硬件/运行顺序敏感性，不降低 fail-closed 门槛。
5. 完整稳定源码回归、冻结 QE/DataAccess/FO/FP 的身份后，再做真实 61 因子、2586 日期、5461 股票的独立 oracle、ABBA、资格 auto、缓存 auto、新进程 provider 和随机次序稳定性检查。资源预检已通过，但未启动真实资格任务。

## 最新闭合情况与剩余验收

- 原 27 项失败在此次完整测试中没有再出现。内部参数共享策略、GPU tie_method provenance、校准初末扫描开销汇总、精确仿射端点均由完整套件覆盖。
- q20/q40 的 64 KiB GPU 分层工作区测试已经原预算通过；修复按预算限制实际启动线程数，未扩大预算。GPU 预算修复个人提交 4daf5e6d5fe0a07fdbad13c688025dc37b7a3d0b 已包含在后续主干。
- source 重用三文件已出版：个人 main e802586b0011b03c5dda2158827fc46609320db5；HKUST QE 605669b4e19bdadd511e9f3d5f76f3d8ab60abe6；代码树 aa484d9f3b7e996cab43aa986117b6c53be345f3，root 独立核验一致。只检查新增读取并保留、核验历史前缀。
- 陈旧测试分别改用 request-level quantile builder 设置、实际导入的 builder alias spy，以及显式 27 个 factor/label GPU 指标和 41 个支持指标集合；CPU/CUDA 数值、计数、轴与 provenance 断言未放宽。真实 27 指标 strict-CUDA 公共 parity 测试通过。
- 26 个跳过包括 12 个需要专用输入的通用 A/B 项、3 个多 GPU 检查、2 个不存在的预处理内核检查、4 个 v1 无 default-auto receipt 检查，以及需显式启用的 source CUDA/runtime 检查。不能把这些跳过算作已验收。
- 81 条警告包括空样本、极值溢出后回退覆盖、fork 弃用及 Pandas 接口弃用；没有静默过滤警告。
- 仍需出版 root 已验证修复、F61 producer 与 provider，再冻结 QE/DataAccess/FO/FP 完整身份。真实 F61 all24 独立 oracle、ABBA、显式资格 auto、缓存 auto、新进程 provider 尚未开始，不能宣称真实最快或默认资格已完成。
- producer/warmup 只读独审未找到具体缺陷，但直接 warmup 测试仅覆盖 CPU all24；下一步补实跑 all15/CUDA warmup 检查。optimizer 原生适配仍 research-only，当前测量慢于既有后端，不应默认切换。
- 后续 root 实际调用 warm_source_profile 补验 all15/all24 × cpu/cuda_strict 共四臂，退出码 0；完整 scalar_metrics/series_metrics 集合、两个因子标识与全部输出有限性均通过。前一次临时检查脚本错误使用 BatchEvaluationBundle.artifacts 报 AttributeError，修正为该批量 API 的 scalar_metrics/series_metrics 后通过，非生产缺陷。此处为 65 日期、100 资产、2 因子的合成预热调用，不是 F61/COS 资格或速度证明；持久测试覆盖扩充仍待完成。
- 随后持久 warmup 测试已扩充为四臂，保留 metric domain、有限性、轴、观测数检查，并验证 backend_used、逐指标后端与 strict-CUDA 执行 receipt。root 独立审读并复测 producer/warmup 三文件：21 passed，1.42 秒。此前 6643 项完整回归发生在这次测试扩充之前，没有生产源码变化；新三项新增覆盖不能伪报成此前完整测试计数。

## 后续标量精度审查与最新回归

- 独立 Luna 审查发现 Fraction(float(value)) 会丢失大于 2^53 的整数及扩展浮点精度。root 新增四反例：非仿射 int64、混合整数/浮点的强制类型相等、uint64 模取负、longdouble 非仿射。真实测试 4 failed，0.11 秒；最小修复后相同测试 4 passed，0.05 秒。
- 精确证明现在对整数使用 int/Fraction，对浮点标量使用其自身 as_integer_ratio；快速相等只在相同 dtype 下证明，快速取负仅用于相同浮点 dtype，避免整数溢出。端点过滤阈值不变，仍须精确证明才能修改系数。
- 数学端点、GPU、独立 oracle、RankIC 和四臂 warmup 复测 55 passed，4 条空样本警告，3.41 秒；未修改、删除或放宽原普通样本断言。
- 随后完整 QE suite（session49282）终态为 1 failed, 6649 passed, 26 skipped, 81 warnings，224.36 秒。唯一失败是生成文档未包含新辅助函数及变化后的源码行号，不是忽略失败或声称此轮全部通过。
- 使用既有 build_metric_reference 生成器重建文档，164 指标、5991 行。原失败所在文档七项加新标量四项独立复测 11 passed，0.36 秒。此后只有生成 Markdown 和本审查记录变化；未再次声称全套零失败。F61/COS 性能资格仍未开始。
- FA 确定性候选顺序/重成员提前拒绝已双推并 root 核验：个人 main4919ab22cf890e941beb69637885a5c42650ea39，HKUST FA3d6a8fc8fabe22241b78be446ba2d1f140c25bac，FA tree368852d596f2ffab9228db502ae9fa73796c1b21。

## 真实 F61 原始轴与标签对齐接线

- peer 用既有严格 axis reader 核验真实轴索引：2588 日期、5461 股票、61 因子，manifest/checksum 一致。既有 load_labels 的注册 t/t+1/t+2 价格口径裁掉尾部两日，资格协议所需 2586 是对齐后的日期数。producer/provider 原先在标签对齐前比较 2586，会阻止真实请求。
- producer 新测试实际 4 failed、2 passed（0.60 秒）；共享 source_f61_axis_admission 仅接受日期数 schema.T 到 schema.T+2、资产与因子数严格匹配，标签加载后仍要求完全相同的 schema.shape。原始 source_rows 对象与校验记录保持不变，没有伪造或裁切原始 receipts。
- 首次修复后的两项测试因误用 SourceProfileReport.request_shape 报 AttributeError，测试改为该 typed API 的 records[0].context.request_shape 后，六项 raw-axis 测试 6 passed（0.52 秒），producer/CLI/四臂 warmup 27 passed（1.68 秒）。前后类型接口误用属于测试检查代码，不计为生产修复。尚未读取真实标签或运行真实 F61 性能资格，不能将编排 mock 的绿色等同为真实 qualification。
