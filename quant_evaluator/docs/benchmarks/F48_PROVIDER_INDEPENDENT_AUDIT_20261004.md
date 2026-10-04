# F48 新进程默认 auto：独立审查记录（2026-10-04）

本记录是报告之间的独立结构性交叉核验，不是重新计算真实数据的 oracle，也不是当前机器环境的重新资格认证。源码修改后仍需重新采集资格证据。

## 输入证据

- 性能报告：`real_cos_f48_linear_shape_profile_current_root_20261004_e20e97d6c317.json`。
- 新进程报告：`real_cos_f48_linear_shape_provider_current_root_20261004_0fac516e100b.json`。
- 两文件均位于本目录；使用严格 `load_source_profile_report` 解析前者。
- 请求规模为 2586 个交易日 × 5461 只股票 × 48 个因子。
- 仅涉及六个指标：quantile_curvature、quantile_tail_asymmetry、quantile_adjacent_spread、quantile_extreme_cliff、top_quantile_cliff、bottom_quantile_cliff。

## 本次直接核验

新进程报告为 complete、run_started=true，包含且仅包含两轮。manifest 摘要、指标顺序及请求规模与性能报告一致。

两轮都采用性能报告的获胜后端 CUDA，qualification_applied=true、qualification_status=qualified_current_source、实际 tile=5、OOM 重试为整数 0，并记录独立 oracle 一致。

首次默认 auto 的来源为 report_candidate，provider 状态为 candidate_validated，cache 状态为 supplied。此处 supplied 表示 provider 经验证后向路由提供资格，不代表调用者绕过 provider 显式传入资格。

第二次默认 auto 的来源为 process_cache，provider 状态为 not_checked，cache 状态为 cache_hit。

两轮 context 和 correctness digest 均与严格解析的性能报告记录一致。六个指标各自的 values_sha256、finite_mask_sha256、observation_counts_sha256 均与获胜性能记录一致，两轮之间也一致。

独立只读 stdin 核验命令终态为 exit 0。首次审查脚本曾直接比较 typed context 的 tuple 与 JSON 的 list 导致断言失败；将 typed context 按 JSON 规范化后所有具名断言通过。这是审查脚本表示差异，不是计算库缺陷。

## 轻量回归

`PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q -p no:cacheprovider quant_evaluator/tests/test_source_auto_authority_oct04.py`

结果：29 passed in 0.06s，exit 0。不启动真实 COS 或 CUDA，不改变源码与资格缓存。

## 不能据此声称

不能声称全部 24 个 source 指标、61 因子规模、多 horizon 实际调用链、所有后端或所有机器均已达到最快。仍需当前源码闭包下的随机配对测速，以及各新请求域独立的完整资格证据。

报告是调用者信任的 JSON，不是签名证明。目前 fresh receipt 不包含输入 profile 文件摘要、provider candidate ID 或 axis-index 文件摘要。将输出身份与选中报告交叉核验提高可审查性，但不构成密码学认证或本次真实数据重算。
