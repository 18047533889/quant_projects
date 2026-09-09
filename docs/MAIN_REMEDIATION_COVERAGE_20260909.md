# 历史整改进入 main 的逐文件复核

## 结论与范围

审计主提交：`59cfe456437b027ce498cec89407add582efa35d`，正式目录 `/home/sunhaiwei/quant_projects`。
对原整改目录 `/home/sunhaiwei/quant_projects_v3_remediation` 相对其基线 `2db45f4e446309003381a51a5e03071d31859502` 的相关改动逐文件比较，包含未跟踪源码及被忽略的 Python/TOML 源码候选，排除生成构建树、缓存及原始数据。

- 470 个相关改动文件：453 个与 main 已提交内容逐字一致。
- 17 个内容不同，已逐项阅读完整差异并确认下表所列合并结果。
- 0 个文件缺失于 main 提交；0 个仅留在工作区；0 个 main 仍停留在旧整改基线。
- JSON 中保留原始 `DIFFERENT_REQUIRES_SEMANTIC_REVIEW` 分类，不用程序假装完成语义审查；本报告是这 17 项的人工审查结论，仅适用于 JSON 中记录的精确文件哈希。

这证明本任务旧整改内容的集成覆盖，不声称审查了所有远程历史分支、其他任务或任意临时草稿。

## 17 项差异说明

| 文件 | 已核对的保留结果 |
|---|---|
| `.gitignore` | 旧整改规则完整保留，只增加 main 已有的三个历史 FE 临时目录忽略规则。 |
| `data_access/_build_info.py` | 保留较新 main 构建元数据和 main 分支名，不倒退为旧整改分支的构建信息。 |
| `factor_assets/tests/profiling/test_v5_development_calibration.py` | Python AST 完全相同，仅格式差异。 |
| `factor_assets/tests/profiling/test_v5_multidimensional_grading.py` | Python AST 完全相同，仅格式差异。 |
| `factor_engine/backend/cleaned_bridge.py` | 保留 main 新增的调用方执行身份、FitScope 与拟合失败凭据传递，同时保留旧整改调用逻辑。 |
| `factor_engine/cleaned_operators/filter_smooth.py` | 保留 main 的数值稳定线性拟合、物理查询端点、Butterworth fs=1.0、全历史重放及未实现 checkpoint 的真实声明。 |
| `factor_engine/tests/operators/test_filter_layer.py` | 与实际 checkpoint/lag 合同一致；用独立 scipy SOS 参考及前缀不变性校验替代错误的单调阶跃假设。 |
| `factor_optimizer/factor_optimizer/search/diagnosis_routing.py` | Python AST 完全相同，仅格式差异。 |
| `factor_optimizer/tests/search/test_execution_plan_checkpoint_v3.py` | Python AST 完全相同，仅格式差异。 |
| `jobs/e2e_v5_acceptance.py` | 验收来源路径从硬编码旧整改目录改为当前源码所在的正式目录。 |
| `quant_evaluator/kernels/gpu/drawdown.py` | 保留主干 CAGR 默认口径、有限值波动率过滤与资本归零吸收边界；与整改回撤合同合并。 |
| `quant_evaluator/metrics/portfolio_stats.py` | 保留主干 CAGR 权威计算、等总敞口权重与成本、信号时成员选择和输入检查；整合整改缺失收益及资本边界。 |
| `quant_evaluator/metrics/probe_portfolio/sharpe.py` | 复利年化委托统一权威函数，避免重复计算语义漂移。 |
| `quant_evaluator/metrics/probe_portfolio_legacy.py` | 旧接口也使用等总敞口记账，支持二维及三维因子。 |
| `quant_evaluator/tests/test_gpu_probe_parity.py` | 保留独立复利净值参考、有限值过滤、负资本拒绝及归零边界测试。 |
| `quant_evaluator/tests/test_v3_risk_variants.py` | 明确分别验证算术年化和 CAGR，额外断言默认使用 CAGR，没有放宽断言。 |
| `quant_evaluator/tests/test_v6_exposure_contracts.py` | Python AST 完全相同，仅格式差异。 |

上述五项 AST 相同通过 `ast.dump(ast.parse(source)) == ast.dump(ast.parse(committed_main))` 独立核验。

## 验证与复现

此前正式主目录完整回归为 5191 通过、0 失败、24 跳过或预期失败，源码前后不变。参见 `docs/MAIN_REMEDIATION_INTEGRATION_20260909.md` 和 `evidence/main_remediation_20260909/{execution,source_hashes}.json`。

额外核对 470 项与回归源码清单的交集：448 项提交哈希全部匹配；其余 22 项为清单范围外的根目录脚本、文档、账本、配置等，不冒称由该源码清单覆盖。17 项差异中除 `.gitignore` 外的 16 项均匹配回归源码清单。

覆盖清单：`evidence/main_remediation_20260909/coverage_audit.json`，每项记录旧源码、原基线、main 提交及工作区 SHA256。

从正式主目录可运行只读审计（仅写指定 JSON 输出）：

```sh
.venv/bin/python scripts/audit_remediation_coverage.py \
  --source /home/sunhaiwei/quant_projects_v3_remediation \
  --output evidence/main_remediation_20260909/coverage_audit.json
```

## 其他任务与保留边界

另一 FE 任务另行核查来自 Mac `/tmp/v8-pending` 与 `.stage-m36-resource` 的 pending-resume、benchmark 草稿，不属于旧 v3 整改分支；其补充合入及测试应以该任务的新提交证据为准。不能用本次 470 文件审计替代它的审查。

旧本地 `codex/v3-remediation-20260906` 分支已删除；原整改目录保留为 detached HEAD 历史证据，不删除未独立核对的独有文件或其他任务分支。今后直接在正式 main 目录按文件分工，不创建新分支、worktree 或整库副本。

代码已集成不等于 V8 所有数据及能力验收已经完成；T70、T116、T117、T119、T120、T123、T165 的原有限定仍然有效。本轮未部署生产或切换交易。
