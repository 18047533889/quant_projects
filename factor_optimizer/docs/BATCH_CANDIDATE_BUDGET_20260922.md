# 自动候选预算与逐方法复验（2026-09-22）

## 已复现的问题

TRAIN 诊断会在预设网格外增加 U 型中心、分层衰减和平滑候选。
原实现扩展后直接在逐因子保护范围之外抛出 ValueError，使整批调用中断；
即使其他因子已经完成，也不能返回整批结果。

新增两因子数据回归：第一只诊断产生 9 个候选，预算为 7，第二只是可修复的反向因子。
交换顺序分别测试，修复前均因预算异常失败（2 failed），不是人为让测试无条件成功。
修复后连同既有定向、形状、时序及验证边界测试共 43 passed、2 warnings。

## 自动化行为

- 预先声明的静态候选预算仍严格验证；衰减正负方向组合也计入静态预算。
- TRAIN 扩展后的候选数量超限：仅该因子返回
  `budget_exceeded_raw_retained`，保持 RAW、不给出验证赢家，其他因子继续。
- 不截断参数网格，不提高预算，不使用 VALIDATION/TEST 挑选应该保留哪些候选。
- `training_diagnostics.candidate_budget` 给出 required、maximum、status 和 evaluated。
  evaluated 是通过训练准入评分的候选数，不包括被拒绝项，也不包括此前的基础处理诊断。
  超预算时 evaluated=0、candidates 为空；不表示整个 TRAIN 诊断零计算。
- COS 示例的 JSON 同样输出 candidate_budget。
- 默认预算仍为 128。若整批因子均超限，逐个明确保留 RAW，不伪装成优化成功。
- 此次隔离仅覆盖候选预算溢出，不声称所有前置诊断异常都已实现逐因子隔离。

## 可复验入口

正式 server-c 主目录运行：

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q \
  factor_optimizer/tests/test_batch_dynamic_budget.py \
  factor_optimizer/tests/test_method_audit_regressions.py \
  factor_optimizer/tests/test_research_batch.py \
  factor_optimizer/tests/test_research_batch_boundaries.py
```

真实审查使用 examples/cos_batch_audit.py 的 load_cos_sample，经 DataAccess 读取
绑定清单的 2 因子、500 日、256 资产；examples/method_audit.py 的 audit_methods
逐项检查前缀不变性、资产重排不变性、行对齐及成本后的训练指标。
默认优化与显式预算 90 分别执行；验证超预算输出逐元素等于输入。
所有选择仅使用 TRAIN，VALIDATION 只确认冻结赢家，TEST 不评分。

## 验收边界

完整优化库与预处理库回归：1843 passed、1 xfailed、16 warnings，99.88 秒。
真实回放共 130 案例：114 executed、16 requires_additional_inputs_or_control、0 failed。
默认预算 128 时，两只因子均生成 103 个候选，分别有 18 和 73 个通过训练评分门槛；
第一只 weekly_4cbd7ca6dccc61dc 的倒 U 型修复通过验证，联合改善下界约 0.11791；
第二只 weekly_db5616cd85a477b3 未通过验证，保留 RAW，不重选。
预算 90 时两只均明确返回 budget_exceeded_raw_retained，输出与输入逐元素相同。
TEST 未评分。逐方法参数及复验证据见 [JSON](batch_budget_real_20260922.json)。

仍然存在的边界：
三类中性化仍需可信历史暴露，三类 DSL 优化需要真实重编译；
ABANDON 和 drop 是控制/行选择，不应统计成数值方法已执行。
HP 非因果过滤维持 OFFLINE_ONLY 及预期失败测试。
不能据此宣称全平台无 bug、所有方法真实验收完成，或未来收益必然提高。
