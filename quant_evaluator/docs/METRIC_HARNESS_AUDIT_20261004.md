# 指标回归测试框架审计与输入修复（2026-10-04）

## 结论与范围

全指标框架过去会捕获所有异常、拼接文字，再允许其中任意一个“缺少输入”
关键词使整个失败被接受；批量/单项比较也会静默丢弃抛异常的候选指标。
因此历史全绿不能作为全部指标独立数值正确的证明。本轮修复的是测试框架，
不借测试输入错误改变生产指标公式，也不放宽输入合同。

三个真实行为反例分别是：先出现已知输入错误、再出现内部 RuntimeError；
ic_decay 发生与 HorizonMeanIC 无关的合同错误；批量候选内部错误被丢弃。
同一命令初始 3 failed（DID NOT RAISE），修复后 3 passed。

## 测试框架行为

- 仅捕获 InvalidContractError 并保留实际异常对象；其他异常立即传播。
- 所有重试都失败时，每一次失败都必须属于声明的输入要求，不能用另一条
  报错文字掩盖未知失败。
- ic_decay 只接受缺少 HorizonMeanIC 构建器的确切合同失败；其他失败不放行。
- 单项/批量候选与测速重试不再吞掉内部计算错误。
- 去除因子轴错误、收益时间轴错误、错误参数值的历史豁免。

## 修复输入而不是掩盖错误

泛化证据原来是两因子 f0/f1，而请求只有 prof。现证据为 prof/v1 单因子，
并在 FactorBatch 中绑定对应版本。日历指标改用已有完整年度、会话对齐的
专用 factor/label/probe/snapshot 输入，删除给整数轴收益直接换 datetime
标签的错误构造。turnover_cost 使用匹配时间轴、明确 cost_drag 腿的 typed
ExecutablePortfolioArtifact，而不是把普通 long-short probe 当执行成本。

三个日历手算反例同时检查低层函数与公开 evaluate 返回值和有效期数：
完整二月三期 -5% 的结果为 0.95^3-1；完整季度/年度三期 -10% 为 0.9^3-1。
每个案例必须 valid=True、observation_count=1，不能只检查输出存在。

## 最新独立复跑

```sh
.venv/bin/python -m pytest -q -p no:cacheprovider \
  quant_evaluator/tests/test_all_metric_harness_fail_closed_oct04.py \
  quant_evaluator/tests/metrics/test_all_metrics_ab_contract.py \
  quant_evaluator/tests/metrics/test_special_input_metrics.py
```

server-c 正式树结果：298 passed、12 skipped，5.88 秒，退出码 0。
12 个跳过是通用框架的暴露相关指标与 ic_decay 专门输入缺口；不能解释为
算法未实现或算法已正确。专门暴露数值测试存在于其他测试文件，未由本次
三文件结果覆盖。确定性比较仍主要检查值，批量/单项会共享生产实现；
这些也不是独立数值 oracle。真实性能资格和默认 auto 接入需要另行实测。
