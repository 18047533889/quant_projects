# DuckDB 窗口语义修复审计（2026-09-14）

## 范围和边界

在 server-c 正式工作树修复通用库缺陷，没有提交 Git、发布因子、重启生产任务或重算历史 COS/报告结果。
本审计不是全库生产认证。注册表枚举、真实 SQL 数值对比、未支持 SQL、适配器未覆盖及超时分别记录；未执行项不能算通过。

审计工具：`factor_engine/tests/backend/audit_sql_window_support.py`。
逐条记录：`evidence/factor_engine/window_support_audit_20260914.json`。
专项测试：`factor_engine/tests/backend/test_sql_window_support_regressions.py`、`test_overnight_sql_min_periods.py`。

## 统一原则

- 最小样本数来自当前规范实现，不能一律设为完整窗口，也不能一律设为 1。
- 区分窗口行数、有限值数、条件命中数、有效配对数、尾部样本数和连续有效路径长度。
- 先确定当前窗口，再在该窗口内计算秩、极值、尾部阈值和回归；不能对不同历史窗口的中间统计再滚动聚合来替代。
- NaN/NULL/无穷值按算子定义处理；统计无定义时不能用 0 或任意 epsilon 冒充。
- 方向、数据源、报告参数和因子公式未在本次修复中改动。

## 已修复的主要计算路径

| 类别 | 明确口径与修复 |
| --- | --- |
| 隔夜/日内 | spread 双腿分别满足完整窗口；sign agreement 满足窗口行数且保留缺失符号计为 false 的定义；opening mispricing 的协方差与方差使用同一有效配对集合。cov 的 5 对样本要求不变。 |
| Top/Bottom-K | 精确排序取 K 个，不使用分位数近似；至少 max(K, min_periods) 个有限值；std 使用 ddof=1。合并原来六个重复 SQL 分支。 |
| 截尾均值 | 当前有限窗口按 floor(trim_ratio * n) 删除两端；默认支持数 max(5, window//2)，尊重显式 min_periods。 |
| 上下偏矩 | 有限值计数至少 max(2,min_periods)；缺失不能落入 ELSE 0；默认 order=1。 |
| 乘积 | 当前注册实现默认完整窗口；尊重 min_periods、skipna，并保留符号、零和溢出处理。 |
| 自相关/条件相关 | 当前窗口有效配对计数，尊重 lag/min_periods；条件相关只使用条件为真且两列有限的配对。 |
| 秩相关 | 每个消费窗口内重新计算双序列的配对平均秩；窗口行数满足 d 且至少 3 对有限样本。不能对逐日 rolling rank 再 rolling corr。 |
| HMA | 三层 WMA 均要求各自完整有限窗口，尊重 floor/round；不改变其他允许部分窗口的线性衰减算子。 |
| 价格均值偏离 | 开头 d-1 行缺失；完整行数窗口中的均值仅使用有限历史样本，保留原参考对当前无穷值的行为。 |
| 极值位置/年龄 | 同一当前窗口内定位；并列取最近一次；按有限值选极值；全空返回缺失；最老端位置与年龄分开。 |
| 有效数/覆盖率 | 门槛按窗口行数，统计有限值数；全空窗口可返回计数 0，不错误返回缺失。 |
| 滚动中位数 | 尊重默认 min_periods=3，而不是硬编码完整窗口。 |
| 分位统计/ES | 分位区间、偏度、峰度、尾部比和 ES 使用同一有限窗口；尊重各自样本/尾部样本门槛、分位参数和退化分母条件。 |
| 绝对值熵 | 用同一窗口的 sum(abs(x)) 和 sum(abs(x)*log(abs(x))) 计算；取消非法嵌套窗口聚合。 |
| 穿越速度/加速度 | 当前缺失返回缺失；有效非穿越为 0；穿越时尺度无定义不加 epsilon 强行出值。 |
| 最大回撤 | 缺失切断价格路径，仅用当前连续有效后缀，不能跨缺口继承旧高点。 |
| 回归统计 | t-stat/partial-corr 默认最小支持数对齐当前实现；slope 尊重 lag/min_periods，系数允许在当前缺失但历史配对足够时输出。尚未实现的回归模式明确不生成 SQL。 |
| Wilder ATR | high/low/前收缺失时 TR 缺失，不让 SQL greatest 跳过缺失腿。 |
| 分组排名权重 | 平均秩处理并列；缺失分组及整日缺失遵守 fallback_policy；window 在该兼容截面算子中本就不参与计算。 |
| digital_count | 统计连续满足阈值的一步变动段长度（上限 d，低于 run 返回 0）；旧 SQL 错写成 COUNT(DISTINCT)，已替换。 |
| winsorize | Polars-long 和 SQL 都在分位点计算前排除非有限值，与当前 Pandas 定义一致，而不是仅清理裁剪后的结果。 |

另修复参考内核边界：`safe_ops._pd_argidx` 仅在有限子集找极值；回归的常数响应不从 lstsq 浮点残差制造 t 值。常数带截距回归斜率为 0，t 值未定义。

## 已禁止错误 SQL、但未完成原生重写的 6 个算子

`ts_distance_corr`、`ts_autocorr_decay_half_life`、`ts_expectile`、`ts_quantile_beta_spread`、`ts_expected_shortfall_asymmetry`、`ts_tail_imbalance`。

原因在 emitter 的 `_SQL_WINDOW_REPAIR_FALLBACKS` 中逐项登记，六段错误模板已移除。编译与 SQL 能力预检都拒绝这些旧实现（包括嵌套在复合公式中）。现有路由可以尝试其他已支持后端；如果数据/策略不允许回退，应明确失败，不能返回近似数值冒充原算子。本次未将“禁止错误 SQL”记为“DuckDB 原生实现已修好”，也未认证所有现场输入的回退路径。

## 验证

专项回归最终结果：272 passed（54.79 秒）；覆盖正常、缺失、无穷值、常数、全空、乱序多标的、未来扰动不改变前缀、多个窗口/门槛及多输入配对。

当前注册表共 1756 条：114 项实际 SQL/参考数值对比通过；1223 项没有本适配器识别的显式窗口签名；392 项没有生成 SQL；18 项参考内核未被适配器覆盖；6 项因已确认算法问题禁用 SQL；2 项没有 Pandas 内核；ADX 1 项需要生产 CSE 测试框架。ADX 的直接展开 SQL 探针超过限定时间，审计进程已明确终止并重新运行其余项目，未把超时算作通过。这些数字是适配器覆盖清单，绝不是 1756 项全都完成数值验证。

修复前现有后端全套回归：2028 passed、55 failed、317 skipped（363.58 秒）。中途全套复跑：2027 passed、57 failed、316 skipped（373.35 秒）。与前次相比额外出现两个 winsorize 失败，已据此修复 Polars-long/SQL 的非有限输入口径；相关三后端测试定向复跑 3 passed、139 deselected（49.84 秒）。其余 55 项失败包含旧名称、缺失测试证据及其他实现问题，尚未全部解决；不能将定向复测拼成一次“全套全绿”的结果。最后的小范围修复后没有再完整重跑该大套测试。

复现入口（在 `/home/sunhaiwei/quant_projects` 下，使用项目 `.venv/bin/python`）：

```text
python -m pytest factor_engine/tests/backend/test_sql_window_support_regressions.py factor_engine/tests/backend/test_overnight_sql_min_periods.py -q
python -m factor_engine.tests.backend.audit_sql_window_support evidence/factor_engine/window_support_audit_20260914.json
python -m pytest factor_engine/tests/backend_parity factor_engine/tests/backend_sql -q --tb=short
```

SQL 模板缓存 generation 已更新。已启动进程不会因源文件修改自动重新加载；历史因子矩阵、COS 值、评估结果和报告曲线也不会自动修复。受影响历史结果需要单独识别并重新落值/评估，不能改页面说明后继续把旧数值当作修复结果。
