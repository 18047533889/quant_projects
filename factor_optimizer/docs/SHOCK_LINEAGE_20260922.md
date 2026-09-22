# 下跌冲击复合因子的处理谱系修复

后续更新：[缩尾退化后保留有效排名](WINSOR_BASELINE_FALLBACK.md) 已定位稀疏日
被缩尾抹平并补充 TRAIN 回退；下文为谱系修复时的历史对照。

## 问题与证据

此前真实 COS 因子 downside_shock_accumulation_asym 能被读取、逐方法回放，
但 baseline_diagnostics 显示 lineage_unknown，基础处理为空。
原因是声明谱系解析器未识别其 ts_pct、ts_delay、ts_std、lt、where、
maximum、flex_max 特征构造，而不是证明这个因子已经做过缩尾或排名。

新增真实表达式及外层 rank 的回归在修复前为 2 failed、12 passed；
修复后与既有成交量谱系、清单绑定测试共 39 passed。
完整优化库与预处理库回归：1859 passed、1 xfailed、16 warnings，99.89 秒；
HP 非因果滤波的预期失败与生产禁用限制不变。
该识别只解释清单绑定的源声明，不执行 DSL，不证明历史落值实现、
PIT 信息可用性或生产准入，不新增读取目标或突破 DataAccess 限制。

## 保守参数合同

核对 FE common/time_series.py、common/elementwise.py、
common/polars_ops.py 与 common/polars_ts_basic.py 的接口后：

| 形式 | 本次接受范围 |
|---|---|
| ts_pct(x, d) | 严格整数 1..10000 |
| ts_delay(x, n) | 严格整数 0..10000，禁止未来负滞后 |
| ts_std(x, n) | 两参数形式，严格整数 2..10000；未冒认其他 ddof 形式 |
| lt(x, y)、maximum(x, y) | 恰好两参数，所有分支均须递归可识别 |
| where(c, x, y) | 恰好三参数，条件和两个分支均须可识别 |
| flex_max(x, c) | 有限非布尔、非正字面标量；正整数滚动窗口暂不接受 |

flex_max 的第二参数不能随意当成截断阈值：
FE 对正整数使用滚动最大值，非正标量才采用逐点比较。
未知分支、内部 rank、布尔窗口和未支持参数继续返回未知，
不是通过放行任意函数来把整个谱系变成“已知”。

本因子没有声明最终截面排名，因此基础方案变为 winsor -> cs_rank。
如果表达式外层已有 rank，保留该步骤并省略新增 cs_rank。
内部特征构造并不冒充最终输出的 winsor/neutralization/smoothing 处理步骤。

## 真实对照设计

同一 DataAccess 清单绑定对象，500 日 × 256 股票，267 TRAIN 日。
控制组明确不给谱系，复现原来的未知谱系路径；处理组使用识别出的绑定谱系。
两组均使用默认联合指标门槛，TRAIN 选择，VALIDATION 仅确认冻结赢家，
TEST 不评分。不按验证表现改动参数或准入门槛。

此对照核验基础处理接线与经济准入，不是速度 A/B，也不承诺强制产生新因子。
真实结果见 [JSON](shock_lineage_ab_20260922.json)。控制组基础方案为空；识别组确实
执行 winsor、cs_rank，但 267 个 TRAIN 日仅 266 日具有配对有效 IC，基础评估
返回 joint_training_metrics_unavailable（不可填补缺失联合指标），故仍退回 RAW。
RAW 训练组合 Sharpe 约 2.0217、RankIC 约 0.04011、最大回撤约 0.00535，
这些是训练描述，不是验证或未来收益。

两组均有 104 候选、10 个通过训练评分，均未得到可靠改善，TEST 未评分。
识别修复并不等于本因子基础处理已被接受。下一步应定位联合方案的具体失效步骤，
检查是否能保留不退化的基础步骤；不能降低覆盖门槛或抹掉该日来获得好看指标。
