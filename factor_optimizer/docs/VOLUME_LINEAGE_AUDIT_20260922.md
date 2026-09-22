# 成交量因子处理谱系与逐方法实测（2026-09-22）

## 修复
声明式表达式识别原先不支持 fillna_const、ts_sum、gt 和 ts_ema，
使 volume_state_persistence 的谱系被标成 incomplete，基础预处理全部跳过。
现支持严格参数形式，记录输出链上的常量填充、EMA、截面排名及执行顺序。
特征内部的 EMA 不误标为对整个输出做过平滑；未知算子或带排名的未知分支仍拒绝推断。
只解析声明，不执行 DSL，也不构成上游时点正确性认证。

常量填充：缺失时 x'=c，否则 x'=x。
EMA 跨期递推为 m_t=αx_t+(1−α)m_(t−1)，α=2/(span+1)；
这里记录来源声明的 span，不重新计算或担保来源实现的初始化/缺失值细节。
允许 span 为 1..10000 的整数，禁止布尔、冲突参数及未支持的 alpha/adjust 形式。
已有输出截面排名仍保留，基础处理不会再次排名。

## 数据及边界
通过 DataAccess 内容绑定读取 COS 真实因子，不复制仓库、不变更下载预算。
因子 SHA256：b53888213fe77c90f14e1ed9eaa817751c2d9c4525ec1db5e30968a2c14f7b19。
清单 SHA256：35e3558548de591e9dda003ad236ff4846051b73628c1d980c7cf3f9cb73067e。
表达式：
`fillna_const(divide(ts_sum(gt(col('Volume'), ts_ema(col('Volume'), 20)), 20), 20), 0.5)`。

500 日 × 256 资产，2024-08-02..2026-08-25；bootstrap 99 次。
TRAIN 诊断搜索；VALIDATION 检验冻结赢家；TEST 未评分。
不根据此次验证结果重新调参。缺少暴露，行业/规模/风格中性化未执行。

## 前后对照
修复前 signature=incomplete，基础处理为空；修复后 known，
基础 winsor + cs_rank 通过训练退化检查，候选数从 104 降到 103。
修复后基础处理训练 RankIC -0.03537194，RAW -0.03537218，收益指标不变。
训练候选 gain≈1.13144，但验证候选 Sharpe -1.17795，RAW 0.99201；
最大回撤 0.05810 对比 0.03824，触发退化门槛，最终保留 RAW。
识别修复不等于获得样本外收益提升。

## 逐方法覆盖
共 65 个方法/参数案例：57 executed，8 requires_additional_inputs_or_control，
0 failed。执行案例检查前缀不变性、资产排列不变性、有限值和训练含成本指标。
成本率 0.001；不将未执行记为通过。
8 项为 ABANDON（控制操作）、三种中性化（缺少暴露）、
LOW_DOF_INTERACTION / OPERATOR_SWAP / WINDOW_REFINEMENT（需要 DSL 重编译）、
MISSINGNESS_FRESHNESS 的 drop（行筛选，不是值变换）。
详细结果见 volume_methods_20260922.json；自动优化前后见 volume_lineage_ab_20260922.json。

## 尚未覆盖
本轮优化库及预处理库完整回归：1825 passed、1 xfailed、16 warnings（104.12 秒）。
HP 滤波的已知非因果行为为预期失败，仍限制 OFFLINE_ONLY。
全仓库 collect-only 得到 24130 项及 14 个收集错误，不能声称全平台验收通过。

列举 metadata 得到 1347 个对象，只读取 4 个小清单；并非全部因子验收。
约 100MB 的新对象超过当前 64MiB 精确对象研究读取上限，未扩大权限或预算。
旧公开示例仍固定旧清单和 8MiB 筛选，本次不声称完成通用发现接口。
历史 PIT 暴露、表达式重编译、最终 TEST 评估仍需后续工作。
