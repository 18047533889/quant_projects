# COS 因子、20 层诊断与训练拟合（2026-09-22）

这是完整自动优化目标的第二阶段进展，不是全部完成的声明。
上一阶段证据见 METHOD_AUDIT_20260922.md。

## 本次接通的流程

DataAccess 指定对象研究读取 → 对齐真实交易日与收益标签 →
仅按 TRAIN 覆盖筛选 512 股票 → QE 权威指标计算 →
20 层形状诊断 → TRAIN 拟合修复中心 → 候选搜索 →
一个冻结赢家的 VALIDATION 确认或回退 RAW。TEST 不评分。

来源为用户指定的
`cos://qs-cold/candidate_pool/sunhaiwei/lqtp_show/factor_values/`，
本次列出 83 个对象。事前规则选按 key 排序的前三个不超过 8 MiB 的 parquet，
不根据收益挑选。每个文件通过 DataAccess 读取，临时文件自动清理。
样本 500 交易日（2024-08-02 至 2026-08-25），512 股票，
TRAIN 267 日、VALIDATION 97 日、TEST 保留 100 日。

## 指标口径

按每个交易日有效因子值用 QE 的分位数边界和 max tie policy 分成 20 层，
第 0 层最低，第 19 层最高。每层至少 10 个有效收益标签，
不拆开同值资产强凑层数，不对空层填 0。
只有全部 20 层都有效的日期才进入层间对照。
完整分层日期必须至少 60 天且覆盖不低于 90% 才生成平均收益剖面。

RankIC 是每天因子与未来标签的截面 Spearman 相关。
RankICIR = mean(RankIC) / sample_std(RankIC)，不年化；
零标准差或证据不足返回缺失，不伪造无穷大。
另报三个连续训练块的 RankIC 均值，暴露跨时期方向变化。

设 q[t,k] 为第 k 层的等权标签收益，多空诊断采用总敞口 1：

$$r_t = \tfrac12(q_{t,19}-q_{t,0}).$$

$$Sharpe = \sqrt{252}\,\frac{mean(r_t)}{sample\_std(r_t)}.$$

$$V_t=\prod_{s\le t}(1+r_s),\qquad
MDD=\max_t\left(1-\frac{V_t}{\max(1,V_1,\ldots,V_t)}\right).$$

以上组合指标调用 QE 实现，不在优化库另写指标内核。
252 为明确的日频年度周期假设，可传 periods_per_year 修改。
缺失收益日期使全区间回撤未知，不偷偷填零；多期重叠标签不当作日收益复利。
低于 -100% 的收益需要资本契约，当前不计算其组合统计，但保留 IC 诊断。
这是无成本、未认证可交易性的 top/bottom 5% 诊断组合，不是可执行净收益回测。

## 从诊断到修复

对完整 TRAIN 剖面寻找内部谷底/峰顶（允许偏离中位数）。
在三个连续训练块中，两端相对该谷底/峰顶的方向必须一致，
才提出 U_SHAPE_REPAIR / INVERTED_U_REPAIR 候选。
拟合中心为 (bin_index + 0.5)/20，搭配事前 power=1、2。
它们受相同候选预算、覆盖率、训练评分和单次验证约束，
不因被诊断命中就自动接受。诊断只是提案，不是显著性认证。

默认网格从原 95 项视诊断结果增加至最多 97 项，仍受 maximum_candidates=128
总上限；仅允许的修复家族可以获得诊断提案。小截面不满足 20 层条件时，
仍使用原候选网格并明确记录诊断缺失，不能冒称已做完整分层。
每个 FactorOptimizationResult.training_diagnostics 保存对应训练诊断。

## 运行

```bash
DATA_ACCESS_COS_CLI=admin-cos \
DATA_ACCESS_COS_CACHE_ROOT=/home/sunhaiwei/.cache/quant-dataaccess/research \
DATA_ACCESS_SKIP_COS_MIRROR=1 ASHARE_PARQUET_ROOT=/home/sunhaiwei/cos_data \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
.venv/bin/python factor_optimizer/examples/cos_batch_audit.py --optimize
```

省略 --optimize 仅做 TRAIN 诊断。没有下载生产行情、写 COS 或发布因子的副作用。
调用 DataAccess 网关时使用用户现有授权，不创建凭据或扩大访问权限。

## 仍须完成，不能降格为已实现

- 基础预处理的时点对齐中性化、缩尾、带谱系的 CS-rank 去重及训练退化保护。
- 基于多空 Sharpe / RankICIR / RankIC / 回撤 / 稳定性 / 换手的联合自动选优；
  当前这些新增组合维度只用于诊断，已有最终选优门槛仍为稳健 RankIC。
- 学习式分层衰减和低换手处理与完整因子流水线的连接。
- 冻结后的 TEST 与单列的全样本描述评估、成本/可交易性证据。
- 全流程同口径性能 A/B，以及更广泛真实样本的方法覆盖。

## 指定 COS 样本的实际结果

完整记录见 [本次自动研究 JSON](cos_automatic_20260922.json)。
三个因子均未产生获验证确认的优化赢家，未改阈值来制造改善：

| 因子后缀 | TRAIN RankIC | 原始 RankICIR | 多空诊断 Sharpe | 多空诊断 MDD | 自动结果 |
|---|---:|---:|---:|---:|---|
| 09e3f39d | 0.002041 | 0.013925 | -0.962482 | 12.8034% | 95 候选，86 个训练评分，保留 RAW |
| 1a8fff63 | -0.001174 | -0.022841 | 缺失 | 缺失 | 95 候选，83 个训练评分，保留 RAW |
| 20fcde8b | 证据不足 | 证据不足 | 缺失 | 缺失 | 仅 13 有效 IC 日，隔离 |

第一项在 267 个训练日都有完整 20 层，每层至少 25 个有效标签；
第二项完整 20 层日为 0，不能因为股票数量够就强称分层有效；
第三项有效 IC 日期不足，不能输出可靠整体评分。
第一项三个训练块 RankIC 分别为 -0.006203、-0.009056、0.021381，
显示明显的时期方向差异。这也是必须继续补齐联合指标与稳定性选优、
不能只拿一个平均 IC 当答案的实际证据。

## 下一实现步骤

下一步优先连接基础预处理与联合选优，不重复改写已接通的数据读取：
暴露缺失时明确报告，谱系不明确时不能声称完成 CS-rank 去重；
冻结预处理/修复计划，在共同样本上比较 QE 权威多维指标，验证失败只回退一次。

## 本轮验证

- 优化库 + 预处理库全套：1,694 passed、1 xfailed、16 warnings；
  唯一预期失败仍为已限制离线的 HP 双边滤波前缀不变性反例。
- DataAccess 全套：2,311 passed、1 failed、11 warnings。失败仍是
  test_check_allowlist.py::test_real_repo_passes，检测到其他并行审查脚本的直读；
  本轮未改这些脚本或添加豁免。
- 平台根测试：14 个收集错误、1 个 PyKX 跳过；错误列表与上一阶段
  METHOD_AUDIT_20260922.md 中列出的 14 个模块相同，未宣称全平台通过。
- 25 个专项测试通过，覆盖真实 Store 投影、授权先于网络、对象变更、
  损坏/失败清理、预算拒绝、路径范围、不看未来的资产选择和诊断，
  空分层、重叠标签、缺失日期、偏心 U 型拟合及其实际自动搜索结果。
- 两次真实 COS 读取后，本次研究缓存目录没有留下因子文件。
