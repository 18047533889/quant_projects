# 自动优化审查：2026-09-22（进行中）

本文件记录证据，不代表全部整改完成。正式工作树为 server-c 主干，
不建立额外分支，不部署或发布生产因子。

后续 COS 指定因子池读取和 20 层训练诊断已有进一步落地，
见 [第二阶段报告](COS_DIAGNOSTICS_20260922.md)；下文保留本阶段的历史记录。

## 已定位并已有修复，等待全套复验

- KAMA 的缺失效率比窗口不能当作零效率比；缺失期间保存已初始化状态。
- 极大有限值的标准化不得因方差溢出而变成全零。
- 候选覆盖率同时约束有效资产单元与有效 IC 日期，不能靠丢弃难预测日期获胜。
- 负向因子允许在 TRAIN 上选择“平滑 + 反向”的冻结组合。
- 上尾、下尾和双尾修复均进入适用的搜索分支。
- 逐方法审查改为遍历全部准入平滑参数，而非每种方法只取第一组。

## 验收计划与未完成事项

1. 重跑优化、预处理全套测试；逐方法验证未来数据不影响过去输出、
   股票排列不影响结果、缺失值与极端值行为。
2. 真实因子全部通过 DataAccess 读取，记录来源、时间区间、资产筛选。
   已替换本地样本示例的直读路径；COS 平铺候选池读取适配尚未完成。
3. 基础预处理：中性化需要真实且时点对齐的暴露；缩尾及排名需明确
   计算顺序和处理谱系，不通过名称猜测重复排名。
4. 20 层诊断须有足够截面资产；训练拟合形状及分层衰减，验证确认。
5. 接入 QE 权威多维指标，纳入多空 Sharpe、RankICIR、RankIC、
   回撤、稳定性和换手；目前 research_batch 的选择目标仍仅为 RankIC。
6. 冻结后才运行最终 TEST 评估；全样本结果单列为描述统计。
7. 数值等价性能 A/B；限制内存和重复变换，不能用更少样本冒充同口径提速。
8. 更新公式、参数及失败回退文档，验证后按个人全仓库 / HKUST 同名子树边界推送。

## 解释边界

“执行成功”不等于“效果提升”。缺少暴露、DSL 或时序窗口的处理必须明确
标注所缺输入，不能伪装为已执行。验证不通过保留 RAW，不反复试验证集。
真实样本表现不构成未来收益保证；不得声称绝对无 bug 或全局最优。

## 本次真实回放结果

最终优化库 + 预处理库回归：**1,681 passed，1 xfailed，16 warnings**。
唯一预期失败为 test_hp_filter_is_prefix_invariant：HP 双边滤波已知不因果，
现有注册限制为 OFFLINE_ONLY，不能用于生产因果平滑。API 文档已重新生成并验证。
逐方法原始结果见 [JSON 记录](method_audit_20260922.json)。

通过 DataAccess 读取现存镜像，不更新或下载生产数据。日期为
2024-08-02 至 2026-08-25；500 日、64 只股票；TRAIN 267 日，
VALIDATION 97 日，TEST 保留 100 日且没有评分。资产覆盖筛选仅使用对齐后的
TRAIN 日期。64 只股票不足以验收具有合理每层样本量的 20 层诊断。

共 256 个方法案例：212 次执行通过（含全部 27 个准入平滑参数 × 4 因子），
44 次明确需要额外输入或属于控制操作，0 次执行失败。每次可执行案例均检查
前缀不变性、资产排列不变性、行对齐及有限输出；不将不适用记为成功。

| 因子 | 自动候选 | 结果 |
|---|---:|---|
| price_smoothness | 95 | TRAIN 无足够稳健提升，保留 RAW |
| volume_ratio_persistence | 95 | TRAIN gain 0.050410，验证下界 -0.029138，保留 RAW |
| ema_trend_simple | 95 | TRAIN 无足够稳健提升，保留 RAW |
| cyclical_trend_confidence | 0 | RAW 有效训练 IC 日不足，隔离并保留 RAW |

前三个因子分别有 86、86、87 个候选得到训练评分，其余明确记录不适用。
第四个虽未进入自动搜索，独立逐方法审查仍覆盖其全部方法案例。
这次没有获验证确认的优化赢家，不能声称已改善真实因子的样本外表现。

## 性能及数据读取修复

- 正反向候选复用底层平滑落值，仅保留最近一个数组。
  400 日 × 64 股票、4 个衰减计划、5 次重复的局部 A/B，
  包括 NaN 的数组逐元素完全一致；中位时间 0.280899s → 0.138756s，
  变换阶段约 2.0244 倍。未声称全流程加速倍数。
- 实际读取暴露了日行情分区配置遗漏：约 510 日请求原先列入 2,588 文件。
  为 ashare_stock_daily_adj 声明按 TradeDate 的每日文件分区，
  使剪枝发生在扫描预算检查之前。3 个合成日文件请求其中一天、
  限制只能扫描 1 文件的回归测试，修复前失败、修复后通过。
- 新增 DataAccess 路径授权回归：未授权路径必须拒绝，不能绕过接口直读。

## 全仓库验收中的现有问题（未掩盖）

DataAccess 全套：2,299 通过，1 失败。失败为
test_check_allowlist.py::test_real_repo_passes，检测到 evidence/ 与 work/ 中
19 个其他审查脚本仍直接读取文件，未擅改这些并行工作。

平台根测试在收集阶段有 14 项错误（1 项可选 PyKX 跳过）：

- tests/backend/test_polars_native_batch2_causality.py
- tests/operators/test_cs_polars_native_batch1.py
- tests/operators/test_r11_round2_closure_audit.py
- tests/operators/test_r11_round3_closure_audit_ext.py
- tests/operators/test_ts_advanced_batch1_polars_native.py
- tests/operators/test_ts_nth_value_polars_causality.py
- tests/r44/test_r44_strict_remote_audit.py
- tests/test_ashare_feature_pipeline.py
- tests/test_sql_statistical_nonfinite.py
- tests/test_status_industry_features.py
- tests/test_status_minute_features.py
- tests/test_status_partition_runner.py
- tests/test_status_publish_validator.py
- tests/test_ts_batch1_rank_if_direct.py

原因包括过时路径、缺失模块、测试桩枚举不完整和冻结注册表。
未将这些错误计入“通过”，也没有关闭检查或加入豁免。
