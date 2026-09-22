# 稀疏 COS 因子的当前来源对照与旧入口训练边界（2026-09-22）

## 发现
三因子采样中的两只 weekly 因子均只有 66 只资产达到 TRAIN 90% 覆盖。
本轮追查 weekly_b1258507388baa03，不假定缺失一定是停牌或优化器错误。
通过 DataAccess 读取已绑定 COS 因子与当前 ashare_stock_daily_adj：
仅 12 只股票、2023-08-21..2025-10-27，扫描预算 550 文件、结果 8 MiB，
不读收益标签、不发布或覆盖生产因子。

样本按存储因子的 TRAIN 覆盖分成高/部分/零三组，各取 ID 排序前四只，
这是定位问题的有意选择，不是随机样本，不能外推总体比例。
TRAIN 日期复用 automatic_time_split，267 日；窗口计算补足 260 日历史。

## 结果和口径
表达式为：
`multiply(rank(neg(ts_days_since_high(col('AdjHigh'), 252))), rank(efficiency_ratio(col('AdjClose'), 20)))`。

现行 pandas 权威口径：距前 252 根内最近最高点的偏移，不含当前根。
若前窗从旧到新记作 x[0]..x[251]，最高值最近一次位置为 j，则输出 251-j。
只有完整前窗才有效；并列最高取最近一次。
效率比为 |C_t-C_(t-20)| / Σ_(i=0..19)|C_(t-i)-C_(t-i-1)|，
要求完整变化窗口，分母为零时缺失。

- 当前输入中，11 只股票具有完整的高价前窗和有效效率比，1 只缺数据。
- 12 只中的 9 只，存储因子有效位置与当前输入推导的有效位置存在差异，
  合计 1629 个 TRAIN 单元不一致。完整逐资产指标见 JSON。
- 当前实际注册 pandas 与 Polars 的 ts_days_since_high 数值及 NaN 位置一致，
  pandas 有效位置也与完整前窗预期一致。
- 当前 pandas FactorEngine 完整表达式重算得到 3025 个有效单元
  （包含补历史后的整个输入范围，不仅 TRAIN）。
- 对照仅有 12 股票，截面 rank 值与全市场不能直接比较；这里比较可计算性，
  不把窄截面数值当作全市场重建结果，也没有重新评分 TEST。

## 不能据此断言的事
历史 manifest 的 backend_used=["cuda"] 是评估环节写入，
不是 FactorEngine 落值后端。已查 jobs/new_mining_intake.py 的
evaluate_factor_batch → batch.backend_used 链路。
一次尝试把 cuda 传给 FactorEngine 后端工厂被明确拒绝，
所以不存在已完成的 CPU/CUDA 因子落值一致性结论，错误也保存在证据中。

记录 completed_at=2026-09-20 13:44:18，
execution_repair_revision=20260914-source-dependency-cpu-floor-pit-native-dq-v2，
但缺当时行情输入内容身份及完整实际物理执行路径。
当前行情可能已修订；因此不能证明差异由哪一历史算子或数据步骤造成。
尚未修改现行最高点天数算子，也未用填零/强制补值掩盖旧输出。
需要可重放的历史输入与计划，或在独立研究身份下完整重算并验收；
不得原地覆盖内容寻址的历史因子文件。

证据：[来源对照 JSON](sparse_source_probe_20260922.json)。
运行基于 server-c 当前共享工作树，包含其他 AI 未提交的 FactorEngine 工作；
本轮不替它们提交，也不将对照解释为干净发布快照的全引擎认证。

## 同轮修复：旧本地样本入口的 TRAIN 边界
real_batch_audit.load_sample 仍把前 300 日传给资产筛选，而非扣除
预热/隔离后的 267 日。现在与 COS 入口一样复用 automatic_time_split，
并在 inputs 保存 asset_selection_split、asset_selection_train_days。

测试通过真实 DataAccess 读取临时小型合成 parquet，不复制仓库或真实数据：
a 只在预热缺失，b 在实际训练期缺失，旧实现错选 b，修复后选 a。
修复前 1 failed、2 passed，修复后与 COS 相关用例共 28 passed。
完整优化库+预处理库：1841 passed、1 xfailed、16 warnings，105.61 秒。
HP 非因果滤波为预期失败，仍限制 OFFLINE_ONLY。

本轮尝试旧四因子真实文件入口，被 DataAccess 路径白名单拒绝，
路径为 weekly_backtest_output/factor_matrices_all/price_smoothness.parquet。
未扩大白名单或绕过 DataAccess；因此不声称旧四因子入口本轮真实运行成功。
已完成的真实证据来自上述授权 COS 与注册行情数据集。
全仓库既有 14 项收集错误本轮未处理。没有受控性能 A/B，不声明加速倍数。
