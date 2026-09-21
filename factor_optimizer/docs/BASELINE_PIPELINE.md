# 自动基础预处理与退化保护（研究模式）

`optimize_factor_batch` 在搜索前接收每个因子的 `TransformLineage`，以及可选的
`exposures` 长表与 `exposure_columns` 列名。只有明确提供的空谱系才表示已知未处理；
缺少谱系或谱系包含未知语义，标为 `lineage_unknown`，不猜测它未被处理。
原始因子永远保留为失败回退基准。

## 顺序与计算公式

默认顺序为截面 1%/99% 缩尾、截面 OLS 中性化、截面排名。

$$x'_{t,i}=\min(\max(x_{t,i},Q_{t,.01}),Q_{t,.99})$$

$$\hat\beta_t=\arg\min_\beta\|x'_t-[1,X_t]\beta\|_2^2,
\quad e_t=x'_t-[1,X_t]\hat\beta_t$$

$$z_{t,i}=\frac{\operatorname{rank}_{\rm average}(e_{t,i})-1}{n_t-1}$$

排名沿用预处理库的端点归一化口径，而不是 pandas 的 rank/n；
缺失值规则与退化截面规则沿用该库实现。
最终非线性排名会破坏严格线性正交，不能把排名后的结果描述为“精确中性”。

- 谱系已声明缩尾时不再次添加默认缩尾。
- 谱系任何位置已声明 CS rank 时不再次添加 CS rank，后续搜索也不重复加入排名候选。
- 暴露未提供时跳过中性化并记录 `neutralization_missing_exposures`，不生成假暴露。
- 暴露表要求 `date`、`asset_id`、`available_time` 及声明的数值列；
  日期/股票键唯一，且使用记录的可用时间不得晚于该截面的决策时间。
  这是调用方数据的时点检查，不是对暴露来源的 PIT 认证。
- 不默认填补未来数据，不用整段时间均值或方差标准化。

## TRAIN 退化检查与 VALIDATION 确认

基础预处理先在 TRAIN 与 RAW 使用相同资产/日期有效交集比较 QE RankIC：

$$d_t=IC_t^{\rm baseline}-IC_t^{\rm RAW}.$$

用移动块 bootstrap 保留时间依赖与缺失日期位置。若均值差的置信上界
低于默认 `-0.01`，认为有证据表明发生明显训练退化，退回 RAW。
覆盖率或有效日期不足同样回退，不把缺失指标填成零。
“没有检出明显退化”不等于“证明有效改善”。

基于保留的基础预处理值进行 TRAIN 诊断与修复候选搜索。
所有修复候选仍与原始 RAW 比较；冻结的执行计划包含基础预处理和后续修复，
输出重放不遗漏前置步骤。VALIDATION 只确认一个 TRAIN 选定方案：

- 纯基础预处理：置信下界必须不低于 `-maximum_baseline_loss`。
- 带进一步修复的候选：维持正向改善及 `minimum_improvement` 门槛。
- 未通过就返回 RAW，不在 VALIDATION 中依次尝试其他方案。

TEST 标签不参加上述任何计算。冻结计划可作用于以后到达的因子值；
含中性化的计划重放时仍需提供声明的暴露表，不在计划中藏入可变数据表。

候选阶段只在 TRAIN 的历史前缀落值（含预热），不会先为每个候选计算验证段。
选出唯一方案后才在截至验证段末尾的前缀重放它，使时间平滑保留训练历史，
再仅评分有效 VALIDATION 日期。验证期暴露错误不能改变已记录的 TRAIN 基础处理决定。

## 调用接口

```python
result = optimize_factor_batch(
    batch, labels, allow_research=True,
    lineages=verified_lineage_by_factor_id,
    exposures=aligned_exposure_table,
    exposure_columns=("size", "industry_1", "industry_2"),
    maximum_baseline_loss=0.01,
)
```

上述暴露列名仅示意，不自动推断行业或市值。数据仍应由 DataAccess 提供。
`result.factors[id].baseline_diagnostics` 记录操作、跳过原因、覆盖率和训练判断；
`training_diagnostics` 对应基础处理保留/回退后的 TRAIN 值。
`baseline_accepted` 表示非劣性确认，区别于 `improved` 的正向改善确认。

## 尚未闭合的范围

当前训练退化保护与研究选优仍使用 RankIC。Sharpe、RankICIR、回撤、
换手和稳定性的联合决策、真实 COS 配方谱系/暴露的自动加载，以及最终冻结 TEST
和全样本分开的报告，仍需继续接入；本说明不声称这些已完成。

## 2026-09-22 验证证据

- 两库完整回归：1,711 passed、1 xfailed、16 warnings，122.03 秒。
  HP 双边滤波前缀不变性仍是已知 XFAIL，OFFLINE_ONLY 限制保留。
- 根测试仍有 14 项既有收集错误，57.43 秒；文件清单与
  [既有审查报告](METHOD_AUDIT_20260922.md) 一致，不计为通过。
- 真实行情控制因子：通过 DataAccess 读取复权 VWAP，构造
  `AdjVwap(t)/AdjVwap(t-5)-1`，明确为空预处理谱系。
  这不是把 COS 原因子的未知配方标为已知。
  日期 2024-08-02 至 2026-08-25，500 日 × 256 股票，267 个有效训练日。
  缩尾/排名训练保护通过，缺少真实暴露故没有中性化。
  96 个候选中训练赢家增益 0.084735，验证下界 -0.026733，最终保留 RAW。
  没有查看 TEST 标签；没有声称获得样本外收益提升。
  [完整记录](baseline_market_audit_20260922.json)。
- 性能 A/B：在内存载入修改前单个模块，与当前实现比较 240×40 合成单因子、
  因果平滑与方向候选的完整研究入口；1 次预热、3 对测量。
  输出数组、候选记录、计划身份和评分完全一致。
  中位 4.454971s → 4.290608s，约 1.0383 倍；小样本波动明显，
  不外推为真实大批量性能承诺。没有复制仓库、数据集或虚拟环境。
  [时间记录](baseline_selection_ab_20260922.json)。
