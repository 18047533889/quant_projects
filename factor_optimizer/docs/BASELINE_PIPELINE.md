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

OLS 现在按有效设计矩阵的秩检查剩余自由度，包含截距：
\(df_t=n_t-\operatorname{rank}([1,X_t])>0\)。
多列冗余行业变量不再仅因列数多于股票数而被误拒绝；
没有剩余自由度的饱和模型返回缺失，不能把求解舍入噪声排名成因子。
当天有效样本中全零的暴露列在求解前删除，避免无信息列改变 SVD
形状、秩判断阈值及数值结果，也减少计算量。
完整截面仅剩机器精度量级残差时归零；公式和阈值见预处理库
[中性化说明](https://github.com/HKUST-QUANT-SOCIETY/factor_preprocess/blob/main/docs/NEUTRALIZATION_NUMERICS.md)。
含中性化的基线身份更新为 baseline.v2-ols-rank-dof，不冒充旧算法。

- 谱系已声明缩尾时不再次添加默认缩尾。
- 谱系任何位置已声明 CS rank 时不再次添加 CS rank，后续搜索也不重复加入排名候选。
- 暴露未提供时跳过中性化并记录 `neutralization_missing_exposures`，不生成假暴露。
- 暴露表要求 `date`、`asset_id`、`available_time` 及声明的数值列；
  日期/股票键唯一，且使用记录的可用时间不得晚于该截面的决策时间。
  这是调用方数据的时点检查，不是对暴露来源的 PIT 认证。
  重复键与可用时间检查只针对本次请求的日期/资产；未参与 TRAIN 的未来
  重复记录不影响 TRAIN 选择，到验证期真正使用这些记录时仍必须拒绝。
- 不默认填补未来数据，不用整段时间均值或方差标准化。

## TRAIN 退化检查与 VALIDATION 确认

基础预处理先在 TRAIN 与 RAW 使用相同资产/日期有效交集比较 QE RankIC：

$$d_t=IC_t^{\rm baseline}-IC_t^{\rm RAW}.$$

用移动块 bootstrap 保留时间依赖与缺失日期位置。若均值差的置信上界
低于默认 `-0.01`，认为有证据表明发生明显训练退化，退回 RAW。
覆盖率或有效日期不足同样回退，不把缺失指标填成零。
“没有检出明显退化”不等于“证明有效改善”。

若含中性化的整个基础流水线在 TRAIN 不可用、覆盖不足或明显退化，
仅在 TRAIN 再检验一次去掉中性化的缩尾/排名流水线。
该回退同样需要通过原有退化和覆盖检查，不是无条件接受；
只删除 neutralize，不偷偷更改其他参数或补造暴露。
诊断保留 neutralization_attempt 的完整失败证据，并标注
neutralization_train_rejected。它表示含中性化流水线未被接受，
不等于已经统计证明唯一原因就是中性化。
如果其余步骤也不合格，仍以 RAW 为基准搜索/回退。

基于保留的基础预处理值进行 TRAIN 诊断与修复候选搜索。
所有修复候选仍与原始 RAW 比较；冻结的执行计划包含基础预处理和后续修复，
输出重放不遗漏前置步骤。VALIDATION 只确认一个 TRAIN 选定方案：

- 纯基础预处理：置信下界必须不低于 `-maximum_baseline_loss`。
- 带进一步修复的候选：维持正向改善及 `minimum_improvement` 门槛。
- 未通过就返回 RAW，不在 VALIDATION 中依次尝试其他方案。

上述“去掉中性化后再检验”只发生在 TRAIN；VALIDATION 遇到暴露错误
仍拒绝唯一已冻结方案并保留 RAW，绝不在那里重新选择另一条预处理流水线。

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

基础预处理 TRAIN 退化保护仍使用 RankIC；后续默认选择与验证已改为
[joint.v1](JOINT_SELECTION.md)，包含 Sharpe、RankICIR、回撤、换手与稳定性。
已提供真实 COS 配方与因子文件的哈希绑定及有限语法的谱系识别；
任意配方的完整谱系、具有可信历史可用时间的真实暴露自动加载，
以及真实冻结 TEST 和全样本分开的最终评估仍未全部完成。
下文早期回放数字来自当时的 RankIC 模式，不代表 joint.v1 的回放结果。

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

## 中性化补充审查（2026-09-22）

[真实因子及暴露的代数验证证据](neutralization_real_20260922.json)：
2024-08-02、256 股票，真实 COS 因子配合 sw_l1 行业和自由流通市值对数。
31 列普通暴露与旧实现残差完全一致；追加 300 个全零“未出现行业”列，
旧实现 0 个有效输出，新实现 256 个，且与普通暴露结果逐值完全一致。
这只是求解器验证，不是历史可交易回测；这些暴露没有传入自动选优，
没有把收益改善当作新证据，也没有评分 TEST。

DataAccess 按一天 time_range 读取行业/估值时，预算预检仍分别匹配
3889/2585 个文件；本轮通过已核对的当天 physical_scope 限定单个文件，
没有扩大扫描预算或直接绕过 DataAccess。通用日期裁剪问题尚待独立修复。

历史可用性调查：2024-08-02 的 5107 条行业记录 UpdateTime 为
2026-06-02 06:50:12（UTC+8）；5107 条估值记录为 2026-06-13 00:13:20。
UpdateTime 不是原始公布时间，不能用它证明 2024 年已可用，也不能反过来
断言信息直到 2026 年才首次公开。2026-08-24 的 5211 条行业/估值记录
分别在当天 16:04/16:10 更新，与午夜决策标记不同。
未伪造 available_time=TradeDate。读取还出现行业 UpdateTime 时区类型的
schema 告警，未以关闭严格校验来声称数据契约已经正确。

最终两库回归：1,768 passed、1 xfailed、16 warnings，133.90 秒。
覆盖冗余暴露、截距导致的饱和回归、全解释截面的舍入噪声、全零行业列
不改变结果、TRAIN 中性化失败后的基础步骤回退，以及未来重复键/晚到
暴露不能改变 TRAIN 选择。中间 API 文档同步检查失败已通过重新生成修复。
根 pytest 仍为 1 skipped、14 collection errors、2 warnings，57.57 秒；
失败模块见 [既有审查清单](METHOD_AUDIT_20260922.md)，不属于已通过的两库测试。
