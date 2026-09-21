# 配方与真实因子值绑定

## 为什么需要这个入口

因子名字相同，不代表配方与数值版本相同。不能从某份历史报告读到 rank，
就假设当前对象使用同一配方；也不能把 report_storage.json 当作处理谱系。

研究接口：

    from factor_optimizer.research_manifest import read_bound_factor
    bound = read_bound_factor(
        store, "declared_landing_manifest", "declared_factor_values", factor_id,
        manifest_params=manifest_parameters,
        factor_params=factor_parameters,
        allow_research=True,
    )

两个数据集必须已经由调用方明确注册在 DataAccessStore 中。
该接口不会根据不可信元数据任意下载其他路径，不读取凭据，
也不会执行 source_python、fe_dsl 或元数据里的文字指令。

## 自动检查

1. DataAccess 有界读取清单，要求一条包含 factors 映射的记录。
2. 对应因子的 verified 必须严格为 true。
3. 仅允许已知研究状态 materialized_not_evaluated、
   evaluated_optimization_pending；质量阻断和未知状态均拒绝。
   “已评估、待优化”不表示统计合格，更不表示生产准入。
4. 声明的数据集与参数解析出的精确 URI 必须等于清单 URI。
5. 通过原 DataAccess 入口读取因子值，再核对：

$$URI_{\mathrm{read}}=URI_{\mathrm{manifest}},\quad
SHA256(\mathrm{bytes}_{\mathrm{read}})=SHA256_{\mathrm{manifest}},\quad
|\mathrm{bytes}_{\mathrm{read}}|=\mathrm{bytes}_{\mathrm{manifest}}.$$

同时保留清单自身的来源 URI 和内容 SHA256。
DataAccess 原有授权、扫描/结果预算、HEAD/GET/HEAD 和临时文件清理仍生效。
本次没有扩大下载上限，没有创建整库或数据集副本。

这些是“生产者声明与具体对象版本绑定”，不是证明 DSL 确实生成了每个数值，
不是历史时点可得性认证，也不是上游从未使用过测试期数据的证明。

## 排名与谱系

DSL 只经过有大小/节点限制的语法解析，不调用 eval/exec。

- contains_cs_rank：语法树某处有 rank 或 cs_rank 调用；
- output_is_cs_rank：根节点是 rank 或 cs_rank 调用；
- col('rank') 中的字符串不算排名操作；
- where(..., rank(...), neg(rank(...))) 有内部排名，但整个输出不等于统一截面排名。

lineage 采用保守识别，仅接受 col、有限数值常量、加减乘除/安全除法、
绝对值/取负、正向历史窗口 ts_delta，以及外层 rank/cs_rank 链。
这是一个明确的初始支持集，不是完整 FE DSL 编译器。
未知函数、参数形式或分支返回 None，不能解释成“明确未处理”。

已识别且没有排名的基本表达式返回空的 TransformLineage；
外层排名返回对应的不可变步骤。传入现有基础处理入口：

    # batch 必须由 bound.factor.table 中的同一份因子值对齐构建。
    result = optimize_factor_batch(
        batch, labels,
        lineages={bound.factor_id: bound.treatment_signature},
        allow_research=True,
    )

明确无预处理的源表达式会尝试默认 winsor / cs_rank；
已有排名时不再次添加。中性化仍需要真实且时点可用的暴露，
没有暴露不会伪造；所有变化仍受 TRAIN 退化检查和 VALIDATION 确认。
接口不自动证明调用者任意组装的 batch 仍来自该对象，不能挂名别的矩阵。

### 不完整公式中的正向排名证据

`bound.lineage` 仍仅表示已识别的完整顺序流程；复杂分支返回 None。
推荐向自动优化器传 `bound.treatment_signature`：已识别流程沿用完整签名，
未识别流程返回 incomplete 签名，但保留语法树中已确认的 rank/cs_rank 存在。
这不会伪造分支的执行顺序，也不等于 output_is_cs_rank。

基础处理遇到 incomplete/unknown 签名仍不添加任何默认处理；如果其中
cs_rank=True，则记录 cs_rank_already_present，并从候选搜索删除
REPRESENTATION_RANK。同样，顺序处理记录中出现“已知排名 + 未知步骤”
也不能抹去已知排名。其他明确允许的研究修复仍可评估，未知公式不被标成
“已确认未预处理”。这项禁止重复排名的规则是用户约束，不代表内部排名
与输出排名在数学上等价。

2026-09-22 补充验证：三个先失败后通过的用例覆盖未知步骤不能抹掉已知
排名、不完整签名实际阻止排名候选、真实绑定接口返回的分支排名证据进入
基础处理决策。两库全套 1772 passed、1 xfailed、16 warnings（135.90 秒）。
另行确认 ts_rank 不被当作截面排名；未知完整性状态仍保留。
根测试仍有同一批 14 项收集错误（56.48 秒），未计入通过。

另选真实 weekly_db5616cd85a477b3，表达式
ts_delta(col('valuation.a_cap'), 20)，由 DataAccess 读取并核对来源清单、
SHA256 和字节数。500 日 × 256 股票，TRAIN 覆盖选择资产，
bootstrap_draws=99，固定 10bps 成本。新签名入口正确产生 winsor / cs_rank，
基础 TRAIN IC 退化检查通过，94 个搜索候选中没有重复排名候选。
然而联合指标含缺失或非有限值，未产生可确认赢家，最终保留 RAW；
这不是收益改善证据。没有提供历史暴露，也没有读取 TEST 指标。
详情见 [真实签名入口回放](rank_signature_real_20260922.json)。

## 依赖与安装

新增 research 可选依赖组，列出 FE、QE、FP、DataAccess。
团队应使用同一发布快照的内部仓库/受控 wheel，不从同名未知第三方包代替。
最低版本号不等于内容锁定；DataAccess 需包含有界 JSON 研究读取接口，
以本项目配套发布快照为准。

## 测试范围

使用真实 DataAccessStore、JSON/Parquet 适配器和小型 Arrow 表；
仅模拟外部网关传输。覆盖 URI、字节数、摘要不匹配，未验证标记，
质量阻断、未知状态，排名作用范围，以及识别结果进入基础处理后的排名去重。
复杂表达式保持未解析，不为了让流程执行而猜测语义。

## 2026-09-22 真实绑定与自动优化

[回放证据](bound_cos_audit_20260922.json)。
从已读取清单中按因子名顺序选择首个已知非阻断状态、大小小于 8 MiB 的对象，
不按收益选样。清单中 68 个条目为 evaluated_optimization_pending、
3 个为 materialized_quality_blocked；满足本次状态与大小条件的有 2 个。

选中 weekly_4cbd7ca6dccc61dc，表达式为
ts_delta(col('valuation.free_cap'), 20)。实际文件 5,400,494 字节，
SHA256 为 35ad096b62be98ce5766fe3df4e651bd7930ee377683d6282171a6984ed14a59，
与清单完全一致。源表 2,588 行、5,462 列，经对齐和 TRAIN 覆盖筛选后，
回放 500 日 × 256 股票，2024-08-02 至 2026-08-25。

明确的源表达式谱系使 winsor / cs_rank 真正进入基础预处理，
267 个有效 TRAIN 日的明显退化检查通过；没有提供可用暴露，未假装执行中性化。
94 个候选中 TRAIN 选出的倒 U 型修复通过固定联合验证门槛：

| VALIDATION 指标 | RAW | 选中候选 |
|---|---:|---:|
| 成本后 Sharpe | -0.1774 | 0.8771 |
| 最大回撤 | 9.5506% | 7.7173% |
| RankIC | -0.02155 | 0.01376 |
| RankICIR | -0.3357 | 0.2122 |
| 平均全名义换手 | 0.04116 | 0.03634 |
| 最差三分块 Sharpe | -3.3453 | -1.1700 |

使用既定 10bps 研究成本，没有为此样本降低门槛。
最差块仍亏损，不能把“验证确认相对改善”写成“曲线始终向上”。
本结果不是 TEST 结果或可交易收益承诺；真实 TEST 未评分，未部署生产因子。
还需补齐历史暴露、更完整 DSL 谱系与完整逐层状态衰减。

最终两库回归 1,744 passed、1 xfailed、16 warnings，149.90 秒。
HP 非因果性为既有预期失败，仍受 OFFLINE_ONLY 限制。
根 pytest 仍为 14 项既有收集错误，56.68 秒，清单见
[已有审查记录](METHOD_AUDIT_20260922.md)；没有宣称根测试通过。
