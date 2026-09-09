# riskfolio_qs v0.2 数据链路初版（仅数据准备与清洗）

## 1. 文档目标与范围

本文给出 v0.2 的初版数据链路，目标是支撑 [riskfolio_qs_v0.2_纯净数学定义.md](riskfolio_qs_v0.2_纯净数学定义.md) 中的方法落地。

本版范围：

1. 输入数据准备。
2. 数据清洗与标准化。
3. 数据适配器与中间层产物设计。
4. 数据质量校验与落盘。

本版不包含：

1. 优化器实现。
2. 约束求解实现。
3. 回测执行实现。

---

## 2. 总体数据分层

建议采用 4 层数据链路：

1. Source 层：上游已对齐数据（Massive 数据、Barra 风险包、策略 alpha、基准权重、上期持仓）。
2. Ingest 层：统一读取后转为标准列名和标准主键。
3. Canonical 层：完成主键对齐、口径清洗、可用性过滤。
4. Bundle 层：按优化输入契约产出标准数据包。

对应目录建议：

1. input_raw/：原始输入路径（只读）。
2. staging/：统一格式后的中间结果。
3. bundle/：最终可直接喂给优化器的数据包。
4. audit/：质量报告、缺失报告、参数快照。

---

## 3. 数学定义到数据输入的映射

纯净数学定义核心对象与数据映射如下：

1. $\mu$（预期收益）
1. 来源：alpha 信号文件。
2. 主键：date, asset。
3. 字段：alpha_raw 或 expected_return。

2. 因子风险输入项 $(F, F_{\mathrm{ret}}, F_{\mathrm{spec}})$
1. 来源优先：外采 Barra 三类数据（暴露、因子收益率、特异性收益率）。
2. 备选：行情收益历史协方差（非 Barra 口径）。
3. 主键：date, asset（及 factor）。

3. $w^b$（基准权重）
1. 来源：基准权重输入文件。
2. 主键：date, asset。

4. $w^{-}$（上期持仓）
1. 来源：上期持仓文件。
2. 主键：date, asset。

5. $F, G, F_{\mathrm{mcap}}$（风格/行业/市值暴露）
1. 风格优先来源：Barra 风格暴露。
2. 行业来源：Barra 行业因子或外部行业映射。
3. 市值来源：StockIndicator.market_cap 或等价市值字段。

6. $F_{\mathrm{ret}}, F_{\mathrm{spec}}$（Barra 因子收益率/特异性收益率）
1. 来源：外采 Barra 收益侧数据。
2. 用途：风险项构建与风险归因。

7. $c, \Lambda$（交易成本参数）
1. 来源：流动性与成交数据估计。
2. 推荐输入：StockDailyBar.Volume, StockDailyBar.Amount, 费用率配置。

---

## 4. 结合对齐版文档的可用上游数据

从对齐版代码验证文档可直接利用的数据资产：

1. 行情主表：StockDailyBar。
2. 可交易股票集合：StockList。
3. 证券主数据：SecurityMaster。
4. 交易日历：Calendar。
5. 指标与市值：StockIndicator, StockValuationDaily。
6. 公司行动：StockDividend, StockCapitalDaily（用于事件过滤与一致性校验）。
7. 代码映射：TickerMap（符号变更修复）。

当前仍需外部补充的数据：

1. Barra 风格暴露矩阵 $F$。
2. Barra 因子收益率向量 $F_{\mathrm{ret}}$。
3. Barra 个股特异性收益率向量 $F_{\mathrm{spec}}$。
4. 基准权重 $w^b$。
5. 上期持仓 $w^{-}$。
6. alpha 信号（$\mu$）。

说明：上述缺口项在 v0.2 为强依赖，未提供时只能退化到“可运行但非目标口径”的占位方案。

---

## 5. 初版数据链路步骤

## Step 0：配置解析

读取配置中的 source_mode（explicit 或 folder），解析所有输入路径。

产物：resolved_input_paths.yaml。

## Step 1：基础读取与标准化（Ingest）

读取 Source 层文件，统一到标准列名：

1. asset（统一 ticker）
2. date（统一到交易日）
3. 数值列统一 float64
4. 类别列统一 string

产物：staging/*.parquet。

## Step 2：主键与时间口径对齐（Canonical）

统一主键为 date + asset，规则：

1. 交易日以 Calendar.is_trading_day 为准。
2. 时间戳统一为 date（本地交易日，不保留时分秒）。
3. 使用 TickerMap 处理历史代码变更。
4. 资产池以当日可交易股票集合为基础（StockList + SecurityMaster 过滤）。

产物：canonical/key_aligned/*.parquet。

## Step 3：行情与收益清洗

基于 StockDailyBar 构建市场输入：

1. 价格字段完整性检查（Open/High/Low/Close > 0）。
2. Ret、Ret_Intra、Ret_Overnight 异常值截尾（winsorize，可配置）。
3. 缺失值处理：
1. 必要字段缺失则该资产当日标为不可用。
2. 非必要字段按规则填补或置空。
4. 公司行动一致性检查（拆股、分红窗口异常跳变标记）。

产物：bundle/market_panel.parquet。

## Step 4：alpha 输入清洗

处理 alpha 文件：

1. 同日同资产去重（保留最新版本或按版本优先级）。
2. 数值异常截尾（横截面 winsorize）。
3. 可选标准化（z-score）与方向统一。
4. 与资产池 inner join，缺失资产置空并记录覆盖率。

产物：bundle/alpha_panel.parquet。

## Step 5：基准与上期持仓清洗

1. benchmark 权重按日归一化（允许容差）。
2. prev_positions 与 benchmark 资产池对齐。
3. 缺失资产默认 0（仅当配置允许）。
4. 校验每日日权重和是否在容差范围。

产物：bundle/benchmark_panel.parquet, bundle/prev_positions_panel.parquet。

## Step 6：Barra 暴露准备（外采读取，不在本系统计算）

优先路线（目标路线）：

1. 读取外采 Barra 风格暴露与行业暴露。
2. 与资产池按 date+asset 对齐。
3. 不在本系统重算 Barra 因子，仅允许做字段映射与缺失校验。

补充路线（兜底，仅研究模式）：

1. 行业暴露用外部行业分类 one-hot。
2. 市值暴露由 market_cap 对数标准化得到 $F_{\mathrm{mcap}}$。
3. 风格暴露缺失时禁止开启相应中性约束。

产物：bundle/exposure_style.parquet, bundle/exposure_industry.parquet, bundle/exposure_mcap.parquet。

## Step 7：Barra 收益项准备（外采读取）

优先路线（Barra）：

1. 读取 $F_{\mathrm{ret}}$ 与 $F_{\mathrm{spec}}$。
2. 校验维度一致性：date、asset、factor 维度一致。
3. 可选派生：按配置窗口从 $F_{\mathrm{ret}}$ 派生 $\Sigma_f$，从 $F_{\mathrm{spec}}$ 派生 $D$。

备选路线（无 Barra 时）：

1. 用市场收益滚动估计 $\Sigma$。
2. 对 $\Sigma$ 做收缩与数值稳定修正。

产物：bundle/risk_model_pack/（至少含 factor_ret 与 specific_ret，可选含 factor_cov 与 specific_risk）。

## Step 8：交易成本输入准备

1. 线性成本系数 $c$：由费率配置 + 流动性分层估计。
2. 冲击系数 $\Lambda$：初版支持对角近似。
3. 成本字段缺失时按配置使用默认分层值。

产物：bundle/cost_panel.parquet, bundle/impact_matrix.parquet。

## Step 9：最终输入包组装与审计

组装标准 InputBundle 对应数据文件，并输出质量报告：

1. 覆盖率（按日期、资产、字段）。
2. 缺失率与异常值统计。
3. 关键约束可用性矩阵（某约束是否可启用）。
4. 数据版本快照（源路径、时间戳、行数、hash）。

产物：bundle/run_{run_id}/ + audit/run_{run_id}/quality_report.json。

---

## 6. 清洗规则（初版冻结建议）

## 6.1 主键与去重

1. 主键：date + asset。
2. 重复记录：按 update_time 或文件优先级保留最后一条。
3. 非交易日记录直接剔除。

## 6.2 缺失值策略

1. 核心字段缺失（close、alpha、benchmark 等）：标记不可用或按配置失败退出。
2. 暴露缺失：
1. 若约束依赖该暴露且 strict=true，则失败。
2. 否则禁用该约束并记录告警。

## 6.3 异常值策略

1. 收益与 alpha：横截面 winsorize（默认 1%/99%）。
2. 暴露值：按风格口径可选截尾与标准化。
3. 成本参数：分层上限截断，避免极值导致优化不可行。

## 6.4 数值稳定性

1. 协方差矩阵最小特征值下界修正。
2. 风险矩阵对称化：$\Sigma = (\Sigma + \Sigma^\top)/2$。
3. 权重和约束容差统一（如 1e-6 到 1e-4 可配置）。

---

## 7. 适配器实现建议（不含优化器）

建议新增/实现模块：

1. io/path_resolver.py：解析 explicit/folder 输入。
2. io/readers.py：统一读取 parquet/csv。
3. adapters/identifier_adapter.py：ticker 映射与资产主键统一。
4. adapters/market_adapter.py：行情清洗与收益面板构建。
5. adapters/alpha_adapter.py：alpha 清洗与标准化。
6. adapters/benchmark_adapter.py：基准与上期持仓处理。
7. adapters/exposure_adapter.py：Barra/行业/市值暴露组装。
8. adapters/risk_adapter.py：Barra 收益项读取与风险派生（可选）。
9. adapters/cost_adapter.py：成本参数构建。
10. core/data_quality.py：质量规则与报告。
11. core/bundle_builder.py：最终输入包落盘。

---

## 8. 初版输入输出契约（数据层）

输入（最小必需）：

1. alpha。
2. market（至少 close/ret）。
3. benchmark。
4. prev_positions。
5. Barra 暴露 + 因子收益率 + 特异性收益率（或历史协方差备选）。

输出（供优化器直接消费）：

1. alpha_panel。
2. market_panel。
3. benchmark_panel。
4. prev_positions_panel。
5. exposure_style / exposure_industry / exposure_mcap。
6. risk_model_pack（含 factor_ret/specific_ret，及可选 factor_cov/specific_risk）。
7. cost_panel / impact_matrix。
8. tradable_mask。
9. quality_report。

---

## 9. 明确可做与暂不能冻结项

可立即实现（不依赖优化器）：

1. 路径解析与标准读取。
2. 主键对齐与去重。
3. 行情、alpha、benchmark、prev_positions 清洗。
4. 质量报告与数据包落盘。

暂不能完全冻结（需上游确认）：

1. Barra 三类输入字段规范与版本规则。
2. 行业分类统一口径（若不直接用 Barra 行业因子）。
3. 成本参数建模口径（费率、冲击系数分层标准）。

严格说明：上述未冻结项不应在 v0.2 中以硬编码默认值替代生产口径，只能作为研究模式兜底。

---

## 10. v0.2 数据链路最小里程碑

1. M1：完成输入路径解析 + 统一读取 + 主键对齐。
2. M2：完成 alpha/market/benchmark/prev_positions 的清洗与落盘。
3. M3：完成暴露与风险包适配接口（含缺口校验）。
4. M4：完成成本参数适配与质量报告。
5. M5：输出标准 bundle 并通过端到端数据回放测试。

该里程碑完成后，即可进入优化器实现阶段。

---

## 11. 字段级输入契约（文件与字段明细）

本节给出可直接实施的文件级契约。字段名以标准化后名称为准，适配器可做源字段映射。

### 11.1 统一主键与通用规则

统一主键：date + asset。

统一类型规则：

1. date：date 或 timestamp（落地后统一到 date）。
2. asset：string（统一大写 ticker）。
3. 权重类：float64。
4. 暴露与风险类：float64。

统一校验规则：

1. 主键不得重复。
2. 同一文件不得同时出现同义字段（如 ticker 与 asset 不一致）。
3. 数值列不得为 inf。

### 11.2 文件 01：alpha 信号文件

建议路径：

1. explicit：input.explicit.alpha
2. folder：<root>/alpha/alpha.parquet

文件格式：parquet 或 csv。

必填字段：

1. date（date/timestamp）
2. asset（string）
3. alpha_raw（float64）

可选字段：

1. alpha_z（float64）
2. alpha_rank（float64）
3. alpha_version（string）
4. update_time（timestamp）

关键校验：

1. 每日覆盖率 >= min_alpha_coverage（配置）。
2. 同日同资产重复时按 update_time 取最新。
3. alpha_raw 缺失率超过阈值则失败。

### 11.3 文件 02：行情文件（market）

建议路径：

1. explicit：input.explicit.market
2. folder：<root>/market/market.parquet

对齐版可映射来源：StockDailyBar。

必填字段：

1. date
2. asset
3. close（float64）

建议必填字段（用于风险与成本）：

1. ret_1d（float64）
2. volume（float64）
3. amount（float64）

可选字段：

1. open, high, low
2. vwap
3. adj_factor

关键校验：

1. close > 0。
2. ret_1d 绝对值超过阈值做异常标记或截尾。
3. date 必须属于交易日历。

### 11.4 文件 03：基准权重文件（benchmark）

建议路径：

1. explicit：input.explicit.benchmark
2. folder：<root>/benchmark/benchmark.parquet

必填字段：

1. date
2. asset
3. benchmark_weight（float64）

可选字段：

1. benchmark_name（string）
2. rebalance_id（string）

关键校验：

1. 每日权重和在容差内接近 1。
2. 权重不得为 NaN。
3. 资产池与 market 可对齐。

### 11.5 文件 04：上期持仓文件（prev_positions）

建议路径：

1. explicit：input.explicit.prev_positions
2. folder：<root>/prev_positions/prev_positions.parquet

必填字段：

1. date
2. asset
3. prev_weight（float64）

关键校验：

1. 每日 prev_weight 与当前资产池可对齐。
2. 缺失资产可按配置补 0，否则失败。

### 11.6 文件 05：Barra 风格暴露文件（style exposure）

建议路径：

1. explicit：input.explicit.risk_factor
2. folder：<root>/risk_factor/barra_style_exposure.parquet

必填字段：

1. date
2. asset
3. factor_name（string）
4. factor_exposure（float64）

宽表兼容字段：

1. date
2. asset
3. 其余列均视作风格因子列

关键校验：

1. 同日同资产同因子不得重复。
2. 因子集合需满足配置要求的最小子集。
3. 缺失因子按 strict_factor_coverage 决定失败或禁用约束。

### 11.7 文件 06：行业分类/行业暴露文件

建议路径：

1. explicit：input.explicit.industry
2. folder：<root>/industry/industry.parquet

两种支持格式：

1. 分类表：date, asset, industry_code
2. 暴露表：date, asset, industry_name, industry_exposure

关键校验：

1. 分类表需可一对一映射到 one-hot。
2. 同日同资产行业不得多重冲突。

### 11.8 文件 07：Barra 因子收益率文件（factor return）

说明：本契约将因子协方差 $\Sigma_f$ 视为可选派生产物，不作为必填外采文件。

建议路径：

1. explicit：input.explicit.barra_factor_ret
2. folder：<root>/risk_factor/barra_factor_ret.parquet

必填字段：

1. date
2. factor_name（string）
3. factor_ret（float64）

关键校验：

1. 同日同因子不得重复。
2. 因子集合与风格暴露文件一致。

### 11.9 文件 08：Barra 特异性收益率文件（specific return）

说明：本契约将特异风险向量/对角阵 $D$ 视为可选派生产物，不作为必填外采文件。

建议路径：

1. explicit：input.explicit.barra_specific_ret
2. folder：<root>/risk_factor/barra_specific_ret.parquet

必填字段：

1. date
2. asset
3. specific_ret（float64）

关键校验：

1. 同日同资产不得重复。
2. 覆盖率低于阈值时失败或降级。

### 11.10 文件 09：可交易状态文件（tradable mask）

建议路径：

1. explicit：input.explicit.tradable
2. folder：<root>/tradable/tradable.parquet

必填字段：

1. date
2. asset
3. tradable（bool）

可选字段：

1. suspend（bool）
2. limit_up（bool）
3. limit_down（bool）

关键校验：

1. tradable 缺失默认 false（或按配置失败）。
2. 不可交易资产在后续资产池中剔除。

### 11.11 文件 10：交易成本参数文件（linear cost）

建议路径：

1. explicit：input.explicit.cost_linear
2. folder：<root>/cost/cost_linear.parquet

必填字段：

1. date
2. asset
3. linear_cost_bps（float64）

可选字段：

1. buy_cost_bps
2. sell_cost_bps

关键校验：

1. 成本值 >= 0。
2. 超过上限阈值做截断或失败。

### 11.12 文件 11：冲击成本参数文件（impact）

建议路径：

1. explicit：input.explicit.cost_impact
2. folder：<root>/cost/cost_impact.parquet

初版支持对角形式必填字段：

1. date
2. asset
3. impact_lambda（float64）

扩展支持矩阵形式字段：

1. date
2. asset_i
3. asset_j
4. lambda_ij

关键校验：

1. lambda >= 0。
2. 矩阵形式需可重建半正定。

### 11.13 文件 12：交易日历文件（calendar）

建议路径：

1. explicit：input.explicit.calendar
2. folder：<root>/calendar/calendar.parquet

可映射来源：Calendar。

必填字段：

1. date
2. is_trading_day（bool）

关键校验：

1. date 连续且无重复。
2. 所有输入数据 date 必须是 calendar 子集。

### 11.14 文件 13：代码映射文件（ticker map，可选）

建议路径：

1. explicit：input.explicit.ticker_map
2. folder：<root>/mapping/ticker_map.parquet

可映射来源：TickerMap。

必填字段：

1. old_asset
2. new_asset
3. effective_date

关键校验：

1. 映射链不得形成循环。
2. 同日冲突映射需失败并人工处理。

---

## 12. 文件级产物清单（Bundle 层）

标准产物建议固定为以下文件：

1. bundle/alpha_panel.parquet
2. bundle/market_panel.parquet
3. bundle/benchmark_panel.parquet
4. bundle/prev_positions_panel.parquet
5. bundle/exposure_style.parquet
6. bundle/exposure_industry.parquet
7. bundle/exposure_mcap.parquet
8. bundle/risk_model/factor_ret.parquet
9. bundle/risk_model/specific_ret.parquet
10. bundle/risk_model/factor_cov.parquet（可选派生）
11. bundle/risk_model/specific_risk.parquet（可选派生）
12. bundle/cost/linear_cost.parquet
13. bundle/cost/impact.parquet
14. bundle/tradable_mask.parquet
15. audit/quality_report.json
16. audit/resolved_input_paths.yaml

每次 run_id 需在独立目录落盘，禁止覆盖历史产物。

---

## 13. 长宽表转换与 Pivot 规则（强制标注）

本节定义所有需要的长表/宽表转换，避免实现时出现列名、主键或聚合口径不一致。

### 13.1 通用规则

1. 标准长表主键：date + asset (+ factor_name / industry_name)。
2. 标准宽表主键：date，列为 asset 或 factor。
3. pivot 前必须先做主键去重。
4. 若出现重复主键且无 update_time，默认失败，不允许隐式聚合。
5. 若有 update_time，先按 update_time 取最新，再 pivot。

### 13.2 alpha：长表 -> 宽表

输入推荐：长表（date, asset, alpha_raw）。

目标：

1. 优化输入：alpha 宽表（index=date, columns=asset, values=alpha_raw）。
2. 审计保留：alpha 长表原始版本。

pivot 规则：

1. index=date。
2. columns=asset。
3. values=alpha_raw。

失败条件：

1. 同一 date+asset 出现多值且无法判定最新记录。

### 13.3 benchmark 与 prev_positions：长表 -> 宽表

输入推荐：

1. benchmark 长表（date, asset, benchmark_weight）。
2. prev 长表（date, asset, prev_weight）。

目标：

1. benchmark 宽表（date x asset）。
2. prev_positions 宽表（date x asset）。

pivot 规则：

1. benchmark values=benchmark_weight。
2. prev values=prev_weight。

额外规则：

1. 对齐同一资产列集合。
2. 缺失资产按配置补 0 或失败。

### 13.4 行业分类：分类表 -> one-hot 暴露矩阵

输入格式 A：分类长表（date, asset, industry_code）。

目标：行业暴露长表与宽表并存。

1. 长表：date, asset, industry_name, industry_exposure（0/1）。
2. 宽表（按资产）：index=date+asset, columns=industry_name, values=industry_exposure。

转换规则：

1. 先将 industry_code 映射为 industry_name。
2. 按 date+asset 做 one-hot。
3. 每个 date+asset 行 one-hot 和应为 1。

失败条件：

1. 同一 date+asset 映射多个行业且未配置优先级。

### 13.5 Barra 风格暴露：长表 <-> 宽表

输入格式 A：长表（date, asset, factor_name, factor_exposure）。
输入格式 B：宽表（date, asset, factor1...factorK）。

目标：统一成两种可互转形态。

1. 优化约束侧常用：长表。
2. 矩阵运算侧常用：宽表（按 date 分片得到 F_t）。

pivot 规则（长 -> 宽）：

1. index=[date, asset]。
2. columns=factor_name。
3. values=factor_exposure。

unpivot 规则（宽 -> 长）：

1. id_vars=[date, asset]。
2. value_vars=全部因子列。
3. 产出 factor_name, factor_exposure。

失败条件：

1. 同一 date+asset+factor_name 多值冲突。

### 13.6 Barra 因子收益率：长表 -> 宽表（主输入）

输入：长表（date, factor_name, factor_ret）。

目标：按日因子收益率宽表 F_ret(t)。

pivot 规则：

1. index=date。
2. columns=factor_name。
3. values=factor_ret。

可选派生：

1. 按配置窗口从 F_ret 派生因子协方差 Sigma_f。

### 13.7 specific return：长表 -> 宽表（主输入）

输入：长表（date, asset, specific_ret）。

目标：

1. 按日资产宽表 F_spec(t)。
2. 可选派生特异风险向量与对角阵 D_t。

排序规则：

1. 必须按当日资产列顺序对齐（与 alpha/benchmark 同序）。

### 13.8 市值暴露：长表 -> 标准化向量

输入：长表（date, asset, market_cap）。

转换：

1. mcap_log=log(market_cap)。
2. 按日做截面标准化得到 F_mcap_exposure。

输出：长表（date, asset, F_mcap_exposure）或宽表（date x asset）。

失败条件：

1. market_cap <= 0 的资产未被清洗。

### 13.9 成本参数：长表 -> 向量/矩阵

线性成本输入：长表（date, asset, linear_cost_bps）。

输出：按日资产向量 c_t（与资产顺序对齐）。

冲击成本输入：

1. 对角形式（date, asset, impact_lambda） -> 对角向量/对角阵。
2. 矩阵形式（date, asset_i, asset_j, lambda_ij） -> 方阵 Lambda_t。

矩阵重建规则同因子协方差。

### 13.10 产物落盘形态约定

建议所有关键对象同时保留：

1. 一份长表（审计友好）。
2. 一份宽表或矩阵分片（计算友好）。

建议文件示例：

1. bundle/exposure_style_long.parquet
2. bundle/exposure_style_wide.parquet
3. bundle/risk_model/factor_ret_long.parquet
4. bundle/risk_model/factor_ret_wide.parquet
5. bundle/risk_model/factor_cov_matrix/{date}.parquet（可选派生）

这样可以在可追溯性与计算效率之间保持平衡。