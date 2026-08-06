# FactorEngine 2026-08 字段对齐与算子整改 Plan

依据:用户提供的新审计(20 章,聚焦字段契约/财务时序/估值/股东/模型/表面)与
`/home/shw/COS_ashare_lqtp_data_dictionary.md`(20 表 ~409 字段,单位/PIT/拼接已实测)。
已逐条对照当前 `main`(124b616) 代码核实。以下分「确认修改」「已存在/不需改」「设计级暂缓」三档。

## 0. 核实结论(摘要)

数据字典关键事实(与审计一致):
- `Return` 是 **bp**(÷10000 得小数);`TurnoverRatio/Roe/ShareRatio/Weight` 是**百分数**(÷100 得小数);`PeRatio/PbRatio/PsRatio/PcfRatio*` 是**倍数**。
- `StockDividend` 物理列:`TradeDate/RightRegDate/ExDividendDate/CashDividend/StockDividend/StockTransfer/UpdateTime`,**无 PubDate、无 ExDate**,`effective_time_only`。
- `IndexDailyBar` 股票代码列是 **Symbol**(非 IndexSymbol);`IndexConstituent` 才用 IndexSymbol。
- `Calendar` 时间列是 **TradeDate**(非 Date)。
- `StockList/ETFList/IndexList/StockIndustry` 是**自然日 D1 快照**(含周末),有完整历史,不是"当前快照"。
- `UpdateTime` = 落盘/同步时间,**非行情/公告/修订可知时点**。
- 前复权 `Factor` 方向**未经验证**;价格本身未复权。
- 中国财报流量为**年初至报告期累计**;同日可披露多报告期;NaN=未披露禁止填 0。

## 1. 确认修改(本轮执行)

### P0-A 字段目录 `fields/catalog.py`
1. **StockDividend**:`time="PubDate", knowledge="PubDate", effective="ExDate"` → `time="TradeDate", knowledge=None, effective="ExDividendDate"`(保留 `effective_only`/`strict_pit_allowed=False`)。
2. **IndexDailyBar**:`instrument="IndexSymbol", required_parameters=("IndexSymbol",)` → `instrument="Symbol", required_parameters=()`。
3. **Calendar**:`time="Date"` → `time="TradeDate"`。
4. **StockList/EtfList/IndexList**:`time=None, current_snapshot_only=True` → `time="TradeDate", current_snapshot_only=False, join_policy="exact"`(恢复 `strict_pit_allowed=True`)。
5. **StockIndustry**:`time=None, current_snapshot_only=True` → `time="TradeDate", current_snapshot_only=False`,join_policy 保留 `state_asof`→ 改 `exact`(D1 完整历史;仍需 `required_parameters=("IndustrySource",)`)。
6. **UpdateTime**:所有 `update_time` 字段 role=`knowledge_time` → 新 role `ingestion_time`;财报表 `revision="UpdateTime"` 保留仅为**快照内确定性去重**,注释明确不得用于跨历史修订排序(若消费点仅作 tiebreak 则安全)。
7. **Factor**:`_f("close/...", adjustment="multiply:Factor")` → `adjustment=None` + `metadata["adjustment_status"]="unverified"`(物理源未复权;方向未经验证前不自动乘因子)。Volume 的 `divide:Factor` 同步移除。
8. **字段覆盖扩展**:登记审计点名的缺失字段——StockValuationDaily(`capitalization/circulating_cap/free_cap/a_cap/a_market_cap`)、StockDividend(`stock_dividend/stock_transfer/right_reg_date`)、StockStatus(`change_date/change_type/public_status_code`)、股东(`shareholder_id/shareholder_rank/shareholder_name/shareholder_class/shares_nature/share_pledge/share_freeze`,`mining_allowed=False`)、minute/ETF/index 日线契约。按 FieldRegistry 校验(one_to_many 必 `mining_allowed=False`)。

### P0-B 字段解析与标准化 fail-closed
9. `fields/resolver.py` + `storage/sources/data_access_source.py`:
   - 新增 `strict_unknown_fields` 运行时开关(默认 False 保持现有行为;production 调用路径开启)。开启时未知字段抛 `UnknownFieldSemanticError`;关闭时走 `allow_untyped_field` 显式回退。
   - `_normalize_contract_columns` 去掉外层 `except Exception: pass`:单元化异常在生产路径抛 `FieldNormalizationError`,研究路径记 warning。
   - 缓存记录 `field_spec_hash/source_unit/canonical_unit/scale_applied`。

### P0-C 财务算子时序
10. `transforms_v2._walk_periods`:`order` 按 `period_ordinal` **升序插入**(同 ordinal 保持先见序),修复 append-顺序乱序(§4.1)。`_values/_lag_value` 随之按财政季度序工作。
11. `flow_semantics_v2`:`fin_ttm_quarterly`/`fin_quarter_from_cumulative`/`fin_ttm_cumulative` 迁移到 `fiscal_strict` 内核(`pd_ttm_from_quarterly`/`pd_quarter_from_cumulative`/`pd_ttm_from_cumulative`),获得 ordinal 排序 + `require_consecutive`(§4.2/4.3)。保持 pandas 接口与参数名,统一先 reindex 到 x。
12. `fin_applicability_mask`(pandas `relation/ops.py` + polars `polars_limit_misc.py`):NaN 保留为 NaN,finite 且 >thr → 1,finite 且 ≤thr → 0(§4.6)。两边一致。

### P0-D 估值/股本
13. `valuation_pe_ttm_lyr_gap`/`valuation_pcf_definition_gap`:`log|x|` 塌缩盈亏符号 → 新增 `*_signed_log` 与 `*_positive` 变体(新 canonical,experimental,extended 表面),旧名保留并加 `semantic_note`/`preferred_replacements`(§5.1)。
14. `valuation_cashflow_disagreement`:修复 **polars 缺 winsorize+zscore 的 pandas/polars parity 分歧**(上一轮只改了 pandas 侧)。
15. `valuation_growth_mismatch`:移除自由 `scale` 掩盖;改 `earnings_yield - profit_growth`(输入须已在字段层归一为小数),`scale` 参数改为受支持的显式扩展(非默认)(§5.2)。
16. `capital_change_age`:ChangeDate 非交易日 → `bisect` 取**之后第一个交易日**(§5.3);且不早于该股本状态的 PubDate 已知时点。
17. `circulating_cap_unlock_proxy` → 新增 `circulating_cap_ratio_change`(daily 变化),旧名标 deprecated(§5.5)。`capital_change_magnitude` 保留 daily-state 语义并注释(§5.4 的 snapshot 拆分标记暂缓)。

### P0-E 股东(按 agent 核实后定)
18. `_ratio_map`:ID 存在但 ShareRatio 缺失 → `snapshot_invalid`,当日该股输出 NaN,不补 0(§6.3)。
19. 同快照重复 ShareholderId:可判定的合并(ShareNumber 求和),否则 `snapshot_invalid`(§6.4)。

### P0-F 表面/挖掘门禁固化
20. §12 的 experimental 家族已由 `production_allowlist()`(evidence fail-closed)排除于生产默认挖掘;补回归测试断言:这些 canonical 不在 production allowlist、`allow_in_production=False`,表面保留 daily 供显式 DSL 使用。

### P0-G 验收测试
21. 新增 `tests/operators/test_field_catalog_alignment_2026.py`:契约正确性(StockDividend/IndexDailyBar/Calendar/List/Industry/UpdateTime/Factor)、未知字段拦截、单位归一(Return/10000、TurnoverRatio/100、ShareRatio/100、Weight/100)、`fin_applicability_mask` NaN 语义、`_walk_periods` ordinal 排序、fin_ttm 迁核、PE signed/positive 变体、growth_mismatch 无 scale、capital_change_age 非交易日、股东 NaN/重复 ID、experimental 不在 production allowlist。

## 2. 核实后判定"已存在/不需改"的章节
- **§4.4 部分**:`fin_turnover` 等已有 `period_id` 驱动,`fiscal_strict` 已带 `require_consecutive`;grain 的**编译期强制**依赖字段→算子绑定,本轮以 `field_contract.py` 扩展 + 文档为准(见暂缓)。
- **§2.9/2.10**:`PubDate` 保守可用时点(next_trading_day)与 `period_selection` 属**事件时钟/编译器**层策略;现有 `knowledge=PubDate` 已保证 `PubDate≤decision`。完整可用时点策略列入暂缓。
- **§6.5-6.9 / §9 / §10 / §11**:名称/描述澄清、checkpoint 系统、expectile 重命名等属**大范围设计项**,本轮不改,列入暂缓。

## 3. 设计级暂缓(需用户决策,避免破坏已验证证据/parity)
- S3-like:全量 409 字段逐字段精修(当前先覆盖审计点名的关键族)。
- §2.9 PubDate 可用时点策略(next_trading_day / same_day_after_close / exact_timestamp)。
- §2.10 `period_selection` 编译器强制(latest_period/annual_only/... 缺失即报错)。
- §4.4 flow-grain 编译期拒绝(需要字段→算子 grain 绑定管线)。
- §6.1/6.2 旧排名槽位股东算子整体 deprecated 并替换 ID 版本的全量迁移。
- §6.8 股东集中度趋势按 snapshot_id 推进(而非 ffill 面板 rolling)。
- §9 日内命名 cleanup、§10 模型 prior/in-sample 默认面收窄、§11 递归算子 checkpoint 逐项接入。

## 4. 执行顺序与完成状态
字段目录(P0-A) → 字段解析/标准化(P0-B) → 财务(P0-C) → 估值(P0-D) → 股东(P0-E) →
表面门禁固化(P0-F) → 测试(P0-G) → 全量回归(验证 3721 基线不回退,必要时重生成 evidence 链)。

### 已完成(2026-08-07)
- **P0-A 字段目录**:StockDividend(time=TradeDate/effective=ExDividendDate/knowledge=None)、
  IndexDailyBar(instrument=Symbol)、Calendar(time=TradeDate)、List 三表 + Industry
  (time=TradeDate, current_snapshot_only=False, strict_pit_allowed=True)、UpdateTime→ingestion_time、
  价格/Volume adjustment=None + metadata{adjusted:False,adjustment_status:unverified}、
  覆盖新增估值股本/分红/状态/股东字段(one_to_many 均 mining_allowed=False)。
- **P0-B 字段标准化**:`_resolve_columns`/`_normalize_contract_columns` 生产模式(默认
  `is_production_mode()`,可 `strict_unknown_fields` 覆盖)fail-closed:未知字段→
  `UnknownFieldSemanticError`,跨数据集字段拒绝,归一化异常→`FieldNormalizationError`;
  研究模式保持 raw 回退。
- **P0-C 财务时序**:`transforms_v2._walk_periods` 按 `period_ordinal` 升序插入、`_lag_value`
  按 ordinal 精确匹配(跳期→NaN);`fin_ttm_quarterly/fin_quarter_from_cumulative/fin_ttm_cumulative`
  双后端迁移到 `fiscal_strict` 内核(require_consecutive=True);`fin_applicability_mask`
  双后端 NaN 保留(未披露≠不适用)。修复 fiscal_strict polars 路径对 Object-dtype
  period_id/fiscal_quarter 的健壮性(真实面板前导 NaN 场景)。
- **P0-D 估值**:新增 `valuation_pe_gap_signed_log/positive`、`valuation_pcf_gap_signed_log/positive`
  变体(盈亏符号敏感),旧名标 compatibility_only;`valuation_cashflow_disagreement` polars
  补齐 winsorize(1/99)+zscore 与 pandas parity;`valuation_growth_mismatch` scale 校验
  (必须正有限,禁止自由 scale 掩盖单位);`capital_change_age` 周末/节假日 ChangeDate→
  下一交易日(searchsorted);`circulating_cap_unlock_proxy`→新增 `circulating_cap_ratio_change`。
- **P0-E 股东**:`_ratio_map` 返回 (map, valid):ID 存在但比例缺失、或重复 ID 比例冲突 →
  snapshot_invalid→NaN(§6.3/6.4);同值重复 ID 去重;`holder_pledge_churn` 描述改诚实
  (非 ID 匹配);集中度描述注明"已披露前十大"。
- **P0-F 表面/门禁**:§12 experimental 家族(财务综合评分/cs_*异常/旧回归/旧日内别名)确认
  不在 `production_allowlist`(evidence fail-closed);补回归测试固化;§10.1 in-sample 回归、
  §9.3/9.4/9.5 日内旧名统一盖章 `diagnostic_only`/`compatibility_only`/`benchmark_only` +
  preferred_replacements。
- **P0-G 测试**:`tests/operators/test_field_catalog_alignment_2026.py`(33 项)覆盖上述全部。

### 暂缓(设计级,需决策后实施)
见 §3 列表。核心:§2.9 PubDate 可用时点策略、§2.10 period_selection 编译器强制、
§4.4 flow-grain 编译期拒绝、§6.8 股东趋势按 snapshot 推进、§9/§10 其余命名 cleanup、
§11 递归算子 checkpoint 逐项接入、全量 409 字段逐字段精修。
