# Review #10 执行报告（2026-08-09）

审计基线：并发会话已提交 `a2dc80c` / `ecffd56` 等 settle pass 之上，本批次与其
WIP 同树编辑。原则：静态确认的 bug → 直接修根因；无法完全证明 → 写性质测试；
失败修根因；每项带永久回归测试；production 全部 fail-closed。

## 一、已修复（fixed，含回归测试）

### R10-P0-001/002 — 缓存真正绑定执行作用域 + data_scope 补 key
`storage/data_scope.py`、`runtime/engine.py`
- **根因**：`_build_cache()` 在构造期只用 `compute_data_scope(data_source)`，且
  该函数漏了 `semantic_filters/read_mode/snapshot_now_only/mining_coverage_threshold/
  enforce_mining_gate/strict_unknown_fields` —— 同源不同读模式的因子共享缓存
  namespace；二级 SourceRef / 执行语义完全不在 key 里。
- **修复**：新增 `ExecutionCacheNamespace` + `compute_execution_cache_scope()`，
  在 run() **编译后**用 plan 的 `source_dependency_hash` + `DataExecutionScope`
  构造 per-run 缓存作用域，`_make_context(cache_scope=...)` 用 `with_scope`
  重设 plan cache —— 不同二级依赖/执行语义绝不共享子树缓存。
- **测试**：read_mode / semantic_filters / snapshot_now_only / mining 门 → scope
  不同；二级依赖不同 / decision_time_policy 不同 → 24 位 execution scope 不同；
  同输入 → 相同（`test_r10_cache_production_gates_2026_08.py`）。

### R10-P0-003 — Production 预编译 plan 重新过门
`runtime/engine.py`
- **根因**：`engine.run(factor, plan=..., analysis=...)` 跳过 `compile()`，而
  PIT / production-operator / fast-path 门全在 compile 里 → research 编译的 plan
  可在 production 直接执行。
- **修复**：新增 `_assert_production_plan_gates()`，production 模式下预编译 plan
  重跑 `assert_production_plan_ops` / map-groups / fastpath / stub / PIT。
- **测试**：`panel_rolling_pca_loading`（experimental）预编译 plan → 抛
  `ProductionPolicyViolation`；research 模式不强制；纯 column plan 通过。

### R10-P0-004..007 — PCA 家族（panel_model.py）
- **004**：`industry_rolling_pca_loading` 调 `_pca_loading` 少传 `prev` →
  `TypeError`（行业 ≥4 成员即崩）。loading 已改无状态 3 参。
- **005**：PCA loading 隐藏 stateful（跨窗 sign 对齐 `prev`）→ **stateless**：
  每窗独立按最大 |loading| 定位强制为正（tie 按面板列序固定）→ full==chunk==
  incremental 精确相等。
- **006**：单股票当天缺值让整截面 NaN → `_pca_transform` 用 `np.where` 把缺失
  股票标准化值钉到训练均值（0），公共分量保持有限；该股票自身 residual 仍 NaN，
  其余股票有效；`_pca_commonality` 训练期缺口同法修补。
- **007**：active 只需 2 个观测太宽松 → `min_obs = max(2, ceil(window*0.7))`，
  输出 `active_breadth/median_coverage/min_coverage` 遥测。
- **测试**：4 成员行业不崩、chunked==full、中段起算==full、单股缺失隔离、
  低覆盖股排除 + 遥测、resid_vol/momentum history==2*(W-1)
  （`test_r10_panel_model_2026_08.py`，含 r9 history contract 14 例）。

### R10-P0-008 — 双层滚动历史显式声明
`runtime/execution_contract.py`
- `panel_rolling_pca_resid_vol/momentum` 加 `_nested_pca_resid_extension`
  compound transform：`rows = 2*(window-1)`，不再被 generic parser 低估为 W。

### R10-P0-009 — panel_model 整数参数 ParamSpec
`cleaned_operators/cross_section/panel_model.py`
- `component/label_horizon/n_regimes/n_experts/window/n_components` 声明
  `ParamSpec(dtype=int, min=...)`；`alpha/l1_ratio` 声明 float 界。`n_experts=3.9`
  在调用边界被拒（原 `int(3.9)==3` 静默截断）。

### R10-P0-010..013 — Universe 契约 fail-closed（market/universe.py）
- **010**：production 模式缺 required 组件（或全缺）→ `UniverseContractError`；
  research 允许 partial mask。
- **011**：`MaskComponentSpec(field, predicate)` + `apply_mask_predicate` ——
  `is_suspend=="0"`、`stock_list.type=='"CS"'`，不再统一 `!=0`。
- **012**：`is_market_eligible` 从 substring（`"close" in "preclose"`）改为
  canonical FieldID / 末段精确匹配。
- **013**：production strict 轴对齐 —— DataFrame mask 必须与 panel 完全同轴，
  不再静默 reindex 填 NaN。
- **测试**：`test_universe_mask_r10_2026_08.py`（17 例，含既有 universe 测试）。

### R10-P0-014..017 — 字段默认 + registry（fields/spec.py、fields/registry.py）
- **014**：`strict_pit_allowed` / `mining_allowed` / Table `strict_pit_allowed`
  默认改为 `None`（UNKNOWN），production 门 fail-closed；catalog `_f/_table`
  显式值不受影响。
- **015**：`role="status"` 自动 `mining_allowed=False`（StateCategorical，不做
  ts_mean/rank/zscore）。
- **016**：`register_table/register(replace=True)` 抢夺第三方 alias → 需要
  `expected_old_identity`，否则 hard fail。
- **017**：新增 tri-state `ResolvedField/UnknownField/AmbiguousField` +
  `resolve_field()`；`get(strict=False)` 兼容返回 None。
- **连带**：`validate_fundamental_period_contracts` 豁免从 `mining_allowed` 改为
  按 **role** 判结构元数据列（否则 None 默认会让该审计静默跳过）。
- **测试**：`test_r10_field_registry_2026_08.py`（10 例）+ 既有 field 测试 47 例。

### R10-P0-018 — 二级依赖哈希补 dialect
`planner/source_dependencies.py`：`_ref_canonical` 加入 `dialect/dialect_version`，
同表同字段不同 dialect 版本不再共享 dependency hash。

### R10-P0-019 — checkpoint 绑定数据 snapshot
`runtime/stateful_incremental.py`
- `input_identity` 增加 `source_snapshot_scope = _source_snapshot_scope(source)`
  （窗口无关的 `compute_data_scope`，剥掉 start/end）。`_effective_identity` 的
  指纹在恢复时经 `require_for_segment` 复核 —— 数据源/读模式/snapshot 变了 →
  指纹不匹配 → fail-closed 到全量重放，不续用旧状态。
- **测试**：`test_r10_stateful_checkpoint_2026_08.py`（3 例：窗口无关、
  snapshot 变更拒绝、指纹区分）。
- `commit_batch` 已是三阶段原子（先全量验证序列化 → 唯一 tmp → 全成后
  `os.replace`），无需改动。

### R10-P0-022/023/028/031/032/033 — Dedup 全面重构
`cleaned_operators/search/factor_dedup.py`
- **022**：`FactorBehaviorSignature` + `multi_regime_signature()`，默认指纹跨
  gaussian/heavy_tail/trend/mean_revert/ties/gaps 六 regime —— Gaussian 下几乎
  一样但极端行情不同的因子不再共享 bucket。
- **023**：`_date_rhos`/`_per_date_duplicates` —— 逐日 Spearman，要求 median
  rho≥阈值 **且** p10≥0.9 **且** >0.999 的日期占比≥0.9，不再是单次 flatten rho
  （90% 相同 + 10% 相反的 alpha regime 不再被误删）。
- **028**：`factor_signatures(compute, *, ast_hash)` 的 `ast_hash` 必填非空，
  不再返回 `""` 让 caller 后填。
- **031**：`dedup_bucket(compute, *, factor_kind, panel=None)` 强制 `factor_kind`，
  按 kind dispatch。
- **032**：OHLC fixture 用 `np.full(np.nan)` 初始化，`cols%4!=0` 不再有未初始化
  垃圾列。
- **033**：event fixture 拆 `event_bool` / `event_signed_intensity` /
  `event_positive_intensity`。
- **测试**：`test_r10_dedup_2026_08.py`（11 例）+ 既有 search 测试 22 例。

### R10-P0-024..027 — source ref / market / coverage（api/source_ref.py、api/
mining_integration.py、runtime/config*.py、market/universe.py）
- **024**：`transform_source_col(..., strict=True)` —— production typed DSL 拒绝
  裸 `close` 隐式换成 StockMinuteBar；compat 面保留。
- **025**：`RunConfig` 增 `market`；`_resolve_market()` 把 `market`(MarketID) 与
  `calendar_id`(日历标识) 拆开，`ResolvedRunKwargs` 同时携带二者，batch key 也
  含 calendar_id —— 不再 `market = config.run.calendar` 混为一谈。
- **026**：`export_dsl_allowlist_json` 未知 market（如 `ahsare`）→ `ValueError`，
  不再回落 US。
- **027**：`UniverseCoverageMetrics` 三分量：universe_breadth / field_coverage /
  mask_retention；`coverage_metrics()` 供拆解，`coverage_ratio` 保留为
  mask_retention 别名 —— 稀疏字段不再虚报 100% 覆盖。
- **测试**：`test_r10_config_market_2026_08.py`（6 例）。

### R10-P0-029 — PCA reconstruction 最少 2 feature
`panel_model.py`：`ts_feature_pca_reconstruction_error`（及旧名
`cs_autoencoder_reconstruction_error`）单 feature → `ValueError`（rank 必然 0、
输出全 NaN 的死端）。既有测试 `test_autoencoder_error_nonneg` 对齐到 2 feature。

### R10-P0-030 — MoE 专家 bin 退化 fail-closed
`panel_model.py`：`np.unique(quantile edges)` 后若 `有效专家数 < n_experts` →
该 cell NaN，不再静默少跑几个专家。

### 连带（非 R10 直接项）
`valuation_growth_mismatch`：并发会话把 `scale` 从搜索面移除后，内核仍收第 3 个
位置参数但元数据未声明 → 严格位置参数门拒绝 `calculate(ey,g,1.0)`。改为声明
`scale` 为非搜索 `ParamSpec(default=1.0, searchable=False)`（pandas + polars 两
注册）。

## 二、并发会话协调
本批次期间并发会话密集提交（`settle pass 1..4` 等），并把工作树 `git add -A`
提交 —— 我的全部改动被其提交保存（已逐项 grep 验证）。其 WIP 同时也在改
`execution_contract.py`（R10 #4 resolution_error）、`factor_dedup.py`（新增
ashare fixture）、`pit_contract.py` 等，与我的改动共存无冲突。

## 三、回归统计
- 新 R10 测试文件：`test_r10_panel_model_2026_08.py`、`test_r10_field_registry_2026_08.py`、
  `test_r10_dedup_2026_08.py`、`test_r10_config_market_2026_08.py`、
  `test_universe_mask_r10_2026_08.py`、`test_r10_stateful_checkpoint_2026_08.py`、
  `test_r10_cache_production_gates_2026_08.py`。
- 既有受影响测试（r7_search_axis1 / cs_model_next_stage / field_catalog_alignment /
  r7_field_contract_gates / r9_history_contract / universe_mask / round6_alignment）全部通过。
- 未通过的非本批项：`test_model_semantics_fixes.py::test_garch_shock_uses_h_t`
  （GARCH 在 ts_model，与本次无关，属并发会话既有/WIP）。

## 四、本轮未做（交接）
- **Ridge/condition-number 统一 train-only 标准化**（优先清单第 11 项）：审计项，
  `regression_models.py`/ts_model 正被并发会话密集编辑，避免冲突，作为下一批
  首项。`ModelScaleMetamorphicAudit` 可自动覆盖。
- 大规模差分/故障注入套件（A/C/H/I/J/K/L/M 等）—— 建议证据重生成后 nightly。
- 证据链重生成（`certify_primitive_evidence.py` → `generate_layer_manifests.py` →
  `generate_operators_catalog.py`）—— 树稳定后跑，本批改动了 panel_model 元数据
  （ParamSpec）与执行契约，重生成会纳入。
