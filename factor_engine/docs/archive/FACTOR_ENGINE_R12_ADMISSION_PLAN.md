# FactorEngine R12：全算子可用化 / 挖掘准入层整改计划

> 基线 HEAD `135b221`。本文是两份 AI（Gemini DeepResearch）整改方案的落地计划：
> 方案 A = 《全算子"可直接挖掘化"最终改造方案》（Admission Matrix + BlockerCode + Mining Lanes）
> 方案 B = 《全算子可用化最终整改任务》（删除不可用公共算子 + `get_mining_operators` 单一发现层 + promote_or_delete + 不变量）。
> 共同目标一句话：**FactorEngine 中凡有研究价值、数据具备、数学成立的算子，都必须被 AlphaProbe/AlphaMiner 实际看见；不能单独做 alpha 的转成 state/gate/condition 角色接入；真正永不可用的从公共体系删除。**

## 现状探测结论（全部实测）

- 注册总量 **1409**，`unclassified=0`（surface: daily 1179 / extended 124 / research 95 / unsafe 7 / legacy 1 / internal 3）。
- catalog 每算子已带六证字段：`implementation_certified / semantic_certified / temporal_certified / source_contract_certified / edge_case_passed / backend_passed`；当前**全部 False**——evidence 工件（`evidence/*.json`）在 HEAD 之后未重生成（已知 R11/R14 延期项），`production_certified` 因此全 False。恢复顺序 = `sync_primitive_evidence → certify_primitive_evidence → certify_factor_operator_evidence → certify_recipe_evidence → manifest/preset/allowlist/llm_prompt`。
- catalog 已有：`role`（仅 8 个非空）、`input_grain/output_grain`（117 个 minute→daily）、`stateful`（48 True）、`same_session_usable`、`window_semantics`、`operator_certification`、`backend_meta`。
- **危险三角已归类 UNSAFE**（`operator_surface.py:42`：arg/tan/cot/sec/csc/cosh/sinh）。
- **原始数学/诊断工具已存在独立 `ResearchToolRegistry`**（fft/pca/wavelet/dropna/fillna/jarque_bera_test/ACF/max_drawdown…），不进 Factor DSL（`test_operator_surface.py` 钉死）。
- 未来函数/随机核已从主 registry 注销：`OperatorRegistry.get("bfill") is None`，测试断言的就是"查不到"。
- **checkpoint 基建完整**：`stateful_contract.py::StatefulCheckpointRegistry`（按字段 type/finite/range 校验），EWM 家族（ts_ema/RSI_WILDER/ATR_WILDER/ADX/MACD_*）`segmented_execution_supported=True`；`runtime/stateful_incremental.py` + `stateful_runtime.execute_stateful_segment` 已打通分段执行。**剩余 ~48 个 full-history 递归算子**（KAMA/Supertrend/PSAR/DMI/DX/NATR/PPO/PVO/Keltner/TSI/DEMA/TEMA/ChaikinOscillator/ForceIndex/ts_sma_cn/state_*/event_refractory/cross_event/directional_change_*/threshold_cycle/state_episode_*/ts_interval_nesting_depth/candle_gap_atr）在 registry 有 spec 但 `segmented_execution_supported=False`——缺每算子增量 kernel + serialize/restore + bit-parity 测试。

## 本轮范围（Phase A–E）

### Phase A — 挖掘发现层单一权威 `mining/operator_catalog.py`（方案 B §十一~§十四，最高优先）
- `MiningRole` 枚举：`ALPHA / CONDITION / STATE / GROUP_STATE / GLOBAL_STATE / INTRADAY_EOD / FUNDAMENTAL_PIT / EVENT / RECIPE_INTERNAL / DIAGNOSTIC / INTERNAL`。
- `MiningOperator` 记录：canonical / role / production_certified / mining_eligible / cost_tier / required_sources / input_semantics / output_unit / searchable_params / allowed_ast_positions / stateful / checkpoint_supported / incremental_supported。
- `get_mining_operators(*, available_sources, target_frequency, market, max_cost, roles)`：**唯一发现入口**。默认仅返回 `production_certified=True` 且角色/源/频符合法者（`mining_eligible=True`）；显式 `include_pending` 才返回未认证（用于补证规划）。
- `assign_mining_role(canonical, catalog)`：确定性映射（六证字段 + role/stateful/grain/scope + 现有 surface/UNSAFE/ResearchToolRegistry）。
- 接线：`api/mining_integration.py` 的 `default_typed_mining_search_space_config` 与 cold-start 改读 `get_mining_operators`（不再直接拼接 surface 集合）。

### Phase B — 准入矩阵 `audit/operator_admission_matrix.py`（方案 A §二/§三）
- 全 canonical 记录：authoring_tier / registered / lifecycle / production_certified / 六证逐门 / 每 backend available+certified / input_grain / output_grain / panel_params / input_semantic_types / output_unit / stateful / recursive / full_history_replay_required / checkpoint_supported / incremental_supported / missing_policy / factor_role / cost_tier / compatibility_only / diagnostic_only / benchmark_only / hidden_from_default_mining / source_required / source_available / **blocker_codes(B01–B32)** / **recommended_action** / **target_mining_lane** / production_eligible / default_mining_eligible。
- 输出：`operator_admission_matrix.csv` / `.json` / `operator_admission_summary.md` + 7 个 lane JSON：`direct_daily / direct_specialized / direct_intraday_eod / direct_high_cost / recipe_only / diagnostic_only / denied`。
- **禁止空泛答案**：每个未入 direct pool 的算子必须带精确 blocker + required fix + data_missing(bool) + code_fixable(bool) + target lane。

### Phase C — promote_or_delete 审计 `scripts/audit_all_registered_operators.py`（方案 B §三十八/§四十二/§四十三）
- 7-case 决策引擎：certify / advanced / condition-state-role / internal / delete-duplicate / delete-no-data / delete-forever。
- 校验 9 条不变量（UNCLASSIFIED==∅ / UNUSED==∅ / FACTOR_SHAPED_RESEARCH_ONLY==∅ / MINING_ELIGIBLE_WITHOUT_CERTIFICATION==∅ / CERTIFIED_FACTOR_NOT_IN_MINING_MANIFEST==∅ / PERMANENTLY_FORBIDDEN_IN_PUBLIC_REGISTRY==∅ / COMPAT_ALIAS_AS_CANONICAL==∅ / SEMANTIC_DUPLICATE==∅ / DEAD_SEARCHABLE_PARAMS==∅）。
- 输出最终 A–J 总表 + 汇总计数（registered public / mining usable / internal / source transform / deleted / unclassified / unused）。

### Phase D — 挖掘 manifest 导出 `scripts/export_mining_manifest.py`（方案 B §三十九/§四十）
- `mining_manifest.json` + `mining_operator_catalog.csv` + `.md`，字段含 allowed_ast_positions / terminal_allowed。
- cold-start 校验门：任何 expression 里的算子必须 ∈ manifest。

### Phase E — 不可用公共算子的收口（方案 B §二~§九，只做分类层；不炸测试）
- 已验证：未来/随机核主 registry 已注销；危险三角已 UNSAFE；原始数学/诊断在 ResearchToolRegistry。
- 补：fin_revision_*/fin_restated_flag/fin_total_operating_accruals 等 no-data 算子 → 准入矩阵 `SOURCE_PIT_BLOCKED / B15_SOURCE_FIELD_MISSING / recommended_action=待真实字段`，公共挖掘不可见（如仍注册则加诊断/隔离标记，不强行 unregister 以免炸 recipe/cold-start）。
- 保持 `unclassified=0` 不变量。

## 后续阶段（明确不在本轮，报告如实标注）
1. **~48 个 full-history 递归算子补 checkpoint/restore**：每算子增量 kernel + `StatefulOperator.initialize_state/update/serialize_state/restore_state` + 500-bar 分段 bit/numeric parity 测试 + `full_history_replay_required=False`。EWM 家族是既有先例（`stateful_contract.py` + `runtime/stateful_incremental.py`），本轮不做以免破坏树。
2. **批量六证 + evidence 重生成**：树稳定后按 §现状结论恢复顺序执行；之后 `mining_eligible` 大规模转 True。
3. **semantic-duplicate 图 / dead-param 自动删除**：依赖真实 A 股数据面板。

## 验证
- Phase A/B/C/D 各自生成产物并自检（100% canonical 有 role+blocker+action+target；invariants 通过）。
- `tests/test_operator_surface.py`、`tests/test_layer_governance.py`、挖掘冒烟 `default_typed_mining_search_space_config` 含/不含新层。
- 相关新测试全绿；`load_all` 0 unclassified。
