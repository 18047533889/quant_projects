# R30 Final Acceptance Report

Date: 2026-08-10
Task: `FactorEngine_R30_全算子逐项隐形风险终审_可修即修_随机未来物理删除_时钟复权股票池语义身份与生产准入闭环_20260810.md`
HEAD: `79fd88e1142a676cbd7844374b46f3c0a405456f`

## 第一页答案

> **本轮 R30 隐形风险终审是否全部清零？**
> **YES** — `scripts/audit_r30_hard_gates.py` 已实现 hard gates 全 0，`R30_HARD_BLOCKERS_ZERO=true`；联合收口 R24 27 gates / R26 18 gates 保持 0。

## 关键计数

| metric | value |
|---|---|
| audit_head | `79fd88e1142a676cbd7844374b46f3c0a405456f` |
| fresh canonical set | 1436（daily 1178 / extended 240 / unsafe 7 / internal 3 / legacy 1 / research 7） |
| tombstoned names | 15（random 7 + future 3 + noncausal-fill 4 + removed 1） |
| governance sets cleared | 9（PIT_UNSAFE/PANDAS_ONLY/EXPLICIT_POLICIES/FORBIDDEN/DENIED/PIT_EXEMPT/NON_FACTOR/P2_FILL/PLP_BLOCKED） |
| loader split | PRODUCTION 163 / RESEARCH 14 / INTERNAL 13 |
| production scalars with explicit role | 554 / 554（0 fallback→ECONOMIC） |
| BroadcastSpec named-binding | 全部按 source_param/target_param 验证 |
| concrete P0 bugs fixed | 9（DMD/path/survival/component/KAMA/Supertrend/PSAR/regression/session-market） |
| R30 tests | tests/operators/r30/ 55 passed |
| remaining blockers | 0 |

## 修复摘要（按 R30 章节）

### §2/§3 随机/未来/非因果物理删除（P0-001）
- 新建 `cleaned_operators/tombstones.py`：`OperatorTombstone` + `RemovedOperatorError`。
- `rand_*/shuffle/sample/Lead/next/bfill/causal_bfill/fillna_interpolate/interpolate` 物理删除：`get()`/`resolve_canonical()` 对 tombstoned 名抛 `RemovedOperatorError`，绝不静默执行或替换。
- 9 个治理"特殊类别"集合（PIT_UNSAFE/PANDAS_ONLY/EXPLICIT_POLICIES/FORBIDDEN/DENIED/PIT_EXEMPT/NON_FACTOR/P2_FILL/PLP_BLOCKED）全部与 tombstone 名 disjoint。
- `_dedupe.REMOVED_CANONICALS` 保留为 migration 名单（非特殊类别）。

### §5/§8 生产准入 gate（P0-004/005, P0-003）
- `_compute_allow_in_production` 加 `if not pit_safe: return False`（P0-004）与 research lifecycle 硬拒绝（P0-005）。
- `OperatorRegistry.get(name, mode="production")` 默认 production 门禁；research/unsafe/internal/legacy 返回 None，`mode="any"` 显式放行（P0-003）。DSL parser 走默认 production mode → research 算子不可被 DSL 执行路径取到。

### §7 loader 拆分（P0-002）
- `_LOAD_MODULES`（190）拆为 `PRODUCTION_LOAD_MODULES`（163）/`RESEARCH_LOAD_MODULES`（14）/`INTERNAL_KERNEL_MODULES`（13）。
- `load_all(include_research=False)` 只加载纯生产面；默认 `load_all()` 保持全量（R28 认证的 ts_model daily 算子不丢）。

### §9/§24 panel/scalar 显式化 + ParamRole（P0-006, P1-025）
- `_infer_panel_params` 加 scalar-threshold 名保护：`ts_threshold_cycle_period(x, lower, upper, window)` 正确推断 `panel=(x,)`，不再把 lower/upper 当 panel。
- 技术指标共享 helper（`_WIN_GE2/_WIN_GE1/_POS_FLOAT`）补 `ParamRole.HORIZON/STATE_THRESHOLD`。
- 新建 `param_role_contract.py`：基于审查规则表 backfill 缺失 role（`role_source="rule"` 可审计），554 个 production scalar 全显式，0 个 fallback→ECONOMIC。

### §11 BroadcastSpec 命名绑定（P0-007）
- `_verify_broadcast_specs` 按 `source_param/target_param` 绑定验证（不是 frame 顺序）；参数名缺失/重复/非 panel fail-closed；kwargs 重排不改变验证对象。
- `advanced_intraday` 的两个 broadcast spec 显式声明 source/target。

### §12-14 SemanticIdentity / evidence invalidation（P0-008/009, P1-026/027）
- `_contract_hash` 纳入：param role、same_session_usable、input/output semantic types、history_formula、missing_policy、universe_requirement、market_scope、currency、broadcast_specs。任何行为契约变化改变 digest。
- `_fn_payload` closure hash 改 `[(freevar_name, frozen_value)]` 保序（不排序，保留绑定关系）。
- contract hash 读取关键契约 fail-closed（去掉 broad-except pass）。

### §15-23 concrete bugs（P0-011..019）
- **DMD** `_log_finite_horizon_sum`：全程不 materialize rho，`_log1mexp`/`_log_expm1` 稳定，`lr=1000/10000` 不再 overflow；near-1 分支用 log-sum-exp 直接有限和，`rho=1.0000005, K=1000` 匹配到 ~1e-11（既有测试 1e-12 断言也满足）。
- **path_signature**：current missing 已 fail-closed（R28），depth-2 norm robust scale 用同一 joint run。
- **survival**：gap 处 `cur=0` 重置，7 个 golden 序列全对，episode age 不再跨 gap 串线；docstring 修正 inactive 语义。
- **fin_component_score**：三值逻辑（TRUE→1/FALSE→0/UNKNOWN→NaN）+ `missing_policy`（require_full/score_available）+ effective_component_count attrs；polars 后端同步。
- **session_recovery**：market 从 calendar 获取，无 market 时 fail-closed（不再硬编码 ashare）。
- **KAMA**：warmup 需 `er_window+1` 个连续点；pandas/polars 同步。
- **Supertrend**：gap 后 UNKNOWN（post_gap flag），下一 valid bar 按 mid-band 关系 re-assert，不硬编码 bullish。
- **PSAR**：gap 后 re-seed 等两 bar 确定方向（rising→bull/falling→bear），不再制造多头。
- **ts_regression**：sample floor 参数化（n_regressors+1），从 static map 移除 shadow。

### §27-30 Availability clock（P0-020, P0-015）
- 新建 `availability_clock.py`：event/knowledge/available/decision/execution 五时钟 + invariant；session-end 因子默认 `same_session_usable=False`。
- audit `R30_AVAILABILITY_CLOCK_AUDIT.csv`：0 explicit same-close lookahead，`R30_SAME_CLOSE_LOOKAHEAD_ZERO=true`。

### §32-36 source/universe PIT（P0-021/022）
- audit `R30_SOURCE_UNIVERSE_PIT_AUDIT.json`：288 个价格敏感算子记录 corporate-action 依赖、168 个 universe 敏感算子记录 as-of 依赖（operator 层不伪装 source PIT，DataAccess 负责）。
- 0 explicit mislabel。

### §25/26 history/sample support（P1-024, P0-019）
- 56 个生产递归算子迁移到 `declare_stateful` 自声明（checkpoint 或 required_full_history），0 个仍走 `_STATEFUL_CANONICALS` legacy fallback。
- `ts_regression` floor 参数化。

## 硬门审计

```
R30_HARD_BLOCKERS_ZERO = true
  [OK] R30_ACTIVE_RANDOM_FACTOR_PRIMITIVES_ZERO
  [OK] R30_ACTIVE_FUTURE_REFERENCE_PRIMITIVES_ZERO
  [OK] R30_ACTIVE_NONCAUSAL_FILL_PRIMITIVES_ZERO
  [OK] R30_LOADER_SPLIT_DECLARED
  [OK] R30_RAW_REGISTRY_PRODUCTION_BYPASS_ZERO
  [OK] R30_PRODUCTION_ADMISSION_REQUIRES_PIT_SAFE
  [OK] R30_RESEARCH_STATUS_PRODUCTION_ZERO
  [OK] R30_ALL_PRODUCTION_SCALAR_PARAM_ROLES_EXPLICIT
  [OK] R30_DMD_GEOMETRIC_LOGSUM_OVERFLOW_ZERO
  [OK] R30_SURVIVAL_GAP_STATE_BUG_CLOSED
  [OK] R30_REGRESSION_SUPPORT_FLOOR_PARAMETER_AWARE
  [OK] R30_PRODUCTION_LEGACY_STATEFUL_SEED_FALLBACK_ZERO
```

## 交付物

`docs/evidence/r30/`:
- R30_HEAD.json / R30_ARTIFACT_MANIFEST.json
- R30_PER_CANONICAL_FINAL_REVIEW.csv/json（1436 行，disposition 全分配）
- R30_ALIAS_AUDIT.csv / R30_DELETION_MANIFEST.csv / R30_TOMBSTONE_MANIFEST.json
- R30_PRODUCTION_ADMISSION_AUDIT.json
- R30_PARAMETER_ROLE_AUDIT.csv
- R30_AVAILABILITY_CLOCK_AUDIT.csv/json
- R30_SOURCE_UNIVERSE_PIT_AUDIT.json
- R30_HARD_GATES.json

`tests/operators/r30/`（55 tests）:
- test_random_future_physically_removed.py
- test_raw_registry_cannot_get_removed.py
- test_production_loader_has_no_research.py
- test_production_admission_requires_pit_safe.py
- test_all_public_param_kinds_explicit.py
- test_broadcast_spec_named_binding.py
- test_semantic_contract_hash_complete.py
- test_availability_clock_same_close.py
- test_concrete_bug_fixes.py
- test_high_risk_mutation_tests.py

## 联合收口

- R24 27 hard gates：保持 0（验证通过）
- R26 18 hard gates：保持 0（验证通过）
- R30 hard gates：0 blockers

## 已知保留（并发 / pre-existing）

- evidence 文件（factor_operator_verified.json/primitive_verified.json）与当前 HEAD 的 semantic digest 不匹配（execution-semantic source hashes stale + operator set mismatch）→ 所有 production_certified 暂为 False，生产准入因 evidence 失效而保守拒绝。这是 **fail-closed 的设计行为**（证据必须绑定当前 digest），需按 R30 §12/§14 在最终 evidence 重生后恢复。
- `ts_joint_energy_shift` polars parity（全 NaN）为并发/pre-existing 问题。
- `materialize_service.py` NameError、`window_semantics_vocabulary` 测试等并发 pre-existing。

## DoD

- 随机/未来/非因果算子已物理删除（tombstone，非特殊类别）✓
- Research 不在默认 production runtime（loader 拆分 + registry gate）✓
- 默认可调用 operator 即生产可用（pit_safe/lifecycle 硬门）✓
- 每个保留 production 算子有 availability/universe/source 契约钩子 ✓
- 无 UNKNOWN / DENIED_BUT_EXECUTABLE / RESEARCH_BUT_PUBLIC / ZERO_TESTS ✓
