# FactorEngine 第五批整改报告（2026-08-08）

针对外部 AI 评审提出的**第五批架构级问题清单**（调用合同 / Stateful 单一事实源 /
Source 归一化与排序 / 认证独立 / Edge 矩阵 / Registry 溯源 / 假 Polars / 搜索空间
canonicalizer / 具体算子 / 治理层），已在本机代码库全部整改或文档化。**未删除任何
算子**。

> 协调说明：另一个 Claude 会话同期在编辑同一棵树（R6 工作）。本批与其工作有交集，
> 已在报告中标注哪些项由其完成、哪些文件当前归其管辖（不在编辑中改动）。

---

## 一、P0-01..08 统一调用合同（全部完成）

| 编号 | 问题 | 修复 | 文件 |
|---|---|---|---|
| R5-01 | Polars backend 绕过中央参数校验，`window=5.9` 在 pandas 拒绝但在 polars 可能 `int→5` | 所有 polars 基类（Series/Scalar/Transform/TwoVar）经 `_prepare_call` 走同一 `validate_operator_call`；删除裸 `float→int` 截断 | `cleaned_operators/base_polars.py` |
| R5-02 | 自定义 `calculate()` 可绕过 validator（scalar_where / gather_ext 等） | `register()` 新增守卫：直接覆盖 `calculate` 且未声明 `_HANDLES_CALL_CONTRACT` 即注册时报错；迁移 gather_ext / scalar_where / `_PolarsRowSum` 走 validator；9 个直接覆盖类加标记 | `cleaned_operators/registry.py`、`gather_ext.py`、`common/scalar_where.py`、`fiscal_event_ops.py`、`overhaul/base.py`、`daily_panel.py`、`spectral_ext.py`、`activity_clock.py`、`weighted_moment_ext.py` |
| R5-03 | Polars 二元运算按列交集（x=[A,B,C] vs y=[A,B] 只算 A/B） | `_align_cols` 严格化：列集不一致 → raise | `cleaned_operators/common/polars_ops.py` |
| R5-04 | polars `where` 未知条件 `cast(Boolean)` 把 null→False（落入 false 分支） | 未知（NaN/null/Inf）条件 → 输出 null；`cond!=0→a, cond==0→b`，与 pandas 一致 | 同上 |
| R5-05 | typed broadcast 只查列数，未查 wide-panel 标的身份 | 广播分支额外校验**列标签完全一致** + instrument 身份子集 | `cleaned_operators/base.py` `_validate_panel_axes` |
| R5-06 | 未声明 kwargs / 额外 positional 不拒绝（隐藏参数可改结果） | 严格拒绝：kwargs 必须 ∈ param_names ∪ param_aliases ∪ 内核别名集，operator 可标 `variadic`；额外 positional 超 param_names 即报错。修复关联：`relation_distribution_skew/kurtosis` 标 variadic；`fin_roe_cash_gap`、`fin_core_earnings_ratio`、`fin_noncore_income_ratio`、`fin_comprehensive_income_gap`、`fin_oci_to_equity`、`fin_discontinued_operation_ratio`、`fin_minority_profit_share` 补 `period_id` 声明（pandas+polars 双后端） | `cleaned_operators/base.py`、`relation/distribution.py`、`fundamental/quality_v2.py`、`fundamental/polars_quality_v2.py` |
| R5-07 | `tolerance` 被整数白名单误判（`pattern_*(tolerance=0.02)` 被拒） | 从 `_INTEGER_PARAM_NAMES` / `_NONNEGATIVE` 移除 `tolerance`、`tau`；用 `ParamSpec` 显式声明（`_normalise_integer` 已支持 spec 优先） | `cleaned_operators/base.py` |
| R5-08 | composite lowering 在 ParamSpec 校验前 `int(5.9)=5`（`window_int` / `_macd_windows`） | lowering helper 全部 `strict_int`：非整数 → 规划期报错，不再静默截断 | `planner/lowerings/_helpers.py`、`planner/lowerings/technical.py` |

## 二、P0-09..13 Stateful 单一事实源（全部完成）

| 编号 | 问题 | 修复 | 文件 |
|---|---|---|---|
| R5-09 | `ts_ema` full（pandas `ewm(adjust=False, ignore_na=False)`）与 checkpoint `_ema_segment` 缺值策略不一致 | 建立 **RecursiveKernel**：EMA 复刻 pandas 精确递推（NaN/Inf 视为缺失 → `old_wt` 按 `(1-alpha)` 衰减、输出 carry；有效观测后 `old_wt` 重置 1.0）。checkpoint 携带 `(weighted_avg, old_wt, valid_count)` | 新建 `recursive_kernel.py`；`stateful_runtime.py` 委托 |
| R5-10 | `RSI_WILDER` / `ADX` seed 用 SMA，full 用 pandas EWM 递归值 | RSI 的 gain/loss 改为 pandas `ewm(alpha=1/w, adjust=False, min_periods=w)` 递归；ADX 全程 `min_periods=1`（无 SMA seed），缺失 bar 的 DM 按 pandas `.where(mask,0.0)` 喂 0.0 | 同上 |
| R5-11 | `ATR_WILDER` 首根 TR 取 `high-low`，full 首根 TR 为 NaN | 首根 / 缺失 prev close 的 TR → None（NaN），ATR 用 pandas EWM | 同上 |
| R5-13 | checkpoint 只绑手工 `semantic_version`，改代码忘 bump 则旧 checkpoint 兼容 | fingerprint 增加 `implementation_hash`（kernel code object 摘要），改递推即失效 | `stateful_runtime.py` `_effective_identity` |
| R5-12 | checkpoint 未绑数据版本（历史修订后旧 checkpoint 继续跑） | `data_version` 字段（snapshot/version/revision/lineage/calendar 等）流入 fingerprint | 同上 |

验证：`tests/operators/test_recursive_kernel_parity.py`（11 个测试）——每个 kernel 对拍
pandas 生产实现（含 NaN gap / 连续 NaN / 首行 NaN / Inf），且**在每个 split 点**缝合
checkpoint 与 full-history 数值一致。stateful 全套 22 测试通过。

## 三、P0-21 / 26-27 Source 归一化与排序（并发会话已覆盖 + 确认）

- **#21 扫描排序**：`build_scan_polars_long` 已由并发会话加 `enforce_source_ordering`
  （P0-10：`(instrument, time)` / 分钟 `(instrument, time, session)`）。已确认存在且生效。
- **#26 单一 FieldPlan**：`storage/sources/field_plan.py` 已由并发会话实现
  `NormalizedFieldPlan` 并接入 `data_access_source.py`。
- **#27 broad except**：`data_access_source.py` 当前由并发会话积极编辑（fields/* 22:00
  被改），本批不介入；其 `_normalize_contract_columns` 已重构为 NormalizedFieldPlan 路径。

## 四、P0-14/15 认证独立与 Edge 矩阵

- **#14 六证独立**：`semantic_cert` 已由并发会话按 P0-18 拆分为每证独立的 evidence
  （`semantic_golden_verified` / `temporal_prefix_verified` / `source_contract_verified`），
  实现证据不再反推语义/时序/来源证书。本批确认无 vacuous 反推。
- **#15 Edge gate vacuous pass**：重写 `edge_requirements.py` —— `INF_REQUIRED` 从
  `NAN_REQUIRED` 独立；新增三态 `edge_evidence_status()`（complete/incomplete/undeclared）；
  `edge_gate_strict` 开关下未声明 edge 要求的算子**不通过**（生产认证应开启 strict 并
  把未声明算子归入 required 集或 `EDGE_IMMUNE`）；`EDGE_IMMUNE` 显式豁免真正 edge-insensitive
  算子。默认保持 lenient 以免在证据重生成前 mass-fail。

## 五、P0-22..25 假 Polars native 清理

- 新增 `polars_backend_kind.py`：把注册为 `polars` 的实现真实分类为
  `polars_expression_native` / `polars_column_numpy_udf` / `pandas_materialization_fallback`。
  `polars_geometry_math`、`register_polars_udf`（rolling_pack）等 `to_pandas()` 桥按源码/
  模块判定为 fallback，供 planner 排除出 auto fast-routing。
- `PanelSchema.value_columns()` 统一 metadata 列跳过规则、canonical signature 唯一化
  属跨模块重构，已在报告延后项列出。

## 六、搜索空间 Canonicalizer + 参数敏感度门（R5 §19 / #34-36）

- 新建 `parameter_canonicalizer.py`：`ParameterCanonicalizer.canonical_key(params)` 对
  齐次权重归一化（(4,2,1)==(8,4,2)==(40,20,10)）、纯尺度移除（rank 不变）、严格正边界
  （Keltner multiplier=0 → 退化拒绝）、对称参数排序；`sensitivity_verified()` 实现
  `parameter_sensitivity_verified` CI 门（每个 searchable 参数至少两个值产生不同输出）。
- `EaseOfMovement.volume_scale` → `ParamSpec(searchable=False)`（P1-86 基础上升级）。
- `UltimateOscillator` 三权重 kernel 内归一化到 `w3=1` + `ParamSpec(searchable=False)`。
- `KeltnerUpper/Lower/Position` multiplier 强制 `> 0`（0 退化到 mid band）。

## 七、具体算子修复（#33 / #37 / #38 / #39 / #49）

| 编号 | 修复 | 文件 |
|---|---|---|
| R5-33 | `bounded_nvi/pvi` 缺失 volume 单独 `vol_valid` 控制：已知下降→收益、已知非下降→0.0、缺失→NaN（不再注入假 0 收益） | `price_volume/liquidity_v2.py` |
| R5-37 | `PSAR` 从**首个 jointly-valid (h,l)** 播种（首行缺失不再污染）；gap 后 reset valid-bar 历史，t-1/t-2 引用改走有效 bar（不再 half-carry half-raw-index）；连续数据与旧实现逐点一致 | `technical/indicators_v2.py` |
| R5-38 | `NATR` 负/零 close → NaN（PositivePrice，不再 `abs()` 洗白坏数据）|pandas + polars 双后端 | `technical/indicators_v2.py`、`technical/polars_indicators_v2.py` |
| R5-39 | `ts_autocorrelation_time` 强制 `1 <= max_lag < window`（超窗 lag 永不被算，是重复搜索节点） | `memory_ext.py` |
| R5-49 | 统一 OHLC 域合同：`ohlc_contract.py` 提供 `ohlc_valid_mask` / `assert_valid_ohlc`（H>=L、H>=max(O,C)、L<=min(O,C)、O,H,L,C>0），坏 vendor bar fail-closed | 新建 `ohlc_contract.py` |

## 八、治理层（#50 / #51）

- **#50 Surface frozenset 重组**：新增 `extend_extended_only()` / `extended_only_canonicals()`；
  **50 个模块**的 `_surface.EXTENDED_ONLY_CANONICALS = frozenset(set(old)|new)` 全部改为
  live 扩展调用，消费者用查询函数不再拿到 import 时快照。已验证 live 查询可见。
- **#51 名字/前缀推断 scope**：`OperatorPolicy.scope` 已是显式字段（并发会话实现）；
  本批确认 `micro_*` 前缀治理仍存在，完整迁移到 OperatorSpec 属后续任务（见延后项）。

## 九、验证

- 所有改动文件 `py_compile` 干净；`load_all()` 正常（1360 算子）。
- 新增回归测试：
  - `tests/operators/test_recursive_kernel_parity.py`（11）：kernel 对拍 pandas + 逐点缝合。
  - `tests/operators/test_fourth_batch_fixes.py`（14）：沿用。
- 套件通过：`test_final_pack`(64)、`test_deepening_2026_08`、`test_financial_next_stage`、
  `test_dynamics_pack`、`test_advanced_ops`、`test_ashare_typed_ops`、`test_stateful_*`(22)、
  `test_causal_operators`、`test_technical_extensions`、`test_relation_index_ops`、
  `test_microstructure_ops` 等，累计 **300+ 通过**。
- 直接功能验证：#3/#4 polars 严格对齐/where 空值、#8 5.9 拒绝、#33 nvi 缺失→NaN、
  #36 Keltner 0 拒绝、#35 UO 比值不变、#38 NATR 负价→NaN、#39 max_lag 校验、
  #37 PSAR 首行缺失/中间 gap re-seed、ParameterCanonicalizer 各用例、surface live 查询。

## 十、当前已知失败（**非本批代码引入**）

| 失败 | 归属 |
|---|---|
| `test_candle_patterns_extended` ×2（`pit_safe is False`） | **证据过期级联**：并发会话新增算子/改源码改变了 evidence hashes 与 operator-set，`infer_operator_policy` 返回 pit_safe=False（覆盖未触碰算子）。恢复路径：并发批次落定后重跑 factor→primitive→recipe→manifests→catalog evidence 链 |
| `test_report_change_breadth_all_rising` | **并发会话语义变化**：`alpha_language_events.py`（20:40 被并发改）的 median-centered change z 对"匀速增长但增速递减"数据输出 -1；测试期望 +1 与现设计矛盾 |
| `test_alpha_language_semantics[ts_location_shift/ts_scale_shift/ts_vol_term_structure]` | 同上：`alpha_language_distribution.py`（20:39 被并发改）对零 MAD/常数数据 fail-closed，与旧测试期望不符 |

以上均在本批编辑之前已存在；本批改动对其零影响。

## 十一、文档化延后项（需与并发会话协调后执行）

1. **#27** source adapter 生产路径去 broad `except Exception`（文件当前归并发会话）。
2. **#22-24** `PanelSchema.value_columns()` 统一 metadata 列跳过；backend 不持有独立
   signature（canonical signature 唯一校验）；`polars_backend_kind` 接入 planner routing。
3. **#51** production eligibility 完全迁移到 OperatorSpec（去掉 `micro_*` 等前缀推断）。
4. **#40** `cs_knn_peer_mean_ex_self` 输出单位继承 target；`cs_knn_graph_dirichlet_energy`
   输出 `unit(target)²`（当前为 ratio 的显式声明）。
5. **R5-15 strict 化**：edge_gate_strict 置 True + 把全部未声明算子归入 required 集或
   EDGE_IMMUNE + 重生成 edge evidence。
6. 并发会话的 `strict_float`/`param_branches` 编辑与本批 `strict_int`/`_LOWERING_DECLARATIONS`
   已兼容共存；后续如有冲突以 `.v2` schema 与 `extend_extended_only` 为准。
