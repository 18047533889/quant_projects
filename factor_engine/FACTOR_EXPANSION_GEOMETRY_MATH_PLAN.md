# FactorEngine 算子扩充：K线区间几何 / 结构数学 / 波动率 / 依赖测度 —— 执行计划

日期：2026-08-08
目标：把用户提供的 AI 算子清单（共 **79 个新算子**，26 个新模块）全部落地为
**daily production** 算子：pandas_numpy 参考实现 + polars 桥接 + DSL 可用 + 测试。
duckdb SQL 下推：对纯表达式子集尽力实现（参考既有 advanced 系列 polars-native 先例）。

> 并发编辑说明：同一 tree 上有多个并发 Claude 会话 + evidence certifier 任务在跑。
> 本计划遵循既有「最小追加式改动」约定：**所有新物理内核放全新模块**，模块内
> `_register_surface()` 自行 union EXTENDED_ONLY_CANONICALS 并注册 polars bridge；
> 对共享文件只做两处追加（`_LOAD_MODULES` 末尾追加导入行；`operator_surface.py`
> 末尾追加 `_DAILY_GEOMETRY_MATH_2026_08` frozenset 并 union 进 DAILY_FACTOR_MIGRATED）。
> 不修改任何既有算子实现；不重新生成 evidence JSON（避免与运行中的 certifier 冲突）。

---

## 〇、注册管线（已验证）

1. 模块内定义 `SeriesOperator` 子类 + `@register_operator(name, category, canonical, source)`
   + `metadata = _metadata(...)`（含 `param_names`、`signature:` 等 tags）。
2. 模块底部 `_register_surface()`：`_surface.EXTENDED_ONLY_CANONICALS |= 名字`，
   并对每个名字 `register_polars_bridge(canon)`（polars 桥 = 精确复刻 pandas 参考）。
3. `operator_surface.py` 中新增 `_DAILY_GEOMETRY_MATH_2026_08` frozenset，union 进
   `DAILY_FACTOR_MIGRATED` → 升 daily 表面。
4. DSL allowlist 由 OperatorRegistry 自动构建，daily 算子无需改 parser。
5. SQL 下推需在 `SQL_IMPLEMENTED_CANONICALS` + `emitter._compile_layer` 注册——仅对子集做。

## 一、79 算子清单与模块分组

| 模块（新文件） | 算子 | 数量 |
|---|---|---|
| `interval_geometry.py` | ts_interval_union_coverage / occupancy_entropy / occupancy_mode_distance / nesting_depth / exploration_efficiency / overlap_component_ratio | 6 |
| `structural_levels.py` | ts_structural_level_density / nearest_structural_level_distance / structural_level_strength | 3 |
| `candle_state_space.py` | ts_vector_state_mahalanobis / ts_vector_state_local_density / ts_multivariate_matrix_profile_novelty / ts_matrix_profile_motif_age | 4 |
| `multiscale_trend.py` | ts_multiscale_trend_consensus / dispersion / curvature | 3 |
| `envelope.py` | ts_envelope_compression / pressure / boundary_dwell | 3 |
| `crossing.py` | ts_crossing_speed / acceleration | 2 |
| `extrema_divergence.py` | ts_extrema_divergence_strength / confirmation_rate | 2 |
| `threshold_cycle.py` | ts_threshold_cycle_period / asymmetry | 2 |
| `state_episode_excursion.py` | state_episode_mfe / mae / efficiency / retrace_ratio / excursion_balance | 5 |
| `jump_robust.py` | intraday_medrv / intraday_minrv / intraday_jump_test_stat | 3 |
| `intraday_vol_ext.py` | intraday_volatility_time_centroid / concentration / entropy / realized_semivariance_balance / rv_signature_curvature | 5 |
| `rough_vol.py` | ts_vol_pvariation_roughness / ts_vol_scaling_break | 2 |
| `binned_response.py` | ts_binned_response_monotonicity / curvature / ts_response_slope_asymmetry | 3 |
| `dependence_ext.py` | ts_chatterjee_xi / ts_hsic / ts_conditional_mutual_information / ts_partial_distance_correlation | 4 |
| `vector_path.py` | ts_vector_path_efficiency / turning_coherence / path_curvature / self_intersection_rate | 4 |
| `event_interval.py` | event_interval_memory / event_local_variation / event_fano_factor | 3 |
| `complexity_ext.py` | ts_lempel_ziv_complexity / ts_forbidden_ordinal_pattern_ratio | 2 |
| `spectral.py` | ts_spectral_centroid / flatness / peak_concentration / quality_factor | 4 |
| `hankel.py` | ts_hankel_effective_rank / singular_gap / ssa_reconstruction_residual | 3 |
| `multifractal.py` | ts_generalized_hurst_exponent / multifractal_width / multifractal_curvature | 3 |
| `memory_ext.py` | ts_autocorrelation_time / ts_fractional_difference | 2 |
| `moments_ext.py` | ts_l_skewness / ts_l_kurtosis / ts_hartigan_dip | 3 |
| `intrinsic_dimension.py` | ts_delay_intrinsic_dimension | 1 |
| `topology_ext.py` | ts_persistence_entropy | 1 |
| `cross_section_ext.py` | cs_knn_local_moran / cs_isotonic_residual / group_tail_coexceedance_density / group_corr_mst_length | 4 |
| `intraday_session.py` | intraday_session_shape_novelty / intraday_profile_pca_residual | 2 |

## 二、执行顺序

1. **参考模块**：interval_geometry.py（含 stateful nesting_depth 等最复杂内核）→ 验证模式。
2. **并行落地**其余 25 个模块（每个子代理只写自己负责的新文件，不碰共享文件）。
3. **中心注册**（本人）：`_LOAD_MODULES` 追加 + `operator_surface.py` daily pack。
4. **集成测试**：load_all 成功、79 算子全部 daily 分类、daily DSL 可解析、合成面板
   pandas+polars 双后端执行与数值校验、PIT/NaN 契约。
5. **SQL 下推**（尽力）：纯表达式子集追加 emitter 分支。
6. 报告。

## 三、数值契约（与既有 advanced 系列一致）

- 全部 trailing window、prefix-causal、确定性；空窗/全 NaN → NaN，禁止 Inf/伪造零。
- 无效参数（window<2、bins<2、k<=0 等）raise ValueError。
- 跨字段输入同面板对齐校验由 `Operator._prepare_call` 自动完成（index/columns 必须一致）。
