# -*- coding: utf-8 -*-
"""重复算子合并 + LQTP / 业界标准命名。

本模块在 ``load_all()`` 末尾调用 ``apply_operator_deduplication()``，
将历史方言名重定向到 canonical 并注销冗余主键。

常量
----
- ``CANONICAL_RENAMES``：旧 canonical → 新 canonical 重命名表；
- ``DEDUPE_ALIASES``：别名 → 保留 canonical 映射；
- ``REMOVED_CANONICALS``：去重后注销的冗余 canonical 列表；
- ``SURFACE_RETRACTIONS``：降级为别名的名字，必须同步退出
  ``operator_surface`` 的静态面登记（否则 layer governance 的
  fail-closed partition gate 会让 ``load_all()`` 直接失败）。

命名原则
--------
时序算子保留 ``ts_*`` canonical，元素算子保留现有生产契约中的名字。
较短的 LQTP 名称作为 DSL 别名，不反向改写 canonical。这样 registry、
policy、SQL/Polars evidence 和持久化 manifest 始终引用同一个稳定主键。
"""
from __future__ import annotations

from factor_engine.cleaned_operators.registry import OperatorRegistry

CANONICAL_RENAMES: dict[str, str] = {
    # 时序滚动：历史 m_* → ts_*
    "m_var": "ts_var",
    "m_median": "ts_median",
    "m_mad": "ts_mad",
    "m_beta": "ts_beta",
    "m_top_n_avg": "ts_top_n_avg",
    "m_top_n_std": "ts_top_n_std",
    "m_bottom_n_avg": "ts_bottom_n_avg",
    "m_bottom_n_sum": "ts_bottom_n_sum",
    "ratios": "ts_ratio",
    # 回归算子统一输出名称
    "ts_regression": "ts_regression_slope",
    # 分组 / 扩展
    "group_demean": "group_neutralize",
    "cum_standardize": "expanding_zscore",
    # --- R67：命名不同但实现逐位一致的重复 canonical ---------------------------
    # 判据（全部已取证）：
    #   a) 合成面板 5 输入变体（base/alt/signed/nan/step）× pandas_numpy + polars
    #      逐位一致 / NaN 掩码相同（max_abs_delta == 0.0）；
    #   b) 富契约 _contract_conflicts(old_cat, new_cat) == 0（param_specs /
    #      units / grains / availability / semantic_version 等全部相等或单侧未声明）；
    #   c) 两侧 runtime backend 集合完全相同；
    #   d) 无任何 alias 指向旧名，旧名也不占冲突别名位；
    #   e) 准入/治理面不变：should_fail_closed() 与
    #      build_operator_spec().allow_in_production 两侧相同。
    # 旧名保留为别名（DEDUPE_ALIASES 再次声明，保证导入期部分加载也 fail-closed），
    # 注册表主键只剩规范名；实现对象由规范名持有，数值行为不变。
    #
    # state 族：category_* 是同一 estimator 的 legacy 拼写（state_* 已是族内规范
    # 前缀；state_episode_duration 早已是 state_age 的 alias）。
    "category_age": "state_episode_age",
    "category_age_lower_bound": "state_episode_age_lower_bound",
    "category_transition_rate": "state_transition_rate",
    # R67 复核：下列 7 对同样满足 a)–d)，但**本轮未落地**，原因分两类（取证见
    # /tmp/r67 的 ab_delta.txt / admission_*.txt / verify_after_merge.json）：
    #   ① 会放宽 fail-closed 准入门禁：ts_ar_forecast / ts_ar_innovation。
    #      should_fail_closed() = is_intentionally_experimental(name) or
    #      is_isolated_from_default_mining(name)，纯名字键、不解析别名；合并后
    #      旧名不再落在这两个集合里 -> should_fail_closed(旧名) 由 True 变 False
    #      （规范名仍为 True）。需先让该判定别名感知或保留旧名闭门，再合并。
    #   ② 数值与准入都一致，但会把当前绿的测试改红（这些测试把旧名钉成 canonical
    #      主键 / 扩展面成员，属本次改动的同步项，不在本文件写权限内）：
    #        holder_weighted_churn  -> holder_id_matched_churn
    #        holder_entry_share    -> holder_id_matched_entry_share
    #        holder_exit_share     -> holder_id_matched_exit_share
    #        fiscal_direction_consistency   -> fiscal_sign_consistency
    #        fiscal_pair_direction_agreement -> fiscal_sign_agreement
    #      补上测试同步后，直接把这几行移入本表 + DEDUPE_ALIASES +
    #      SURFACE_RETRACTIONS 即可。
}

# 别名 → 保留的 canonical
DEDUPE_ALIASES: dict[str, str] = {
    # --- 时序滚动 ---
    "SMA": "ts_mean",
    "move": "ts_mean",
    "window_mean": "ts_mean",
    "running_mean": "ts_mean",
    "ma": "ts_mean",
    "window_sum": "ts_sum",
    "sum_n": "ts_sum",
    "window_max": "ts_max",
    "window_min": "ts_min",
    "window_std": "ts_std",
    "std_n": "ts_std",
    "deltas": "ts_delta",
    "delta": "ts_delta",
    "ts_decay": "ts_decay_linear",
    "Sum": "ts_sum",
    "Kurt": "ts_kurt",
    "Skew": "ts_skew",
    "Corr": "ts_corr",
    "Cov": "ts_cov",
    "Covariance": "ts_cov",
    "m_zscore": "ts_zscore",
    "returns": "ts_pct",
    "m_pct_change": "ts_pct",
    "pct_change": "ts_pct",
    "m_var": "ts_var",
    "m_median": "ts_median",
    "m_mad": "ts_mad",
    "m_beta": "ts_beta",
    "m_top_n_avg": "ts_top_n_avg",
    "m_top_n_std": "ts_top_n_std",
    "m_bottom_n_avg": "ts_bottom_n_avg",
    "m_bottom_n_sum": "ts_bottom_n_sum",
    "m_top_n_sum": "ts_topk_sum",
    "Percentile": "ts_quantile",
    "percentile": "ts_quantile",
    "running_std": "ts_std",
    "running_sum": "ts_sum",
    "cum_standardize": "expanding_zscore",
    "DECAY_LINEAR": "ts_decay_linear",
    "TS_DECAY_LINEAR": "ts_decay_linear",
    "decay_linear": "ts_decay_linear",
    "EMA": "ts_ema",
    "ema": "ts_ema",
    "ewm_mean": "ts_ema",
    "ratios": "ts_ratio",
    # --- 回归统计量 ---
    "Var": "ts_var",
    # NEW-037: an alias is only legal when the maths is EQUIVALENT.  ``Mad`` /
    # ``mad`` are the single-centre mean-absolute-deviation (``mean(|x-mean|)``);
    # ``ts_mad`` is a DIFFERENT legacy double-rolling estimator (rolling median
    # -> abs -> rolling mean).  Aliasing Mad->ts_mad silently redefined the
    # statistic, so the alias now points at the mathematically equal
    # ``ts_mean_abs_deviation``.
    "Mad": "ts_mean_abs_deviation",
    "Median": "ts_median",
    "Beta": "beta",
    "Intercept": "intercept",
    "Residual": "residual",
    "R2": "r_squared",
    "var": "ts_var",
    "mad": "ts_mean_abs_deviation",
    "median": "ts_median",
    "corr": "ts_corr",
    "cov": "ts_cov",
    "kurt": "ts_kurt",
    "skew": "ts_skew",
    "sum": "ts_sum",
    "ts_regression": "ts_regression_slope",
    "TS_REGRESSION_SLOPE": "ts_regression_slope",
    # --- 元素 / 信号 ---
    "negate": "neg",
    "inv": "inverse",
    "reciprocal": "inverse",
    "if_else": "where",
    "fmax": "maximum",
    "fmin": "minimum",
    "sqr": "square",
    "cumulative_max": "expanding_max",
    "cumulative_mean": "expanding_mean",
    "cumulative_min": "expanding_min",
    "clamp": "clip",
    "cap": "clip",
    "CLIP": "clip",
    "dft": "fft",
    "idft": "ifft",
    "safe_div": "safe_div_null",
    # --- delay ---
    "DELAY": "ts_delay",
    "Delay": "ts_delay",
    "Ref": "ts_delay",
    "delay": "ts_delay",
    "m_delay": "ts_delay",
    "shift": "ts_delay",
    # --- 截面 / 分组 ---
    "standardize": "zscore",
    "panel_rank": "rank",
    "panel_zscore": "zscore",
    "panel_standardize": "zscore",
    "industry_neutralize": "group_neutralize",
    "industry_neutral": "group_neutralize",
    "ind_neutralize": "group_neutralize",
    "neutralize": "group_neutralize",
    "c_neutralize": "group_neutralize",
    "panel_neutralize": "group_neutralize",
    "group_demean": "group_neutralize",
    "NEUTRALIZE": "group_neutralize",
    "INDUSTRY_NEUTRAL": "group_neutralize",
    "INDUSTRY_NEUTRALIZE": "group_neutralize",
    "IND_NEUTRALIZE": "group_neutralize",
    "market_cap_neutralize": "size_neutralize",
    "cap_neutralize": "size_neutralize",
    "size_industry_neutralize": "industry_size_neutralize",
    # --- 累计 / 扩展 ---
    "mean_agg": "avg",
    "cum_avg": "expanding_mean",
    "cum_rank": "expanding_rank",
    "cum_std": "expanding_std",
    "cum_last": "ffill",
    "last": "ffill",
    "last_not_null": "ffill",
    "first": "first_not_null",
    # --- 时序别名（已有） ---
    "Mean": "ts_mean",
    "TS_MEAN": "ts_mean",
    "m_avg": "ts_mean",
    "mean": "ts_mean",
    "Max": "ts_max",
    "TS_MAX": "ts_max",
    "m_max": "ts_max",
    "Min": "ts_min",
    "TS_MIN": "ts_min",
    "m_min": "ts_min",
    "Std": "ts_std",
    "TS_STD": "ts_std",
    "m_std": "ts_std",
    "std": "ts_std",
    "TS_SUM": "ts_sum",
    "m_sum": "ts_sum",
    "Delta": "ts_delta",
    "Diff": "ts_delta",
    "TS_DELTA": "ts_delta",
    "TS_PCT": "ts_pct",
    "TS_KURT": "ts_kurt",
    "TS_SKEW": "ts_skew",
    "TS_CORR": "ts_corr",
    "TS_COV": "ts_cov",
    "m_percentile": "ts_quantile",
    "m_argmax": "ts_argmax",
    "m_argmin": "ts_argmin",
    "TS_TOPK_SUM": "ts_topk_sum",
    # --- R67：上述重复 canonical 降级为别名（与 CANONICAL_RENAMES 同源） ------
    # rename_canonical 已经登记这些边；这里再次声明是为了在"只加载了部分模块"
    # 的导入期场景下仍能 fail-closed 地保证旧名可解析到规范实现，且旧名永不
    # 重新占回 canonical 主键。
    "category_age": "state_episode_age",
    "category_age_lower_bound": "state_episode_age_lower_bound",
    "category_transition_rate": "state_transition_rate",
}

# 注销的冗余 canonical（实现类仍可在源码中，但不再占主键）
REMOVED_CANONICALS: tuple[str, ...] = (
    # Removed from the factor runtime: future-looking, misleading, random,
    # non-shape-preserving, or full-sample operators.
    "Lead",
    "next",
    "bfill",
    "causal_bfill",
    "fillna_interpolate",
    "interpolate",
    "norm",
    "norm_l1",
    "norm_linf",
    "rand_exp",
    "rand_lognormal",
    "rand_normal",
    "rand_poisson",
    "rand_uniform",
    "sample",
    "shuffle",
    "SMA",
    "move",
    "window_mean",
    "running_mean",
    "window_sum",
    "window_max",
    "window_min",
    "window_std",
    "deltas",
    "ts_decay",
    "Sum",
    "Kurt",
    "Skew",
    "Corr",
    "Cov",
    "Covariance",
    "Var",
    "Mad",
    "Median",
    "m_zscore",
    "Beta",
    "Intercept",
    "Residual",
    "R2",
    "negate",
    "inv",
    "reciprocal",
    "fmax",
    "fmin",
    "sqr",
    "cumulative_max",
    "cumulative_mean",
    "cumulative_min",
    "if_else",
    "clamp",
    "dft",
    "idft",
    "standardize",
    "panel_rank",
    "panel_zscore",
    "panel_standardize",
    "industry_neutralize",
    "neutralize",
    "panel_neutralize",
    "mean_agg",
    "cum_avg",
    "cum_rank",
    "cum_std",
    "cum_last",
    "last",
    "last_not_null",
    "first",
    "returns",
    "Percentile",
    "running_std",
    "running_sum",
    # LQTP 对齐：与 ema 重复的独立实现
    "ewm_mean",
)

# 静态算子面撤回表：这些名字在上面的 CANONICAL_RENAMES 里被降级为别名，
# 因此必须同时从 ``operator_surface`` 的静态面登记中撤下。
#
# 为什么必须在 _dedupe.py 里做：``layer_governance.finalize_layer_governance()``
# 在 `apply_operator_deduplication()` 之后运行，并对六个静态分区做 fail-closed
# 校验 —— ``missing = {label: sorted(names - set(OperatorRegistry._operators))}``，
# 只要某个分区仍登记着一个已不是活跃 canonical 的名字，``load_all()`` 就直接抛
# ``static operator surface contains inactive canonicals``。也就是说
# "从 canonical 面移除" 在语义上就包含 "从枚举 canonical 面的静态面清单里移除"。
#
# 手法与仓库既有先例一致（``cleaned_operators/direction_concentration.py`` 的
# ``_r11_round3_split_and_surfaces()`` 把 ``ts_abs_concentration`` 从 daily 面
# 降级）：``REVIEWED_MIGRATION_MANIFEST``（daily 面的真源）与 legacy
# ``DAILY_FACTOR_MIGRATED`` frozenset **必须同步收缩**，否则
# ``test_r7_surface_orthogonal`` 的 ``len(manifest) == len(DAILY_FACTOR_MIGRATED)``
# 表面正交不变量会被破坏；extended 面走模块自带的 ``retract_extended_only()``。
#
# 注：本表只覆盖 daily-migrated + extended 两个分区（这 10 个名字的实际归属，
# 见 /tmp/r67/probe_surface.json）。若将来自某个名字还出现在 DAILY_CANONICALS /
# RESEARCH_ONLY / UNSAFE / LEGACY_ONLY / INTERNAL_ONLY 里，_retract_from_static_surface
# 需要相应扩展并会以 ``剩留面登记`` 的 RuntimeError 明确报警（见下）。
SURFACE_RETRACTIONS: tuple[str, ...] = (
    "category_age",
    "category_age_lower_bound",
    "category_transition_rate",
)

# finalize_layer_governance() 校验的六个静态分区（顺序无关）。
_SURFACE_PARTITIONS: tuple[str, ...] = (
    "DAILY_CANONICALS",
    "EXTENDED_ONLY_CANONICALS",
    "RESEARCH_ONLY_CANONICALS",
    "UNSAFE_CANONICALS",
    "LEGACY_ONLY_CANONICALS",
    "INTERNAL_ONLY_CANONICALS",
)


def _retract_from_static_surface(names: tuple[str, ...]) -> None:
    """把已降级为别名的 canonical 从 operator_surface 的静态面登记中撤下。

    参数:
        names: 已不再占 canonical 主键的名字。

    返回:
        None

    异常:
        RuntimeError: 某个名字仍留存在脚本未覆盖的分区登记里（fail-closed，
            避免 load_all 在后面以更难定位的方式失败）。
    """
    if not names:
        return
    from factor_engine.cleaned_operators import operator_surface as _surface

    targets = frozenset(names)
    # daily 面：review manifest 是 daily 面的真源（daily_factor_migrated() 直接
    # 读它），legacy DAILY_FACTOR_MIGRATED frozenset 必须同步收缩。
    for name in targets:
        _surface.REVIEWED_MIGRATION_MANIFEST.pop(name, None)
    _surface.DAILY_FACTOR_MIGRATED = frozenset(
        c for c in _surface.DAILY_FACTOR_MIGRATED if c not in targets
    )
    # extended / research 面：模块自带的 live mutator。
    _surface.retract_extended_only(targets)
    _surface.retract_research_only(targets)
    # classify_canonical() 走 _DAILY_SURFACE_CACHE 热路径，必须重建，否则该名字
    # 会被静态地判成 daily（而不是回落到别名解析到规范名）。
    _surface._rebuild_daily_surface_cache()

    remaining = {
        attr: sorted(targets & set(getattr(_surface, attr) or ()))
        for attr in _SURFACE_PARTITIONS
        if targets & set(getattr(_surface, attr) or ())
    }
    if remaining:
        raise RuntimeError(
            "SURFACE_RETRACTIONS left names on static surface partitions: "
            f"{remaining}.  Extend _retract_from_static_surface() for those "
            "partitions before landing the dedupe entry."
        )


_DEDUPE_APPLIED = False


def apply_operator_deduplication() -> None:
    """在全部算子注册与基础别名加载后执行去重合并。

    依次执行：canonical 重命名 → 冗余 canonical 注销 → 别名登记 →
    静态算子面撤回（``SURFACE_RETRACTIONS``）。
    幂等设计，重复调用仅第一次生效。

    返回:
        None

    异常:
        RuntimeError: 去重目标 canonical 无 runtime 实现时，或静态面撤回后
            仍有名字留在未覆盖的分区登记里。
    """
    global _DEDUPE_APPLIED
    if _DEDUPE_APPLIED:
        return
    # P0-B1: flush deferred in-window alias edges (e.g. polars_statistics ->
    # ts_mean_abs_deviation) before dedupe validates every alias target.
    OperatorRegistry.publish_pending_aliases()
    for old, new in CANONICAL_RENAMES.items():
        OperatorRegistry.rename_canonical(old, new)

    # Remove retired canonical spellings before registering them as aliases;
    # alias/canonical collisions are always errors in the Registry.
    for name in REMOVED_CANONICALS:
        OperatorRegistry.unregister(name)

    for alias, canonical in DEDUPE_ALIASES.items():
        if alias == canonical:
            raise RuntimeError(
                f"dedupe self-loop: {alias!r} aliases itself (R16-075 — a "
                "canonical->canonical edge has no value and pollutes the alias graph)"
            )
        if canonical not in OperatorRegistry._operators:
            raise RuntimeError(f"dedupe target missing: {canonical!r} (alias {alias!r})")
        OperatorRegistry.register_alias(alias, canonical)

    # 最后一步：把降级为别名的名字从静态算子面登记中撤下。layer_governance 的
    # fail-closed partition gate 在本次调用之后运行，见 SURFACE_RETRACTIONS 注释。
    _retract_from_static_surface(SURFACE_RETRACTIONS)

    _DEDUPE_APPLIED = True
