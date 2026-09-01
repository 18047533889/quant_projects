# factor_engine 勘察进度存档（只读勘察，无副作用）
# 2026-09-01 文档勘察员任务进度

## 已完成勘察
1. 模块地图：api/expr/ir/planner/planning/backend/cleaned_operators/storage/runtime/service/mining/identity/security/execution/semantic/telemetry/fields/market/factor_recipes/export/research_operators/research_tools/validation 全部存在并有内容。
2. DSL API：读了 api/__init__.py —— 显式导出 col/field/rank/ts_mean/ts_std/ts_std_dev/zscore/delay/make_cleaned_call_factory/Factor；其余算子经 __getattr__ 按 build_dsl_allowlist() 懒加载。
3. 后端矩阵：读了 backend/factory.py（auto/pandas/pandas_modin/polars/polars_long/hybrid_long/polars_lazy/duckdb_sql/clickhouse_sql/debug/q_kdb）。
4. 算子全景：cleaned_operators 447 个 .py；operators_catalog.json canonical=1737，daily=1238，extended=480，research=8，unsafe=7，legacy=1，internal=3；136 个有别名；scope ts=871/elementwise=503/fundamental_period=136/cs=96/group=59/session_intraday=70。
5. 文档现状：dsl_operators_reference.md 仅 31 行（分类枚举，无逐算子公式）；算子全览.md 3806 行/259 个算子节，76 个显式 OpDoc 公式条目，14 个"以实现为准"泛化条目；operator_core_specs.yaml 86 条×24 字段。
6. 运行时：FactorEngine.run/run_many/materialize/materialize_many 签名已读；factor_lake 流程 staging→publish 带审批门；examples/profiles/prod.yaml 磁盘上不存在（文档引用但缺失）。
7. 服务层：15 个路由（health/livez/readyz/metrics/operators/validate-spec/research/compute/production/compute/production/materialize/jobs/compute/jobs/materialize/jobs/{run_id}...），端口 8088。
8. 其他功能：mining（MiningRole/DirectUseStatus）、identity（SHA-256 公式身份）、security（派生数据权限继承+factor_id 校验）、execution（仅 memory_manager）、semantic（DataKnowledgeIdentity）、telemetry（opt-in metrics/traces/health）均真实存在。
9. 测试规模：静态统计 1145 个测试文件 / 14408 个测试函数；实际采集 runtime 665 / api 16 / ir 19 / backend 1080(+17 collection errors, 缺 polars)；全量采集超时（SIGTERM）。
10. 版本：pyproject 0.3.1；changelog_shw 最新为 2026-08-10 R25（DataAccess 第38版）/ FE 侧最新 2026-08-09 R14（第36版）。

## 关键缺口（供重写 README/算子深度文档参考）
- examples/profiles/prod.yaml 不存在（README/完全指南/runtime/config.py `_PROFILES_DIR` 引用 examples/profiles，但 git 里 examples/ 零文件被跟踪；`load_profile("prod")` 会 FileNotFoundError）。README 引用的 examples/simple_factor.py、examples/config_driven_factor.yaml 同样缺失。
- 算子全览.md 的 DSL 白名单 469 名/256 规范算子（2026-07-17 生成）已严重过期 vs 现 catalog 1737 canonical。
- 仅 76 个算子有显式公式语义（operator_doc_semantics.py OpDoc），1737 中 1478+ 个没有逐算子公式说明。
- backend_coverage.md 诚实披露：polars/duckdb parity_verified=0，production_safe=0；BACKEND_COVERAGE.md production_admitted=0。
- docs/README.md 引用 examples/README.md、scripts/README.md，但 examples/ 目录整体缺失。
- 算子深度文档核心源文件：cleaned_operators/docs/operator_doc_semantics.py（_EXPLICIT 76 条 OpDoc）+ _operator_doc_batch.py（0 条）。
- daily DSL 中许多 WQ 习惯名以别名形式存在：SMA→ts_mean、EMA→ts_ema、neutralize→group_neutralize、ts_regression→ts_regression_slope、if_else→where、returns→ts_pct、safe_div→safe_div_null；RSI/OBV/ROC/TRIX/CCI/STOCH/pe_ttm 无 canonical 亦无别名（需自查实现/文档缺口）。

## 最终代表性算子清单（127 个，全部经 catalog 验证存在）
截面(13): rank zscore normalize winsorize cs_demean cs_mean cs_std cs_regression cs_resid cs_quantile cs_mad rank_corr group_neutralize
时序(26): ts_mean ts_std ts_rank ts_corr ts_delta ts_sum ts_min ts_max ts_median ts_skew ts_kurt ts_decay_linear ts_decay_exp_window ts_argmax ts_argmin ts_quantile ts_autocorr ts_beta ts_max_drawdown ts_topk_sum ts_ema ts_pct ts_days_since ts_true_streak ts_count_if ts_mean_if
分组(7): group_rank group_zscore group_mean group_std group_sum group_neutralize group_quantile_spread
均线(8): ts_mean ts_ema WMA DEMA TEMA HMA KAMA ALMA
技术/振荡(20): RSI_WILDER MACD_line MACD_signal MACD_hist ATR_WILDER CMF CMO ForceIndex MFI PPO PSAR Supertrend TSI UltimateOscillator ADX DMI_plus KeltnerUpper bollinger_pct_b bollinger_width rolling_obv
事件/条件(8): trade_when where event_frequency event_rate_pct event_decay_asof directional_change_state limit_up_close ts_true_streak
量价/流动性(11): adv amihud_illiquidity abnormal_volume abnormal_turnover vwap_deviation vwap_distance_pct volume_momentum turnover_momentum free_float_ratio float_share_ratio price_volume_divergence
收益/波动(6): ts_pct garman_klass_vol parkinson_vol yang_zhang_vol return_volume_corr return_per_turnover
分钟级(5): intraday_volatility intraday_vwap_deviation intraday_medrv intraday_realized_semivariance_balance session_event_recovery_score
基本面/PIT(9): book_to_price altman_z_score holder_concentration fin_staleness fin_surprise_zscore yoy_by_period valuation_pe_ttm_lyr_gap ashare_limit_up_touch fin_beat_streak
基础数学/安全(14): abs log sqrt power add subtract multiply divide protected_div safe_div_null clip floor ceil sign

## 测试规模（实际采集，非全量）
- 全量 pytest --collect-only 超时(SIGTERM 143)。
- 实采：tests/api 16、tests/ir 19、tests/runtime 665、tests/backend 1080(+17 collection error, 因缺 polars 模块)、tests/storage+integration+planner+market 793(+4 errors)。
- 静态统计：1145 个测试文件 / 14408 个测试函数（find+grep，含 stress）。
- README 声称 "1144+ tests" 与实际相符。
