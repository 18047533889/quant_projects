# FactorEngine 第四轮复审执行计划 (Review #4)

> 执行日期：2026-08-08
> 输入：外部 AI 复审单（102 项：P0 21 / P1 68 / P2 12，编号见下）
> 目标：一次性全部修复；先底层合同、再算子批次；每批自测 + 末尾全量回归。

## 0. 已由并发会话修复（不计入本轮工作量，仅验证）

- **R4-04**（递归 EWM/Wilder stateful governance）：`production_hardening.FULL_HISTORY_REPLAY_CANONICALS`
  已含 DMI_plus/DMI_minus/DX/NATR/PPO{,_signal,_hist}/PVO{,_signal,_hist}/
  Keltner{...}/TSI{,_signal}/DEMA/TEMA/ChaikinOscillator/ForceIndex；
  `ir/analyzer` 已处理 PPO/PVO/TSI/DEMA/TEMA 复合 lookback。→ 仅验证。
- **R4-82/83**（dynamic KNN k-peers / tie-inclusive）：待核查是否并发会话已动。
- **R4-20/21**（jump_robust / intraday_vol_ext 分钟压缩）：待核查是否已有 session-slot 实现。

## 1. 分批执行清单

### Batch A —— 底层 contract（✅ 已完成 2026-08-08）
| # | 文件 | 状态 |
|---|---|---|
| A1 | `cleaned_operators/base.py` | ✅ `ParamSpec`（dtype/min/max/choices/searchable/active_when/history_semantics）+ `OperatorMetadata.param_specs`；整数校验 ParamSpec→param_types→白名单三级；扩展 `_INTEGER_PARAM_NAMES`（confirmation/tolerance/horizon/min_anchors/min_tail_count/min_bin_count/cutoff/block/tau…）；EnumSpec choices 校验 |
| A2 | `cleaned_operators/base.py` | ✅ 广播拆分：`_TYPED_BROADCAST_TAGS`（daily_to_minute/scalar_to_cs/same_trading_date/session_boundary）+ 仪器集 subset 校验 |
| A3 | `cleaned_operators/base.py` | ✅ `validate_operator_call()` 中央校验；`_prepare_call` 委托它；绕过模块（daily_panel/weighted_moment_ext/activity_clock/spectral_ext）已接入 |
| A4 | `cleaned_operators/registry.py` | ✅ 非 declared-source 替换必须 expected_old_source + replacement_reason + semantic_version；overwrite log 记录 semantic_version |
| A5 | `cleaned_operators/registration_audit.py` | ✅ 全 canonical 位置 arity 硬门（pandas 参考；后端 arity < pandas → raise）。修复 KAMA（composite_fastpath+polars_signal）、group_winsorize（polars_native a 对齐）、holder_concentration_slope/index_event_decay/event_historical_response_*（polars arity + missing_policy/require_full_horizon 贯穿） |
| A6 | `ir/analyzer.py` | ✅ `register_history_requirement(canonical, fn)` + 实现方法优先；复合 lookback 加法（volume_autocorr=w-1+lag、event-response=hw-1+horizon）；结构 level 加 confirmation |
| A7 | `production_hardening.py` | ✅ threshold_cycle / state_episode_* / interval_nesting_depth 入 FULL_HISTORY_REPLAY + STATEFUL_CHECKPOINTS |

### Batch B —— P0 数学/语义错误
| 文件 | 项 |
|---|---|
| `sequence_complexity.py` | R4-08（permutation transition 相邻判断 `start_next==start+1`）、R4-64（trailing contiguous suffix）、R4-65（tie kernel 选择） |
| `ts_model/wavelet_spectral.py` | R4-09（统一 WaveletKernel：不删 NaN、x[-m:]、0·log0、dyadic scale、ddof 一致） |
| `dependence_ext.py` | R4-14（HSIC 中心化分母）、R4-15（partial dCor 重写→research_only）、R4-49（Chatterjee tie-aware）、R4-50（CMI 样本门槛） |
| `complexity_ext.py` | R4-17（LZ /ln(bins)）、R4-62（contiguous suffix）、R4-63（forbidden ordinal 期望修正）、R4-64 |
| `candle_pattern_engine_v2.py` | R4-10（3_inside/3_outside 重写）、R4-11（warmup→NaN 非 0）、R4-12（sign(close-open) 允许 0 / EventBool 与 direction 分离）、R4-91（active_when）、R4-92（semantic_family） |
| `dynamic_knn.py` | R4-13（average-tie rank）、R4-82（k-peer 一致性）、R4-83（tie-inclusive 确定性） |
| `structural_levels.py` | R4-18（pivot 严格峰谷交替状态机）、R4-19（strength 距离衰减方向）、R4-89（history+confirmation）、R4-90（density 归一） |
| `report_timing.py` | R4-16（filing-delay-surprise 事件时钟）、R4-84（revision magnitude ReportPeriod 一致性） |

### Batch C —— stateful
| 文件 | 项 |
|---|---|
| `technical/indicators_v2.py` | R4-04 验证 + 剩余递归算子入 governance |
| `price_volume/liquidity_v2.py` | R4-04、R4-29（up/down volume ratio、signed volume unknown→NaN）、R4-96 |
| `threshold_cycle.py` | R4-05（FULL_HISTORY_REPLAY/checkpoint、missing→censor） |
| `state_episode_excursion.py` | R4-06（checkpoint: direction/entry/mfe/mae/path） |
| `interval_geometry.py` | R4-07（nesting depth stateful + gap→NaN） |

### Batch D —— missing / time-axis
| 文件 | 项 |
|---|---|
| `jump_robust.py` | R4-20（保留 minute slot，缺失→fail session） |
| `intraday_vol_ext.py` | R4-21（session-slot-aware 时间轴） |
| `intraday_activity_duration.py` | R4-29/74（missing 不压缩、unknown→NaN） |
| `vector_path.py` | R4-51（gap 使窗口 invalid）、R4-52（curvature 标准化坐标）、R4-53（denominator 排除相邻段） |
| `rough_vol.py` | R4-59（保留原时间轴）、R4-60（整数校验）、R4-61（改名 pvariation_scaling_exponent） |
| `memory_ext.py` | R4-56（integrated autocorrelation time 归一）、R4-57（contiguous cohort）、R4-64 |
| `event_interval.py` | R4-54（EventBool 0/1/NaN）、R4-55（fano_factor full-block）、R4-96 |
| `common/daily_panel.py` | R4-02（validate_operator_call）、R4-22（period_lag legacy→alias） |

### Batch E —— fundamental
| 文件 | 项 |
|---|---|
| `fundamental/transforms_v2.py` | R4-23（consecutive vs last-N-visible 分两类）、R4-24（x_axis=fiscal_ordinal）、R4-25（prior-history vs inclusive）、R4-28（fin_mad→fin_mean_abs_deviation + 真 MAD） |
| `fundamental/transforms_repairs_v2.py` | R4-26（missing→NaN）、R4-27（days_since_update observed clock） |
| `fundamental/expectation_v2.py` | R4-26/27/28 按适用 |
| `fiscal_strict.py` | R4-22/23/24/84 |

### Batch F —— typed / CS
| 文件 | 项 |
|---|---|
| `extrema_divergence.py` | R4-45（ConfirmedExtremaKernel 峰谷交替）、R4-46（one-to-one monotonic 匹配 q1<q2） |
| `binned_response.py` | R4-47（min_per_bin≥3/5、N≥bins·min）、R4-93（bins 小网格 {3,5}） |
| response slope（`ts_response_slope_asymmetry` 所在文件） | R4-48（标准 β·σx/σy，不除 residual RMSE） |
| `first_passage.py` | R4-85（unit contract）、R4-86（barrier≤0→ValueError） |
| `extreme_tail.py` | R4-87（lower-tail 定义 threshold−x）、R4-88（missing→censor） |
| `hankel.py` | R4-66（min_contiguous_fraction≥0.8）、R4-67（文档同步）、R4-68（int 校验）、R4-95 |
| `cross_section_local.py` | R4-13/65（复用 AverageTieRank）、R4-82/83 |

### Batch G —— 元数据 / governance / P2
| 文件 | 项 |
|---|---|
| `activity_clock.py` | R4-02、R4-69（scale_window<5 出 grammar）、R4-70（metadata unit 修正）、R4-71（equal-peak 取最近 peak） |
| `spectral_ext.py` | R4-02、R4-95 |
| `weighted_moment_ext.py` | R4-02 |
| multifractal 模块 | R4-94（满 window 才输出）、R4-95 |
| `_aliases.py` | R4-22（legacy alias 化）、R4-101 |
| metadata/catalog | R4-95（window_semantics）、R4-97（EnumSpec）、R4-98（input_units 强制）、R4-102（catalog backend set == runtime backend set） |

## 2. 全局统一原则（所有批次遵守）
1. **绝不 dropna 后重连时间轴**——保持原 time/session slot。
2. **unknown ≠ zero / no-event**——只有确定"无事件"才能输出 0。
3. **session slot 不可压缩**；缺失分钟 → fail/censor，不是相邻。
4. **int 校验集中**：所有整数参数走 ParamSpec/`_normalise_integer`，杜绝 `int(5.9)→5`。
5. **stateful 显式声明**：递归算子必须 checkpoint_contract 或 FULL_HISTORY_REPLAY。
6. **event 语义统一** EventBool{0,1,NaN} / StateSigned{-1,0,+1,NaN} / Intensity。
7. **additive 编辑**：不动证据 JSON；不重生成 recipe evidence；对共享文件最小 append。
8. 每批完成后跑该文件对应测试，末尾全量回归。

## 3. 验收
- 全量测试通过（除并发会话持有的 recipe evidence / data_access stale / six-gate 认证不一致）。
- `docs/operator_market_capabilities.json` / `docs/market_field_manifest.json` 重新生成，CI audit 通过。
- 计划文档本表逐项打勾。

## 4. 执行进度追踪

### Batch A 底层 contract —— ✅ 完成（详见 §1 Batch A 状态表）

### Batch B P0 数学 —— 已派 subagent（B）
| 文件 | 项 | 状态 |
|---|---|---|
| sequence_complexity.py | R4-08/64/65 | 进行中 |
| ts_model/wavelet_spectral.py | R4-09 | 进行中 |
| dependence_ext.py | R4-14/15/49/50 | 进行中 |
| complexity_ext.py | R4-17/62/63/64 | 进行中 |
| candle_pattern_engine_v2.py | R4-10/11/12/91/92 | 进行中 |
| dynamic_knn.py | R4-13/82/83 | 进行中 |
| structural_levels.py | R4-18/19/89/90 | 进行中 |
| report_timing.py | R4-16/84 | 进行中 |

### Batch C stateful —— 已派 subagent（C）
| 文件 | 项 | 状态 |
|---|---|---|
| technical/indicators_v2.py | R4-04 验证 | 进行中 |
| price_volume/liquidity_v2.py | R4-29/96 + R4-04 验证 | 进行中 |
| threshold_cycle.py | R4-05 | 进行中 |
| state_episode_excursion.py | R4-06 | 进行中 |
| interval_geometry.py | R4-07 | 进行中 |

### Batch D missing/time-axis —— 已派 subagent（D）
| 文件 | 项 | 状态 |
|---|---|---|
| jump_robust.py | R4-20 | 进行中 |
| intraday_vol_ext.py | R4-21 | 进行中 |
| intraday_activity_duration.py | R4-29/74 | 进行中 |
| vector_path.py | R4-51/52/53 | 进行中 |
| rough_vol.py | R4-59/60/61 | 进行中 |
| memory_ext.py | R4-56/57 | 进行中 |
| event_interval.py | R4-54/55/96 | 进行中 |
| common/daily_panel.py | R4-02 验证 + R4-22 | 进行中 |

### Batch E fundamental —— 已派 subagent（E）
| 文件 | 项 | 状态 |
|---|---|---|
| fundamental/transforms_v2.py | R4-23/24/25/28 | 进行中 |
| fundamental/transforms_repairs_v2.py | R4-26/27 | 进行中 |
| fundamental/expectation_v2.py | R4-26/27/28 | 进行中 |
| fiscal_strict.py | R4-22/23/24/84 验证 | 进行中 |

### Batch F typed/CS —— 已派 subagent（F）
| 文件 | 项 | 状态 |
|---|---|---|
| extrema_divergence.py | R4-45/46 | 进行中 |
| binned_response.py | R4-47/93 | 进行中 |
| response-slope 模块 | R4-48 | 进行中 |
| first_passage.py | R4-85/86 | 进行中 |
| extreme_tail.py | R4-87/88 | 进行中 |
| hankel.py | R4-66/67/68/95 | 进行中 |
| cross_section_local.py | R4-13/65/82/83 | 进行中 |

### Batch G metadata/governance —— 已派 subagent（G）
| 文件 | 项 | 状态 |
|---|---|---|
| activity_clock.py | R4-02 验证/69/70/71 | 进行中 |
| spectral_ext.py | R4-02 验证 + R4-95 | 进行中 |
| weighted_moment_ext.py | R4-02 验证 | 进行中 |
| multifractal 模块 | R4-94/95 | 进行中 |
| _aliases.py | R4-22/101 | 进行中 |
| metadata/catalog | R4-95/97/98/102 | 进行中 |
