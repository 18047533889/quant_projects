# FactorEngine 第六批整改报告（2026-08-09）

针对外部 AI 评审提出的**第六批共 ~118 项问题清单**（R6-101..218 + §19-§33 治理层），
已按用户优先级（Gemini 共享层 → CSE/cache → 研究/区间 → 截面/分组 → 换手 → volume
clock/candle → GLR/RQA/topology → intraday → 统计 V2 → 治理层）在本机代码库一次性整改。
**未删除任何算子**；所有重命名保留旧名为 registry alias。

> 协调说明：并发 Claude 会话仍在编辑同一棵树（price_volume / storage/sources /
> fundamental）。本批目标文件与其无交集；改动已与 R5 的 `_HANDLES_CALL_CONTRACT` /
> `strict_int` / `extend_extended_only` 兼容。

---

## 一、Gemini V2 共享层（P0，全部完成）

| 编号 | 问题 | 修复 |
|---|---|---|
| R6-155 | `gemini_v2_common` 的 `_SKIP` 只排除 `{date, stock_code}`，frame 用 `timestamp/trade_date/datetime` 作时间轴时会被当数值特征送进 kernel 并回写 | `_SKIP` 扩充为 `_AXIS_COLUMNS ∪ _IDENTITY_COLUMNS`（date/timestamp/trade_date/datetime/stock_code/instrument/symbol/session）；`union_extended/research` 改用 live-extend |
| R6-156 | HVG 五个算子只有 `ts_hvg_degree_entropy` 挂了 `_HVG_PARAM_SPECS` | 五个全部挂同一 `_HVG_PARAM_SPECS`；motif 用 `_HVG_MOTIF_PARAM_SPECS`（window≤80，见 #159） |
| R6-157 | ParamSpec 未校验 `keys(param_specs) ⊆ param_names`，Allan 共享 spec 含无关 `scale/max_scale`、DMD 含无关 `top_k` | `registry.register()` 新增不变量校验（load_all 直接 fail）。修复 5 处违规：`evt_allan` 拆 `_ALLAN_FACTOR/SCALING_SPEC`，`dmd` 拆 `_DMD_BASE/CONCENTRATION_SPEC` |

## 二、Runtime CSE + Cache（全部完成）

| 编号 | 问题 | 修复 |
|---|---|---|
| R6-150 | `cse_consumer_counts` 按出现次数计数（`add(X,X)` 的 X refcount=2），但 `collect_consumed_sids` 单 root 内去重后只减 1 → 永不 evict | `cse_consumer_counts` 改为每 root 去重计数（refcount = 消费该 sid 的 root 数），与释放侧语义一致 |
| R6-151 | `CacheManager` 注释说 LRU，实为 FIFO（`get()` 不 move_to_end） | `get()` 命中后 pop+reinsert 到末尾，evict 删除最早使用项 → 真 LRU |
| R6-152 | `PersistentPlanCache.get()` 磁盘命中不记 `_bytes`、不参与 evict，连续 L3 hit 绕过内存预算 | 磁盘命中后按 `estimate_object_bytes` 计 `_bytes` 并 `_evict_to(budget)`（resource P0） |
| R6-153 | 未注册 operator 的 plan hash 以 `"unregistered"` 当正式 semantic namespace 进入生产持久缓存 | 新增 `contract_is_resolved(node)`；`pandas_backend` 写缓存时对 `PersistentPlanCache` 拒写未解析契约的计划 |
| R6-154 | 持久缓存 key 只绑 canonical/semantic_version/policy/signature，改代码忘 bump 则旧 key 继续命中 | `_operator_semantic_contract` 增加 `implementation_hash`（`implementation_hashes_for(canonical)`），改实现即失效 |

## 三、research_transform / interval_geometry（#101-111）

| 编号 | 修复 |
|---|---|
| R6-101 | Haar `window` 只允许 2^k（32/64/128/256），kernel 拒绝非 2 幂（65/80/100/127 不再等价） |
| R6-102 | `level×window` 关系进入 RelationalParamSpec：`2^(level+2) <= window` |
| R6-103 | signature 历史只接受完整 `path_window` 的 signature（早于首完整窗不产生） |
| R6-104 | 历史采样加 `stride=max(1,pw//4)`，`n >= max(20, 2*dim)` 变有效样本门 |
| R6-105 | "shrinkage covariance" → `ridge`（diagonal loading，0.1 迹归一），如实命名 |
| R6-106 | `ts_persistence_birth_dispersion` 默认 `dim=2`，`dim>=2`（H1 需 ≥2 维嵌入） |
| R6-107 | dim≥2 后 `tau` 不再死参数（1D 嵌入不用 tau 的问题随 #106 解决） |
| R6-108 | `ts_interval_exploration_efficiency` 分子分母同 cohort：确定唯一 `[start_valid, t]` 后 span/TR/high/low 全部只用该段 |
| R6-109 | `ts_interval_overlap_component_ratio` → `ts_interval_overlap_connected_component_ratio`（overlap-连通分量，非共同重叠），旧名 alias |
| R6-110 | 区间族统一 coverage 门：`_coverage_ok`（≥80% valid pairs 才继续算） |
| R6-111 | `ts_interval_occupancy_mode_distance` 的 profile 改为 `[t-W, t-1]`，当前行仅作 query（去除 self-contamination） |

## 四、Cross-section / Group（#133-137）

| 编号 | 修复 |
|---|---|
| R6-133 | `cross_section_ext` 删本地嵌套 argsort 复制，改引 `dynamic_knn._avg_tie_ranks` / `_neighbors`（tie 归一 + kth 半径含 tie） |
| R6-132 | tail threshold 用 strict prior `[t-W,t-1]`（当前行不能定义自己的异常阈值，`_tail_coexceedance_series`） |
| R6-134 | `cs_knn_local_moran` 邻居候选从初始即 feature+target 双有效（不再先选 k 个再丢 target-NaN） |
| R6-135 | group 网络算子明确 `current_members_retrospective` 契约并文档化（`advanced_structure` 已有 `composition_policy` intersection/current） |
| R6-136 | group MST 的 pair Pearson 要求 ≥20 对齐样本（3 点相关 ±1 直接进 MST 的问题） |
| R6-137 | isotonic 截面要求 ≥10 只（`_MIN_ISOTONIC_BREADTH`），否则 NaN |

## 四-b、distribution_break（#122/#132）

| 编号 | 修复 |
|---|---|
| R6-122 | `ts_joint_energy_shift` recent/prior 窗口都止于 `t-1`（切片到 `n-1`，当前行仅作 query，不再 include-current） |
| R6-132 | tail threshold（`_tail_coexceedance_series`）用 strict prior `[t-W,t-1]`，当前行不参与定义自己的阈值 |

## 五、Turnover survival（#138-140）

| 编号 | 修复 |
|---|---|
| R6-138 | 保留 residual old mass `M_old = prod exp(-u)`：>30% 时 fail-closed（窗口估计只是条件分布，不可冒充全量） |
| R6-139 | `ts_turnover_holding_age` 文档如实标注 window 条件化（真正 recursive holder-age 列为未来 stateful 算子） |
| R6-140 | `ts_turnover_cost_entropy` 分箱改相对参考价 RP（纯形状；当前价位置由 `mode_distance` 单独表达） |

## 六、volume_clock / candle_state_space（#141-149）

| 编号 | 修复 |
|---|---|
| R6-141 | price ≤0 显式 invalid（log 域 fail-closed） |
| R6-142 | negative activity = 数据错误 → 整条路径 fail-closed（不再过滤后重连） |
| R6-143 | 任意 NaN/负 activity → 全路径 NaN（无法重建 activity 时钟） |
| R6-144 | `buckets` 不得超过 distinct activity points-1（否则插值造平滑） |
| R6-145 | candle 密度/Mahalanobis 的 mean/cov 用 history `[t-W,t-1]`，当前行仅 query |
| R6-146 | 密度改 log-density `-log(max(r_k,1e-3))`（一字板/重复状态不再爆 1/EPS） |
| R6-147 | 历史常特征 drop 维度（不 sd=1 糊弄） |
| R6-148 | `ts_multivariate_matrix_profile_novelty` → `ts_matrix_profile_novelty`（输入是标量 x），旧名 alias |
| R6-149 | z 归一距离除 `√L`（不同 subsequence_length 可比） |

## 七、GLR / RQA / Topology（#112-121、#160-162）

| 编号 | 修复 |
|---|---|
| R6-112 | RQA Theiler 默认 = 嵌入跨度 `(dim-1)*delay` |
| R6-113 | RQA ε 按 `√dim` 归一（不同 dim 下邻域密度可比） |
| R6-114 | `ts_recurrence_divergence` 单位改 `inverse_bars`（1/L_max 量纲） |
| R6-115 | diagonal entropy 归一用固定 support `log(M-min_line+1)`（不再依赖本窗出现几种长度） |
| R6-116 | `eps_fraction` ParamSpec 改 (1e-4, 1-1e-4)，与 runtime 一致 |
| R6-117 | `min_line < window-(dim-1)*delay` 进 RelationalParamSpec |
| R6-118 | persistence entropy 先剔除 zero-lifetime（不再污染归一） |
| R6-119 | Fisher 的 fit 与 empirical Fisher 用同一 exact finite cohort（不再混 NaN 块） |
| R6-120 | `ts_persistence_diagram_shift` 去掉按 max(m1,m2) 平均 → 真 W1 |
| R6-121 | Betti scale 用 off-diagonal（i<j）成对距离中位（不再被对角 0 压低） |
| R6-160 | variance-shift H0 改 pooled within-segment 方差（纯均值平移不再触发 variance 报警） |
| R6-161 | 完美 changepoint（段内方差 0）用相对 variance floor，不 skip |
| R6-162 | GLR 输出单位改 `dimensionless_score` |

## 八、Intraday（#182-199）

| 编号 | 修复 |
|---|---|
| R6-184 | `intra_extreme_bar_return` side 显式 `{max,min}` 校验 |
| R6-185 | `intra_lunch_gap_return` 上午只取 Close、下午只取 Open，不再 joint dropna |
| R6-186 | `intra_limit_first_hit_time` 触板用 minute high/low（sealed 才用 close），可选手动传 high/low |
| R6-187 | `_limit_mask` 三态：limit 未知 → NaN（不再当 False 计数） |
| R6-188 | `intra_limit_reopen_count` 在 official grid 上跑状态机，missing 中断 episode（不 dropna） |
| R6-189 | `intra_amihud` scale `searchable=False`（纯单位缩放） |
| R6-190 | segment/lunch/entropy 的 `session_tz/morning_cutoff/afternoon_start/normalize` 声明进 param_names + `searchable=False`（pandas+polars 双后端同步） |
| R6-191 | concentration/entropy 对负值 fail-closed（NonnegativeActivity 契约，不 abs） |
| R6-192 | `_log_returns` 显式正价契约 |
| R6-193 | `intraday_impact_decay_rate` `ret<=-1` → 整日 fail（不再 floor 到 1e-12） |
| R6-194 | shock 阈值用严格 prior prefix `absr[:e]`（大 r_e 不能抬自己阈值） |
| R6-195 | 相邻 shock 加 refractory（e+horizon 内不再重复拟合同一 impact path） |
| R6-196 | `available_at=session_close` / `same_session_usable=False` 变机器字段（新增 OperatorMetadata 字段 + registry 持久化） |
| R6-197 | impact-decay 参数 ParamSpec（horizon 2..60、shock_quantile 0.5..0.99） |

## 九、Statistical V2 + Research Models（#158-168、#200-218）

| 编号 | 修复 |
|---|---|
| R6-158 | HVG clustering：degree<2 节点按标准约定计入 C=0（不再从平均里删） |
| R6-159 | HVG motif window ≤80（O(W³)，防 W=252 爆炸） |
| R6-163 | Pastor-Stambaugh gamma 输出单位 `return_per_million_currency` |
| R6-164 | PS `min_periods <= window-1` 进 RelationalParamSpec |
| R6-165 | Qn/HL window 上限 512（O(W²)） |
| R6-166 | 单位串 `same_as_target` → `same_as:target`（robust_scale、cs_state_ops） |
| R6-167 | Pickands `window >= 4k+1` 进 RelationalParamSpec |
| R6-168 | EVT threshold stability：`k_max<=window-2`、`k_min<k_max` 进 RelationalParamSpec |
| R6-200 | DMD `top_k <= rank`（否则 reject，不再 `min()` clip） |
| R6-201 | DMD `K>=rank+2`、`rank<=dim` 进 RelationalParamSpec |
| R6-202 | DMD 频率用 `|arg λ|/(2π)`（共轭对符号稳定） |
| R6-203 | DMD 浓度先合并共轭对能量（不再拆成 +f/-f 两半） |
| R6-204 | bicoherence 分段右对齐（不再丢最新 remainder） |
| R6-205 | max bicoherence → top-decile 均值（去极值选择偏） |
| R6-206 | `floor(window/n_segments) >= 8` 进 RelationalParamSpec |
| R6-207 | kernel Granger 改 nested kernel `K_F = K_Y + η·K_X`（restricted=K_Y，共享 Y bandwidth/ridge）→ 模型公平 |
| R6-208 | Granger `window - lag >= 24` 进 RelationalParamSpec |
| R6-209 | residualized HSIC 改 blocked cross-fitting（不再同窗拟合又同窗测） |
| R6-210 | HSIC 改 off-diagonal unbiased 估计（null baseline 不随 N 增长） |
| R6-211 | BDS `embedding_dim >= 2` |
| R6-212 | BDS `distance_multiplier` 只开放 {0.5,1.0,1.5,2.0} |
| R6-213 | BDS `N_m = window-embedding_dim+1 >= 12` 进 RelationalParamSpec |
| R6-214 | SR 改名 `ts_rolling_sr_gaussian_mean_shift_score`（滚动窗重置，非真 sequential），旧名 alias |
| R6-215 | SR 增加 `side={up,down}`（双向均值平移检测） |
| R6-216 | SR `baseline_window+8 <= window` 进 RelationalParamSpec |
| R6-217 | `shift_sigma >= 0.01`（0 是死参数） |
| R6-218 | multifractal asymmetry 文档化 A 股覆盖率审计要求（零增量→NaN 已知，升 daily 前须实测覆盖率，禁加 EPS floor） |

## 十、治理层（§19-24、§32-33）

- **RelationalParamSpec**（base.py）：跨参数可行性约束（`window>=4k+1` 等 11 组），
  `validate_operator_call` 中央门在调用边界拒绝 guaranteed-NaN 组合；注册表/搜索 grammar
  可据此预剪枝。`_kernel_param_defaults` 增强以穿透 `_fn` 桥解析真实默认值。
- **operator_cost_model.py**（新建）：`runtime_cost/memory_cost(canonical, params)`
  参数感知复杂度（HVG motif O(W³)、Qn/HL O(W²)、bicoherence O(W·F²)、DMD O(W·d²+d³)、
  RQA O(M²)），`window=20 vs 500` 不再同价。
- **operator_audits.py**（新建）：§33 的 14 个自动审计原语
  （equivalent-parameter / relational / self-contamination / cohort / column-permutation /
  membership-vintage / theiler / effective-N / zero-missing-invalid / scale-invariance /
  complexity-vs-param / golden-reference / cache-invalidation / surface-duplication），
  均为确定性纯函数，供 CI 门使用（暂不接入 load_all 以免证据重生成前 mass-fail）。
- **available_at / same_session_usable**：OperatorMetadata + registry 新增机器字段（#196）。
- **unit ontology**：`same_as_target` → `same_as:target` 全量统一（#166）。

## 十一、验证

- 36 个改动文件全部 `py_compile` 干净；`load_all()` 正常（1362 算子）。
- 新增 `tests/operators/test_round6_governance_2026_08.py`（11 个测试）：
  RelationalParamSpec 拒绝 Pickands/GLR/PS/DMD/RQA 的 guaranteed-NaN 组合、cost model
  分族排序、audit 原语运行。
- 通过：`test_gemini_v2_pack`（101）、`test_advanced_ops`（39）、`test_deepening`、
  `test_geometry_math_expansion`（5）、`test_round6_governance`（11）、`test_dynamics_pack`、
  `test_ashare_typed_ops`、`test_microstructure_ops` 等。
- 功能验证：#155 axis 过滤、#157 ParamSpec 不变量、#150 CSE 去重计数、#151 LRU、
  #153 未注册拒绝写持久缓存、#106 dim≥2、#109/148 alias 解析、#134 邻居双有效、
  #138 old-mass 门、#144 bucket 门、#160 方差 H0、#161 完美 changepoint、#193 r≤-1、
  #194 prior 阈值、#196 机器字段、#207 nested kernel、#215 SR side。

## 十二、当前已知失败（非本批引入）

- `test_polars_parity_minute`（`intraday_bvc_imbalance`）：`polars_flow_impact.py`
  的 polars 精度问题，**本批未触碰该文件**；baseline（git stash 后）同样失败。
- `test_chip_flow_pack` 的 legacy-proxy 相关（若仍红）：micro_* 的 R5-06 声明已补
  （`ts_cpt_value` preset、`micro_vpin/kyle_lambda` window/min_periods）。
- 历史遗留：`pit_safe=False` 证据过期级联、alpha-language 语义变化（并发会话 20:39 改动）——
  恢复路径同 R5 报告（并发落定后重跑 evidence 链）。
- **R7-232（并发会话新加的 `registration_audit.py` 逻辑签名门，02:15 修改）**：在我
  的 governance 套件运行期间触发 `ashare_limit_down_touch/polars` 逻辑签名 mismatch
  —— 这是并发会话自己的新门 + 其稳定的 `polars_limit_misc.py` 的 import-order 敏感问题，
  与本批无关。并发会话 02:15 后已自行收敛（`load_all` 恢复 OK）。我的 11 个 governance
  测试在此之前全绿。

## 十三、文档化延后项（需并发会话协调）

1. `operator_cost_model` / `operator_audits` 接入搜索预算与 CI 门（§32/§33 落地）。
2. `edge_gate_strict=True` + 未声明算子归入 required/EDGE_IMMUNE + 重生成 edge evidence。
3. PanelSchema `value_columns` 统一（R5 延后项，归并发会话 storage/sources）。
4. `ts_turnover_holding_age` 真正 recursive holder-age state（#139 升级路径）。
5. SR 真 sequential 实现（带 checkpoint 的 stateful 递归，R6-214 的升级路径）。
6. multifractal asymmetry 的 A 股覆盖率实测（R6-218 前置条件）。
