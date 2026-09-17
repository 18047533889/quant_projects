"""Reviewed signature and unit repairs; no generic factor fallback."""
import ast
from factor_engine.tools.catalog_r20_return_units import migrate_formula as repair_units
def C(name,*args,**kwargs):
    return ast.Call(ast.Name(name,ast.Load()),list(args),[ast.keyword(k,v) for k,v in kwargs.items()])
def L(value): return ast.Constant(value)
def field(name,table="StockDailyBarAdj"): return C("field",L(name),table=L(table))
def benchmark():
    return C("subtract",C("safe_div_null",
        C("source_col",L("BenchmarkIndexDailyBar"),L("Close"),L("index"),L("000300.SH")),
        C("source_col",L("BenchmarkIndexDailyBar"),L("PreClose"),L("index"),L("000300.SH"))),L(1.0))
def migrate_formula(formula,logic=""):
    formula,notes=repair_units(formula)
    try: tree=ast.parse(formula,mode="eval")
    except SyntaxError: return formula,notes
    class Repair(ast.NodeTransformer):
        def visit_Name(self,node):
            if node.id=="index_ret":
                notes.append("SOURCE_DEFINITION: benchmark return explicitly uses CSI300 000300.SH Close/PreClose, decimal units.")
                return benchmark()
            return node
        def visit_Call(self,node):
            node=self.generic_visit(node)
            if not isinstance(node.func,ast.Name): return node
            n=node.func.id;a=node.args;k={v.arg:v for v in node.keywords}
            if n in {"ts_chatterjee_xi","ts_hsic","ts_conditional_mutual_information","ts_distance_correlation_partial_proxy","ts_lag_of_peak_corr"} and "source" in k and "y" not in k:
                k["source"].arg="y"
                notes.append(f"PARAMETER_ALIAS: {n} source is the second observed input y; x and temporal direction unchanged.")
                k={v.arg:v for v in node.keywords}
            if n=="period_lag" and "require_consecutive" in k and isinstance(k["require_consecutive"].value,ast.Constant) and k["require_consecutive"].value.value is True:
                node.keywords=[v for v in node.keywords if v.arg!="require_consecutive"]
                notes.append("EXACT_CONTRACT: period_lag already retrieves exact fiscal-ordinal minus periods; missing target period remains null. Redundant require_consecutive=True removed; revision policy retained.")
            if n=="field" and len(a)==1 and isinstance(a[0],ast.Constant) and a[0].value in {"high_limit","low_limit"} and "table" in k and isinstance(k["table"].value,ast.Constant) and k["table"].value.value=="StockDailyBarAdj":
                k["table"].value=L("StockDailyBar")
                notes.append("SOURCE_CORRECTION: official daily upper/lower limit prices must be RAW StockDailyBar, not adjusted.")
            if n=="ts_conditional_transfer_entropy" and "window" in k and "bins" in k:
                w=k["window"].value;b=k["bins"].value;lag=k.get("lag")
                lag=lag.value.value if lag and isinstance(lag.value,ast.Constant) else 1
                if isinstance(w,ast.Constant) and isinstance(b,ast.Constant) and isinstance(w.value,int) and isinstance(b.value,int):
                    minimum=3*b.value**4+lag
                    if w.value<minimum:
                        k["window"].value=L(minimum)
                        notes.append(f"SEMANTIC_DEFINITION: CTE window={minimum} is minimum supported for bins={b.value}, lag={lag}; support criterion preserved.")
            if n=="ts_spectral_entropy":
                x=a[0] if a else (k["x"].value if "x" in k else None)
                if x is not None:
                    src=ast.unparse(x)
                    if "turnover" in src.lower() or src.startswith("relative_volume("):
                        node.func.id="ts_activity_spectral_entropy"
                        notes.append("TYPE_CORRECTION: activity input uses normalized activity spectral entropy, not return spectrum.")
                    elif src.startswith("amihud_illiquidity(") or src.startswith("atr_pct("):
                        node.func.id="ts_signal_spectral_entropy"
                        notes.append("SEMANTIC_DEFINITION: same normalized periodogram entropy on the selected liquidity/ATR level, via explicit scalar-signal canonical.")
            if n=="ts_current_drawdown_duration" and len(a)==1 and "StockDailyBarAdj" in ast.unparse(a[0]) and "ret" in ast.unparse(a[0]):
                node.args=[field("close")]
                notes.append("SEMANTIC_DEFINITION: drawdown duration uses adjusted close wealth level, not returns or rolling cumulative-return index.")
            if n in {"ts_markov_state_entropy","ts_ordinal_irreversibility"} and "min_periods" in k:
                k["min_periods"].arg="min_count" if n=="ts_markov_state_entropy" else "min_patterns"
                notes.append(f"PARAMETER_ALIAS: {n} obsolete min_periods mapped to declared support-count parameter.")
            if n.startswith("fiscal_") and "signal" in k:
                from factor_engine.cleaned_operators.registry import OperatorRegistry
                op=OperatorRegistry.get(n,"pandas_numpy")
                if op is not None and "x" in op.metadata.param_names and "signal" not in op.metadata.param_names:
                    k["signal"].arg="x";notes.append(f"PARAMETER_ALIAS: {n} signal renamed to declared x; expression unchanged.")
            if n=="cs_shrinkage_mahalanobis" and len(a)==3 and "f4" not in k:
                node.args.append(C("ts_std",field("ret"),L(20)))
                notes.append("SEMANTIC_DEFINITION: fourth Mahalanobis feature explicitly chosen as 20-session adjusted-return volatility.")
            if n=="state_episode_excursion_balance" and len(a)==3:
                node.args=a[:2];notes.append("PARAMETER_CORRECTION: excursion balance takes path and state, not an undocumented third panel.")
            if n in {"ts_run_concentration","ts_run_strength","ts_run_efficiency","ts_transition_intensity"} and len(a)==1 and "state" not in k:
                node.args.append(C("gt",a[0],C("ts_mean",a[0],L(20))))
                notes.append(f"SEMANTIC_DEFINITION: {n} state is input above its own trailing 20-session mean.")
            if n in {"ts_interval_union_coverage","ts_interval_nesting_depth","ts_interval_occupancy_entropy"} and len(a)==1:
                node.args=[C("ts_min",a[0],L(20)),C("ts_max",a[0],L(20))]
                notes.append(f"SEMANTIC_DEFINITION: {n} interval is the chosen signal's trailing 20-session min/max band.")
            if n=="ts_interval_nesting_depth" and "window" in k:
                node.keywords=[v for v in node.keywords if v.arg!="window"]
                notes.append("PARAMETER_CORRECTION: interval nesting is the formal interval collection statistic and has no window knob.")
            if n=="ts_interval_occupancy_mode_distance" and len(a)==1:
                node.args=[a[0],C("ts_min",a[0],L(20)),C("ts_max",a[0],L(20))]
                notes.append("SEMANTIC_DEFINITION: occupancy mode measured against input's 20-session min/max intervals.")
            if n=="ts_interval_exploration_efficiency" and len(a)==1:
                node.args=[C("ts_max",a[0],L(20)),C("ts_min",a[0],L(20)),a[0]]
                notes.append("SEMANTIC_DEFINITION: exploration uses signal as close and its 20-session extrema as same-unit high/low.")
            if n in {"ts_envelope_pressure","ts_envelope_boundary_dwell","ts_envelope_compression"} and len(a)==1:
                mid=C("ts_mean",a[0],L(20));width=C("multiply",C("ts_std",a[0],L(20)),L(2.0))
                upper=C("add",mid,width);lower=C("subtract",mid,width)
                node.args=[upper,lower,mid] if n=="ts_envelope_compression" else [a[0],upper,lower]
                notes.append(f"SEMANTIC_DEFINITION: {n} envelope is 20-session mean +/- 2 standard deviations of the original input.")
            if n in {"ts_weighted_downside_deviation","ts_weighted_semivariance","ts_weighted_expected_shortfall","ts_weighted_drawdown_area"} and len(a)==1:
                if n=="ts_weighted_drawdown_area" and "ret" in ast.unparse(a[0]):node.args[0]=field("close")
                node.args.append(ast.Name("turnover_ratio",ast.Load()))
                notes.append(f"SEMANTIC_DEFINITION: {n} uses nonnegative turnover-ratio observation weights; drawdown uses adjusted price level.")
            if n=="ts_stratified_mean_spread" and len(a)==1:
                node.args.append(ast.Name("turnover_ratio",ast.Load()))
                notes.append("SEMANTIC_DEFINITION: target spread sorted by observed turnover ratio.")
            if n in {"ts_energy_break_score","ts_joint_energy_shift","ts_feature_mode_share","ts_feature_subspace_rotation"} and len(a)==1:
                node.args.extend([ast.Name("turnover_ratio",ast.Load()),C("ts_std",field("ret"),L(20))])
                notes.append(f"SEMANTIC_DEFINITION: {n} adds turnover and 20-session return volatility as explicitly chosen second/third features.")
            if n in {"ts_vector_path_efficiency","ts_vector_self_intersection_rate"} and len(a)==1:
                node.args.append(ast.Name("turnover_ratio",ast.Load()))
                notes.append(f"SEMANTIC_DEFINITION: {n} second path feature is turnover ratio.")
            if n in {"ts_conditional_mutual_information","ts_distance_correlation_partial_proxy"} and len(a)==2 and "z" not in k:
                node.args.append(ast.Name("turnover_ratio",ast.Load()))
                notes.append(f"SEMANTIC_DEFINITION: {n} conditions on observed turnover ratio.")
            if n in {"ts_conditional_transfer_entropy","ts_corr_if"} and len(a)==2 and "condition" not in k:
                node.args.append(C("gt",ast.Name("turnover_ratio",ast.Load()),C("ts_mean",ast.Name("turnover_ratio",ast.Load()),L(20))))
                notes.append(f"SEMANTIC_DEFINITION: {n} conditions on above-20-session-mean turnover.")
            if n in {"ts_copula_central_asymmetry","ts_extrema_confirmation_rate","ts_hsic","ts_lag_of_peak_corr","ts_lagged_mutual_information"} and len(a)==1:
                node.args.append(ast.Name("turnover_ratio",ast.Load()))
                notes.append(f"SEMANTIC_DEFINITION: {n} second observed input is turnover ratio.")
            if n=="ts_score_rank_weighted_mean" and len(a)==1:
                node.args.append(ast.Name("turnover_ratio",ast.Load()))
                notes.append("SEMANTIC_DEFINITION: observation scores are turnover ratios.")
            if n=="ts_response_slope_asymmetry" and len(a)==1:
                node.args.append(benchmark())
                notes.append("SEMANTIC_DEFINITION: response slope is relative to CSI300 decimal return.")
            if n=="ts_price_delay" and len(a)==1:
                node.args=[field("ret"),benchmark()]
                notes.append("SEMANTIC_DEFINITION: price delay uses adjusted-stock return against explicitly identified CSI300 return, not a level/activity panel.")
            if n=="KeltnerPosition" and len(a)==6 and not isinstance(a[3],ast.Constant):
                node.args=[*a[:3],L(20),a[4],a[5]]
                notes.append("PARAMETER_CORRECTION: Keltner EMA window=20 replaces erroneous fourth close panel; ATR window/multiplier retained.")
            if n in {"ALMA","ATR_WILDER","HMA","RSI_WILDER","WMA"} and len(a)==5:
                node.args=([a[1],a[2],a[3],L(14)] if n=="ATR_WILDER" else [a[3],L(20)])
                notes.append(f"SEMANTIC_DEFINITION: {n} OHLCV sketch selects actual declared price inputs and a 14/20-session window.")
            if n=="ts_kalman_trend" and "q" in k:
                value=k["q"].value
                node.keywords=[v for v in node.keywords if v.arg!="q"]+[ast.keyword("q_level",value),ast.keyword("q_trend",value)]
                notes.append("SEMANTIC_DEFINITION: legacy Kalman q explicitly sets both level and trend process noise.")
            if n.startswith("ashare_"):
                raw=lambda key:field(key,"StockDailyBar")
                valid=C("gt",raw("volume"),L(0.0))
                up=C("limit_up_close",raw("close"),raw("high_limit"),L(0.005))
                down=C("limit_down_close",raw("close"),raw("low_limit"),L(0.005))
                fixed=None
                if n=="ashare_limit_distance" and len(a)==3:
                    fixed=[raw("close"),raw("high_limit")]
                if len(a)==5:
                    if n=="ashare_limit_asymmetry":fixed=[up,down,L(20)]
                    elif n=="ashare_limit_distance":fixed=[raw("close"),raw("high_limit")]
                    elif n=="ashare_limit_open_failed":fixed=[raw("open"),raw("low"),raw("high_limit"),L(0.005)]
                    elif n=="ashare_suspension_episode_length":fixed=[raw("is_suspend")]
                    elif n=="ashare_limit_event_density":fixed=[C("maximum",up,down),L(20)]
                    elif n=="ashare_limit_touch_count":fixed=[raw("high"),raw("low"),raw("high_limit"),raw("low_limit"),L(20)]
                    elif n=="ashare_one_price_limit_streak":fixed=[raw("open"),raw("high"),raw("low"),raw("close"),raw("high_limit"),raw("low_limit"),valid]
                    elif n=="ashare_open_at_upper_limit":fixed=[raw("open"),raw("high_limit"),L(0.005)]
                    elif n in {"ashare_days_since_limit_up","ashare_days_since_limit_down"}:fixed=[up if n.endswith("_up") else down,L(252)]
                    elif n in {"ashare_limit_up_volume_ratio","ashare_limit_down_volume_ratio"}:fixed=[raw("volume"),up if "_up_" in n else down,L(20)]
                if fixed is not None:
                    node.args=fixed
                    notes.append(f"SOURCE_SIGNATURE_CORRECTION: {n} uses raw official OHLC/limit prices and declared event signature; tolerance CNY 0.005, window 20 or days-since cap 252 where missing.")
            if n in {"ts_regression_r2","ts_regression_resid"} and len(a)==2 and isinstance(a[1],ast.Constant) and isinstance(a[1].value,int):
                x,w=a;nobs=w.value
                slope=C("ts_time_slope",x,w,min_periods=L(nobs))
                if n=="ts_regression_r2":
                    out=C("safe_div_null",C("multiply",C("square",slope),L(nobs*(nobs+1)/12)),C("square",C("ts_std",x,w,L(1))))
                else:
                    out=C("subtract",x,C("add",C("ts_mean",x,w),C("multiply",slope,L((nobs-1)/2))))
                notes.append(f"EXACT_DEFINITION: {n}(x,{nobs}) interpreted as full-finite-window OLS against time positions; expanded slope/mean/sample-variance identity, no invented market regressor.")
                return out
            if n in {"ts_quantile_transport_slope","ts_quantile_transport_curvature","ts_mmd_rbf_shift"} and "window" in k and isinstance(k["window"].value,ast.Constant):
                total=k["window"].value.value
                if isinstance(total,int) and total>=6:
                    node.keywords=[v for v in node.keywords if v.arg!="window"]+[ast.keyword("recent_window",L(total//3)),ast.keyword("old_window",L(total-total//3))]
                    notes.append(f"SEMANTIC_DEFINITION: {n} window={total} split into recent one-third and preceding two-thirds.")
            if n in {"ts_min","ts_max"} and len(a)==2 and not isinstance(a[1],ast.Constant):
                node.func.id="minimum" if n=="ts_min" else "maximum"
                notes.append("SYNTAX_CORRECTION: min/max of two panel expressions is elementwise, not a rolling window.")
            if n in {"ts_vector_turning_coherence","ts_vector_path_curvature","ts_extrema_divergence_strength","ts_beta_break_score"} and len(a)==1:
                node.args.append(ast.Name("turnover_ratio",ast.Load()) if n!="ts_beta_break_score" else benchmark())
                notes.append(f"SEMANTIC_DEFINITION: {n} adds explicit turnover feature (beta uses CSI300 return).")
            if n in {"ts_feature_effective_rank","cs_sliced_wasserstein_copula_shift"} and len(a)==1:
                node.args.extend([ast.Name("turnover_ratio",ast.Load()),C("ts_std",field("ret"),L(20))])
                notes.append(f"SEMANTIC_DEFINITION: {n} uses original feature plus turnover and 20-session return volatility.")
            if n in {"ts_vector_state_local_density","ts_vector_state_mahalanobis","cs_knn_local_moran"} and len(a)==1:
                node.args.extend([ast.Name("turnover_ratio",ast.Load()),C("ts_std",field("ret"),L(20)),C("ts_sum",field("ret"),L(20))])
                notes.append(f"SEMANTIC_DEFINITION: {n} auxiliary coordinates are turnover, 20-session volatility and momentum.")
            if n in {"ts_conditional_mutual_information","ts_distance_correlation_partial_proxy"} and {"x","y"}<=k.keys() and "z" not in k:
                node.keywords.append(ast.keyword("z",ast.Name("turnover_ratio",ast.Load())))
                notes.append(f"SEMANTIC_DEFINITION: {n} conditioning z is turnover.")
            if n=="group_spd_feature_structure_shift" and len(a)==2:
                node.args=[a[0],ast.Name("turnover_ratio",ast.Load()),C("ts_std",field("ret"),L(20)),a[1]]
                notes.append("SEMANTIC_DEFINITION: group SPD shift adds turnover and return volatility, preserving supplied group.")
            if n=="group_ex_self_weighted_mean" and len(a)==2:
                node.args=[a[0],ast.Name("free_market_cap",ast.Load()),a[1]]
                notes.append("SEMANTIC_DEFINITION: peer weighted mean uses free-float market cap and the supplied group.")
            if n=="panel_factor_pocket_strength" and len(a)==1:
                node.args.append(C("ts_sum",field("ret"),L(20)))
                notes.append("SEMANTIC_DEFINITION: pocket factor is trailing 20-session adjusted-return momentum.")
            if n in {"ts_turnover_profit_share","ts_turnover_holding_age","ts_turnover_age_dispersion"} and len(a)==1:
                node.args=[field("close"),ast.Name("turnover_ratio",ast.Load())]
                notes.append(f"SIGNATURE_CORRECTION: {n} requires adjusted price and observed turnover, not a lone return/activity input.")
            if n=="state_hold" and len(a)==1:
                node.args.append(C("ne",a[0],C("delay",a[0],L(1))))
                notes.append("SEMANTIC_DEFINITION: hold updates on observed input changes.")
            if n=="cross_event" and len(a)==1:
                node.args.append(C("ts_mean",a[0],L(20)))
                notes.append("SEMANTIC_DEFINITION: cross_event baseline is the input's trailing 20-session mean.")
            if n in {"state_ewm_if","state_latch","ts_transition_count"} and len(a)==1:
                above=C("gt",a[0],C("ts_mean",a[0],L(20)))
                if n=="state_ewm_if":node.args.append(above)
                elif n=="state_latch":node.args=[above,C("lt",a[0],C("ts_mean",a[0],L(20)))]
                else:node.args=[above]
                notes.append(f"SEMANTIC_DEFINITION: {n} event/state determined by input relative to its trailing 20-session mean.")
            if n=="ts_recurrence_trapping_time" and len(a)==7:
                node.args=a[:5]+[a[6]]
                notes.append("PARAMETER_CORRECTION: trapping-time kernel has no separate obsolete sixth run-length control; min_periods retained.")
            if n=="ts_time_since_change" and "window" in k:
                k["window"].arg="max_lookback"
                if a:node.args[0]=C("gt",a[0],C("ts_mean",a[0],L(20)))
                notes.append("SIGNATURE_DEFINITION: window maps to max_lookback; changes defined on above-mean state.")
            if n.startswith("ts_matrix_profile_"):
                from factor_engine.cleaned_operators.registry import OperatorRegistry
                op=OperatorRegistry.get(n,"pandas_numpy")
                names=set(op.metadata.param_names) if op is not None else set()
                for old,new in (("subsequence_length","m"),("window","history_window")):
                    if old in k and new in names and old not in names:
                        k[old].arg=new
                        notes.append(f"PARAMETER_ALIAS: {n} {old} -> {new} per declared contract.")
                    elif new in k and old in names and new not in names:
                        k[new].arg=old
                        notes.append(f"PARAMETER_ALIAS: {n} {new} -> {old} per declared contract.")
            if n in {"turnover_shock","abnormal_turnover"} and len(a)==3:
                node.args=[C("ts_mean",a[0],a[1]),a[2]]
                notes.append(f"SEMANTIC_DEFINITION: {n}(x,short,long) measures the short-window mean against its preceding long-window baseline; both horizons retained.")
            if n=="ts_quantilogram" and len(a)==2 and not isinstance(a[1],ast.Constant):
                node.func.id="ts_cross_quantilogram"
                notes.append("SIGNATURE_CORRECTION: two-panel quantilogram uses the cross-series canonical; formal default window=120, lower quantiles=0.1, lag=1.")
            if n in {"ts_support_level","ts_resistance_level","ts_support_slope","ts_resistance_slope"} and len(a)==4:
                node.args=a[:3]+[L(250),a[3]]
                notes.append("PARAMETER_CORRECTION: preserve left/right/point-count; add explicit trailing pivot history=250 bars.")
            if n in {"ts_distance_to_support","ts_distance_to_resistance","ts_support_break","ts_resistance_break"} and len(a)==5:
                node.args=a[:4]+[L(250),a[4]]
                notes.append("PARAMETER_CORRECTION: preserve left/right/point-count; add explicit trailing pivot history=250 bars.")
            if n in {"ts_pivot_high_age","ts_pivot_low_age","ts_last_pivot_low","ts_last_pivot_high"} and len(a)==3:
                node.args=a+[L(250)]
                notes.append("PARAMETER_DEFINITION: confirmed pivot search history explicitly 250 bars; left/right confirmation retained.")
            if n=="group_multi_level_rank_consistency" and len(a)==2:
                node.args=[a[0]]+[C("source_col",L("StockIndustry"),L("IndustryCode"),L("IndustrySource"),L(level)) for level in ("sw_l1","sw_l2","sw_l3")]
                notes.append("SOURCE_DEFINITION: multi-level consistency explicitly uses SW level 1/2/3 group identities, not three copies of one group.")
            if n=="composition_clr_component" and len(a)==9 and sum(ast.dump(a[0])==ast.dump(x) for x in a[1:])==1:
                node.args=[a[0]]+[x for x in a[1:] if ast.dump(x)!=ast.dump(a[0])]
                notes.append("SIGNATURE_CORRECTION: target is already first composition part; remove duplicate target slot while retaining all eight distinct economic components.")
            if ((n=="composition_normalized_entropy" and len(a)>8) or (n=="composition_entropy" and len(a)==2)) and not k:
                import math
                def summed(xs):
                    out=xs[0]
                    for x in xs[1:]:out=C("add",out,x)
                    return out
                total=summed(a)
                weighted=summed([C("multiply",x,C("log_positive_or_nan",x)) for x in a])
                out=C("subtract",C("log_positive_or_nan",total),C("safe_div_null",weighted,total))
                if n=="composition_normalized_entropy":out=C("safe_div_null",out,L(math.log(len(a))))
                notes.append(f"EXPLICIT_DEFINITION: {n} uses Shannon entropy over all {len(a)} supplied positive components via log(sum(x))-sum(x*log(x))/sum(x), divided by log(N) only for normalized entropy; no component dropped; nonpositive components yield null.")
                return out
            if n=="report_benford_js_divergence" and len(a)==6:
                node.func.id="report_vector_benford_js_divergence"
                notes.append("SEMANTIC_DEFINITION: six contemporaneous report monetary items form one first-digit distribution; explicit research six-item JS-distance canonical, not a rolling time-series statistic; incomplete/zero vectors null.")
            if n=="intraday_impact_asymmetry" and len(a)==2 and {"horizon","shock_quantile"}<=k.keys():
                node.keywords=[v for v in node.keywords if v.arg not in {"horizon","shock_quantile"}]
                node.args=[a[0],C("multiply",C("sign",a[0]),a[1])]
                notes.append("SEMANTIC_REDEFINITION: formal full-session buy/sell impact asymmetry uses minute-return-sign times amount as explicitly estimated signed flow (not observed L2 flow); horizon/shock_quantile belong only to impact-decay term and are removed from asymmetry.")
            if n=="ADX" and len(a)==5:
                node.args=[a[1],a[2],a[3],L(14)];notes.append("SEMANTIC_DEFINITION: OHLCV ADX sketch uses high/low/close and 14-session window.")
            if n=="RSX" and len(a)==5:
                node.args=[a[3],L(14)];notes.append("SEMANTIC_DEFINITION: OHLCV RSX sketch uses close and length 14.")
            return node
    tree=Repair().visit(tree)
    return (ast.unparse(ast.fix_missing_locations(tree)),notes) if notes else (formula,[])
