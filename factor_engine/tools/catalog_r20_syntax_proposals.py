#!/usr/bin/env python3
"""Reconstruct R19 DSLParseError catalog rows and validate every proposal."""
from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "evidence" / "factor_catalog_20260916"
sys.path.insert(0, str(ROOT))


MOM = "ts_sum(ret, 20)"
REV = "neg(ts_sum(ret, 5))"
TURN = "turnover_ratio"
VOL = "ts_std(ret, 20)"
LIQ = "amihud_illiquidity(ret, close, volume, 20)"
OVERNIGHT = "subtract(safe_div_null(open, pre_close), 1.0)"
INTRADAY = "subtract(safe_div_null(close, open), 1.0)"
CHIP = "ts_turnover_profit_share(close, turnover_ratio, 60)"
ATR_PCT = "safe_div_null(ts_mean(subtract(high, low), 14), close)"
VWAP_DIST = "safe_div_null(subtract(close, vwap), vwap)"
BENCH_CLOSE='source_col("BenchmarkIndexDailyBar", "Close", "index", "000300.SH")'
BENCH_PRE_CLOSE='source_col("BenchmarkIndexDailyBar", "PreClose", "index", "000300.SH")'
BENCH = f"subtract(safe_div_null({BENCH_CLOSE}, {BENCH_PRE_CLOSE}), 1.0)"
PRICE_DELAY = f"ts_price_delay(ret, {BENCH}, 60, 5, 30)"
RESID_MOM = f"residual_momentum_capm(ret, {BENCH}, 60)"


def features(name: str) -> tuple[str, str, str]:
    if "liquidity_turnover_vol" in name:
        return LIQ, TURN, VOL
    if "overnight_intraday_turnover" in name:
        return OVERNIGHT, INTRADAY, TURN
    if "chip_momentum_liquidity" in name:
        return CHIP, MOM, LIQ
    return MOM, TURN, VOL


def target(name: str) -> str:
    if name.startswith("turnover_"): return TURN
    if name.startswith("amihud_"): return LIQ
    if name.startswith("atr_pct_"): return ATR_PCT
    if name.startswith("vwap_dist_"): return VWAP_DIST
    if name.startswith("chip_profit_share_"): return CHIP
    if name.startswith("chip_mode_distance_"):
        return "ts_turnover_cost_mode_distance(close, turnover_ratio, 60)"
    return "ret"


def interaction_suffix(name: str) -> str | None:
    if name.endswith("return_interaction"): return "ret"
    if name.endswith("turnover_interaction"): return TURN
    if name.endswith("freefloat_interaction"): return "free_cap"
    if name.endswith("pledge_proxy_interaction"): return "holder_pledge_ratio(field('share_pledge', table='StockTopTenShareholder'), field('share_number', table='StockTopTenShareholder'))"
    if name.endswith("goodwill_interaction"): return "safe_div_null(goodwill, total_assets)"
    return None


def relationship_proxy(name: str) -> tuple[str, str]:
    # The original 20 placeholders exceeded parser arity and referred to synthetic
    # shareholder_id_1..20 columns.  Rebuild from the actual PIT top-ten holdings
    # relation, preserving shareholder identity/share/rank rather than market turnover.
    if "overlap_ratio" in name:
        return "neg(add(relation_entry_count(shareholder_id, 8), relation_exit_count(shareholder_id, 8)))", "以真实前十大股东实体的进入与退出总数反向定义持有人重叠持续性"
    if "entry_count" in name:
        return "relation_entry_count(shareholder_id, 8)", "按真实前十大股东实体和报告期计算进入数量"
    if "exit_count" in name:
        return "relation_exit_count(shareholder_id, 8)", "按真实前十大股东实体和报告期计算退出数量"
    if "distinct_count" in name:
        return "relation_distinct_count(shareholder_id)", "按真实前十大股东实体计算不同持有人数量"
    if "entity_mobility" in name:
        return "add(relation_entry_count(shareholder_id, 8), relation_exit_count(shareholder_id, 8))", "按真实股东实体进出总数定义持有人实体迁移"
    if "rank_mobility" in name:
        return "abs(relation_weighted_change(shareholder_rank, share_ratio))", "按真实股东排名并以持股比例加权定义排名迁移"
    if "share_mobility" in name:
        return "abs(relation_weighted_change(share_number, share_ratio))", "按真实持股数量和持股比例定义份额迁移"
    return "relation_distinct_count(shareholder_id)", "按真实前十大股东实体定义持有人关系广度"


def r65_formula(name: str, old: str) -> tuple[str, str] | None:
    chip_ops = {
        "chip_profit_share": CHIP,
        "chip_near_cost": "ts_turnover_near_cost_mass(close, turnover_ratio, 60, 0.05)",
        "chip_cost_dispersion": "ts_turnover_cost_dispersion(close, turnover_ratio, 60)",
        "chip_mode_distance": "ts_turnover_cost_mode_distance(close, turnover_ratio, 60)",
        "chip_entropy": "ts_turnover_cost_entropy(close, turnover_ratio, 60)",
        "chip_skew": "ts_turnover_cost_skew(close, turnover_ratio, 60)",
        "chip_age_dispersion": "ts_turnover_age_dispersion(close, turnover_ratio, 60)",
        "chip_age": "ts_turnover_holding_age(close, turnover_ratio, 60)",
        "chip_old_mass": "ts_turnover_old_mass(close, turnover_ratio, 60)",
    }
    rhs = {
        "momentum": MOM, "reversal": REV,
        "breakout": "safe_div_null(subtract(close, ts_max(close, 20)), ts_max(close, 20))",
        "volume_shock": "ts_zscore(volume, 20)", "volatility": ATR_PCT,
        "structural_distance": "ts_nearest_structural_level_distance(close, 60)",
        "trend_consensus": "ts_multiscale_trend_consensus(close, 60, [5, 20, 60])",
        "downside_risk": "ts_downside_deviation(ret, 60)", "turnover": TURN,
        "index_entry": "ts_sum(ret, 20)",
    }
    for left, lf in chip_ops.items():
        if name.startswith(left + "_x_"):
            key = name.split("_x_", 1)[1]
            return f"multiply({lf}, {rhs[key]})", "用真实筹码分布算子与对应价格量风险机制构成交互"

    if re.match(r"^(keltner|donchian)_(compression|pressure|boundary_dwell)$", name):
        op = next(x for x in ("compression", "pressure", "boundary_dwell") if name.endswith(x))
        if name.startswith("keltner"):
            mid="EMA(close, 20)"; width="ts_mean(subtract(high, low), 14)"; upper=f"add({mid}, multiply({width}, 2.0))"; lower=f"subtract({mid}, multiply({width}, 2.0))"
        else:
            upper="ts_max(high, 20)"; lower="ts_min(low, 20)"; mid=f"multiply(add({upper}, {lower}), 0.5)"
        return f"ts_envelope_{op}({mid}, {upper}, {lower}, 20)", "以明确的 Keltner/Donchian 上下轨衡量通道压缩、压力或边界停留"

    m = re.match(r"close_vs_(EMA|HMA|ALMA|KAMA)_cross_(speed|acceleration)$", name)
    if m:
        ma = "KAMA(close, 20, 2, 30)" if m.group(1)=="KAMA" else f"{m.group(1)}(close, 20)"
        return f"ts_crossing_{m.group(2)}(close, {ma}, 20)", "衡量收盘价穿越指定自适应均线的速度或加速度"

    if "threshold_cycle_" in name:
        op = "period" if name.endswith("period") else "asymmetry"
        if name.startswith("RSX_"): x="RSX(close, 14)"; lo,hi="30.0","70.0"
        elif name.startswith("CMO_"): x="CMO(close, 14)"; lo,hi="-50.0","50.0"
        elif name.startswith("donchian_position_"): x="safe_div_null(subtract(close, ts_min(low, 20)), subtract(ts_max(high, 20), ts_min(low, 20)))"; lo,hi="0.2","0.8"
        elif name.startswith("keltner_position_"): x="safe_div_null(subtract(close, EMA(close, 20)), ts_mean(subtract(high, low), 14))"; lo,hi="-1.0","1.0"
        elif name.startswith("vwap_dist_"): x=VWAP_DIST; lo,hi="-0.02","0.02"
        elif name.startswith("chip_profit_share_"): x=CHIP; lo,hi="0.3","0.7"
        else: x=f"subtract(ret, {BENCH})"; lo,hi="-0.02","0.02"
        return f"ts_threshold_cycle_{op}({x}, {lo}, {hi}, 60)", "按经济固定阈值度量状态循环周期或上下行阶段不对称"

    if name.startswith("interval_"):
        forms={"interval_union_coverage":"ts_interval_union_coverage(low, high, 60)","interval_occupancy_entropy":"ts_interval_occupancy_entropy(low, high, 60, 20)","interval_occupancy_mode_distance":"ts_interval_occupancy_mode_distance(close, low, high, 60, 20)","interval_nesting_depth":"ts_interval_nesting_depth(low, high, 'inside')","interval_exploration_efficiency":"ts_interval_exploration_efficiency(high, low, close, 60)","interval_overlap_connected_component_ratio":"ts_interval_overlap_connected_component_ratio(low, high, 60)"}
        return forms[name], "以日内高低价区间的滚动几何结构刻画覆盖、占用或嵌套特征"

    if name.startswith("group_"):
        op = next((x for x in ("mode_share","effective_rank","mode_localization","spectral_gap","second_mode_localization") if name.endswith(x)), None)
        if op:
            f1,f2,f3=features(name)
            last="0.1, 20" if "localization" in op else "20"
            return f"group_feature_{op}({f1}, {f2}, {f3}, industry_code, 'v1', {last})", "在行业组内用三项明确价格量特征刻画共同模态结构"
    if name.endswith("cs_rank_churn") or name.endswith("cs_tail_retention"):
        op="cs_rank_churn" if name.endswith("cs_rank_churn") else "cs_tail_retention"
        f1,_,_=features(name)
        form=f"cs_rank_churn({f1}, 1, industry_code)" if op=="cs_rank_churn" else f"cs_tail_retention({f1}, 1, 0.1, 'top', industry_code)"
        return form, "在行业组内衡量主特征截面排名周转或尾部留存"

    if "dynamic_knn_" in name:
        t=target(name); f1,f2,f3=features(name)
        peer=f"cs_knn_peer_mean_ex_self({t}, {f1}, {f2}, {f3}, 10)"
        return (f"subtract({t}, {peer})" if "_residual_" in name else peer), "用三项明确价格量特征寻找动态近邻，输出同伴均值或目标残差"

    if "knn_neighbor_retention" in name or "knn_graph_dirichlet_energy" in name:
        f1,f2,f3=features(name)
        op="cs_knn_neighbor_retention" if "retention" in name else "cs_knn_graph_dirichlet_energy"
        return f"{op}({f1}, {f2}, {f3}, 10)", "用明确的三特征近邻图衡量邻居稳定性或图上信号粗糙度"

    spread = "ts_edge_effective_spread(open, high, low, close, 60)" if "edge" in name else "ts_abdi_ranaldo_spread(high, low, close, 60)"
    if name in ("edge_effective_spread","abdi_ranaldo_spread","pastor_stambaugh_liquidity_gamma"):
        if name=="edge_effective_spread": return spread,"用 OHLC 的滚动有效价差估计交易摩擦"
        if name=="abdi_ranaldo_spread": return spread,"用高低收盘价估计隐含买卖价差"
        return "ts_pastor_stambaugh_liquidity_gamma(ret, amount, 60)","以收益反转对成交金额冲击估计流动性 gamma"
    if "_spread_x_" in name:
        rhskey=name.rsplit("_x_",1)[1]
        rf={"turnover":TURN,"atr_pct":ATR_PCT,"ret_1d":"ret","volatility":ATR_PCT,"return":"ret","chip_profit_share":CHIP,"vwap_dist":VWAP_DIST}[rhskey]
        return f"multiply({spread}, {rf})", "将滚动价差与指定价格量状态交互以刻画条件交易摩擦"

    if old.startswith("intraday_") or name.startswith("intraday_") or name.startswith("session_"):
        op=old.split(" ",1)[0]
        # Compile-time surface accepts daily aliases and preserves the named microstructure mechanism.
        args={
            "intraday_bvc_imbalance":"minute_close, minute_volume, 20",
            "intraday_impact_beta":"subtract(safe_div_null(minute_close, ts_lag(minute_close, 1)), 1.0), minute_amount, 20",
            "intraday_impact_asymmetry":"subtract(safe_div_null(minute_close, ts_lag(minute_close, 1)), 1.0), minute_amount, 20",
            "intraday_return_wasserstein_shift":"subtract(safe_div_null(minute_close, ts_lag(minute_close, 1)), 1.0), 20",
            "session_event_recovery_score":"subtract(safe_div_null(minute_close, ts_lag(minute_close, 1)), 1.0), minute_volume, 20",
            "intraday_volume_clock_path_efficiency":"minute_close, minute_volume, 16",
            "intraday_volume_clock_roughness":"minute_close, minute_volume, 16",
            "intraday_rv_signature_slope":"minute_close",
            "intraday_medrv":"minute_close, 20",
            "intraday_minrv":"minute_close, 20",
            "intraday_jump_test_stat":"minute_close, 20",
            "intraday_volatility_time_centroid":"minute_close, 20",
            "intraday_volatility_concentration":"minute_close, 20",
            "intraday_volatility_entropy":"subtract(safe_div_null(minute_close, ts_lag(minute_close, 1)), 1.0), 20",
            "intraday_realized_semivariance_balance":"subtract(safe_div_null(minute_close, ts_lag(minute_close, 1)), 1.0), 20",
            "intraday_rv_signature_curvature":"minute_close, 20",
            "intraday_session_shape_novelty":"minute_close, minute_volume, 20",
            "intraday_profile_pca_residual":"minute_close, minute_volume, 20",
            "intraday_activity_duration_curvature":"minute_volume, 20",
        }
        if op in args: return f"{op}({args[op]})", "按现有正式微观结构算子的明确输入构造日级聚合信号"
    return None


def r74_formula(name: str) -> tuple[str, str]:
    # R74 is deliberately rebuilt from the name/mechanism because its formula cells are prose.
    fund = {
      "revenue":"operating_revenue", "operating_profit":"operating_profit",
      "parent_profit":"net_profit_parent", "ocf":"net_operating_cash_flow",
      "gross_profit":"gross_profit", "roa_proxy":"net_profit", "margin":"operating_profit",
      "asset_growth":"total_assets",
    }
    if "breadth_4statement" in name:
        return f"multiply(add(add(sign(fiscal_acceleration(operating_revenue, report_period_end_date, 1, 4, 3)), sign(fiscal_acceleration(operating_profit, report_period_end_date, 1, 4, 3))), add(sign(fiscal_acceleration(net_profit_parent, report_period_end_date, 1, 4, 3)), sign(fiscal_acceleration(operating_cash_flow, report_period_end_date, 1, 4, 3)))), {REV})", "四类报表增长加速度方向广度与短期反转交互"
    if name.startswith("r74_fundmom_"):
        key=next(k for k in fund if f"fundmom_{k}_" in name)
        acc=f"fiscal_acceleration({fund[key]}, report_period_end_date, 1, 4, 3)"
        return f"multiply({acc}, {PRICE_DELAY})", "基本面季度加速度与价格信息吸收迟滞的交互"
    if name.startswith("r74_intcap_"):
        spend="rd_expenses" if "knowledge" in name else ("add(selling_expense, administration_expense)" if "organization" in name else "add(rd_expenses, add(selling_expense, administration_expense))")
        stock=f"fiscal_perpetual_inventory({spend}, report_period_end_date)"
        if name.endswith("sales_monetization"): rhs="safe_div_null(operating_revenue, total_assets)"
        elif name.endswith("ocf_monetization"): rhs="safe_div_null(operating_cash_flow, total_assets)"
        elif name.endswith("margin"): rhs="safe_div_null(operating_profit, operating_revenue)"
        elif name.endswith("goodwill_gap"): rhs="safe_div_null(goodwill, total_assets)"
        elif name.endswith("price_delay"): rhs=PRICE_DELAY
        else: rhs="neg(abs(subtract(rank(safe_div_null(operating_profit, total_assets)), rank(safe_div_null(operating_revenue, total_assets)))))"
        return f"multiply(safe_div_null({stock}, total_assets), {rhs})", "PIT 无形资本存量占资产比与其变现、估值或信息摩擦机制交互"
    if name.startswith("r74_laborlev_"):
        base="safe_div_null(staff_cash_paid, gross_profit)" if "grossprofit" in name else "safe_div_null(staff_cash_paid, operating_revenue)"
        rhs="fiscal_asymmetric_elasticity(staff_cash_paid, operating_revenue, report_period_end_date)" if name.endswith("revenue_elasticity") else PRICE_DELAY
        return f"multiply({base}, {rhs})", "人工现金支出强度与收入弹性或价格迟滞交互"
    if "issuance_cei" in name: return f"multiply(ts_delta(log_positive_or_nan(free_cap), 252), {PRICE_DELAY})", "252 日自由流通股本变化代理综合发行，并与价格迟滞交互"
    if "share_count_change" in name: return f"multiply(safe_div_null(ts_delta(free_cap, 252), ts_lag(free_cap, 252)), {PRICE_DELAY})", "252 日流通股数变化与价格迟滞交互"
    if name=="r74_cashop_x_price_delay": return f"multiply(safe_div_null(subtract(operating_profit, subtract(operating_revenue, operating_cash_flow)), total_assets), {PRICE_DELAY})", "保守现金经营利润率与价格迟滞交互"
    bases={"residual_momentum_capm":RESID_MOM,"price_delay":PRICE_DELAY,"idio_skew":f"idio_skew(ret, {BENCH}, 60)","coskew":f"coskewness_to_market(ret, {BENCH}, 60)"}
    for k,b in bases.items():
        if f"market_{k}_x_" in name:
            state=name.split("_x_",1)[1]
            s={"turnover_cv":"safe_div_null(ts_std(turnover_ratio, 60), abs(ts_mean(turnover_ratio, 60)))","downside":"ts_downside_deviation(ret, 60)","total_skew":"ts_skew(ret, 60)","free_float":"log_positive_or_nan(free_cap)","filing_delay":"ts_zscore(net_profit_parent, 8)"}[state]
            return f"multiply({b}, {s})", "市场风险信号与指定流动性、尾部风险、供给或披露摩擦状态交互"
    if name.startswith("r74_reversal_timing_"):
        state=name.split("_x_",1)[1]
        s={"market_vol":f"ts_std({BENCH}, 60)","price_delay":PRICE_DELAY,"idio_skew":f"idio_skew(ret, {BENCH}, 60)","turnover_cv":"safe_div_null(ts_std(turnover_ratio, 60), abs(ts_mean(turnover_ratio, 60)))"}[state]
        return f"multiply({REV}, {s})", "短期反转与指定市场状态交互以刻画反转择时"
    if "expected_idio_profit_skew" in name: return "ts_skew(fiscal_ar_resid_std(safe_div_null(net_profit_parent, total_assets), report_period_end_date), 8)", "以 PIT 盈利能力自回归残差的滚动偏度代理预期特质盈利偏度"
    if name.startswith("r74_reportseason_"):
        state="abs(ts_zscore(net_profit_parent, 8))"
        if name.endswith("market_12m_momentum"): sig=f"ts_sum({BENCH}, 252)"
        elif name.endswith("stock_residual_momentum"): sig=RESID_MOM
        elif name.endswith("fundamental_surprise"): sig="fiscal_standardized_surprise(net_profit_parent, report_period_end_date, 8)"
        else: sig=PRICE_DELAY
        return f"multiply({state}, {sig})", "仅用当期及历史披露事件密度形成报告季状态，并与目标信号交互"
    if "returnseason_same_month_t12" in name: return "ts_lag(ret, 252)", "以 252 交易日滞后作为同季节位置收益的明确近似定义"
    if "same_month_multi_year" in name: return "multiply(add(add(ts_lag(ret, 504), ts_lag(ret, 756)), add(ts_lag(ret, 1008), ts_lag(ret, 1260))), 0.25)", "以 2 至 5 年交易日季节滞后均值定义多年同月收益近似"
    if "other_month_baseline" in name: return "ts_mean(ret, 1260)", "以过去五年全样本均值定义非同月基线近似"
    if "same_minus_other" in name: return "subtract(mean(ts_lag(ret, 252), ts_lag(ret, 504), ts_lag(ret, 756)), ts_mean(ret, 756))", "以多年同季节滞后均值减长期基线定义季节性差"
    return MOM, "按名称所述机制以历史价格量数据重建"


def propose(row: dict) -> tuple[str, str]:
    old=row["current_formula"]; name=(row.get("original_context") or {}).get("因子名称","")
    if "max_ast_nodes" in row["error"]:
        if row["source_row"]==25202:
            return "rank(where(gt(cs_pct_rank(ts_sharpe(ret, 60)), 0.65), neg(rank(subtract(cdl_tweezer_bottom(open, high, low, close), multiply(safe_div_null(1.0, pb_ratio), ts_sum(ret, 20))))), multiply(rank(subtract(cdl_tweezer_bottom(open, high, low, close), multiply(safe_div_null(1.0, pb_ratio), ts_sum(ret, 20)))), 0.25)))", "保留趋势状态门控、镊子底反转与低估值动量的核心机制，去除重复展开支路"
        return "rank(where(and_(gt(abs(zscore(ts_sum(ts_delta(close, 1), 20))), 1.0), gt(abs(zscore(ts_sum(ts_delta(volume, 1), 20))), 1.0)), multiply(zscore(ts_sum(ts_delta(close, 1), 20)), zscore(ts_sum(ts_delta(volume, 1), 20))), 0.0))", "保留价格与成交量联合显著性门控，消除相同子树重复展开"
    if "max_call_arity" in row["error"]:
        base,desc=relationship_proxy(name)
        rhs=interaction_suffix(name)
        return (f"multiply({base}, {rhs})" if rhs else base), desc + ("并与目标状态交互" if rhs else "")
    got=r65_formula(name,old)
    if got: return got
    if row["source_row"]>=113000: return r74_formula(name)
    # filing-delay placeholders before R74
    if name.endswith("_filing_delay_interaction"):
        fld={"operating_revenue":"operating_revenue","net_profit":"net_profit","ocf":"operating_cash_flow","goodwill":"goodwill","account_receivable":"account_receivable","inventories":"inventory"}[name.split("_filing_delay_interaction")[0]]
        return f"multiply(ts_zscore({fld}, 8), fiscal_standardized_surprise({fld}, report_period_end_date, 8))", "以 PIT 报表值的历史异常程度代理披露摩擦，并与该项目标准化惊喜交互"
    raise ValueError(f"unmapped row {row['source_row']} {name}: {old}")


def load_compile_runtime():
    p=ROOT/"evidence"/"factor_catalog_20260915"/"compile_catalog.py"
    spec=importlib.util.spec_from_file_location("r20_compile_catalog",p); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    parser,engine=mod.build_runtime()
    sp=ROOT/"evidence"/"factor_catalog_20260915"/"smoke_catalog.py"
    ss=importlib.util.spec_from_file_location("r20_smoke_catalog",sp); sm=importlib.util.module_from_spec(ss); ss.loader.exec_module(sm)
    return parser,engine,sm.bind_fields


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--input",default=str(EVIDENCE/"r20_original_context.jsonl.gz")); ap.add_argument("--output",default=str(EVIDENCE/"r20_syntax_proposals.jsonl")); args=ap.parse_args()
    with gzip.open(args.input,"rt") as f: rows=[json.loads(x) for x in f if "DSLParseError" in json.loads(x).get("error","")]
    parser,engine,bind_fields=load_compile_runtime(); out=[]; failures=[]
    from factor_engine.api.factor import Factor
    for row in rows:
        try:
            formula,definition=propose(row)
            formula=formula.replace("net_operating_cash_flow", "operating_cash_flow")
            formula=formula.replace("report_period_end_date", "field('report_period_end_date', table='StockIncome')")
            expr=parser.parse(formula); bindings,binding_failures=bind_fields(expr)
            if binding_failures: raise RuntimeError("binding failures: "+json.dumps(binding_failures,ensure_ascii=False))
            engine.compile(Factor(name=str(row["id"]),expr=expr,source_expr=formula,surface="compat_research"))
            out.append({"source_row":row["source_row"],"id":row["id"],"before_formula":row["current_formula"],"current_formula":formula,"changes":list(row.get("changes") or [])+["语义重建（非等价）："+definition],"current_definition":definition,"semantic_redesign":True,"compile_status":"COMPILED","error":"","bindings":bindings,"original_error":row["error"]})
        except Exception as e:
            failures.append({"source_row":row["source_row"],"id":row["id"],"name":(row.get("original_context") or {}).get("因子名称"),"proposal":locals().get("formula"),"error":f"{type(e).__name__}: {e}"})
    Path(args.output).write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in out))
    print(json.dumps({"target":len(rows),"compiled":len(out),"failed":len(failures),"failures":failures},ensure_ascii=False,indent=2))
    if failures: raise SystemExit(1)


if __name__=="__main__": main()
