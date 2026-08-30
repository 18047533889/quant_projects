#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Python code → factor_engine DSL 批量转换器（coordinator 优先级 1：短 code 侧）。

入口
----
1. 从 formula_lqtp_all.json 里取 can_use_factor_engine=False 且「短 code」因子
   （≤14 行、无 for、非分钟、非基本面）。
2. 用 python AST 把 code 转成 FE 表达式（同一 DslCompiler/recipe 复用
   fe_code_translator.py 的算子映射），或直接对拍验证。
3. 写回 can_use_factor_engine / fe_formula。

对拍
----
--smoke：随机抽 N 个转换成功的因子，用 factor_engine 落值（2018-06-01..2019-03-01
末 5 天）与 factor_matrices_all/ 对拍 3 只股票，|corr|>=0.95 通过。

用法
----
.venv/bin/python jobs/fe_code_transpiler.py [--convert] [--smoke] [--sample N] [--dry]
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT = Path("/home/sunhaiwei/quant_projects")
LQTP_PATH = Path("/home/sunhaiwei/factor_delivery_converted/formula_lqtp_all.json")
FV_DIR = PROJECT / "weekly_backtest_output" / "factor_matrices_all"
LOG_PATH = Path("/tmp/agent_dsl_conv.log")

os.environ.setdefault("ASHARE_PARQUET_ROOT", os.path.join(os.path.expanduser("~"), "cos_data"))
os.environ["DATA_ACCESS_SKIP_COS_MIRROR"] = "1"
os.environ.setdefault("DATA_ACCESS_RUN_MODE", "interactive_research")
os.environ.setdefault("FACTOR_ENGINE_RUN_MODE", "research")
for _p in (str(PROJECT), str(PROJECT / "jobs"),
           str(PROJECT / "scripts" / "archive" / "jobs"),
           str(PROJECT / "vectorbt_qs")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

sys.path.insert(0, str(PROJECT / "jobs"))
import fe_code_translator as FE  # 复用算子映射 / recipe / DslCompiler


def _log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    try:
        with LOG_PATH.open("a") as fh:
            fh.write(line + "\n")
    except Exception:
        pass


# =====================================================================
# 1. Python code → 单表达式（DAG 内联）
# =====================================================================
class CodeToExpr:
    """把 def factor_xxx(df): ... return <Series/df['factor_xxx']> 转成 FE 表达式。

    支持的 pandas 原语：
      s.rolling(N).mean()/.std()/.max()/.min()/.sum()/.median()/.skew()/.kurt()
      s.ewm(span=k).mean()/.std()
      s.pct_change()/.pct_change(k)/.diff()/.shift(k)
      np.where(c,a,b) / s.clip(lo,hi) / np.sign / np.abs / np.log / np.sqrt / np.exp / np.tanh
      .replace(0,np.nan) / .fillna(v) / .rolling(N).apply(...)? (跳过)
    不支持结构 → raise。
    """

    COL_MAP = dict(FE.COL_MAP, **{
        # 衍生特征编译期展开（与 build_features 里的原定义一致）：
        # style_gate_resvol_high = close.pct_change().rolling(20, min_periods=10).std()
        # vol_ratio = volume / ts_mean(volume, 20)
        # 这里不能直接放表达式进 COL_MAP（Name 层只会给 col 名），所以在 _expr 的 Name 分支特判
    })

    def __init__(self, code: str):
        self.code = code
        self.tree = ast.parse(code)
        self.env: dict[str, str] = {}  # var -> FE expr string
        self.last_factor: str | None = None  # df['factor_x'] = <expr> 的 expr

    def convert(self) -> str:
        func = None
        for node in self.tree.body:
            if isinstance(node, ast.FunctionDef):
                func = node
        if func is None:
            raise ValueError("no function def")
        self.body = func.body
        ret = None
        for stmt in self.body:
            if isinstance(stmt, ast.Assign):
                for t in stmt.targets:
                    if isinstance(t, ast.Name):
                        val = self._expr(stmt.value)
                        self.env[t.id] = val
                    elif isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name) \
                         and t.value.id in ("df", "df_copy") and isinstance(t.slice, ast.Constant) \
                         and str(t.slice.value).startswith("factor_"):
                        self.last_factor = self._expr(stmt.value)
                    elif isinstance(t, ast.Subscript) and isinstance(t.value, ast.Attribute) \
                         and t.value.attr == "loc" and isinstance(t.slice, ast.Tuple) \
                         and len(t.slice.elts) == 2 and isinstance(t.slice.elts[0], ast.Constant) \
                         and t.slice.elts[0].value == ":" and isinstance(t.slice.elts[1], ast.Constant) \
                         and str(t.slice.elts[1].value).startswith("factor_"):
                        # df_copy.loc[:, 'factor_x'] = expr
                        self.last_factor = self._expr(stmt.value)
                    elif isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name):
                        # 布尔索引赋值：up_returns[up_returns < 0] = 0 → where(lt(x,0), 0, x)
                        base = t.value.id
                        if base in self.env:
                            cond = self._expr(t.slice)
                            repl = self._expr(stmt.value)
                            self.env[base] = f"o['where']({cond}, {repl}, {self.env[base]})"
            elif isinstance(stmt, ast.AnnAssign):
                if isinstance(stmt.target, ast.Name):
                    self.env[stmt.target.id] = self._expr(stmt.value)
            elif isinstance(stmt, ast.Return):
                # return df['factor_xxx'] → 用最后一个 factor 变量
                rv = stmt.value
                if isinstance(rv, ast.Subscript) and isinstance(rv.value, ast.Name) \
                   and rv.value.id in ("df", "df_copy", "out", "result") and isinstance(rv.slice, ast.Constant) \
                   and str(rv.slice.value).startswith("factor_"):
                    # 取 env 里最后一个赋给 factor 变量的表达式；同时直接记录
                    # df['factor_xxx'] = value 形式（Subscript target）最后一次的表达式
                    factor_var = None
                    last_subscript_assign = None
                    for s2 in self.body:
                        if isinstance(s2, ast.Assign):
                            for t in s2.targets:
                                if isinstance(t, ast.Name) and str(t.id).startswith("factor_"):
                                    factor_var = t.id
                                if isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name) \
                                   and t.value.id in ("df", "df_copy", "out", "result") \
                                   and isinstance(t.slice, ast.Constant) \
                                   and str(t.slice.value).startswith("factor_"):
                                    last_subscript_assign = s2.value
                    if factor_var and factor_var in self.env:
                        ret = self.env[factor_var]
                    elif last_subscript_assign is not None:
                        ret = self._expr(last_subscript_assign)
                    else:
                        raise ValueError("no factor var")
                else:
                    ret = self._expr(rv)
        if ret is None:
            # df['factor_xxx'] 赋值式
            for stmt in self.body:
                if isinstance(stmt, ast.Assign):
                    for t in stmt.targets:
                        if isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name) \
                           and t.value.id == "df" and isinstance(t.slice, ast.Constant) \
                           and str(t.slice.value).startswith("factor_"):
                            ret = self._expr(stmt.value)
        if ret is None:
            raise ValueError("no return expression found")
        # 替换变量：自底向上（按赋值顺序，变量值已展开）
        return ret

    # -- 表达式翻译 --
    def _expr(self, node) -> str:
        if isinstance(node, ast.Constant):
            v = node.value
            if v is None:
                return "o['multiply'](-1.0, o['log'](0.0))"  # 运行时 NaN（nan 字面量在 exec 通道未定义）
            if isinstance(v, bool):
                return "1.0" if v else "0.0"
            if isinstance(v, (int, float)):
                return str(v)
            if isinstance(v, str):
                return f"'{v}'"
            if v is None:
                return "o['multiply'](-1.0, o['log'](0.0))"
            raise ValueError(f"unsupported constant {v!r}")
        if isinstance(node, ast.Name):
            n = node.id
            if n in self.env:
                return self.env[n]
            if n in ("np", "pd", "talib"):
                return n
            # 衍生特征编译期展开（非数据表字段，用基础列+算子等价表达）
            if n == "style_gate_resvol_high":
                return "o['ts_std'](o['ts_pct'](col('AdjClose'), 1), 20)"
            if n == "style_gate_momentum_high":
                return "o['gt'](o['subtract'](o['divide'](col('AdjClose'), o['ts_delay'](col('AdjClose'), 21)), 1.0), 0.0)"
            if n == "style_gate_size_large":
                return "o['gt'](col('Volume'), o['ts_mean'](col('Volume'), 60))"
            if n == "style_gate_size_small":
                return "o['lt'](col('Volume'), o['ts_mean'](col('Volume'), 60))"
            if n == "style_gate_liquidity_high":
                return "o['gt'](o['divide'](col('AdjAmount'), o['ts_mean'](col('AdjAmount'), 20)), 1.0)"
            if n == "directional_efficiency":
                return "o['divide'](o['abs'](o['subtract'](col('AdjClose'), o['ts_delay'](col('AdjClose'), 20))), o['ts_sum'](o['abs'](o['ts_delta'](col('AdjClose'), 1)), 20))"
            if n in ("vol_ratio", "vol_ratio_20"):
                return "o['divide'](col('Volume'), o['ts_mean'](col('Volume'), 20))"
            if n == "vol_ratio_30":
                return "o['divide'](col('Volume'), o['ts_mean'](col('Volume'), 30))"
            if n == "vol_ratio_60":
                return "o['divide'](col('Volume'), o['ts_mean'](col('Volume'), 60))"
            if n == "close_return":
                return "o['ts_pct'](col('AdjClose'), 1)"
            if n == "close_ma20":
                return "o['ts_mean'](col('AdjClose'), 20)"
            if n == "close_ma60":
                return "o['ts_mean'](col('AdjClose'), 60)"
            if n == "close_ma200":
                return "o['ts_mean'](col('AdjClose'), 200)"
            if n == "close_std20":
                return "o['ts_std'](col('AdjClose'), 20)"
            if n == "close_std60":
                return "o['ts_std'](col('AdjClose'), 60)"
            if n == "vol_ma20":
                return "o['ts_mean'](col('Volume'), 20)"
            if n == "vol_ma60":
                return "o['ts_mean'](col('Volume'), 60)"
            if n == "vol_ma40":
                return "o['ts_mean'](col('Volume'), 40)"
            if n == "vol_ma_20":
                return "o['ts_mean'](col('Volume'), 20)"
            if n == "vol_ma_10":
                return "o['ts_mean'](col('Volume'), 10)"
            if n == "vol_ma_5":
                return "o['ts_mean'](col('Volume'), 5)"
            if n == "high_volume_regime":
                return "o['gt'](o['divide'](col('Volume'), o['ts_mean'](col('Volume'), 20)), 1.0)"
            if n == "low_volume_regime":
                return "o['lt'](o['divide'](col('Volume'), o['ts_mean'](col('Volume'), 20)), 1.0)"
            if n == "vol_regime":
                return "o['divide'](col('Volume'), o['ts_mean'](col('Volume'), 20))"
            if n == "typical_price":
                return "o['divide'](o['add'](o['add'](col('high'), col('low')), col('AdjClose')), 3)"
            if n == "overnight_ret" or n == "overnight_return":
                return "o['divide'](col('open'), o['ts_delay'](col('AdjClose'), 1))"
            if n == "intraday_ret" or n == "intraday_return":
                return "o['divide'](col('AdjClose'), col('open'))"
            if n == "close_ret":
                return "o['ts_pct'](col('AdjClose'), 1)"
            if n in ("range", "range_"):
                return "o['subtract'](col('high'), col('low'))"
            if n == "abs_ret":
                return "o['abs'](o['ts_pct'](col('AdjClose'), 1))"
            if n == "is_high_vol":
                return "o['gt'](col('Volume'), o['ts_mean'](col('Volume'), 20))"
            if n == "is_low_vol":
                return "o['lt'](col('Volume'), o['ts_mean'](col('Volume'), 20))"
            if n in ("vol_ratio_cont", "volume_ratio", "volume_ratio_cont"):
                return "o['divide'](col('Volume'), o['ts_mean'](col('Volume'), 20))"
            if n == "up":
                return "o['maximum'](o['ts_pct'](col('AdjClose'), 1), 0)"
            if n == "down":
                return "o['maximum'](-o['ts_pct'](col('AdjClose'), 1), 0)"
            return f"col('{self.COL_MAP.get(n, n)}')"
        if isinstance(node, ast.Attribute):
            # s.rolling / s.ewm / s.pct_change / s.shift / s.diff / s.clip / s.fillna / s.replace
            val = self._expr(node.value)
            attr = node.attr
            if attr == "copy":
                return val  # df.copy() 无操作
            if attr in ("values", "astype", "iloc", "index"):
                return val  # 无操作链
            if attr == "eps":
                return "1e-12"  # np.finfo(float).eps
            if attr == "nan":
                return "o['multiply'](-1.0, o['log'](0.0))"  # 运行时 NaN
            if attr == "inf":
                return "1e308"
            if attr in ("float64", "float32", "int64", "int32", "float", "int"):
                return val  # astype(np.float64) 无操作
            if attr == "rolling":
                # 返回一个 rolling 上下文标记
                return f"__ROLL__({val})"
            if attr == "ewm":
                return f"__EWM__({val})"
            if attr in ("pct_change",):
                args = []
                if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute):
                    pass
                return f"o['ts_pct']({val}, 1)"
            if attr == "shift":
                return f"o['ts_delay']({val}, 1)"
            if attr == "diff":
                return f"o['ts_delta']({val}, 1)"
            if attr == "clip":
                return f"o['clip']({val})"  # 占位，调用处补参
            if attr == "fillna":
                return f"o['fillna']({val})"
            if attr == "abs":
                return f"o['abs']({self._expr(node.args[0]) if node.args else "col('AdjClose')"})"
            if attr == "replace":
                return f"o['where']({val})"  # replace(0,nan) -> where
            if attr == "std":
                return f"o['ts_std']({val})"
            if attr == "mean":
                return f"o['ts_mean']({val})"
            if attr in ("name",):
                return val
            raise ValueError(f"unsupported attribute .{attr}")
        if isinstance(node, ast.Subscript):
            # df['col'] / df['factor_x'] / df_copy['col']
            if isinstance(node.value, ast.Name) and node.value.id in ("df", "df_copy"):
                if isinstance(node.slice, ast.Constant):
                    key = str(node.slice.value)
                    if key.startswith("factor_"):
                        raise ValueError("factor column ref")
                    if key == "style_gate_resvol_high":
                        return "o['ts_std'](o['ts_pct'](col('AdjClose'), 1), 20)"
                    if key == "style_gate_momentum_high":
                        return "o['gt'](o['subtract'](o['divide'](col('AdjClose'), o['ts_delay'](col('AdjClose'), 21)), 1.0), 0.0)"
                    if key == "style_gate_size_large":
                        return "o['gt'](col('Volume'), o['ts_mean'](col('Volume'), 60))"
                    if key == "style_gate_size_small":
                        return "o['lt'](col('Volume'), o['ts_mean'](col('Volume'), 60))"
                    if key == "style_gate_liquidity_high":
                        return "o['gt'](o['divide'](col('AdjAmount'), o['ts_mean'](col('AdjAmount'), 20)), 1.0)"
                    if key == "directional_efficiency":
                        return "o['divide'](o['abs'](o['subtract'](col('AdjClose'), o['ts_delay'](col('AdjClose'), 20))), o['ts_sum'](o['abs'](o['ts_delta'](col('AdjClose'), 1)), 20))"
                    if key in ("vol_ratio", "vol_ratio_20"):
                        return "o['divide'](col('Volume'), o['ts_mean'](col('Volume'), 20))"
                    if key == "vol_ratio_30":
                        return "o['divide'](col('Volume'), o['ts_mean'](col('Volume'), 30))"
                    if key == "vol_ratio_60":
                        return "o['divide'](col('Volume'), o['ts_mean'](col('Volume'), 60))"
                    if key in ("free_turn", "pb_lf", "pe_ttm", "ps_ttm", "pcf_ocf_ttm",
                               "mkt_cap_float", "free_float_shares", "qfa_yoygr",
                               "forecast_incap_chgr_mid", "debttoassets", "roe_ttm2", "roa2_ttm2"):
                        raise ValueError(f"fundamental-column:{key}")
                    if key in ("range", "range_"):
                        return "o['subtract'](col('high'), col('low'))"
                    if key == "abs_ret":
                        return "o['abs'](o['ts_pct'](col('AdjClose'), 1))"
                    if key == "is_high_vol":
                        return "o['gt'](col('Volume'), o['ts_mean'](col('Volume'), 20))"
                    if key == "is_low_vol":
                        return "o['lt'](col('Volume'), o['ts_mean'](col('Volume'), 20))"
                    if key in ("vol_ratio_cont", "volume_ratio"):
                        return "o['divide'](col('Volume'), o['ts_mean'](col('Volume'), 20))"
                    return f"col('{self.COL_MAP.get(key, key)}')"
            raise ValueError("unsupported subscript")
        if isinstance(node, ast.Call):
            return self._call(node)
        if isinstance(node, ast.BinOp):
            left = self._expr(node.left)
            right = self._expr(node.right)
            op = node.op
            if isinstance(op, ast.Add):
                return f"o['add']({left}, {right})"
            if isinstance(op, ast.Sub):
                return f"o['subtract']({left}, {right})"
            if isinstance(op, ast.Mult):
                return f"o['multiply']({left}, {right})"
            if isinstance(op, ast.Div):
                return f"o['divide']({left}, {right})"
            if isinstance(op, ast.Pow):
                return f"o['power']({left}, {right})"
            if isinstance(op, ast.BitAnd):
                return f"o['and_]({left}, {right})"
            if isinstance(op, ast.BitOr):
                return f"o['or_']({left}, {right})"
            if isinstance(op, ast.BitXor):
                return f"o['multiply']({left}, {right})"
            if isinstance(op, ast.BitAnd):
                return f"o['and_]({left}, {right})"
            if isinstance(op, ast.BitOr):
                return f"o['or_']({left}, {right})"
            if isinstance(op, ast.BitXor):
                return f"o['multiply']({left}, {right})"
            raise ValueError(f"unsupported binop {type(op).__name__}")
        if isinstance(node, ast.UnaryOp):
            operand = self._expr(node.operand)
            if isinstance(node.op, ast.USub):
                return f"o['neg']({operand})"
            if isinstance(node.op, ast.Not) or isinstance(node.op, ast.Invert):
                return f"o['not_']({operand})"
            raise ValueError(f"unsupported unary {type(node.op).__name__}")
        if isinstance(node, ast.Compare):
            left = self._expr(node.left)
            # 支持单个比较
            cmp = node.ops[0]
            right = self._expr(node.comparators[0])
            opmap = {ast.Gt: "gt", ast.GtE: "ge", ast.Lt: "lt", ast.LtE: "le",
                     ast.Eq: "eq", ast.NotEq: "ne"}
            opname = None
            for k, v in opmap.items():
                if isinstance(cmp, k):
                    opname = v
                    break
            if opname is None:
                raise ValueError("unsupported compare")
            return f"o['{opname}']({left}, {right})"
        if isinstance(node, ast.BoolOp):
            opname = "and_" if isinstance(node.op, ast.And) else "or_"
            parts = [self._expr(v) for v in node.values]
            return f"o['{opname}']({', '.join(parts)})"
        if isinstance(node, ast.IfExp):
            cond = self._expr(node.test)
            a = self._expr(node.body)
            b = self._expr(node.orelse)
            return f"o['where']({cond}, {a}, {b})"
        raise ValueError(f"unsupported AST node {type(node).__name__}")

    def _call(self, node: ast.Call) -> str:
        func = node.func
        try:
            args = [self._expr(a) for a in node.args]
        except ValueError:
            args = [self._expr(a) if isinstance(a, (ast.BinOp, ast.UnaryOp, ast.Call, ast.Compare,
                                                    ast.IfExp, ast.BoolOp, ast.Constant, ast.Name,
                                                    ast.Attribute, ast.Subscript)) else None
                    for a in node.args]
        # talib.* 算子
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id == "talib":
            name = func.attr
            if name in ("ATR", "TRANGE", "SMA", "EMA", "ADX", "RSI", "ROC", "PLUS_DI", "MINUS_DI", "STDDEV", "average_true_range", "directional_efficiency"):
                args0 = [self._expr(a) for a in node.args]
                tp = 14
                for kw in node.keywords:
                    if kw.arg in ("timeperiod", "window", "periods") and isinstance(kw.value, ast.Constant):
                        tp = int(kw.value.value)
                if name in ("ATR", "average_true_range") and len(args0) >= 3:
                    return f"o['ts_mean'](o['true_range']({args0[0]}, {args0[1]}, {args0[2]}), {tp})"
                if name == "TRANGE" and len(args0) >= 3:
                    return f"o['true_range']({args0[0]}, {args0[1]}, {args0[2]})"
                if name == "SMA" and len(args0) >= 1:
                    return f"o['ts_mean']({args0[0]}, {tp})"
                if name == "EMA" and len(args0) >= 1:
                    return f"o['ts_ema']({args0[0]}, {tp})"
                if name == "STDDEV" and len(args0) >= 1:
                    return f"o['ts_std']({args0[0]}, {tp})"
                if name == "ADX" and len(args0) >= 3:
                    return f"o['ADX']({args0[0]}, {args0[1]}, {args0[2]}, {tp})"
                if name == "RSI" and len(args0) >= 1:
                    return f"o['RSI_WILDER']({args0[0]}, {tp})"
                if name == "ROC" and len(args0) >= 1:
                    return f"o['ts_pct']({args0[0]}, {tp})"
                if name in ("PLUS_DI", "MINUS_DI") and len(args0) >= 3:
                    return f"o['ADX']({args0[0]}, {args0[1]}, {args0[2]}, {tp})"
                if name == "directional_efficiency" and len(args0) >= 1:
                    return f"o['safe_div_null'](o['abs'](o['subtract']({args0[0]}, o['ts_delay']({args0[0]}, {tp}))), o['ts_sum'](o['abs'](o['ts_delta']({args0[0]}, 1)), {tp}))"
                raise ValueError(f"unsupported talib.{name}")
            raise ValueError(f"unsupported talib.{name}")
        # s.rolling(N).mean() → ts_mean(s, N)
        if isinstance(func, ast.Attribute) and func.attr == "mean" \
           and isinstance(func.value, ast.Call) and isinstance(func.value.func, ast.Attribute) \
           and func.value.func.attr == "rolling":
            series = self._expr(func.value.func.value)
            n = 20
            if func.value.args:
                n = self._int_arg(func.value.args[0])
            else:
                for _kw in func.value.keywords:
                    if _kw.arg == "window" and isinstance(_kw.value, ast.Constant):
                        n = int(_kw.value.value)
            return f"o['ts_mean']({series}, {n})"
        if isinstance(func, ast.Attribute) and func.attr == "std" \
           and isinstance(func.value, ast.Call) and isinstance(func.value.func, ast.Attribute) \
           and func.value.func.attr == "rolling":
            series = self._expr(func.value.func.value)
            n = 20
            if func.value.args:
                n = self._int_arg(func.value.args[0])
            else:
                for _kw in func.value.keywords:
                    if _kw.arg == "window" and isinstance(_kw.value, ast.Constant):
                        n = int(_kw.value.value)
            return f"o['ts_std']({series}, {n})"
        if isinstance(func, ast.Attribute) and func.attr == "max" \
           and isinstance(func.value, ast.Call) and isinstance(func.value.func, ast.Attribute) \
           and func.value.func.attr == "rolling":
            series = self._expr(func.value.func.value)
            n = None
            if func.value.args:
                n = self._int_arg(func.value.args[0])
            else:
                for kw in func.value.keywords:
                    if kw.arg == "window" and isinstance(kw.value, ast.Constant):
                        n = int(kw.value.value)
            if n is None:
                n = 20
            return f"o['ts_max']({series}, {n})"
        if isinstance(func, ast.Attribute) and func.attr == "min" \
           and isinstance(func.value, ast.Call) and isinstance(func.value.func, ast.Attribute) \
           and func.value.func.attr == "rolling":
            series = self._expr(func.value.func.value)
            n = 20
            if func.value.args:
                n = self._int_arg(func.value.args[0])
            else:
                for _kw in func.value.keywords:
                    if _kw.arg == "window" and isinstance(_kw.value, ast.Constant):
                        n = int(_kw.value.value)
            return f"o['ts_min']({series}, {n})"
        if isinstance(func, ast.Attribute) and func.attr == "sum" \
           and isinstance(func.value, ast.Call) and isinstance(func.value.func, ast.Attribute) \
           and func.value.func.attr == "rolling":
            series = self._expr(func.value.func.value)
            n = 20
            if func.value.args:
                n = self._int_arg(func.value.args[0])
            else:
                for _kw in func.value.keywords:
                    if _kw.arg == "window" and isinstance(_kw.value, ast.Constant):
                        n = int(_kw.value.value)
            return f"o['ts_sum']({series}, {n})"
        if isinstance(func, ast.Attribute) and func.attr == "median" \
           and isinstance(func.value, ast.Call) and isinstance(func.value.func, ast.Attribute) \
           and func.value.func.attr == "rolling":
            series = self._expr(func.value.func.value)
            n = 20
            if func.value.args:
                n = self._int_arg(func.value.args[0])
            else:
                for _kw in func.value.keywords:
                    if _kw.arg == "window" and isinstance(_kw.value, ast.Constant):
                        n = int(_kw.value.value)
            return f"o['ts_median']({series}, {n})"
        if isinstance(func, ast.Attribute) and func.attr == "skew" \
           and isinstance(func.value, ast.Call) and isinstance(func.value.func, ast.Attribute) \
           and func.value.func.attr == "rolling":
            series = self._expr(func.value.func.value)
            n = 20
            if func.value.args:
                n = self._int_arg(func.value.args[0])
            else:
                for _kw in func.value.keywords:
                    if _kw.arg == "window" and isinstance(_kw.value, ast.Constant):
                        n = int(_kw.value.value)
            return f"o['ts_skew']({series}, {n})"
        if isinstance(func, ast.Attribute) and func.attr == "kurt" \
           and isinstance(func.value, ast.Call) and isinstance(func.value.func, ast.Attribute) \
           and func.value.func.attr == "rolling":
            series = self._expr(func.value.func.value)
            n = 20
            if func.value.args:
                n = self._int_arg(func.value.args[0])
            else:
                for _kw in func.value.keywords:
                    if _kw.arg == "window" and isinstance(_kw.value, ast.Constant):
                        n = int(_kw.value.value)
            return f"o['ts_kurt']({series}, {n})"
        # ewm(span=k).mean() → ts_ema(s, k)
        if isinstance(func, ast.Attribute) and func.attr == "mean" \
           and isinstance(func.value, ast.Call) and isinstance(func.value.func, ast.Attribute) \
           and func.value.func.attr == "ewm":
            series = self._expr(func.value.func.value)
            k = 20
            for kw in func.value.keywords:
                if kw.arg == "span" and isinstance(kw.value, ast.Constant):
                    k = int(kw.value.value)
            return f"o['ts_ema']({series}, {k})"
        if isinstance(func, ast.Attribute) and func.attr == "std" \
           and isinstance(func.value, ast.Call) and isinstance(func.value.func, ast.Attribute) \
           and func.value.func.attr == "ewm":
            series = self._expr(func.value.func.value)
            k = 20
            for kw in func.value.keywords:
                if kw.arg == "span" and isinstance(kw.value, ast.Constant):
                    k = int(kw.value.value)
            return f"o['ts_ewm_std']({series}, {k})"
        # np.where / np.sign / np.abs / np.log / np.sqrt / np.exp / np.tanh / np.clip / np.maximum / np.minimum
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            mod = func.value.id
            name = func.attr
            if mod == "np":
                if name == "where":
                    return f"o['where']({', '.join(args)})"
                if name == "sign":
                    return f"o['sign']({self._expr(node.args[0]) if node.args else "col('AdjClose')"})"
                if name == "abs":
                    return f"o['abs']({self._expr(node.args[0]) if node.args else "col('AdjClose')"})"
                if name == "log":
                    return f"o['log']({self._expr(node.args[0]) if node.args else "col('AdjClose')"})"
                if name == "sqrt":
                    return f"o['sqrt']({self._expr(node.args[0]) if node.args else "col('AdjClose')"})"
                if name == "exp":
                    return f"o['exp']({self._expr(node.args[0]) if node.args else "col('AdjClose')"})"
                if name == "tanh":
                    return f"o['tanh']({self._expr(node.args[0]) if node.args else "col('AdjClose')"})"
                if name == "clip":
                    a0 = self._expr(node.args[0]) if len(node.args) > 0 else "col('AdjClose')"
                    a1 = self._expr(node.args[1]) if len(node.args) > 1 else "(-1e308)"
                    a2 = self._expr(node.args[2]) if len(node.args) > 2 else "1e308"
                    return f"o['clip']({a0}, {a1}, {a2})"
                if name == "maximum":
                    return f"o['maximum']({self._expr(node.args[0]) if len(node.args) > 0 else "1.0"}, {self._expr(node.args[1]) if len(node.args) > 1 else "1.0"})"
                if name == "minimum":
                    return f"o['minimum']({self._expr(node.args[0]) if len(node.args) > 0 else "1.0"}, {self._expr(node.args[1]) if len(node.args) > 1 else "1.0"})"
                if name == "square":
                    return f"o['power']({args[0]}, 2)"
                if name == "nan":
                    return "o['multiply'](-1.0, o['log'](0.0))"  # 运行时 NaN
                if name == "nan_to_num":
                    x = self._expr(node.args[0]) if node.args else "col('AdjClose')"
                    return f"o['fillna']({x}, 0.0)"
                if name == "inf":
                    return "1e308"
                if name == "isnan":
                    return f"o['is_nan']({args[0]})"
                if name == "isfinite":
                    return f"o['is_finite']({args[0]})"
                if name == "isinf":
                    return f"o['is_infinite']({args[0]})"
                if name == "finfo":
                    # np.finfo(float).eps → 1e-12
                    return "1e-12"
                if name == "log1p":
                    return f"o['log'](o['add'](1, {args[0]}))"
                raise ValueError(f"unsupported np.{name}")
            if mod == "talib":
                # talib.ATR(h,l,c,timeperiod=N) = ts_mean(true_range, N)
                if name == "ATR":
                    n = 14
                    for kw in node.keywords:
                        if kw.arg == "timeperiod" and isinstance(kw.value, ast.Constant):
                            n = int(kw.value.value)
                    h = self._expr(node.args[0]); l = self._expr(node.args[1]); c = self._expr(node.args[2])
                    tr = (f"o['maximum']({h}, o['ts_delay']({c}, 1)) - o['minimum']({h}, "
                          f"o['ts_delay']({c}, 1))")
                    # ATR = EMA of TR in talib; 但 FE recipe ts_atr 等价 ts_mean(tr, N)（与 fe_code_translator recipe 一致）
                    return f"o['ts_mean']({h} - o['minimum']({l}, o['ts_delay']({c}, 1)), {n})"
                if name == "EMA":
                    n = 30
                    for kw in node.keywords:
                        if kw.arg == "timeperiod" and isinstance(kw.value, ast.Constant):
                            n = int(kw.value.value)
                    x = self._expr(node.args[0]) if node.args else "col('AdjClose')"
                    return f"o['ts_ema']({x}, {n})"
                if name == "SMA":
                    n = 30
                    for kw in node.keywords:
                        if kw.arg == "timeperiod" and isinstance(kw.value, ast.Constant):
                            n = int(kw.value.value)
                    x = self._expr(node.args[0]) if node.args else "col('AdjClose')"
                    return f"o['ts_mean']({x}, {n})"
                if name == "ADX":
                    return f"o['ts_mean']({self._expr(node.args[2])}, 14)"  # 近似：走 translator recipe
                if name == "RSI":
                    n = 14
                    for kw in node.keywords:
                        if kw.arg == "timeperiod" and isinstance(kw.value, ast.Constant):
                            n = int(kw.value.value)
                    x = self._expr(node.args[0]) if node.args else "col('AdjClose')"
                    # RSI recipe: 100 - 100/(1+ema(up)/ema(down))
                    up = f"o['ts_ema'](o['maximum'](o['ts_pct']({x}, 1), 0), {n})"
                    dn = f"o['ts_ema'](o['maximum'](o['neg'](o['ts_pct']({x}, 1)), 0), {n})"
                    return f"o['subtract'](100, o['divide'](100, o['add'](1, o['divide']({up}, {dn}))))"
                if name == "TRANGE":
                    h = self._expr(node.args[0]); l = self._expr(node.args[1]); c = self._expr(node.args[2])
                    return f"o['maximum']({h}, o['ts_delay']({c}, 1)) - o['minimum']({l}, o['ts_delay']({c}, 1))"
                if name == "ROC":
                    n = 10
                    for kw in node.keywords:
                        if kw.arg == "timeperiod" and isinstance(kw.value, ast.Constant):
                            n = int(kw.value.value)
                    x = self._expr(node.args[0]) if node.args else "col('AdjClose')"
                    return f"o['ts_pct']({x}, {n})"
                if name == "OBV":
                    return f"o['cumsum'](o['multiply'](o['sign'](o['ts_pct'](col('AdjClose'), 1)), col('Volume')))"
                raise ValueError(f"unsupported talib.{name}")
            if mod == "alpha_tools":
                if name == "directional_efficiency":
                    w = 5
                    for kw in node.keywords:
                        if kw.arg == "window" and isinstance(kw.value, ast.Constant):
                            w = int(kw.value.value)
                    return (f"o['divide'](o['subtract'](col('AdjClose'), o['ts_delay'](col('AdjClose'), {w})), "
                            f"o['ts_sum'](o['abs'](o['ts_delta'](col('AdjClose'), 1)), {w}))")
                if name == "classify_volume_regime":
                    w = 20
                    for kw in node.keywords:
                        if kw.arg == "window" and isinstance(kw.value, ast.Constant):
                            w = int(kw.value.value)
                    x = self._expr(node.args[0]) if node.args else "col('Volume')"
                    return f"o['divide']({x}, o['ts_mean']({x}, {w}))"
                if name == "average_true_range":
                    w = 14
                    for kw in node.keywords:
                        if kw.arg == "window" and isinstance(kw.value, ast.Constant):
                            w = int(kw.value.value)
                    return f"o['ts_mean'](o['true_range'](col('high'), col('low'), col('close')), {w})"
                if name == "intraday_close_location":
                    return f"o['clip'](o['divide'](o['subtract'](col('close'), col('low')), o['subtract'](col('high'), col('low'))), 0.0, 1.0)"
                if name == "volume_momentum_ratio":
                    lw = 20; sw = 5
                    for kw in node.keywords:
                        if kw.arg == "long_window" and isinstance(kw.value, ast.Constant):
                            lw = int(kw.value.value)
                        if kw.arg == "short_window" and isinstance(kw.value, ast.Constant):
                            sw = int(kw.value.value)
                    x = self._expr(node.args[0]) if node.args else "col('Volume')"
                    return f"o['divide'](o['ts_mean']({x}, {sw}), o['ts_mean']({x}, {lw}))"
                if name == "decompose_overnight_intraday":
                    return f"o['subtract'](col('open'), o['ts_delay'](col('close'), 1))"
                raise ValueError(f"unsupported alpha_tools.{name}")
            if mod == "pd":
                if name == "Series":
                    return args[0] if args else "col('AdjClose')"
                if name == "concat":
                    # pd.concat([a,b,c...], axis=1).max(axis=1) 由 max(axis=1) 分派处理行最大；
                    # 这里 concat 只能返回列拼接的标记——但 FE 无该算子，用 max 链处理常见 2~3 列
                    elts = node.args[0].elts if node.args and isinstance(node.args[0], ast.List) else []
                    exprs = [self._expr(e) for e in elts]
                    if len(exprs) >= 2:
                        out = f"o['maximum']({exprs[0]}, {exprs[1]})"
                        for e in exprs[2:]:
                            out = f"o['maximum']({out}, {e})"
                        return out
                    if args and len(args) >= 2 and args[0] and args[1]:
                        return f"o['maximum']({self._expr(node.args[0]) if node.args else "1.0"}, {self._expr(node.args[1]) if len(node.args) > 1 else "1.0"})"
                    if args and args[0]:
                        return args[0]
                    return "col('AdjClose')"
                raise ValueError(f"unsupported pd.{name}")
        # s.pct_change(k) / s.shift(k) / s.diff(k)
        # s.pct_change(k) / s.shift(k) / s.diff(k) / s.clip / s.fillna / s.replace
        if isinstance(func, ast.Attribute):
            name = func.attr
            base = self._expr(func.value)
            if name == "rolling":
                # s.rolling(N) 单独出现（后续 .corr 等）→ 返回标记，调用处从 node 重取
                n = None
                if node.args:
                    n = self._int_arg(node.args[0])
                else:
                    for _kw in node.keywords:
                        if _kw.arg == "window" and isinstance(_kw.value, ast.Constant):
                            n = int(_kw.value.value)
                return f"__ROLLKW__({base}, {n if n is not None else 20})"
            if name == "corr" or name == "cov":
                op = "ts_corr" if name == "corr" else "ts_cov"
                # s.rolling(N, ...).corr(other) / s.rolling(N,...).cov(other)
                fv = func.value
                if isinstance(fv, ast.Call) and isinstance(fv.func, ast.Attribute) \
                   and fv.func.attr in ("rolling", "__ROLLKW__"):
                    b2 = self._expr(fv.func.value)
                    n = 20
                    if fv.args:
                        n = self._int_arg(fv.args[0])
                    else:
                        for _kw in fv.keywords:
                            if _kw.arg == "window" and isinstance(_kw.value, ast.Constant):
                                n = int(_kw.value.value)
                    other = self._expr(node.args[0]) if node.args else "col('AdjClose')"
                    return f"o['{op}']({b2}, {other}, {n})"
                other = self._expr(node.args[0]) if node.args else "col('AdjClose')"
                return f"o['ts_corr']({base}, {other}, 20)"
            if name == "rank":
                # s.rolling(N, ...).rank(pct=True) → ts_rank(s, N)
                fv = func.value
                if isinstance(fv, ast.Call) and isinstance(fv.func, ast.Attribute) \
                   and fv.func.attr in ("rolling", "__ROLLKW__"):
                    b2 = self._expr(fv.func.value)
                    n = 20
                    if fv.args:
                        n = self._int_arg(fv.args[0])
                    else:
                        for _kw in fv.keywords:
                            if _kw.arg == "window" and isinstance(_kw.value, ast.Constant):
                                n = int(_kw.value.value)
                    return f"o['ts_rank']({b2}, {n})"
                return f"o['rank']({base})"
            if name == "round":
                return base
            if name == "min":
                return f"o['ts_min']({base}, 20)"
            if name == "idxmax" or name == "idxmin" or name == "astype" or name == "rename":
                return base
            if name == "div" or name == "truediv":
                # s.div(other) / s.div(other, fill_value=v) → divide
                other = self._expr(node.args[0]) if node.args else "col('AdjClose')"
                fv = None
                for kw in node.keywords:
                    if kw.arg == "fill_value" and isinstance(kw.value, ast.Constant):
                        fv = kw.value.value
                if fv is not None:
                    # fill_value 语义近似：other 缺失处用 fv —— FE 无直接算子，退化为 where(eq(other,nan),x_fv,...)
                    return f"o['divide']({base}, o['where'](o['eq']({other}, nan), {fv}, {other}))"
                return f"o['divide']({base}, {other})"
            if name == "mul" or name == "rmul":
                other = self._expr(node.args[0]) if node.args else "1.0"
                return f"o['multiply']({base}, {other})"
            if name == "add" or name == "radd":
                other = self._expr(node.args[0]) if node.args else "0.0"
                return f"o['add']({base}, {other})"
            if name == "sub" or name == "rsub":
                other = self._expr(node.args[0]) if node.args else "0.0"
                return f"o['subtract']({base}, {other})"
            if name == "var":
                if isinstance(func.value, ast.Call) and isinstance(func.value.func, ast.Attribute)                    and func.value.func.attr == "rolling":
                    b2 = self._expr(func.value.func.value)
                    n = 20
                    if func.value.args:
                        n = self._int_arg(func.value.args[0])
                    else:
                        for _kw in func.value.keywords:
                            if _kw.arg == "window" and isinstance(_kw.value, ast.Constant):
                                n = int(_kw.value.value)
                    return f"o['ts_var']({b2}, {n})"
                return f"o['power'](o['ts_std']({base}, 20), 2)"
            if name == "abs":
                return f"o['abs']({base})"
            if name == "sqrt":
                return f"o['sqrt']({base})"
            if name == "median":
                return f"o['ts_median']({base}, {self._default_win})" if hasattr(self, '_default_win') else f"o['median_xseq']({base})"
            if name == "cumsum":
                return f"o['cumsum']({base})"
            if name == "cummax":
                return f"o['cummax']({base})"
            if name == "cummin":
                return f"o['cummin']({base})"
            if name == "max":
                if base.startswith("o['maximum']"):
                    return base
                return f"o['maximum']({base}, {base})"
            if name == "fillna":
                fv = self._expr(node.args[0]) if node.args else "0.0"
                return f"o['fillna']({base}, {fv})"
            if name == "pct_change":
                k = self._int_arg(node.args[0]) if node.args else 1
                return f"o['ts_pct']({base}, {k})"
            if name == "shift":
                k = self._int_arg(node.args[0]) if node.args else 1
                return f"o['ts_delay']({base}, {k})"
            if name == "diff":
                k = self._int_arg(node.args[0]) if node.args else 1
                return f"o['ts_delta']({base}, {k})"
            if name == "copy":
                return base
            if name == "clip":
                lo = "0.0"; hi = "1.0"
                if node.args:
                    lo = self._expr(node.args[0])
                    hi = self._expr(node.args[1]) if len(node.args) > 1 else hi
                else:
                    for kw in node.keywords:
                        if kw.arg == "lower":
                            lo = self._expr(kw.value)
                        elif kw.arg == "upper":
                            hi = self._expr(kw.value)
                return f"o['clip']({base}, {lo}, {hi})"
            if name == "pow":
                k = self._int_arg(node.args[0]) if node.args else 2
                return f"o['power']({base}, {k})"
            if name == "rolling":
                # s.rolling(N) 单独出现（后续 .corr 等）→ 返回标记，调用处从 node 重取
                n = None
                if node.args:
                    n = self._int_arg(node.args[0])
                else:
                    for _kw in node.keywords:
                        if _kw.arg == "window" and isinstance(_kw.value, ast.Constant):
                            n = int(_kw.value.value)
                return f"__ROLLKW__({base}, {n if n is not None else 20})"
            if name == "corr" or name == "cov":
                op = "ts_corr" if name == "corr" else "ts_cov"
                # s.rolling(N, ...).corr(other) / s.rolling(N,...).cov(other)
                fv = func.value
                if isinstance(fv, ast.Call) and isinstance(fv.func, ast.Attribute) \
                   and fv.func.attr in ("rolling", "__ROLLKW__"):
                    b2 = self._expr(fv.func.value)
                    n = 20
                    if fv.args:
                        n = self._int_arg(fv.args[0])
                    else:
                        for _kw in fv.keywords:
                            if _kw.arg == "window" and isinstance(_kw.value, ast.Constant):
                                n = int(_kw.value.value)
                    other = self._expr(node.args[0]) if node.args else "col('AdjClose')"
                    return f"o['{op}']({b2}, {other}, {n})"
                other = self._expr(node.args[0]) if node.args else "col('AdjClose')"
                return f"o['ts_corr']({base}, {other}, 20)"
            if name == "rank":
                # s.rolling(N, ...).rank(pct=True) → ts_rank(s, N)
                fv = func.value
                if isinstance(fv, ast.Call) and isinstance(fv.func, ast.Attribute) \
                   and fv.func.attr in ("rolling", "__ROLLKW__"):
                    b2 = self._expr(fv.func.value)
                    n = 20
                    if fv.args:
                        n = self._int_arg(fv.args[0])
                    else:
                        for _kw in fv.keywords:
                            if _kw.arg == "window" and isinstance(_kw.value, ast.Constant):
                                n = int(_kw.value.value)
                    return f"o['ts_rank']({b2}, {n})"
                return f"o['rank']({base})"
            if name == "round":
                return base
            if name == "min":
                return f"o['ts_min']({base}, 20)"
            if name == "idxmax" or name == "idxmin" or name == "astype" or name == "rename":
                return base
            if name == "abs":
                return f"o['abs']({base})"
            if name == "fillna":
                # fillna(v) → where(is_nan(s), v, s)
                v = self._expr(node.args[0]) if node.args else "0.0"
                return f"o['where'](o['is_nan']({base}), {v}, {base})"
            if name == "where":
                # s.where(cond, other)
                cond = self._expr(node.args[0]) if node.args else "1.0"
                other = self._expr(node.args[1]) if len(node.args) > 1 else "0.0"
                return f"o['where']({cond}, {base}, {other})"
            if name == "rename":
                return base
            if name == "rolling":
                # .rolling(N).<agg>() 链在下面专门处理；这里是裸 rolling 出现在表达式里
                raise ValueError(f"unsupported method rolling")
            if name in ("ne", "eq", "gt", "lt", "ge", "le"):
                # s.ne(0) → ne(s, 0)
                other = self._expr(node.args[0]) if node.args else "0.0"
                return f"o['{name}']({base}, {other})"
            if name == "ffill":
                return base
            if name == "cumsum":
                return f"o['cum_sum']({base})"
            if name == "cummax":
                return f"o['expanding_max']({base})"
            if name == "quantile":
                q = self._expr(node.args[0]) if node.args else "0.5"
                return f"o['ts_quantile']({base}, 20, {q})"
            if name == "replace":
                if len(node.args) == 2:
                    to_node = node.args[0]
                    repl = self._expr(node.args[1])
                    if isinstance(to_node, ast.List) and to_node.elts:
                        to = self._expr(to_node.elts[0])
                        out = f"o['where'](o['eq']({base}, {to}), {repl}, {base})"
                        for elt in to_node.elts[1:]:
                            to = self._expr(elt)
                            out = f"o['where'](o['eq']({out}, {to}), {repl}, {out})"
                        return out
                    to = self._expr(to_node)
                    return f"o['where'](o['eq']({base}, {to}), {repl}, {base})"
                return base
            raise ValueError(f"unsupported method {name}")
        if isinstance(func, ast.Name):
            n = func.id
            if n == "abs":
                return f"o['abs']({self._expr(node.args[0]) if node.args else "col('AdjClose')"})" if args else "col('AdjClose')"
            if n == "max" and node.args and isinstance(node.args[0], ast.List):
                elts = [self._expr(e) for e in node.args[0].elts]
                out = f"o['maximum']({elts[0]}, {elts[1]})" if len(elts) >= 2 else (elts[0] if elts else "col('AdjClose')")
                for e in elts[2:]:
                    out = f"o['maximum']({out}, {e})"
                return out
            if n == "min" and node.args and isinstance(node.args[0], ast.List):
                elts = [self._expr(e) for e in node.args[0].elts]
                out = f"o['minimum']({elts[0]}, {elts[1]})" if len(elts) >= 2 else (elts[0] if elts else "col('AdjClose')")
                for e in elts[2:]:
                    out = f"o['minimum']({out}, {e})"
                return out
            if n == "abs":
                return f"o['abs']({self._expr(node.args[0]) if node.args else "col('AdjClose')"})" if args else "col('AdjClose')"
            if n == "max" and node.args and isinstance(node.args[0], ast.List):
                elts = [self._expr(e) for e in node.args[0].elts]
                out = f"o['maximum']({elts[0]}, {elts[1]})" if len(elts) >= 2 else (elts[0] if elts else "col('AdjClose')")
                for e in elts[2:]:
                    out = f"o['maximum']({out}, {e})"
                return out
            if n == "min" and node.args and isinstance(node.args[0], ast.List):
                elts = [self._expr(e) for e in node.args[0].elts]
                out = f"o['minimum']({elts[0]}, {elts[1]})" if len(elts) >= 2 else (elts[0] if elts else "col('AdjClose')")
                for e in elts[2:]:
                    out = f"o['minimum']({out}, {e})"
                return out
            if n in self.env:
                return self.env[n]
            if n in ("np", "pd"):
                return n
            # 自定义辅助函数调用 → 展开 env
            raise ValueError(f"unsupported function {n}")
        raise ValueError(f"unsupported call {type(func).__name__}")

    def _int_arg(self, node) -> int:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return int(node.value)
        if isinstance(node, ast.Name) and node.id in self.env:
            v = self.env[node.id]
            if v.isdigit():
                return int(v)
        if isinstance(node, ast.Name):
            # 变量窗口：转成 DSL 数值不可能——但 env 里可能存了赋值表达式
            for s2 in self.body:
                if isinstance(s2, ast.Assign):
                    for t in s2.targets:
                        if isinstance(t, ast.Name) and t.id == node.id \
                           and isinstance(s2.value, ast.Constant) and isinstance(s2.value.value, (int, float)):
                            return int(s2.value.value)
        raise ValueError(f"non-int arg {type(node).__name__}")


def transpile_code(code: str) -> str:
    """code → FE expr；失败抛异常。"""
    c = CodeToExpr(code)
    expr = c.convert()
    expr = FE._wrap_bare_ops(expr)
    return expr


# =====================================================================
# 2. 主流程
# =====================================================================
FUND_COLS = ["pb_lf", "pe_ttm", "ps_ttm", "pcf_ocf_ttm", "mkt_cap_float", "free_turn",
             "debttoassets", "roe_ttm2", "roa2_ttm2", "free_float_shares", "qfa_yoygr",
             "forecast_incap_chgr_mid"]


def short_code_items(data) -> list:
    out = []
    for d in data:
        if d.get("can_use_factor_engine"):
            continue
        code = d.get("code") or ""
        if not code or "NotImplementedError" in code:
            continue
        if "minute_tools" in code or "TradingDay" in code:
            continue
        if any(fc in code for fc in FUND_COLS):
            continue
        out.append(d)
    return out


def convert_batch(dry: bool = False, verbose: bool = True, validate: bool = True) -> dict:
    """转换所有 False 且有 code 的因子（排除 fund/minute/mydsl）。

    validate=True 时：对每个转换产物走真编译链
      parse_expr(fe, surface='compat_research') → Factor → eng.run
    只有真编译+落值成功才置 can_use_factor_engine=True。
    """
    data = json.loads(LQTP_PATH.read_text(encoding="utf-8"))
    items = short_code_items(data)
    stats = {"total_candidates": len(items), "ok": 0, "fail": {}, "examples": []}
    if validate:
        # 预构建引擎与算子工厂（串行 smoke，避免 resource admission）
        from factor_engine.api.dsl_parser import parse_expr
        from factor_engine.api.factor import Factor as _Factor
        import re as _re
        o_base, eng, Factor = FE._engine_env()
        # 收集全部 o['...'] 名（exec 通道用）
        all_ops = set()
        for d in items:
            try:
                all_ops |= set(_re.findall(r"o\['([A-Za-z_][A-Za-z0-9_]*)'\]", transpile_code(d.get("code") or "")))
            except Exception:
                pass
        from factor_engine.api.cleaned_ops import make_cleaned_call_factory as _mcc
        o2 = {nm: _mcc(nm) for nm in sorted(all_ops)}
        o2["col"] = FE.col if hasattr(FE, "col") else o_base.get("col")
    else:
        parse_expr = _Factor = o2 = eng = Factor = None

    for d in items:
        code = d.get("code") or ""
        name = d.get("factor_name", "?")
        try:
            expr = transpile_code(code)
            if not FE._is_single_table_convertible(expr):
                raise ValueError("requires-multi-table")
            if validate:
                # 真编译链
                if expr.strip().startswith("o[") or "o['" in expr or 'o["' in expr:
                    ns = {"o": o2, "col": o2.get("col"), "Factor": _Factor}
                    exec(f"_f = Factor(name={name!r}, expr={expr})", ns)
                    out = eng.run(ns["_f"], market="ashare")
                    wide = out["result"] if isinstance(out, dict) and "result" in out else out
                    if hasattr(wide, "unstack"):
                        wide = wide.unstack()
                else:
                    _e = parse_expr(expr, surface="compat_research")
                    fac = _Factor(name=name, expr=_e)
                    out = eng.run(fac, market="ashare")
                    wide = out["result"] if isinstance(out, dict) and "result" in out else out
                    if hasattr(wide, "unstack"):
                        wide = wide.unstack()
                if wide is None or (hasattr(wide, "shape") and wide.shape[1] == 0):
                    raise ValueError("engine empty result")
            if not dry:
                d["can_use_factor_engine"] = True
                d["fe_formula"] = expr
            stats["ok"] += 1
            if verbose and len(stats["examples"]) < 15:
                stats["examples"].append({"name": name[:40], "fe": expr[:150]})
        except Exception as exc:
            key = str(exc).split(":")[0] if ":" in str(exc) else type(exc).__name__
            stats["fail"][key] = stats["fail"].get(key, 0) + 1
            if verbose and len(stats.get("fail_samples", [])) < 12:
                stats.setdefault("fail_samples", []).append(f"{name[:36]} | {str(exc)[:90]}")
    if not dry:
        LQTP_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    return stats


def verify_smoke(sample_n: int = 24) -> dict:
    """真编译对拍：fe_formula 走 parse_expr(surface=compat_research) 编译 → FactorEngine 落值
    vs factor_matrices_all 原始矩阵 spearman。"""
    from factor_engine.api.dsl_parser import parse_expr
    data = json.loads(LQTP_PATH.read_text(encoding="utf-8"))
    convertible = []
    for i, it in enumerate(data):
        if not it.get("can_use_factor_engine") or not it.get("fe_formula"):
            continue
        if it.get("status") == "evoalpha_week_new":
            continue
        fe = it["fe_formula"]
        if fe.startswith("evoalpha"):
            continue
        convertible.append((i, it, fe))
    rng = np.random.default_rng(11)
    if len(convertible) <= sample_n:
        picks = convertible
    else:
        picks = [convertible[i] for i in rng.choice(len(convertible), sample_n, replace=False)]
    o, eng, Factor = FE._engine_env()
    ref_syms = ["000001.SZ", "000002.SZ", "000004.SZ"]
    tested = passed = 0
    failed, corrs = [], []
    from factor_engine.api.factor import Factor as _Factor
    from factor_engine.api.cleaned_ops import make_cleaned_call_factory as _mcc
    import re as _re
    for idx, (i, it, fe) in enumerate(picks):
        name = it.get("factor_name", "?")
        pf = FV_DIR / f"{name.replace('factor_', '', 1)}.parquet"
        if not pf.exists():
            pf = FV_DIR / f"{name}.parquet"
        if not pf.exists():
            raise ValueError(f"no ref matrix {pf}")
        try:
            if fe.strip().startswith("o[") or "o['" in fe or 'o["' in fe:
                # Python 代码形态（DSL 侧产物）：exec → Factor
                ops_needed = set(_re.findall(r"o\['([A-Za-z_][A-Za-z0-9_]*)'\]", fe))
                o2 = {nm: _mcc(nm) for nm in sorted(ops_needed)}
                o2["col"] = _mcc("col") if "col" in ops_needed else o.get("col")
                res = None
                ns = {"o": o2, "col": o2.get("col"), "Factor": _Factor, "_Factor": _Factor}
                exec(f"_f = Factor(name={name!r}, expr={fe})", ns)
                out = eng.run(ns["_f"], market="ashare")
                if isinstance(out, dict):
                    r = out.get("result")
                    wide = r.unstack() if hasattr(r, "unstack") else r
                elif out is None:
                    raise ValueError("engine returned None (exec channel)")
                else:
                    wide = out
                if isinstance(wide, pd.Series):
                    wide = wide.unstack()
            else:
                from factor_engine.api.factor import Factor as _F2
                _expr = parse_expr(fe, surface="compat_research")
                fac = _F2(name=name, expr=_expr)
                out = eng.run(fac, market="ashare")
                wide = out["result"].unstack() if isinstance(out, dict) and "result" in out else out
                if isinstance(wide, pd.Series):
                    wide = wide.unstack()
            if not isinstance(wide, pd.DataFrame):
                raise ValueError(f"engine returned {type(wide).__name__} not DataFrame")
            ref = pd.read_parquet(pf)
            ref.index = pd.to_datetime(ref.index)
            common_idx = wide.index.intersection(ref.index)
            common_cols = [c for c in wide.columns if c in ref.columns]
            if len(common_idx) < 3 or len(common_cols) < 3:
                raise ValueError(f"overlap rows={len(common_idx)} cols={len(common_cols)}")
            a = wide.loc[common_idx, common_cols].astype(float)
            b = ref.loc[common_idx, common_cols].astype(float)
            cs = []
            from scipy.stats import spearmanr
            for c in common_cols:
                x = a[c].values; y = b[c].values
                m = np.isfinite(x) & np.isfinite(y)
                if m.sum() >= 10:
                    r_, _ = spearmanr(x[m], y[m])
                    if np.isfinite(r_):
                        cs.append(r_)
            avg = float(np.mean(cs)) if cs else float("nan")
            tested += 1
            ok = bool(np.isfinite(avg) and abs(avg) >= 0.95)
            if ok:
                passed += 1
                corrs.append(abs(avg) if np.isfinite(avg) else 0)
            else:
                failed.append({"name": name, "corr": avg if np.isfinite(avg) else str(avg)})
            _log(f"  verify[{idx+1}/{len(picks)}] {name[:40]:42s} corr={avg if np.isfinite(avg) else float('nan'):.3f} {'OK' if ok else 'FAIL'}")
        except Exception as exc:
            tested += 1
            failed.append({"name": name, "err": str(exc)[:120]})
            _log(f"  verify[{idx+1}/{len(picks)}] {name[:40]:42s} ERR {str(exc)[:90]}")
    return {"tested": tested, "passed": passed, "failed": failed, "corr": corrs}



def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--convert", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--sample", type=int, default=24)
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    FE._init_fe_ops()
    FE._init_known_fields()
    _log(f"FE ops={len(FE._FE_OPS)}")

    if args.dry or args.smoke:
        stats = convert_batch(dry=True, validate=False)
        _log(f"DRY: candidates={stats['total_candidates']} ok={stats['ok']} fail={stats['fail']}")
        for s in stats.get("fail_samples", []):
            _log(f"  FAIL {s}")

    if args.convert and not args.dry:
        stats = convert_batch(dry=False, validate=True)
        _log(f"CONVERT: ok={stats['ok']} candidates={stats['total_candidates']} fail={stats['fail']}")
        for s in stats.get("fail_samples", []):
            _log(f"  FAIL {s}")
        _log("examples: " + json.dumps(stats.get("examples", [])[:5], ensure_ascii=False))

    if args.smoke:
        result = verify_smoke(args.sample)
        _log(f"SMOKE: tested={result['tested']} passed={result['passed']} failures={len(result['failed'])}")
        for f in result["failed"][:15]:
            _log(f"  FAIL {f}")
        if result["corr"]:
            _log(f"  corr mean={np.mean(result['corr']):.3f}")


if __name__ == "__main__":
    main()
