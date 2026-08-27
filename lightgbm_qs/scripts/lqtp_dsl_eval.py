# -*- coding: utf-8 -*-
"""本地 LQTP DSL 求值器: 用后复权 OHLCV 面板重算平台 LQTP 因子。

平台 functions.yaml 复权口径 (与表内 Factor 一致):
  open=Open*Factor, high=High*Factor, low=Low*Factor, close=Close*Factor,
  pre_close=PreClose*Factor, vwap=Vwap*Factor,
  volume=Volume/Factor, amount=Amount, ret=Return, factor=Factor
后复权量 (volume/Factor) 的 rolling 计算即平台口径。
"""
import re, math, numpy as np, pandas as pd

# ---------- 时序列算子 (输入 date x asset DataFrame) ----------
def _roll(df, n, fn):
    return df.rolling(n, min_periods=max(1, min(n, 2))).apply(fn, raw=True)
def _ts_mean(x,n): return x.rolling(n,min_periods=1).mean()
def _ts_sum(x,n):  return x.rolling(n,min_periods=1).sum()
def _ts_std(x,n):  return x.rolling(n,min_periods=1).std(ddof=0)
def _ts_min(x,n):  return x.rolling(n,min_periods=1).min()
def _ts_max(x,n):  return x.rolling(n,min_periods=1).max()
def _ts_delta(x,n): return x - x.shift(n)
def _ts_pct(x,n):   return x / x.shift(n) - 1.0
def _delay(x,n):    return x.shift(n)
def _ts_corr(a,b,n): return a.rolling(n,min_periods=2).corr(b)
def _ts_cov(a,b,n):  return a.rolling(n,min_periods=2).cov(b)
def _ts_skew(x,n):
    return x.rolling(n,min_periods=3).skew()
def _ts_kurt(x,n):
    return x.rolling(n,min_periods=4).kurt()
def _ema(x, span):
    return x.ewm(span=span, adjust=False).mean()
def _sma(x, n, m=None):
    m = m or n
    return x.rolling(n,min_periods=1).mean()
def _decay_linear(x, n):
    w = np.arange(1, n+1, dtype=float)
    w = w / w.sum()
    # 卷积近似 (线性加权) -> 用 rolling.apply 替代为 conv: 用 pandas 无直接, 但可用 sum 分解: 太复杂, 保留apply但只在必要时
    return x.rolling(n,min_periods=1).apply(lambda r: float(np.dot(r, w[-len(r):])), raw=True)
def _ts_decay_linear(x,n):
    return _decay_linear(x,n)
def _ts_rank(x,n):
    # 窗口内当前值 rank (末位 rank) -> rolling.rank(pct=False) 返回窗口内最后一个元素的 rank
    return x.rolling(n,min_periods=1).rank(pct=False)
def _ts_rank_pct(x,n):
    return x.rolling(n,min_periods=1).rank(pct=True)
def _ts_quantile(x,n,q):
    # 平台有时 ts_quantile(x, q, n) 或 ts_quantile(x, n, q); 尝试识别
    import numbers
    if isinstance(n,(int,float)) and not isinstance(q,(int,float)):
        n,q = q,n
    return x.rolling(int(n),min_periods=1).quantile(q)
def _ts_argmax(x,n):
    return x.rolling(n,min_periods=1).apply(lambda r: float(np.nanargmax(r)), raw=True)
def _ts_argmin(x,n):
    return x.rolling(n,min_periods=1).apply(lambda r: float(np.nanargmin(r)), raw=True)
def _ts_topk_sum(x,n,k):
    return x.rolling(n,min_periods=1).apply(lambda r: float(np.sort(r)[-min(k,len(r)):].sum()), raw=True)
def _ts_bottomk_sum(x,n,k):
    return x.rolling(n,min_periods=1).apply(lambda r: float(np.sort(r)[:min(k,len(r))].sum()), raw=True)
def _ts_zscore(x,n):
    m=x.rolling(n,min_periods=1).mean(); s=x.rolling(n,min_periods=1).std(ddof=0)
    return (x-m)/s.replace(0,np.nan)
def _ts_expanding_rank(x):
    return x.expanding(min_periods=1).rank(pct=False)
def _ts_moment(x,n,k):
    return x.rolling(n,min_periods=1).apply(lambda r: float(pd.Series(r).kurt()) if k==4 else float(pd.Series(r).skew()), raw=False)
def _ts_product(x,n):
    return x.rolling(n,min_periods=1).apply(lambda r: float(np.nanprod(r)), raw=True)
def _ts_hump_decay(x, alpha=1.0):
    # alpha 0.02 typical: weight proportional to r^(2*(1-alpha)) * (1-r)^(2*alpha) 简单: 线性衰减叠加
    n=20
    w = np.array([1.0/(2.0**alpha)**i if alpha>0 else 1.0 for i in range(n)],dtype=float); w/=w.sum()
    w=w[::-1]
    return x.rolling(n,min_periods=1).apply(lambda r: float(np.dot(r, w[-len(r):])), raw=True)
def _ts_max_buildup(x,n):
    return x.rolling(n,min_periods=1).max()  # 近似: 区间最大值(累积)
def _ts_regression_slope(x, y=None, n=None):
    if y is None and n is None:
        return x.interpolate()
    # y 回归 on x
    def slope(r):
        xx=r[:,0]; yy=r[:,1]
        m=~np.isnan(xx)&~np.isnan(yy)
        if m.sum()<2: return np.nan
        return float(np.polyfit(xx[m], yy[m], 1)[0])
    res_dict={}
    arr_x=x.to_numpy(); arr_y=y.to_numpy()
    for c in range(arr_x.shape[1]):
        col=[]
        for r in range(arr_x.shape[0]):
            lo=max(0,r-n+1)
            xx=arr_x[lo:r+1,c]; yy=arr_y[lo:r+1,c]
            m=~np.isnan(xx)&~np.isnan(yy)
            col.append(float(np.polyfit(xx[m], yy[m], 1)[0]) if m.sum()>=2 else np.nan)
        res_dict[x.columns[c]]=col
    return pd.DataFrame(res_dict, index=x.index).reindex(columns=x.columns)
def _ts_poly2_coeff(x,n):
    def coeff(r):
        if len(r)<3: return np.nan
        return float(np.polyfit(np.arange(len(r)), r, 2)[0])
    return x.rolling(n,min_periods=3).apply(coeff, raw=True)

# ---------- 截面算子 ----------
def _rank(x):
    return x.rank(axis=1, pct=False)
def _cs_rank(x):
    return x.rank(axis=1, pct=True)
def _cs_zscore(x):
    m=x.mean(axis=1); s=x.std(axis=1, ddof=0)
    return x.sub(m,axis=0).div(s.replace(0,np.nan),axis=0)
def _cs_mean(x):
    m=x.mean(axis=1)
    return pd.DataFrame({c: m for c in x.columns}, index=x.index)
def _cs_sum(x):
    m=x.sum(axis=1)
    return pd.DataFrame({c: m for c in x.columns}, index=x.index)
def _cs_regression(a,b,c=None):
    m=a.rank(axis=1).corrwith(b.rank(axis=1),axis=1)
    return pd.DataFrame({col: m for col in a.columns}, index=a.index)
def _rank_corr(a,b,n=None):
    if n is None:
        # 截面 rank corr per date
        return a.rank(axis=1).corrwith(b.rank(axis=1),axis=1)
    return a.rolling(n,min_periods=2).corr(b)
def _scale(x, a=1.0):
    return x * a
def _cap(x, lo, hi):
    if isinstance(x,(int,float)):
        return float(min(max(x,lo),hi))
    if hasattr(x,'shape') is False or (len(getattr(x,'shape',[]))==0):
        return float(min(max(float(x),lo),hi))
    return x.clip(lo, hi)

def _to_bool(c):
    import pandas as pd
    if isinstance(c, pd.DataFrame):
        return (c.astype(float)!=0).astype(bool)
    if hasattr(c,"dtype") and c.dtype==bool: return c
    return (c.astype(float)!=0).astype(bool)

# ---------- 标量/逐元素 ----------
def _safe_div(a,b):
    if isinstance(b,(int,float)):
        return a / b if b!=0 else pd.DataFrame(np.nan,index=a.index,columns=a.columns)
    return a / b.replace(0,np.nan)
def _where(c,a,b):
    import pandas as _pd
    if isinstance(c,(int,float)):
        c=_pd.DataFrame(bool(c), index=a.index, columns=a.columns)
    c=_to_bool(c)
    if isinstance(a,(int,float)):
        a=_pd.DataFrame(a, index=c.index, columns=c.columns)
    if isinstance(b,(int,float)):
        b=_pd.DataFrame(b, index=c.index, columns=c.columns)
    return a.where(c, b)
def _iif(c,a,b):
    import pandas as _pd
    if isinstance(c,(int,float)):
        c=_pd.DataFrame(bool(c), index=a.index, columns=a.columns)
    c=_to_bool(c)
    if isinstance(a,(int,float)):
        a=_pd.DataFrame(a, index=c.index, columns=c.columns)
    if isinstance(b,(int,float)):
        b=_pd.DataFrame(b, index=c.index, columns=c.columns)
    return a.where(c, b)
def _and(a,b):
    return ((a>0)&(b>0)).astype(float)
def _or(a,b):
    return ((a>0)|(b>0)).astype(float)
def _abs(x): return abs(x) if isinstance(x,(int,float)) else x.abs()
def _log(x): return np.log(x) if isinstance(x,(int,float)) else np.log(x.clip(lower=1e-12))
def _safe_log(x):
    if isinstance(x,pd.DataFrame):
        return pd.DataFrame(np.where(x>0, np.log(x), np.nan), index=x.index, columns=x.columns)
    return np.where(x>0, np.log(x), np.nan)
def _sqrt(x): return np.sqrt(x) if isinstance(x,(int,float)) else np.sqrt(x.clip(lower=0))
def _power(x,p): return x**p if isinstance(x,(int,float)) else x.pow(p)
def _sign(x): return np.sign(x) if isinstance(x,(int,float)) else np.sign(x)
def _signed_sqrt(x): return np.sign(x)*np.sqrt(abs(x)) if isinstance(x,(int,float)) else np.sign(x)*np.sqrt(x.abs())
def _sigmoid(x): return 1.0/(1.0+np.exp(-x)) if isinstance(x,(int,float)) else 1.0/(1.0+np.exp(-x))
def _coalesce(a,b): return a.fillna(b)
def _is_nan(x): return x.isna().astype(float)
def _round(x): return x.round()
def _pct_change(x, n=1): return x/x.shift(int(n))-1.0

# ---------- 字段 (后复权面板) ----------
def build_panels(ohlcv_adj):
    fields=['Open','High','Low','Close','Vwap','Volume','Amount']
    P={}
    for f in fields:
        P[f]=ohlcv_adj[ohlcv_adj.field==f].pivot_table(index='date',columns='asset',values='value')
    F=P['Close']*0.0
    # factor = Close_adj / Close_raw; 但我们没有raw, 用 volume 反推? 
    # 直接: close_adj 已复权, factor 用于 volume. 平台 volume=Volume/Factor.
    # 这里需要 Factor 面板. 从 cos_data StockDailyBar 直接读.
    import glob, os
    bar_dir='/home/sunhaiwei/cos_data/StockDailyBar'
    facs=[]
    for d in P['Close'].index:
        f=f'{bar_dir}/{d.date()}.parquet'
        if os.path.exists(f):
            b=pd.read_parquet(f,columns=['Symbol','Factor']).set_index('Symbol')
            s=b['Factor'].reindex(P['Close'].columns)
            s.name=d; facs.append(s)
    Factor=pd.concat(facs,axis=1).T
    P['close']=P['Close']; P['high']=P['High']; P['low']=P['Low']; P['open']=P['Open']
    P['vwap']=P['Vwap']
    P['pre_close']=P['Close'].shift(1)
    P['volume']=P['Volume']/Factor      # 后复权量
    P['amount']=P['Amount']             # 平台 amount 不复权
    P['ret']=P['Close'].pct_change()
    P['factor']=Factor
    P['daily_return']=P['Close'].pct_change()
    P['returns']=P['Close'].pct_change()
    _a=(P['High']-P['Low']).copy()
    _b=(P['High']-P['Close'].shift(1)).abs().copy()
    _c=(P['Low']-P['Close'].shift(1)).abs().copy()
    _tr=pd.concat([_a,_b,_c], axis=1)
    _tr=pd.DataFrame(_tr.groupby(_tr.index).max())  # date x asset? no
    P['true_range']=pd.concat([_a,_b,_c], keys=['a','b','c'], axis=1).max(axis=1, level=None) if False else P['High']-P['Low']
    P['true_range']=_a.where(_a>=_b,_b)
    P['true_range']=P['true_range'].where(P['true_range']>=_c,_c)
    P['is_suspend']=P['Close'].isna().astype(float)
    return P

# ---------- 递归下降 DSL 解析器 ----------
import ast as _ast

class _Parser:
    def __init__(self, s):
        self.toks = self._tokenize(s); self.i=0
    def _tokenize(self, s):
        # 识别 数字/标识符/运算符/括号/逗号
        toks=[]; j=0; n=len(s)
        while j<n:
            c=s[j]
            if c in ' \t\n': j+=1; continue
            if c.isdigit() or (c=='.' and j+1<n and s[j+1].isdigit()):
                k=j
                while k<n and (s[k].isdigit() or s[k]=='.' or s[k] in 'eE'): 
                    if s[k] in 'eE': 
                        k+=1
                        if k<n and s[k] in '+-': k+=1
                        continue
                    k+=1
                tok=s[j:k]; toks.append(('num', int(tok) if (tok.isdigit() or (tok.startswith('-') and tok[1:].isdigit())) and '.' not in tok else float(tok))); j=k; continue
            if c.isalpha() or c=='_':
                k=j
                while k<n and (s[k].isalnum() or s[k]=='_'): k+=1
                toks.append(('id', s[j:k])); j=k; continue
            if s.startswith('**',j): toks.append(('op','**')); j+=2; continue
            if s.startswith('==',j): toks.append(('op','==')); j+=2; continue
            if s.startswith('>=',j): toks.append(('op','>=')); j+=2; continue
            if s.startswith('<=',j): toks.append(('op','<=')); j+=2; continue
            if c in '+-*/<>=':
                toks.append(('op',c)); j+=1; continue
            if c=='(': toks.append(('lp','(')); j+=1; continue
            if c==')': toks.append(('rp',')')); j+=1; continue
            if c==',': toks.append(('comma',',')); j+=1; continue
            # string literal (quoted)
            if c in "'\"":
                k=j+1
                while k<n and s[k]!=c: k+=1
                toks.append(('str', s[j+1:k])); j=k+1; continue
            raise ValueError(f"unknown char {c!r} at {j} in {s}")
        toks.append(('end',''))
        return toks
    def peek(self): return self.toks[self.i]
    def next(self): t=self.toks[self.i]; self.i+=1; return t
    def expect(self, kind):
        t=self.next()
        if t[0]!=kind: raise ValueError(f"expect {kind} got {t} in expr")
        return t
    def parse(self): 
        e=self._expr()
        if self.peek()[0]!='end': raise ValueError(f"trailing {self.peek()} in expr")
        return e
    # 优先级: or < and < compare < +- < */ < unary < ** < primary
    def _expr(self): return self._logical()
    def _logical(self):
        left=self._or()
        return left
    def _or(self):
        left=self._andop()
        while self.peek()[0]=='id' and self.peek()[1] in ('or','and'):
            op=self.next()[1]; right=self._andop(); left=('op',op,left,right)
        return left
    def _andop(self):
        left=self._cmp()
        while self.peek()[0]=='id' and self.peek()[1] in ('and','or'):
            op=self.next()[1]; right=self._cmp(); left=('op',op,left,right)
        return left
    def _cmp(self):
        left=self._add()
        while self.peek()[0]=='op' and self.peek()[1] in ('<','>','=','==','<=','>=','<=' ,'>='):
            op=self.next()[1]; right=self._add(); left=('cmp',op,left,right)
        return left
    def _add(self):
        left=self._mul()
        while self.peek()[0]=='op' and self.peek()[1] in ('+','-'):
            op=self.next()[1]; right=self._mul(); left=('arith',op,left,right)
        return left
    def _mul(self):
        left=self._unary()
        while self.peek()[0]=='op' and self.peek()[1] in ('*','/','%'):
            op=self.next()[1]; right=self._unary(); left=('arith',op,left,right)
        return left
    def _unary(self):
        if self.peek()[0]=='op' and self.peek()[1] in ('-','+'):
            op=self.next()[1]; return ('neg',op,self._unary())
        return self._power()
    def _power(self):
        base=self._primary()
        if self.peek()[0]=='op' and self.peek()[1]=='**':
            self.next(); return ('pow',base,self._unary())
        return base
    def _primary(self):
        t=self.peek()
        if t[0]=='num': self.next(); return ('num',t[1])
        if t[0]=='str': self.next(); return ('str',t[1])
        if t[0]=='id':
            self.next()
            if self.peek()[0]=='lp':
                self.next(); args=[]
                if self.peek()[0]!='rp':
                    args.append(self._expr())
                    while self.peek()[0]=='comma':
                        self.next(); args.append(self._expr())
                self.expect('rp')
                return ('call', t[1], args)
            return ('field', t[1])
        if t[0]=='lp':
            self.next(); e=self._expr(); self.expect('rp'); return e
        raise ValueError(f"unexpected {t}")

def parse_dsl(s):
    return _Parser(s).parse()

def _eval(node, env, ops):
    kind=node[0]
    if kind=='num': return node[1]
    if kind=='str': return node[1]
    if kind=='field':
        return env[node[1]]
    if kind=='neg':
        return -_eval(node[2],env,ops)
    if kind=='pow':
        return _eval(node[1],env,ops) ** _eval(node[2],env,ops)
    if kind=='arith':
        a=_eval(node[2],env,ops); b=_eval(node[3],env,ops)
        o=node[1]
        if o=='+': return a+b
        if o=='-': return a-b
        if o=='*': return a*b
        if o=='/': return a/b.replace(0,np.nan) if hasattr(b,'replace') else a/b
    if kind=='cmp':
        a=_eval(node[2],env,ops); b=_eval(node[3],env,ops); o=node[1]
        if o=='<': return (a<b).astype(float)
        if o=='>': return (a>b).astype(float)
        if o in ('=','=='): return (a==b).astype(float)
        if o=='<=': return (a<=b).astype(float)
        if o=='>=': return (a>=b).astype(float)
    if kind=='op':
        a=_eval(node[2],env,ops); b=_eval(node[3],env,ops); o=node[1]
        if o=='and': return ((a>0)&(b>0)).astype(float)
        if o=='or': return ((a>0)|(b>0)).astype(float)
    if kind=='call':
        fn=node[1]; args=[_eval(a,env,ops) for a in node[2]]
        if fn in ops: return ops[fn](*args)
        raise ValueError(f"unknown operator {fn}")
    raise ValueError(f"unknown node {kind}")

def eval_formula(formula, env, ops):
    return _eval(parse_dsl(formula), env, ops)

# 注册算子表 (与实现对应)
OPS = {
 'ts_mean':_ts_mean,'ts_sum':_ts_sum,'ts_std':_ts_std,'ts_min':_ts_min,'ts_max':_ts_max,
 'ts_delta':_ts_delta,'ts_pct':_ts_pct,'delay':_delay,'ts_corr':_ts_corr,'ts_cov':_ts_cov,
 'ts_skew':_ts_skew,'ts_kurt':_ts_kurt,'ema':_ema,'sma':_sma,'decay_linear':_decay_linear,
 'ts_decay_linear':_ts_decay_linear,'ts_rank':_ts_rank,'ts_rank_pct':_ts_rank_pct,
 'ts_quantile':_ts_quantile,'zscore':_cs_zscore,'ts_argmax':_ts_argmax,'ts_argmin':_ts_argmin,
 'ts_topk_sum':_ts_topk_sum,'ts_bottomk_sum':_ts_bottomk_sum,'ts_zscore':_ts_zscore,
 'ts_expanding_rank':_ts_expanding_rank,'ts_moment':_ts_moment,'ts_product':_ts_product,
 'ts_hump_decay':_ts_hump_decay,'ts_max_buildup':_ts_max_buildup,'ts_regression_slope':_ts_regression_slope,
 'ts_poly2_coeff':_ts_poly2_coeff,
 'rank':_rank,'cs_rank':_cs_rank,'cs_zscore':_cs_zscore,'cs_mean':_cs_mean,'cs_sum':_cs_sum,
 'cs_regression':_cs_regression,'rank_corr':_rank_corr,'scale':_scale,'cap':_cap,'clip':_cap,
 'safe_div':_safe_div,'where':_where,'iif':_iif,'and':_and,'abs':_abs,'log':_log,
 'safe_log':_safe_log,'sqrt':_sqrt,'power':_power,'sign':_sign,'signed_sqrt':_signed_sqrt,
 'sigmoid':_sigmoid,'coalesce':_coalesce,'is_nan':_is_nan,'round':_round,'pct_change':_pct_change,
}
