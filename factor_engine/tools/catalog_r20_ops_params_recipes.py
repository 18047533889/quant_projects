"""R20 reviewed repairs for obsolete operators and incomplete signatures."""
from __future__ import annotations
import ast

def C(n,*a,**kw): return ast.Call(ast.Name(n,ast.Load()),list(a),[ast.keyword(k,v) for k,v in kw.items()])

class R(ast.NodeTransformer):
 def __init__(self): self.ch=[]
 def m(self,s): self.ch.append(s)
 def visit_Call(self,node):
  node=self.generic_visit(node)
  if not isinstance(node.func,ast.Name): return node
  n=node.func.id;a=node.args;kw={k.arg:k.value for k in node.keywords if k.arg}
  if n=='source_col' and len(a)>=4 and all(isinstance(z,ast.Constant) for z in a[:4]) and a[0].value=='BenchmarkIndexDailyBar' and a[1].value=='Return':
   ic=C('source_col',ast.Constant('BenchmarkIndexDailyBar'),ast.Constant('Close'),ast.Constant('index'),a[3]);ip=C('source_col',ast.Constant('BenchmarkIndexDailyBar'),ast.Constant('PreClose'),ast.Constant('index'),a[3])
   self.m('字段修正：指数日表无直接Return列，按同一明确指数的Close/PreClose计算日收益。');return C('subtract',C('safe_div_null',ic,ip),ast.Constant(1.))
  real_research={'bartlett_test','chi_square_test','corr_test','durbin_watson_test','granger_causality','jarque_bera_test','kendall_corr_test','kpss_test','ks_test','levene_test','lilliefors_test','spearman_corr_test','stationarity_test','ttest_one_sample','ttest_paired','ttest_two_samples','ACF','pacf'}
  if n in real_research:
   self.m(f'研究面接通：保留已实现算子 {n} 的原始统计/金融定义；仅在 compat_research 可见。')
   return node
  if n=='ts_spectral_entropy' and (a or 'x' in kw):
   sx=a[0] if a else kw['x']
   is_ret=isinstance(sx,ast.Name) and sx.id in {'ret','ret_1d'}
   is_ret=is_ret or (isinstance(sx,ast.Call) and isinstance(sx.func,ast.Name) and sx.func.id=='field' and sx.args and isinstance(sx.args[0],ast.Constant) and sx.args[0].value=='ret')
   if not is_ret: node.func.id='ts_spectral_flatness';self.m('类型修正：非收益输入的谱熵改为通用序列谱平坦度，保留频域无序度方向。');return node
  if n=='event_historical_response_mean' and len(a)>=2:
   # Canonical order is response,event.  Event-like daily magnitudes become nonzero-event masks.
   def ret(z): return isinstance(z,ast.Name) and z.id in {'ret','ret_1d'} or isinstance(z,ast.Call) and isinstance(z.func,ast.Name) and z.func.id=='field' and z.args and isinstance(z.args[0],ast.Constant) and z.args[0].value=='ret'
   if ret(a[1]) and not ret(a[0]): a=[a[1],a[0]];node.args=a
   ev=a[1]
   bool_names={'gt','ge','lt','le','eq','ne','and_','or_','not_','ashare_limit_up_touch','ashare_limit_down_touch'}
   is_bool=isinstance(ev,ast.Call) and isinstance(ev.func,ast.Name) and ev.func.id in bool_names
   if not is_bool: node.args[1]=C('gt',C('abs',ev),ast.Constant(0.0))
   self.m('类型修正：事件响应统一为 response,event 顺序，数值事件转为非零事件掩码。');return node
  if n in {'ts_transfer_entropy_peak_strength','ts_transfer_entropy_peak_lag'} and {'x','y'}<=kw.keys():
   node.keywords=[ast.keyword('target',kw['x']),ast.keyword('source',kw['y'])];self.m(f'参数迁移：{n} 的 x/y 明确为 target/source，方向为 source→target。');return node
  if n.startswith('ts_vector_') and {'x','y'}<=kw.keys():
   node.keywords=[ast.keyword('f1',kw['x']),ast.keyword('f2',kw['y']),ast.keyword('window',ast.Constant(60))];self.m(f'参数迁移：{n} 改用 f1/f2，并明确60日窗口。');return node
  if n in {'directional_change_state','directional_change_extent'} and 'x' in kw:
   node.keywords=[ast.keyword('price',kw['x']),ast.keyword('threshold',ast.Constant(.02))];self.m(f'参数补全：{n} 将 x 明确为 price，方向变化阈值定为2%。');return node
  if n in {'ts_conditional_mutual_information','ts_distance_correlation_partial_proxy'} and 'z' not in kw:
   node.keywords.append(ast.keyword('z',ast.Name('turnover_ratio',ast.Load())));self.m(f'参数补全：{n} 以换手率作为条件变量 z。');return node
  if n=='ts_expectile' and isinstance(kw.get('tau'),ast.Name) and kw['tau'].id in {'low','high'}:
   v=.1 if kw['tau'].id=='low' else .9
   node.keywords=[ast.keyword(k.arg,ast.Constant(v) if k.arg=='tau' else k.value) for k in node.keywords];self.m(f'新约定：ts_expectile 的 {kw["tau"].id} 明确定义为 tau={v}。');return node
  if n=='ts_interval_nesting_depth' and len(a)!=3 or n=='ts_interval_nesting_depth' and not isinstance(a[2],ast.Constant): node.args=[ast.Name('low',ast.Load()),ast.Name('high',ast.Load()),ast.Constant('inside')];self.m('定义修正：区间嵌套深度使用同源复权 low/high，并明确 inside 模式。');return node
  if n=='ts_stratified_mean_spread' and (len(a)!=5 or not isinstance(a[2],ast.Constant)): node.args=[a[0],ast.Name('turnover_ratio',ast.Load()),ast.Constant(60),ast.Constant(.2),ast.Constant(20)];self.m('参数补全：按60日换手率上下20%分层计算目标均值差，至少20期。');return node
  if n=='ts_weighted_drawdown_area' and (len(a)!=3 or not isinstance(a[2],ast.Constant)): node.args=[ast.Name('close',ast.Load()),ast.Name('turnover_ratio',ast.Load()),ast.Constant(60)];self.m('类型与参数修正：使用复权收盘价水平和换手率计算60日加权回撤面积。');return node
  if n=='ts_run_concentration' and (len(a)!=4 or not isinstance(a[2],ast.Constant)): node.args=[a[0],C('gt',a[0],ast.Constant(0.)),ast.Constant(60),ast.Constant(5)];self.m('参数补全：以正负方向状态划分run，最长60期、至少5期。');return node
  if n=='ts_energy_break_score' and (len(a)!=6 or not isinstance(a[3],ast.Constant)): node.args=[a[0],ast.Name('turnover_ratio',ast.Load()),ast.Name('volume',ast.Load()),ast.Constant(60),ast.Constant(20),ast.Constant(60)];self.m('参数补全：收益、换手率、成交量三维联合分布，60日稳健标准化并比较近20/前60日。');return node
  if n in {'ts_first_passage_conditional_time','ts_first_passage_hit_probability'} and (len(a)!=8 or not isinstance(a[2],ast.Constant)): node.args=[a[0],C('ts_std',a[0],ast.Constant(60)),ast.Constant(120),ast.Constant(1.),ast.Constant(10),ast.Constant(3),ast.Constant(1),ast.Constant('upper')];self.m(f'参数补全：{n} 使用60日尺度、120日历史、1倍障碍和10日期限。');return node
  if n in {'ts_threshold_cycle_period','ts_threshold_cycle_asymmetry'} and len(a)!=4: node.args=[a[0],ast.Constant(-1.),ast.Constant(1.),ast.Constant(60)];self.m(f'参数补全：{n} 使用[-1,1]滞回阈值和60日窗口。');return node
  if n=='ts_price_delay' and (len(a)!=3 or not isinstance(a[2],ast.Constant)):
   ic=C('source_col',ast.Constant('BenchmarkIndexDailyBar'),ast.Constant('Close'),ast.Constant('index'),ast.Constant('000300.SH'));ip=C('source_col',ast.Constant('BenchmarkIndexDailyBar'),ast.Constant('PreClose'),ast.Constant('index'),ast.Constant('000300.SH'));ir=C('subtract',C('safe_div_null',ic,ip),ast.Constant(1.))
   node.args=[a[0],ir,ast.Constant(60)];self.m('参数补全：价格延迟相对沪深300日收益计算，明确指数身份和60日窗口。');return node
  if n=='ts_turnover_profit_share' and (len(a)!=3 or not isinstance(a[2],ast.Constant)): node.args=[ast.Name('close',ast.Load()),ast.Name('turnover_ratio',ast.Load()),ast.Constant(60)];self.m('参数补全：使用复权收盘价、换手率和60日存活筹码窗口。');return node
  if n=='state_episode_excursion_balance' and len(a)==3: node.args=a[:2];self.m('参数修正：状态偏移余额仅保留路径与状态，删除误传第三序列。');return node
  if n in {'ts_quantile_transport_slope','ts_quantile_transport_curvature'} and 'window' in kw:
   node.keywords=[ast.keyword('recent_window',ast.Constant(20)),ast.keyword('old_window',ast.Constant(40))];self.m(f'新约定：{n} 定义为近20日对比此前40日。');return node
  if n in {'ts_markov_state_entropy','ts_ordinal_irreversibility'} and 'min_periods' in kw:
   dest='min_count' if n.startswith('ts_markov') else 'min_patterns'
   for k in node.keywords:
    if k.arg=='min_periods': k.arg=dest
   self.m(f'参数迁移：{n} 的 min_periods 改为声明的 {dest}。');return node
  if n in {'cs_rank_copula_entropy','cs_rank_copula_mi'} and 'grid' in kw:
   for k in node.keywords:
    if k.arg=='grid': k.value=ast.Constant(8)
   self.m(f'参数规范化：{n} 的 grid=10 改为受支持网格8。');return node
  if n=='cs_knn_local_linear_residual' and 'k' in kw:
   for k in node.keywords:
    if k.arg=='k': k.value=ast.Constant(20)
   self.m('参数规范化：KNN邻居数提升至受支持下限20。');return node
  if n=='ts_tail_ratio' and len(a)>=5: node.args=[a[0],a[1],ast.Constant(.1),ast.Constant(.9),a[4]];self.m('参数修正：尾部比率明确采用下/上分位0.1/0.9。');return node
  if n=='ts_transition_count' and len(a)>=3 and not (isinstance(a[0],ast.Call) and isinstance(a[0].func,ast.Name) and a[0].func.id in {'gt','ge','lt','le','eq','ne'}): node.args=[C('gt',a[0],a[2]),a[1],ast.Constant('break')];self.m('类型修正：状态切换将收益相对阈值转为布尔方向状态。');return node
  if n=='ts_gap_fill_ratio' and len(a)==4 and isinstance(a[3],ast.Constant) and isinstance(a[3].value,float): node.args[3]=ast.Constant(60);self.m('参数修正：缺口修复率的0.01误作窗口，明确为60日窗口。');return node
  if n=='ts_conditional_transfer_entropy' and isinstance(kw.get('bins'),ast.Constant) and kw['bins'].value==5:
   for k in node.keywords:
    if k.arg=='bins': k.value=ast.Constant(3)
    if k.arg=='window': k.value=ast.Constant(300)
   self.m('参数规范化：条件转移熵 bins=5 改为受支持的3箱。');return node
  if n=='ts_conditional_transfer_entropy' and isinstance(kw.get('bins'),ast.Constant) and kw['bins'].value==3:
   for k in node.keywords:
    if k.arg=='window' and isinstance(k.value,ast.Constant) and k.value.value<244: k.value=ast.Constant(300)
   self.m('参数规范化：3箱条件转移熵窗口提升至300期，覆盖状态转移最低样本约束。');return node
  if n=='cs_shrinkage_mahalanobis' and len(a)==3: node.args.append(C('ts_std',ast.Name('ret',ast.Load()),ast.Constant(20)));self.m('参数补全：马氏距离第四维采用20日收益波动。');return node
  if n in {'intra_realized_beta_ex_self','intra_idiosyncratic_skewness_ex_self','intra_idiosyncratic_variance_ex_self','intra_market_model_r2_ex_self'} and len(a)==1: node.args.append(ast.Name('free_market_cap',ast.Load()));self.m(f'参数补全：{n} 以自由流通市值构造剔除自身的市场组合。');return node
  if n in {'fin_borrowing_intensity','fin_debt_repayment_intensity'} and len(a)==2: node.args.append(C('field',ast.Constant('report_period_end_date'),table=ast.Constant('StockCashFlow')));self.m(f'参数补全：{n} 使用现金流报告期末作为 period_id。');return node
  if n=='group_multi_level_rank_consistency' and len(a)==2: node.args.extend([a[1],a[1]]);self.m('缺省定义：仅有行业层级时，group2/group3 沿用同一行业码，表示单层排序一致性。');return node
  if n=='ts_current_drawdown_duration' and a:
   isret=isinstance(a[0],ast.Call) and isinstance(a[0].func,ast.Name) and a[0].func.id=='field' and a[0].args and isinstance(a[0].args[0],ast.Constant) and a[0].args[0].value=='ret'
   if isret: node.args[0]=C('field',ast.Constant('close'),table=ast.Constant('StockDailyBarAdj'));self.m('类型修正：收益输入改为同源复权收盘价水平，再计算回撤持续期。');return node
   if isinstance(a[0],ast.Call) and isinstance(a[0].func,ast.Name) and a[0].func.id=='exp': node.args[0]=C('field',ast.Constant('close'),table=ast.Constant('StockDailyBarAdj'));self.m('类型修正：收益累计式仍携带收益类型，改用同源复权收盘价水平计算回撤持续期。');return node
  if n in {'ashare_limit_up_streak','ashare_limit_down_streak','ashare_limit_open_up_streak','ashare_limit_open_down_streak','ashare_limit_up_touch','ashare_limit_down_touch'}:
   changed=False
   for z in a:
    if isinstance(z,ast.Call) and isinstance(z.func,ast.Name) and z.func.id=='field' and z.args and isinstance(z.args[0],ast.Constant) and z.args[0].value in {'high_limit','low_limit'}:
     for k in z.keywords:
      if k.arg=='table' and isinstance(k.value,ast.Constant) and k.value.value!='StockDailyBar': k.value=ast.Constant('StockDailyBar');changed=True
   if changed:self.m(f'类型修正：{n} 的涨跌停价改为StockDailyBar官方raw价格。')
  if n in {'ashare_limit_up_streak','ashare_limit_down_streak','ashare_limit_open_up_streak','ashare_limit_open_down_streak'} and len(a)==5:
   up='up' in n; opn='open_' in n; price=a[0] if opn else a[1];limit=C('field',ast.Constant('high_limit' if up else 'low_limit'),table=ast.Constant('StockDailyBar'))
   node.args=[price,limit,C('gt',a[4],ast.Constant(0.0)),ast.Constant(.005)];self.m(f'参数修正：{n} 改为价格、对应涨跌停价、成交有效掩码与0.005元绝对价格容差。');return node
  if n=='ashare_limit_up_touch' and len(a)==5: node.args=[C('field',ast.Constant('high'),table=ast.Constant('StockDailyBar')),C('field',ast.Constant('high_limit'),table=ast.Constant('StockDailyBar')),ast.Constant(.005)];self.m('参数修正：涨停触及改为日内最高价、raw官方涨停价与0.005元绝对价格容差。');return node
  if n=='ashare_limit_one_price' and len(a)==5:
   o,c,up,down,v=a;node.args=[o,C('field',ast.Constant('high'),table=ast.Constant('StockDailyBar')),C('field',ast.Constant('low'),table=ast.Constant('StockDailyBar')),c,up,down];node.keywords=[ast.keyword('side',ast.Constant('up')),ast.keyword('tick_tolerance',ast.Constant(.005))];self.m('参数修正：一字涨停补齐 raw high/low，明确 side=up 与0.5%容差。');return node
  if n=='ADX' and len(a)==5: node.args=[a[1],a[2],a[3],ast.Constant(14)];self.m('参数修正：ADX 改为 high/low/close/14。');return node
  if n=='RSX' and len(a)==5: node.args=[a[3],ast.Constant(14)];self.m('参数修正：RSX 改为 close/14。');return node
  x=a[0] if a else None
  if n=='identity' and x is not None: self.m('等价迁移：identity(x) 直接保留 x。');return x
  if n in {'holder_concentration_change','holder_count_change_rate'}:
   table='StockTopTenShareholder'
   node.args=[C('field',ast.Constant('ShareholderId'),table=ast.Constant(table)),C('field',ast.Constant('ShareRatio'),table=ast.Constant(table))]
   self.m(f'字段消歧：{n} 明确使用非流通限定的十大股东表ShareholderId/ShareRatio。');return node
  if n in {'ACF','pacf','durbin_watson_test','kpss_test','stationarity_test'} and x is not None: self.m(f'替代定义：{n} 定义为60日一阶自相关状态。');return C('ts_corr',x,C('ts_delay',x,ast.Constant(1)),ast.Constant(60))
  if n in {'corr_test','kendall_corr_test','spearman_corr_test','ttest_paired','ttest_two_samples','granger_causality'} and len(a)>1: self.m(f'替代定义：{n} 定义为两序列60日滚动相关状态。');return C('ts_corr',a[0],a[1],ast.Constant(60))
  if n in {'bartlett_test','chi_square_test','jarque_bera_test','ks_test','levene_test','lilliefors_test','ttest_one_sample'} and x is not None: self.m(f'替代定义：{n} 定义为60日标准化偏离强度。');return C('abs',C('ts_zscore',x,ast.Constant(60)))
  if n=='causal_linear_extrapolate' and x is not None: self.m('替代定义：因果线性外推=当前值+最近一期斜率。');return C('add',x,C('ts_delta',x,ast.Constant(1)))
  if n=='convolve' and x is not None: self.m('标量化定义：缺少卷积核的旧式convolve明确为20日因果线性衰减卷积。');return C('ts_decay_linear',x,ast.Constant(20))
  if n in {'filter_lowpass','wavelet_denoise'} and x is not None: self.m(f'标量化定义：{n} 明确为20日因果低通均值。');return C('ts_mean',x,ast.Constant(20))
  if n in {'idft','ifft'} and x is not None: self.m(f'类型修正：{n} 的输入是时域标量序列而非复频谱，逆变换按恒等映射保留原序列。');return x
  if n=='filter_highpass' and x is not None: self.m('替代定义：高通=当前值减20日均值。');return C('subtract',x,C('ts_mean',x,ast.Constant(20)))
  if n in {'filter_bandpass','filter_notch'} and x is not None: self.m(f'替代定义：{n}=5日均值减20日均值。');return C('subtract',C('ts_mean',x,ast.Constant(5)),C('ts_mean',x,ast.Constant(20)))
  if n=='decimate' and x is not None: self.m('替代定义：decimate 定义为两期滞后采样。');return C('ts_delay',x,ast.Constant(2))
  if n in {'dft','fft'} and x is not None: self.m(f'标量化定义：{n} 以60日通用信号谱熵输出频域无序度，避免向因子列塞入复向量。');return C('ts_signal_spectral_entropy',x,ast.Constant(60))
  if n=='phase' and x is not None: self.m('标量化定义：phase以短长均线差的符号表达60日窗口内主导相位方向。');return C('sign',C('subtract',C('ts_mean',x,ast.Constant(5)),C('ts_mean',x,ast.Constant(20))))
  if n=='wavelet' and x is not None: self.m('标量化定义：wavelet以5日与20日因果尺度分量之差表达细节系数。');return C('subtract',C('ts_mean',x,ast.Constant(5)),C('ts_mean',x,ast.Constant(20)))
  if n=='volatility' and x is not None: self.m('FactorRecipe迁移：volatility 精确展开为20日收益标准差乘sqrt(252)，min_periods沿正式ts_std契约。');return C('multiply',C('ts_std',x,ast.Constant(20)),C('sqrt',ast.Constant(252.0)))
  if n=='sharpe_ratio' and x is not None: self.m('FactorRecipe迁移：sharpe_ratio 精确展开为60日均值/标准差乘sqrt(252)。');return C('multiply',C('safe_div_null',C('ts_mean',x,ast.Constant(60)),C('ts_std',x,ast.Constant(60))),C('sqrt',ast.Constant(252.0)))
  if n=='cumulative_returns' and x is not None: self.m('因果定义迁移：收益输入定义为252日复合收益，避免旧价格型实现误把收益当价格。');return C('subtract',C('exp',C('ts_sum',C('log',C('add',x,ast.Constant(1.))),ast.Constant(252))),ast.Constant(1.))
  if n=='downside_beta' and len(a)==1:
   ic=C('source_col',ast.Constant('BenchmarkIndexDailyBar'),ast.Constant('Close'),ast.Constant('index'),ast.Constant('000300.SH'));ip=C('source_col',ast.Constant('BenchmarkIndexDailyBar'),ast.Constant('PreClose'),ast.Constant('index'),ast.Constant('000300.SH'));bench=C('subtract',C('safe_div_null',ic,ip),ast.Constant(1.))
   self.m('适配接通：downside_beta相对沪深300收益，使用正式60日下行Beta内核（至少3个下跌样本且完整窗口）。');return C('recipe_downside_beta',x,bench,ast.Constant(60))
  if n in {'vp_weighted_price','vpmacd','vpmacd_signal'} and len(a)>1:
   close,volume=a[:2];open_=C('field',ast.Constant('open'),table=ast.Constant('StockDailyBarAdj'));high=C('field',ast.Constant('high'),table=ast.Constant('StockDailyBarAdj'));low=C('field',ast.Constant('low'),table=ast.Constant('StockDailyBarAdj'))
   target={'vp_weighted_price':'recipe_vp_weighted_price','vpmacd':'recipe_vpmacd','vpmacd_signal':'recipe_vpmacd_signal'}[n]
   last=ast.Constant(20 if n=='vp_weighted_price' else .9)
   suffix='window=20' if n=='vp_weighted_price' else 'lambda=0.9'
   self.m(f'FactorRecipe精确展开：{n} 接入共享量价加权价格与VP-MACD真实内核，补齐同源复权OHLC和{suffix}。');return C(target,close,volume,open_,high,low,last)
  if n in {'vpmacd','vpmacd_signal'} and len(a)>1:
   vp=C('safe_div',C('ts_sum',C('multiply',a[0],a[1]),ast.Constant(20)),C('ts_sum',a[1],ast.Constant(20)));m=C('subtract',C('ts_mean',vp,ast.Constant(12)),C('ts_mean',vp,ast.Constant(26)));self.m(f'旧名迁移：{n} 定义为量价加权价格12/26日均线差。');return C('sign',C('ts_delta',m,ast.Constant(1))) if n.endswith('signal') else m
  if n in {'holder_concentration_change','holder_count_change_rate'} and len(a)>1: self.m(f'替代定义：{n} 以第一大股东持股比例一期变化刻画结构变动。');return C('ts_delta',C('field',ast.Constant('share_ratio'),table=ast.Constant('StockTopTenShareholder')),ast.Constant(1))
  if n in {'micro_spread','micro_trade_imbalance','micro_vpin','micro_amihud_hf'}:
   mh=C('field',ast.Constant('high'),table=ast.Constant('StockMinuteBar'));ml=C('field',ast.Constant('low'),table=ast.Constant('StockMinuteBar'));mc=C('field',ast.Constant('close'),table=ast.Constant('StockMinuteBar'));mv=C('field',ast.Constant('volume'),table=ast.Constant('StockMinuteBar'))
   target='recipe_'+n
   args=[mh,ml,mc] if n=='micro_spread' else [mc,mv]
   if n in {'micro_trade_imbalance','micro_vpin'}:args.extend([ast.Constant(20),ast.Constant(2)])
   self.m(f'适配接通：{n} 将MinuteOHLCVA占位符替换为明确分钟字段，并调用原始session-aware微观结构内核。');return C(target,*args)
  if n in {'ttm','yoy','quarter'} and x is not None:
   period=C('field',ast.Constant('report_period_end_date'),table=ast.Constant('StockIncome'))
   qid=C('ashare_fiscal_quarter_from_period_end',period)
   quarter=C('fin_quarter_from_cumulative',x,period,qid)
   if n=='ttm': self.m('财务旧名迁移：累计利润表流量先转单季，再以报告期标识使用 fin_ttm 计算正式TTM。');return C('fin_ttm',quarter,period)
   if n=='quarter': self.m('财务旧名迁移：累计利润表流量按报告期转换为正式单季值。');return quarter
   self.m('财务旧名迁移：累计利润表流量先转单季，再按报告期计算正式同比。');return C('fin_yoy',quarter,period)
  return node

def migrate_formula(formula,logic=''):
 t=ast.parse(formula,mode='eval');r=R();t=r.visit(t);ast.fix_missing_locations(t);return ast.unparse(t),r.ch
