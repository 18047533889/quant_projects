from __future__ import annotations
import gzip,json,re,sys
from pathlib import Path
ROOT=Path('/home/sunhaiwei/quant_projects'); E=ROOT/'evidence/factor_catalog_20260916'; sys.path.insert(0,str(ROOT/'evidence/factor_catalog_20260915'))
BC='source_col("BenchmarkIndexDailyBar","Close","index","000300.SH")'; BP='source_col("BenchmarkIndexDailyBar","PreClose","index","000300.SH")'; IW='source_col("IndexConstituent","Weight","IndexSymbol","000300.SH")'
BR=f'subtract(safe_div_null({BC},{BP}),1.0)'; IM=f'gt({IW},0.0)'; PL='holder_pledge_ratio(field("share_pledge", table="StockTopTenShareholder"), field("share_number", table="StockTopTenShareholder"))'; RP='field("report_period_end_date", table="StockIndicator")'; TRIO='field("roe", table="StockIndicator"), field("gross_profit_margin", table="StockIndicator"), field("net_profit_margin", table="StockIndicator"), '+RP
def ranked(table,col,start=1,end=10): return [f'source_col("{table}","{col}","ShareholderRank",{i})' for i in range(start,end+1)]
def strict_sum(xs):
 out=xs[0]
 for x in xs[1:]: out=f'add({out},{x})'
 return out
TOP_R=ranked('StockTopTenShareholder','ShareRatio'); FLOAT_R=ranked('StockTopTenFloatShareholder','ShareRatio')
TOP_HHI='relation_hhi('+','.join(TOP_R)+')'; FLOAT_HHI='relation_hhi('+','.join(FLOAT_R)+')'
TOP_SUM=strict_sum(TOP_R); FLOAT_SUM=strict_sum(FLOAT_R)
def replace_bare_name(f,name,replacement):
 import ast
 try:
  tree=ast.parse(f,mode='eval'); rep=ast.parse(replacement,mode='eval').body
  class Replace(ast.NodeTransformer):
   def visit_Name(self,node): return rep if node.id==name else node
  return ast.unparse(ast.fix_missing_locations(Replace().visit(tree)).body)
 except Exception:return f
def sn(f,n,r): return replace_bare_name(f,n,r)
def rc(f,name,repl):
 start=f.find(name+'(')
 if start<0:return f
 i=start+len(name); depth=0
 while i<len(f):
  if f[i]=='(': depth+=1
  elif f[i]==')':
   depth-=1
   if depth==0:return f[:start]+repl+f[i+1:]
  i+=1
 return f
def latent_repairs(f,n):
 f=f.replace("safe_div_null(field('ret', table='StockDailyBarAdj'), 10000)","field('ret', table='StockDailyBarAdj')").replace('safe_div_null(field("ret", table="StockDailyBarAdj"), 10000)','field("ret", table="StockDailyBarAdj")')
 f=f.replace('tau=low','tau=0.1').replace('tau=high','tau=0.9')
 f=f.replace('ts_transfer_entropy_peak_strength(x=','ts_transfer_entropy_peak_strength(target=').replace(', y=',', source=')
 f=f.replace('ts_transfer_entropy_peak_lag(x=','ts_transfer_entropy_peak_lag(target=').replace(', y=',', source=')
 f=f.replace('ts_cross_spectral_coherence(x=', 'ts_cross_spectral_coherence(x=').replace('ts_cross_spectral_coherence(x=field(\'ret\', table=\'StockDailyBarAdj\'), source=', 'ts_cross_spectral_coherence(x=field(\'ret\', table=\'StockDailyBarAdj\'), y=')
 f=f.replace('ts_cross_spectral_phase(x=field(\'ret\', table=\'StockDailyBarAdj\'), source=', 'ts_cross_spectral_phase(x=field(\'ret\', table=\'StockDailyBarAdj\'), y=')
 f=f.replace("ts_first_passage_bias(x=log(close), scale=ts_std(ret,20), side='symmetric')","ts_first_passage_bias(x=log(close), scale=ts_std(ret,20))")
 f=f.replace('ts_spectral_entropy(x=subtract(overnight_return(open,pre_close),open_close_return(open,close)))','ts_spectral_flatness(subtract(overnight_return(open,pre_close),open_close_return(open,close)),60)')
 f=f.replace('index_entry_exit_event(IndexSymbol)',f'index_entry_exit_event({IM})')
 if 'index_weight_change(' in f: f=rc(f,'index_weight_change',f'index_weight_change({IW},20)'); n.append('index_weight_change改为沪深300真实成分权重的20日变化')
 rawup='ashare_limit_up_touch(field("high",table="StockDailyBar"),field("high_limit",table="StockDailyBar"),0.005)'
 rawdn='ashare_limit_down_touch(field("low",table="StockDailyBar"),field("low_limit",table="StockDailyBar"),0.005)'
 valid='subtract(1.0,field("is_suspend",table="StockDailyBar"))'
 reps=[('ashare_limit_down_streak',f'ashare_limit_down_streak(field("close",table="StockDailyBar"),field("low_limit",table="StockDailyBar"),{valid},0.005)'),('ashare_limit_open_down_streak',f'ashare_limit_open_down_streak(field("open",table="StockDailyBar"),field("low_limit",table="StockDailyBar"),{valid},0.005)'),('ashare_limit_open_up_streak',f'ashare_limit_open_up_streak(field("open",table="StockDailyBar"),field("high_limit",table="StockDailyBar"),{valid},0.005)'),('ashare_limit_up_streak',f'ashare_limit_up_streak(field("close",table="StockDailyBar"),field("high_limit",table="StockDailyBar"),{valid},0.005)'),('ashare_limit_touch_count','ashare_limit_touch_count(field("high",table="StockDailyBar"),field("low",table="StockDailyBar"),field("high_limit",table="StockDailyBar"),field("low_limit",table="StockDailyBar"),20,"up",0.005)'),('ashare_limit_up_touch',rawup),('ashare_limit_one_price','ashare_limit_one_price(field("open",table="StockDailyBar"),field("high",table="StockDailyBar"),field("low",table="StockDailyBar"),field("close",table="StockDailyBar"),field("high_limit",table="StockDailyBar"),field("low_limit",table="StockDailyBar"),"up",0.005)')]
 for name,repl in reps:
  if name+'(' in f: f=rc(f,name,repl); n.append(f'{name}畸形模板按正式签名修复，保留原算子语义')
 # A generated subset reversed response/event positional arguments.
 import ast
 try:
  t=ast.parse(f,mode='eval')
  for x in ast.walk(t):
   if isinstance(x,ast.Call) and isinstance(x.func,ast.Name) and x.func.id=='ts_first_passage_bias':
    kept=[]
    for kw in x.keywords:
     if kw.arg=='side' and isinstance(kw.value,ast.Constant) and kw.value.value=='symmetric':
      n.append('ts_first_passage_bias双边bias使用正式默认定义，删除冗余symmetric参数')
     else: kept.append(kw)
    x.keywords=kept
   if isinstance(x,ast.Call) and isinstance(x.func,ast.Name) and x.func.id=='ts_spectral_entropy':
    signal=x.args[0] if x.args else next((kw.value for kw in x.keywords if kw.arg=='x'),None)
    text=ast.unparse(signal) if signal is not None else ''
    if any(token in text for token in ('PriceContinuous','price_continuous','overnight_return','open_close_return')):
     x.func.id='ts_signal_spectral_entropy'; n.append('连续价格信号使用research-only ts_signal_spectral_entropy，不套用收益输入门禁')
   if isinstance(x,ast.Call) and isinstance(x.func,ast.Name) and x.func.id=='event_historical_response_mean' and len(x.args)>=2:
    a=ast.unparse(x.args[0]); b=ast.unparse(x.args[1])
    if ('ret' in b and any(k in a for k in ('surprise','magnitude','fiscal_pct_change','index_weight_change'))): x.args[0],x.args[1]=x.args[1],ast.Call(func=ast.Name(id='ne',ctx=ast.Load()),args=[x.args[0],ast.Constant(value=0.0)],keywords=[]); n.append('修复历史事件响应算子的response/event反置，并将非零事件强度正式化为布尔事件')
  f=ast.unparse(t.body)
 except Exception: pass
 return f
def align_financial_periods(f,n):
 """Bind fiscal period inputs to the same statement as the measured value."""
 import ast
 from factor_engine.fields import FIELD_REGISTRY
 financial={'StockBalance','StockIncome','StockCashFlow','StockIndicator'}
 source_tables={}
 for spec in FIELD_REGISTRY.fields():
  if spec.table in financial:
   source_tables.setdefault(str(spec.source_name).lower(),set()).add(spec.table)
   source_tables.setdefault(str(spec.name).lower(),set()).add(spec.table)
 def value_table(node):
  found=[]
  for x in ast.walk(node):
   if isinstance(x,ast.Call) and isinstance(x.func,ast.Name) and x.func.id=='field':
    for kw in x.keywords:
     if kw.arg=='table' and isinstance(kw.value,ast.Constant) and kw.value.value in financial: found.append(kw.value.value)
   elif isinstance(x,ast.Name):
    ts=source_tables.get(x.id.lower(),set())
    if len(ts)==1: found.extend(ts)
  return found[0] if found and len(set(found))==1 else None
 def period_field(table):
  return ast.Call(func=ast.Name(id='field',ctx=ast.Load()),args=[ast.Constant('report_period_end_date')],keywords=[ast.keyword(arg='table',value=ast.Constant(table))])
 try:
  tree=ast.parse(f,mode='eval'); changed=[]
  for call in ast.walk(tree):
   if not isinstance(call,ast.Call) or not isinstance(call.func,ast.Name): continue
   name=call.func.id; value=None; period_pos=None
   if name in {'ttm_from_cumulative','fin_quarter_from_cumulative','period_lag','period_average','period_change','period_pct_change'} and call.args:
    value=call.args[0]; period_pos=1
   elif name.startswith('fiscal_'):
    value=next((kw.value for kw in call.keywords if kw.arg in {'x','signal'}),call.args[0] if call.args else None); period_pos=1
   table=value_table(value) if value is not None else None
   if table and period_pos is not None:
    if len(call.args)>period_pos: call.args[period_pos]=period_field(table); changed.append((name,table))
    else:
     for kw in call.keywords:
      if kw.arg in {'period_id','fiscal_period_id'}: kw.value=period_field(table); changed.append((name,table))
   if name=='report_revision_magnitude' and call.args:
    table=value_table(call.args[0])
    if table:
     value=call.args[0]; prior=ast.Call(ast.Name('delay',ast.Load()),[value,ast.Constant(1)],[])
     period=period_field(table); previous=ast.Call(ast.Name('delay',ast.Load()),[period_field(table),ast.Constant(1)],[])
     event=ast.Call(ast.Name('and_',ast.Load()),[ast.Call(ast.Name('eq',ast.Load()),[period,previous],[]),ast.Call(ast.Name('ne',ast.Load()),[value,prior],[])],[])
     call.args=[value,prior,period,previous,event]; call.keywords=[]; changed.append((name,table))
  if changed:
   f=ast.unparse(tree.body); n.append('财务值与报告期字段按同一来源表对齐：'+','.join(sorted({t for _,t in changed})))
 except Exception:
  pass
 return f
def collapse_expanded_price_delay(f,n):
 import ast
 try:
  tree=ast.parse(f,mode='eval')
  class Collapse(ast.NodeTransformer):
   def visit_Call(self,node):
    node=self.generic_visit(node)
    if isinstance(node.func,ast.Name) and node.func.id=='safe_div_null':
     calls=[x.func.id for x in ast.walk(node) if isinstance(x,ast.Call) and isinstance(x.func,ast.Name)]
     if calls.count('ts_corr')>=5 and calls.count('source_col')>=2:
      n.append('重复展开的价格延迟恒等改写为正式ts_price_delay算子，窗口60、最大滞后4')
      return ast.Call(ast.Name('ts_price_delay',ast.Load()),[ast.Call(ast.Name('field',ast.Load()),[ast.Constant('ret')],[ast.keyword('table',ast.Constant('StockDailyBarAdj'))]),ast.parse(BR,mode='eval').body,ast.Constant(60),ast.Constant(4),ast.Constant(20)],[])
    return node
  tree=Collapse().visit(tree); return ast.unparse(ast.fix_missing_locations(tree).body)
 except Exception:return f
def redesign(f,c):
 n=[]; ne=False
 try:
  import ast
  from factor_engine.fields import FIELD_REGISTRY
  names={x.id for x in ast.walk(ast.parse(f,mode='eval')) if isinstance(x,ast.Name) and x.id.endswith('_SP')}
  for old in sorted(names,key=len,reverse=True):
   base=old[:-3]; specs=[s for s in FIELD_REGISTRY.fields() if s.table in {'StockIncome','StockCashFlow'} and str(s.source_name).lower()==base.lower()]
   if len(specs)==1:
    s=specs[0]; pd=f'field("report_period_end_date",table="{s.table}")'; rep=f'fin_quarter_from_cumulative(field("{s.name}",table="{s.table}"),{pd},ashare_fiscal_quarter_from_period_end({pd}))'; f=sn(f,old,rep); n.append(f'{old}按{s.table}累计流量和报告期重建为真实单季值')
 except Exception: pass
 d={'edge_spread':('ts_edge_effective_spread(open,high,low,close,20)','edge_spread定义为20日EDGE有效价差'),'oi_spread':('subtract(overnight_return(open,pre_close),open_close_return(open,close))','oi_spread定义为隔夜减日内收益'),'chip_profit_share':('ts_turnover_profit_share(close,turnover_ratio,60)','chip_profit_share定义为60日换手存活筹码获利占比'),'chip_mode_distance':('ts_turnover_cost_mode_distance(close,turnover_ratio,60)','chip_mode_distance定义为60日筹码主成本峰距离'),'return_volatility':('ts_std(ret,20)','return_volatility定义为20日收益波动率'),'IndexClose':(BC,'IndexClose绑定沪深300收盘'),'IndexWeight':(IW,'IndexWeight绑定沪深300成分权重'),'Top10PledgeRatio':('holder_top10_pledge_ratio("StockTopTenShareholder")','Top10PledgeRatio由十个真实rank槽聚合质押股数/持股数'),'is_st':('field("public_status_code",table="StockStatus")','is_st改用真实上市状态代码识别状态转换'),'contract_liability':('field("contract_liabilities",table="StockBalance")','合同负债绑定StockBalance'),'symmetric':("'symmetric'",'symmetric修复为枚举字符串')}
 for x,(r,z) in d.items():
  if re.search(rf'\b{x}\b',f): f=sn(f,x,r); n.append(z)
 if re.search(r'\bIndexReturn\b',f): f=sn(f,'IndexReturn',f'multiply({BR},10000.0)'); n.append('IndexReturn绑定沪深300收益并保留原基点尺度')
 for x,r,z in [('index_entry',f'gt(index_entry_exit_event({IM}),0.0)','index_entry定义为沪深300纳入事件'),('index_exit',f'lt(index_entry_exit_event({IM}),0.0)','index_exit定义为沪深300剔除事件')]:
  if re.search(rf'\b{x}\b',f): f=sn(f,x,r); n.append(z)
 if re.search(r'\bcapital_supply_shock\b',f): x='field("total_capital",table="StockCapitalDaily")'; f=sn(f,'capital_supply_shock',f'ne({x},delay({x},1))'); n.append('capital_supply_shock定义为总股本变化事件')
 for x,r,z in [('SHORT','ts_sum(ret,5)','SHORT按原解释定义为5日短周期动量；原式外层neg表示短周期反转'),('VOLCONF','ts_std(ret,20)','VOLCONF不可用，非等价用20日波动率代理'),('VOL','ts_std(ret,20)','VOL不可用，非等价用20日波动率')]:
  if re.search(rf'\b{x}\b',f): f=sn(f,x,r); n.append(z); ne=True
 for op in ('report_change_breadth','report_change_coherence'):
  p=rf'{op}\(ReportVector\s*,\s*FiscalPeriodId\)'
  if re.search(p,f): f=re.sub(p,f'{op}({TRIO})',f); n.append(f'ReportVector非字段，非等价改为ROE/毛利率/净利率三维{op}'); ne=True
 tabs=str(c.get('输入表') or c.get('original_tables') or '')
 if re.search(r'\bPubDate\b',f): cs=[t for t in ('StockIndicator','StockIncome','StockCashFlow','StockBalance') if t in tabs]; t=cs[0] if len(cs)==1 else 'StockIndicator'; f=sn(f,'PubDate',f'field("pub_date",table="{t}")'); n.append(f'PubDate绑定{t}')
 if re.search(r'\bReportPeriodEndDate\b',f): t='StockBalance' if 'StockBalance' in tabs and 'StockIndicator' not in tabs else 'StockIndicator'; f=sn(f,'ReportPeriodEndDate',f'field("report_period_end_date",table="{t}")'); n.append(f'ReportPeriodEndDate绑定{t}')
 if re.search(r'\bFiscalPeriodId\b',f):
  t='StockCashFlow' if 'StockCashFlow' in tabs and 'StockIndicator' not in tabs else ('StockIndicator' if 'StockIndicator' in tabs else ('StockTopTenShareholder' if 'StockTopTenShareholder' in tabs and 'StockBalance' not in tabs else 'StockBalance')); f=sn(f,'FiscalPeriodId',f'field("report_period_end_date",table="{t}")'); n.append(f'FiscalPeriodId绑定{t}报告期末')
 if re.search(r'\bperiod_id\b',f):
  t='StockCashFlow' if 'StockCashFlow' in f and 'StockIncome' not in f else ('StockIncome' if 'StockIncome' in f else 'StockIndicator'); pd=f'field("report_period_end_date",table="{t}")'; f=re.sub(r'\bperiod_id\b(?!\s*=)',pd,f); n.append(f'period_id绑定{t}报告期末')
 if re.search(r'\bfiscal_quarter\b',f):
  t='StockCashFlow' if 'StockCashFlow' in f and 'StockIncome' not in f else ('StockIncome' if 'StockIncome' in f else 'StockIndicator'); pd=f'field("report_period_end_date",table="{t}")'; f=sn(f,'fiscal_quarter',f'ashare_fiscal_quarter_from_period_end({pd})'); n.append(f'fiscal_quarter由{t}报告期末确定')
 freeze=ranked('StockTopTenShareholder','ShareFreeze'); number=ranked('StockTopTenShareholder','ShareNumber')
 for x,r,z in [('Top10HHI',TOP_HHI,'Top10HHI由十个真实rank持股比例按观测top10口径计算HHI'),('FloatTop10HHI',FLOAT_HHI,'FloatTop10HHI由十个真实流通股东rank持股比例计算HHI'),('Top1ShareRatio',TOP_R[0],'Top1ShareRatio绑定真实第1名持股比例'),('Top2to10ShareRatio',strict_sum(TOP_R[1:]),'Top2to10ShareRatio为真实rank2至10持股比例严格求和；任一披露未知则null'),('Top10FreezeRatio','safe_div_null('+strict_sum(freeze)+','+strict_sum(number)+')','Top10FreezeRatio由十个真实rank冻结股数/持股数严格合计构造；未知披露不填0'),('ILL','amihud_illiquidity(ret,close,volume,20)','ILL定义为20日Amihud非流动性'),('amount_change','ts_delta(amount,1)','amount_change定义为成交额一日变化'),('turnover_decimal','turnover_ratio','turnover_decimal绑定小数换手率')]:
  if re.search(rf'\b{x}\b',f): f=sn(f,x,r); n.append(z)
 if 'index_weight(' in f: f=rc(f,'index_weight',IW); n.append('旧index_weight草图改为沪深300真实成分权重源')
 for x,r,z in [('ShareNumber','field("share_number",table="StockTopTenShareholder")','ShareNumber绑定前十大股东'),('SharePledge','field("share_pledge",table="StockTopTenShareholder")','SharePledge绑定前十大股东')]:
  try:
   import ast
   present=any(isinstance(node,ast.Name) and node.id==x for node in ast.walk(ast.parse(f,mode='eval')))
  except Exception: present=False
  if present: f=replace_bare_name(f,x,r); n.append(z)
 if any(re.search(rf'\bshare_rank_{i}\b',f) for i in range(1,11)):
  m=re.search(r'(relation_category_share|relation_peer_weighted_mean_ex_self)\([^)]*\)',f)
  if m: f=f[:m.start()]+'holder_concentration(field("share_number",table="StockTopTenShareholder"))'+f[m.end():]; n.append('share_rank_1..10非字段，非等价改为前十大股东集中度'); ne=True
 # R20 additional batch: spell out generated placeholders using observable,
 # point-in-time inputs.  These are deliberately definitions, not aliases to
 # unrelated convenient columns.
 if 'capital_change_age(CapitalChangeDate)' in f:
  f=f.replace('capital_change_age(CapitalChangeDate)','ts_days_since(ne(field("total_capital",table="StockCapitalDaily"),delay(field("total_capital",table="StockCapitalDaily"),1)))'); n.append('股本变更日不可观测；变更年龄定义为总股本变化事件的交易日龄')
 explicit={
  'ShareholderId':('field("shareholder_id",table="StockTopTenShareholder")','ShareholderId明确绑定前十大股东表'),
  'ShareRatio':('field("share_ratio",table="StockTopTenShareholder")','ShareRatio明确绑定前十大股东表'),
  'IndexVolume':('source_col("BenchmarkIndexDailyBar","Volume","index","000300.SH")','IndexVolume明确绑定沪深300指数成交量'),
  'IndexAmount':('source_col("BenchmarkIndexDailyBar","Amount","index","000300.SH")','IndexAmount明确绑定沪深300指数成交额'),
  'filing_event':('ne(field("pub_date",table="StockIndicator"),delay(field("pub_date",table="StockIndicator"),1))','filing_event定义为可得公告日期变化事件'),
  'FiscalAvgAssets':('period_average(field("total_assets",table="StockBalance"),field("report_period_end_date",table="StockBalance"),periods=2,revision_policy="latest_available")','FiscalAvgAssets定义为相邻报告期平均总资产'),
  'EMA_fast':('ema(close,12)','EMA_fast定义为12日EMA'),
  'EMA_slow':('ema(close,26)','EMA_slow定义为26日EMA'),
  'EFF':('safe_div_null(abs(subtract(close,delay(close,20))),ts_sum(abs(subtract(close,delay(close,1))),20))','EFF定义为20日Kaufman路径效率比'),
 }
 for old,(rep,note) in explicit.items():
  try:
   import ast
   present=any(isinstance(x,ast.Name) and x.id==old for x in ast.walk(ast.parse(f,mode='eval')))
  except Exception: present=False
  if present: f=replace_bare_name(f,old,rep); n.append(note)
 positive_balance={
  'PositiveFinancialAsset_Trading':'trading_assets','PositiveFinancialAsset_Derivative':'derivative_financial_asset',
  'PositiveFinancialAsset_Bond':'bond_invest','PositiveFinancialAsset_OtherBond':'other_bond_invest',
  'PositiveFinancialAsset_Equity':'other_equity_tools_invest','PositiveFinancialAsset_OtherNC':'other_non_current_financial_assets',
  'PositiveFunding_ST':'shortterm_loan','PositiveFunding_1Y':'non_current_liability_in_one_year',
  'PositiveFunding_LT':'longterm_loan','PositiveFunding_Bond':'bonds_payable','PositiveFunding_Lease':'lease_liability',
  'PositiveOpLiab_AP':'accounts_payable','PositiveOpLiab_Notes':'notes_payable',
  'PositiveOpLiab_Contract':'contract_liabilities','PositiveOpLiab_Advance':'advance_peceipts',
  'PositiveOpLiab_Payroll':'employee_payable','PositiveOpLiab_Tax':'taxes_payable','PositiveOpLiab_Other':'other_payable',
 }
 for old,canon in positive_balance.items():
  if re.search(rf'\b{old}\b',f):
   f=sn(f,old,f'maximum(field("{canon}",table="StockBalance"),1e-12)'); n.append(f'{old}定义为StockBalance.{canon}的正值闭合分量')
 holder_defs={
  'Top2to5ShareRatio':strict_sum(TOP_R[1:5]),
  'Top10Ownership':TOP_SUM,
  'FloatTop10Ownership':FLOAT_SUM,
  'Top10Entropy':'relation_entropy('+','.join(TOP_R)+')',
  'FloatTop10Entropy':'relation_entropy('+','.join(FLOAT_R)+')',
  'Top10Churn':'holder_top10_daily_id_churn("StockTopTenShareholder")',
  'FloatTop10Churn':'holder_top10_daily_id_churn("StockTopTenFloatShareholder")',
  'Top10PledgeConcentration':'holder_concentration(field("share_pledge",table="StockTopTenShareholder"))',
 }
 for old,rep in holder_defs.items():
  if re.search(rf'\b{old}\b',f):
   f=sn(f,old,rep)
   if old.endswith('Churn'): n.append(f'{old}定义为交易日网格的新披露ID匹配变动事件：无变化日为0、信息未知为null，不将事件值跨日持久化')
   else: n.append(f'{old}由对应股东明细显式构造')
 states={
  'positive_momentum_state':'gt(ts_sum(field("ret",table="StockDailyBarAdj"),20),0.0)',
  'negative_momentum_state':'lt(ts_sum(field("ret",table="StockDailyBarAdj"),20),0.0)',
  'high_turnover_state':'gt(turnover_ratio,ts_mean(turnover_ratio,60))',
  'high_volume_state':'gt(volume,ts_mean(volume,60))',
  'high_vol_state':'gt(ts_std(ret,20),ts_mean(ts_std(ret,20),60))',
  'illiquidity_state':'gt(amihud_illiquidity(ret,close,volume,20),ts_mean(amihud_illiquidity(ret,close,volume,20),60))',
  'above_vwap_state':'gt(close,safe_div_null(ts_sum(amount,20),ts_sum(volume,20)))',
  'breakout_state':'gt(close,delay(ts_max(high,20),1))',
  'limit_stress_state':'or_(ashare_limit_up_touch(field("high",table="StockDailyBar"),field("high_limit",table="StockDailyBar"),0.005),ashare_limit_down_touch(field("low",table="StockDailyBar"),field("low_limit",table="StockDailyBar"),0.005))',
  'chip_supply_wall_state':'gt(ts_turnover_cost_mode_distance(close,turnover_ratio,60),0.0)',
  'drawdown_state':'lt(close,ts_max(close,60))',
  'chip_profit_state':'gt(ts_turnover_profit_share(close,turnover_ratio,60),0.5)',
  'opposite_or_recovery_state':'gt(field("ret",table="StockDailyBarAdj"),0.0)',
 }
 for old,rep in states.items():
  if re.search(rf'\b{old}\b',f): f=sn(f,old,rep); n.append(f'{old}按原始文字含义定义为可观测布尔状态')
 # Generated relation calls previously fed an unaggregated one-to-many column
 # into scalar slots.  Materialize the ten disclosed ranks explicitly.
 if 'holder_class_js_shift(' in f and ('ShareholderId' in f or 'shareholder_id' in f):
  f=rc(f,'holder_class_js_shift','holder_top10_daily_id_churn("StockTopTenShareholder")'); n.append('原式缺少五类股东聚合槽；明确重定义为前十股东ID匹配的日频披露变动事件'); ne=True
 for op,rep,note in [
  ('holder_company_ownership_hhi','holder_company_ownership_hhi('+','.join(TOP_R)+')','公司总股本口径HHI使用十个真实rank持股比例'),
  ('holder_observed_topk_hhi',TOP_HHI,'观测top10口径HHI使用十个真实rank持股比例'),
  ('holder_concentration',TOP_HHI,'股东集中度明确为观测top10口径HHI并使用十个真实rank持股比例')]:
  if op+'(' in f and ('StockTopTenShareholder' in f or 'ShareholderId' in f): f=rc(f,op,rep); n.append(note)
 f=f.replace("safe_div_null(field('share_pledge', table='StockTopTenShareholder'), field('share_number', table='StockTopTenShareholder'))",'holder_top10_pledge_ratio("StockTopTenShareholder")')
 f=f.replace('safe_div_null(field("share_pledge", table="StockTopTenShareholder"), field("share_number", table="StockTopTenShareholder"))','holder_top10_pledge_ratio("StockTopTenShareholder")')
 f=latent_repairs(f,n)
 f=align_financial_periods(f,n)
 f=align_financial_periods(f,n)
 f=collapse_expanded_price_delay(f,n)
 f=f.replace('ts_crossing_speed(x=', 'ts_crossing_speed(x=').replace(', source=',', y=') if 'ts_crossing_speed' in f else f
 f=f.replace('ts_crossing_acceleration(x=', 'ts_crossing_acceleration(x=').replace(', source=',', y=') if 'ts_crossing_acceleration' in f else f
 return f,n,ne
def main():
 from compile_catalog import build_runtime
 from smoke_catalog import bind_fields
 from factor_engine.api.factor import Factor
 xs=[json.loads(s) for s in gzip.open(E/'r19-combined-r5.jsonl.gz','rt')]; xs=[x for x in xs if x.get('status')=='COMPILE_FAILED' and 'FIELD_BINDING_FAILED' in x.get('error','')]; cs={int(x['source_row']):x['original_context'] for x in map(json.loads,gzip.open(E/'r20_original_context.jsonl.gz','rt'))}; p,en=build_runtime(); out=[]
 for x in xs:
  c=cs[int(x['source_row'])]; f,n,ne=redesign(x['current_formula'],c); r=dict(source_row=x['source_row'],id=x['id'],before_formula=x['current_formula'],current_formula=f,changes=n,current_definition='原始经济解释：'+str(c.get('经济解释') or c.get('logic') or '原记录未提供')+'；当前定义：'+'；'.join(n),semantic_redesign=ne,compile_status='COMPILE_FAILED',error='',bindings=[])
  try:
   q=p.parse(f); b,fail=bind_fields(q)
   if fail: raise ValueError('FIELD_BINDING_FAILED: '+json.dumps(fail,ensure_ascii=False))
   en.compile(Factor(name=str(x['id']),expr=q,source_expr=f,surface='compat_research')); r['compile_status']='COMPILED'; r['bindings']=b
  except Exception as ex: r['error']=f'{type(ex).__name__}: {ex}'
  out.append(r)
 (E/'r20_fields_proposals.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in out)); from collections import Counter; print(json.dumps({'rows':len(out),'status':Counter(r['compile_status'] for r in out),'errors':Counter(r['error'][:260] for r in out if r['error']).most_common(30)},ensure_ascii=False,indent=2))
if __name__=='__main__': main()
