#!/usr/bin/env python3
"""Repair R20 extra MinuteOHLCVA and shareholder-rank binding failures."""
from __future__ import annotations
import argparse, ast, gzip, glob, importlib.util, json, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]; E=ROOT/'evidence/factor_catalog_20260916'; sys.path.insert(0,str(ROOT))
F=lambda n,t: f"field('{n}', table='{t}')"
MC=F('minute_close','StockMinuteBarAdj'); MO=F('minute_open','StockMinuteBarAdj'); MH=F('minute_high','StockMinuteBarAdj'); ML=F('minute_low','StockMinuteBarAdj')
MV=F('minute_volume','StockMinuteBarAdj'); MA=F('minute_amount','StockMinuteBarAdj'); MW=F('minute_vwap','StockMinuteBarAdj')
MR=f"subtract(safe_div_null({MC}, ts_lag({MC}, 1)), 1.0)"; FLOW=f"multiply(sign({MR}), {MA})"; EVENT=f"gt(abs(ts_zscore({MR}, 60)), 2.0)"; STATE=f"sign({MR})"

def runtime():
 p=ROOT/'evidence/factor_catalog_20260915/compile_catalog.py'; s=importlib.util.spec_from_file_location('xcompile',p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); parser,engine=m.build_runtime()
 p=ROOT/'evidence/factor_catalog_20260915/smoke_catalog.py'; s=importlib.util.spec_from_file_location('xsmoke',p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return parser,engine,m.bind_fields

def minute_args(op, manifest):
 ins=(manifest.get(op) or {}).get('input_fields') or []
 d={'close':MC,'price':MC,'x':MR,'y':f'ts_zscore({MV}, 60)','returns':MR,'ret':MR,'open':MO,'high':MH,'low':ML,'volume':MV,'amount':MA,'activity':MV,'value':MV,'flow':FLOW,
    'event':EVENT,'event_mask':EVENT,'state':STATE,'state_a':STATE,'state_b':f'sign(ts_delta({MV}, 1))','industry':'industry_code','session_id':F('quote_time','StockMinuteBarAdj'),
    'high_limit':F('high_limit','StockDailyBar'),'low_limit':F('low_limit','StockDailyBar'),'pre_close':F('pre_close','StockDailyBarAdj'),'locked':'eq('+MH+', '+ML+')'}
 return [d.get(x, MR) for x in ins]

class Bundle(ast.NodeTransformer):
 def __init__(self,manifest): self.manifest=manifest; self.changed=[]
 def visit_Call(self,node):
  self.generic_visit(node)
  if isinstance(node.func,ast.Name) and node.args and isinstance(node.args[0],ast.Name) and node.args[0].id=='MinuteOHLCVA':
   args=minute_args(node.func.id,self.manifest)
   node.args=[ast.parse(x,mode='eval').body for x in args]+node.args[1:]; self.changed.append((node.func.id,args))
  return node

class RequiredArgs(ast.NodeTransformer):
 def visit_Call(self,node):
  self.generic_visit(node)
  if not isinstance(node.func,ast.Name): return node
  n=node.func.id
  scalar=lambda x: ast.Constant(value=x)
  if n in ('intra_state_follow_beta','intra_state_follow_corr','intra_state_follow_ratio','intra_state_interval_moment','intra_state_sum','intra_state_vwap') and len(node.args) in (1,2,3): node.args.append(scalar(1.0))
  elif n=='intra_state_pair_same_slot_corr' and len(node.args)==2: node.args=[node.args[0],scalar(1.0),node.args[1],scalar(1.0)]
  elif n=='ts_run_efficiency' and len(node.args)==1: node.args.append(ast.parse(f'gt({ast.unparse(node.args[0])}, 0.0)',mode='eval').body)
  elif n=='ts_weighted_expected_shortfall' and len(node.args)==1: node.args.append(ast.parse(MV,mode='eval').body)
  elif n=='intraday_bvc_imbalance' and len(node.args)==3: node.args=[node.args[0],node.args[1],scalar(20),node.args[2]]
  elif n in ('intraday_volume_clock_path_efficiency','intraday_volume_clock_roughness') and len(node.args)==3: node.args=[node.args[0],node.args[1],scalar(16),node.args[2]]
  return node

class SevenMinuteArgs(ast.NodeTransformer):
 def __init__(self,manifest): self.manifest=manifest; self.changed=[]
 def visit_Call(self,node):
  self.generic_visit(node)
  names=[a.id for a in node.args[:7] if isinstance(a,ast.Name)]
  if isinstance(node.func,ast.Name) and names==['MinuteOpen','MinuteHigh','MinuteLow','MinuteClose','MinuteVolume','MinuteAmount','MinuteVwap']:
   args=minute_args(node.func.id,self.manifest); node.args=[ast.parse(x,mode='eval').body for x in args]+node.args[7:]; self.changed.append(node.func.id)
  return node

class FreeMarketCap(ast.NodeTransformer):
 def visit_Call(self,node):
  self.generic_visit(node)
  if isinstance(node.func,ast.Name) and node.func.id in ('intra_idiosyncratic_skewness_ex_self','intra_idiosyncratic_variance_ex_self','intra_market_model_r2_ex_self','intra_realized_beta_ex_self') and len(node.args)==1:
   node.args.append(ast.parse("field('free_market_cap', table='StockValuationDaily')",mode='eval').body)
  return node

def minute_formula(f,manifest):
 t=ast.parse(f,mode='eval'); v=Bundle(manifest); t=v.visit(t); s=SevenMinuteArgs(manifest); t=s.visit(t); t=RequiredArgs().visit(t); ast.fix_missing_locations(t)
 if not v.changed and not s.changed: raise ValueError('NO_MINUTE_BUNDLE_CALL')
 return ast.unparse(t.body), '将 MinuteOHLCVA 占位包按算子正式输入签名展开为 StockMinuteBarAdj 的真实分钟 OHLCVA、分钟收益/事件/状态输入'

def freecap_formula(f):
 t=ast.parse(f,mode='eval'); t=FreeMarketCap().visit(t); ast.fix_missing_locations(t); return ast.unparse(t.body),'为分钟横截面市场模型显式绑定 StockValuationDaily.FreeMarketCap'

def relation_base(name):
 if 'entropy_change' in name: return "ts_delta(relation_weighted_change(share_number, share_ratio), 1)", '真实前十大股东持股加权变化的一阶变化'
 if 'hhi_change' in name or 'concentration_acceleration' in name: return "ts_delta(abs(relation_weighted_change(share_number, share_ratio)), 1)", '真实前十大股东持股集中变化的加速度'
 if 'weighted_change' in name: return "relation_weighted_change(share_number, share_ratio)", '真实前十大股东持股数量按持股比例加权变化'
 if 'weighted_std' in name or 'distribution_skew' in name or 'kurtosis' in name: return "ts_std(relation_weighted_change(share_number, share_ratio), 8)", '真实前十大股东持股加权变化的跨报告期离散度'
 if 'diffusion' in name or 'entropy' in name: return "relation_distinct_count(shareholder_id)", '真实前十大股东实体关系广度'
 if 'rank_weighted' in name: return "abs(relation_weighted_change(shareholder_rank, share_ratio))", '真实前十大股东排名按持股比例加权变化'
 return "abs(relation_weighted_change(share_number, share_ratio))", '真实前十大股东持股集中度变化'

def relation_formula(row):
 name=(row.get('original_context') or {}).get('因子名称',''); b,desc=relation_base(name)
 if name.endswith('_direct'): return b,desc
 if '_return_interaction' in name: rhs=F('ret','StockDailyBarAdj')
 elif '_turnover_interaction' in name: rhs='turnover_ratio'
 elif '_freefloat_interaction' in name: rhs='safe_div_null(market_cap, free_market_cap)'
 elif '_pledge_proxy_interaction' in name: rhs=f"holder_pledge_ratio({F('share_pledge','StockTopTenShareholder')}, {F('share_number','StockTopTenShareholder')})"
 elif '_goodwill_interaction' in name: rhs='safe_div_null(goodwill, total_assets)'
 else: raise ValueError('UNKNOWN_RELATION_SUFFIX '+name)
 return f'multiply({b}, {rhs})',desc+'并与命名状态交互'

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--input',default=str(E/'r20_additional_original_context.jsonl.gz')); ap.add_argument('--output',default=str(E/'r20_minute_relation_proposals.jsonl')); a=ap.parse_args()
 originals=[json.loads(l) for l in gzip.open(a.input,'rt')]; om={x['source_row']:x for x in originals}
 latest={}
 for p in glob.glob(str(E/'r20-extra-s*.jsonl.gz')):
  for l in gzip.open(p,'rt'): x=json.loads(l); latest[x['source_row']]=x
 ids={x['source_row'] for x in originals if any(k.lower() in x.get('error','').lower() for k in ('MinuteOHLCVA','shareholder_id_','share_rank_'))}
 seven=('MinuteOpen','MinuteHigh','MinuteLow','MinuteClose','MinuteVolume','MinuteAmount','MinuteVwap')
 ids.update(sr for sr,x in latest.items() if x.get('status')!='COMPILED' and all(k in x.get('current_formula','') for k in seven))
 ids.update(sr for sr,x in latest.items() if x.get('status')!='COMPILED' and 'free_market_cap' in x.get('error','').lower() and 'intra_' in x.get('current_formula',''))
 rows=[]
 for sr in sorted(ids):
  r=dict(om[sr]); r['_work_formula']=latest.get(sr,r).get('current_formula',r['current_formula']); rows.append(r)
 man=json.load(open(ROOT/'benchmarks/operator_manifest.json')); manifest={x['name']:x for x in man['operators']}; parser,engine,bind=runtime(); from factor_engine.api.factor import Factor
 out=[]; bad=[]
 for r in rows:
  try:
   wf=r['_work_formula']
   if 'MinuteOHLCVA' in r['error'] or all(k in wf for k in seven): f,d=minute_formula(wf,manifest)
   elif 'free_market_cap' in latest.get(r['source_row'],{}).get('error','').lower(): f,d=freecap_formula(wf)
   else: f,d=relation_formula(r)
   e=parser.parse(f); bs,bf=bind(e)
   if bf: raise RuntimeError(json.dumps(bf,ensure_ascii=False))
   engine.compile(Factor(name=str(r['id']),expr=e,source_expr=f,surface='compat_research'))
   out.append({'source_row':r['source_row'],'id':r['id'],'before_formula':r['current_formula'],'current_formula':f,'changes':list(r.get('changes') or [])+['语义重建（非等价）：'+d],'current_definition':d,'semantic_redesign':True,'compile_status':'COMPILED','error':'','bindings':bs,'original_error':r['error']})
  except Exception as ex: bad.append({'source_row':r['source_row'],'id':r['id'],'formula':locals().get('f'),'error':type(ex).__name__+': '+str(ex)})
 Path(a.output).write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in out)); print(json.dumps({'target':len(rows),'compiled':len(out),'failed':len(bad),'failures':bad},ensure_ascii=False,indent=2)); raise SystemExit(bool(bad))
if __name__=='__main__': main()
