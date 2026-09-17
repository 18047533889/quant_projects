from __future__ import annotations
import glob,gzip,json,sys
from collections import Counter
from pathlib import Path
ROOT=Path('/home/sunhaiwei/quant_projects'); E=ROOT/'evidence/factor_catalog_20260916'
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'evidence/factor_catalog_20260915'))
from factor_engine.tools.catalog_r20_fields_proposals import redesign
from factor_engine.tools.catalog_r20_root_recipes import migrate_formula
def main():
 from compile_catalog import build_runtime
 from smoke_catalog import bind_fields
 from factor_engine.api.factor import Factor
 xs=[]
 for p in sorted(glob.glob(str(E/'r20-extra-s*.jsonl.gz'))):
  for x in map(json.loads,gzip.open(p,'rt')):
   if x.get('status')=='COMPILE_FAILED' and 'FIELD_BINDING_FAILED' in x.get('error','') and not any(z in x.get('error','') for z in ('MinuteOHLCVA','shareholder_id_','share_rank_')): xs.append(x)
 cs={int(x['source_row']):x['original_context'] for x in map(json.loads,gzip.open(E/'r20_additional_original_context.jsonl.gz','rt'))}
 p,en=build_runtime(); out=[]
 for x in xs:
  c=cs[int(x['source_row'])]; f,n,ne=redesign(x['current_formula'],c); f,rn=migrate_formula(f); n.extend(rn); changes=list(x.get('changes') or [])+n
  r=dict(source_row=x['source_row'],id=x['id'],before_formula=x['before_formula'],current_formula=f,changes=changes,current_definition='原始经济解释：'+str(c.get('经济解释') or '原记录未提供')+'；当前定义：'+'；'.join(n),semantic_redesign=bool(ne or any('SEMANTIC_REDESIGN' in z for z in changes)),compile_status='COMPILE_FAILED',error='',bindings=[],original_error=x.get('error',''))
  try:
   q=p.parse(f); b,fail=bind_fields(q)
   if fail: raise ValueError('FIELD_BINDING_FAILED: '+json.dumps(fail,ensure_ascii=False))
   en.compile(Factor(name=str(x['id']),expr=q,source_expr=f,surface='compat_research')); r['compile_status']='COMPILED'; r['bindings']=b
  except Exception as ex:r['error']=f'{type(ex).__name__}: {ex}'
  out.append(r)
 dst=E/'r20_fields_additional_proposals.jsonl'; dst.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in out))
 print(json.dumps({'rows':len(out),'status':Counter(x['compile_status'] for x in out),'errors':Counter(x['error'][:280] for x in out if x['error']).most_common(40)},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
