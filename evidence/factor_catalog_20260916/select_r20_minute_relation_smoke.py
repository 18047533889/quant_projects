#!/usr/bin/env python3
import gzip,json,re
from pathlib import Path
E=Path(__file__).resolve().parent
xs=[json.loads(l) for l in open(E/'r20_minute_relation_proposals.jsonl')]
chosen=[]; seen=set()
for x in xs:
 f=x['current_formula']; minute='StockMinuteBarAdj' in str(x['bindings'])
 rel='StockTopTenShareholder' in str(x['bindings'])
 if not (minute or rel): continue
 m=re.search(r'\b((?:intra|intraday|session)_\w+)\(',f); fam=m.group(1) if m else ('relation_'+re.search(r'(relation_\w+)\(',f).group(1) if 'relation_' in f else 'other')
 if fam in seen: continue
 if minute and sum('StockMinuteBarAdj' in str(z['bindings']) for z in chosen)>=16: continue
 if rel and sum('StockTopTenShareholder' in str(z['bindings']) for z in chosen)>=4: continue
 seen.add(fam); chosen.append(x)
 if len(chosen)>=20: break
for x in xs:
 if 'StockTopTenShareholder' in str(x['bindings']) and x not in chosen and sum('StockTopTenShareholder' in str(z['bindings']) for z in chosen)<4: chosen.append(x)
with gzip.open(E/'r20-minute-relation-smoke-input.jsonl.gz','wt') as o:
 for x in chosen:
  tables=sorted({b['table'] for b in x['bindings']}); fields=sorted({b['table']+'.'+b['column'] for b in x['bindings']})
  o.write(json.dumps({'source_row':x['source_row'],'id':x['id'],'formula':x['current_formula'],'fields':'|'.join(fields),'tables':'|'.join(tables),'domain':'intraday_or_relation','pit':'PIT'},ensure_ascii=False)+'\n')
print(json.dumps({'selected':len(chosen),'rows':[x['source_row'] for x in chosen]},ensure_ascii=False))
