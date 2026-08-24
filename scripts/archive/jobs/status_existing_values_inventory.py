#!/usr/bin/env python3
"""Inventory existing COS value artifacts without relabeling alpha factors as Status features."""
from __future__ import annotations
import argparse, collections, hashlib, json, re
from pathlib import Path

REGISTRY = Path('/home/sunhaiwei/.quantsociety/Status_Stage0_Master_Feature_Registry_587_AI执行版_V1.html')

def registry_rows(path):
    text=path.read_text(encoding='utf-8')
    m=re.search(r'<script id="registry-json" type="application/json">(.*?)</script>',text,re.S)
    if not m: raise ValueError('registry-json missing')
    return json.loads(m.group(1))

def parse_listing(path):
    rows=[]
    for line in path.read_text(errors='replace').splitlines():
        if '|' not in line: continue
        key=line.split('|',1)[0].strip()
        if not key or key in {'KEY','TOTAL OBJECTS:'}: continue
        if key.startswith(('factor_pool/','candidate_pool/')):
            fields=[x.strip() for x in line.split('|')]
            rows.append({'key':key,'type':fields[1] if len(fields)>1 else None,'modified':fields[2] if len(fields)>2 else None,'size':fields[4] if len(fields)>4 else None,'source_listing':str(path)})
    return rows

def parent_prefix(key):
    parts=key.split('/')
    if key.startswith('factor_pool/ashare/') and len(parts)>=5: return '/'.join(parts[:5])
    if key.startswith('candidate_pool/hsunbj/'):
        if 'factor_values/' in key:
            i=parts.index('factor_values'); return '/'.join(parts[:min(len(parts),i+4)])
        if 'ashare/status/' in key: return '/'.join(parts[:6])
        return '/'.join(parts[:4])
    return '/'.join(parts[:3])

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--factor-listing',required=True); ap.add_argument('--candidate-listing',required=True); ap.add_argument('--output-dir',required=True); args=ap.parse_args()
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    rows=parse_listing(Path(args.factor_listing))+parse_listing(Path(args.candidate_listing))
    groups=collections.defaultdict(lambda:{'objects':0,'bytes_text':collections.Counter(),'keys_sample':[],'parquet_objects':0,'manifest_objects':0,'date_min':None,'date_max':None})
    for row in rows:
        g=groups[parent_prefix(row['key'])]; g['objects']+=1
        if len(g['keys_sample'])<8: g['keys_sample'].append(row['key'])
        if row['key'].endswith('.parquet'): g['parquet_objects']+=1
        if '/manifest' in row['key'] or row['key'].endswith('/afv.json') or row['key'].endswith('/published.json'): g['manifest_objects']+=1
        m=re.search(r'/(\d{4}-\d{2}-\d{2})\.parquet$',row['key'])
        if m:
            d=m.group(1); g['date_min']=d if g['date_min'] is None or d<g['date_min'] else g['date_min']; g['date_max']=d if g['date_max'] is None or d>g['date_max'] else g['date_max']
    reg=registry_rows(REGISTRY)
    buckets=collections.Counter(r.get('exec_bucket') for r in reg)
    inventory={'inventory_version':'status-existing-values-v1','registry_rows':len(reg),'registry_buckets':buckets,'source_files':[str(Path(args.factor_listing)),str(Path(args.candidate_listing))],'objects_total':len(rows),'groups':dict(groups),'classification':{'status_features':{'keys':['candidate_pool/hsunbj/ashare/status/'],'note':'Only outputs with Status manifest and registry linkage are Status artifacts.'},'alpha_factor_values':{'prefixes':['factor_pool/ashare/','candidate_pool/hsunbj/factor_values/'],'note':'Existing alpha/GTJA/CogAlpha values remain separate and are not relabeled.'},'blocked_unmapped':{'reason':'No canonical Status feature_id mapping in the observed path/metadata.'}}}
    # JSON serializable counters
    inventory['registry_buckets']=dict(buckets)
    for g in inventory['groups'].values(): g['bytes_text']=dict(g['bytes_text'])
    (out/'existing_values_inventory.json').write_text(json.dumps(inventory,ensure_ascii=False,indent=2,default=str)+'\n')
    blocked=[]
    for row in rows:
        if row['key'].startswith(('factor_pool/','candidate_pool/hsunbj/factor_values/')):
            blocked.append({'source_key':row['key'],'classification':'UNMAPPED_ALPHA_FACTOR','reason':'No one-to-one Status registry feature_id mapping; retain original factor namespace.'})
    (out/'unmapped_existing_values.json').write_text(json.dumps(blocked,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'objects':len(rows),'groups':len(groups),'status_source_objects':sum(1 for r in rows if '/ashare/status/' in r['key']),'unmapped_objects':len(blocked)},ensure_ascii=False))
if __name__=='__main__': main()
