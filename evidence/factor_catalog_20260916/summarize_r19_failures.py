"""Summarize actual compile failures without conflating binding errors with absent data."""
import argparse,collections,gzip,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument("evidence",type=Path);p.add_argument("output",type=Path)
args=p.parse_args();counts=collections.Counter();examples=collections.defaultdict(list);total=0
with gzip.open(args.evidence,"rt") as f:
 for line in f:
  r=json.loads(line)
  if r["status"]!="COMPILE_FAILED":continue
  total+=1;e=r["error"]
  if "FIELD_BINDING_FAILED" in e:kind="字段未绑定：缺少字段、来源表或指数身份；不等于已证实无数据"
  elif e.startswith("DSLUnknownOperatorError"):kind="未注册算子或旧名称"
  elif e.startswith("DSLParseError"):kind="表达式语法、文字占位或参数数量限制"
  elif e.startswith("OperatorParameterError"):kind="参数名、数量、范围或类型错误"
  elif "InputContractError" in e:kind="输入语义或单位不匹配"
  elif "Imputation" in e:kind="禁止填造未披露财务数据"
  elif "Invariant" in e:kind="优化前后语义不一致"
  else:kind=e.split(":")[0]
  counts[kind]+=1
  if len(examples[kind])<5:examples[kind].append({k:r[k] for k in ("source_row","id","current_formula","error")})
result={"evidence":str(args.evidence),"total_failed":total,"counts":dict(counts),"examples":dict(examples)}
with args.output.open("x",encoding="utf-8") as f:json.dump(result,f,ensure_ascii=False,indent=2)
print(json.dumps({"total_failed":total,"counts":dict(counts)},ensure_ascii=False))
