import gzip,json,hashlib
from pathlib import Path
p=Path(__file__).with_name("r20-holder-macro-smoke-input.jsonl.gz")
with gzip.open(p,"xt") as f:
    for i,name in enumerate(("holder_top10_daily_id_churn","holder_top10_weighted_std","holder_top10_pledge_ratio")):
        row={"source_row":i+2,"id":f"R20_{name}","formula":f"{name}('StockTopTenShareholder')"}
        f.write(json.dumps(row)+"\n")
print(hashlib.sha256(p.read_bytes()).hexdigest())
