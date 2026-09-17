import gzip,json,hashlib
from pathlib import Path
p=Path(__file__).with_name("r20-rank-smoke-input.jsonl.gz")
with gzip.open(p,"xt") as f:
    for rank in (1,2):
        row={"source_row":rank+1,"id":f"R20_rank_identity_{rank}",
             "formula":f"source_col('StockTopTenShareholder','ShareNumber','ShareholderRank',{rank})"}
        f.write(json.dumps(row)+"\n")
print(hashlib.sha256(p.read_bytes()).hexdigest())
