"""Bounded parallel compilation, no reads or factor publication."""
import argparse,concurrent.futures,json,os,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--prefix",required=True)
    ap.add_argument("--proposals",required=True)
    args=ap.parse_args()
    available=next(int(x.split()[1])*1024 for x in Path("/proc/meminfo").read_text().splitlines() if x.startswith("MemAvailable:"))
    slots=min(8,max(1,int(available*.5)//(2*1024**3)))
    env=dict(os.environ,PYTHONPATH="evidence/factor_catalog_20260915:.",OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1")
    def run(shard):
        prefix=f"{args.prefix}-s{shard}"
        cmd=[str(ROOT/".venv/bin/python"),"evidence/r3/test_watchdog.py","--log",prefix+".log",
             "--max-rss-mib","2048","--timeout","1800","--",str(ROOT/".venv/bin/python"),
             "evidence/factor_catalog_20260916/compile_r20_full.py",
             "--source","evidence/factor_catalog_20260916/factor_catalog_review_r19_final.csv.gz",
             "--prefix",prefix,"--shard",str(shard),"--shards","8","--proposals",args.proposals]
        result=subprocess.run(cmd,cwd=ROOT,env=env,capture_output=True,text=True)
        print(json.dumps({"shard":shard,"returncode":result.returncode,"watchdog":result.stdout[-3000:],"stderr":result.stderr[-1000:]}),flush=True)
        return result.returncode
    print(json.dumps({"slots":slots,"available_bytes":available,"shards":8}),flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=slots) as pool:
        codes=list(pool.map(run,range(8)))
    if any(codes):raise SystemExit(1)
if __name__=="__main__":main()
