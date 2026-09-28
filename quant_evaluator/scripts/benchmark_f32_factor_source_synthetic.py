"""On-demand synthetic F32 source benchmark; never reads COS."""
from __future__ import annotations
import argparse, json, resource, subprocess, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from quant_evaluator.api.factor_source import evaluate_factor_source_batch
from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.factor_tile_source import FactorTile
from quant_evaluator.contracts.label_bundle import LabelBundle

T,N,F,TILE=2586,5461,32,2
METRICS=("rank_ic","rank_ic_series")
MIN_RAM=24*1024**3
MIN_VRAM=14*1024**3

class SyntheticSource:
    def __init__(self,seed):
        self.factor_ids=tuple(f"f{i}" for i in range(F))
        self.time_axis=AxisRef("time","int64",T,np.arange(T,dtype=np.int64))
        self.asset_axis=AxisRef("asset","int64",N,np.arange(N,dtype=np.int64))
        self.dtype="float64"; self.snapshot_id=f"synthetic-f32-seed-{seed}"
        self.max_tile_size=TILE; self.seed=seed; self.reads=[]; self.closed=False
        self.seed_offset=seed-442
    def read_tile(self,start,end):
        if self.closed or not 0<=start<end<=F or end-start>TILE: raise ValueError("invalid tile")
        # Match the measured panel exactly: each factor has its own RNG stream.
        x=np.empty((T,N,end-start),dtype=np.float64)
        for offset,k in enumerate(range(start,end)):
            x[:,:,offset]=np.random.default_rng(10000+k+self.seed_offset).standard_normal((T,N))
        batch=FactorBatch(self.factor_ids[start:end],self.time_axis,self.asset_axis,x)
        self.reads.append((start,end)); return FactorTile(start,end,batch,self.snapshot_id)
    def close(self): self.closed=True

def available_ram():
    try:
        for line in open("/proc/meminfo",encoding="ascii"):
            if line.startswith("MemAvailable:"): return int(line.split()[1])*1024
    except OSError: pass
    return None

def gpu_snapshot():
    try:
        p=subprocess.run(["nvidia-smi","--query-gpu=name,memory.total,memory.free","--format=csv,noheader,nounits"],capture_output=True,text=True,timeout=5)
        if p.returncode: return {"status":"unavailable","detail":p.stderr.strip()[:300]}
        ds=[]
        for line in p.stdout.splitlines():
            name,total,free=(v.strip() for v in line.split(",",2)); free=int(free)
            eff=int(free*.4); ds.append({"name":name,"total_mib":int(total),"free_mib":free,"effective_free_mib":eff,"passes":eff*1024**2>=MIN_VRAM})
        return {"status":"ok","devices":ds}
    except Exception as e: return {"status":"unavailable","detail":f"{type(e).__name__}: {e}"}

def preflight(ram,gpu):
    rp=ram is not None and ram>=MIN_RAM
    devices=gpu.get("devices",[])
    gp=(gpu.get("status")=="ok" and len(devices)==1
        and devices[0].get("name")=="NVIDIA L20" and devices[0].get("passes") is True)
    return {"pass":rp and gp,"ram_available_bytes":ram,"ram_minimum_bytes":MIN_RAM,"ram_pass":rp,"gpu_memory":gpu,"gpu_pass":gp,"factor_tile_bytes":T*N*TILE*8,"factor_tensor_materialized":False,"disk_panel_bytes":0}

def labels_for(seed):
    ta=AxisRef("time","int64",T,np.arange(T,dtype=np.int64)); aa=AxisRef("asset","int64",N,np.arange(N,dtype=np.int64))
    times=tuple(range(T)); y=np.random.default_rng(seed).standard_normal((T,N))
    return LabelBundle("synthetic_forward_return",y,1,decision_time=times,label_start_time=times,label_end_time=tuple(t+1 for t in times),asset_axis=aa,source_ref="synthetic:benchmark",calendar_ref="synthetic:integer_axis")

def run_one(backend,seed):
    src=SyntheticSource(seed); y=labels_for(seed); tic=time.perf_counter()
    result=evaluate_factor_source_batch(src,y,metrics=METRICS,backend=backend,max_tile_size=TILE)
    sec=time.perf_counter()-tic
    expected=[(i,min(i+TILE,F)) for i in range(0,F,TILE)]
    assert src.reads==expected and result.factor_ids==src.factor_ids
    row={"backend_requested":backend,"backend_used":result.metadata.get("backend_used"),"auto_backend_reason":result.metadata.get("auto_backend_reason"),"seconds":sec,"factor_tiles_processed":result.metadata.get("factor_tiles_processed"),"source_tile_reads":len(src.reads),"peak_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,"metadata":result.metadata}
    vals={"metrics":{m:np.array(result.get_metric(m),copy=True) for m in METRICS},
          "observation_counts":{m:np.array(result.observation_counts[m],copy=True) for m in METRICS}}
    src.close(); return row,vals

def compare(a,b):
    out={}
    for m in METRICS:
        left,right=a["metrics"][m],b["metrics"][m]
        np.testing.assert_allclose(left,right,rtol=1e-8,atol=1e-10,equal_nan=True)
        ok=np.isfinite(left)&np.isfinite(right); err=float(np.max(np.abs(left[ok]-right[ok]))) if ok.any() else 0.
        counts_a,counts_b=a["observation_counts"][m],b["observation_counts"][m]
        np.testing.assert_array_equal(counts_a,counts_b)
        out[m]={"metric_values":{"shape":list(left.shape),"allclose":True,"rtol":1e-8,"atol":1e-10,"max_abs_error":err},
                "observation_counts":{"shape":list(counts_a.shape),"equal":True}}
    return out

def run_full(seed,output):
    gate=preflight(available_ram(),gpu_snapshot())
    if not gate["pass"]: raise RuntimeError(f"preflight failed: {json.dumps(gate)}")
    order=("cpu","cuda_strict","cpu","auto"); runs=[]; vals=[]
    for b in order:
        row,v=run_one(b,seed); runs.append(row); vals.append(v)
        print(f"{b}: {row['seconds']:.3f}s route={row['backend_used']} tiles={row['source_tile_reads']}",flush=True)
    parity={"cpu_repeat":compare(vals[0],vals[2]),"cuda_vs_cpu":compare(vals[0],vals[1]),"auto_vs_cpu":compare(vals[0],vals[3])}
    assert runs[1]["backend_used"]==runs[3]["backend_used"]=="cuda"
    report={"status":"PASS","created_utc":datetime.now(timezone.utc).isoformat(),"source_kind":"deterministic_synthetic_on_demand","real_cos_data_used":False,"cos_accessed":False,"shape_T_N_F":[T,N,F],"factor_dtype":"float64","tile_width":TILE,"metrics":METRICS,"seed":seed,"preflight":gate,"runs":runs,"parity":parity,"pass":True,"limitations":["Synthetic timings are not COS I/O or production throughput evidence.","RAM is host available-memory snapshot; RSS is process high-water.","VRAM preflight uses free memory times 40% safety fraction."]}
    if output: output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+"\n")
    print(json.dumps({"status":"PASS","seconds":[round(x["seconds"],3) for x in runs],"routes":[x["backend_used"] for x in runs],"parity":"pass"}),flush=True)
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--run-full",action="store_true"); p.add_argument("--seed",type=int,default=442); p.add_argument("--output",type=Path); a=p.parse_args()
    gate=preflight(available_ram(),gpu_snapshot())
    if not a.run_full:
        print(json.dumps({"mode":"preflight_only","synthetic_only":True,"shape_T_N_F":[T,N,F],"tile_width":TILE,"metrics":METRICS,"preflight":gate},indent=2)); return
    if not gate["pass"]: raise SystemExit(f"preflight failed; not run: {json.dumps(gate)}")
    run_full(a.seed,a.output)
if __name__=="__main__": main()
