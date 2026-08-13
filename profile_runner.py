#!/usr/bin/env python3
"""
Comprehensive profiling script for quant platform.
Profiles CPU, memory, and I/O hotspots across all major packages.
"""
import cProfile
import pstats
import io
import sys
import tracemalloc
import time
from pathlib import Path

# Add packages to path
ROOT = Path(__file__).parent
for pkg in ["factor_engine", "dataaccess", "factor_preprocess", "quant_evaluator", "factor_optimizer"]:
    sys.path.insert(0, str(ROOT / pkg))

def profile_cpu_hotspots():
    """Profile CPU-intensive operations."""
    print("\n" + "="*80)
    print("CPU PROFILING - Identifying computation hotspots")
    print("="*80)
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    results = {}
    
    # Profile 1: Factor Engine operations
    try:
        print("\n[1/5] Profiling Factor Engine...")
        from cleaned_operators import load_all
        load_all()
        
        from benchmarks.bench_fp_transforms import run_fp_benchmark
        t0 = time.perf_counter()
        fp_result = run_fp_benchmark()
        results['factor_engine'] = {
            'elapsed': time.perf_counter() - t0,
            'status': 'success'
        }
    except Exception as e:
        results['factor_engine'] = {'status': 'error', 'error': str(e)}
        print(f"  ERROR: {e}")
    
    # Profile 2: Quant Evaluator
    try:
        print("\n[2/5] Profiling Quant Evaluator...")
        from benchmarks.bench_qe_metrics import run_qe_benchmark
        t0 = time.perf_counter()
        qe_result = run_qe_benchmark()
        results['quant_evaluator'] = {
            'elapsed': time.perf_counter() - t0,
            'status': 'success'
        }
    except Exception as e:
        results['quant_evaluator'] = {'status': 'error', 'error': str(e)}
        print(f"  ERROR: {e}")
    
    # Profile 3: Factor Optimizer
    try:
        print("\n[3/5] Profiling Factor Optimizer...")
        from benchmarks.bench_fo_search import run_fo_benchmark
        t0 = time.perf_counter()
        fo_result = run_fo_benchmark()
        results['factor_optimizer'] = {
            'elapsed': time.perf_counter() - t0,
            'status': 'success'
        }
    except Exception as e:
        results['factor_optimizer'] = {'status': 'error', 'error': str(e)}
        print(f"  ERROR: {e}")
    
    # Profile 4: Factor Preprocess
    try:
        print("\n[4/5] Profiling Factor Preprocess...")
        from benchmarks.bench_fp_transforms import run_fp_benchmark
        t0 = time.perf_counter()
        # Reuse FP benchmark as proxy
        results['factor_preprocess'] = {
            'elapsed': 0.1,  # Placeholder
            'status': 'skipped'
        }
    except Exception as e:
        results['factor_preprocess'] = {'status': 'error', 'error': str(e)}
    
    # Profile 5: DataAccess
    try:
        print("\n[5/5] Profiling DataAccess...")
        # Lightweight test
        results['dataaccess'] = {
            'elapsed': 0.1,
            'status': 'skipped'
        }
    except Exception as e:
        results['dataaccess'] = {'status': 'error', 'error': str(e)}
    
    profiler.disable()
    
    # Analyze results
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s)
    ps.strip_dirs()
    ps.sort_stats('cumulative')
    ps.print_stats(50)  # Top 50 functions
    
    return s.getvalue(), results

def profile_memory():
    """Profile memory allocations."""
    print("\n" + "="*80)
    print("MEMORY PROFILING - Identifying allocation hotspots")
    print("="*80)
    
    tracemalloc.start()
    
    # Run lightweight benchmark
    try:
        from benchmarks.bench_fo_search import run_fo_benchmark
        print("\nRunning memory trace...")
        run_fo_benchmark()
    except Exception as e:
        print(f"ERROR: {e}")
    
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')
    
    tracemalloc.stop()
    
    lines = []
    lines.append("\nTop 20 Memory Allocations:")
    lines.append("-" * 80)
    for stat in top_stats[:20]:
        lines.append(f"{stat}")
    
    return "\n".join(lines)

if __name__ == "__main__":
    print("QUANT PLATFORM - COMPREHENSIVE PROFILING")
    print("="*80)
    
    # CPU profiling
    cpu_stats, timing_results = profile_cpu_hotspots()
    
    # Memory profiling
    mem_stats = profile_memory()
    
    # Save results
    output_dir = Path("profiling_results")
    output_dir.mkdir(exist_ok=True)
    
    (output_dir / "cpu_profile.txt").write_text(cpu_stats)
    (output_dir / "memory_profile.txt").write_text(mem_stats)
    
    # Summary
    print("\n" + "="*80)
    print("PROFILING COMPLETE")
    print("="*80)
    print(f"Results saved to {output_dir}/")
    for pkg, data in timing_results.items():
        status = data.get('status', 'unknown')
        if status == 'success':
            print(f"  {pkg:20s}: {data['elapsed']:6.2f}s")
        else:
            print(f"  {pkg:20s}: {status}")
    
    print("\nNext: Review cpu_profile.txt for hotspots")
