"""Synthetic run_many admission/streaming regression; not a production benchmark.

Run from repository root with PYTHONPATH=. and explicit --output (new directory).
Every factor is numerically checked and its on-disk result is read back.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import threading
import time
import traceback

import numpy as np
import pandas as pd
import psutil

from factor_engine.api.columns import col
from factor_engine.api.factor import Factor
from factor_engine.backend.pandas_backend import PandasBackend
from factor_engine.runtime.engine import FactorEngine
from factor_engine.runtime.perf_config import PerfConfig
from factor_engine.storage.datasource import DataSource


class Source(DataSource):
    def __init__(self, values):
        self.values = values

    def load_column(self, name):
        if name != 'close':
            raise KeyError(name)
        return self.values

    def load_columns(self, names):
        return {name: self.load_column(name) for name in names}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--roots', type=int, default=1000)
    parser.add_argument('--wave-size', type=int, default=512)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--assets', type=int, default=32)
    parser.add_argument('--gpu-evaluate', action='store_true')
    parser.add_argument('--native-fusion', action='store_true')
    parser.add_argument('--stream-api', action='store_true',
                        help='Use a single public run_many_stream call with a factor generator')
    parser.add_argument('--sink-queue-mib', type=int,
                        help='Explicit result queue budget in MiB; required with --stream-api')
    parser.add_argument('--sink-delay-ms', type=float, default=0.0,
                        help='Synthetic per-result sink delay for backpressure measurements')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if min(args.roots, args.wave_size, args.workers, args.assets) < 1:
        parser.error('roots, wave-size, workers and assets must be positive')
    if args.gpu_evaluate and args.assets < 20:
        parser.error('CUDA RankIC requires at least 20 assets')
    if args.stream_api and (args.sink_queue_mib is None or args.sink_queue_mib < 1):
        parser.error('--stream-api requires --sink-queue-mib >= 1')
    if args.sink_delay_ms < 0:
        parser.error('--sink-delay-ms must be non-negative')
    args.output.mkdir(parents=True, exist_ok=False)
    index = pd.MultiIndex.from_product(
        [pd.date_range('2024-01-01', periods=8), [f'A{i:03}' for i in range(args.assets)]],
        names=['timestamp', 'instrument'],
    )
    source_values = pd.Series(np.arange(len(index), dtype=float), index=index)
    values_path = args.output / 'values.npy'
    values = np.lib.format.open_memmap(
        values_path, mode='w+', dtype='float64', shape=(args.roots, len(index))
    )
    seen = np.zeros(args.roots, dtype=bool)
    engine = FactorEngine(backend=PandasBackend(), data_source=Source(source_values))
    perf = PerfConfig(max_workers=args.workers, memory_limit_bytes=4 * 1024**3,
                      result_budget_bytes=64 * 1024**2, native_fusion=args.native_fusion)
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    baseline_rss = peak_rss
    stopped = threading.Event()

    def sample():
        nonlocal peak_rss
        while not stopped.wait(0.05):
            peak_rss = max(peak_rss, process.memory_info().rss)

    monitor = threading.Thread(target=sample, daemon=True)
    monitor.start()
    started = time.monotonic()
    first_result_s = None
    completed_outputs = 0
    sink_lock = threading.Lock()

    def sink(name, result):
        # The benchmark remains correct if callbacks arrive on worker threads.
        with sink_lock:
            write_result(name, result)

    def write_result(name, result):
        nonlocal first_result_s, completed_outputs
        if args.sink_delay_ms:
            time.sleep(args.sink_delay_ms / 1000.0)
        position = int(name[1:])
        if seen[position]:
            raise AssertionError(f'duplicate output {name}')
        if first_result_s is None:
            first_result_s = time.monotonic() - started
        if not result.index.equals(index):
            raise AssertionError(f'axis mismatch: {name}')
        expected = source_values.to_numpy() + position
        np.testing.assert_array_equal(np.asarray(result), expected)
        values[position] = np.asarray(result)
        seen[position] = True
        completed_outputs += 1
        if args.stream_api and (completed_outputs % args.wave_size == 0
                                or completed_outputs == args.roots):
            values.flush()
            progress = {'completed_sink_outputs': completed_outputs,
                        'seconds': time.monotonic() - started,
                        'rss_bytes': process.memory_info().rss}
            with (args.output / 'progress.jsonl').open('a') as journal:
                journal.write(json.dumps(progress) + '\n')
            print(json.dumps(progress), flush=True)

    waves = []
    streaming_summary = None
    try:
        if args.stream_api:
            streaming_summary = engine.run_many_stream(
                (Factor(name=f'f{i}', expr=col('close') + float(i))
                 for i in range(args.roots)),
                wave_size=args.wave_size, n_jobs=args.workers, perf=perf,
                enable_cse=True, sink=sink,
                sink_queue_bytes=args.sink_queue_mib * 1024**2,
            )
            if streaming_summary['results']:
                raise AssertionError('stream API retained root outputs')
            if streaming_summary['completed_factors'] != args.roots:
                raise AssertionError('stream API reported incorrect completed count')
        for start in (() if args.stream_api else range(0, args.roots, args.wave_size)):
            stop = min(args.roots, start + args.wave_size)
            factors = [Factor(name=f'f{i}', expr=col('close') + float(i))
                       for i in range(start, stop)]
            wave_started = time.monotonic()
            result = engine.run_many_parallel(
                factors, n_jobs=args.workers, perf=perf, enable_cse=True,
                result_policy='sink', sink=sink,
            )
            if result['results']:
                raise AssertionError('sink execution retained root outputs')
            values.flush()
            waves.append({'start': start, 'stop': stop,
                          'seconds': time.monotonic() - wave_started,
                          'rss_bytes': process.memory_info().rss})
            del result, factors
            with (args.output / 'progress.jsonl').open('a') as journal:
                journal.write(json.dumps(waves[-1]) + '\n')
            print(json.dumps(waves[-1]), flush=True)
        if not seen.all():
            raise AssertionError(f'missing outputs: {(~seen).sum()}')
        values.flush()
        del values
        disk_values = np.load(values_path, mmap_mode='r')
        for start in range(0, args.roots, args.wave_size):
            stop = min(args.roots, start + args.wave_size)
            expected = np.arange(start, stop)[:, None] + source_values.to_numpy()[None, :]
            np.testing.assert_array_equal(disk_values[start:stop], expected)
        report = {
            'status': 'PASS', 'roots': args.roots, 'rows_per_root': len(index),
            'scale_dimensions': {
                'F_factor_count': args.roots,
                'T_timestamps': 8,
                'N_assets': args.assets,
                'input_valid_rate': 1.0,
                'formula_family': 'column_plus_distinct_scalar',
                'max_lookback': 0,
                'unique_subgraph_ratio': None,
                'unique_subgraph_ratio_status': 'NOT_MEASURED',
                'backend': 'pandas',
                'cache_state': 'fresh_engine; process/library warm state not controlled',
            },
            'workers': args.workers, 'wave_size': args.wave_size,
            'sink_queue_bytes': (args.sink_queue_mib * 1024**2
                                 if args.stream_api else None),
            'sink_delay_ms': args.sink_delay_ms,
            'native_fusion': args.native_fusion,
            'public_entry': 'run_many_stream' if args.stream_api else 'run_many_parallel_external_waves',
            'streaming_summary': streaming_summary,
            'seconds': time.monotonic() - started, 'first_result_s': first_result_s,
            'baseline_rss_bytes': baseline_rss, 'peak_rss_bytes': peak_rss,
            'output_bytes': values_path.stat().st_size, 'waves': waves,
            'limitations': 'Synthetic add/column expressions, tiny panel, research mode; '
                           'not full operators, market history or production certification.',
        }
        if args.gpu_evaluate:
            from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
            from quant_evaluator.contracts.label_bundle import LabelBundle
            from quant_evaluator.runtime.evaluator import evaluate

            # The lake file is mapped, then viewed as (T,N,F); no all-factor copy.
            panel = disk_values.reshape(args.roots, 8, args.assets).transpose(1, 2, 0)
            fb = FactorBatch(tuple(f'f{i}' for i in range(args.roots)),
                             AxisRef('time', 'int64', 8),
                             AxisRef('asset', 'str', args.assets), panel)
            times = tuple(range(8))
            lb = LabelBundle('synthetic_forward_return',
                             np.tile(np.arange(args.assets, dtype=float), (8, 1)),
                             1, decision_time=times, label_start_time=times,
                             label_end_time=tuple(t + 1 for t in times))
            gpu_started = time.monotonic()
            evaluated = evaluate(fb, lb, backend='cuda_strict', metrics=('rank_ic_series',))
            np.testing.assert_allclose(evaluated.series_metrics['rank_ic_series'], 1., atol=1e-12)
            report['gpu_seconds'] = time.monotonic() - gpu_started
            report['gpu_metadata'] = evaluated.metadata
            report['gpu_oracle'] = 'All factor and label ranks identical: RankIC=1, every output checked'
        report['total_seconds_including_gpu'] = time.monotonic() - started
        report['peak_rss_including_gpu_bytes'] = peak_rss
        (args.output / 'report.json').write_text(json.dumps(report, indent=2))
        print(json.dumps({k: v for k, v in report.items() if k != 'waves'}), flush=True)
    except BaseException:
        (args.output / 'FAILURE.json').write_text(json.dumps({
            'status': 'INCOMPLETE', 'completed_sink_outputs': int(seen.sum()),
            'completed_waves': None if args.stream_api else len(waves),
            'completed_sink_batches': completed_outputs // args.wave_size,
            'traceback': traceback.format_exc(),
            'note': 'Partial outputs are not a full-run validation certificate.',
        }, indent=2))
        raise
    finally:
        stopped.set()
        monitor.join()


if __name__ == '__main__':
    main()
