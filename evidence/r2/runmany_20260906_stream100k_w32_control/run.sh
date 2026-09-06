#!/usr/bin/env bash
set -u
run_output=$1
run_control=$2
run_roots=$3
run_wave_size=$4
run_workers=${5:-4}
printf '%s\n' "$$" > "$run_control/pid"
trap 'run_exit=$?; printf "%s\n" "$run_exit" > "$run_control/exit_code"' EXIT
if test -e "$run_output"; then
    printf '%s\n' "Output already exists: $run_output" >&2
    exit 2
fi
cd /home/sunhaiwei/quant_projects || exit 3
PYTHONPATH=. POLARS_MAX_THREADS=4 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
    .venv/bin/python benchmarks/benchmark_run_many_streaming_20260906.py \
    --roots "$run_roots" --wave-size "$run_wave_size" --workers "$run_workers" --assets 32 \
    --gpu-evaluate --native-fusion --stream-api --output "$run_output"
