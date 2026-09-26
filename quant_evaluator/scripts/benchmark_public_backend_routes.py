"""Bounded per-metric CPU/CUDA A/B for the public evaluate() facade.

Default is a read-only plan. --smoke runs one tiny synthetic rank_ic pair.
--run loads the already registered adjusted stock bars once, then evaluates
one metric/backend per child. Exposure styles are explicitly synthetic and
must not be interpreted as production risk data.
"""
from __future__ import annotations

import argparse
from datetime import datetime, time, timezone
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import resource
import statistics
import time as clock
import traceback
from zoneinfo import ZoneInfo

import numpy as np

from quant_evaluator.scripts.benchmark_backend_tournament import panel, real_registered_panel


PORTFOLIO = ("sharpe_ratio", "sortino_ratio", "win_rate", "max_drawdown", "calmar_ratio")
EXPOSURE = ("industry_exposure", "size_exposure", "beta_exposure",
            "liquidity_exposure", "volatility_exposure", "momentum_exposure",
            "max_absolute_style_exposure", "exposure_drift", "purity_ratio")
RISK = ("worst_calendar_month", "worst_calendar_quarter", "worst_calendar_year",
        "worst_rolling_21d", "worst_rolling_63d", "worst_rolling_252d")
_SOURCES = None


def metric_groups():
    from quant_evaluator.runtime.gpu_executor import GPUExecutor
    base = tuple(sorted(GPUExecutor.SUPPORTED_METRICS - set(PORTFOLIO) - set(EXPOSURE)))
    if len(base) != 18 or len(GPUExecutor.SUPPORTED_METRICS) != 32:
        raise RuntimeError("GPUExecutor capability matrix changed; inspect before benchmarking")
    return {"base": base, "portfolio": PORTFOLIO, "risk": RISK, "exposure": EXPOSURE}


def prepare_sources(batch, labels, *, include_exposure):
    from quant_evaluator.contracts.artifact_types import ProbePortfolioArtifact
    from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
    from quant_evaluator.contracts.label_bundle import LabelBundle
    from data_access.r30.calendar_snapshot import CalendarSnapshot

    x = np.asarray(batch.values)
    y = np.asarray(labels.values)
    valid = np.isfinite(x) & np.isfinite(y)[:, :, None]
    signed = np.where(valid, np.sign(x) * y[:, :, None], 0.0)
    n = valid.sum(axis=1)
    pnl = np.divide(signed.sum(axis=1), n, out=np.full(n.shape, np.nan),
                    where=n > 0)
    if np.any(np.isfinite(pnl) & (pnl < -1)):
        raise ValueError("diagnostic probe PnL below -100%")
    dates = tuple(labels.observation_time or labels.decision_time)
    probe = ProbePortfolioArtifact(
        pnl, time_index=dates, factor_ids=batch.factor_ids,
        provenance={"source": "registered_adjvwap_next_day",
                    "method": "signed_factor_equal_weight_diagnostic",
                    "execution_certified": False},
    )
    local_dates = tuple(str(day)[:10] for day in dates)
    aware = tuple(datetime.combine(
        datetime.fromisoformat(day).date(), time(15), tzinfo=ZoneInfo("Asia/Shanghai")
    ) for day in local_dates)
    calendar_batch = FactorBatch(
        batch.factor_ids,
        AxisRef("time", "datetime", len(aware), np.asarray(aware, dtype=object)),
        batch.asset_axis, batch.values, validity=batch.validity,
    )
    calendar_labels = LabelBundle(
        labels.target_id, labels.values, labels.horizon,
        decision_time=aware, label_start_time=aware,
        label_end_time=tuple(
            aware[i + 1] if i + 1 < len(aware) else aware[i].replace(hour=16)
            for i in range(len(aware))
        ), asset_axis=batch.asset_axis,
    )
    calendar_probe = ProbePortfolioArtifact(
        pnl, time_index=aware, factor_ids=batch.factor_ids,
        provenance=dict(probe.provenance),
    )
    calendar = CalendarSnapshot(
        market="benchmark_observed_ashare_sessions",
        source_version="registered_adjvwap_observed_dates_v1",
        timezone="Asia/Shanghai", trading_days=local_dates,
        sessions=(), early_close=(),
        snapshot_id=hashlib.sha256("|".join(local_dates).encode()).hexdigest(),
    )
    sources = {"base": (batch, labels, {}),
               "portfolio": (batch, labels, {"portfolio_returns": probe}),
               "risk_rolling": (batch, labels, {"portfolio_returns": probe}),
               "risk_calendar": (calendar_batch, calendar_labels,
                                 {"portfolio_returns": calendar_probe,
                                  "calendar_snapshot": calendar})}
    if include_exposure:
        from quant_evaluator.metrics.exposure_evidence import ExposurePanel
        rng = np.random.default_rng(260926)
        styles = ("industry", "size", "beta", "liquidity", "volatility", "momentum")
        risk = rng.standard_normal((batch.num_times, batch.num_assets, len(styles)))
        exposure = ExposurePanel(
            risk, style_names=styles,
            source_ref="synthetic:benchmark-public-backend-risks",
            provider="deterministic_fixture_on_registered_axes",
            date_index=dates,
            security_ids=tuple(batch.asset_axis.values.tolist()),
            factor_ids=batch.factor_ids,
            universe_snapshot_ref="benchmark:registered-asset-axis",
        )
        sources["exposure"] = (batch, labels, {
            "exposure_panel": exposure,
            "metric_parameters": {
                "max_absolute_style_exposure": {"min_finite": 5},
                "purity_ratio": {"min_finite": 5},
            },
        })
    return sources


def source_for(metric):
    if metric in EXPOSURE:
        return "exposure"
    if metric in PORTFOLIO:
        return "portfolio"
    if metric.startswith("worst_calendar_"):
        return "risk_calendar"
    if metric.startswith("worst_rolling_"):
        return "risk_rolling"
    return "base"


def _worker(conn, metric, backend, repeats):
    try:
        from quant_evaluator.runtime.evaluator import evaluate
        from quant_evaluator.contracts.backend_policy import GPUExecutionPolicy
        batch, labels, source_kwargs = _SOURCES[source_for(metric)]
        kwargs = dict(source_kwargs)
        if "metric_parameters" in kwargs:
            kwargs["metric_parameters"] = {
                metric: kwargs["metric_parameters"].get(metric, {})
            }
        if metric.startswith("worst_calendar_"):
            kwargs["metric_parameters"] = {metric: {"partial_policy": "include"}}
        kwargs["metrics"] = (metric,)
        kwargs["backend"] = backend
        if backend == "cuda_strict":
            kwargs["gpu_policy"] = GPUExecutionPolicy(
                max_vram_fraction=0.4, max_host_result_bytes=64 * 1024**2,
                strict_backend=True,
            )
        times = []
        for _ in range(repeats + 1):
            start = clock.perf_counter()
            result = evaluate(batch, labels, **kwargs)
            times.append(clock.perf_counter() - start)
        artifact = result.artifacts[metric]
        values = np.asarray(artifact.values, dtype=np.float64)
        evidence = tuple(
            (result.get_metric(metric, fid).valid,
             result.get_metric(metric, fid).observation_count,
             result.get_metric(metric, fid).sample_unit)
            if result.get_metric(metric, fid) is not None else None
            for fid in batch.factor_ids
        )
        counts = getattr(artifact, "counts", None)
        valid_mask = getattr(artifact, "valid_mask", None)
        provenance = dict(artifact.provenance)
        conn.send({
            "status": "ok", "cold_s": times[0], "warm_s": times[1:],
            "warm_median_s": statistics.median(times[1:]),
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "backend_used": result.metadata.get("backend_used"),
            "gpu_peak_vram": result.metadata.get("peak_vram"),
            "artifact_type": type(artifact).__name__,
            "values": values, "evidence": evidence,
            "counts": None if counts is None else np.asarray(counts),
            "valid_mask": None if valid_mask is None else np.asarray(valid_mask),
            "provenance_observation_counts": provenance.get("observation_counts"),
            "input_factor_hash": provenance.get("factor_value_bytes_hash"),
            "execution_backend": provenance.get("execution_backend"),
            "no_fallback": provenance.get("no_fallback"),
        })
    except BaseException as exc:
        conn.send({"status": "error", "error": f"{type(exc).__name__}: {exc}",
                   "traceback": traceback.format_exc(limit=5)})
    finally:
        conn.close()


def run_one(ctx, metric, backend, repeats, timeout_s):
    receiver, sender = ctx.Pipe(duplex=False)
    process = ctx.Process(target=_worker, args=(sender, metric, backend, repeats))
    process.start()
    sender.close()
    if receiver.poll(timeout_s):
        try:
            answer = receiver.recv()
        except EOFError:
            answer = {"status": "crash", "exitcode": process.exitcode}
    else:
        answer = {"status": "timeout", "timeout_s": timeout_s}
        process.terminate()
    process.join(5)
    if process.is_alive():
        process.kill()
        process.join(5)
    receiver.close()
    return answer


def parity(cpu, gpu):
    x, y = cpu["values"], gpu["values"]
    same_shape = x.shape == y.shape
    same_finite = same_shape and bool(np.array_equal(np.isfinite(x), np.isfinite(y)))
    numeric = same_shape and bool(np.allclose(x, y, rtol=1e-8, atol=1e-10, equal_nan=True))
    same_counts = (cpu["counts"] is None and gpu["counts"] is None) or (
        cpu["counts"] is not None and gpu["counts"] is not None and
        np.array_equal(cpu["counts"], gpu["counts"])
    )
    same_mask = (cpu["valid_mask"] is None and gpu["valid_mask"] is None) or (
        cpu["valid_mask"] is not None and gpu["valid_mask"] is not None and
        np.array_equal(cpu["valid_mask"], gpu["valid_mask"])
    )
    same_evidence = cpu["evidence"] == gpu["evidence"]
    same_provenance_counts = cpu["provenance_observation_counts"] == gpu["provenance_observation_counts"]
    same_input_hash = cpu["input_factor_hash"] == gpu["input_factor_hash"]
    gpu_routed = gpu["backend_used"] == "cuda" and bool(gpu["gpu_peak_vram"])
    return {"pass": all((same_shape, same_finite, numeric, same_counts, same_mask,
                          same_evidence, same_provenance_counts, same_input_hash,
                          gpu_routed)),
            "same_shape": same_shape, "same_finite": same_finite,
            "numeric_1e-8_1e-10": numeric, "same_counts": bool(same_counts),
            "same_mask": bool(same_mask), "same_evidence": same_evidence,
            "same_provenance_counts": same_provenance_counts,
            "same_input_hash": same_input_hash, "gpu_routed": gpu_routed,
            "max_abs": float(np.max(np.abs(x[np.isfinite(x) & np.isfinite(y)] -
                                           y[np.isfinite(x) & np.isfinite(y)])))
            if same_shape and np.any(np.isfinite(x) & np.isfinite(y)) else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--groups", default="base,portfolio,risk,exposure")
    parser.add_argument("--metrics", default="")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--alternate-order", action="store_true",
                        help="Reverse CPU/CUDA order for every second metric")
    parser.add_argument("--timeout-s", type=float, default=120)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    groups = metric_groups()
    selected_groups = args.groups.split(",")
    if any(group not in groups for group in selected_groups):
        parser.error("unknown metric group")
    selected = tuple(dict.fromkeys(mid for group in selected_groups for mid in groups[group]))
    if args.metrics:
        requested = set(args.metrics.split(","))
        if requested - set(selected):
            parser.error(f"requested metrics outside groups: {sorted(requested - set(selected))}")
        selected = tuple(mid for mid in selected if mid in requested)
    if args.smoke:
        selected = ("rank_ic",)
    if args.repeats < 1 or args.timeout_s <= 0:
        parser.error("repeats and timeout must be positive")
    if not args.smoke and not args.run:
        print(json.dumps({"mode": "plan_only", "groups": groups,
                          "selected_metrics": selected,
                          "note": "adaptive_quantile_count is CPU-planned, outside GPUExecutor whitelist"},
                         indent=2))
        return
    global _SOURCES
    read_start = clock.perf_counter()
    if args.smoke:
        batch, labels = panel(24, 40, 1, 260926)
        source = {"kind": "tiny_synthetic", "days": 24, "assets": 40, "factors": 1}
    else:
        batch, labels, source = real_registered_panel()
    load_s = clock.perf_counter() - read_start
    _SOURCES = ({"base": (batch, labels, {})} if args.smoke else
                prepare_sources(batch, labels, include_exposure=any(
                    mid in EXPOSURE for mid in selected
                )))
    ctx = mp.get_context("fork")
    report = {"started_at_utc": datetime.now(timezone.utc).isoformat(),
              "source": source, "source_load_s": load_s,
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "groups": selected_groups, "repeats": args.repeats,
              "alternate_order": args.alternate_order,
              "timeout_s": args.timeout_s, "results": {}}
    for metric_index, metric in enumerate(selected):
        raw = {}
        backend_order = (("cuda_strict", "cpu") if args.alternate_order and metric_index % 2
                         else ("cpu", "cuda_strict"))
        for backend in backend_order:
            print(f"{metric} {backend}: running", flush=True)
            raw[backend] = run_one(ctx, metric, backend, args.repeats, args.timeout_s)
            print(f"{metric} {backend}: {raw[backend]['status']}", flush=True)
        row = {"backend_order": backend_order}
        for backend, answer in raw.items():
            row[backend] = {k: v for k, v in answer.items()
                            if k not in ("values", "counts", "valid_mask", "evidence",
                                         "traceback", "provenance_observation_counts",
                                         "input_factor_hash")}
        if all(raw[b]["status"] == "ok" for b in ("cpu", "cuda_strict")):
            row["parity"] = parity(raw["cpu"], raw["cuda_strict"])
        report["results"][metric] = row
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
