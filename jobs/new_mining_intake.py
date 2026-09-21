#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""本周新挖因子：冻结候选 → 分批落值/评估 → 发布页面 → 上传 COS。

复用正式主链路：
  jobs/incremental_factor_intake.py 的 admit/land/evaluate/write_report_manifest

子命令
------
freeze   : 从 xlubs mining_outputs 选出的候选里做门禁+去重，冻结成 candidates.json
audit    : 只做 DSL/历史需求/矩阵覆盖审计，不算值
wave     : 把待办候选切成 wave，用隔离子进程执行（内存/时间护栏），逐因子写 report_manifest.json
status   : 打印真实进度
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path("/home/sunhaiwei/quant_projects")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "jobs"))

WORK = Path(os.environ.get("NEWMINING_WORK", str(ROOT / "work/newmining_20260919")))
CANDIDATES = WORK / "candidates.json"
STATE = WORK / "state.json"
QUEUE_LOCK = WORK / "queue.lock"

IC_MIN, ICIR_MIN = 0.015, 0.15
EVAL_VERSION = "equal-amount-gross100-actual-volume-v3-20260907"

# ---------------------------------------------------------------------------
# DSL repairs applied at landing time
# ---------------------------------------------------------------------------
# A miner occasionally emits a call the engine rejects for a *unit* reason, not
# a semantic one.  Rewriting it to the author's evident intent is authorised by
# the report owner and is recorded in the manifest/report as a repair, so a
# repaired factor is never silently presented as the miner's own expression.
#   winsorize(x, lo[, hi]) takes unit-interval quantiles; the author wrote `1`
#   (meaning "1%") where the DSL needs `0.01`, and the engine refuses lo=1>0.95.
DSL_REPAIRS = {
    "alphasage_20260910154931_3c69f233": dict(
        find="winsorize(ts_regression_slope(circulating_market_cap, turnover_ratio, 30), 1)",
        replace="winsorize(ts_regression_slope(circulating_market_cap, turnover_ratio, 30), 0.01)",
        note="winsorize 第二参应为 (0,1) 分位；作者写 1（本意 1%），按 winsorize(x, 0.01) 修复",
    ),
    "alphasage_20260911220613_bc370a5c": dict(
        find="log(turnover_ratio)",
        replace="safe_log(turnover_ratio)",
        note=("log 未做零值保护：只要有 1 只股票换手率为 0，log(0)=-inf 就会把整个截面的 "
              "zscore 变成 NaN。实测 2588 天里 2580 天含零值股票 → 全部不可用，"
              "仅 8 天无零值（其中 5 天可用）。改用引擎自带的 safe_log 后恢复正常覆盖"),
    ),
}


def apply_dsl_repair(record):
    """Return (record, repair_note|None). Original record is never mutated."""
    repair = DSL_REPAIRS.get(str(record.get("page_name")))
    if not repair:
        return record, None
    fixed = dict(record)
    touched = False
    for key in ("fe_formula", "lqtp_formula", "source_formula"):
        value = fixed.get(key)
        if not isinstance(value, str) or repair["find"] not in value:
            continue
        if repair["replace"] in value:
            continue          # already repaired: never turn safe_log into safe_safe_log
        fixed[key] = value.replace(repair["find"], repair["replace"])
        touched = True
    if not touched:
        return record, None
    return fixed, dict(applied=True, note=repair["note"],
                       original=record.get("fe_formula"), repaired=fixed.get("fe_formula"))


# ---------------------------------------------------------------------------
# candidate freezing
# ---------------------------------------------------------------------------
def _norm_formula(text: str) -> str:
    return re.sub(r"\s+", "", str(text or ""))


def existing_formulas():
    """Formulas already present in the report pool or landed matrices."""
    seen = set()
    import incremental_factor_intake as intake
    pool = []
    try:
        pool = intake.load_pool()
    except Exception as exc:
        print(f"[warn] pool unreadable: {exc}", flush=True)
    for record in pool:
        for key in ("fe_formula", "dsl", "lqtp_formula"):
            if record.get(key):
                seen.add(_norm_formula(record[key]))
    return seen, {str(r.get("page_name")) for r in pool}


def freeze(selected_path: Path, out: Path, only_this_week: bool):
    rows = json.load(open(selected_path))
    seen_formulas, seen_pages = existing_formulas()
    print(f"[freeze] pool formulas={len(seen_formulas)} pages={len(seen_pages)}", flush=True)

    candidates, rejected = [], []
    for r in rows:
        fid = r["factor_id"]
        formula = str(r.get("formula") or "")
        ic, icir = r.get("ic"), r.get("icir")
        # signed gate after direction freeze (no abs() substitution)
        passed = (ic is not None and ic > IC_MIN) or (icir is not None and icir > ICIR_MIN)
        if not passed:
            rejected.append(dict(factor_id=fid, reason="below_gate", ic=ic, icir=icir))
            continue
        if r["algo"] != "alphasage" or not formula:
            rejected.append(dict(factor_id=fid, reason=f"surface_not_executable:{r.get('algo')}/{r.get('expr_type')}",
                                 ic=ic, icir=icir))
            continue
        if only_this_week and str(r.get("campaign", "")) < "20260914":
            rejected.append(dict(factor_id=fid, reason="earlier_campaign", campaign=r.get("campaign")))
            continue
        if fid in seen_pages:
            rejected.append(dict(factor_id=fid, reason="already_in_pool_page"))
            continue
        if _norm_formula(formula) in seen_formulas:
            rejected.append(dict(factor_id=fid, reason="duplicate_formula_of_pool"))
            continue
        if (ROOT / "weekly_backtest_output/factor_matrices_all" / f"{fid}.parquet").exists():
            rejected.append(dict(factor_id=fid, reason="matrix_already_landed"))
            continue
        candidates.append(dict(
            page_name=fid, factor_name=fid, fe_formula=formula,
            source_formula=formula,
            lqtp_formula=r.get("expr"),
            source_metadata=r.get("src"), campaign=r.get("campaign"),
            mined_by=r.get("mined_by"), market=r.get("market"), domain=r.get("domain"),
            frequency=r.get("freq"),
            mining_metrics=dict(ic=ic, icir=icir, test_ic=r.get("t_ic"),
                                test_icir=r.get("t_icir"), turnover=r.get("turnover"),
                                metric_kind=r.get("metric_kind")),
            is_unlisted_miner=True, can_use_factor_engine=True,
            window_start=r.get("start"), window_end=r.get("end"),
        ))

    candidates = sorted(candidates, key=lambda c: (-(c["mining_metrics"]["ic"] or -9),
                                                  -(c["mining_metrics"]["icir"] or -9)))
    payload = dict(
        generated_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        threshold=dict(rank_ic=IC_MIN, rank_icir=ICIR_MIN, combine="or",
                       note="方向冻结后的带符号阈值；未取绝对值"),
        source_manifest=str(selected_path),
        candidates=candidates, rejected=rejected,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1))
    os.replace(tmp, out)
    print(f"[freeze] candidates={len(candidates)} rejected={len(rejected)} -> {out}")
    import collections
    print("         reject reasons:", dict(collections.Counter(x["reason"] for x in rejected)))
    return payload


# ---------------------------------------------------------------------------
# state
# ---------------------------------------------------------------------------
def load_state():
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {"factors": {}}


def save_state(state):
    WORK.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=1, allow_nan=False))
    os.replace(tmp, STATE)


def mem_available_gib():
    mem = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    return int(mem["MemAvailable"].split()[0]) / 1024**2


# ---------------------------------------------------------------------------
# wave execution
# ---------------------------------------------------------------------------
def _force_non_negative_rankic(intake, records, batch, backend, vwap):
    """Publish every factor in the direction whose full-window RankIC is >= 0.

    The intake direction is trained on 2016-01-04..2018-06-30 only, so the
    full-window mean IC of a factor can still come out negative after that flip.
    The report convention is "RankIC 转正：负则加负号", so those factors are
    re-evaluated with the opposite fixed direction and QuantEvaluator recomputes
    IC, decile net values and the long-short series consistently — the stored
    artifact is a real evaluation, never a sign-patched copy.
    """
    negatives = {}
    for record in records or []:
        name = str(record.get("page_name"))
        evaluated = (batch.get("factors") or {}).get(name)
        direction = getattr(evaluated, "direction", None)
        mean_ic = getattr(evaluated, "mean_rank_ic", None)
        if evaluated is None or direction not in (1, -1) or mean_ic is None:
            continue
        if float(mean_ic) < 0:
            negatives[name] = (record, -int(direction), float(mean_ic))
    if not negatives:
        return 0
    names = list(negatives)
    try:
        redone = intake.evaluate_factor_batch(
            [negatives[name][0] for name in names], batch_size=1, backend=backend,
            vwap=vwap, direction_map={name: negatives[name][1] for name in names})
    except Exception as error:  # a failed flip must not lose the evaluation
        print(f"[wave] direction flip raised for {len(names)} factor(s): "
              f"{type(error).__name__}: {error}", flush=True)
        return 0
    flipped = 0
    for name in names:
        fixed = (redone.get("factors") or {}).get(name)
        if fixed is None or float(getattr(fixed, "mean_rank_ic", -1.0)) < 0:
            print(f"[wave] direction flip did not turn RankIC positive for {name}; "
                  "keeping the trained direction", flush=True)
            continue
        batch["factors"][name] = fixed
        batch["backend_used"] |= set(redone.get("backend_used") or ())
        if redone.get("dates") is not None:
            batch["dates"] = redone["dates"]
        flipped += 1
        print(f"[wave] rank_ic flipped to non-negative: {name} "
              f"{negatives[name][2]:+.5f} -> {float(fixed.mean_rank_ic):+.5f}", flush=True)
    return flipped


def run_wave_child(records, out_dir: Path, *, batch_size, fe_backend, qe_backend, force_landing,
                   window_years, landing_start=None):
    """Land and evaluate one wave; write a single manifest for the wave."""
    import incremental_factor_intake as intake
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "report_manifest.json"

    # A DSL repair (unit slip in a call, see DSL_REPAIRS) is applied here so the
    # engine sees an executable expression; the note travels into the manifest.
    repaired = {}
    fixed_records = []
    for original in records or []:
        fixed, note = apply_dsl_repair(original)
        if note:
            repaired[str(original.get("page_name"))] = note
        fixed_records.append(fixed)
    records = fixed_records
    if repaired:
        print(f"[wave] dsl repairs applied: {json.dumps(repaired, ensure_ascii=False)}", flush=True)

    # Wider landing windows cut redundant warm-up re-reads (each window re-reads
    # 550 days of overlap).  Memory stays bounded by the wave/process guard.
    _original_windowed = intake.land_factor_batch_windowed
    def _windowed(records_, **kwargs):
        kwargs.setdefault("window_years", window_years)
        if landing_start:
            # Give every factor real pre-window history so a warm-up of a few
            # bars does not push the first usable cross-section past the
            # report window start.
            kwargs.setdefault("start_date", landing_start)
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        return _original_windowed(records_, **kwargs)
    intake.land_factor_batch_windowed = _windowed

    admitted, admission_errors = intake.admit_report_factors(records)
    print(f"[wave] admitted={len(admitted)} admission_errors={len(admission_errors)} window_years={window_years}", flush=True)
    landing = intake.land_missing_factors(admitted, batch_size=batch_size,
                                          backend_name=fe_backend, force=force_landing)
    print(f"[wave] landing={json.dumps({k: (v if k != 'landed' else len(v)) for k, v in landing.items()}, ensure_ascii=False, default=str)[:800]}", flush=True)
    evaluable, landing_errors = intake.evaluation_candidates_after_landing(admitted, landing)

    batch = {"factors": {}, "backend_used": set(), "fallbacks": {},
             "unavailable": dict(admission_errors), "dates": None,
             "evaluation_provenance": {}}
    batch["unavailable"].update(landing_errors)
    stream_eval = os.environ.get("FACTOR_REPORT_STREAM_EVAL", "1") != "0"
    if stream_eval:
        # Score one factor at a time and republish the manifest after each one,
        # so "the factor exists" and "the factor has an evaluation" happen
        # together: nothing is ever left landed-but-unscored, and a wave killed by
        # the memory guard or the time limit keeps every factor already scored
        # instead of losing the whole batch.  The label/vwap panel is loaded once
        # and reused, so per-factor scoring adds no panel reads.
        vwap = (intake.load_vwap(start_date=str(intake.FULL_WINDOW_START.date()),
                                 end_date=str(intake.FULL_WINDOW_END.date()))
                if evaluable else None)
        for index, record in enumerate(evaluable, 1):
            name = str(record["page_name"])
            try:
                part = intake.evaluate_factor_batch([record], batch_size=1,
                                                    backend=qe_backend, vwap=vwap)
            except Exception as error:  # one bad factor must not cost the wave
                batch["unavailable"][name] = f"evaluation raised {type(error).__name__}: {error}"
                print(f"[wave] {name} evaluation raised {type(error).__name__}: {error}", flush=True)
            else:
                batch["factors"].update(part.get("factors") or {})
                batch["fallbacks"].update(part.get("fallbacks") or {})
                batch["unavailable"].update(part.get("unavailable") or {})
                batch["backend_used"] |= set(part.get("backend_used") or ())
                if part.get("dates") is not None:
                    batch["dates"] = part["dates"]
                batch["evaluation_provenance"] = (part.get("evaluation_provenance")
                                                  or batch["evaluation_provenance"])
                _force_non_negative_rankic(intake, [record], batch, qe_backend, vwap)
            intake.write_report_manifest(records, batch, target=target)
            print(f"[wave] scored {index}/{len(evaluable)} {name} "
                  f"(scored={len(batch['factors'])} unavailable={len(batch['unavailable'])})",
                  flush=True)
        if not evaluable:
            intake.write_report_manifest(records, batch, target=target)
    else:
        vwap = (intake.load_vwap(start_date=str(intake.FULL_WINDOW_START.date()),
                                 end_date=str(intake.FULL_WINDOW_END.date()))
                if evaluable else None)
        batch = intake.evaluate_factor_batch(evaluable, batch_size=batch_size,
                                             backend=qe_backend, vwap=vwap)
        _force_non_negative_rankic(intake, evaluable, batch, qe_backend, vwap)
        batch["unavailable"].update(admission_errors)
        batch["unavailable"].update(landing_errors)
        intake.write_report_manifest(records, batch, target=target)
    if repaired:
        manifest = json.loads(target.read_text())
        for name, note in repaired.items():
            entry = (manifest.get("factors") or {}).get(name)
            if isinstance(entry, dict):
                entry["dsl_repair"] = note
        target.write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    print(f"[wave] manifest={target} factors={len(batch['factors'])} unavailable={len(batch['unavailable'])}",
          flush=True)
    return target


def run_waves(limit, batch_size, fe_backend, qe_backend, memory_gib, threads, reuse, force_landing,
              root_workers, window_years, host_mem_floor_gib=8.0, wave_time_limit_s=None):
    payload = json.loads(CANDIDATES.read_text())
    candidates = {c["page_name"]: c for c in payload["candidates"]}
    state = load_state()
    def has_result(name):
        """A factor counts as done only when its own evaluated manifest exists."""
        manifest = WORK / "factors" / name / "report_manifest.json"
        return manifest.exists() and manifest.stat().st_size > 2

    pending = [name for name in candidates if not has_result(name)]
    if reuse:
        # Retry only entries whose previous run produced no usable manifest.
        pending = [n for n in pending if not has_result(n)]
    if limit:
        pending = pending[:limit]
    print(f"[waves] pending={len(pending)} of {len(candidates)}", flush=True)
    if not pending:
        return

    WORK.mkdir(parents=True, exist_ok=True)
    with QUEUE_LOCK.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for offset in range(0, len(pending), batch_size):
            wave_names = pending[offset:offset + batch_size]
            wave = [candidates[n] for n in wave_names]
            for n in wave_names:
                (WORK / "factors" / n).mkdir(parents=True, exist_ok=True)
                (WORK / "factors" / n / "candidate.json").write_text(
                    json.dumps([candidates[n]], ensure_ascii=False, indent=1))
            wave_id = hashlib.sha256("|".join(wave_names).encode()).hexdigest()[:16]
            folder = WORK / "waves" / f"wave_{wave_id}"
            folder.mkdir(parents=True, exist_ok=True)
            candidate_path = folder / "candidate.json"
            candidate_path.write_text(json.dumps(wave, ensure_ascii=False, indent=1))
            target = folder / "report_manifest.json"

            while mem_available_gib() < max(memory_gib, host_mem_floor_gib) + 4:
                state["status"] = "waiting_for_memory"
                save_state(state)
                print(f"[waves] waiting for memory: avail={mem_available_gib():.1f}GiB", flush=True)
                time.sleep(30)

            cmd = [sys.executable, str(Path(__file__).resolve()), "--run-wave", str(candidate_path),
                   "--wave-out", str(folder), "--batch-size", str(batch_size),
                   "--fe-backend", fe_backend, "--qe-backend", qe_backend]
            if force_landing:
                cmd.append("--force-landing")
            cmd += ["--window-years", str(window_years)]
            env = dict(os.environ, DUCKDB_MAX_THREADS=str(threads), OMP_NUM_THREADS=str(threads),
                       OPENBLAS_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads),
                       POLARS_MAX_THREADS=str(threads), FACTOR_REPORT_FE_BACKEND=fe_backend,
                       FACTOR_REPORT_ROOT_WORKERS=str(max(1, root_workers)),
                       ASHARE_PARQUET_ROOT=str(Path.home() / "cos_data"),
                       DATA_ACCESS_SKIP_COS_MIRROR="1")
            for n in wave_names:
                state["factors"][n] = dict(status="running", started_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                                           evaluation_version=EVAL_VERSION)
            save_state(state)
            print(f"[waves] start wave {wave_id} n={len(wave_names)} names={wave_names}", flush=True)
            started = time.monotonic()
            reason = None
            with (folder / "run.log").open("a") as log:
                child = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env,
                                         start_new_session=True)
                peak = 0
                while child.poll() is None:
                    try:
                        status = Path(f"/proc/{child.pid}/status").read_text()
                        peak = max(peak, int(re.search(r"VmRSS:\s+(\d+)", status).group(1)) / 1024**2)
                    except Exception:
                        pass
                    # The host is shared: neighbours (other users' mining jobs)
                    # spike to tens of GiB and drive MemAvailable down.  Killing
                    # our wave just because the *host* is tight throws away its
                    # work even when we are a negligible contributor (we have
                    # seen 1.8GiB waves killed at a 6GiB floor).  Only abort
                    # when we are a real contributor, or in a genuine emergency.
                    available = mem_available_gib()
                    if peak >= 8.0 and available < host_mem_floor_gib:
                        reason = f"host low-memory guard (own peak {peak:.1f}GiB, avail {available:.1f}GiB)"
                        os.killpg(child.pid, signal.SIGTERM)
                        break
                    if available < 2.0:
                        reason = f"host emergency memory (avail {available:.1f}GiB)"
                        os.killpg(child.pid, signal.SIGTERM)
                        break
                    if peak > memory_gib:
                        reason = f"process memory limit ({peak:.1f}GiB > {memory_gib}GiB)"
                        os.killpg(child.pid, signal.SIGTERM)
                        break
                    if time.monotonic() - started > (wave_time_limit_s or max(1800, len(wave_names) * 900)):
                        reason = "wave time limit"
                        os.killpg(child.pid, signal.SIGTERM)
                        break
                    time.sleep(3)
                code = child.wait()
            elapsed = time.monotonic() - started
            print(f"[waves] wave {wave_id} exit={code} elapsed={elapsed:.0f}s peak={peak:.1f}GiB reason={reason}",
                  flush=True)
            manifest = json.loads(target.read_text()) if code == 0 and target.exists() else None
            for n in wave_names:
                item_dir = WORK / "factors" / n
                result = state["factors"][n]
                result.update(exit_code=code, peak_rss_gib=round(peak, 2),
                              elapsed_s=round(elapsed, 1), wave=wave_id, run_log=str(folder / "run.log"))
                entry = (manifest or {}).get("factors", {}).get(n)
                if entry is not None:
                    isolated = dict(manifest, factors={n: entry})
                    if entry.get("artifact"):
                        isolated["factors"] = {n: dict(entry, artifact=str((target.parent / entry["artifact"]).resolve()))}
                    (item_dir / "report_manifest.json").write_text(json.dumps(isolated, ensure_ascii=False, indent=1))
                    import incremental_factor_intake as intake
                    result["status"] = ("unavailable" if entry.get("status") == "unavailable"
                                        else "evaluated" if intake.passes_weekly_gate(entry.get("metrics") or {})
                                        else "below_gate")
                    result["metrics"] = entry.get("metrics") or {}
                    result["direction"] = entry.get("direction")
                    result["is_flipped"] = entry.get("is_flipped")
                else:
                    result["status"] = "failed"
                    result["reason"] = reason or "wave process failed; see run.log"
            save_state(state)


def _manifest_bases():
    """Every directory that may hold per-factor manifests (root + all shards)."""
    bases = [WORK / "factors"]
    shards = WORK / "shards"
    if shards.is_dir():
        bases += sorted(p for p in shards.glob("*/factors") if p.is_dir())
    return bases


def _current_manifests():
    """name -> (manifest path, entry) for every candidate that has one."""
    out = {}
    for base in _manifest_bases():
        for manifest in sorted(base.glob("*/report_manifest.json")):
            name = manifest.parent.name
            if name in out or manifest.stat().st_size <= 2:
                continue
            try:
                entry = (json.loads(manifest.read_text()).get("factors") or {}).get(name)
            except Exception:
                continue
            if entry:
                out[name] = (manifest, entry)
    return out


def _repair_class(name, entry):
    """Classify a candidate as retryable, or say why it is not.

    Retryable:
      ``resource``  — the engine's admission controller denied the root because
                      a concurrently running job had consumed the host memory
                      headroom.  Re-landing alone on a quiet box clears it.
      ``coverage``  — a matrix is on disk that now satisfies the full-window
                      rule, but the stored verdict says otherwise (the rule
                      itself was fixed, or an allowance changed).  Only the
                      evaluation has to be re-run; re-landing is neither needed
                      nor possible, because the panel starts on
                      FULL_WINDOW_START — there is no earlier history to give a
                      windowed operator its warm-up from.
      ``missing``   — never landed at all (including "a leftover file from the
                      shared matrix directory exists, but our own landing never
                      recorded one").

    Non-retryable: formula defects the engine cannot execute, and cross-sections
    that are degenerate by construction (constant by definition), where any
    "repair" would invent a different factor.
    """
    if entry is None:
        return "missing"
    reason = str(entry.get("reason") or "")
    if entry.get("status") != "unavailable":
        return None
    if "ResourceBudgetExceeded" in reason or "CPUWidthUnavailable" in reason:
        return "resource"
    source = None
    try:
        from factor_report_sources import resolve_raw_matrix
        source = resolve_raw_matrix(name)
    except Exception:
        source = None
    # ``matrix_path`` is written only when *this* campaign's landing produced
    # the matrix.  The matrix search directory is shared with the pre-existing
    # weekly-backtest library, so a factor that we never managed to land still
    # resolves to a leftover foreign file: its coverage then reads
    # "not full window" and — with no known DSL repair — the factor falls out
    # of the retry set for good.  Judging ownership from the manifest instead
    # of from "a file exists somewhere" is what keeps those factors retryable.
    owned = bool(str(entry.get("matrix_path") or "").strip())
    if source is None or source.path is None or not owned:
        # No self-produced matrix on disk: the landing simply never succeeded
        # (plan binding, admission, crash, guard kill).  It deserves a retry,
        # and with the backend fallback chain now covering plan-binding errors
        # it can genuinely succeed where it failed before.
        return "missing"
    if source.is_full_window:
        return "coverage"
    if name in DSL_REPAIRS:
        return "dsl_fix"
    # A matrix we did land ourselves, not full-window, with no known DSL
    # repair: the formula's own domain (a rolling window over a barely-dense
    # field, or a transform that is undefined on most of its range) leaves too
    # few usable cross-sections.  Re-landing reproduces the same numbers, so
    # this is reported as-is instead of burning another hour.
    return None


def repair_all(batch_size, fe_backend, qe_backend, window_years, threads, root_workers,
               rounds=2, landing_start="2015-01-05", memory_gib=32,
               host_mem_floor_gib=6.0, wave_time_limit_s=None):
    """Retry every candidate whose failure is a landing/environment defect.

    Must run with no other FactorEngine job on the host: the admission denials
    this exists to undo are *caused* by a concurrent shard, so retrying while
    one is still running merely reproduces them.
    """
    payload = json.loads(CANDIDATES.read_text())
    candidates = {c["page_name"]: c for c in payload["candidates"]}
    folder = WORK / "waves" / "repair_all"
    folder.mkdir(parents=True, exist_ok=True)

    # A factor that fails again with exactly the same reason is structurally
    # unfixable by re-landing; retrying it in every round would burn half an
    # hour per attempt for nothing.  Each failure signature is attempted at
    # most twice, then left as-is and reported honestly.
    attempts = {}

    def signature(entry):
        if entry is None:
            return "no-manifest"
        return str(entry.get("reason") or entry.get("status") or "?")

    for round_index in range(1, rounds + 1):
        known = _current_manifests()
        todo, classes, exhausted = [], {}, 0
        for name in candidates:
            got = known.get(name)
            entry = got[1] if got else None
            kind = _repair_class(name, entry)
            if kind is None:
                continue
            sig = signature(entry)
            if attempts.get((name, sig), 0) >= 2:
                exhausted += 1
                continue
            attempts[(name, sig)] = attempts.get((name, sig), 0) + 1
            todo.append(name)
            classes[name] = kind
        print(f"[repair] round {round_index}: {len(todo)} retryable "
              f"({sum(1 for v in classes.values() if v == 'resource')} resource, "
              f"{sum(1 for v in classes.values() if v == 'coverage')} coverage(eval-only), "
              f"{sum(1 for v in classes.values() if v == 'dsl_fix')} dsl_fix, "
              f"{sum(1 for v in classes.values() if v == 'missing')} never-landed)"
              f"; {exhausted} skipped (same failure twice, left as reported)",
              flush=True)
        if not todo:
            return

        progressed = False
        # Coverage repairs need no FactorEngine landing at all: the matrix is
        # already on disk and only the stored verdict is stale.  Run those first
        # (minutes, no memory risk) before spending hours on real landings.
        ordered = ([n for n in todo if classes.get(n) == "coverage"]
                   + [n for n in todo if classes.get(n) != "coverage"])
        for offset in range(0, len(ordered), batch_size):
            names = ordered[offset:offset + batch_size]
            eval_only = all(classes.get(n) == "coverage" for n in names)
            wave = [candidates[n] for n in names]
            candidate_path = folder / "candidate.json"
            candidate_path.write_text(json.dumps(wave, ensure_ascii=False, indent=1))
            env = dict(os.environ, FACTOR_REPORT_ROOT_WORKERS=str(max(1, root_workers)),
                       DUCKDB_MAX_THREADS=str(threads), OMP_NUM_THREADS=str(threads),
                       POLARS_MAX_THREADS=str(threads), MKL_NUM_THREADS=str(threads),
                       OPENBLAS_NUM_THREADS=str(threads),
                       FACTOR_REPORT_FE_BACKEND=fe_backend,
                       ASHARE_PARQUET_ROOT=str(Path.home() / "cos_data"),
                       DATA_ACCESS_SKIP_COS_MIRROR="1")
            cmd = [sys.executable, str(Path(__file__).resolve()), "--run-wave", str(candidate_path),
                   "--wave-out", str(folder), "--batch-size", str(len(names)),
                   "--fe-backend", fe_backend, "--qe-backend", qe_backend,
                   "--window-years", str(window_years)]
            if not eval_only:
                cmd += ["--force-landing", "--landing-start", landing_start]
            if not eval_only:
                while mem_available_gib() < max(memory_gib, host_mem_floor_gib) + 4:
                    print(f"[repair] waiting for memory: avail={mem_available_gib():.1f}GiB", flush=True)
                    time.sleep(30)
            started = time.monotonic()
            time_limit = wave_time_limit_s or max(1800, len(names) * 900)
            reason = None
            peak = 0
            with (folder / "run.log").open("a") as log:
                child = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env,
                                         start_new_session=True)
                while child.poll() is None:
                    try:
                        status = Path(f"/proc/{child.pid}/status").read_text()
                        peak = max(peak, int(re.search(r"VmRSS:\s+(\d+)", status).group(1)) / 1024**2)
                    except Exception:
                        pass
                    # The host is shared: neighbours (other users' mining jobs)
                    # spike to tens of GiB and drive MemAvailable down.  Killing
                    # our wave just because the *host* is tight throws away its
                    # work even when we are a negligible contributor (we have
                    # seen 1.8GiB waves killed at a 6GiB floor).  Only abort
                    # when we are a real contributor, or in a genuine emergency.
                    available = mem_available_gib()
                    if peak >= 8.0 and available < host_mem_floor_gib:
                        reason = f"host low-memory guard (own peak {peak:.1f}GiB, avail {available:.1f}GiB)"
                        os.killpg(child.pid, signal.SIGTERM)
                        break
                    if available < 2.0:
                        reason = f"host emergency memory (avail {available:.1f}GiB)"
                        os.killpg(child.pid, signal.SIGTERM)
                        break
                    if peak > memory_gib:
                        reason = f"process memory limit ({peak:.1f}GiB > {memory_gib}GiB)"
                        os.killpg(child.pid, signal.SIGTERM)
                        break
                    if time.monotonic() - started > time_limit:
                        reason = "wave time limit"
                        os.killpg(child.pid, signal.SIGTERM)
                        break
                    time.sleep(3)
                child.wait()
            print(f"[repair] wave exit={child.returncode} elapsed={time.monotonic() - started:.0f}s "
                  f"peak={peak:.1f}GiB reason={reason} n={len(names)}", flush=True)
            manifest_path = folder / "report_manifest.json"
            if not manifest_path.exists():
                print("[repair] wave produced no manifest", flush=True)
                continue
            manifest = json.loads(manifest_path.read_text())
            state = load_state()
            for n in names:
                entry = (manifest.get("factors") or {}).get(n)
                if entry is None:
                    continue
                previous = (known.get(n) or (None, None))[1] or {}
                prev_reason = str(previous.get("reason") or "")
                new_reason = str(entry.get("reason") or "")
                if entry.get("status") == "unavailable" and new_reason == prev_reason:
                    # Identical verdict: keep the earlier evidence untouched.
                    continue
                # "Not retryable" is not the same thing as "fixed".  A wave that
                # merely trades one failure for another must not be reported as
                # progress, and its new reason is recorded rather than hidden.
                improved = entry.get("status") != "unavailable"
                isolated = dict(manifest, factors={n: entry})
                if entry.get("artifact"):
                    isolated["factors"] = {n: dict(entry, artifact=str((folder / entry["artifact"]).resolve()))}
                (WORK / "factors" / n).mkdir(parents=True, exist_ok=True)
                (WORK / "factors" / n / "report_manifest.json").write_text(
                    json.dumps(isolated, ensure_ascii=False, indent=1))
                import incremental_factor_intake as intake
                st = state["factors"].setdefault(n, {})
                st["status"] = ("unavailable" if entry.get("status") == "unavailable"
                                else "evaluated" if intake.passes_weekly_gate(entry.get("metrics") or {})
                                else "below_gate")
                st["metrics"] = entry.get("metrics") or {}
                st["reason"] = entry.get("reason") or ""
                st["repair"] = dict(kind=classes.get(n), landing_start=landing_start,
                                    round=round_index, improved=bool(improved),
                                    previous_reason=prev_reason,
                                    at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
                print(f"[repair] {n} ({classes.get(n)}) -> {st['status']}", flush=True)
                progressed = progressed or improved
            save_state(state)
            manifest_path.unlink(missing_ok=True)
        if not progressed:
            print("[repair] no further progress; stopping", flush=True)
            return


def print_status():
    payload = json.loads(CANDIDATES.read_text()) if CANDIDATES.exists() else {"candidates": []}
    state = load_state()
    import collections
    total = len(payload["candidates"])
    counts = collections.Counter(state["factors"].get(c["page_name"], {}).get("status", "pending")
                                 for c in payload["candidates"])
    print(f"candidates={total}")
    for k, v in counts.most_common():
        print(f"  {k:14s} {v}")
    gate = [c for c in payload["candidates"]
            if state["factors"].get(c["page_name"], {}).get("status") == "evaluated"]
    print(f"evaluated-and-above-gate={len(gate)}")
    fails = [(n, s.get("reason")) for n, s in state["factors"].items()
             if s.get("status") == "failed"]
    for n, r in fails[:20]:
        print(f"  FAIL {n}: {r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--selected", type=Path,
                    default=WORK / "new_mining_selected.json")
    ap.add_argument("--only-this-week", action="store_true")
    ap.add_argument("--run-wave", type=Path)
    ap.add_argument("--wave-out", type=Path)
    ap.add_argument("--waves", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=6)
    ap.add_argument("--fe-backend", default="auto")
    ap.add_argument("--qe-backend", choices=("auto", "cpu", "cuda", "cuda_strict"), default="auto")
    ap.add_argument("--memory-gib", type=int, default=24)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--root-workers", type=int, default=6)
    ap.add_argument("--window-years", type=int, default=3)
    ap.add_argument("--landing-start", default=None,
                    help="earliest landing date; earlier than the report window start gives warm-up history")
    ap.add_argument("--repair-warmup", action="store_true",
                    help="re-land only candidates that failed coverage purely on a short warm-up boundary")
    ap.add_argument("--repair-all", action="store_true",
                    help="re-land every candidate whose failure is an environment/warm-up defect "
                         "(admission denials caused by a concurrent shard, window-start shortfall). "
                         "Run with no other FactorEngine job on the host.")
    ap.add_argument("--repair-rounds", type=int, default=2)
    ap.add_argument("--wave-time-limit-s", type=int, default=None,
                    help="per-wave wall clock cap; default is max(1800, 900 * wave size)")
    ap.add_argument("--host-mem-floor-gib", type=float, default=8.0,
                    help="abort a wave when host MemAvailable drops below this")
    ap.add_argument("--reuse", action="store_true")
    ap.add_argument("--force-landing", action="store_true")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()

    if args.freeze:
        freeze(args.selected, CANDIDATES, args.only_this_week)
        return
    if args.run_wave:
        records = json.loads(args.run_wave.read_text())
        run_wave_child(records, args.wave_out, batch_size=args.batch_size,
                       fe_backend=args.fe_backend, qe_backend=args.qe_backend,
                       force_landing=args.force_landing, window_years=args.window_years,
                       landing_start=args.landing_start)
        return
    if args.repair_warmup:
        # Superseded: a warm-up shortfall is no longer a landing problem (the
        # panel starts on FULL_WINDOW_START, so no earlier history exists) — it
        # is a coverage-verdict problem handled by --repair-all's `coverage`
        # class, which re-runs the evaluation and skips landing entirely.
        print("[repair] --repair-warmup is superseded by --repair-all (coverage class)", flush=True)
        repair_all(args.batch_size, args.fe_backend, args.qe_backend, args.window_years,
                   args.threads, args.root_workers, rounds=args.repair_rounds,
                   landing_start=args.landing_start or "2016-01-04",
                   memory_gib=args.memory_gib,
                   host_mem_floor_gib=args.host_mem_floor_gib,
                   wave_time_limit_s=args.wave_time_limit_s)
        return
    if args.repair_all:
        repair_all(args.batch_size, args.fe_backend, args.qe_backend, args.window_years,
                   args.threads, args.root_workers, rounds=args.repair_rounds,
                   landing_start=args.landing_start or "2016-01-04",
                   memory_gib=args.memory_gib,
                   host_mem_floor_gib=args.host_mem_floor_gib,
                   wave_time_limit_s=args.wave_time_limit_s)
        return
    if args.waves:
        run_waves(args.limit, args.batch_size, args.fe_backend, args.qe_backend,
                  args.memory_gib, args.threads, args.reuse, args.force_landing,
                  args.root_workers, args.window_years,
                  host_mem_floor_gib=args.host_mem_floor_gib,
                  wave_time_limit_s=args.wave_time_limit_s)
        return
    if args.status:
        print_status()
        return
    ap.print_help()


if __name__ == "__main__":
    main()
