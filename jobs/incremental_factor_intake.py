#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""增量因子入库：26 条新因子 → 去重/落盘/评估/聚类/优化/页面注入/json 写回。

用法:
  OMP_NUM_THREADS=31 .venv/bin/python jobs/incremental_factor_intake.py --manifest /tmp/new50_selected.json
  OMP_NUM_THREADS=31 .venv/bin/python jobs/incremental_factor_intake.py --manifest /tmp/new50_selected.json --limit 2

断点续跑：状态 /tmp/intake_state.json {factor_name: {"stage": "...", "result": {...}}}
已完成 stage 跳过。每 stage 完成立即写盘（write-early）。

口径（与既有 470 池一致）：
  - 收益 = AdjVwap.pct_change().shift(-2)  (vwap-to-vwap 后复权，企业级 shift(-2))
  - 评估窗 2016-01-04..2018-06-30（与 manifest rank_ic_local 同口径，已实测复现）
  - 逐日 spearman rankic，min_universe=30
  - is_flipped=True 的因子评估负矩阵（×-1 后让 rank_ic 为正）
"""
import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from html import escape
from datetime import datetime
from zoneinfo import ZoneInfo
from uuid import uuid4

PIPELINE_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
EVALUATION_CONVENTION_VERSION = "equal-amount-gross100-actual-volume-v3-20260907"


def canonical_actual_volume_dsl(formula):
    """Explicit user contract: adj-table prices, actual Volume, no Factor usage."""
    import ast
    def column(node):
        if isinstance(node, ast.Name):
            return node.id
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "col" and len(node.args) == 1
                and isinstance(node.args[0], ast.Constant)):
            return node.args[0].value
        return None
    def volume():
        return ast.Call(func=ast.Name(id="col", ctx=ast.Load()), args=[ast.Constant("Volume")], keywords=[])
    class Normalize(ast.NodeTransformer):
        def visit_BinOp(self, node):
            if isinstance(node.op, ast.Div) and column(node.left) in {"Volume", "volume"} and column(node.right) == "Factor":
                return volume()
            return self.generic_visit(node)
        def visit_Call(self, node):
            if (isinstance(node.func, ast.Name) and node.func.id in {"safe_div", "safe_div_null", "divide"}
                    and len(node.args) == 2 and column(node.args[0]) in {"Volume", "volume"}
                    and column(node.args[1]) == "Factor"):
                return volume()
            if column(node) in {"volume", "Volume"}:
                return volume()
            return self.generic_visit(node)
        def visit_Name(self, node):
            return volume() if node.id in {"volume", "Volume"} else node
    tree = Normalize().visit(ast.parse(formula, mode="eval"))
    if any(column(n) == "Factor" for n in ast.walk(tree)):
        raise ValueError("Factor forbidden: explicit source-based expansion required; do not erase arbitrary Factor arithmetic")
    return ast.unparse(ast.fix_missing_locations(tree))


def invalidate_old_evaluations(state):
    """Run under queue lock; preserve prior receipts, never relabel old results."""
    for name, entry in state.get("factors", {}).items():
        if entry.get("evaluation_version") == EVALUATION_CONVENTION_VERSION:
            continue
        state.setdefault("evaluation_history", {}).setdefault(name, []).append(dict(entry))
        state["factors"][name] = dict(entry, status="retry_pending",
                                      evaluation_version=EVALUATION_CONVENTION_VERSION,
                                      optimization_status="requires_reevaluation")


def report_time():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")


def evaluation_banner(document, provenance=None):
    """Rendering time never substitutes for the persisted evaluation time."""
    p = provenance or {}
    text = (f"评估完成时间（北京时间）：{p.get('completed_at') or '未记录，旧结果待重测'} · "
            f"运行编号：{p.get('run_id') or '未知'} · "
            f"主链路源码 SHA256：{p.get('pipeline_source_sha256') or '未知'} · "
            f"HTML 生成时间（北京时间）：{report_time()}")
    banner = '<aside id="evaluation-version" style="padding:14px 24px;background:#fff3cd;color:#513c06;overflow-wrap:anywhere">' + escape(text) + '</aside>'
    return document.replace('<body>', '<body>' + banner, 1)

os.environ.setdefault("OMP_NUM_THREADS", "8")

ROOT = Path("/home/sunhaiwei/quant_projects")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "jobs"))
from factor_report_sources import FULL_WINDOW_END, FULL_WINDOW_START, resolve_raw_matrix
from factor_engine.reporting.quant_evaluator_adapter import evaluate_report_arrays

POOL_JSON = Path("/home/sunhaiwei/factor_delivery_converted/formula_lqtp_all.json")
CLUSTERS_JSON = ROOT / "weekly_backtest_output/factor_clusters.json"
OPT_META_JSON = ROOT / "weekly_backtest_output/optimized_meta.json"
MATRICES_DIR = ROOT / "weekly_backtest_output/factor_matrices_all"
OPT_DIR = ROOT / "weekly_backtest_output/optimized_factors"
REPORTS_DIR = ROOT / "factor_engine/docs/reports/2026-08-23"
FACTORS_DIR = REPORTS_DIR / "factors"
INDEX_HTML = REPORTS_DIR / "index.html"
REPORT_MANIFEST_JSON = REPORTS_DIR / "report_manifest.json"
STATE_JSON = Path("/tmp/intake_state.json")
DONE_JSON = Path("/tmp/intake_done.json")

STAGES = ["dedup_check", "landing", "eval", "cluster_assign", "optimize_lite",
          "page_inject", "json_writeback"]

EVAL_START = "2016-01-04"
EVAL_END = "2018-06-30"
MIN_UNIVERSE = 30
COMMISSION_RATE = 0.0001  # User-confirmed: one-way traded notional, 1 bp.

# 渲染产物中禁止出现的算法名字符串（零出现）
BANNED = ["cogalpha", "alphasage", "evoalpha", "factorminer", "qwen",
          "alpha_sage", "alpha158"]


# --------------------------------------------------------------------------
# state
# --------------------------------------------------------------------------
def load_state():
    if STATE_JSON.exists():
        return json.loads(STATE_JSON.read_text())
    return {}


def save_state(state):
    STATE_JSON.write_text(json.dumps(state, ensure_ascii=False, indent=1))


def stage_done(state, factor_name, stage):
    s = state.get(factor_name)
    return bool(s and s.get("stage") == stage)


# --------------------------------------------------------------------------
# loaders
# --------------------------------------------------------------------------
def load_manifest(path):
    return json.loads(Path(path).read_text())


def discover_price_dsl_candidates(root, existing_records, *, since=None):
    """Import explicit DSL metadata, never candidate matrices or Python code.

    Initial supported surface is price-only native DSL. Return/volume units
    and foreign DSL dialects require their own validated translation contract.
    """
    import ast
    from datetime import datetime, timedelta
    if since is None:
        today = datetime.now().date()
        since = (today - timedelta(days=today.weekday())).strftime("%Y%m%d")
    fields = {"close": "AdjClose", "open": "AdjOpen", "high": "AdjHigh",
              "low": "AdjLow", "pre_close": "AdjPreClose", "vwap": "AdjVwap",
              "high_limit": "AdjHighLimit", "low_limit": "AdjLowLimit"}
    known = set(fields.values())
    seen_names = {r.get("page_name") for r in existing_records}
    seen_formulas = {re.sub(r"\s+", "", str(r.get("fe_formula") or "")) for r in existing_records}
    candidates, rejected = [], []
    for path in sorted(Path(root).glob("*/*/metadata*.json")):
        if path.parent.name[:8] < since:
            continue
        payload = json.loads(path.read_text())
        if payload.get("meta", {}).get("market") != "ashare":
            continue
        for record in payload.get("factors", []):
            if record.get("expression_type") not in {"dsl", "python"}:
                continue
            name = str(record.get("factor_name") or record.get("factor_id") or "")
            formula = record.get("formula") or record.get("factor_expression")
            if not name or name in seen_names or not isinstance(formula, str):
                continue
            try:
                original_formula = formula
                if record.get("expression_type") == "python":
                    from fe_code_transpiler import transpile_native_dsl
                    formula = transpile_native_dsl(formula)
                tree = ast.parse(formula, mode="eval")
                call_names = set()
                for node in ast.walk(tree):
                    if isinstance(node, ast.Call):
                        if not isinstance(node.func, ast.Name):
                            raise ValueError("only named DSL operators are accepted")
                        call_names.add(id(node.func))
                    if isinstance(node, (ast.Attribute, ast.Subscript, ast.Lambda, ast.NamedExpr)):
                        raise ValueError("not a native arithmetic DSL expression")
                for node in ast.walk(tree):
                    if isinstance(node, ast.Name) and id(node) not in call_names:
                        if node.id in fields:
                            node.id = fields[node.id]
                        elif node.id not in known:
                            raise ValueError(f"field requires a verified unit mapping: {node.id}")
                translated = ast.unparse(tree)
                key = re.sub(r"\s+", "", translated)
                if key in seen_formulas:
                    continue
                entry = dict(page_name=name, factor_name=name, fe_formula=translated,
                             source_formula=original_formula, source_metadata=str(path),
                             source_language=record.get("expression_type"),
                             campaign=path.parent.name, intake_since=since,
                             is_unlisted_miner=True, can_use_factor_engine=True)
                factor_from_record(entry)
                candidates.append(entry)
                seen_formulas.add(key)
                seen_names.add(name)
            except (ValueError, SyntaxError, TypeError) as exc:
                rejected.append(dict(factor_name=name, reason=str(exc), source=str(path)))
    return candidates, rejected


SOURCE_FIELDS = ("code", "python_code", "python_formula", "source_code", "source_formula",
                 "dsl", "lqtp_formula", "local_formula", "source_metadata", "campaign")


def preserve_formula_source(record):
    """Capture input evidence before canonicalization; never replace it with DSL."""
    if "formula_source" not in record:
        snapshot = {key: record[key] for key in (*SOURCE_FIELDS, "fe_formula") if key in record}
        record["formula_source"] = json.loads(json.dumps(snapshot, ensure_ascii=False))
    return record["formula_source"]


def formula_provenance(record):
    source = preserve_formula_source(record)
    return {**{key: record[key] for key in SOURCE_FIELDS if key in record},
            "formula_source": source,
            "formula_source_sha256": hashlib.sha256(
                json.dumps(source, ensure_ascii=False, sort_keys=True).encode()).hexdigest()}


def formula_source_html(record):
    from html import escape
    source = record.get("formula_source") or {k: record[k] for k in SOURCE_FIELDS if k in record}
    blocks = []
    for key, value in source.items():
        if value and key not in {"source_metadata", "campaign"}:
            blocks.append(f'<h4>{escape(key)}</h4><pre>{escape(str(value))}</pre>')
    return ('<section id="formula-source"><h2>原始公式与 Python 来源（转换前）</h2>'
            '<p>以下为保留的输入证据，不代表可执行 DSL；实际落值使用上方公式。'
            '未提供的 Python 不从 DSL 反向编造。</p>'
            + ('<p>来源由当前因子库补回；旧评估未记录原始源码快照，不能证明当时源码版本。</p>'
               if record.get("formula_source_backfilled") else '')
            + ''.join(blocks) + '</section>')


def load_pool():
    return json.loads(POOL_JSON.read_text())


def resolve_weekly_daily_field(name):
    """Use DataAccess's dataset-scoped authority; never guess units or market."""
    from data_access import get_store
    field = get_store().resolve_fields(
        [name], dataset="ashare_stock_valuation_daily")[0]
    if (field.dataset != "ashare_stock_valuation_daily"
            or field.market != "ashare" or field.frequency != "daily"
            or field.temporal_model != "panel" or field.join_policy != "exact"
            or not field.mining_allowed):
        raise ValueError(f"field requires a separate temporal adapter: {name}")
    return f"valuation.{field.logical_name}"


def resolve_weekly_field_expression(name):
    """Bind registered fields through FE's maintained DataAccess/PIT adapters."""
    from factor_engine.fields.market_registry import MULTI_MARKET_FIELD_REGISTRY
    from factor_engine.api.source_ref import make_source_ref, encode_source_ref

    field = MULTI_MARKET_FIELD_REGISTRY.resolve_field("ashare", name, strict=False)
    if field is None or not field.mining_allowed:
        # DataAccess also registers fields not yet in the FE catalog.
        return f"col({resolve_weekly_daily_field(name)!r})"
    if field.table == "StockValuationDaily" and field.temporal_model == "exact":
        # The child DataAccessSource applies the registered canonical units.
        return f"col({'valuation.' + field.name!r})"
    if field.table in {"StockIncome", "StockBalance", "StockCashFlow", "StockIndicator"}:
        if field.temporal_model != "financial_pit" or not field.strict_pit_allowed:
            raise ValueError(f"uncertified financial availability contract: {name}")
        spec = make_source_ref(field.table, field.source_name)
        expression = f"col({encode_source_ref(spec)!r})"
        # Financial SourceRefs read physical values before PIT row-bundle selection.
        scale = float(field.scale_to_canonical)
        return expression if scale == 1.0 else f"multiply({expression}, {scale!r})"
    raise ValueError(f"registered field needs frequency/source adapter: {name} ({field.table})")


def requested_dsl_records(path):
    """Normalize explicitly supplied daily formulas; keep unresolved fields visible."""
    import ast
    prices = {"close": "AdjClose", "open": "AdjOpen", "high": "AdjHigh",
              "low": "AdjLow", "pre_close": "AdjPreClose", "vwap": "AdjVwap",
              "amount": "AdjAmount", "high_limit": "AdjHighLimit", "low_limit": "AdjLowLimit"}
    physical = set(prices.values()) | {"Volume", "Factor"}
    records, deferred, seen = [], [], set()
    for line_number, raw in enumerate(Path(path).read_text().splitlines(), 1):
        raw = raw.strip()
        if not raw or raw in seen:
            continue
        seen.add(raw)
        name = "weekly_" + hashlib.sha256(raw.encode()).hexdigest()[:16]
        try:
            tree = ast.parse(raw, mode="eval")
            class Fields(ast.NodeTransformer):
                def visit_Call(self, node):
                    if not isinstance(node.func, ast.Name):
                        raise ValueError("only named DSL operators are supported")
                    if node.func.id in {"intraday_activity_duration_curvature", "intra_close_participation", "intraday_bvc_imbalance"}:
                        from factor_engine.api.source_ref import make_source_ref, encode_source_ref
                        op = node.func.id
                        params = {"bar_minutes": 1, "min_coverage": 1.0, "min_bars": 240,
                                  "minute_dataset": "ashare_stock_minute", "timestamp_convention": "bar_end"}
                        inputs = [a.id if isinstance(a, ast.Name) else None for a in node.args]
                        kwargs = {k.arg: ast.literal_eval(k.value) for k in node.keywords}
                        if op == "intraday_activity_duration_curvature" and inputs in [["minute_volume"], ["minute_amount"]]:
                            if set(kwargs) - {"buckets"}: raise ValueError("unsupported activity parameters")
                            params.update(feature="fe_activity_" + inputs[0][7:], buckets=kwargs.get("buckets", 10))
                        elif op == "intra_close_participation" and inputs[0:1] == ["minute_volume"] and len(inputs) in {1, 2}:
                            if set(kwargs) - {"tail_minutes"}: raise ValueError("unsupported participation parameters")
                            if len(inputs) == 2 and kwargs: raise ValueError("duplicate tail_minutes")
                            params.update(feature="fe_close_participation", tail_minutes=ast.literal_eval(node.args[1]) if len(inputs) == 2 else kwargs.get("tail_minutes", 30))
                        elif op == "intraday_bvc_imbalance" and inputs == ["minute_close", "minute_volume"]:
                            if set(kwargs) - {"scale_window"}: raise ValueError("unsupported BVC parameters")
                            params.update(feature="fe_bvc_imbalance", scale_window=kwargs.get("scale_window", 20))
                        else:
                            raise ValueError(f"unsupported minute input signature: {op}")
                        spec = make_source_ref("StockMinuteBar", "Close", transform="intraday_feature", transform_params=params)
                        return ast.parse(f"col({encode_source_ref(spec)!r})", mode="eval").body
                    node.args = [self.visit(a) for a in node.args]
                    for kw in node.keywords:
                        kw.value = self.visit(kw.value)
                    return node
                def visit_Name(self, node):
                    if node.id == "ret":
                        return ast.parse("subtract(safe_div_null(AdjClose, AdjPreClose), 1.0)", mode="eval").body
                    if node.id == "volume":
                        return ast.parse("col('Volume')", mode="eval").body
                    if node.id in prices:
                        return ast.copy_location(ast.Name(id=prices[node.id], ctx=ast.Load()), node)
                    if node.id not in physical:
                        try:
                            expression = resolve_weekly_field_expression(node.id)
                        except Exception as exc:
                            raise ValueError(
                                f"requires multi-dataset field binding: {node.id}; {exc}") from exc
                        return ast.copy_location(ast.parse(expression, mode="eval").body, node)
                    return node
            tree = Fields().visit(tree)
            if any(isinstance(n, (ast.Attribute, ast.Subscript, ast.Lambda, ast.NamedExpr)) for n in ast.walk(tree)):
                raise ValueError("not arithmetic DSL")
            formula = ast.unparse(ast.fix_missing_locations(tree))
            records.append(dict(page_name=name, factor_name=name, fe_formula=formula,
                                source_formula=raw, source_metadata=str(path), source_line=line_number,
                                is_unlisted_miner=True, can_use_factor_engine=True))
        except (ValueError, SyntaxError) as exc:
            deferred.append(dict(page_name=name, source_formula=raw, source_line=line_number,
                                 reason=str(exc), status="requires_adapter"))
    return sorted(records, key=lambda r: len(r["fe_formula"])), deferred


def passes_weekly_gate(metrics):
    import math
    return any(value is not None and math.isfinite(value) and value >= threshold
               for value, threshold in ((metrics.get("rank_ic"), .015),
                                        (metrics.get("ic_ir"), .15)))


def reopen_failed_candidates(state, names):
    """Explicitly retry failures only, retaining each earlier result for audit."""
    names = list(dict.fromkeys(names))
    for name in names:
        prior = state.get("factors", {}).get(name)
        if prior is None or prior.get("status") not in {"failed", "unavailable"}:
            raise ValueError(f"retry requires a failed/unavailable candidate: {name}")
    for name in names:
        prior = state["factors"][name]
        state.setdefault("retry_history", {}).setdefault(name, []).append(dict(prior))
        state["factors"][name] = dict(prior, status="retry_pending")


def overnight_pending_waves(records, state, batch_size):
    """Called under queue.lock; interrupted workers are resumable, not still running."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    for entry in state["factors"].values():
        if entry.get("status") == "running":
            entry["status"] = "retry_pending"
    pending = [r for r in records if state["factors"].get(r["page_name"], {}).get("status")
               not in {"passed", "below_gate", "failed", "unavailable"}]
    return [pending[i:i + batch_size] for i in range(0, len(pending), batch_size)]


def run_overnight_queue(formulas, output_dir, *, memory_gib=8, retry_names=(), wait_for_lock=False,
                        worker_threads=2, batch_size=4, fe_backend="auto", qe_backend="auto",
                        pool_retest=False, reuse_landed=False, auto_finalize=False):
    """Resume isolated mainline evaluations with host and process memory guards."""
    import subprocess
    import signal
    import fcntl
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "queue.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | (0 if wait_for_lock else fcntl.LOCK_NB))
        if pool_retest:
            records, deferred = load_pool(), []
            # Formula provenance remains raw; training alone decides direction.
            records = [dict(r, source_formula=r.get("source_formula") or r.get("fe_formula") or r.get("dsl", ""))
                       for r in records]
            records.sort(key=lambda r: r.get("page_name") != "intraday_overnight_gap_smoothed")
        else:
            records, deferred = requested_dsl_records(formulas)
        state_path = output_dir / "queue_state.json"
        state = json.loads(state_path.read_text()) if state_path.exists() else {"factors": {}}
        known = {record["page_name"] for record in records}
        if set(retry_names) - known:
            raise ValueError("retry candidates must be present in the supplied formulas")
        reopen_failed_candidates(state, retry_names)
        invalidate_old_evaluations(state)
        waves = overnight_pending_waves(records, state, batch_size)
        state["deferred"] = deferred
        state["gate"] = {"rank_ic": .015, "rank_ic_ir": .15, "combine": "or",
                         "window": "full-window exploratory screening; not sealed OOS"}
        def save():
            tmp = state_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2, allow_nan=False))
            os.replace(tmp, state_path)
        save()
        for wave in waves:
            record = wave[0]
            name = record["page_name"]
            while True:
                mem = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
                available_kib = int(mem["MemAvailable"].split()[0])
                if available_kib >= (min(memory_gib, 8) + 10) * 1024**2:
                    task_memory_gib = min(memory_gib, available_kib / 1024**2 - 10)
                    break
                state["status"] = "waiting_for_memory"
                save()
                time.sleep(30)
            # Per-factor artifacts stay isolated for optimization and publication.
            for item in wave:
                item_folder = output_dir / item["page_name"]
                item_folder.mkdir(exist_ok=True)
                (item_folder / "candidate.json").write_text(json.dumps([item], ensure_ascii=False, indent=2))
            wave_id = hashlib.sha256("|".join(r["page_name"] for r in wave).encode()).hexdigest()[:16]
            folder = output_dir / ("wave_" + wave_id)
            folder.mkdir(exist_ok=True)
            candidate_path = folder / "candidate.json"
            candidate_path.write_text(json.dumps(wave, ensure_ascii=False, indent=2))
            target = folder / "report_manifest.json"
            command = [sys.executable, str(Path(__file__).resolve()), "--manifest", str(candidate_path),
                       "--batch-size", str(batch_size), "--fe-backend", fe_backend, "--qe-backend", qe_backend,
                       "--evaluate-only", "--output-manifest", str(target)]
            if not reuse_landed and (pool_retest or any(r["page_name"] in retry_names for r in wave)):
                command.append("--force-landing")
            state["status"] = "running"
            for item in wave:
                state["factors"][item["page_name"]] = dict(
                    status="running", evaluation_version=EVALUATION_CONVENTION_VERSION,
                    formula=item.get("fe_formula", ""), source_formula=item["source_formula"],
                    folder=str(output_dir / item["page_name"]), wave_log=str(folder / "run.log"))
            save()
            print(f"[overnight start] {name}", flush=True)
            threads = str(max(1, worker_threads))
            env = dict(os.environ, DUCKDB_MAX_THREADS=threads, OMP_NUM_THREADS=threads, OPENBLAS_NUM_THREADS=threads,
                       MKL_NUM_THREADS=threads, POLARS_MAX_THREADS=threads,
                       FACTOR_REPORT_FE_BACKEND=fe_backend)
            result_policy = {"process_memory_gib": task_memory_gib, "host_reserve_gib": 10,
                             "worker_threads": int(threads), "batch_size": len(wave),
                             "fe_backend": fe_backend, "qe_backend": qe_backend}
            for item in wave:
                state["factors"][item["page_name"]]["resource_policy"] = result_policy
            save()
            reason = None
            with (folder / "run.log").open("a") as log:
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                         env=env, start_new_session=True)
                started = time.monotonic()
                peak_kib = 0
                while child.poll() is None:
                    try:
                        status = Path(f"/proc/{child.pid}/status").read_text()
                        rss = int(re.search(r"VmRSS:\s+(\d+)", status).group(1))
                        peak_kib = max(peak_kib, rss)
                    except (FileNotFoundError, AttributeError):
                        rss = 0
                    mem_now = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
                    emergency = int(mem_now["MemAvailable"].split()[0]) < 10 * 1024**2
                    timeout_seconds = min(7200, max(1800, len(wave) * 900))
                    if emergency or rss > task_memory_gib * 1024**2 or time.monotonic() - started > timeout_seconds:
                        reason = ("host low-memory guard" if emergency else "process memory limit"
                                  if rss > task_memory_gib * 1024**2 else f"{timeout_seconds // 60}-minute batch time limit")
                        os.killpg(child.pid, signal.SIGTERM)
                        try:
                            child.wait(timeout=15)
                        except subprocess.TimeoutExpired:
                            os.killpg(child.pid, signal.SIGKILL)
                        break
                    time.sleep(2)
                code = child.wait()
            manifest = json.loads(target.read_text()) if code == 0 and target.exists() else None
            for item in wave:
                item_name = item["page_name"]
                result = state["factors"][item_name]
                result.update(exit_code=code, peak_rss_kib=peak_kib)
                item_folder = output_dir / item_name
                entry = manifest.get("factors", {}).get(item_name) if manifest else None
                if entry is not None:
                    isolated = dict(manifest, factors={item_name: entry})
                    if entry.get("artifact"):
                        isolated["factors"] = {item_name: dict(entry, artifact=str((target.parent / entry["artifact"]).resolve()))}
                    (item_folder / "report_manifest.json").write_text(json.dumps(isolated, ensure_ascii=False, indent=2))
                    result["status"] = "unavailable" if entry.get("status") == "unavailable" else (
                        "passed" if passes_weekly_gate(entry["metrics"]) else "below_gate")
                    result["metrics"] = entry.get("metrics", {})
                    result["optimization_status"] = "pending_library_integration"
                else:
                    result.update(status="failed", reason=reason or "see wave_log")
                print(f"[overnight done] {item_name}: {result['status']}", flush=True)
            save()
            if auto_finalize:
                # Publish verified raw results first, even if optimization fails.
                for stage, action in (("publish", publish_completed_queue),
                                      ("optimize", lambda directory: run_optimization_queue(
                                          directory, names={r["page_name"] for r in wave})),
                                      ("publish_optimized", publish_completed_queue)):
                    if os.environ.get("FACTOR_REPORT_LANDING_FIRST", "1") != "0" and stage != "publish":
                        continue
                    try:
                        action(output_dir)
                    except Exception as exc:
                        state.setdefault("finalization_errors", []).append(
                            dict(stage=stage, time=report_time(), error=str(exc)))
                        save()
                        print(f"[finalize error] {stage}: {exc}", flush=True)
        state["status"] = "evaluation_finished_optimization_and_publication_pending"
        if auto_finalize:
            state["status"] = "evaluation_finished_finalize_attempted"
        save()
        if auto_finalize and os.environ.get("FACTOR_REPORT_LANDING_FIRST", "1") != "0":
            state["status"] = "landing_evaluation_finished_optimizing"
            save()
            try:
                run_optimization_queue(output_dir)
                publish_completed_queue(output_dir)
                state["status"] = "evaluation_finished_finalize_attempted"
            except Exception as exc:
                state.setdefault("finalization_errors", []).append(
                    dict(stage="deferred_optimization", time=report_time(), error=str(exc)))
                state["status"] = "evaluation_finished_optimization_needs_attention"
            save()
    return state


def run_optimization_queue(output_dir, *, names=None):
    """Resume eligible weekly optimizations serially in isolated child processes."""
    import fcntl
    import subprocess
    output_dir = Path(output_dir)
    with (output_dir / "optimizer.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        intake = json.loads((output_dir / "queue_state.json").read_text())
        code_digest = hashlib.sha256(Path(__file__).read_bytes())
        for library_dir in (ROOT / "factor_optimizer/factor_optimizer",
                            ROOT / "factor_preprocess/factor_preprocess",
                            ROOT / "factor_engine/reporting", ROOT / "quant_evaluator"):
            for source in sorted(library_dir.rglob("*.py")):
                code_digest.update(str(source.relative_to(ROOT)).encode())
                code_digest.update(source.read_bytes())
        library_identity = code_digest.hexdigest()
        target = output_dir / "optimizer_state.json"
        state = json.loads(target.read_text()) if target.exists() else {}
        def save():
            tmp = target.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(state, ensure_ascii=False, indent=1, allow_nan=False))
            os.replace(tmp, target)
        for name, entry in intake["factors"].items():
            if names is not None and name not in names:
                continue
            if (entry["status"] != "passed" or
                    entry.get("evaluation_version") != EVALUATION_CONVENTION_VERSION):
                continue
            folder = Path(entry["folder"])
            # Eligibility comes from this factor's verified manifest, never another
            # factor's warning in a shared batch log.
            # Identity changes invalidate the successful-stage checkpoint.
            matrix = MATRICES_DIR / f"{name}.parquet"
            identity = hashlib.sha256((
                (folder / "candidate.json").read_text() + str(matrix.stat().st_mtime_ns)
                + library_identity
            ).encode()).hexdigest()
            if (state.get(name, {}).get("status") in {"optimized", "failed", "timeout"}
                    and state[name].get("identity") == identity):
                continue
            while True:
                memory = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
                if int(memory["MemAvailable"].split()[0]) >= 12 * 1024**2:
                    break
                state[name] = dict(status="waiting_for_memory", identity=identity)
                save()
                time.sleep(30)
            state[name] = dict(status="running", identity=identity)
            save()
            print(f"[optimizer start] {name}", flush=True)
            command = [sys.executable, str(Path(__file__).resolve()), "--manifest",
                       str(folder / "candidate.json"), "--optimize-only"]
            with (folder / "optimizer.log").open("a") as log:
                try:
                    result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                            timeout=2700, check=False)
                    status = "optimized" if result.returncode == 0 else "failed"
                except subprocess.TimeoutExpired:
                    status = "timeout"
            state[name] = dict(status=status, identity=identity, log=str(folder / "optimizer.log"),
                               publication="pending_validation")
            save()
            print(f"[optimizer done] {name}: {status}", flush=True)


def load_clusters():
    return json.loads(CLUSTERS_JSON.read_text())


def load_opt_meta():
    return json.loads(OPT_META_JSON.read_text())


def matrix_source(page):
    """Resolve a raw matrix by verified date coverage, never by folder name."""
    return resolve_raw_matrix(page)


def matrix_path(page):
    source = matrix_source(page)
    return source.path if source.is_full_window else None


# --------------------------------------------------------------------------
# vwap 面板缓存（2016-01-04..2018-06-30）
# --------------------------------------------------------------------------
_HAS_VWAP = None


def load_vwap(*, source=None, start_date=EVAL_START, end_date=EVAL_END):
    global _HAS_VWAP
    uses_default_window = start_date == EVAL_START and end_date == EVAL_END
    if source is None and uses_default_window and _HAS_VWAP is not None:
        return _HAS_VWAP
    import pandas as pd
    managed_source = source is None
    if source is None:
        configure_report_scan_budget()
        os.environ.setdefault("ASHARE_PARQUET_ROOT", str(Path.home() / "cos_data"))
        os.environ.setdefault("DATA_ACCESS_SKIP_COS_MIRROR", "1")
        from factor_engine.storage.factory import build_data_source

        source = build_data_source({
            "type": "data_access",
            "dataset": "ashare_stock_daily_adj",
            "start_date": start_date,
            "end_date": end_date,
        })
    try:
        values = source.load_column("AdjVwap")
    except Exception as exc:
        from data_access.core.exceptions import ResourceAdmissionError
        start, end = pd.Timestamp(start_date), pd.Timestamp(end_date)
        if not isinstance(exc, ResourceAdmissionError) or not managed_source or (end-start).days <= 366:
            raise
        # Full-history scan can itself exceed admission, even without contention.
        # Keep DataAccess governance; reduce each request instead of disabling it.
        print("[vwap] scan admission exceeded; reading disjoint calendar years", flush=True)
        parts = []
        cursor = start
        while cursor <= end:
            stop = min(end, pd.Timestamp(year=cursor.year, month=12, day=31))
            parts.append(load_vwap(start_date=cursor.strftime("%Y-%m-%d"),
                                   end_date=stop.strftime("%Y-%m-%d")))
            cursor = stop + pd.Timedelta(days=1)
        panel = pd.concat(parts).sort_index()
        if panel.index.has_duplicates:
            raise ValueError("duplicate dates in chunked VWAP")
        if uses_default_window:
            _HAS_VWAP = panel
        return panel
    if not isinstance(values.index, pd.MultiIndex) or values.index.nlevels != 2:
        raise ValueError("DataAccess AdjVwap must use timestamp × instrument MultiIndex")
    panel = values.unstack(level=-1).sort_index().astype("float64")
    panel.index = pd.to_datetime(panel.index)
    if uses_default_window and source is not None and source.__class__.__module__.startswith("factor_engine"):
        _HAS_VWAP = panel
    return panel


def load_matrix(page, flip=False):
    import pandas as pd
    p = matrix_path(page)
    if p is None:
        return None
    m = pd.read_parquet(p)
    if flip:
        m = -m
    return m


# --------------------------------------------------------------------------
# 评估核心：逐日 spearman rankic / ic_ir（vwap-to-vwap shift(-2)）
# --------------------------------------------------------------------------
def eval_matrix(mat, vwap=None):
    """Evaluate a raw matrix through the canonical QuantEvaluator adapter."""
    import numpy as np
    if vwap is None:
        vwap = load_vwap()
    common = mat.index.intersection(vwap.index)
    if len(common) == 0:
        return {"rank_ic": 0.0, "ic_ir": 0.0, "n_days": 0, "error": "no overlap"}
    fv = mat.reindex(index=common)
    vv = vwap.reindex(index=common)
    cols = vv.columns.intersection(fv.columns)
    if len(cols) == 0:
        return {"rank_ic": 0.0, "ic_ir": 0.0, "n_days": 0, "error": "no common cols"}
    fv = fv[cols].values.astype(np.float64)
    vv = vv[cols]
    fwd = vv.pct_change(fill_method=None).shift(-2).values.astype(np.float64)
    evaluated = evaluate_report_arrays(
        fv,
        fwd,
        commission_rate=COMMISSION_RATE,
        n_quantiles=10,
        min_assets=MIN_UNIVERSE,
        min_ic_periods=20,
        direction_training_periods=len(common),
    )
    return {
        "rank_ic": evaluated.mean_rank_ic,
        "ic_ir": evaluated.rank_ic_ir,
        "n_days": evaluated.valid_return_periods,
        "std": evaluated.rank_ic_std,
        "direction": evaluated.direction,
        "ls_sharpe": evaluated.sharpe,
        "ls_annual": evaluated.annualized_return,
        "ls_cumulative": evaluated.cumulative_return,
        "ls_mdd": evaluated.max_drawdown,
        "ls_winrate": evaluated.win_rate,
        "rank_ic_winrate": getattr(evaluated, "rank_ic_win_rate", None),
        "g10_annual": getattr(evaluated, "top_quantile_annualized_return", None),
        "g1_annual": getattr(evaluated, "bottom_quantile_annualized_return", None),
    }


# --------------------------------------------------------------------------
# stage: dedup_check
# --------------------------------------------------------------------------
def stage_dedup_check(factor):
    page = factor["page_name"]
    pool = load_pool()
    in_pool = page in {r.get("page_name") for r in pool}
    matrix = matrix_path(page)
    result = {
        "page_name": page,
        "in_pool": in_pool,
        "matrix_exists": matrix is not None,
        "matrix_path": str(matrix) if matrix else None,
        "ok": (not in_pool) and matrix is not None,
    }
    return result


# --------------------------------------------------------------------------
# stage: landing
# --------------------------------------------------------------------------
def stage_landing(factor):
    page = factor["page_name"]
    matrix = matrix_path(page)
    if matrix is not None:
        return {"landed": True, "path": str(matrix), "skipped": True}
    return {"landed": False, "error": "matrix missing"}


def land_factor_batch(factors, *, engine, sink, workers=None, full_history=False):
    """Compile and land one incremental wave through FactorEngine ``run_many``.

    The sink boundary keeps factor matrices out of the aggregate result so a
    weekly batch does not retain every full panel in RAM.
    """
    parsed = []
    for record in factors:
        name = str(record.get("page_name") or record.get("factor_name") or "").strip()
        formula = str(record.get("fe_formula") or "").strip()
        if not name or not formula:
            raise ValueError(f"factor record requires page_name and fe_formula: {name or '<unnamed>'}")
        parsed.append(factor_from_record(record))

    workers = int(os.environ.get("FACTOR_REPORT_ROOT_WORKERS", "1")) if workers is None else int(workers)
    if not 1 <= workers <= 16:
        raise ValueError("factor report root workers must be between 1 and 16")
    workers = min(workers, max(1, len(parsed)))
    runner = engine.run_many_parallel if workers > 1 else engine.run_many
    parallel_options = {"n_jobs": workers} if workers > 1 else {"warmup_clusters": True}
    print(f"[factor landing] factors={len(parsed)} root_workers={workers} streaming_sink=True", flush=True)
    runner(
        parsed,
        enable_cse=True,
        auto_warmup=not full_history,
        trim_warmup=not full_history,
        input_dq_check=True,
        pit_enforce=True,
        pit_forbid_forward_fill=True,
        **parallel_options,
        result_policy="sink",
        sink=sink,
    )
    return {"landed": [factor.name for factor in parsed], "count": len(parsed)}


def factor_from_record(record):
    """Build a Factor from canonical JSON AST or native FactorEngine DSL."""
    from factor_engine.api.dsl_parser import parse_factor
    from factor_engine.api.factor import Factor

    preserve_formula_source(record)
    name = str(record.get("page_name") or record.get("factor_name") or "").strip()
    formula = str(record.get("fe_formula") or record.get("dsl") or "").strip()
    if not formula or re.fullmatch(r"MYDSL\([^()]*\)", formula):
        code = next((record[k] for k in ("code", "python_code", "python_formula", "source_code")
                     if record.get(k)), None)
        if not code:
            raise ValueError("No executable DSL or original Python source supplied")
        from fe_code_transpiler import transpile_native_dsl
        formula = transpile_native_dsl(code)
    try:
        node = json.loads(formula)
    except json.JSONDecodeError:
        if "o[" in formula:
            from fe_code_transpiler import native_dsl_from_expression
            formula = native_dsl_from_expression(formula)
        # Native DSL bare identifiers resolve through the semantic catalog.
        # Volume resolves to adjusted logical `volume`; an explicit /Factor
        # would then adjust twice (or fail derived-field projection). Bind only
        # these explicitly spelled physical inputs via FE's existing col API.
        # Lowercase logical fields retain their catalog semantics.
        import ast

        class PhysicalInputs(ast.NodeTransformer):
            def visit_Name(self, node):
                if node.id in {"Volume", "Factor"}:
                    return ast.copy_location(ast.Call(
                        func=ast.Name(id="col", ctx=ast.Load()),
                        args=[ast.Constant(value=node.id)], keywords=[]), node)
                return node

        formula = canonical_actual_volume_dsl(formula)
        tree = ast.parse(formula, mode="eval")
        bound = PhysicalInputs().visit(tree)
        formula = ast.unparse(ast.fix_missing_locations(bound))
        # Persist the executable formula so reports and future retries use the
        # same explicit physical bindings; source_formula remains untouched.
        record["fe_formula"] = formula
        return parse_factor(formula, name=name)
    if not isinstance(node, dict) or "kind" not in node:
        return parse_factor(formula, name=name)
    record["fe_formula"] = formula
    if '"Factor"' in formula:
        raise ValueError("Factor forbidden in JSON DSL; convert to certified actual-volume expression")

    from factor_engine.api.cleaned_ops import make_cleaned_call_factory
    from factor_engine.api.columns import col
    import factor_engine.cleaned_operators as cleaned_operators

    cleaned_operators.load_all(include_research=False)
    operator_aliases = {"and": "and_", "or": "or_", "not": "not_"}

    def build(current):
        kind = current.get("kind")
        if kind == "column":
            return col("Volume" if str(current["name"]) in {"Volume", "volume"} else str(current["name"]))
        if kind == "literal":
            return current.get("value")
        if kind != "call":
            raise ValueError(f"unsupported FactorEngine JSON node kind: {kind!r}")
        operator = operator_aliases.get(str(current.get("op")), str(current.get("op")))
        arguments = [build(argument) for argument in current.get("args", [])]
        keywords = dict(current.get("kwargs") or {})
        return make_cleaned_call_factory(operator)(*arguments, **keywords)

    return Factor(name=name, expr=build(node), source_expr=formula)


def ensure_local_minute_history(root=None, history_root=None):
    """Incrementally expose existing local daily partitions; never copy/overwrite data."""
    root = Path(root or Path.home() / "cos_data/StockMinuteBar")
    history_root = Path(history_root or "/srv/quant/research/model/mining/users/zhangborui/data/a_share/lqtp_data/StockMinuteBar")
    if not history_root.is_dir():
        raise FileNotFoundError(f"registered minute history directory missing: {history_root}")
    root.mkdir(parents=True, exist_ok=True)
    added = 0
    for original in history_root.glob("*.parquet"):
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}\.parquet", original.name):
            continue
        destination = root / original.name
        if destination.exists():
            continue
        if destination.is_symlink():
            raise FileNotFoundError(f"broken minute history binding: {destination}")
        try:
            destination.symlink_to(original.resolve(strict=True))
            added += 1
        except FileExistsError:
            pass  # Concurrent incremental writer owns this partition.
    if added:
        from data_access.read.manifest import bump_manifest_epoch
        bump_manifest_epoch(root)
    return added


def configure_report_scan_budget():
    """Scan I/O bytes are not resident memory; keep DA's memory gate intact."""
    from data_access.runtime.resource_governor import get_global_governor
    limit = int(os.environ.get("FACTOR_REPORT_SCAN_INFLIGHT_GIB", "64"))
    if not 1 <= limit <= 128:
        raise ValueError("report scan-inflight budget must be 1..128 GiB")
    get_global_governor().set_max_total_scan_bytes_inflight(limit * 1024**3)


def build_incremental_engine(*, start_date, end_date, backend_name="polars_long", records=(), instrument_filter=None):
    """Build the mainline FactorEngine over its canonical DataAccess source."""
    configure_report_scan_budget()
    os.environ.setdefault("ASHARE_PARQUET_ROOT", str(Path.home() / "cos_data"))
    os.environ.setdefault("DATA_ACCESS_SKIP_COS_MIRROR", "1")
    import ast
    from factor_engine.api.source_ref import decode_source_ref
    for record in records:
        try:
            nodes = ast.walk(ast.parse(str(record.get("fe_formula", "")), mode="eval"))
            refs = [decode_source_ref(n.value) for n in nodes
                    if isinstance(n, ast.Constant) and isinstance(n.value, str)]
        except SyntaxError:
            refs = []
        if any(ref and ref.table in {"StockMinuteBar", "MinuteBar"} for ref in refs):
            added = ensure_local_minute_history(Path(os.environ["ASHARE_PARQUET_ROOT"]) / "StockMinuteBar")
            if added: print(f"[minute history] bound {added} existing local daily partitions", flush=True)
            break
    from factor_engine.backend.factory import build_backend
    from factor_engine.runtime.engine import FactorEngine
    from factor_engine.storage.factory import build_data_source

    daily = {
        "type": "data_access",
        "dataset": "ashare_stock_daily_adj",
        **({"start_date": start_date} if start_date is not None else {}),
        "end_date": end_date,
        "read_auto": True,
        **({"instrument_filter": instrument_filter} if instrument_filter is not None else {}),
    }
    config = {
        "type": "composite", "anchor": "pv", "anchor_column": "AdjClose",
        "sources": {"pv": daily, "valuation": {
            "type": "data_access", "dataset": "ashare_stock_valuation_daily",
            **({"start_date": start_date} if start_date is not None else {}),
            "end_date": end_date, "read_auto": True,
            **({"instrument_filter": instrument_filter} if instrument_filter is not None else {}),
        }},
        # A daily valuation panel must not be forward-filled across missing days.
        "joins": {"valuation": "exact"},
    }
    needs_valuation = any("valuation." in str(r.get("fe_formula", "")) for r in records)
    if needs_valuation:
        # CompositeDataSource exposes Series; use FE's maintained long-table
        # bridge for long backends instead of calling a missing scan method.
        config = {"type": "long_table", "inner": config}
    source = build_data_source(config if needs_valuation else daily)
    # Logical SourceRef children inherit the enclosing read window even when
    # the daily anchor is wrapped by a composite/long-table adapter.
    source.start_date = start_date
    source.end_date = end_date
    source.instrument_filter = instrument_filter
    # Public HybridBackend 'auto' is not yet wired to run_many. The report's
    # explicit auto policy uses the supported SQL + Polars long-table backend.
    resolved_backend = "auto_long" if backend_name == "auto" else backend_name
    print(f"[FE route] requested={backend_name} resolved={resolved_backend}", flush=True)
    return FactorEngine(build_backend(resolved_backend), source, run_mode="research")


def matrix_sink(output_dir):
    """Create an atomic streaming sink for wide report matrices."""
    import pandas as pd

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def write(name, value):
        series = value.get("result") if isinstance(value, dict) else value
        if not isinstance(series, pd.Series) or not isinstance(series.index, pd.MultiIndex):
            raise TypeError(f"run_many result for {name!r} must be a MultiIndex Series")
        matrix = series.unstack(level=-1).sort_index().astype("float32")
        target = output_dir / f"{name}.parquet"
        temporary = target.with_suffix(".parquet.tmp")
        matrix.to_parquet(temporary)
        os.replace(temporary, target)

    return write


def admit_report_factors(records):
    """One fail-closed DSL/history gate for landing, evaluation and auditing."""
    from factor_engine.ir.analyzer import Analyzer
    analyzer = Analyzer(production=False)
    admitted, rejected = [], {}
    for record in records:
        name = record["page_name"]
        try:
            factor = factor_from_record(record)
            analysis = analyzer.lower(factor.expr)
            record["history_mode"] = ("continuous-source-history" if needs_full_history(analysis)
                                      else "bounded-overlap")
            admitted.append(record)
        except Exception as exc:
            rejected[name] = f"{type(exc).__name__}: {exc}"
    return admitted, rejected


def needs_full_history(analysis):
    from factor_engine.runtime.execution_contract import (
        factor_history_requirement, is_full_history_lookback,
    )
    requirement = factor_history_requirement(getattr(analysis, "ir", None))
    return bool(getattr(analysis, "requires_full_history", False)
            or is_full_history_lookback(getattr(analysis, "lookback", 0))
            or requirement.is_full_history)


def require_bounded_history(analysis, name):
    """Guard retained for consumers that cannot replay continuous history."""
    if needs_full_history(analysis):
        raise ValueError(
            f"{name}: full-history replay/state continuity required; yearly "
            "windowed landing with finite overlap is not certified")


def land_factor_batch_windowed(
    records,
    *,
    backend_name="polars_long",
    start_date=FULL_WINDOW_START,
    end_date=FULL_WINDOW_END,
    window_years=1,
    warmup_days=550,
    output_dir=None,
):
    """Land a factor wave in bounded date windows, then atomically merge.

    FactorEngine still evaluates each factor wave with ``run_many``.  The date
    window prevents a ten-year long table from being materialized in RAM;
    overlap supplies rolling/EMA history and is trimmed before persistence.
    """
    import pandas as pd
    from factor_engine.ir.analyzer import Analyzer

    # Use the engine's execution-contract authority, not operator-name guesses.
    # Run before any source reads or partial output writes.
    analyzer = Analyzer(production=False)
    full_history = False
    for record in records:
        factor = factor_from_record(record)
        full_history = full_history or needs_full_history(analyzer.lower(factor.expr))

    history_batch = int(os.environ.get("FACTOR_REPORT_HISTORY_BATCH", "1"))
    if not 1 <= history_batch <= 4:
        raise ValueError("continuous-history batch must be 1..4")
    if full_history and len(records) > history_batch:
        # A full-source replay must not hold many ten-year panels in memory.
        landed = []
        for offset in range(0, len(records), history_batch):
            result = land_factor_batch_windowed(
                records[offset:offset + history_batch], backend_name=backend_name, start_date=start_date,
                end_date=end_date, window_years=window_years,
                warmup_days=warmup_days, output_dir=output_dir)
            landed.extend(result["landed"])
        return {"landed": landed, "count": len(landed), "history_mode": "continuous-source-history"}

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    names = [str(record["page_name"]) for record in records]
    with tempfile.TemporaryDirectory(prefix="factor_engine_landing_") as temp_name:
        temp_dir = Path(temp_name)
        cursor = start
        part = 0
        while cursor <= end:
            window_end = end if full_history else min(end, cursor + pd.DateOffset(years=window_years) - pd.Timedelta(days=1))
            load_start = cursor - pd.Timedelta(days=warmup_days)
            engine = build_incremental_engine(
                start_date=None if full_history else str(load_start.date()),
                end_date=str(window_end.date()),
                backend_name=backend_name,
                records=records,
            )

            def sink(name, value, *, _part=part, _start=cursor, _end=window_end):
                series = value.get("result") if isinstance(value, dict) else value
                if not isinstance(series, pd.Series) or not isinstance(series.index, pd.MultiIndex):
                    raise TypeError(f"run_many result for {name!r} must be a MultiIndex Series")
                dates = pd.to_datetime(series.index.get_level_values(0))
                selected = series[(dates >= _start) & (dates <= _end)]
                matrix = selected.unstack(level=-1).sort_index().astype("float32")
                matrix.to_parquet(temp_dir / f"{name}.{_part:03d}.parquet")

            if full_history:
                print(f"[factor landing] continuous source history; no annual state reset; roots={len(records)}", flush=True)
                land_factor_batch(records, engine=engine, sink=sink, workers=len(records), full_history=True)
            else:
                land_factor_batch(records, engine=engine, sink=sink)
            cursor = window_end + pd.Timedelta(days=1)
            part += 1

        for name in names:
            pieces = [pd.read_parquet(path) for path in sorted(temp_dir.glob(f"{name}.*.parquet"))]
            if not pieces:
                raise RuntimeError(f"FactorEngine produced no window for {name}")
            matrix = pd.concat(pieces).sort_index()
            matrix = matrix[~matrix.index.duplicated(keep="last")]
            destination = Path(output_dir) if output_dir is not None else MATRICES_DIR
            destination.mkdir(parents=True, exist_ok=True)
            target = destination / f"{name}.parquet"
            temporary = target.with_suffix(".parquet.tmp")
            matrix.to_parquet(temporary)
            os.replace(temporary, target)
    return {"landed": names, "count": len(names), "date_windows": part}


def landing_failure_reason(exc):
    detail = f"{type(exc).__name__}: {exc}"
    report = getattr(exc, "report", None)
    if report is not None and callable(getattr(report, "to_dict", None)):
        detail += "; diagnostics=" + json.dumps(report.to_dict(), ensure_ascii=False, default=str)
    return detail


def land_missing_factors(factors, *, batch_size=8, backend_name="polars_long", force=False):
    """Land missing executable DSL factors in bounded run_many waves."""
    receipts = ROOT / "weekly_backtest_output" / "actual_volume_landing_receipts"
    def needs_refresh(record):
        formula = str(record.get("fe_formula", ""))
        if not re.search(r"\b(?:Volume|volume|Factor)\b", formula):
            return False
        receipt = receipts / (hashlib.sha256(record["page_name"].encode()).hexdigest() + ".json")
        return not receipt.exists() or json.loads(receipt.read_text()).get("formula") != formula
    missing = [record for record in factors
               if force or needs_refresh(record) or matrix_path(record["page_name"]) is None
               or not matrix_source(record["page_name"]).is_full_window]
    # Capability metadata is advisory and is absent on older/new-mining
    # manifests.  The formula itself is authoritative: always try FE first.
    pending = [record for record in missing if str(record.get("fe_formula") or "").strip()]
    unsupported = [record["page_name"] for record in missing if record not in pending]
    if not pending:
        return {"landed": [], "count": 0, "pending": len(missing), "python_fallback": unsupported}
    landed = []
    errors = {}
    backend_chain = ["auto", "polars_long", "duckdb_sql", "pandas"] if backend_name == "auto" else [backend_name]
    def land_partition(wave, backend_index=0):
        try:
            selected_backend = backend_chain[backend_index]
            result = land_factor_batch_windowed(wave, backend_name=selected_backend)
            landed.extend(result["landed"])
            receipts.mkdir(parents=True, exist_ok=True)
            for record in wave:
                if record["page_name"] in result["landed"]:
                    record["landing_backend"] = "auto_long" if selected_backend == "auto" else selected_backend
                    receipt = receipts / (hashlib.sha256(record["page_name"].encode()).hexdigest() + ".json")
                    temporary = receipt.with_suffix(".tmp")
                    temporary.write_text(json.dumps({"formula": record["fe_formula"], "version": EVALUATION_CONVENTION_VERSION}))
                    os.replace(temporary, receipt)
        except Exception as wave_error:
            if len(wave) == 1:
                detail = landing_failure_reason(wave_error)
                capability_error = (isinstance(wave_error, NotImplementedError)
                    or (isinstance(wave_error, AttributeError)
                        and "has no attribute 'scan_polars_long'" in detail)) or any(
                    word in detail.lower() for word in ("unsupported", "not supported", "not implemented", "emitter", "capability"))
                if capability_error and backend_index + 1 < len(backend_chain):
                    wave[0].setdefault("landing_backend_fallbacks", []).append(
                        {"backend": backend_chain[backend_index], "reason": detail})
                    print(f"[FE fallback] {wave[0]['page_name']}: {backend_chain[backend_index]} -> {backend_chain[backend_index+1]}: {detail}", flush=True)
                    land_partition(wave, backend_index + 1)
                    return
                errors[wave[0]["page_name"]] = landing_failure_reason(wave_error)
                return
            # Bisect only failed partitions; healthy siblings retain shared
            # reads/CSE and parallel root execution. Do not retry a singleton twice.
            middle = len(wave) // 2
            print(f"[landing split] failed={len(wave)} left={middle} right={len(wave)-middle}", flush=True)
            land_partition(wave[:middle], backend_index)
            land_partition(wave[middle:], backend_index)
    history_batch = int(os.environ.get("FACTOR_REPORT_HISTORY_BATCH", "1"))
    if not 1 <= history_batch <= 4:
        raise ValueError("continuous-history batch must be 1..4")
    continuous = [r for r in pending if r.get("history_mode") == "continuous-source-history"]
    bounded = [r for r in pending if r.get("history_mode") != "continuous-source-history"]
    for group, size in ((bounded, batch_size), (continuous, history_batch)):
        for offset in range(0, len(group), size):
            land_partition(group[offset:offset + size])
    return {
        "landed": landed,
        "count": len(landed),
        "pending": len(missing),
        "python_fallback": unsupported,
        "factor_engine_errors": errors,
    }


def evaluation_candidates_after_landing(records, landing):
    """Failed refreshes cannot silently reuse old matrices or old stage state."""
    failures = {name: f"FactorEngine landing failed; old matrix excluded: {cause}"
                for name, cause in landing.get("factor_engine_errors", {}).items()}
    for name in landing.get("python_fallback", []):
        failures[name] = "executable DSL unavailable; old matrix excluded"
    return [r for r in records if r["page_name"] not in failures], failures


# --------------------------------------------------------------------------
# stage: eval
# --------------------------------------------------------------------------
def stage_eval(factor):
    page = factor["page_name"]
    mat = load_matrix(page, flip=False)
    if mat is None:
        return {"error": "matrix missing"}
    res = eval_matrix(mat)
    res["is_flipped"] = res.get("direction") == -1
    res["matrix_path"] = str(matrix_path(page))
    return res


def evaluate_factor_batch(
    factors,
    *,
    vwap=None,
    matrix_loader=None,
    evaluator=evaluate_report_arrays,
    batch_size=8,
    backend="auto",
):
    """Evaluate factor matrices in bounded QuantEvaluator tiles."""
    import numpy as np
    import pandas as pd
    from factor_engine.reporting.quant_evaluator_adapter import evaluate_report_batch
    provenance = {"run_id": uuid4().hex, "started_at": report_time(),
                  "pipeline_source_sha256": PIPELINE_SOURCE_SHA256}

    if not factors:
        return {"factors": {}, "backend_used": set(), "fallbacks": {},
                "unavailable": {}, "dates": pd.DatetimeIndex([]),
                "evaluation_provenance": dict(provenance, completed_at=report_time())}
    if evaluator is evaluate_report_arrays:
        evaluator = evaluate_report_batch
    vwap = load_vwap(
        start_date=str(FULL_WINDOW_START.date()),
        end_date=str(FULL_WINDOW_END.date()),
    ) if vwap is None else vwap
    matrix_loader = matrix_loader or (lambda name: load_matrix(name, flip=False))
    names = [str(record["page_name"]) for record in factors]
    evaluated = {}
    backends = set()
    fallbacks = {}
    unavailable = {}
    for offset in range(0, len(names), max(1, batch_size)):
        tile_names = names[offset:offset + max(1, batch_size)]
        matrices = [matrix_loader(name) for name in tile_names]
        missing = [name for name, matrix in zip(tile_names, matrices) if matrix is None]
        for name in missing:
            source = matrix_source(name)
            unavailable[name] = source.reason or "verified full-window factor matrix unavailable"
        available = [(name, matrix) for name, matrix in zip(tile_names, matrices) if matrix is not None]
        if not available:
            continue
        tile_names = [name for name, _ in available]
        matrices = [matrix for _, matrix in available]
        # DataAccess owns the report universe.  A sparse or newly-landed factor
        # must contribute NaNs on its missing cells; it must never truncate the
        # dates/assets (and therefore labels) of every other factor in a tile.
        dates = vwap.index
        columns = vwap.columns
        labels = vwap.reindex(index=dates, columns=columns).pct_change(fill_method=None).shift(-2)
        values = np.stack([
            matrix.reindex(index=dates, columns=columns).values.astype(np.float64)
            for matrix in matrices
        ], axis=-1)
        # A zero-filled factor is not proof an asset existed at signal time.
        # Use observed current-day adjusted prices, never future-return validity,
        # for the causal universe before QE assigns cross-sectional quantiles.
        signal_prices = vwap.reindex(index=dates, columns=columns).to_numpy(dtype=float)
        signal_eligible = np.isfinite(signal_prices) & (signal_prices > 0)
        values = np.where(signal_eligible[:, :, None], values, np.nan)
        train_periods = int((pd.DatetimeIndex(dates) <= pd.Timestamp(EVAL_END)).sum())
        result = evaluator(
            values,
            labels.values.astype(np.float64),
            commission_rate=COMMISSION_RATE,
            factor_ids=tuple(tile_names),
            backend=backend,
            n_quantiles=10,
            min_assets=20,
            min_ic_periods=20,
            direction_training_periods=train_periods,
        )
        for name, factor_result in result.factors.items():
            ic = getattr(factor_result, "rank_ic_series", None)
            if ic is not None and not np.isfinite(ic).any():
                unavailable[name] = (
                    "全窗矩阵未产生任何有效 RankIC；需审计横截面常数、"
                    "有效股票数和标签对齐，不能作为已完成评估发布"
                )
                continue
            evaluated[name] = factor_result
        backends.add(result.backend_used)
        if result.backend_fallback_reason:
            fallbacks[",".join(tile_names)] = result.backend_fallback_reason
    return {
        "factors": evaluated,
        "backend_used": backends,
        "fallbacks": fallbacks,
        "unavailable": unavailable,
        "dates": pd.DatetimeIndex(vwap.index),
        "evaluation_provenance": dict(provenance, completed_at=report_time()),
    }


def report_evaluation_dict(evaluated):
    """Project the canonical QE result into the incremental manifest schema."""
    return {
        "commission_rate": getattr(evaluated, "commission_rate", 0.0),
        "rank_ic": evaluated.mean_rank_ic,
        "ic_ir": evaluated.rank_ic_ir,
        "std": evaluated.rank_ic_std,
        "n_days": evaluated.valid_return_periods,
        "direction": evaluated.direction,
        "is_flipped": evaluated.direction == -1,
        "ls_sharpe": evaluated.sharpe,
        "ls_annual": evaluated.annualized_return,
        "ls_cumulative": evaluated.cumulative_return,
        "ls_mdd": evaluated.max_drawdown,
        "ls_winrate": evaluated.win_rate,
        "rank_ic_winrate": getattr(evaluated, "rank_ic_win_rate", None),
        "g10_annual": getattr(evaluated, "top_quantile_annualized_return", None),
        "g1_annual": getattr(evaluated, "bottom_quantile_annualized_return", None),
    }


def write_report_manifest(records, batch_evaluation, *, target=REPORT_MANIFEST_JSON):
    """Atomically publish the sole metric source for index and detail pages."""
    import numpy as np

    target = Path(target)
    artifact_dir = target.parent / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    dates = batch_evaluation.get("dates")
    by_name = {str(record["page_name"]): record for record in records}
    factors = {}
    for name, evaluated in batch_evaluation["factors"].items():
        metrics = report_evaluation_dict(evaluated)
        record = by_name[name]
        raw_formula = str(record.get("fe_formula") or record.get("formula") or "")
        direction = metrics.pop("direction")
        entry = {
            **formula_provenance(record),
            "factor_name": record.get("factor_name", name),
            "source_formula": record.get("source_formula"),
            "landing_backend": record.get("landing_backend"),
            "landing_backend_fallbacks": record.get("landing_backend_fallbacks", []),
            "source_metadata": record.get("source_metadata"),
            "campaign": record.get("campaign"),
            "direction": direction,
            "is_flipped": metrics.pop("is_flipped"),
            "raw_formula": raw_formula,
            "effective_formula": f"neg({raw_formula})" if direction == -1 and raw_formula else raw_formula,
            "matrix_path": str(matrix_path(name) or ""),
            "metrics": metrics,
        }
        if dates is not None and hasattr(evaluated, "rank_ic_series"):
            safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
            artifact = artifact_dir / f"{safe_name}.npz"
            temporary = artifact.with_suffix(".npz.tmp")
            with temporary.open("wb") as stream:
                np.savez_compressed(
                    stream,
                    dates=np.asarray(dates, dtype="datetime64[ns]"),
                    rank_ic_series=np.asarray(evaluated.rank_ic_series),
                    quantile_returns=np.asarray(evaluated.quantile_returns),
                    quantile_nav=np.asarray(evaluated.quantile_nav),
                    long_short_returns=np.asarray(evaluated.long_short_returns),
                    long_short_nav=np.asarray(evaluated.long_short_nav),
                    long_short_nav_aligned=np.asarray(evaluated.long_short_nav_aligned),
                )
            os.replace(temporary, artifact)
            entry["artifact"] = artifact.relative_to(target.parent).as_posix()
            entry["artifact_sha256"] = hashlib.sha256(artifact.read_bytes()).hexdigest()
        factors[name] = entry
    for name, reason in batch_evaluation.get("unavailable", {}).items():
        record = by_name[name]
        raw_formula = str(record.get("fe_formula") or record.get("formula") or "")
        factors[name] = {
            **formula_provenance(record),
            "factor_name": record.get("factor_name", name),
            "status": "unavailable",
            "reason": reason,
            "direction": None,
            "is_flipped": None,
            "raw_formula": raw_formula,
            "effective_formula": None,
            "matrix_path": "",
            "metrics": {},
        }
    payload = {
        "schema_version": 1,
        "evaluation_version": EVALUATION_CONVENTION_VERSION,
        "portfolio_convention": {"policy": "equal-stock-gross-100-v1", "gross_exposure": 1.0,
                                 "short_margin_ratio": 1.0, "weighting": "equal absolute weight across both legs",
                                 "missing_selected_return": "invalidate day; do not reselect stocks"},
        "evaluation_provenance": batch_evaluation.get("evaluation_provenance"),
        "price_convention": "adj_vwap_t1_to_t2",
        "cost_model": {"one_way_commission_rate": COMMISSION_RATE,
                       "turnover_model": "target_weights_half_L1_times_two; opening charged",
                       "excluded_costs": ["stamp_duty", "slippage", "borrow_cost"],
                       "label": "net of commission only; not all-in trading costs"},
        "training_window": [EVAL_START, EVAL_END],
        "backend_used": sorted(batch_evaluation["backend_used"]),
        "backend_fallbacks": batch_evaluation["fallbacks"],
        "factors": factors,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    def json_safe(value):
        if isinstance(value, float) and not __import__("math").isfinite(value):
            return None
        if isinstance(value, dict):
            return {key: json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [json_safe(item) for item in value]
        return value

    payload = json_safe(payload)
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=1, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, target)
    return payload


def _report_result_from_artifact(entry, artifact_path):
    """Load the chart contract emitted by ``write_report_manifest``.

    Publishing deliberately has no matrix or price-data dependency: a page is
    a pure projection of a hash-verified QuantEvaluator artifact.
    """
    import numpy as np
    import pandas as pd

    expected = str(entry.get("artifact_sha256") or "")
    actual = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    if not expected or actual != expected:
        raise ValueError(f"artifact checksum mismatch for {artifact_path.name}")
    with np.load(artifact_path, allow_pickle=False) as arrays:
        required = {
            "dates", "rank_ic_series", "quantile_returns", "quantile_nav",
            "long_short_returns", "long_short_nav", "long_short_nav_aligned",
        }
        missing = required.difference(arrays.files)
        if missing:
            raise ValueError(f"artifact {artifact_path.name} misses {sorted(missing)}")
        values = {key: arrays[key].copy() for key in required}
    dates = pd.DatetimeIndex(values.pop("dates"))
    if values["quantile_nav"].ndim != 2 or len(dates) != len(values["rank_ic_series"]):
        raise ValueError(f"artifact {artifact_path.name} has inconsistent chart axes")
    metrics = dict(entry.get("metrics") or {})
    return SimpleNamespace(
        commission_rate=metrics.get("commission_rate", 0.0),
        mean_rank_ic=metrics.get("rank_ic"),
        rank_ic_ir=metrics.get("ic_ir"),
        rank_ic_std=metrics.get("std"),
        valid_return_periods=metrics.get("n_days", 0),
        direction=entry.get("direction", 1),
        sharpe=metrics.get("ls_sharpe"),
        annualized_return=metrics.get("ls_annual"),
        cumulative_return=metrics.get("ls_cumulative"),
        max_drawdown=metrics.get("ls_mdd"),
        win_rate=metrics.get("ls_winrate"),
        rank_ic_win_rate=metrics.get("rank_ic_winrate"),
        top_quantile_annualized_return=metrics.get("g10_annual"),
        bottom_quantile_annualized_return=metrics.get("g1_annual"),
        **values,
    ), dates


def _unavailable_report_page(page, entry):
    reason = escape(str(entry.get("reason") or "required full-window matrix is unavailable"))
    formula = escape(str(entry.get("raw_formula") or "—"))
    return f"""<!doctype html><html lang=\"zh-CN\"><meta charset=\"utf-8\"/>
<title>{escape(page)} — 数据待补齐</title><body><main>
<p><a href=\"../index.html\">← 返回汇总</a></p><h1><code>{escape(page)}</code></h1>
<h2>全窗评估暂不可用</h2><p>{reason}</p>
<p>本页未展示或填充任何回测指标；待主链路完成 DataAccess → FactorEngine 落值后，将由同一报告发布流程自动更新。</p>
<h2>FactorEngine DSL</h2><pre>{formula}</pre>
</main></body></html>"""


def _write_manifest_index(report_dir, factors):
    """Render the home page directly from the canonical manifest entries."""
    rows = []
    def formatted(value, spec):
        import math
        return format(value, spec) if value is not None and math.isfinite(value) else "—"
    for number, (page, entry) in enumerate(sorted(factors.items()), start=1):
        metrics = entry.get("metrics") or {}
        status = entry.get("status", "available")
        if status == "unavailable":
            cells = ("—", "—", "—", "暂不可用（详情见原因）")
        else:
            cells = (
                formatted(metrics.get('rank_ic'), '+.4f'),
                formatted(metrics.get('ic_ir'), '+.3f'),
                formatted(metrics.get('ls_sharpe'), '+.2f'),
                ("已翻正" if entry.get("is_flipped") else "原方向") +
                {"passed": " · 通过筛选", "below_gate": " · 未达筛选门槛"}.get(entry.get("screening_status"), ""),
            )
        rows.append(
            f"<tr><td>{number}</td><td><a href=\"factors/factor_{escape(page)}.html\"><code>{escape(page)}</code></a></td>"
            f"<td>{cells[0]}</td><td>{cells[1]}</td><td>{cells[2]}</td><td>{cells[3]}</td></tr>"
        )
    available = sum(entry.get("status") != "unavailable" for entry in factors.values())
    html = f"""<!doctype html><html lang=\"zh-CN\"><meta charset=\"utf-8\"/>
<title>FactorEngine 因子报告</title><body><main><h1>FactorEngine 因子报告</h1>
<p>共 {len(factors)} 个因子；{available} 个已完成全窗评估。所有指标及图表均来自同一 QuantEvaluator manifest/artifact。</p>
<table><thead><tr><th>#</th><th>因子</th><th>RankIC</th><th>IR</th><th>LS Sharpe</th><th>状态</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></main></body></html>"""
    output = Path(report_dir) / "index.html"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".html.tmp")
    temporary.write_text(html, encoding="utf-8")
    os.replace(temporary, output)


def html_eligible(entry):
    if not str(entry.get("raw_formula") or "").strip():
        return False
    # Legacy intermediate aliases have conflicting definitions and no certified expansion.
    if re.search(r"\b(EWMA_up_vol|EWMA_down_vol|Factor)\b", entry.get("raw_formula", "")):
        return False
    return entry.get("status", "available") == "available" and passes_weekly_gate(entry.get("metrics", {}))


def withdraw_factor_html(report_dir, name):
    """Recoverable removal from the served site; evaluation evidence stays intact."""
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        raise ValueError("unsafe factor page name")
    source = Path(report_dir) / "factors" / f"factor_{name}.html"
    if not source.exists():
        return False
    archive = ROOT / "weekly_backtest_output" / "unpublished_html"
    archive.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.move(str(source), str(archive / f"{name}_{uuid4().hex}.html"))
    return True


def publish_completed_queue(queue_dir, report_dir=REPORTS_DIR):
    """Incrementally publish completed artifacts; preserve the rich homepage."""
    import fcntl
    queue_dir, report_dir = Path(queue_dir), Path(report_dir)
    state = json.loads((queue_dir / "queue_state.json").read_text())
    source_records = {r["page_name"]: r for r in load_pool()} if POOL_JSON.exists() else {}
    cache_path = report_dir / "published_evaluations.json"
    with (report_dir / "publication.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
        errors, published, withdrawn = {}, 0, 0
        excluded = {name for name, receipt in cache.items()
                    if receipt.get("screening") != "passed" or not html_eligible(receipt.get("entry", {}))}
        excluded.update(name for name, item in state["factors"].items()
                        if item.get("status") in {"below_gate", "unavailable", "failed"})
        # Old pages may predate the publication manifest entirely. Audit their
        # formula header too, not only entries already tracked in the cache.
        for legacy in (report_dir / "factors").glob("factor_*.html"):
            with legacy.open(encoding="utf-8") as stream:
                header = stream.read(131072)
            if re.search(r'<div\b[^>]*class="formula-wrap"[^>]*>\s*</div>', header):
                excluded.add(legacy.stem.removeprefix("factor_"))
        for name in excluded:
            withdrawn += withdraw_factor_html(report_dir, name)
        for name, item in state["factors"].items():
            if item.get("status") != "passed":
                continue
            path = Path(item["folder"]) / "report_manifest.json"
            if not path.exists():
                continue
            payload = json.loads(path.read_text())
            if payload.get("evaluation_version") != EVALUATION_CONVENTION_VERSION:
                errors[name] = "stale evaluation convention; reevaluation required"
                continue
            entry = payload.get("factors", {}).get(name)
            if entry is None:
                continue
            entry = dict(entry)
            if not entry.get("formula_source") and name in source_records:
                # Recover source evidence only. Never substitute a newer DSL for
                # the expression associated with an already evaluated artifact.
                for key, value in formula_provenance(source_records[name]).items():
                    if not entry.get(key):
                        entry[key] = value
                entry["formula_source_backfilled"] = True
            if not html_eligible(entry):
                excluded.add(name)
                withdrawn += withdraw_factor_html(report_dir, name)
                continue
            excluded.discard(name)
            if entry.get("artifact"):
                artifact = path.parent / entry["artifact"]
                if not artifact.exists() and item.get("wave_log"):
                    artifact = Path(item["wave_log"]).parent / entry["artifact"]
                entry["artifact"] = str(artifact.resolve())
            provenance = payload.get("evaluation_provenance")
            optimization = load_opt_meta().get(name, {}) if OPT_META_JSON.exists() else {}
            receipt = dict(entry=entry, provenance=provenance, screening=item["status"],
                           formula_render_version="preserved-source-v1",
                           evaluation_version=EVALUATION_CONVENTION_VERSION,
                           optimization_sha256=hashlib.sha256(json.dumps(
                               optimization, sort_keys=True).encode()).hexdigest())
            if cache.get(name) == receipt and (report_dir / "factors" / f"factor_{name}.html").exists():
                continue
            try:
                with tempfile.TemporaryDirectory(prefix="report_publish_") as staging:
                    staging = Path(staging)
                    source = staging / "manifest.json"
                    source.write_text(json.dumps(dict(payload, factors={name: entry})))
                    publish_report_from_manifest(source, report_dir=staging, update_homepage=False)
                    (report_dir / "factors").mkdir(exist_ok=True)
                    target = report_dir / "factors" / f"factor_{name}.html"
                    temporary = target.with_suffix(".html.tmp")
                    temporary.write_bytes((staging / "factors" / target.name).read_bytes())
                    os.replace(temporary, target)
                cache[name] = receipt
                published += 1
                print(f"[published] {name}: {item['status']}", flush=True)
            except Exception as exc:
                errors[name] = str(exc)
        # Reuse the canonical table renderer without replacing the full homepage.
        with tempfile.TemporaryDirectory(prefix="report_index_") as staging:
            _write_manifest_index(staging, {
                n: dict(r["entry"], screening_status=r["screening"])
                for n, r in cache.items()
                if r.get("evaluation_version") == EVALUATION_CONVENTION_VERSION
                and r.get("screening") == "passed" and html_eligible(r["entry"]) and n not in excluded})
            table = Path(staging, "index.html").read_text()
            content = table.split("<main>", 1)[1].split("</main>", 1)[0]
        section = ('<section id="latest-evaluations"><h2>最新已完成重测</h2><p>同步时间（北京时间）：'
                   + escape(report_time()) + '。仅展示 RankIC ≥ 0.015 或 RankIC IR ≥ 0.15 的可用因子；仅详情页标注的评估时间代表重测时间。'
                   '其他旧报告区块尚未全量重测。</p>' + content + '</section>')
        index = report_dir / "index.html"
        document = index.read_text()
        document = re.sub(r'<section id="latest-evaluations">.*?</section>', '', document, flags=re.S)
        def remove_stale_row(match):
            names = re.findall(r'href="factors/factor_([^"<>]+)\.html"', match.group(0))
            return '' if any(n in cache or n in excluded for n in names) else match.group(0)
        document = re.sub(r'<tr\b[^>]*>.*?</tr>', remove_stale_row, document, flags=re.S)
        def remove_excluded_highlight(match):
            names = re.findall(r'href="factors/factor_([^"<>]+)\.html"', match.group(0))
            return '' if any(n in excluded for n in names) else match.group(0)
        document = re.sub(r'<li\b[^>]*>.*?</li>', remove_excluded_highlight, document, flags=re.S)
        document = document.replace('<main>', '<main>' + section, 1) if '<main>' in document else document.replace('<body>', '<body>' + section, 1)
        for target, text in [(index, document), (cache_path, json.dumps(cache, ensure_ascii=False, indent=1))]:
            temporary = target.with_suffix(target.suffix + ".tmp")
            temporary.write_text(text, encoding="utf-8")
            os.replace(temporary, target)
        publication_result = {"published_now": published, "withdrawn": withdrawn, "errors": errors}
    # Refresh the established all-factor 2026 page after releasing its shared
    # publication lock. The report only reads completed QE artifacts.
    if report_dir.resolve() == REPORTS_DIR.resolve() and published:
        import subprocess
        try:
            subprocess.run([sys.executable, str(ROOT / "jobs/analyze_2026_robustness.py"),
                            "--mode", "completed"], cwd=ROOT, check=True, timeout=120)
        except Exception as exc:
            publication_result["robustness_refresh_error"] = str(exc)
    return publication_result


def publish_report_from_manifest(manifest_path, *, report_dir=REPORTS_DIR, update_homepage=True):
    """Publish all pages from one verified QE manifest; never recompute metrics."""
    manifest_path = Path(manifest_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or not isinstance(payload.get("factors"), dict):
        raise ValueError("unsupported report manifest schema")
    report_dir = Path(report_dir)
    output_factors = report_dir / "factors"
    output_factors.mkdir(parents=True, exist_ok=True)
    published = available = unavailable = 0
    for page, entry in payload["factors"].items():
        if not html_eligible(entry):
            withdraw_factor_html(report_dir, page)
            continue
        status = entry.get("status", "available")
        if status == "unavailable":
            (output_factors / f"factor_{page}.html").write_text(
                evaluation_banner(_unavailable_report_page(page, entry), payload.get("evaluation_provenance")), encoding="utf-8")
            unavailable += 1
            published += 1
            continue
        artifact_ref = entry.get("artifact")
        if not artifact_ref:
            raise ValueError(f"available factor {page} has no chart artifact")
        result, dates = _report_result_from_artifact(entry, manifest_path.parent / artifact_ref)
        factor = {
            **{key: entry[key] for key in (*SOURCE_FIELDS, "formula_source", "formula_source_sha256", "formula_source_backfilled") if key in entry},
            "page_name": page,
            "factor_name": entry.get("factor_name", page),
            "fe_formula": entry.get("raw_formula", ""),
            "is_flipped": bool(entry.get("is_flipped")),
            "evaluation_provenance": payload.get("evaluation_provenance"),
        }
        rendered = stage_page_inject(
            factor, report_evaluation_dict(result), report_result=result,
            report_dates=dates, out_dir=output_factors,
        )
        if rendered.get("mode") != "full":
            raise RuntimeError(f"factor {page} did not render all charts: {rendered}")
        available += 1
        published += 1
    if update_homepage:
        _write_manifest_index(report_dir, {n: e for n, e in payload["factors"].items() if html_eligible(e)})
        index = report_dir / "index.html"
        index.write_text(evaluation_banner(index.read_text(), payload.get("evaluation_provenance")), encoding="utf-8")
    return {"published": published, "available": available, "unavailable": unavailable}


# --------------------------------------------------------------------------
# stage: cluster_assign
# --------------------------------------------------------------------------
def stage_cluster_assign(factor, eval_result):
    """与 154 簇代表矩阵算 spearman（抽样对齐窗口），|ρ|≥0.7 归簇，否则新簇。"""
    import numpy as np
    from scipy.stats import spearmanr
    page = factor["page_name"]
    is_flipped = bool(factor.get("is_flipped", False))
    clusters = load_clusters()
    reps = clusters.get("representatives") or {}
    fmat = load_matrix(page, flip=False)
    if fmat is None:
        return {"assigned": None, "error": "matrix missing"}

    # 抽样窗口：每簇比较用 60 个对齐交易日（快），取矩阵与代表共同 index 均分抽样
    def sample_rows(mat):
        if mat is None or len(mat) == 0:
            return mat
        n = len(mat)
        if n <= 120:
            return mat
        idx = np.linspace(0, n - 1, 120).astype(int)
        return mat.iloc[idx]

    fmat_s = sample_rows(fmat)
    best = None
    for cid, rep in reps.items():
        rep_factor = rep.get("factor") if isinstance(rep, dict) else rep
        rep_mat = load_matrix(rep_factor, flip=False)
        if rep_mat is None:
            continue
        rmat_s = sample_rows(rep_mat)
        common = fmat_s.index.intersection(rmat_s.index)
        if len(common) < 30:
            continue
        fv = fmat_s.reindex(index=common)
        rv = rmat_s.reindex(index=common)
        cols = fv.columns.intersection(rv.columns)
        if len(cols) < 30:
            continue
        fv = fv[cols]
        rv = rv[cols]
        rho_sum, n = 0.0, 0
        for dt in common:
            x = fv.loc[dt]
            y = rv.loc[dt]
            mask = x.notna() & y.notna()
            if mask.sum() < 30:
                continue
            r, _ = spearmanr(x[mask], y[mask])
            if np.isfinite(r):
                rho_sum += r
                n += 1
        if n == 0:
            continue
        rho = rho_sum / n
        if best is None or abs(rho) > abs(best["rho"]):
            best = {"cluster_id": cid, "rho": float(rho)}

    if best is not None and abs(best["rho"]) >= 0.7:
        res = {"assigned": best["cluster_id"], "rho": best["rho"], "new": False,
               "representative": reps[best["cluster_id"]].get("factor")}
    else:
        # 追加新簇：cluster_members + representatives + page_to_cluster（不改旧键）
        cm = clusters.setdefault("cluster_members", {})
        suffix = len(cm) + 1
        new_id = f"new_{suffix:02d}"
        while new_id in cm or new_id in reps:
            suffix += 1
            new_id = f"new_{suffix:02d}"
        cm[new_id] = [page]
        clusters.setdefault("representatives", {})[new_id] = {
            "factor": page, "best_rankic_ir": 0.0, "best_mean_rankic": 0.0,
            "cluster_size": 1}
        clusters.setdefault("page_to_cluster", {})[page] = new_id
        clusters["num_clusters"] = len(cm)
        CLUSTERS_JSON.write_text(json.dumps(clusters, ensure_ascii=False, indent=1))
        res = {"assigned": new_id, "rho": (best["rho"] if best else None),
               "new": True, "representative": page}

    # 同时把 page_to_cluster 补上（即使是归簇也写回一次）
    if best is not None and abs(best["rho"]) >= 0.7:
        clusters.setdefault("page_to_cluster", {})[page] = best["cluster_id"]
        CLUSTERS_JSON.write_text(json.dumps(clusters, ensure_ascii=False, indent=1))
    return res


# --------------------------------------------------------------------------
# stage: optimize_lite
# --------------------------------------------------------------------------
def stability_objective(metrics, worst_year_return):
    """Versioned, explicit stability-first policy over QE net-of-commission metrics."""
    import math
    from factor_optimizer.search.desirability import desirability_for
    values = [metrics.get(k) for k in ("ls_mdd", "ls_sharpe", "rankic_ir", "ls_annual")]
    if any(v is None or not math.isfinite(v) for v in values + [worst_year_return]):
        raise ValueError("stability selection requires finite QE net performance metrics")
    if metrics["ls_annual"] <= 0 or metrics.get("mean_rankic", 0) <= 0:
        raise ValueError("non-positive net return/IC cannot win by appearing flat")
    components = {
        "max_drawdown": desirability_for("decreasing", [(0.,1.),(.1,.8),(.3,.3),(.6,0.)], metrics["ls_mdd"]),
        "net_sharpe": desirability_for("increasing", [(0.,0.),(1.,.5),(2.,.8),(3.,1.)], metrics["ls_sharpe"]),
        "worst_year": desirability_for("increasing", [(-.3,0.),(0.,.5),(.2,1.)], worst_year_return),
        "rankic_ir": desirability_for("increasing", [(0.,0.),(.15,.4),(.4,.8),(.8,1.)], metrics["rankic_ir"]),
    }
    weights = {"max_drawdown": .45, "net_sharpe": .25, "worst_year": .20, "rankic_ir": .10}
    return {"policy": "net-stability-v1", "weights": weights, "components": components,
            "score": sum(weights[k] * v for k,v in components.items())}


OBJECTIVE_PROFILES = {
    "stability": {"ls_mdd": .30, "drawdown_duration": .15, "worst_year": .20, "year_dispersion": .15, "ls_sharpe": .20},
    "balanced": {"ls_annual": .30, "ls_sharpe": .25, "calmar": .25, "ls_mdd": .20},
    "ic_stability": {"rankic_ir": .45, "rolling_ic_std": .25, "negative_ic_windows": .30},
    "low_cost": {"fee_drag": .40, "stress_annual": .30, "ls_mdd": .15, "ls_sharpe": .15},
}


def diversified_objectives(metrics):
    """Validation-only scores. Missing required measurements make a profile ineligible."""
    import math
    from factor_optimizer.search.desirability import desirability_for
    ranges = {
        "ls_mdd": (False, 0., .60), "drawdown_duration": (False, 0., 504.),
        "worst_year": (True, -.30, .20), "year_dispersion": (False, 0., .50),
        "ls_sharpe": (True, 0., 3.), "ls_annual": (True, 0., .40),
        "calmar": (True, 0., 3.), "rankic_ir": (True, 0., .80),
        "rolling_ic_std": (False, 0., .10), "negative_ic_windows": (False, 0., .50),
        "fee_drag": (False, 0., .001), "stress_annual": (True, 0., .40),
    }
    finite = lambda x: isinstance(x, (int, float)) and math.isfinite(x)
    common = all(finite(metrics.get(k)) and metrics[k] > 0 for k in ("mean_rankic", "ls_annual"))
    out = {}
    for profile, weights in OBJECTIVE_PROFILES.items():
        missing = [k for k in weights if not finite(metrics.get(k))]
        if not common or missing:
            out[profile] = {"eligible": False, "score": None, "missing": missing,
                            "reason": "missing metrics" if missing else "non-positive validation IC/net return"}
            continue
        components = {}
        for key in weights:
            increasing, lo, hi = ranges[key]
            components[key] = desirability_for("increasing" if increasing else "decreasing",
                [(lo, 0. if increasing else 1.), (hi, 1. if increasing else 0.)], metrics[key])
        out[profile] = {"eligible": True, "policy": "multi-objective-v1", "weights": weights,
                        "components": components, "score": sum(weights[k]*components[k] for k in weights)}
    return out


def objective_comparison(trials):
    champions = {}
    for profile in OBJECTIVE_PROFILES:
        valid = [(n, r) for n, r in trials.items() if r.get("objectives", {}).get(profile, {}).get("eligible")]
        if valid:
            name, trial = max(valid, key=lambda pair: (pair[1]["objectives"][profile]["score"], pair[0] == "raw"))
            champions[profile] = {"variant": name, "score": trial["objectives"][profile]["score"],
                                  "expression": trial["expression"], "status": "validation_candidate"}
    keys = ("ls_annual", "ls_sharpe", "ls_mdd", "fee_drag")
    import math
    valid = {n: r for n, r in trials.items()
             if r.get("objective_metrics", {}).get("mean_rankic", 0) > 0
             and r.get("objective_metrics", {}).get("ls_annual", 0) > 0
             and all(r.get("objective_metrics", {}).get(k) is not None
                     and math.isfinite(r["objective_metrics"][k]) for k in keys)}
    vectors = {n: tuple(r["objective_metrics"][k] * (1 if i < 2 else -1) for i, k in enumerate(keys)) for n, r in valid.items()}
    frontier = [n for n, v in vectors.items() if not any(
        all(a >= b for a,b in zip(other,v)) and any(a > b for a,b in zip(other,v))
        for m,other in vectors.items() if m != n)]
    return {"policy": "multi-objective-v1", "default": "stability", "champions": champions,
            "pareto_frontier": sorted(frontier), "pareto_dimensions": list(keys),
            "scope": "validation only; alternatives require full DSL replay before deployment"}


def optimizer_dsl(base, variant):
    """FE spelling is deliberately not the FP Python API spelling."""
    templates = {"raw": "{base}", "cs_rank": "rank_pct({base})",
                 "cs_zscore": "c_zscore({base})",
                 "winsor_1pct": "winsorize({base}, 0.01, 0.99)",
                 "winsor_5pct": "winsorize({base}, 0.05, 0.95)"}
    if variant not in templates:
        raise ValueError(f"no certified DSL mapping for {variant}")
    expression = templates[variant].format(base=base)
    factor_from_record({"page_name": "optimizer_expression_check", "fe_formula": expression})
    return expression


def lock_optimizer_direction(values, labels, train, *, already_directed=False):
    """Determine direction once BEFORE preprocessing, never per trial."""
    initial = evaluate_report_arrays(values[train], labels[train], min_assets=MIN_UNIVERSE,
                                    fixed_direction=1 if already_directed else None,
                                    commission_rate=COMMISSION_RATE)
    direction = 1 if already_directed else initial.direction
    return values * direction, direction


def stage_optimize_lite(factor, eval_result):
    """Bounded FO search over actual FP transforms, evaluated only by QE."""
    import numpy as np
    import pandas as pd
    import fcntl
    # These libraries use src-style project directories alongside this job;
    # select the current checkout rather than an older installed distribution.
    for library in ("factor_optimizer", "factor_preprocess"):
        sys.path.insert(0, str(ROOT / library))
    from factor_preprocess.registry.transforms import get_default_registry
    from factor_optimizer.search.runner import SearchConfig, SearchRunner
    from factor_optimizer.contracts.search_budget import SearchBudget
    from factor_optimizer.contracts.splits import SplitPlan, EvaluationProtocol
    from factor_optimizer.contracts.trial import Trial, TrialStatus
    from factor_optimizer.contracts.treatment_integrity import build_integrity_evidence

    page = factor["page_name"]
    # Existing full-history matrices are not certified by yearly overlap.
    from factor_engine.ir.analyzer import Analyzer
    parsed = factor_from_record(factor)
    require_bounded_history(Analyzer(production=False).lower(parsed.expr), page)
    mat = load_matrix(page, flip=False)
    if mat is None:
        raise ValueError(f"{page}: verified full-window matrix unavailable")
    vwap = load_vwap(start_date=str(FULL_WINDOW_START.date()),
                     end_date=str(FULL_WINDOW_END.date()))
    mat = mat.reindex(index=vwap.index, columns=vwap.columns)
    labels = vwap.pct_change(fill_method=None).shift(-2).to_numpy(dtype=float)
    dates = pd.DatetimeIndex(vwap.index)
    values = mat.to_numpy(dtype=float)
    values = np.where(np.isfinite(vwap.to_numpy()) & (vwap.to_numpy() > 0), values, np.nan)
    train = np.asarray(dates <= pd.Timestamp(EVAL_END))
    train[np.flatnonzero(train)[-2:]] = False  # purge t+2 label boundary
    values, input_direction = lock_optimizer_direction(
        values, labels, train, already_directed=bool(factor.get("matrix_direction_applied", False)))
    raw_base = _dsl_text(factor)
    directed_base = f"neg({raw_base})" if input_direction < 0 else raw_base
    directed_base = (f"where(is_finite(AdjVwap), where(gt(AdjVwap, 0.0), {directed_base}, "
                     "safe_div_null(0.0, 0.0)), safe_div_null(0.0, 0.0))")
    validation = np.asarray((dates >= "2018-07-01") & (dates <= "2023-12-31"))
    validation[np.flatnonzero(validation)[-2:]] = False
    test = np.asarray(dates >= "2024-01-01")
    split = SplitPlan("weekly-treatment-v1", train.tolist(), validation.tolist(), test.tolist(),
                      {"test_note": "previously explored; not sealed OOS"},
                      time_index=tuple(dates), label_horizon=2)
    registry = get_default_registry()
    recipes = {"raw": (None, {}),
               "cs_rank": ("cs_rank", {"axis": 1, "pct": True}),
               "cs_zscore": ("cs_zscore", {"axis": 1, "ddof": 1}),
               "winsor_1pct": ("cs_winsor", {"axis": 1, "lower": .01, "upper": .99}),
               "winsor_5pct": ("cs_winsor", {"axis": 1, "lower": .05, "upper": .95})}
    results = {}
    def transform(name, data):
        op, params = recipes[name]
        # This report adapter owns wide NumPy panels. get_execution's current
        # FE bridge expects a long DataFrame; use the library's registered
        # native array implementation explicitly, without a silent format fallback.
        return data.copy() if op is None else registry.get_function(op)(data, **params)
    def metrics(result):
        return {"mean_rankic": result.mean_rank_ic, "rankic_ir": result.rank_ic_ir,
                "ls_sharpe": result.sharpe, "ls_annual": result.annualized_return,
                "ls_mdd": result.max_drawdown, "n": result.valid_return_periods}
    # Evaluator receives search arrays only, never test observations.
    search_mask = train | validation
    search_values, search_labels = values[search_mask], labels[search_mask]
    search_train, search_validation = train[search_mask], validation[search_mask]
    def evaluate(trial, fidelity):
        name = trial.trial_id
        treated = transform(name, search_values)
        training = evaluate_report_arrays(treated[search_train], search_labels[search_train],
                                         min_assets=MIN_UNIVERSE, fixed_direction=1,
                                         commission_rate=COMMISSION_RATE)
        valid = evaluate_report_arrays(treated[search_validation], search_labels[search_validation],
                                       min_assets=MIN_UNIVERSE, fixed_direction=1,
                                       commission_rate=COMMISSION_RATE)
        op, params = recipes[name]
        evidence = build_integrity_evidence(name, op or "raw", params,
                                            search_values, treated, expected_parameters=params)
        results[name] = dict(train=metrics(training), validation=metrics(valid),
                             direction=1, parameters=params,
                             transform=op, integrity=evidence.to_dict())
        results[name]["execution_origin"] = "identity" if op is None else "FP_NATIVE_ARRAY"
        results[name]["expression"] = optimizer_dsl(directed_base, name)
        results[name]["direction_policy"] = "train-once-before-preprocessing-v1"
        from quant_evaluator.metrics.probe_portfolio.sharpe import compute_portfolio_metrics
        validation_dates = dates[validation]
        yearly = {}
        for year in sorted(set(validation_dates.year)):
            observed = valid.long_short_returns[validation_dates.year == year]
            observed = observed[np.isfinite(observed)]
            if len(observed) >= 60:
                yearly[str(year)] = compute_portfolio_metrics(observed, periods_per_year=252,
                                                            min_periods=20)["annualized_return"]
        stats = compute_portfolio_metrics(valid.long_short_returns, periods_per_year=252, min_periods=20)
        from quant_evaluator.metrics.risk.drawdown_analysis import compute_drawdown_duration
        duration = compute_drawdown_duration(valid.long_short_returns, min_periods=20)
        ic_windows = pd.Series(valid.rank_ic_series).rolling(60, min_periods=40).mean().dropna()
        stressed = evaluate_report_arrays(treated[search_validation], search_labels[search_validation],
            min_assets=MIN_UNIVERSE, fixed_direction=1, commission_rate=COMMISSION_RATE * 5)
        drag = np.asarray(valid.long_short_returns) - np.asarray(stressed.long_short_returns)
        finite_drag = drag[np.isfinite(drag)]
        objective_metrics = dict(metrics(valid),
            worst_year=min(yearly.values()) if yearly else None,
            year_dispersion=float(np.std(list(yearly.values()), ddof=1)) if len(yearly)>1 else None,
            drawdown_duration=float(duration.get("max_drawdown_duration", np.nan)), calmar=stats.get("calmar"),
            rolling_ic_std=float(ic_windows.std()) if len(ic_windows)>1 else None,
            negative_ic_windows=float((ic_windows < 0).mean()) if len(ic_windows) else None,
            fee_drag=float(finite_drag.mean()) if len(finite_drag) else None,
            stress_annual=stressed.annualized_return)
        objectives = diversified_objectives(objective_metrics)
        objective = objectives["stability"]
        results[name].update(objective=objective, objectives=objectives, objective_metrics=objective_metrics,
                             validation_yearly_returns=yearly)
        return {"evaluation_id": f"{page}:{name}", "score": objective["score"] if objective["eligible"] else -1., "cost": 1.,
                "treatment_integrity_evidence": evidence}
    proposals = iter(Trial(trial_id=name, mutation_id=f"{page}:{name}",
                           status=TrialStatus.PROPOSED) for name in recipes)
    config = SearchConfig(budget=SearchBudget(max_trials=len(recipes),
                          max_evaluations=len(recipes), max_cost_units=len(recipes)),
                          enable_multifidelity=False, plateau_window=len(recipes)+1)
    session = SearchRunner(config, lambda: next(proposals),
                           EvaluationProtocol(split, evaluate)).run(f"{page}:preprocess-v1")
    winner = session.best_trial_id
    if winner is None or not results[winner]["objective"]["eligible"]:
        raise ValueError(f"{page}: optimizer produced no valid winner: {session.to_dict()}")
    direction = results[winner]["direction"]
    selected = transform(winner, values)
    expression = optimizer_dsl(directed_base, winner)
    # Do not publish an FP result with a merely decorative DSL. Execute the
    # complete expression through FE and require replay parity before saving.
    with tempfile.TemporaryDirectory(prefix="optimizer_fe_replay_") as replay_dir:
        land_factor_batch_windowed(
            [dict(page_name=page, fe_formula=expression)],
            backend_name=os.environ.get("FACTOR_REPORT_FE_BACKEND", "auto"),
            output_dir=replay_dir)
        replay = pd.read_parquet(Path(replay_dir) / f"{page}.parquet").reindex(
            index=mat.index, columns=mat.columns).to_numpy(dtype=float)
        if not np.allclose(selected, replay, rtol=1e-4, atol=1e-6, equal_nan=True):
            raise ValueError(f"{page}: optimized FE DSL replay differs from FP treatment; publication blocked")
        selected = replay
    heldout = evaluate_report_arrays(selected[test], labels[test], min_assets=MIN_UNIVERSE,
                                     fixed_direction=direction, commission_rate=COMMISSION_RATE)
    baseline_heldout = evaluate_report_arrays(values[test], labels[test], min_assets=MIN_UNIVERSE,
                                              fixed_direction=results["raw"]["direction"],
                                              commission_rate=COMMISSION_RATE)
    op, params = recipes[winner]
    expression = optimizer_dsl(directed_base, winner)
    # Persist actual winner evaluation; rendering must not evaluate or borrow raw charts.
    full_result = evaluate_report_arrays(selected, labels, min_assets=MIN_UNIVERSE,
                                        fixed_direction=1, commission_rate=COMMISSION_RATE)
    chart_manifest = OPT_DIR / page / "report_manifest.json"
    write_report_manifest(
        [dict(page_name=page, fe_formula=expression)],
        dict(factors={page: full_result}, dates=mat.index, backend_used={"cpu"},
             fallbacks=[], evaluation_provenance={"evaluated_at": report_time(),
             "pipeline_source_sha256": PIPELINE_SOURCE_SHA256}), target=chart_manifest)
    record = dict(evaluation_version=EVALUATION_CONVENTION_VERSION,
                  chart_manifest=str(chart_manifest),
                  chart_manifest_sha256=hashlib.sha256(chart_manifest.read_bytes()).hexdigest(),
                  best=winner, best_mean_rankic=results[winner]["validation"]["mean_rankic"],
                  best_rankic_ir=results[winner]["validation"]["rankic_ir"],
                  steps=[] if op is None else [op], dsl_preproc_ops=[expression],
                  effective_formula=expression, variants=results, is_flipped=input_direction < 0,
                  input_direction=input_direction, matrix_direction_applied=True,
                  direction_policy="train-once-before-preprocessing-v1",
                  dsl_status="full-expression FE replay parity verified",
                  optimizer="factor_optimizer.SearchRunner", preprocess="factor_preprocess.registry",
                  selection_window="2018-07-01..2023-12-31; t+2 purged", direction_window=EVAL_START+".."+EVAL_END,
                  test_note="2024+ previously explored; not sealed OOS", heldout=metrics(heldout),
                  baseline_heldout=metrics(baseline_heldout),
                  objective=results[winner]["objective"], objective_comparison=objective_comparison(results),
                  stress_commission_rate=COMMISSION_RATE * 5, commission_rate=COMMISSION_RATE,
                  excluded_costs=["stamp_duty", "slippage", "borrow_cost"],
                  session=session.to_dict())
    def clean(value):
        if isinstance(value, dict): return {k: clean(v) for k,v in value.items()}
        if isinstance(value, (list, tuple)): return [clean(v) for v in value]
        if isinstance(value, float) and not np.isfinite(value): return None
        return value
    OPT_DIR.mkdir(parents=True, exist_ok=True)
    out = OPT_DIR / f"{page}.parquet"
    temporary = out.with_suffix(".parquet.tmp")
    pd.DataFrame(selected * direction, index=mat.index, columns=mat.columns).to_parquet(temporary)
    os.replace(temporary, out)
    with OPT_META_JSON.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        meta = load_opt_meta() if OPT_META_JSON.exists() else {}
        meta[page] = clean(record)
        temporary = OPT_META_JSON.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(meta, ensure_ascii=False, indent=1, allow_nan=False))
        os.replace(temporary, OPT_META_JSON)
    return dict(variant=winner, path=str(out), direction=direction,
                rank_ic=record["best_mean_rankic"], ic_ir=record["best_rankic_ir"])


# --------------------------------------------------------------------------
# stage: page_inject
# --------------------------------------------------------------------------
def _render_json_tree(node, depth=0):
    """FE JSON 树 → 可读 DSL 文本（用于详情页公式展示）。"""
    if not isinstance(node, dict):
        return str(node)
    kind = node.get("kind")
    if kind == "column":
        return node.get("name", "?")
    if kind == "literal":
        v = node.get("value")
        if isinstance(v, str):
            return v
        if isinstance(v, float) and v == int(v):
            return str(int(v))
        return str(v)
    if kind == "call":
        op = node.get("op", "?")
        args = [_render_json_tree(a) for a in node.get("args", [])]
        infix = {"add": "+", "subtract": "-", "multiply": "*", "divide": "/",
                 "lt": "<", "gt": ">", "leq": "<=", "geq": ">=", "eq": "==",
                 "neq": "!=", "and_": "&&", "or_": "||"}
        if op in infix and len(args) == 2:
            return f"({args[0]} {infix[op]} {args[1]})"
        if op in ("negate", "abs", "log", "sqrt", "sign", "rank", "zscore",
                  "cs_rank", "cs_zscore", "ts_mean", "ts_std", "ts_sum",
                  "ts_delta", "ts_cov", "ts_corr", "ewm_mean", "where",
                  "power", "max", "min", "clip", "delay") and len(args) >= 1:
            return f"{op}(" + ", ".join(args) + ")"
        if len(args) >= 1:
            return f"{op}(" + ", ".join(args) + ")"
        return op
    return str(node)


def _dsl_text(factor):
    """详情页公式展示文本。优先 FE JSON 树渲染，fallback local_formula。"""
    fe = factor.get("fe_formula", "")
    if fe:
        try:
            tree = json.loads(fe)
            txt = _render_json_tree(tree)
            if txt and len(txt) > 5:
                return txt
        except Exception:
            return str(fe)
    # A source shorthand or Python body is not an executable DSL fallback.
    if factor.get("dsl"):
        candidate = dict(factor)
        factor_from_record(candidate)  # parser/binding validation, preserving input
        return candidate.get("fe_formula", "")
    return ""


def _check_banned(text, page=None):
    """检查违禁算法名。page 是因子标识名，若违禁词只作为 page 名出现（title/h1/链接），
    属于因子标识不算违规；正文/描述中出现才违规。
    先剥离 base64 图片数据（图二进制编码可能恰好含词子串），再检查纯文本。"""
    # 去掉 <img src="data:image/...base64..."> 内容
    clean = re.sub(r'data:image/[^"]+', '', text)
    low = clean.lower()
    for w in BANNED:
        if w not in low:
            continue
        idxs = [m.start() for m in re.finditer(re.escape(w), low)]
        if page and w in page.lower():
            # page 名本身含该词：标题/文件名/返回链接等标识性出现可容忍。
            flagged = 0
            main_idx = low.find("<main>")
            for i in idxs:
                if main_idx != -1 and i > main_idx:
                    ctx = clean[max(0, i - 120): i + 200]
                    # 因子统计摘要表里的因子名称行不算
                    if f"<td>因子名称</td><td><code>{page}</code>" in ctx:
                        continue
                    # 因子族行里代表因子 == page 也不算（簇代表是它自己）
                    if f"代表因子: {page}" in ctx:
                        continue
                    if ">该因子为" in ctx:
                        continue
                    flagged += 1
            if flagged:
                return w
        else:
            return w
    return None


def stage_page_inject(factor, eval_result, *, report_result=None, report_dates=None, out_dir=None):
    """为每条新因子生成详情页（复用 render_evoalpha14_pages 的模板+图表函数）。
    图可省略或复用 optimize 图函数。禁止算法名字符串。"""
    page = factor["page_name"]
    output_dir = Path(out_dir) if out_dir is not None else FACTORS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    is_flipped = bool((eval_result or {}).get("is_flipped", factor.get("is_flipped", False)))
    dsl_text = _dsl_text(factor)
    if not str(dsl_text or "").strip():
        raise ValueError("Cannot publish factor metrics without an executable DSL")
    fe_formula_raw = factor.get("fe_formula", "")
    if is_flipped:
        dsl_text = f"neg({dsl_text})" if dsl_text else dsl_text
        fe_formula_raw = f"neg({fe_formula_raw})" if fe_formula_raw else fe_formula_raw
    note = "本周新挖增量因子"

    # 尝试 import 复用 evo14 渲染器
    try:
        import render_evoalpha14_pages as R
        has_tpl = True
    except Exception:
        has_tpl = False

    if not has_tpl:
        # 内联简版：无图单页
        html = _minimal_page(page, dsl_text, fe_formula_raw, is_flipped, eval_result)
        out = output_dir / f"factor_{page}.html"
        out.write_text(evaluation_banner(html, factor.get("evaluation_provenance")), encoding="utf-8")
        return {"page": str(out), "mode": "minimal"}

    try:
        opt_meta = load_opt_meta()
        cluster = load_clusters()
        qg = (cluster.get("quality_gate") or {}).get(page, {})
        cluster_id = (cluster.get("page_to_cluster") or {}).get(page, "—")
        rep_entry = (cluster.get("representatives") or {}).get(cluster_id) or {}
        rep_factor = rep_entry.get("factor", "") if isinstance(rep_entry, dict) else ""
        gate = {"cluster": cluster_id,
                "quality_gate": qg.get("quality_gate", "—"),
                "gate_msg": qg.get("message", ""),
                "rep_factor": rep_factor}

        if report_result is None or report_dates is None:
            raise ValueError("canonical QuantEvaluator result is required for full report rendering")
        import pandas as pd
        dates = pd.DatetimeIndex(report_dates)
        ic_series = pd.Series(report_result.rank_ic_series, index=dates)
        decile_navs = {
            "dates": dates.tolist(),
            **{
                f"G{group + 1}": report_result.quantile_nav[:, group].tolist()
                for group in range(report_result.quantile_nav.shape[1])
            },
            "LS": report_result.long_short_nav_aligned.tolist(),
        }
        base = {
            "cost_note": (f"按单边成交额佣金 {getattr(report_result, 'commission_rate', 0.0) * 10000:g} bp 扣费；"
                          "印花税、滑点及融券成本尚未计入。采用目标权重换手模型，非完整成交仿真。"),
            "report_start": str(dates.min().date()),
            "report_end": str(dates.max().date()),
            "mean_rankic": report_result.mean_rank_ic,
            "std_rankic": report_result.rank_ic_std,
            "rankic_ir": report_result.rank_ic_ir,
            "rankic_winrate": report_result.rank_ic_win_rate,
            "ls_sharpe": report_result.sharpe,
            "ls_annual": report_result.annualized_return,
            "ls_cum": report_result.cumulative_return,
            "ls_mdd": report_result.max_drawdown,
            "ls_winrate": report_result.win_rate,
            "g10_annual": report_result.top_quantile_annualized_return,
            "g1_annual": report_result.bottom_quantile_annualized_return,
            "n_periods": report_result.valid_return_periods,
            "ic": ic_series,
            "decile_navs": decile_navs,
        }
        charts = {
            "svg_ts": R.plot_ic_timeseries_svg(ic_series, page),
            "monthly": R.plot_ic_monthly_heatmap(ic_series, page),
            "decile": R.plot_decile_nav(decile_navs, page),
            "ls": R.plot_long_short_nav(decile_navs, page),
            "dist": R.plot_ic_distribution(ic_series, page),
        }
        # Optimization comparisons have their own canonical batch artifact;
        # never silently recompute them inside the page renderer.
        opt_charts = ""

        dsl_note = "factor_engine DSL（本因子由增量管线落值）"
        req_cols = _required_columns(dsl_text)
        optimization = opt_meta.get(page, {})
        if (optimization.get("evaluation_version") != EVALUATION_CONVENTION_VERSION
                or optimization.get("direction_policy") != "train-once-before-preprocessing-v1"
                or optimization.get("dsl_status") != "full-expression FE replay parity verified"):
            optimization = {}
        if optimization.get("chart_manifest"):
            try:
                opt_path = Path(optimization["chart_manifest"])
                if hashlib.sha256(opt_path.read_bytes()).hexdigest() != optimization.get("chart_manifest_sha256"):
                    raise ValueError("optimized manifest checksum mismatch")
                opt_payload = json.loads(opt_path.read_text())
                opt_entry = opt_payload["factors"][page]
                if opt_entry["raw_formula"] != optimization["effective_formula"]:
                    raise ValueError("optimized formula mismatch")
                opt_result, opt_dates = _report_result_from_artifact(
                    opt_entry, opt_path.parent / opt_entry["artifact"])
                opt_ic = pd.Series(opt_result.rank_ic_series, index=opt_dates)
                opt_navs = dict(dates=opt_dates.tolist(), LS=opt_result.long_short_nav_aligned.tolist(),
                    **{f"G{i+1}": opt_result.quantile_nav[:, i].tolist()
                       for i in range(opt_result.quantile_nav.shape[1])})
                opt_charts = '<h3>最优变体全窗评估图（不是验证窗择优分数）</h3>'
                opt_charts += '<h4>优化前后同图对比</h4>' + R.chart_img_or_notice(
                    R.plot_optimization_comparison(decile_navs, opt_navs, page), "对比数据不足")
                opt_charts += '<p>G1/G10 是各分组满仓买入的参考净值，G1 不是空头账户净值。多空按所有入选股票等金额、总绝对敞口100%配置；两侧人数相等时各占50%。因此多空可以低于满仓G10；不能将两条分组累计净值相减。原始方案获选或排序未变时，前后曲线可能重合。</p>'
                opt_charts += R.plot_ic_timeseries_svg(opt_ic, page + " optimized")
                for title, picture in (
                    ("优化后月度 RankIC", R.plot_ic_monthly_heatmap(opt_ic, page)),
                    ("优化后十分层净值", R.plot_decile_nav(opt_navs, page)),
                    ("优化后多空净值（扣佣金）", R.plot_long_short_nav(opt_navs, page)),
                    ("优化后 RankIC 分布", R.plot_ic_distribution(opt_ic, page))):
                    opt_charts += '<h4>' + title + '</h4>' + R.chart_img_or_notice(picture, "数据不足")
            except Exception as exc:
                opt_charts = '<p>优化图表校验未通过：' + escape(str(exc)) + '</p>'
        elif optimization:
            opt_charts = '<p>该优化记录尚无独立全窗图表产物，待主优化链路重算；不复用原始图冒充优化图。</p>'
        html = R.build_html(page, note, dsl_text, dsl_note, req_cols, base,
                            optimization, gate, {}, charts, opt_charts)

        # 替换回测区间为我们的评估口径说明
        html = html.replace("本周新挖", "本周新挖（增量）")
        html = html.replace("回测区间 2019-01-02 ~ 2026-08-24",
                            f"全窗评估 {FULL_WINDOW_START.date()} ~ {FULL_WINDOW_END.date()}"
                            "（方向仅由 2016-01-04 ~ 2018-06-30 训练窗确定；vwap-to-vwap shift(-2)）")
        out = output_dir / f"factor_{page}.html"
        out.write_text(evaluation_banner(html, factor.get("evaluation_provenance")), encoding="utf-8")

        banned = _check_banned(html, page=page)
        if banned:
            # 违规就降级为 minimal 无图页
            html2 = _minimal_page(page, dsl_text, fe_formula_raw, is_flipped, eval_result)
            out.write_text(evaluation_banner(html2, factor.get("evaluation_provenance")), encoding="utf-8")
            return {"page": str(out), "mode": "minimal", "banned_in_full": banned}
        # Preserve source verbatim (HTML escaped), outside presentation-name filters.
        html = html.replace("</body>", formula_source_html(factor) + "</body>")
        out.write_text(evaluation_banner(html, factor.get("evaluation_provenance")), encoding="utf-8")
        return {"page": str(out), "mode": "full"}
    except Exception as exc:
        print(f"    [page_inject fallback] {type(exc).__name__}: {str(exc)[:120]}", flush=True)
        html = _minimal_page(page, dsl_text, fe_formula_raw, is_flipped, eval_result)
        out = output_dir / f"factor_{page}.html"
        out.write_text(evaluation_banner(html, factor.get("evaluation_provenance")), encoding="utf-8")
        return {"page": str(out), "mode": "minimal"}


def _required_columns(dsl_text):
    import ast
    tree = ast.parse(dsl_text, mode="eval")
    functions = {id(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)}
    used = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and id(node) not in functions:
            used.add(node.id)
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "col" and node.args
                and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)):
            used.add(node.args[0].value)
    return ", ".join(sorted(used))


def _opt_compare_charts(page, raw_mat, opt_mat, is_flipped):
    if opt_mat is None:
        return ""
    try:
        import render_optimized_pages as R2
        import pandas as pd
        vwap = R2.load_vwap()
        charts = ""
        raw_ic = R2.daily_rankic_series(raw_mat, vwap)
        if is_flipped:
            raw_ic = -raw_ic
        opt_ic = R2.daily_rankic_series(opt_mat, vwap)
        try:
            ic = R2.plot_ic_compare(raw_ic, opt_ic, page, is_flipped)
            if ic:
                charts += f'  <img src="data:image/png;base64,{ic}" style="width:100%;border-radius:8px;margin-bottom:8px" alt="RankIC对比"/>\n'
        except Exception:
            pass
        try:
            dec = R2.plot_decile_compare(raw_mat, opt_mat, vwap, page, is_flipped)
            if dec:
                charts += f'  <img src="data:image/png;base64,{dec}" style="width:100%;border-radius:8px;margin-bottom:8px" alt="十分层对比"/>\n'
        except Exception:
            pass
        try:
            ls = R2.plot_ls_compare(raw_mat, opt_mat, vwap, page, is_flipped)
            if ls:
                charts += f'  <img src="data:image/png;base64,{ls}" style="width:100%;border-radius:8px" alt="多空对比"/>\n'
        except Exception:
            pass
        if not charts:
            charts = '  <p style="color:#94a3b8;font-size:0.8rem">（对比图生成失败）</p>\n'
        return charts
    except Exception:
        return ""


def _minimal_page(page, dsl_text, fe_formula_raw, is_flipped, eval_result):
    """降级简版页面（无图，仅指标/公式/来源）。"""
    import html
    import math
    e = eval_result or {}
    def metric(key, precision):
        value = e.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            return "—", ""
        return f"{value:+.{precision}f}", "pos" if value >= 0 else "neg"
    ic, ic_cls = metric("rank_ic", 4)
    ir, ir_cls = metric("ic_ir", 3)
    page = html.escape(str(page))
    dsl_text = html.escape(str(dsl_text or "（无已验证 DSL）"))
    flip_badge = '<span class="badge badge-yellow">⚠ 按训练窗方向翻转</span>' if is_flipped else ""
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>{page}</title>
<style>
:root{{--bg:#eef2f7;--panel:#fff;--fg:#0f172a;--muted:#64748b;--line:#e2e8f0;--primary:#1e4d8c;--pos:#16a34a;--neg:#dc2626}}
*{{box-sizing:border-box}}
body{{margin:0;font-family:"Segoe UI","Microsoft YaHei",system-ui,sans-serif;color:var(--fg);background:var(--bg)}}
header{{background:linear-gradient(135deg,#0f2744,#1e4d8c 60%,#0d9488);color:#fff;padding:28px 48px 22px}}
header h1{{margin:0 0 6px;font-size:1.5rem;word-break:break-all}}
header .meta{{opacity:0.85;font-size:0.85rem;margin-top:4px}}
main{{max-width:1100px;margin:0 auto;padding:24px}}
.back{{display:inline-block;margin-bottom:16px;color:#93c5fd;font-weight:500;text-decoration:none;font-size:0.88rem}}
.grid-4{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:12px}}
.metric{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px;text-align:center;box-shadow:0 4px 24px rgba(15,23,42,0.06)}}
.metric b{{display:block;font-size:1.35rem}}
.metric span{{color:var(--muted);font-size:0.73rem}}
.pos{{color:var(--pos)}}.neg{{color:var(--neg)}}
.card{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:18px 20px;box-shadow:0 4px 24px rgba(15,23,42,0.06);margin-bottom:16px}}
h2{{font-size:0.95rem;color:var(--primary);margin:0 0 12px;border-bottom:1px solid var(--line);padding-bottom:8px}}
.formula-wrap{{background:#f8fafc;border:1px solid var(--line);border-radius:8px;padding:16px;font-family:"Courier New",monospace;font-size:0.85rem;word-break:break-all;line-height:1.8;white-space:pre-wrap}}
.badge{{display:inline-block;padding:2px 8px;border-radius:12px;font-size:0.73rem;margin-left:6px}}
.badge-yellow{{background:#fef3c7;color:#92400e}}
.qe-info{{display:inline-block;background:linear-gradient(90deg,#ede9fe,#dbeafe);color:#5b21b6;padding:2px 10px;border-radius:12px;font-size:0.72rem;font-weight:600;margin-left:8px}}
.meta-table{{width:100%;border-collapse:collapse;font-size:0.84rem}}
.meta-table td{{padding:7px 10px;border-bottom:1px solid var(--line)}}
.meta-table td:first-child{{color:var(--muted);width:160px;font-weight:500}}
</style>
</head>
<body>
<header>
<a class="back" href="../index.html">&#8592; 返回汇总</a>
<h1><code>{page}</code>{flip_badge}<span class="qe-info">⚡ quant_evaluator</span></h1>
<div class="meta">本周新挖（增量）· 评估区间 {EVAL_START} ~ {EVAL_END} · 收益口径 Vwap 后复权 vwap-to-vwap（shift(-2)）</div>
</header>
<main>
<div class="grid-4">
<div class="metric"><b class="{ic_cls}">{ic}</b><span>RankIC</span></div>
<div class="metric"><b class="{ir_cls}">{ir}</b><span>RankIC IR</span></div>
<div class="metric"><b>{html.escape(str(e.get("n_days") or "—"))}</b><span>评估交易日</span></div>
<div class="metric"><b>{"是" if is_flipped else "否"}</b><span>已翻正</span></div>
</div>
<div class="card">
<h2>📐 因子表达式（FactorEngine DSL）</h2>
<div class="formula-wrap">{dsl_text or "（无 DSL 文本，见 source 字段）"}</div>
</div>
<div class="card">
<h2>🧬 优化因子（预处理 + 择优）</h2>
<p style="font-size:0.82rem;color:#64748b;margin:0">本页为降级展示：完整图表生成未完成，优化结果未在本页验证。不能据此认定已完成优化；请通过主链路重新生成并验证报告。</p>
</div>
</main>
</body>
</html>
"""


# --------------------------------------------------------------------------
# stage: json_writeback
# --------------------------------------------------------------------------
def stage_json_writeback(factor, eval_result, cluster_result):
    """重新 Read 池 json → append 新条目 → dump → 校验 can_use 计数。"""
    page = factor["page_name"]
    fname = factor.get("factor_name") or f"factor_{page}"
    fe_formula = factor.get("fe_formula", "")
    is_flipped = bool(eval_result.get("is_flipped", factor.get("is_flipped", False)))
    dsl_text = _dsl_text(factor)
    lqtp_formula = factor.get("local_formula", "") or dsl_text

    pool = load_pool()
    orig_total = len(pool)
    orig_can_use = sum(1 for r in pool if r.get("can_use_factor_engine"))
    if page in {r.get("page_name") for r in pool}:
        return {"skipped": True, "reason": "already in pool"}

    entry = {
        "page_name": page,
        "factor_name": fname,
        "status": "incremental_intake",
        "dsl": dsl_text,
        "lqtp_formula": lqtp_formula,
        "fe_formula": fe_formula,
        "code": factor.get("code", ""),
        "is_flipped": is_flipped,
        "can_use_factor_engine": True,
        "note": "本周新挖增量因子（incremental_intake）",
        "is_unlisted_miner": True,
    }
    entry.update(formula_provenance(factor))
    for key in ("intake_since",):
        if key in factor:
            entry[key] = factor[key]
    pool.append(entry)
    POOL_JSON.write_text(json.dumps(pool, ensure_ascii=False, indent=1))

    new_can_use = sum(1 for r in pool if r.get("can_use_factor_engine"))
    ok = new_can_use == orig_can_use + 1
    return {"added": page, "total": len(pool), "can_use": new_can_use,
            "orig_can_use": orig_can_use, "check_ok": ok,
            "total_check": len(pool) == orig_total + 1}


# --------------------------------------------------------------------------
# 首页：header 计数更新 + 新挖区块追加 26 行
# --------------------------------------------------------------------------
def update_index(new_entries):
    """更新 index.html：
    1) +14 → +40、含本周新挖 14 → 40、470 → 496（总数 470+26）
    2) all-factors 表尾追加 26 行（新因子标「新」badge）
    3) 在 robustness 前插入「本周新挖」区块（列出 26 条）
    """
    html = INDEX_HTML.read_text(encoding="utf-8")
    new_entries = list({str(entry["page_name"]): entry for entry in new_entries}.values())
    # Replace the previous weekly section and matching table rows on retries.
    html = re.sub(r'<section\b[^>]*\bid="new-mining"[^>]*>.*?</section>',
                  '', html, flags=re.DOTALL)
    pages_to_replace = {str(entry["page_name"]) for entry in new_entries}
    def keep_other_row(match):
        row = match.group(0)
        links = re.findall(r'href="factors/factor_([^"<>]+)\.html"', row)
        return '' if pages_to_replace.intersection(links) else row
    html = re.sub(r'<tr\b[^>]*>.*?</tr>', keep_other_row, html, flags=re.DOTALL)
    pool_pages = {str(record.get("page_name")) for record in load_pool() if record.get("page_name")}
    total = len(pool_pages)
    n_new = len({str(entry["page_name"]) for entry in new_entries})

    # 1) header 统计
    html = re.sub(r"<b>\d+</b><span>因子总数</span>",
                  f"<b>{total}</b><span>因子总数</span>", html)
    html = re.sub(r"<b>\+\d+</b><span>本周新挖</span>",
                  f"<b>+{n_new}</b><span>本周新挖</span>", html)
    html = re.sub(r'<h2 id="all-factors">全部 \d+ 个因子</h2>',
                  f'<h2 id="all-factors">全部 {total} 个因子</h2>', html)
    html = re.sub(r"含本周新挖 \d+", f"含本周新挖 {n_new}", html)
    html = re.sub(r"存活 (\d+)/\d+", rf"存活 \1/{total}", html)

    # 2) 新挖因子区块（插入 all-factors 表后，robustness 前）
    rows = []
    for i, en in enumerate(new_entries, start=max(1, total - n_new + 1)):
        page = en["page_name"]
        ic = en.get("rank_ic", 0.0)
        ir = en.get("ic_ir", 0.0)
        flipped = en.get("is_flipped", False)
        flip_tag = '<span class="tag tag-flip">翻正</span>' if flipped else ""
        new_tag = '<span class="tag" style="background:#dcfce7;color:#166534">新</span>'
        ic_cls = "pos" if ic >= 0 else "neg"
        rows.append(
            f'<tr><td class="rank">{i}</td>'
            f'<td><a href="factors/factor_{page}.html"><code>{page}</code></a>{new_tag}{flip_tag}</td>'
            f'<td class="{ic_cls}">{ic:.4f}</td>'
            f'<td>{ir:.3f}</td>'
            + ''.join(f'<td>{_homepage_metric(en.get(key), percent)}</td>' for key, percent in
                    (("ls_sharpe", False), ("ls_annual", True), ("ls_mdd", True),
                     ("ls_winrate", True), ("g10_annual", True))) + '</tr>'
        )
    # 插入到 </tbody></table> 之后第一个 </table> 之前? all-factors 表是第一个 table。
    # 直接在 all-factors 的 </table> 后追加新挖区块
    marker = '</tbody>\n  </table>'
    block = '\n' + '\n'.join(rows) + '\n    </tbody>\n  </table>\n'
    # 替换 all-factors 表的结尾：在其 </tbody></table> 前插入新行
    idx_tbl = html.find('id="all-factors"')
    idx_tbody_end = html.find('</tbody>', idx_tbl)
    if idx_tbl < 0 or idx_tbody_end < 0:
        raise ValueError("homepage is missing the all-factors table; refusing an invalid insertion")
    html = html[:idx_tbody_end] + '\n' + '\n'.join(rows) + '\n' + html[idx_tbody_end:]

    # 3) 「本周新挖」区块（放 robustness 前）
    new_section = _new_mining_section(new_entries)
    anchor = '<section id="robustness-2026">'
    html = html.replace(anchor, new_section + '\n' + anchor, 1)

    INDEX_HTML.write_text(html, encoding="utf-8")
    banned = _check_banned(new_section, page="")
    return {"total": total, "n_new": n_new, "rows_added": len(rows),
            "banned_in_new_section": banned}


def _homepage_metric(value, percent=False):
    import math
    if value is None or not math.isfinite(value):
        return "—"
    return f"{value:.1%}" if percent else f"{value:.2f}"


def _new_mining_section(new_entries):
    rows = ""
    for i, en in enumerate(new_entries, start=1):
        page = en["page_name"]
        ic = en.get("rank_ic", 0.0)
        ir = en.get("ic_ir", 0.0)
        cluster_id = en.get("cluster_id", "—")
        flipped = en.get("is_flipped", False)
        flip_tag = '<span class="tag tag-flip">翻正</span>' if flipped else ""
        new_tag = '<span class="tag" style="background:#dcfce7;color:#166534">新</span>'
        ic_cls = "pos" if ic >= 0 else "neg"
        rows += (
            f'<tr><td class="rank">{i}</td>'
            f'<td><a href="factors/factor_{page}.html"><code>{page}</code></a>{new_tag}{flip_tag}</td>'
            f'<td class="{ic_cls}">{ic:.4f}</td>'
            f'<td>{ir:.3f}</td>'
            f'<td>{cluster_id}</td></tr>'
        )
    return f'''
<section id="new-mining">
<h2>🆕 本周新挖增量因子（{len(new_entries)}）</h2>
<p style="font-size:0.82rem;color:#64748b">本次增量入库 {len(new_entries)} 条，评估口径与既有池一致（vwap-to-vwap shift(-2) 后复权）。</p>
<table>
  <thead><tr><th>#</th><th>因子</th><th>RankIC</th><th>IR</th><th>因子族</th></tr></thead>
  <tbody>{rows}</tbody>
</table>
</section>
'''


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--overnight-formulas", type=Path)
    ap.add_argument("--publish-queue", type=Path, help="Publish completed queue entries and incrementally synchronize the rich homepage")
    ap.add_argument("--retest-pool", action="store_true",
                    help="Force-land the existing report pool in resumable guarded batches; no automatic publication")
    ap.add_argument("--reuse-landed", action="store_true",
                    help="Reuse validated matrices; still land missing/invalid coverage through mainline")
    ap.add_argument("--auto-finalize", action="store_true",
                    help="Publish each evaluated wave, run eligible optimization, then refresh publication")
    ap.add_argument("--optimize-queue", type=Path,
                    help="Resume optimization of passed weekly candidates in this queue directory")
    ap.add_argument("--overnight-dir", type=Path, default=Path("/tmp/weekly_overnight_20260906"))
    ap.add_argument("--worker-memory-gib", type=int, choices=range(1, 49), default=8)
    ap.add_argument("--worker-threads", type=int, choices=range(1, 33), default=2)
    ap.add_argument("--wait-for-queue", action="store_true",
                    help="Wait for the current queue owner, then refresh formulas and resume")
    ap.add_argument("--overnight-retry-name", action="append", default=[],
                    help="Explicit failed/unavailable candidate to retry; repeatable")
    ap.add_argument("--manifest")
    ap.add_argument("--discover-root", type=Path,
                    help="discover new explicit DSL metadata from shared mining outputs")
    ap.add_argument("--since", help="minimum campaign date YYYYMMDD; defaults to this Monday")
    ap.add_argument("--candidate-name", action="append", default=[],
                    help="select a discovered candidate by its source name (repeatable)")
    ap.add_argument("--publish-from-manifest", type=Path,
                    help="render index/detail HTML only from a verified QE report manifest")
    ap.add_argument("--publish-details-only", action="store_true",
                    help="Publish selected detail pages without replacing the full homepage")
    ap.add_argument("--report-dir", type=Path, default=REPORTS_DIR,
                    help="target report directory for --publish-from-manifest")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--fe-backend", default="auto",
                    help="Report auto policy resolves to auto_long (SQL pushdown + Polars long); explicit backends remain supported")
    ap.add_argument("--qe-backend", choices=("auto", "cpu", "cuda", "cuda_strict"), default="auto")
    ap.add_argument("--skip-landing", action="store_true")
    ap.add_argument("--force-landing", action="store_true",
                    help="Recompute admitted DSL matrices, replacing each only after all windows succeed")
    ap.add_argument("--optimize-only", action="store_true",
                    help="Run FO/FP optimization on verified landed matrices; do not publish")
    ap.add_argument("--audit-only", action="store_true",
                    help="Audit executable DSL, history requirements and matrix coverage without computing")
    ap.add_argument("--evaluate-only", action="store_true",
                    help="evaluate and write the manifest, but do not mutate pages/pool/state")
    ap.add_argument("--output-manifest", type=Path, default=REPORT_MANIFEST_JSON)
    args = ap.parse_args()
    if args.publish_queue:
        print(json.dumps(publish_completed_queue(args.publish_queue, args.report_dir), ensure_ascii=False))
        return

    if args.optimize_queue:
        run_optimization_queue(args.optimize_queue)
        return

    if args.overnight_formulas or args.retest_pool:
        run_overnight_queue(args.overnight_formulas, args.overnight_dir,
                            retry_names=args.overnight_retry_name, wait_for_lock=args.wait_for_queue,
                            memory_gib=args.worker_memory_gib, worker_threads=args.worker_threads,
                            batch_size=args.batch_size, fe_backend=args.fe_backend, qe_backend=args.qe_backend,
                            pool_retest=args.retest_pool, reuse_landed=args.reuse_landed,
                            auto_finalize=args.auto_finalize)
        return

    if args.publish_from_manifest is not None:
        result = publish_report_from_manifest(
            args.publish_from_manifest, report_dir=args.report_dir,
            update_homepage=not args.publish_details_only,
        )
        print(f"[publish] {json.dumps(result, ensure_ascii=False)}", flush=True)
        return
    if not args.manifest and not args.discover_root:
        ap.error("--manifest is required unless --publish-from-manifest is used")

    if args.discover_root:
        manifest, rejected = discover_price_dsl_candidates(args.discover_root, load_pool(), since=args.since)
        if args.candidate_name:
            selected = set(args.candidate_name)
            manifest = [r for r in manifest if r["page_name"] in selected]
            if selected.difference(r["page_name"] for r in manifest):
                ap.error("requested candidates are absent, duplicate, or unsupported")
        args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
        (args.output_manifest.parent / "intake_sources.json").write_text(
            json.dumps(dict(candidates=manifest, rejected=rejected), ensure_ascii=False, indent=2))
        print(f"[discover DSL] candidates={len(manifest)} rejected={len(rejected)}", flush=True)
    else:
        manifest = load_manifest(args.manifest)
    if args.limit:
        manifest = manifest[:args.limit]
    if args.candidate_name and not args.discover_root:
        requested = set(args.candidate_name)
        manifest = [r for r in manifest if r.get("page_name") in requested or r.get("factor_name") in requested]
        if not manifest:
            raise ValueError("no requested candidates found")

    if args.optimize_only:
        for factor in manifest:
            print(json.dumps(stage_optimize_lite(factor, None), ensure_ascii=False), flush=True)
        return

    admitted, admission_errors = admit_report_factors(manifest)
    if args.audit_only:
        from dataclasses import asdict
        audit = {}
        for record in manifest:
            name = record["page_name"]
            source = matrix_source(name)
            audit[name] = {"coverage": asdict(source),
                           **formula_provenance(record),
                           "executable_dsl": record.get("fe_formula"),
                           "dsl_history_error": admission_errors.get(name),
                           "eligible": source.is_full_window and name not in admission_errors}
            if len(audit) % 25 == 0:
                print(f"[audit] {len(audit)}/{len(manifest)}", flush=True)
        args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output_manifest.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(audit, ensure_ascii=False, indent=1, default=str))
        os.replace(temporary, args.output_manifest)
        print(f"[audit done] {len(audit)} factors; {args.output_manifest}", flush=True)
        return

    landing = {}
    if not args.skip_landing:
        landing = land_missing_factors(
            admitted,
            batch_size=max(1, args.batch_size),
            backend_name=args.fe_backend,
            force=args.force_landing,
        )
        print(f"[run_many] {json.dumps(landing, ensure_ascii=False)}", flush=True)

    evaluable, landing_errors = evaluation_candidates_after_landing(admitted, landing)
    batch_evaluation = evaluate_factor_batch(
        evaluable,
        batch_size=max(1, args.batch_size),
        backend=args.qe_backend,
    )
    batch_evaluation["unavailable"].update(admission_errors)
    batch_evaluation["unavailable"].update(landing_errors)
    written_manifest = write_report_manifest(
        manifest, batch_evaluation, target=args.output_manifest,
    )
    print(
        f"[quant_evaluator] backends={sorted(batch_evaluation['backend_used'])} "
        f"fallbacks={batch_evaluation['fallbacks']}",
        flush=True,
    )
    if args.evaluate_only:
        print(
            f"[evaluate-only] factors={len(written_manifest['factors'])} "
            f"manifest={args.output_manifest}",
            flush=True,
        )
        return

    state = load_state()
    done = {}
    for factor in manifest:
        factor["evaluation_provenance"] = batch_evaluation.get("evaluation_provenance")
        fname = factor["factor_name"]
        page = factor["page_name"]
        if page in batch_evaluation["unavailable"]:
            # Invalidate previously completed stages before any optimization or publication.
            state.pop(fname, None)
            save_state(state)
            print(f"[blocked] {page}: {batch_evaluation['unavailable'][page]}", flush=True)
            continue
        print(f"\n=== {page} ({fname}) ===", flush=True)
        if fname not in state:
            state[fname] = {"stage": "", "result": {}, "page": page}
        for stage in STAGES:
            if stage_done(state, fname, stage):
                print(f"  [skip] {stage}", flush=True)
                continue
            try:
                if stage == "dedup_check":
                    res = stage_dedup_check(factor)
                elif stage == "landing":
                    res = stage_landing(factor)
                elif stage == "eval":
                    res = report_evaluation_dict(batch_evaluation["factors"][page])
                    res["matrix_path"] = str(matrix_path(page))
                elif stage == "cluster_assign":
                    res = stage_cluster_assign(
                        factor, state[fname]["result"].get("eval", {}))
                elif stage == "optimize_lite":
                    res = stage_optimize_lite(
                        factor, state[fname]["result"].get("eval", {}))
                elif stage == "page_inject":
                    res = stage_page_inject(
                        factor,
                        state[fname]["result"].get("eval", {}),
                        report_result=batch_evaluation["factors"][page],
                        report_dates=batch_evaluation["dates"],
                    )
                elif stage == "json_writeback":
                    res = stage_json_writeback(
                        factor,
                        state[fname]["result"].get("eval", {}),
                        state[fname]["result"].get("cluster_assign", {}))
                else:
                    res = {}
            except Exception:
                print(f"  [FAIL] {stage}\n{traceback.format_exc()}", flush=True)
                save_state(state)
                sys.exit(1)
            state[fname]["result"][stage] = res
            state[fname]["stage"] = stage
            save_state(state)
            msg = json.dumps(res, ensure_ascii=False)
            print(f"  [done] {stage}: {msg[:220]}", flush=True)
        done[page] = {
            "factor_name": fname,
            "eval": state[fname]["result"].get("eval", {}),
            "cluster": state[fname]["result"].get("cluster_assign", {}),
            "optimize": state[fname]["result"].get("optimize_lite", {}),
            "page": state[fname]["result"].get("page_inject", {}),
        }

    # 全部完成后：更新首页
    if not args.limit or len(manifest) >= 26:
        new_entries = []
        for fname, st in state.items():
            if st.get("stage") == "json_writeback" and st.get("page"):
                page = st["page"]
                ev = st["result"].get("eval", {})
                cl = st["result"].get("cluster_assign", {})
                new_entries.append({
                    "page_name": page,
                    "rank_ic": ev.get("rank_ic", 0.0),
                    "ic_ir": ev.get("ic_ir", 0.0),
                    "is_flipped": ev.get("is_flipped", False),
                    "cluster_id": cl.get("assigned", "—"),
                })
        if new_entries:
            upd = update_index(new_entries)
            print(f"\n[index] 首页已更新: {json.dumps(upd, ensure_ascii=False)[:200]}", flush=True)

    DONE_JSON.write_text(json.dumps(done, ensure_ascii=False, indent=1))
    print(f"\nALL DONE -> {DONE_JSON}", flush=True)


if __name__ == "__main__":
    main()
