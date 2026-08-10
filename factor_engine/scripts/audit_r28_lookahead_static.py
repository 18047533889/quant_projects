#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R28 Phase 3: static lookahead hazard scan over the operator/backend/planner code.

Comment- and docstring-stripped (tokenize-based) scan so comments cannot masquerade
as code.  Every hazard line is emitted with a ``review_status`` (default
``unreviewed``) and ``resolution``; the R28 gate requires
``UNREVIEWED_STATIC_HAZARD == 0``, i.e. every emitted row must eventually carry a
review status.  Output goes to ``docs/evidence/r28/R28_STATIC_LOOKAHEAD_SCAN.{csv,json}``.

Run:  python3 scripts/audit_r28_lookahead_static.py
"""
from __future__ import annotations

import csv
import io
import json
import re
import sys
import tokenize
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "docs" / "evidence" / "r28"

SCAN_DIRS = [
    "cleaned_operators",
    "research_operators",
    "research_tools",
    "backend",
    "planner",
    "mining",
]

#: (pattern-name, severity, regex) — matched against code-only lines.
PATTERNS: list[tuple[str, str, re.Pattern]] = [
    ("shift_negative", "P0", re.compile(r"\.shift\(\s*-\s*\d")),
    ("lead_next_call", "P0", re.compile(r"(?<![A-Za-z0-9_])(Lead|lead|next)\s*\(")),
    ("bfill", "P0", re.compile(r"\.bfill\s*\(|backfill\s*\(")),
    ("fillna_interpolate", "P0", re.compile(r"fillna_interpolate|\.interpolate\s*\(")),
    ("centered_rolling", "P0", re.compile(r"center\s*=\s*True|center\s*:\s*True")),
    ("future_index", "P0", re.compile(r"iloc\[[^\]]*\+\s*1\]|values\[[^\]]*\+\s*1\]|\[[rt]\s*\+\s*1\s*:\s*\]")),
    ("filtfilt", "P0", re.compile(r"filtfilt|sosfiltfilt")),
    ("random_call", "P0", re.compile(r"np\.random\.|numpy\.random\.|default_rng\s*\(|RandomState\s*\(|random\.(rand|randn|randint|uniform|normal|choice|shuffle|sample|seed)\s*\(")),
    ("random_cv", "P0", re.compile(r"KFold\s*\([^)]*shuffle\s*=\s*True|train_test_split\s*\([^)]*shuffle\s*=\s*True|cross_val_score")),
    ("full_sample_fit", "P1", re.compile(r"\.fit_transform\s*\(|StandardScaler\s*\(|RobustScaler\s*\(|MinMaxScaler\s*\(|QuantileTransformer\s*\(|PowerTransformer\s*\(")),
    ("model_fit", "P1", re.compile(r"\.fit\s*\(")),
    ("pca_svd", "P1", re.compile(r"\bPCA\s*\(|TruncatedSVD\s*\(|KernelPCA\s*\(")),
    ("bidirectional_smoother", "P1", re.compile(r"KalmanSmoother|RTS|forward_backward|sosfiltfilt|savgol_filter")),
    ("full_sample_stat", "P2", re.compile(r"np\.(mean|std|var|quantile|median|corrcoef|cov)\s*\(|\.mean\(\s*\)|\.std\(\s*\)")),
]

#: Reviewed hazards (file:line:pattern -> (review_status, resolution)).  Every
#: hazard emitted by the scanner MUST have an entry here; the R28 gate
#: ``UNREVIEWED_STATIC_HAZARD == 0`` is satisfied iff no row is left unreviewed.
#: Each review is the result of inspecting the actual code at HEAD 37008c7f.
REVIEWS: dict[tuple[str, int, str], tuple[str, str]] = {
    # --- lead_next_call (17): every hit is the Python BUILTIN ``next(it, default)``
    #     iterator call, never the forbidden ``next``/``Lead`` factor operator.
    #     ``next`` / ``Lead`` are NOT registered for runtime (verified fresh-enum).
    ("cleaned_operators/_causal.py", 537, "lead_next_call"): ("reviewed", "builtin next(iter) — not the forbidden next operator; no future read"),
    ("cleaned_operators/_causal.py", 540, "lead_next_call"): ("reviewed", "builtin next(iter) — not the forbidden next operator; no future read"),
    ("cleaned_operators/common/_polars_bridge.py", 266, "lead_next_call"): ("reviewed", "builtin next(iter) over time-axis columns — benign"),
    ("cleaned_operators/common/_polars_bridge.py", 308, "lead_next_call"): ("reviewed", "builtin next(iter) over time-axis columns — benign"),
    ("cleaned_operators/common/_polars_bridge.py", 494, "lead_next_call"): ("reviewed", "builtin next(iter) over time-axis columns — benign"),
    ("cleaned_operators/common/scalar_where.py", 57, "lead_next_call"): ("reviewed", "builtin next(iter) over input frames — benign"),
    ("cleaned_operators/contract_hardening.py", 243, "lead_next_call"): ("reviewed", "builtin next(iter) over implementations dict — benign"),
    ("cleaned_operators/fiscal_event_ops.py", 177, "lead_next_call"): ("reviewed", "builtin next(iter) over frames — benign"),
    ("cleaned_operators/fiscal_event_ops.py", 199, "lead_next_call"): ("reviewed", "builtin next(iter) over inputs — benign"),
    ("cleaned_operators/operator_spec.py", 186, "lead_next_call"): ("reviewed", "builtin next(iter) over backends map — benign"),
    ("cleaned_operators/operator_spec.py", 470, "lead_next_call"): ("reviewed", "builtin next(iter) over backends map — benign"),
    ("cleaned_operators/rolling_pack.py", 104, "lead_next_call"): ("reviewed", "builtin next(iter) over columns — benign"),
    ("cleaned_operators/semantic_audit.py", 661, "lead_next_call"): ("reviewed", "builtin next(iter) over names — benign"),
    ("research_operators/__init__.py", 21, "lead_next_call"): ("reviewed", "builtin next(iter) over implementations — benign"),
    ("backend/composite_evidence.py", 110, "lead_next_call"): ("reviewed", "builtin next(iter) over backends — benign"),
    ("backend/polars_expr_emitter.py", 1087, "lead_next_call"): ("reviewed", "builtin next(iter) over cols — benign"),
    ("backend/sql_pushdown/emitter.py", 7366, "lead_next_call"): ("reviewed", "builtin next(iter) over plans — benign"),
    # --- random_call (19): all are FIXED-SEED ``np.random.default_rng(seed)``
    #     surrogates inside research/audit/diagnostic statistics (taskbook §九B).
    #     Deterministic, reproducible, independent of global RNG state; the
    #     seeded operator (ts_transfer_entropy_peak_excess / ts_best_lag_corr_excess)
    #     is extended-surface and NOT production-certified (PENDING) -> not a
    #     production random factor terminal (R28_RANDOM_FACTOR_TERMINALS_ZERO holds).
    ("cleaned_operators/advanced_information.py", 761, "random_call"): ("reviewed", "fixed-seed default_rng surrogate null for TE peak-excess significance; deterministic; extended/PENDING"),
    ("cleaned_operators/advanced_structure.py", 385, "random_call"): ("reviewed", "fixed-seed default_rng(_SEED); deterministic internal fixture"),
    ("cleaned_operators/closure_audit.py", 492, "random_call"): ("reviewed", "fixed-seed default_rng(_stable_seed()); audit fixture"),
    ("cleaned_operators/closure_audit.py", 740, "random_call"): ("reviewed", "fixed-seed default_rng(_stable_seed()); audit fixture"),
    ("cleaned_operators/closure_audit.py", 823, "random_call"): ("reviewed", "fixed-seed default_rng(_stable_seed()); audit fixture"),
    ("cleaned_operators/common/polars_robust_stats.py", 387, "random_call"): ("reviewed", "fixed-seed default_rng(seed+row) block surrogate for ts_best_lag_corr_excess; deterministic; extended/PENDING"),
    ("cleaned_operators/downside_risk.py", 399, "random_call"): ("reviewed", "fixed-seed default_rng(seed+row) block surrogate for ts_best_lag_corr_excess; deterministic; extended/PENDING"),
    ("cleaned_operators/math_certificate.py", 1383, "random_call"): ("reviewed", "fixed-seed default_rng(seed); hostile fixture generator (offline)"),
    ("cleaned_operators/operator_audits.py", 53, "random_call"): ("reviewed", "fixed-seed default_rng(seed); audit fixture"),
    ("cleaned_operators/operator_audits.py", 464, "random_call"): ("reviewed", "fixed-seed default_rng(1).permutation; column-permutation oracle (offline)"),
    ("cleaned_operators/research_spectral.py", 251, "random_call"): ("reviewed", "fixed-seed default_rng(seed) phase surrogate; research-only spectral"),
    ("cleaned_operators/search/factor_dedup.py", 56, "random_call"): ("reviewed", "fixed-seed default_rng(seed); dedup probe fixture"),
    ("cleaned_operators/search/factor_dedup.py", 90, "random_call"): ("reviewed", "fixed-seed default_rng(_PROBE_SEED); dedup probe fixture"),
    ("cleaned_operators/search/factor_dedup.py", 170, "random_call"): ("reviewed", "fixed-seed default_rng(seed); dedup probe fixture"),
    ("cleaned_operators/semantic_audit.py", 147, "random_call"): ("reviewed", "fixed-seed default_rng(seed); audit fixture"),
    ("cleaned_operators/semantic_audit.py", 157, "random_call"): ("reviewed", "fixed-seed default_rng(seed); audit fixture"),
    ("cleaned_operators/semantic_audit.py", 615, "random_call"): ("reviewed", "fixed-seed default_rng(42); column-permutation audit"),
    ("cleaned_operators/semantic_audit.py", 948, "random_call"): ("reviewed", "fixed-seed default_rng(31); PSD-geometry audit"),
    ("cleaned_operators/semantic_audit.py", 1016, "random_call"): ("reviewed", "fixed-seed default_rng(17); golden-hill audit"),
    # --- bidirectional_smoother (4)
    ("cleaned_operators/composition.py", 120, "bidirectional_smoother"): ("reviewed", "false positive: '_MISCLASSIFIED_PARTS' contains substring 'ARTS'"),
    ("cleaned_operators/composition.py", 184, "bidirectional_smoother"): ("reviewed", "false positive: '_MISCLASSIFIED_PARTS' contains substring 'ARTS'"),
    ("cleaned_operators/hvg_ext.py", 364, "bidirectional_smoother"): ("reviewed", "ts_hvg_forward_backward_asymmetry = trailing-window HVG graph-asymmetry statistic; 'forward/backward' refers to the graph measure, computed causally on window [t-w+1,t]; NOT a time smoother"),
    ("cleaned_operators/hvg_ext.py", 427, "bidirectional_smoother"): ("reviewed", "same trailing-window HVG asymmetry canonical registration; causal"),
}


def code_only_lines(path: Path) -> list[tuple[int, str]]:
    """Return (lineno, code-text) pairs with comments/strings/docstrings stripped."""
    try:
        with path.open("rb") as fh:
            toks = list(tokenize.tokenize(fh.readline))
    except (tokenize.TokenError, SyntaxError, IndentationError):
        # Fallback: naive line-based comment strip for non-parseable files.
        out = []
        for i, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            code = re.sub(r"#.*$", "", line).strip()
            if code:
                out.append((i, code))
        return out
    out: list[tuple[int, str]] = []
    current_lineno = 0
    buf: list[str] = []
    for tok in toks:
        if tok.type == tokenize.ENCODING or tok.type == tokenize.ENDMARKER:
            continue
        if tok.type == tokenize.COMMENT or tok.type == tokenize.STRING:
            continue
        if tok.type == tokenize.NEWLINE:
            line = " ".join(buf).strip()
            if line:
                out.append((current_lineno, line))
            buf = []
            continue
        if tok.type in (tokenize.NL, tokenize.INDENT, tokenize.DEDENT):
            continue
        if buf and tok.start[0] != current_lineno:
            line = " ".join(buf).strip()
            if line:
                out.append((current_lineno, line))
            buf = []
        current_lineno = tok.start[0]
        buf.append(tok.string)
    if buf:
        line = " ".join(buf).strip()
        if line:
            out.append((current_lineno, line))
    return out


def map_symbol(path: Path, lineno: int) -> str:
    """Best-effort enclosing function/class name for a line."""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return ""
    sym = ""
    for i in range(min(lineno, len(lines)) - 1, -1, -1):
        m = re.match(r"\s*(?:async\s+)?def\s+(\w+)", lines[i])
        if m:
            return sym + ("::" + m.group(1) if sym else m.group(1))
        m = re.match(r"\s*class\s+(\w+)", lines[i])
        if m:
            sym = m.group(1)
    return sym


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for rel_dir in SCAN_DIRS:
        base = REPO / rel_dir
        if not base.exists():
            continue
        for py in sorted(base.rglob("*.py")):
            if "__pycache__" in py.parts or "build/" in py.as_posix():
                continue
            rel = py.relative_to(REPO).as_posix()
            for lineno, code in code_only_lines(py):
                for name, severity, pat in PATTERNS:
                    if pat.search(code):
                        rows.append(
                            {
                                "file": rel,
                                "line": lineno,
                                "symbol": map_symbol(py, lineno),
                                "pattern": name,
                                "severity": severity,
                                "code": code[:160],
                                "canonicals": "",
                                "review_status": "unreviewed",
                                "resolution": "",
                            }
                        )
    # dedupe identical (file,line,pattern)
    seen = set()
    deduped = []
    for r in rows:
        key = (r["file"], r["line"], r["pattern"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)
    rows = deduped

    # Apply the review table: every hazard must be reviewed (gate:
    # UNREVIEWED_STATIC_HAZARD == 0).  Unreviewed rows are reported loudly.
    unreviewed_keys = []
    for r in rows:
        key = (r["file"], r["line"], r["pattern"])
        review = REVIEWS.get(key)
        if review is None:
            r["review_status"] = "unreviewed"
            unreviewed_keys.append(key)
            continue
        r["review_status"], r["resolution"] = review
    if unreviewed_keys:
        print("!! UNREVIEWED STATIC HAZARDS (must be reviewed before acceptance):")
        for k in unreviewed_keys:
            print(f"   {k[0]}:{k[1]} [{k[2]}]")
        print(f"   count={len(unreviewed_keys)}")

    json_path = OUT / "R28_STATIC_LOOKAHEAD_SCAN.json"
    json_path.write_text(
        json.dumps(
            {
                "schema_version": "factor_engine.r28.static_lookahead.v1",
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "total_hazards": len(rows),
                "unreviewed": sum(1 for r in rows if r["review_status"] == "unreviewed"),
                "hazards": rows,
            },
            indent=1,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    csv_path = OUT / "R28_STATIC_LOOKAHEAD_SCAN.csv"
    fieldnames = ["file", "line", "symbol", "pattern", "severity", "code", "canonicals", "review_status", "resolution"]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    from collections import Counter
    counts = Counter(r["pattern"] for r in rows)
    sev = Counter(r["severity"] for r in rows)
    n_reviewed = sum(1 for r in rows if r["review_status"] == "reviewed")
    n_unreviewed = len(rows) - n_reviewed
    print(f"R28 static lookahead scan: {len(rows)} hazards")
    print(f"  severity: {dict(sev)}")
    print(f"  by pattern: {dict(counts)}")
    print(f"  reviewed={n_reviewed} unreviewed={n_unreviewed}")
    if n_unreviewed:
        print("  !! UNREVIEWED_STATIC_HAZARD != 0")
        sys.exit(1)
    print("  UNREVIEWED_STATIC_HAZARD == 0  (all hazards reviewed)")
    sys.exit(0)


if __name__ == "__main__":
    main()
