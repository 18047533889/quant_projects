#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FactorEngine 全库源码静态审计（Master Prompt §22 Static Audit）。

对 ``cleaned_operators`` 全目录运行 AST / 文本检查。命中不是错误本身，而是
"必须人工/规则分类"的高风险模式。最终报告输出到 ``build/static_audit/``。

分类（22.1..22.8）：
  22.1 参数静默变形       int( / max(*int( / min(*int( / np.clip( / .clip(
  22.2 time-axis 压缩     dropna( / [np.isfinite( / valid_values( / aligned_pairs(
  22.3 silent reindex     .reindex( / pd.concat( / set_axis(
  22.4 truthiness         != 0 / bool( / astype(bool) / .ne(0)
  22.5 hidden parameter   (inspect.signature 对比 - 单独脚本, 这里只扫 kwargs)
  22.6 EPS               1e-12 / EPS / + EPS / np.finfo / 1.4826 / 1.345
  22.7 hidden constant   0.05 0.1 0.3 0.5 0.8 20 60 120 252 512 200
  22.8 future-dependent  whole-panel .any()/.all()/.min()/.max() 先于逐 t 计算

R16-045: 每次命中携带 stable finding ID + severity + owning canonical + 处置
状态（disposition）。未处置的 HIGH / UNKNOWN 命中使 ``main()`` 以非零退出——
高危命中不再只是打印。
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DIR = ROOT / "cleaned_operators"
OUT_DIR = ROOT / "build" / "static_audit"
DISPOSITIONS_PATH = OUT_DIR / "dispositions.json"

# --------------------------------------------------------------------------
# 22.1 参数静默变形
# --------------------------------------------------------------------------
RE_INT_CAST = re.compile(r"\bint\(")
RE_MAX_INT = re.compile(r"\bmax\([^\n]{0,120}?\bint\(")
RE_MIN_INT = re.compile(r"\bmin\([^\n]{0,120}?\bint\(")
RE_NP_CLIP = re.compile(r"\bnp\.clip\(|\.clip\(")
RE_ASINT_CAST = re.compile(r"\.astype\(int\)|\.astype\(\"int")

# 22.2 time-axis 压缩
RE_DROPNA = re.compile(r"\bdropna\(")
RE_ISFINITE_COMPRESS = re.compile(r"\[\s*np\.isfinite\(")
RE_VALID_VALUES = re.compile(r"\bvalid_values\(")
RE_ALIGNED_PAIRS = re.compile(r"\baligned_pairs\(")
RE_FINITE_COMPRESS = re.compile(r"=\s*[a-z_]+\[\s*np\.isfinite\(")
# 压缩后紧跟 lag/fft/embedding 的高危提示
RE_COMPRESS_THEN_LAG = re.compile(r"np\.isfinite\([^)]*\)\].{0,200}?\b(shift|lag|fft|embed|np\.roll)")

# 22.3 silent reindex
RE_REINDEX = re.compile(r"\.reindex\(")
RE_PD_CONCAT = re.compile(r"\bpd\.concat\(")
RE_SET_AXIS = re.compile(r"\.set_axis\(")

# 22.4 truthiness
RE_NEQ0 = re.compile(r"!=\s*0")
RE_BOOL = re.compile(r"\bbool\(")
RE_ASTYPE_BOOL = re.compile(r"\.astype\(\s*bool|\.astype\(\s*np\.bool_|\.astype\(bool")
RE_NE0 = re.compile(r"\.ne\(\s*0\s*\)")

# 22.6 EPS
RE_EPS = re.compile(r"1e-12|1e-10|\bEPS\b|np\.finfo|\b1\.4826\b|\b1\.345\b|\+ *EPS\b")

# 22.7 hidden constants (经济阈值 / horizon / estimator / numerical / support)
RE_HIDDEN_CONSTS = re.compile(
    r"(?<![a-zA-Z_])(0\.05|0\.1|0\.2|0\.3|0\.5|0\.8|1\.345|1\.4826|20|60|120|252|512|200)(?![a-zA-Z_])"
)

# 22.8 whole-panel future-dependent validation
RE_ANY_ALL = re.compile(r"\.(any|all|min|max)\(\s*(axis|\)|$)")

# R16-045: category -> severity (undisposed HIGH/UNKNOWN block release).
_CATEGORY_SEVERITY: dict[str, str] = {
    "22.1_int_cast": "HIGH",
    "22.1_max_int": "HIGH",
    "22.1_min_int": "HIGH",
    "22.1_clip": "HIGH",
    "22.1_astype_int": "HIGH",
    "22.2_compress_then_lag": "HIGH",
    "22.2_dropna": "MEDIUM",
    "22.2_isfinite_compress": "MEDIUM",
    "22.2_valid_values": "MEDIUM",
    "22.2_aligned_pairs": "MEDIUM",
    "22.3_reindex": "MEDIUM",
    "22.3_pd_concat": "MEDIUM",
    "22.3_set_axis": "MEDIUM",
    "22.4_neq0": "HIGH",
    "22.4_bool": "HIGH",
    "22.4_astype_bool": "HIGH",
    "22.4_ne0": "HIGH",
    "22.6_eps": "MEDIUM",
    "22.7_hidden_const": "LOW",
    "22.8_whole_panel_reduce": "UNKNOWN",
}

# Category checks as (regex, category) applied line-by-line, grouped by pass.
_LINE_CHECKS: list[tuple[re.Pattern, str]] = [
    (RE_INT_CAST, "22.1_int_cast"),
    (RE_MAX_INT, "22.1_max_int"),
    (RE_MIN_INT, "22.1_min_int"),
    (RE_NP_CLIP, "22.1_clip"),
    (RE_ASINT_CAST, "22.1_astype_int"),
    (RE_DROPNA, "22.2_dropna"),
    (RE_ISFINITE_COMPRESS, "22.2_isfinite_compress"),
    (RE_VALID_VALUES, "22.2_valid_values"),
    (RE_ALIGNED_PAIRS, "22.2_aligned_pairs"),
    (RE_COMPRESS_THEN_LAG, "22.2_compress_then_lag"),
    (RE_REINDEX, "22.3_reindex"),
    (RE_PD_CONCAT, "22.3_pd_concat"),
    (RE_SET_AXIS, "22.3_set_axis"),
    (RE_NEQ0, "22.4_neq0"),
    (RE_BOOL, "22.4_bool"),
    (RE_ASTYPE_BOOL, "22.4_astype_bool"),
    (RE_NE0, "22.4_ne0"),
    (RE_EPS, "22.6_eps"),
    (RE_HIDDEN_CONSTS, "22.7_hidden_const"),
    (RE_ANY_ALL, "22.8_whole_panel_reduce"),
]


@dataclass
class Hit:
    finding_id: str          # R16-045: stable ID, e.g. "SA-22.1_int_cast-<file>-<line>"
    category: str
    severity: str            # HIGH / MEDIUM / LOW / UNKNOWN
    file: str                # repo-relative path
    line: int
    owning_canonical: str    # R16-045: canonical whose source block contains the line
    code: str                # the matched line, stripped
    pattern: str = ""


@dataclass
class ScanResult:
    hits: list[Hit] = field(default_factory=list)


def _finding_id(category: str, path: Path, lineno: int) -> str:
    return f"SA-{category}-{path.stem}-{lineno}"


def _severity(category: str) -> str:
    return _CATEGORY_SEVERITY.get(category, "UNKNOWN")


def _canonical_line_ranges(path: Path) -> dict[int, str]:
    """AST: line -> owning canonical.  Non-operator lines map to '<module>'."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError, OSError):
        return {}
    ranges: dict[int, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            continue
        for dec in node.decorator_list:
            if not isinstance(dec, ast.Call):
                continue
            kw = {k.arg: k.value for k in getattr(dec, "keywords", [])}
            canon_node = kw.get("canonical")
            if canon_node is not None and isinstance(canon_node, ast.Constant):
                canonical = str(canon_node.value)
                for ln in range(node.lineno, node.end_lineno + 1):
                    ranges.setdefault(ln, canonical)
                break
    return ranges


def _repo_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _is_generated(name: str) -> bool:
    return name.startswith("__pycache__") or name.endswith(".pyc") or name.startswith(".")


def scan_file(path: Path, res: ScanResult) -> None:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (UnicodeDecodeError, OSError):
        return
    canonical_ranges = _canonical_line_ranges(path)
    for i, line in enumerate(lines, start=1):
        owning = canonical_ranges.get(i, "<module>")
        for regex, category in _LINE_CHECKS:
            if regex.search(line):
                res.hits.append(
                    Hit(
                        finding_id=_finding_id(category, path, i),
                        category=category,
                        severity=_severity(category),
                        file=_repo_path(path),
                        line=i,
                        owning_canonical=owning,
                        code=line.strip()[:200],
                    )
                )


def scan_tree(root: Path) -> ScanResult:
    res = ScanResult()
    if root.is_file():
        if root.suffix == ".py":
            scan_file(root, res)
        return res
    for path in sorted(root.rglob("*.py")):
        if any(_is_generated(p) for p in path.parts):
            continue
        scan_file(path, res)
    return res


def _load_dispositions() -> dict[str, dict[str, str]]:
    if DISPOSITIONS_PATH.exists():
        try:
            return json.loads(DISPOSITIONS_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def write_report(res: ScanResult, dispositions: dict[str, dict[str, str]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    def disposition_of(h: Hit) -> str:
        rec = dispositions.get(h.finding_id)
        return (rec or {}).get("disposition", "undisposed")

    csv_path = OUT_DIR / "static_audit_hits.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["finding_id", "severity", "category", "file", "line",
                         "owning_canonical", "disposition", "code"])
        for hit in sorted(res.hits, key=lambda h: (h.severity, h.category, h.file, h.line)):
            writer.writerow([hit.finding_id, hit.severity, hit.category, hit.file,
                             hit.line, hit.owning_canonical, disposition_of(hit), hit.code])

    counts = Counter(h.category for h in res.hits)
    sev_counts = Counter(h.severity for h in res.hits)
    undisposed = [h for h in res.hits
                  if h.severity in ("HIGH", "UNKNOWN") and disposition_of(h) == "undisposed"]
    md_path = OUT_DIR / "static_audit_report.md"
    lines = [
        "# FactorEngine Static Audit (§22 + R16-045) Report",
        "",
        f"- scanned: `cleaned_operators/**/*.py`",
        f"- total hits: {len(res.hits)}",
        f"- severity: {dict(sev_counts)}",
        f"- undisposed HIGH/UNKNOWN (release blockers): {len(undisposed)}",
        "",
        "## 22.1 参数静默变形 (int() / max(int()) / clip)",
        "",
        "| pattern | hits |",
        "|---|---|",
    ]
    for cat in sorted(counts):
        lines.append(f"| {cat} | {counts[cat]} |")
    lines += [
        "",
        "## 判定规则",
        "",
        "- `int(x)` on a scalar param value = 非法参数修正 (FORBIDDEN unless proven output clipping).",
        "- `dropna`/`[np.isfinite(...)]` followed by lag/FFT/embedding = time-axis reconnection (HIGH RISK).",
        "- `.reindex`/`pd.concat`/`set_axis` in multi-panel math = must prove BroadcastSpec, else SameAxis fail-closed.",
        "- `!= 0`/`bool(...)`/`.ne(0)` on condition/event/member/state input = strict semantic validator required.",
        "- `EPS`/`1e-12` used to make an undefined statistic finite = FORBIDDEN.",
        "- whole-panel `.any()/.all()/.min()/.max()` before per-t compute = future-dependent control flow.",
        "",
        "## R16-045 undisposed HIGH / UNKNOWN findings (must be disposed before release)",
        "",
    ]
    if undisposed:
        for h in sorted(undisposed, key=lambda h: (h.file, h.line)):
            lines.append(
                f"- `{h.finding_id}` [{h.severity}] `{h.file}:{h.line}` "
                f"(canonical={h.owning_canonical}) — {h.code}"
            )
    else:
        lines.append("- none ✓")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"static audit hits: {len(res.hits)} (severity {dict(sev_counts)})")
    print(f"  csv: {csv_path}")
    print(f"  md : {md_path}")
    print(f"  undisposed HIGH/UNKNOWN: {len(undisposed)}")
    for cat in sorted(counts):
        print(f"    {cat}: {counts[cat]}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default=str(DEFAULT_DIR))
    parser.add_argument("--no-strict", action="store_true",
                        help="R16-045: exit 0 even with undisposed HIGH/UNKNOWN hits")
    args = parser.parse_args(argv)
    res = scan_tree(Path(args.dir))
    dispositions = _load_dispositions()
    write_report(res, dispositions)

    # R16-045: undisposed HIGH / UNKNOWN findings are CI blockers.
    if not args.no_strict:
        undisposed = [
            h for h in res.hits
            if h.severity in ("HIGH", "UNKNOWN")
            and (dispositions.get(h.finding_id) or {}).get("disposition", "undisposed") == "undisposed"
        ]
        if undisposed:
            print(
                f"\nSTRICT FAIL: {len(undisposed)} undisposed HIGH/UNKNOWN static-audit "
                f"finding(s); dispose them in {DISPOSITIONS_PATH} or --no-strict",
                file=sys.stderr,
            )
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
