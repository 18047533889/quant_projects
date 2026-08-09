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

用法:
  python3 tools/audit_operator_source.py            # 全目录扫描, 输出 build/static_audit/
  python3 tools/audit_operator_source.py --dir P    # 指定目录
"""
from __future__ import annotations

import ast
import csv
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DIR = ROOT / "cleaned_operators"
OUT_DIR = ROOT / "build" / "static_audit"

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


@dataclass
class Hit:
    category: str          # e.g. "22.1_int_cast"
    file: str              # repo-relative path
    line: int
    code: str              # the matched line, stripped
    pattern: str = ""


@dataclass
class ScanResult:
    hits: list[Hit] = field(default_factory=list)

    def add(self, category: str, path: Path, lineno: int, code: str, pattern: str = ""):
        self.hits.append(
            Hit(
                category=category,
                file=str(path.relative_to(ROOT)),
                line=lineno,
                code=code.strip()[:200],
                pattern=pattern,
            )
        )


def _is_generated(name: str) -> bool:
    return name.startswith("__pycache__") or name.endswith(".pyc") or name.startswith(".")


def scan_file(path: Path, res: ScanResult) -> None:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (UnicodeDecodeError, OSError):
        return

    # 22.1
    for i, line in enumerate(lines, start=1):
        if RE_INT_CAST.search(line):
            res.add("22.1_int_cast", path, i, line)
        if RE_MAX_INT.search(line):
            res.add("22.1_max_int", path, i, line)
        if RE_MIN_INT.search(line):
            res.add("22.1_min_int", path, i, line)
        if RE_NP_CLIP.search(line):
            res.add("22.1_clip", path, i, line)
        if RE_ASINT_CAST.search(line):
            res.add("22.1_astype_int", path, i, line)
    # 22.2
    for i, line in enumerate(lines, start=1):
        if RE_DROPNA.search(line):
            res.add("22.2_dropna", path, i, line)
        if RE_ISFINITE_COMPRESS.search(line) or RE_FINITE_COMPRESS.search(line):
            res.add("22.2_isfinite_compress", path, i, line)
        if RE_VALID_VALUES.search(line):
            res.add("22.2_valid_values", path, i, line)
        if RE_ALIGNED_PAIRS.search(line):
            res.add("22.2_aligned_pairs", path, i, line)
        if RE_COMPRESS_THEN_LAG.search(line):
            res.add("22.2_compress_then_lag", path, i, line)
    # 22.3
    for i, line in enumerate(lines, start=1):
        if RE_REINDEX.search(line):
            res.add("22.3_reindex", path, i, line)
        if RE_PD_CONCAT.search(line):
            res.add("22.3_pd_concat", path, i, line)
        if RE_SET_AXIS.search(line):
            res.add("22.3_set_axis", path, i, line)
    # 22.4
    for i, line in enumerate(lines, start=1):
        if RE_NEQ0.search(line):
            res.add("22.4_neq0", path, i, line)
        if RE_BOOL.search(line):
            res.add("22.4_bool", path, i, line)
        if RE_ASTYPE_BOOL.search(line):
            res.add("22.4_astype_bool", path, i, line)
        if RE_NE0.search(line):
            res.add("22.4_ne0", path, i, line)
    # 22.6
    for i, line in enumerate(lines, start=1):
        if RE_EPS.search(line):
            res.add("22.6_eps", path, i, line)
    # 22.7
    for i, line in enumerate(lines, start=1):
        if RE_HIDDEN_CONSTS.search(line):
            res.add("22.7_hidden_const", path, i, line)
    # 22.8 (整面板 any/all/min/max 校验 —— 保守: 只标记 .any()/.all() 无 axis)
    for i, line in enumerate(lines, start=1):
        if re.search(r"\.(any|all)\(\s*\)", line) or re.search(r"\.(min|max)\(\s*\)", line):
            res.add("22.8_whole_panel_reduce", path, i, line)


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


def write_report(res: ScanResult) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # CSV 全量
    csv_path = OUT_DIR / "static_audit_hits.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["category", "file", "line", "code", "pattern"])
        for hit in sorted(res.hits, key=lambda h: (h.category, h.file, h.line)):
            writer.writerow([hit.category, hit.file, hit.line, hit.code, hit.pattern])

    # 按类别统计
    from collections import Counter

    counts = Counter(h.category for h in res.hits)
    md_path = OUT_DIR / "static_audit_report.md"
    lines = [
        "# FactorEngine Static Audit (§22) Report",
        "",
        f"- scanned: `cleaned_operators/**/*.py`",
        f"- total hits: {len(res.hits)}",
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
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"static audit hits: {len(res.hits)}")
    print(f"  csv: {csv_path}")
    print(f"  md : {md_path}")
    for cat in sorted(counts):
        print(f"  {cat}: {counts[cat]}")


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default=str(DEFAULT_DIR))
    args = parser.parse_args()
    res = scan_tree(Path(args.dir))
    write_report(res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
