#!/usr/bin/env python3
"""列出仍缺 Polars 的 canonical（按类别分组，供研发排期）。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

FE_ROOT = Path(__file__).resolve().parents[1]

_BUCKETS = (
    ("signal_processing", ("fft", "ifft", "convolve", "correlate", "decimate", "filter_", "wavelet")),
    ("linear_algebra", ("mat_", "eig", "svd", "pca", "qr_decompose", "lu_decompose")),
    ("distribution", ("cdf_", "pdf_", "quantile_normal", "quantile_t")),
    ("random", ("rand_", "shuffle")),
    ("hypothesis", ("test", "ACF", "pacf", "granger", "stationarity")),
    ("other", ()),
)


def _bucket(name: str) -> str:
    """按名称前缀将 canonical 归入研发排期类别。"""
    low = name.lower()
    for label, prefixes in _BUCKETS:
        if any(low.startswith(p.lower()) or p.lower() in low for p in prefixes if p):
            return label
    return "other"


def main() -> int:
    """列出仍缺 Polars backend 的 canonical，按类别分组输出。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    root = str(FE_ROOT.parent)
    if root not in sys.path:
        sys.path.insert(0, root)
    fe = str(FE_ROOT)
    if fe not in sys.path:
        sys.path.insert(0, fe)

    from cleaned_operators import load_all
    from cleaned_operators.registry import OperatorRegistry

    load_all()
    canon = [c for c in OperatorRegistry.list_canonical() if OperatorRegistry.backends_for(c)]
    pandas_only = sorted(
        c
        for c in canon
        if "polars" not in OperatorRegistry.backends_for(c) and "pandas_numpy" in OperatorRegistry.backends_for(c)
    )
    grouped: dict[str, list[str]] = {}
    for name in pandas_only:
        grouped.setdefault(_bucket(name), []).append(name)

    if args.json:
        import json

        print(json.dumps({"count": len(pandas_only), "by_bucket": grouped}, ensure_ascii=False, indent=2))
    else:
        print(f"pandas-only canonical: {len(pandas_only)}")
        for label in sorted(grouped):
            items = grouped[label]
            print(f"\n[{label}] ({len(items)})")
            for item in items:
                print(f"  - {item}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
