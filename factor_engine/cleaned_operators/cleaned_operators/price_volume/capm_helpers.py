# -*- coding: utf-8
"""CAPM 类算子：双输入 panel 对齐 + 滚动特质指标。"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from factor_engine.cleaned_operators.price_volume.beta_helpers import align_benchmark_to_ret


def apply_capm_kernel_panel(
    ret: pd.DataFrame,
    benchmark_ret: pd.DataFrame,
    window: int,
    kernel: Callable[..., np.ndarray],
    *,
    min_obs: int = 5,
    **kernel_kwargs,
) -> pd.DataFrame:
    """对 panel 逐列应用 CAPM 类 numpy 核（ret, benchmark, window）。"""
    if benchmark_ret is None:
        raise ValueError(f"{kernel.__name__} 需要 (ret, benchmark_ret, window)")
    w = max(3, int(window))
    bench = align_benchmark_to_ret(ret, benchmark_ret)
    out = pd.DataFrame(index=ret.index, columns=ret.columns, dtype=float)
    for col in ret.columns:
        arr = kernel(ret[col].to_numpy(dtype=float), bench[col].to_numpy(dtype=float), w, **kernel_kwargs)
        out[col] = arr
    return out
