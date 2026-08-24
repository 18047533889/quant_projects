# -*- coding: utf-8 -*-
"""Backend-independent production and LQTP compatibility tiers."""
from __future__ import annotations

import collections.abc

from factor_engine.cleaned_operators.operator_surface import extended_only_canonicals


def pandas_first_production_canonicals() -> frozenset[str]:
    """Derive the pandas-first production candidate set from the CURRENT
    extended-only surface (review #309).

    The previous module-level ``frozenset(EXTENDED_ONLY_CANONICALS)`` was an
    import-time snapshot: any operator registered onto the live extended surface
    (via ``extend_extended_only``) after ``production_tiers`` was imported never
    appeared in it.  This function always reads the live surface, so it is
    correct both during ``load_all`` (after every operator module has run its
    ``extend_extended_only``) and after registry freeze.
    """
    return frozenset(extended_only_canonicals())


class _LiveExtendedOnlyView(collections.abc.Set):
    """Backward-compatible live view for ``PANDAS_FIRST_PRODUCTION_CANONICALS``.

    Consumers that import the old module-level name keep working unmodified: the
    view is a read-only ``collections.abc.Set`` whose membership / iteration /
    length always reflect the CURRENT extended-only surface at call time, so a
    late ``extend_extended_only`` mutation is visible without any consumer edit.
    New code should prefer :func:`pandas_first_production_canonicals`.
    """

    def __contains__(self, item: object) -> bool:
        return item in extended_only_canonicals()

    def __iter__(self):
        return iter(extended_only_canonicals())

    def __len__(self) -> int:
        return len(extended_only_canonicals())

    def __repr__(self) -> str:
        return f"LiveExtendedOnlyView({len(self)} canonicals)"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, (set, frozenset, _LiveExtendedOnlyView)):
            return frozenset(self) == frozenset(other)
        return NotImplemented

    def __or__(self, other: object) -> frozenset[str]:
        return frozenset(self) | frozenset(other)

    def __sub__(self, other: object) -> frozenset[str]:
        return frozenset(self) - frozenset(other)

    def __and__(self, other: object) -> frozenset[str]:
        return frozenset(self) & frozenset(other)


# Backward-compatible name.  It is a LIVE view of the extended-only surface,
# NOT a snapshot: ``canonical in PANDAS_FIRST_PRODUCTION_CANONICALS`` and
# ``sorted(PANDAS_FIRST_PRODUCTION_CANONICALS)`` always reflect the current
# ``extended_only_canonicals()``.  Daily primitives retain triple-backend
# evidence; every extended factor operator has a Pandas/Numpy semantic-reference
# production path, while Polars/SQL are promoted independently when their own
# parity/evidence gates pass.
PANDAS_FIRST_PRODUCTION_CANONICALS: _LiveExtendedOnlyView = _LiveExtendedOnlyView()

# Backward-compatible name used by LQTP compatibility code.  These operators are
# no longer on a research-only factor surface; they are promoted to Extended.
FACTOR_LIKE_RESEARCH_CANONICALS: frozenset[str] = frozenset({
    "coskewness_to_market","digital_count","expanding_rank","group_decay_linear",
    "hump_decay","idio_skew","idio_vol","intraday_vwap_deviation",
    "lqtp_historical_cvar","rank_corr","residual_momentum_capm",
    "rolling_beta_to_market","tail_beta","trade_when","ts_max_buildup","ts_moment",
    "ts_poly2_coeff","ts_poly2_resid","ts_sma_cn","ts_sum_decay",
})

LQTP_COMPAT_PARSE_CANONICALS: frozenset[str] = frozenset({
    "coskewness_to_market","cs_resid","expanding_rank","hump_decay","idio_skew","idio_vol",
    "lqtp_historical_cvar","rank_corr","real_turnover_rate","residual_momentum_capm",
    "rolling_beta_to_market","scale","sigmoid","tail_beta","true_range","ts_argmax","ts_argmin",
    "ts_decay_linear","ts_ema","ts_ewm_corr","ts_ewm_cov","ts_ewm_std","ts_ewm_var","ts_kurt",
    "ts_mad","ts_max_buildup","ts_moment","ts_poly2_resid","ts_product","ts_quantile",
    "ts_regression_slope","ts_skew","ts_sma_cn","ts_time_slope","ts_topk_sum",
})

LQTP_SOURCE_DEPENDENT_NAMES: frozenset[str] = frozenset({
    "ebitda_approx","enterprise_value","turnover_base",
})
