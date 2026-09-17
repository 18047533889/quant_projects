"""Strict A-share report-period helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd

from factor_engine.cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator


def ashare_fiscal_quarter_from_period_end(period_end: pd.DataFrame) -> pd.DataFrame:
    """Map standard A-share quarter ends to 1..4; all other values fail closed."""
    if not isinstance(period_end, pd.DataFrame):
        raise TypeError("period_end must be a pandas DataFrame")
    mapping = {(3, 31): 1.0, (6, 30): 2.0, (9, 30): 3.0, (12, 31): 4.0}
    def convert(value):
        if isinstance(value, (bool, np.bool_)):
            return np.nan
        try:
            if isinstance(value, (int, np.integer)):
                text = str(int(value))
                if len(text) != 8:
                    return np.nan
                stamp = pd.to_datetime(text, format="%Y%m%d", errors="raise")
            else:
                stamp = pd.Timestamp(value)
        except (TypeError, ValueError, OverflowError):
            return np.nan
        if pd.isna(stamp):
            return np.nan
        return mapping.get((stamp.month, stamp.day), np.nan)
    return period_end.apply(lambda column: column.map(convert)).astype(float)


@register_operator(
    name="ashare_fiscal_quarter_from_period_end",
    canonical="ashare_fiscal_quarter_from_period_end",
    category="fundamental_period",
    source="ashare_fiscal_period",
)
class AshareFiscalQuarterFromPeriodEnd(SeriesOperator):
    metadata = OperatorMetadata(
        name="ashare_fiscal_quarter_from_period_end",
        category="fundamental_period",
        description="Strict quarter number from standard A-share report-period end.",
        param_names=["period_end"],
        panel_params=("period_end",),
        input_arity=1,
        tags=["fundamental", "period_aware", "pit_safe", "causal", "ashare_only"],
    )

    def _calculate_series(self, period_end):
        return ashare_fiscal_quarter_from_period_end(period_end)
