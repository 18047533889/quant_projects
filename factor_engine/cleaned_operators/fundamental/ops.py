# -*- coding: utf-8 -*-
"""
基本面与财报衍生算子。

语义
----
对财报时序字段做 TTM、同比、季度化、两期平均等变换，例如：
- ``ttm``：滚动十二个月；
- ``yoy``：同比增速；
- ``quarter`` / ``avg2``：季度化或两期平均。

依赖 canonical 基本面字段（见 ``docs/canonical_data_fields.md``）；与价量时序算子分文件维护。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from cleaned_operators.base import (
    Operator,
    OperatorMetadata,
    SeriesOperator,
    ScalarOperator,
    TwoVarOperator,
    register_operator,
)


@register_operator(name="ttm", category="fundamental", business_category="fundamental", canonical="ttm", source="factor_dsl_np")
class LqtpTtmOp(SeriesOperator):
    metadata = OperatorMetadata(
        name="ttm",
        category="fundamental",
        description="滚动十二个月（TTM）累加",
        param_names=[],
        return_type="series",
    )

    def _calculate_series(self, *args, **kwargs):
        from cleaned_operators._numpy_kernels import ttm_
        # 兼容 DataFrame 输入：逐列应用 numpy 函数
        if len(args) == 1 and hasattr(args[0], "apply"):
            return args[0].apply(lambda s: ttm_(s.values, **kwargs) if kwargs else ttm_(s.values))
        return ttm_(*args, **kwargs)


@register_operator(name="quarter", category="fundamental", business_category="fundamental", canonical="quarter", source="factor_dsl_np")
class LqtpQuarterOp(SeriesOperator):
    metadata = OperatorMetadata(
        name="quarter",
        category="fundamental",
        description="累计值转单季度",
        param_names=[],
        return_type="series",
    )

    def _calculate_series(self, *args, **kwargs):
        from cleaned_operators._numpy_kernels import quarter_
        # 兼容 DataFrame 输入：逐列应用 numpy 函数
        if len(args) == 1 and hasattr(args[0], "apply"):
            return args[0].apply(lambda s: quarter_(s.values, **kwargs) if kwargs else quarter_(s.values))
        return quarter_(*args, **kwargs)


@register_operator(name="yoy", category="fundamental", business_category="fundamental", canonical="yoy", source="factor_dsl_np")
class LqtpYoyOp(SeriesOperator):
    metadata = OperatorMetadata(
        name="yoy",
        category="fundamental",
        description="同比增速",
        param_names=[],
        return_type="series",
    )

    def _calculate_series(self, *args, **kwargs):
        from cleaned_operators._numpy_kernels import yoy_
        # 兼容 DataFrame 输入：逐列应用 numpy 函数
        if len(args) == 1 and hasattr(args[0], "apply"):
            return args[0].apply(lambda s: yoy_(s.values, **kwargs) if kwargs else yoy_(s.values))
        return yoy_(*args, **kwargs)


@register_operator(name="avg2", category="fundamental", business_category="fundamental", canonical="avg2", source="factor_dsl_np")
class LqtpAvg2Op(SeriesOperator):
    metadata = OperatorMetadata(
        name="avg2",
        category="fundamental",
        description="当期与上期均值",
        param_names=[],
        return_type="series",
    )

    def _calculate_series(self, *args, **kwargs):
        from cleaned_operators._numpy_kernels import avg2_
        # 兼容 DataFrame 输入：逐列应用 numpy 函数
        if len(args) == 1 and hasattr(args[0], "apply"):
            return args[0].apply(lambda s: avg2_(s.values, **kwargs) if kwargs else avg2_(s.values))
        return avg2_(*args, **kwargs)
