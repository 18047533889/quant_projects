"""CogAlpha / factor_engine → LQTP evaluation helpers.

Primary converter
-----------------
- ``lqtp_converter`` — unified FE DSL / Python → LQTP API
- ``convert_to_lqtp`` — CLI (``python -m scripts.cogalpha_lqtp.convert_to_lqtp``)

See ``README.md`` in this package.
"""

from scripts.cogalpha_lqtp.lqtp_converter import (
    ConvertResult,
    convert_auto,
    convert_dsl,
    convert_python,
    convert_report_factors,
)

__all__ = [
    "ConvertResult",
    "convert_auto",
    "convert_dsl",
    "convert_python",
    "convert_report_factors",
]
