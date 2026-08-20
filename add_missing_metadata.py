#!/usr/bin/env python3
"""
Add metadata to operators that are missing it
"""

import re

# Read the file
with open('cleaned_operators/polars_native/ts_batch1.py', 'r') as f:
    content = f.read()

# Operators missing metadata (from our earlier analysis):
# They have: decorator, class, docstring, then immediately def _calculate_series
operators_to_fix = [
    ('ts_distance_cov', ['x', 'y', 'window'], 'Distance covariance'),
    ('ts_ffill_limited', ['feature', 'max_fill'], 'Forward fill with maximum fill limit'),
    ('ts_median3_causal', ['feature'], '3-point median filter (causal: uses current and 2 past)'),
    ('ts_rolling_median_causal', ['feature', 'window'], 'Rolling median (alias for ts_median)'),
    ('ts_robust_ema', ['feature', 'span', 'clip_std'], 'Robust EMA that clips outliers before smoothing'),
    ('ts_run_strength', ['feature', 'window'], 'Average absolute return during runs (consecutive same-sign moves)'),
    ('ts_run_efficiency', ['feature', 'window'], 'Ratio of run-based distance to total distance'),
    ('ts_quantile_regression_slope', ['feature', 'window', 'quantile'], 'Quantile regression slope (simplified via weighted least squares)'),
    ('ts_expectile', ['feature', 'window', 'tau'], 'Rolling expectile (asymmetric mean)'),
    ('ts_support_level', ['feature', 'window', 'buffer'], 'Support level (rolling minimum with buffer)'),
    ('ts_resistance_level', ['feature', 'window', 'buffer'], 'Resistance level (rolling maximum with buffer)'),
    ('ts_distance_to_support', ['feature', 'window'], 'Distance from current price to support level'),
    ('ts_distance_to_resistance', ['feature', 'window'], 'Distance from current price to resistance level'),
    ('ts_breakout_high', ['feature', 'window', 'threshold'], 'Boolean: price breaks above resistance'),
    ('ts_breakdown_low', ['feature', 'window', 'threshold'], 'Boolean: price breaks below support'),
]

for op_name, param_names, description in operators_to_fix:
    # Find the class for this operator
    pattern = rf'(@register_operator\(name="{op_name}"[^)]+\)\nclass (\w+)\(SeriesOperator\):\n    """[^"]+"""\n)(    def _calculate_series)'

    match = re.search(pattern, content)
    if match:
        before = match.group(1)
        method_def = match.group(3)
        class_name = match.group(2)

        # Build metadata
        metadata_code = f'''    metadata = OperatorMetadata(
        name="{op_name}",
        category="time_series",
        description="{description}",
        param_names={param_names},
        return_type="series",
        tags=["time_series", "rolling", "pit_safe", "polars_native"],
    )
'''

        # Add param_specs based on param_names
        param_specs = []
        if 'window' in param_names:
            param_specs.append('        "window": ParamSpec(dtype=int, min=1, param_role=ParamRole.WINDOW),')
        if 'short_window' in param_names:
            param_specs.append('        "short_window": ParamSpec(dtype=int, min=1, param_role=ParamRole.WINDOW),')
        if 'long_window' in param_names:
            param_specs.append('        "long_window": ParamSpec(dtype=int, min=1, param_role=ParamRole.WINDOW),')
        if 'span' in param_names:
            param_specs.append('        "span": ParamSpec(dtype=int, min=1, param_role=ParamRole.WINDOW),')
        if 'max_fill' in param_names:
            param_specs.append('        "max_fill": ParamSpec(dtype=int, min=1, default=5, param_role=ParamRole.SCALAR),')
        if 'clip_std' in param_names:
            param_specs.append('        "clip_std": ParamSpec(dtype=float, min=0.0, default=3.0, param_role=ParamRole.SCALAR),')
        if 'quantile' in param_names:
            param_specs.append('        "quantile": ParamSpec(dtype=float, min=0.0, max=1.0, default=0.5, param_role=ParamRole.SCALAR),')
        if 'tau' in param_names:
            param_specs.append('        "tau": ParamSpec(dtype=float, min=0.0, max=1.0, default=0.5, param_role=ParamRole.SCALAR),')
        if 'buffer' in param_names:
            param_specs.append('        "buffer": ParamSpec(dtype=float, min=0.0, max=1.0, default=0.02, param_role=ParamRole.SCALAR),')
        if 'threshold' in param_names:
            param_specs.append('        "threshold": ParamSpec(dtype=float, default=1.0, param_role=ParamRole.SCALAR),')

        if param_specs:
            metadata_code += '    metadata.param_specs = {\n'
            metadata_code += '\n'.join(param_specs)
            metadata_code += '\n    }\n'

        metadata_code += '\n'

        # Replace
        replacement = before + metadata_code + method_def
        content = content.replace(match.group(0), replacement)
        print(f"Added metadata to {class_name} ({op_name})")
    else:
        print(f"WARNING: Could not find {op_name}")

# Write back
with open('cleaned_operators/polars_native/ts_batch1.py', 'w') as f:
    f.write(content)

print("\nDone adding metadata!")
