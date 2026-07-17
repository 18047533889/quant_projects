# Factor Operator Overhaul

The audited overhaul enforces the following runtime contracts:

- a registered `polars` backend must execute without converting the panel to pandas;
- unsupported complex operators fall back to the pandas backend instead of advertising false Polars coverage;
- daily operators are shape preserving and causal;
- financial statement transforms operate on distinct point-in-time report periods, not forward-filled daily rows;
- composite indicators such as MACD reuse registered primitive operators;
- historical duplicate names remain aliases rather than independent canonicals.

New primitives include rolling regression outputs, joint cross-sectional neutralization, explicit top/bottom-K aggregations, and rolling tail means. Fixed financial ratios are maintained as factor recipes under `factor_recipes/`.
