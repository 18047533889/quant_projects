# R24 LabelIR Spec

R24-174..187: the label layer owns a LabelIR with CONTROLLED forward semantics; the feature DSL
forbids any forward op.

## LabelOp
FORWARD_RETURN, FORWARD_EXCESS_RETURN, FORWARD_RESIDUAL_RETURN, FORWARD_TOTAL_RETURN.

## FeatureIR rule (R24-176)
Any forward op in a FEATURE IR is forbidden (assert_feature_ir_no_forward).

## Single authority (R24-177/178)
The default label formula is `forward_return(close, h)` — matching the runtime builder
`close[t+h]/close[t]-1`.  A backward `ts_pct(close, h)` is NEVER a label formula.

## Corporate action (R24-179/180)
Default target is a return ratio (CA-safe); raw close forward ratio only with a CA-safe assertion.

## Strict horizon (R24-181) / bar clock (R24-182/183)
horizon_bars is a strict positive int; gap_bars is a BAR count, never a calendar Timedelta.

## Missing / delisting (R24-186/187)
label_missing_policy / terminal_return_policy govern delisting / suspension / missing close.
