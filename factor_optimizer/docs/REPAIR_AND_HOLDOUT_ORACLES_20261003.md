# Repair-method and holdout regression oracles

## Why these tests exist

Correct output on a dense, sorted panel is insufficient for daily batch use.
The new `test_repair_method_oracles_oct03.py` checks EWMA, IIR, KAMA and Kalman
through actual frozen `compile_value_repair` plans on sparse, interleaved asset
histories, irregular calendar gaps and repeated caller indices. Historical
time scale and parameters are frozen; these tests do not tune on VALID or TEST.

Each method is compared with hand-derived reference values. Changing late A
observations must not change earlier A outputs or any B output. Insufficient
KAMA warmup is explicitly missing rather than an empty, vacuous comparison.
U/inverted-U tests calculate tied percentile ranks independently per date:

$$r_i=\frac{\operatorname{average\_rank}(x_i)-1}{n-1},\qquad
U_i=|r_i-c|^p,\qquad U_i^{\mathrm{inverse}}=-U_i.$$

These formulas describe the tested multi-member finite cross-sections; missing
and infinite values are excluded under this repair contract. Singleton rules
remain the separately documented FP contract.

## TRAIN winner only: no validation retry

`test_research_batch_holdout_oracles_oct03.py` constructs a fixed symmetric
factor with sign flip and U-distance treatments. TRAIN selects SIGN over a
positive-gain U runner-up. VALID deliberately favors U and rejects SIGN.
The optimizer must retain RAW and must not evaluate U as a replacement VALID
candidate. The exact compiled U plan is evaluated independently after the
optimization, proving that a genuinely VALID-positive alternative existed.
Independent Spearman calculations also check the intended ordering.

This guards against implicitly fitting the holdout by trying successive TRAIN
runners-up. It is a correctness regression, not proof that these treatments
improve any real market factor or that every optimizer method is bug-free.
Production execution and FE backend admission remain separate requirements.
