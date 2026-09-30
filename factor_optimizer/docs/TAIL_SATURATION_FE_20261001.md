# Tail saturation through FactorEngine

Research `TAIL_SATURATION` plans now call FE's canonical `winsorize` operator
through `adapters/fe_tail_saturation.py`. Keep `allow_research=True`; this
change does not authorize production factor publication or fit on validation
data. The cross-sectional thresholds use the current date's observed finite
values, with linear quantile interpolation.

For saturation quantile q in (0.5, 1), bounds are `[0,q]` for `top`, `[1-q,1]`
for `bottom`, and `[1-q,q]` for `both`. In each finite cross-section:

$$y_i = \min(\max(x_i,Q_{lower}),Q_{upper}).$$

The adapter converts NaN/Inf inputs to missing through FE. It groups dates
into power-of-two width buckets and pads rows with NaN, which FE excludes
from quantiles. Ordinal columns preserve observations without an asset pivot.
It restores outputs by row position. Unused categorical dates add no rows;
the public plan still rejects null date/asset IDs and duplicate identities.

Each batch targets at most one million padded cells. One cross-section is
the floor, so a wider section can exceed that target. The adapter also retains
input-sized values, positions and output; the target is not a hard RSS cap.

Bounded A/B samples agreed with the previous FP transform. For 128 dates of
500 rows, median time was 0.03633 s (FP) and 0.00708 s (batched FE). A sparse
sample with 33,477 rows and 107 distinct date widths measured 0.02950 s and
0.02083 s. These samples do not establish gains on every workload.
