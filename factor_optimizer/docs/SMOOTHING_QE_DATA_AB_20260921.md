# Smoothing repair data A/B (synthetic research evidence)

This check exercises Factor Optimizer `compile_smoothing_repair`, real Factor Preprocess long-panel execution, and Quant Evaluator `evaluate(..., metrics=["rank_ic_series"])`.

The deterministic fixture has 40 assets and 220 daily observations. Parameters are prespecified constants; this test does not fit them from synthetic training data, and the context reference is provenance rather than evidence of fitting. Dates 0–99 provide historical warmup and only dates 100–219 are scored. Comparisons use the same assets and intersection of valid dates. Undefined IC values remain NaN and are never changed to zero.

Seeds 17, 71, and 509 cover a persistent latent signal with observation noise and an instantaneous-signal negative control (the label equals the current noisy observation; it is not a reversal process). The test evaluates SMA, EWMA, IIR, KAMA, Kalman, relative decay, absolute decay, and RAW. Each treatment independently must improve by more than 0.12 IC in the persistent/noisy regime; each must remain below 0.60 IC while RAW is effectively 1.0 in the instantaneous control. A constant factor must produce no valid IC. Every repair is checked for prefix invariance after future inputs are poisoned.

A fixed-seed common block bootstrap resamples paired daily IC differences for the a-priori confirmatory treatment, SMA, and passes those draws to `compare_paired_draws`. SMA is not selected using the holdout, and the test does not compare independently resampled marginal intervals.

Recorded mean holdout rank IC from the passing run (`15 passed in 4.49s`):

| Regime / seed | RAW | SMA | EWMA | IIR | KAMA | Kalman | decay rel | decay abs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| slow / 17 | .588290 | .805593 | .819511 | .819511 | .757108 | .819511 | .825508 | .802802 |
| slow / 71 | .559700 | .770905 | .796671 | .796671 | .741671 | .796671 | .801063 | .770355 |
| slow / 509 | .603679 | .816442 | .790439 | .790439 | .738136 | .790439 | .802037 | .812971 |
| instantaneous / 17 | 1.000000 | .492439 | .499207 | .499207 | .452087 | .499207 | .502939 | .489271 |
| instantaneous / 71 | 1.000000 | .454511 | .471159 | .471159 | .438449 | .471159 | .474340 | .459317 |
| instantaneous / 509 | 1.000000 | .514844 | .507814 | .507814 | .472162 | .507814 | .513956 | .511396 |

This is synthetic, non-live, non-production evidence. It does not certify a production factor, choose live parameters, estimate trading costs, or authorize deployment.
