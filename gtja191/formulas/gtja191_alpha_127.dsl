# GTJA-191 Alpha 127
# source: (MEAN((100*(CLOSE-MAX(CLOSE,12))/(MAX(CLOSE,12)))^2))^(1/2)

power(ts_mean(power(100 * (close - ts_max(close, 12)) / (ts_max(close, 12) + 1e-8), 2), 20), 0.5)
