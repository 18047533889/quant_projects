# GTJA-191 Alpha 125
# source: (RANK(DECAYLINEAR(CORR((VWAP), MEAN(VOLUME,80),17), 20)) / RANK(DECAYLINEAR(DELTA(((CLOSE * 0.5) + (VWAP * 0.5)), 3), 16)))

rank(ts_decay_linear(ts_corr(col('vwap'), ts_mean(volume, 80), 17), 20)) / rank(ts_decay_linear(ts_delta(close * 0.5 + col('vwap') * 0.5, 3), 16))
