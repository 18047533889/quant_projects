# GTJA-191 Alpha 125
# source: (RANK(DECAYLINEAR(CORR((VWAP), MEAN(VOLUME,80),17), 20)) / RANK(DECAYLINEAR(DELTA(((CLOSE * 0.5) + (VWAP * 0.5)), 3), 16)))

rank(ts_decay_linear(ts_corr((high + low + close) / 3, ts_mean(volume, 80), 17), 20)) / rank(ts_decay_linear(ts_delta(close * 0.5 + (high + low + close) / 3 * 0.5, 3), 16))
