# GTJA-191 Alpha 044
# source: (TSRANK(DECAYLINEAR(CORR(((LOW )), MEAN(VOLUME,10), 7), 6),4) + TSRANK(DECAYLINEAR(DELTA((VWAP), 3), 10), 15))

ts_rank(ts_decay_linear(ts_corr(low, ts_mean(volume, 10), 7), 6), 4) + ts_rank(ts_decay_linear(ts_delta(col('vwap'), 3), 10), 15)
