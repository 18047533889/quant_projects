# GTJA-191 Alpha 092
# source: (MAX(RANK(DECAYLINEAR(DELTA(((CLOSE * 0.35) + (VWAP *0.65)), 2), 3)), TSRANK(DECAYLINEAR(ABS(CORR((MEAN(VOLUME,180)), CLOSE, 13)), 5), 15)) * -1)

flex_max(rank(ts_decay_linear(ts_delta(close * 0.35 + col('vwap') * 0.65, 2), 3)), ts_rank(ts_decay_linear(abs(ts_corr(ts_mean(volume, 180), close, 13)), 5), 15)) * -1
