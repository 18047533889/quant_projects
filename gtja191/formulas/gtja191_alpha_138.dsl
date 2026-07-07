# GTJA-191 Alpha 138
# source: ((RANK(DECAYLINEAR(DELTA((((LOW * 0.7) + (VWAP *0.3))), 3), 20)) - TSRANK(DECAYLINEAR(TSRANK(CORR(TSRANK(LOW, 8), TSRANK(MEAN(VOLUME,60), 17), 5), 19), 16), 7)) * -1)

((rank(ts_decay_linear(ts_delta((((low * 0.7) + (((high + low + close) / 3) *0.3))), 3), 20)) - ts_rank(ts_decay_linear(ts_rank(ts_corr(ts_rank(low, 8), ts_rank(ts_mean(volume,60), 17), 5), 19), 16), 7)) * -1)
