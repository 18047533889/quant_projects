# GTJA-191 Alpha 073
# source: ((TSRANK(DECAYLINEAR(DECAYLINEAR(CORR((CLOSE), VOLUME, 10), 16), 4), 5) - RANK(DECAYLINEAR(CORR(VWAP, MEAN(VOLUME,30), 4),3))) * -1)

((ts_rank(ts_decay_linear(ts_decay_linear(ts_corr((close), volume, 10), 16), 4), 5) - rank(ts_decay_linear(ts_corr(((high + low + close) / 3), ts_mean(volume,30), 4),3))) * -1)
