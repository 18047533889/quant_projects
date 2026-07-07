# GTJA-191 Alpha 064
# source: (MAX(RANK(DECAYLINEAR(CORR(RANK(VWAP), RANK(VOLUME), 4), 4)), RANK(DECAYLINEAR(MAX(CORR(RANK(CLOSE), RANK(MEAN(VOLUME,60)), 4), 13), 14))) * -1)

(max(rank(ts_decay_linear(ts_corr(rank(((high + low + close) / 3)), rank(volume), 4), 4)), rank(ts_decay_linear(ts_max(ts_corr(rank(close), rank(ts_mean(volume,60)), 4), 13), 14))) * -1)
