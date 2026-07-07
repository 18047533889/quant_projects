# GTJA-191 Alpha 061
# source: (MAX(RANK(DECAYLINEAR(DELTA(VWAP, 1), 12)), RANK(DECAYLINEAR(RANK(CORR((LOW),MEAN(VOLUME,80), 8)), 17))) * -1)

(max(rank(ts_decay_linear(ts_delta(((high + low + close) / 3), 1), 12)), rank(ts_decay_linear(rank(ts_corr((low),ts_mean(volume,80), 8)), 17))) * -1)
