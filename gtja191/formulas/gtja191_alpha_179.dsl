# GTJA-191 Alpha 179
# source: (RANK(CORR(VWAP, VOLUME, 4)) *RANK(CORR(RANK(LOW), RANK(MEAN(VOLUME,50)), 12)))

rank(ts_corr((high + low + close) / 3, volume, 4)) * rank(ts_corr(rank(low), rank(ts_mean(volume, 50)), 12))
