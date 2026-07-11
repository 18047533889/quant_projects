# GTJA-191 Alpha 119
# source: (RANK(DECAYLINEAR(CORR(VWAP, SUM(MEAN(VOLUME,5), 26), 5), 7)) - RANK(DECAYLINEAR(TSRANK(MIN(CORR(RANK(OPEN), RANK(MEAN(VOLUME,15)), 21), 9), 7), 8)))

rank(ts_decay_linear(ts_corr(col('vwap'), ts_sum(ts_mean(volume, 5), 26), 5), 7)) - rank(ts_decay_linear(ts_rank(ts_min(ts_corr(rank(open), rank(ts_mean(volume, 15)), 21), 9), 7), 8))
