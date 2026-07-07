# GTJA-191 Alpha 131
# source: (RANK(DELAT(VWAP, 1))^TSRANK(CORR(CLOSE,MEAN(VOLUME,50), 18), 18))

power(rank(ts_delta(((high + low + close) / 3), 1)), ts_rank(ts_corr(close, ts_mean(volume, 50), 18), 18))
