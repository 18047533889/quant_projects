# GTJA-191 Alpha 121
# source: ((RANK((VWAP - MIN(VWAP, 12)))^TSRANK(CORR(TSRANK(VWAP, 20), TSRANK(MEAN(VOLUME,60), 2), 18), 3)) * -1)

power(rank((high + low + close) / 3 - ts_min((high + low + close) / 3, 12)), ts_rank(ts_corr(ts_rank((high + low + close) / 3, 20), ts_rank(ts_mean(volume, 60), 2), 18), 3)) * -1
