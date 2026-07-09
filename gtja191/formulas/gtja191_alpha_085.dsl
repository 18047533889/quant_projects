# GTJA-191 Alpha 085
# source: (TSRANK((VOLUME / MEAN(VOLUME,20)), 20) * TSRANK((-1 * DELTA(CLOSE, 7)), 8))

ts_rank(volume / ts_mean(volume, 20), 20) * ts_rank(-1 * ts_delta(close, 7), 8)
