# GTJA-191 Alpha 005
# source: (-1 * TSMAX(CORR(TSRANK(VOLUME, 5), TSRANK(HIGH, 5), 5), 3))

-1 * ts_max(ts_corr(ts_rank(volume, 5), ts_rank(high, 5), 5), 3)
