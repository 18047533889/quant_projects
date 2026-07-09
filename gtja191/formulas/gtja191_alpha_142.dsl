# GTJA-191 Alpha 142
# source: (((-1 * RANK(TSRANK(CLOSE, 10))) * RANK(DELTA(DELTA(CLOSE, 1), 1))) * RANK(TSRANK((VOLUME /MEAN(VOLUME,20)), 5)))

-1 * rank(ts_rank(close, 10)) * rank(ts_delta(ts_delta(close, 1), 1)) * rank(ts_rank(volume / ts_mean(volume, 20), 5))
