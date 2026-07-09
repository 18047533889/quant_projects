# GTJA-191 Alpha 113
# source: (-1 * ((RANK((SUM(DELAY(CLOSE, 5), 20) / 20)) * CORR(CLOSE, VOLUME, 2)) * RANK(CORR(SUM(CLOSE, 5), SUM(CLOSE, 20), 2))))

-1 * (rank(ts_sum(ts_delay(close, 5), 20) / 20) * ts_corr(close, volume, 2) * rank(ts_corr(ts_sum(close, 5), ts_sum(close, 20), 2)))
