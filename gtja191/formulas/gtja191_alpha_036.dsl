# GTJA-191 Alpha 036
# source: RANK(SUM(CORR(RANK(VOLUME), RANK(VWAP)), 6), 2)

ts_rank(ts_sum(ts_corr(rank(volume), rank(col('vwap')), 6), 2))
