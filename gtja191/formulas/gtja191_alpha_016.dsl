# GTJA-191 Alpha 016
# source: (-1 * TSMAX(RANK(CORR(RANK(VOLUME), RANK(VWAP), 5)), 5))

-1 * ts_max(rank(ts_corr(rank(volume), rank(col('vwap')), 5)), 5)
