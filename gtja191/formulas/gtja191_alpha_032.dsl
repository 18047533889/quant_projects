# GTJA-191 Alpha 032
# source: (-1 * SUM(RANK(CORR(RANK(HIGH), RANK(VOLUME), 3)), 3))

(-1 * ts_sum(rank(ts_corr(rank(high), rank(volume), 3)), 3))
