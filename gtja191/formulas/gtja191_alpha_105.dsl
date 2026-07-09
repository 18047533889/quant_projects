# GTJA-191 Alpha 105
# source: (-1 * CORR(RANK(OPEN), RANK(VOLUME), 10))

-1 * ts_corr(rank(open), rank(volume), 10)
