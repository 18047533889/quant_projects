# GTJA-191 Alpha 083
# source: (-1 * RANK(COVIANCE(RANK(HIGH), RANK(VOLUME), 5)))

-1 * rank(ts_cov(rank(high), rank(volume), 5))
