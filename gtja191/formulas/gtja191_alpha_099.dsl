# GTJA-191 Alpha 099
# source: (-1 * RANK(COVIANCE(RANK(CLOSE), RANK(VOLUME), 5)))

(-1 * rank(ts_cov(rank(close), rank(volume), 5)))
