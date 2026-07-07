# GTJA-191 Alpha 001
# source: (-1 * CORR(RANK(DELTA(LOG(VOLUME), 1)), RANK(((CLOSE - OPEN) / OPEN)), 6))

(-1 * ts_corr(rank(ts_delta(log(volume), 1)), rank(((close - open) / open)), 6))
