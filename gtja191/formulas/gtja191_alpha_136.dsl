# GTJA-191 Alpha 136
# source: ((-1 * RANK(DELTA(RET, 3))) * CORR(OPEN, VOLUME, 10))

((-1 * rank(ts_delta(ret, 3))) * ts_corr(open, volume, 10))
