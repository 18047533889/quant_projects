# GTJA-191 Alpha 104
# source: (-1 * (DELTA(CORR(HIGH, VOLUME, 5), 5) * RANK(STD(CLOSE, 20))))

-1 * (ts_delta(ts_corr(high, volume, 5), 5) * rank(ts_std(close, 20)))
