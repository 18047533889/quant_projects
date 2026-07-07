# GTJA-191 Alpha 042
# source: ((-1 * RANK(STD(HIGH, 10))) * CORR(HIGH, VOLUME, 10))

((-1 * rank(ts_std(high, 10))) * ts_corr(high, volume, 10))
