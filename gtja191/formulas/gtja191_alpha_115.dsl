# GTJA-191 Alpha 115
# source: (RANK(CORR(((HIGH * 0.9) + (CLOSE * 0.1)), MEAN(VOLUME,30), 10))^RANK(CORR(TSRANK(((HIGH + LOW) / 2), 4), TSRANK(VOLUME, 10), 7)))

power(rank(ts_corr(high * 0.9 + close * 0.1, ts_mean(volume, 30), 10)), rank(ts_corr(ts_rank((high + low) / 2, 4), ts_rank(volume, 10), 7)))
