# GTJA-191 Alpha 123
# source: ((RANK(CORR(SUM(((HIGH + LOW) / 2), 20), SUM(MEAN(VOLUME,60), 20), 9)) < RANK(CORR(LOW, VOLUME, 6))) * -1)

((rank(ts_corr(ts_sum(((high + low) / 2), 20), ts_sum(ts_mean(volume,60), 20), 9)) < rank(ts_corr(low, volume, 6))) * -1)
