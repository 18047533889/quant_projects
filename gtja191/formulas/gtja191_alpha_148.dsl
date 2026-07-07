# GTJA-191 Alpha 148
# source: ((RANK(CORR((OPEN), SUM(MEAN(VOLUME,60), 9), 6)) < RANK((OPEN - TSMIN(OPEN, 14)))) * -1)

((rank(ts_corr((open), ts_sum(ts_mean(volume,60), 9), 6)) < rank((open - ts_min(open, 14)))) * -1)
