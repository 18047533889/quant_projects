# GTJA-191 Alpha 141
# source: (RANK(CORR(RANK(HIGH), RANK(MEAN(VOLUME,15)), 9))* -1)

rank(ts_corr(rank(high), rank(ts_mean(volume, 15)), 9)) * -1
