# GTJA-191 Alpha 056
# source: (RANK((OPEN - TSMIN(OPEN, 12))) < RANK((RANK(CORR(SUM(((HIGH + LOW) / 2), 19), SUM(MEAN(VOLUME,40), 19), 13))^5)))

rank(open - ts_min(open, 12)) < rank(rank(ts_corr(ts_sum((high + low) / 2, 19), ts_sum(ts_mean(volume, 40), 19), power(13)), 5))
