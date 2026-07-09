# GTJA-191 Alpha 101
# source: ((RANK(CORR(CLOSE, SUM(MEAN(VOLUME,30), 37), 15)) < RANK(CORR(RANK(((HIGH * 0.1) + (VWAP * 0.9))), RANK(VOLUME), 11))) * -1)

(rank(ts_corr(close, ts_sum(ts_mean(volume, 30), 37), 15)) < rank(ts_corr(rank(high * 0.1 + (high + low + close) / 3 * 0.9), rank(volume), 11))) * -1
