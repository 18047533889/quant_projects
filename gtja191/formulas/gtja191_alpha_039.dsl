# GTJA-191 Alpha 039
# source: ((RANK(DECAYLINEAR(DELTA((CLOSE), 2),8)) - RANK(DECAYLINEAR(CORR(((VWAP * 0.3) + (OPEN * 0.7)), SUM(MEAN(VOLUME,180), 37), 14), 12))) * -1)

(rank(ts_decay_linear(ts_delta(close, 2), 8)) - rank(ts_decay_linear(ts_corr((high + low + close) / 3 * 0.3 + open * 0.7, ts_sum(ts_mean(volume, 180), 37), 14), 12))) * -1
