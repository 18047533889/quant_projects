# GTJA-191 Alpha 025
# source: ((-1 * RANK((DELTA(CLOSE, 7) * (1 - RANK(DECAYLINEAR((VOLUME / MEAN(VOLUME,20)), 9)))))) * (1 + RANK(SUM(RET, 250))))

((-1 * rank((ts_delta(close, 7) * (1 - rank(ts_decay_linear((volume / ts_mean(volume,20)), 9)))))) * (1 + rank(ts_sum(ret, 250))))
