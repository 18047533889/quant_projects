# GTJA-191 Alpha 140
# source: MIN(RANK(DECAYLINEAR(((RANK(OPEN) + RANK(LOW)) - (RANK(HIGH) + RANK(CLOSE))), 8)), TSRANK(DECAYLINEAR(CORR(TSRANK(CLOSE, 8), TSRANK(MEAN(VOLUME,60), 20), 8), 7), 3))

flex_min(rank(ts_decay_linear(rank(open) + rank(low) - (rank(high) + rank(close)), 8)), ts_rank(ts_decay_linear(ts_corr(ts_rank(close, 8), ts_rank(ts_mean(volume, 60), 20), 8), 7), 3))
