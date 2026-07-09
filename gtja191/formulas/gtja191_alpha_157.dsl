# GTJA-191 Alpha 157
# source: (MIN(PROD(RANK(RANK(LOG(SUM(TSMIN(RANK(RANK((-1 * RANK(DELTA((CLOSE - 1), 5))))), 2), 1)))), 1), 5) + TSRANK(DELAY((-1 * RET), 6), 5))

ts_min(ts_product(rank(rank(log(ts_sum(ts_min(rank(rank(-1 * rank(ts_delta(close - 1, 5)))), 2), 1))))), 5) + ts_rank(ts_delay(-1 * ret, 6), 5)
