# GTJA-191 Alpha 033
# source: ((((-1 * TSMIN(LOW, 5)) + DELAY(TSMIN(LOW, 5), 5)) * RANK(((SUM(RET, 240) - SUM(RET, 20)) / 220))) * TSRANK(VOLUME, 5))

(-1 * ts_min(low, 5) + ts_delay(ts_min(low, 5), 5)) * rank((ts_sum(ret, 240) - ts_sum(ret, 20)) / 220) * ts_rank(volume, 5)
