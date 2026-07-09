# GTJA-191 Alpha 107
# source: (((-1 * RANK((OPEN - DELAY(HIGH, 1)))) * RANK((OPEN - DELAY(CLOSE, 1)))) * RANK((OPEN - DELAY(LOW, 1))))

-1 * rank(open - ts_delay(high, 1)) * rank(open - ts_delay(close, 1)) * rank(open - ts_delay(low, 1))
