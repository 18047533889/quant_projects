# GTJA-191 Alpha 012
# source: (RANK((OPEN - (SUM(VWAP, 10) / 10)))) * (-1 * (RANK(ABS((CLOSE - VWAP)))))

rank(open - ts_sum(col('vwap'), 10) / 10) * (-1 * rank(abs(close - col('vwap'))))
