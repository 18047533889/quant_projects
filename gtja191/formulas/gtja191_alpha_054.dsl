# GTJA-191 Alpha 054
# source: (-1 * RANK((STD(ABS(CLOSE - OPEN)) + (CLOSE - OPEN)) + CORR(CLOSE, OPEN,10)))

(-1 * rank((ts_std(abs(close - open)) + (close - open)) + ts_corr(close, open,10)))
