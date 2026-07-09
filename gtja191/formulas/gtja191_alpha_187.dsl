# GTJA-191 Alpha 187
# source: SUM((OPEN<=DELAY(OPEN,1)?0:MAX((HIGH-OPEN),(OPEN-DELAY(OPEN,1)))),20)

ts_sum(where(open <= ts_delay(open, 1), 0, flex_max(high - open, open - ts_delay(open, 1))), 20)
