# GTJA-191 Alpha 187
# source: SUM((OPEN<=DELAY(OPEN,1)?0:MAX((HIGH-OPEN),(OPEN-DELAY(OPEN,1)))),20)

ts_sum(if_else(open <= delay(open, 1), 0, max(high - open, open - delay(open, 1))), 20)
