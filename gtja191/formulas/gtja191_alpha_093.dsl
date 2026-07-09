# GTJA-191 Alpha 093
# source: SUM((OPEN>=DELAY(OPEN,1)?0:MAX((OPEN-LOW),(OPEN-DELAY(OPEN,1)))),20)

ts_sum(where(open >= ts_delay(open, 1), 0, flex_max(open - low, open - ts_delay(open, 1))), 20)
