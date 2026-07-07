# GTJA-191 Alpha 093
# source: SUM((OPEN>=DELAY(OPEN,1)?0:MAX((OPEN-LOW),(OPEN-DELAY(OPEN,1)))),20)

ts_sum(if_else(open >= delay(open, 1), 0, max(open - low, open - delay(open, 1))), 20)
