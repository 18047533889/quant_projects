# GTJA-191 Alpha 118
# source: SUM(HIGH-OPEN,20)/SUM(OPEN-LOW,20)*100

ts_sum(high - open, 20) / ts_sum(open - low, 20) * 100
