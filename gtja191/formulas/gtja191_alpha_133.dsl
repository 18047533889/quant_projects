# GTJA-191 Alpha 133
# source: ((20-HIGHDAY(HIGH,20))/20)*100-((20-LOWDAY(LOW,20))/20)*100

(20 - (20 - ts_argmax(high, 20))) / 20 * 100 - (20 - (20 - ts_argmin(low, 20))) / 20 * 100
