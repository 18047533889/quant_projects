# GTJA-191 Alpha 051
# source: SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)/(SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)+SUM(((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12))

ts_sum(where(high + low <= ts_delay(high, 1) + ts_delay(low, 1), 0, flex_max(abs(high - ts_delay(high, 1)), abs(low - ts_delay(low, 1)))), 12) / (ts_sum(where(high + low <= ts_delay(high, 1) + ts_delay(low, 1), 0, flex_max(abs(high - ts_delay(high, 1)), abs(low - ts_delay(low, 1)))), 12) + ts_sum(where(high + low >= ts_delay(high, 1) + ts_delay(low, 1), 0, flex_max(abs(high - ts_delay(high, 1)), abs(low - ts_delay(low, 1)))), 12) + 1e-08)
