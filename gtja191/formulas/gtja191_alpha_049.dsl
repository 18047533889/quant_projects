# GTJA-191 Alpha 049
# source: SUM(((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)/(SUM(((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX (ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)+SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12))

ts_sum(if_else((high + low) >= delay(high, 1) + delay(low, 1), 0, max(abs(high - delay(high, 1)), abs(low - delay(low, 1)))), 12) / (ts_sum(if_else((high + low) >= delay(high, 1) + delay(low, 1), 0, max(abs(high - delay(high, 1)), abs(low - delay(low, 1)))), 12) + ts_sum(if_else((high + low) <= delay(high, 1) + delay(low, 1), 0, max(abs(high - delay(high, 1)), abs(low - delay(low, 1)))), 12) + 1e-8)
