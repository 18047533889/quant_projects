# GTJA-191 Alpha 166
# source: -20* (20-1)^1.5*SUM(CLOSE/DELAY(CLOSE,1)-1-MEAN(CLOSE/DELAY(CLOSE,1)-1,20),20)/((20-1)*(20-2)(SUM((CLOSE/DELAY(CLOSE,1),20)^2,20))^1.5)

-1656.381598545456 * ts_sum(close / ts_delay(close, 1) - 1 - ts_mean(close / ts_delay(close, 1) - 1, 20), 20) / (342 * power(ts_sum(power(close / ts_delay(close, 1), 20), 20), 1.5) + 1e-08)
