# GTJA-191 Alpha 166
# source: -20* (20-1)^1.5*SUM(CLOSE/DELAY(CLOSE,1)-1-MEAN(CLOSE/DELAY(CLOSE,1)-1,20),20)/((20-1)*(20-2)(SUM((CLOSE/DELAY(CLOSE,1),20)^2,20))^1.5)

-20 * power(20 - 1, 1.5) * ts_sum(close / delay(close, 1) - 1 - ts_mean(close / delay(close, 1) - 1, 20), 20) / ((20 - 1) * (20 - 2) * power(ts_sum(power(close / delay(close, 1), 20), 20), 1.5) + 1e-8)
