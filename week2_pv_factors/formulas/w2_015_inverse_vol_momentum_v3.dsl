(close / delay(close, 5) - 1) * (1.5 - 0.5 * ts_rank(ts_std(ts_delta(close, 1), 20), 20))
