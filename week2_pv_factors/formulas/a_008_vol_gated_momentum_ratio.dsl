(close / delay(close, 5) - 1) * (0.5 + 0.5 * ts_rank(ts_atr(high, low, close, 14) / close, 20)) * volume / ts_mean(volume, 20)
