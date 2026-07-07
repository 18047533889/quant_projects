ema(close / (high - low), 8) * (sigmoid(5 * (volume - 0.85 * ts_mean(volume, 25))) * ema(volume, 10) + (1 - sigmoid(5 * (volume - 0.85 * ts_mean(volume, 25)))) * ema(volume, 30))
