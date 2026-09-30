# Factor diagnosis constant detection

The constant flag is defined on finite observations admitted by the factor
validity mask. With no valid observations it is true. Otherwise:

$$
\mathrm{is\_constant}(x) = [\min(x_{valid}) = \max(x_{valid})].
$$

The former `std(x_valid) == 0` test incorrectly classified distinct values
such as `[1e-200, 2e-200]` as constant because the variance underflowed.
The extrema rule also handles large constant values without variance overflow.
Distribution means use the existing reduction for ordinary inputs. If its sum
overflows despite every admitted observation being finite, the mean is computed
after division by the maximum absolute valid value and then rescaled. This
keeps a mean such as `mean([1e308, 1e308])` finite at `1e308`.

On server-c, seven alternating runs over 2,000,000 float64 normal values
(NumPy RNG seed 30) measured median 0.003247 s for std/min/max/mean versus
0.001085 s for min/max/mean: 2.99x for this statistics section. This is a
microbenchmark, not an end-to-end evaluation speedup or COS benchmark.
