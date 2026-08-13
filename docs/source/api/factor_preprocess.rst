Factor Preprocess Module
=========================

.. automodule:: factor_preprocess
   :members:
   :undoc-members:
   :show-inheritance:

Outlier Treatment
-----------------

.. autofunction:: factor_preprocess.winsorize

.. autofunction:: factor_preprocess.truncate

.. autofunction:: factor_preprocess.mad_outlier_removal

Standardization
---------------

.. autofunction:: factor_preprocess.standardize

.. autofunction:: factor_preprocess.zscore

.. autofunction:: factor_preprocess.rank_normalize

.. autofunction:: factor_preprocess.quantile_normalize

Neutralization
--------------

.. autofunction:: factor_preprocess.neutralize

.. autofunction:: factor_preprocess.industry_neutralize

.. autofunction:: factor_preprocess.orthogonalize

Missing Data
------------

.. autofunction:: factor_preprocess.fill_missing

.. autofunction:: factor_preprocess.forward_fill

.. autofunction:: factor_preprocess.interpolate

Factor Combination
------------------

.. autofunction:: factor_preprocess.combine_factors

.. autofunction:: factor_preprocess.ic_weighted_combination

.. autofunction:: factor_preprocess.optimal_combination

Cross-Sectional Operations
---------------------------

.. autofunction:: factor_preprocess.demean

.. autofunction:: factor_preprocess.demedian

.. autofunction:: factor_preprocess.cross_sectional_zscore

Time Series Operations
----------------------

.. autofunction:: factor_preprocess.smooth

.. autofunction:: factor_preprocess.exponential_smooth

.. autofunction:: factor_preprocess.rolling_demean
