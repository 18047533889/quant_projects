"""
Example: Using adapters with mock providers.

Demonstrates how to use the adapters layer without requiring
factor_assets or dataaccess packages installed.
"""

import numpy as np
from factor_preprocess.adapters.factor_assets import FactorAssetsAdapter
from factor_preprocess.adapters.data_access import DataAccessAdapter


# Mock FactorSet (simulating factor_assets.FactorSet)
class MockFactorSet:
    def __init__(self, set_id, name, factor_ids):
        self.set_id = set_id
        self.name = name
        self.factor_ids = factor_ids
        self.size = len(factor_ids)
        self.created_at = "2024-01-01T00:00:00"
        self.universe_ref = "TOP1000"
        self.frequency = "daily"


# Simple mock provider
class SimpleFactorProvider:
    def get_factor_values(self, factor_id, start_date=None, end_date=None, universe=None):
        # Generate mock data
        n_dates, n_assets = 100, 500
        return {
            'values': np.random.randn(n_dates, n_assets),
            'dates': np.array([f"2024-01-{i+1:02d}" for i in range(min(31, n_dates))]),
            'assets': np.array([f"ASSET_{i}" for i in range(n_assets)]),
            'metadata': {'factor_id': factor_id, 'source': 'example'},
        }

    def get_factor_batch(self, factor_ids, start_date=None, end_date=None, universe=None):
        n_dates, n_assets, n_factors = 100, 500, len(factor_ids)
        values = np.random.randn(n_dates, n_assets, n_factors)

        return {
            'values': values,
            'dates': np.array([f"2024-{i//30+1:02d}-{i%30+1:02d}" for i in range(n_dates)]),
            'assets': np.array([f"ASSET_{i}" for i in range(n_assets)]),
            'factor_ids': factor_ids,
            'metadata': {fid: {'factor_id': fid, 'source': 'example'} for fid in factor_ids},
        }

    def validate_factor_set(self, factor_set):
        return True


class SimpleExposureProvider:
    def get_industry_exposure(self, market, start_date=None, end_date=None,
                              assets=None, industry_classification="default"):
        n_dates, n_assets = 100, 500
        # Mock industry codes
        industries = np.array([f"IND_{i % 10}" for i in range(n_assets)])
        values = np.tile(industries, (n_dates, 1))

        return {
            'values': values,
            'dates': np.array([f"2024-{i//30+1:02d}-{i%30+1:02d}" for i in range(n_dates)]),
            'assets': np.array([f"ASSET_{i}" for i in range(n_assets)]),
            'classification': industry_classification,
            'categories': [f"IND_{i}" for i in range(10)],
            'metadata': {'source': 'example', 'market': market},
        }

    def get_size_exposure(self, market, start_date=None, end_date=None,
                         assets=None, size_metric="market_cap"):
        n_dates, n_assets = 100, 500
        # Mock market caps
        values = np.random.lognormal(mean=20, sigma=2, size=(n_dates, n_assets))

        return {
            'values': values,
            'dates': np.array([f"2024-{i//30+1:02d}-{i%30+1:02d}" for i in range(n_dates)]),
            'assets': np.array([f"ASSET_{i}" for i in range(n_assets)]),
            'metric': size_metric,
            'metadata': {'source': 'example', 'market': market},
        }

    def get_sector_exposure(self, market, start_date=None, end_date=None,
                           assets=None, sector_classification="default"):
        n_dates, n_assets = 100, 500
        sectors = np.array([f"SEC_{i % 5}" for i in range(n_assets)])
        values = np.tile(sectors, (n_dates, 1))

        return {
            'values': values,
            'dates': np.array([f"2024-{i//30+1:02d}-{i%30+1:02d}" for i in range(n_dates)]),
            'assets': np.array([f"ASSET_{i}" for i in range(n_assets)]),
            'classification': sector_classification,
            'categories': [f"SEC_{i}" for i in range(5)],
            'metadata': {'source': 'example', 'market': market},
        }

    def get_beta_exposure(self, market, start_date=None, end_date=None,
                         assets=None, window_days=252):
        n_dates, n_assets = 100, 500
        values = np.random.normal(loc=1.0, scale=0.3, size=(n_dates, n_assets))

        return {
            'values': values,
            'dates': np.array([f"2024-{i//30+1:02d}-{i%30+1:02d}" for i in range(n_dates)]),
            'assets': np.array([f"ASSET_{i}" for i in range(n_assets)]),
            'window_days': window_days,
            'metadata': {'source': 'example', 'market': market},
        }

    def get_custom_exposure(self, exposure_name, market, start_date=None,
                           end_date=None, assets=None, **kwargs):
        n_dates, n_assets = 100, 500
        values = np.random.randn(n_dates, n_assets)

        return {
            'values': values,
            'dates': np.array([f"2024-{i//30+1:02d}-{i%30+1:02d}" for i in range(n_dates)]),
            'assets': np.array([f"ASSET_{i}" for i in range(n_assets)]),
            'exposure_name': exposure_name,
            'metadata': {'source': 'example', 'market': market, **kwargs},
        }


def main():
    print("=" * 70)
    print("Factor Preprocess Adapters Example")
    print("=" * 70)

    # 1. Create adapters with custom providers
    print("\n1. Creating adapters with mock providers...")
    factor_provider = SimpleFactorProvider()
    exposure_provider = SimpleExposureProvider()

    factor_adapter = FactorAssetsAdapter(factor_provider)
    exposure_adapter = DataAccessAdapter(exposure_provider)
    print("   ✓ Adapters created")

    # 2. Load factor data
    print("\n2. Loading factor data...")
    factor_set = MockFactorSet(
        set_id="momentum_value_quality",
        name="Momentum + Value + Quality",
        factor_ids=("momentum_12m", "book_to_price", "roe")
    )

    factor_data = factor_adapter.load_factor_set(factor_set)
    print(f"   ✓ Loaded {len(factor_data['factor_ids'])} factors")
    print(f"   Shape: {factor_data['values'].shape}")
    print(f"   Factors: {factor_data['factor_ids']}")

    # 3. Load exposure data
    print("\n3. Loading exposure data...")
    exposures = exposure_adapter.fetch_multi_exposure(
        market="ashare",
        exposure_types=["industry", "size", "beta"]
    )
    print(f"   ✓ Loaded {len(exposures)} exposures")
    for exp_type, exp_data in exposures.items():
        print(f"   - {exp_type}: {exp_data['values'].shape}")

    # 4. Use in preprocessing pipeline
    print("\n4. Using in preprocessing pipeline...")
    from factor_preprocess.transforms.cross_sectional import cs_zscore

    # Cross-sectional standardization
    cs_standardized = cs_zscore(factor_data['values'])
    print(f"   ✓ Cross-sectional zscore: {cs_standardized.shape}")

    # Compute some statistics
    mean_factor = np.nanmean(cs_standardized, axis=(0, 1))
    print(f"   ✓ Factor means after standardization: {mean_factor}")
    print(f"   ✓ All factors ready for model input")

    # 5. Prepare for neutralization (example)
    print("\n5. Preparing for neutralization...")
    industry_exposure = exposures['industry']['values']
    size_exposure = exposures['size']['values']

    print(f"   ✓ Industry exposure: {industry_exposure.shape}")
    print(f"   ✓ Size exposure: {size_exposure.shape}")
    print(f"   Ready for OLS neutralization")

    # 6. Metadata preservation
    print("\n6. Metadata preservation...")
    set_meta = factor_data['set_metadata']
    print(f"   Set ID: {set_meta['set_id']}")
    print(f"   Set Name: {set_meta['set_name']}")
    print(f"   Universe: {set_meta['universe']}")
    print(f"   Frequency: {set_meta['frequency']}")

    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
