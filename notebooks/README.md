# Jupyter Notebook Integration for Factor Engine

Complete Jupyter notebook integration with tutorials, utilities, and examples.

## Directory Structure

```
notebooks/
├── tutorials/           # Step-by-step learning notebooks
├── examples/            # Real-world example notebooks
└── utils/              # Helper functions for notebooks
```

## Tutorials

Progressive learning path from basics to production:

1. **01_basic_factor_evaluation.ipynb** - Factor engine fundamentals
   - Creating factor expressions with DSL
   - Running factors with DebugBackend
   - Understanding output format

2. **02_data_preprocessing.ipynb** - Real data handling
   - Data quality checks and cleaning
   - Universe filtering
   - Custom DataSource implementation

3. **03_factor_optimization.ipynb** - Parameter tuning
   - Grid search for optimal parameters
   - IC metrics and stability testing
   - Walk-forward validation
   - Overfitting detection

4. **04_multi_factor_selection.ipynb** - Factor combination
   - Correlation analysis
   - IC-weighted combination
   - Incremental IC analysis
   - Portfolio construction

5. **05_full_workflow.ipynb** - End-to-end production workflow
   - Research hypothesis to production
   - Rigorous validation framework
   - Monitoring setup
   - Documentation templates

## Examples

Real-world implementation patterns:

- **momentum_strategy.ipynb** - Cross-sectional momentum with universe filtering, rebalancing, and performance analysis
- **backend_comparison.ipynb** - Pandas vs Polars vs DuckDB performance benchmarking

## Utilities

`notebooks/utils/notebook_helpers.py` provides:

- `ProgressBar` - Visual progress tracking
- `display_factor_result()` - Styled factor output
- `display_metrics()` - Metric cards with color coding
- `display_dataframe_summary()` - DataFrame statistics
- `display_warning()` / `display_success()` - Status messages
- `NotebookTimer` - Context manager for timing operations

## Quick Start

```python
import sys
sys.path.insert(0, '/home/shw/quant_projects/factor_engine')
sys.path.insert(0, '/home/shw/quant_projects/notebooks')

from utils import display_success, display_metrics, NotebookTimer
from api import col, rank, ts_mean, Factor
from backend.debug_backend import DebugBackend
from runtime.engine import FactorEngine

# Create and run a factor
expr = rank(ts_mean(col('close'), 20))
factor = Factor('momentum_20', expr, '1d', 'equities')

engine = FactorEngine(backend=DebugBackend(), data_source=...)
result = engine.run(factor)

display_success("Factor calculated successfully")
```

## Styling

Notebooks use a consistent dark theme design system:
- Near-black surfaces with tonal gradients
- Cyan accent (#38BDF8) for primary highlights
- Emerald (#6EE7B7) for success states
- Amber (#E9A568) for warnings
- Fluid typography with clamp() sizing
- Token-based spacing and radius

## Running Notebooks

All notebooks are self-contained and executable:

```bash
cd /home/shw/quant_projects/notebooks
jupyter notebook
```

Or use JupyterLab:

```bash
jupyter lab
```

## Next Steps

1. Start with Tutorial 01 for factor engine basics
2. Work through tutorials 02-05 progressively
3. Review examples for real-world patterns
4. Build your own factors following the workflows
5. Use the utilities for consistent notebook styling

## Requirements

- factor_engine installed and accessible
- Python 3.8+
- pandas, numpy
- IPython for display utilities (optional, degrades gracefully)

## Contributing

When adding new notebooks:
- Follow the established naming convention
- Include clear learning objectives
- Add comprehensive comments
- Use the provided utility functions for consistent styling
- Ensure notebooks are fully executable without external dependencies
