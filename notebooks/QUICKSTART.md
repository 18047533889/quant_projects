# Jupyter Notebooks Quick Start

## Installation

Ensure you have Jupyter installed:

```bash
pip install jupyter notebook
# or
pip install jupyterlab
```

## Launch Notebooks

```bash
cd /home/shw/quant_projects/notebooks
jupyter notebook
# or
jupyter lab
```

Your browser will open to the notebook interface.

## Learning Path

### Beginners - Start Here

1. **tutorials/01_basic_factor_evaluation.ipynb**
   - 15 minutes
   - Learn factor creation, engine setup, basic evaluation

### Intermediate

2. **tutorials/02_data_preprocessing.ipynb**
   - 20 minutes
   - Data quality, cleaning, universe filtering

3. **tutorials/03_factor_optimization.ipynb**
   - 30 minutes
   - Grid search, IC analysis, walk-forward validation

4. **tutorials/04_multi_factor_selection.ipynb**
   - 30 minutes
   - Factor combination, correlation analysis, portfolio construction

### Advanced

5. **tutorials/05_full_workflow.ipynb**
   - 45 minutes
   - Complete research-to-production workflow

### Real-World Examples

- **examples/momentum_strategy.ipynb** - Cross-sectional momentum with performance analysis
- **examples/backend_comparison.ipynb** - Pandas vs Polars vs DuckDB benchmarking

## Common Tasks

### Create a Simple Factor

```python
import sys
sys.path.insert(0, '/home/shw/quant_projects/factor_engine')

from api import col, rank, ts_mean, Factor
from backend.debug_backend import DebugBackend
from runtime.engine import FactorEngine
from storage.datasource import DataSource

class DemoDataSource(DataSource):
    def load_column(self, name: str):
        raise NotImplementedError()

expr = rank(ts_mean(col('close'), 20))
factor = Factor('my_factor', expr, '1d', 'equities')

engine = FactorEngine(backend=DebugBackend(), data_source=DemoDataSource())
result = engine.run(factor)
print(result['result'])
```

### Use Display Helpers

```python
import sys
sys.path.insert(0, '/home/shw/quant_projects/notebooks')

from utils import display_success, display_metrics, NotebookTimer

# Time operations
with NotebookTimer("My calculation"):
    result = expensive_operation()

# Display metrics
display_metrics({
    'IC': 0.045,
    'Sharpe': 1.2,
    'Win_Rate': 0.65
}, "Factor Performance")

# Status messages
display_success("Factor calculation complete")
```

## Troubleshooting

### ImportError: No module named 'factor_engine'

Make sure the path is correct:

```python
import sys
sys.path.insert(0, '/home/shw/quant_projects/factor_engine')
```

### Kernel Crashes on Large Datasets

Try reducing dataset size or using DuckDB backend for out-of-core processing.

### Display Helpers Don't Show Styled Output

Install IPython: `pip install ipython`

If still plain text, the helpers degrade gracefully - functionality remains intact.

## Tips

- **Start Small**: Use `n_stocks=50, n_days=100` while developing
- **Save Often**: Jupyter doesn't auto-save, use Ctrl+S frequently
- **Clear Output**: Use Cell → All Output → Clear before sharing notebooks
- **Restart Kernel**: If state gets messy, use Kernel → Restart & Clear Output
- **Export Results**: Use `df.to_csv()` to save factor values

## Next Steps

1. Open `tutorials/01_basic_factor_evaluation.ipynb`
2. Run all cells (Cell → Run All)
3. Experiment by modifying parameters
4. Build your own factors following the patterns

Happy researching!
