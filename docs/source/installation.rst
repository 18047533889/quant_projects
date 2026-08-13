.. _installation:

Installation
============

This guide covers installation of the Quant Projects platform.

Requirements
------------

System Requirements
~~~~~~~~~~~~~~~~~~~

* Python 3.10 or higher
* Linux (Ubuntu 20.04+) or macOS
* 16GB+ RAM recommended
* 50GB+ disk space for data cache

Python Dependencies
~~~~~~~~~~~~~~~~~~~

Core dependencies:

* numpy >= 1.24
* pandas >= 2.0
* polars >= 0.18
* duckdb >= 0.8
* pyarrow >= 12.0
* scipy >= 1.10
* scikit-learn >= 1.3

Installation Methods
--------------------

Development Installation
~~~~~~~~~~~~~~~~~~~~~~~~

For active development, install in editable mode:

.. code-block:: bash

   # Clone the repository
   cd /path/to/quant_projects

   # Create virtual environment
   python3 -m venv .venv
   source .venv/bin/activate

   # Install all modules in editable mode
   pip install -e ./dataaccess
   pip install -e ./factor_engine
   pip install -e ./factor_preprocess
   pip install -e ./factor_optimizer
   pip install -e ./factor_assets
   pip install -e ./quant_evaluator
   pip install -e ./research_control

Individual Module Installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Install only specific modules:

.. code-block:: bash

   # Just the factor engine
   pip install -e ./factor_engine

   # Engine + data access
   pip install -e ./dataaccess
   pip install -e ./factor_engine

Configuration
-------------

Data Sources
~~~~~~~~~~~~

Configure data source credentials:

.. code-block:: bash

   # Create config directory
   mkdir -p ~/quant_projects/config

   # Add COS credentials (for cloud storage)
   cat > ~/quant_projects/config/cos_config.yaml << EOF
   bucket: your-bucket-name
   region: your-region
   secret_id: YOUR_SECRET_ID
   secret_key: YOUR_SECRET_KEY
   EOF

Market Data Setup
~~~~~~~~~~~~~~~~~

Initialize market data catalogs:

.. code-block:: python

   from dataaccess import initialize_catalogs

   # Initialize A-share catalog
   initialize_catalogs(market='ashare')

   # Initialize US catalog
   initialize_catalogs(market='us')

Cache Configuration
~~~~~~~~~~~~~~~~~~~

Configure cache directories:

.. code-block:: bash

   export QUANT_CACHE_DIR=/path/to/cache
   export QUANT_DATA_DIR=/path/to/data

Verification
------------

Verify Installation
~~~~~~~~~~~~~~~~~~~

Run basic smoke tests:

.. code-block:: python

   # Test data access
   from dataaccess import get_market_data
   df = get_market_data('000001.SZ', start='2024-01-01', end='2024-01-31')
   print(df.head())

   # Test factor engine
   from factor_engine import FactorEngine
   engine = FactorEngine()
   print(f"Loaded {len(engine.list_operators())} operators")

Run Test Suite
~~~~~~~~~~~~~~

Run the full test suite:

.. code-block:: bash

   # Run all tests
   pytest

   # Run specific module tests
   pytest dataaccess/tests
   pytest factor_engine/tests

   # Run with coverage
   pytest --cov=dataaccess --cov=factor_engine

Build Documentation
~~~~~~~~~~~~~~~~~~~

Build and view documentation locally:

.. code-block:: bash

   cd docs
   make html
   # Open build/html/index.html in browser

Troubleshooting
---------------

Import Errors
~~~~~~~~~~~~~

If you encounter import errors:

.. code-block:: bash

   # Ensure modules are in Python path
   export PYTHONPATH=/path/to/quant_projects:$PYTHONPATH

   # Or install in editable mode
   pip install -e ./dataaccess
   pip install -e ./factor_engine

Missing Dependencies
~~~~~~~~~~~~~~~~~~~~

Install missing optional dependencies:

.. code-block:: bash

   # For performance profiling
   pip install memory_profiler line_profiler

   # For visualization
   pip install matplotlib seaborn plotly

   # For notebook support
   pip install jupyter ipykernel

DuckDB Issues
~~~~~~~~~~~~~

If DuckDB fails to load:

.. code-block:: bash

   # Reinstall DuckDB
   pip uninstall duckdb
   pip install duckdb --no-cache-dir

Memory Issues
~~~~~~~~~~~~~

For large computations, increase memory limits:

.. code-block:: python

   from factor_engine import configure_resources

   configure_resources(
       max_memory_gb=32,
       enable_spill=True,
       spill_dir='/path/to/fast/disk'
   )

Next Steps
----------

* :ref:`quickstart` - Run your first factor computation
* :ref:`usage-examples` - Detailed usage examples
* :ref:`api-reference` - API documentation
