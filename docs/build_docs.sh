#!/bin/bash
# Build documentation script for Quant Projects

set -e

echo "======================================"
echo "Building Quant Projects Documentation"
echo "======================================"
echo ""

# Check if we're in the docs directory
if [ ! -f "source/conf.py" ]; then
    echo "Error: Must run from docs/ directory"
    exit 1
fi

# Check dependencies
echo "Checking dependencies..."
python3 -c "import sphinx" 2>/dev/null || {
    echo "Error: Sphinx not installed"
    echo "Install with: pip install sphinx sphinx-rtd-theme sphinx-autodoc-typehints myst-parser"
    exit 1
}

# Clean previous build
echo "Cleaning previous build..."
make clean

# Build HTML documentation
echo ""
echo "Building HTML documentation..."
make html

# Check build status
if [ $? -eq 0 ]; then
    echo ""
    echo "======================================"
    echo "Documentation built successfully!"
    echo "======================================"
    echo ""
    echo "View documentation:"
    echo "  file://$(pwd)/build/html/index.html"
    echo ""
    echo "Or serve locally:"
    echo "  cd build/html && python3 -m http.server 8000"
    echo "  Then open: http://localhost:8000"
    echo ""
else
    echo ""
    echo "Error: Documentation build failed"
    exit 1
fi
