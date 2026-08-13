#!/usr/bin/env python3
"""
Fix metadata integration for all Polars native operators in ts_batch1.py
"""

import re

# Read the file
with open('cleaned_operators/polars_native/ts_batch1.py', 'r') as f:
    content = f.read()

# Pattern to find lines that start with indentation + (self, ... without def _calculate_series
# These are broken method definitions
lines = content.split('\n')
fixed_lines = []
i = 0

while i < len(lines):
    line = lines[i]

    # Check if this line is a broken method signature: starts with spaces, then (self,
    if re.match(r'^(\s+)\(self,', line):
        # This is a broken method definition - add def _calculate_series before it
        indent = re.match(r'^(\s+)', line).group(1)
        fixed_lines.append(f'{indent}def _calculate_series(self,{line[len(indent)+6:]}')
    else:
        fixed_lines.append(line)

    i += 1

# Write back
content = '\n'.join(fixed_lines)

# Now fix operators missing metadata attribute entirely
# Find operators that have a docstring but no metadata
operators_needing_metadata = []

# Pattern: class name, docstring, then immediately def _calculate_series (no metadata)
pattern = r'(@register_operator\([^)]+\)\nclass (\w+)\(SeriesOperator\):\n    """([^"]+)"""\n    def _calculate_series)'
matches = re.finditer(pattern, content)

for match in matches:
    decorator = match.group(0).split('\n')[0]
    class_name = match.group(2)
    description = match.group(3)

    # Extract name from decorator
    name_match = re.search(r'name="([^"]+)"', decorator)
    if name_match:
        op_name = name_match.group(1)
        print(f"Found operator without metadata: {class_name} ({op_name})")

# Write the fixed content
with open('cleaned_operators/polars_native/ts_batch1.py', 'w') as f:
    f.write(content)

print("Fixed broken method signatures!")
