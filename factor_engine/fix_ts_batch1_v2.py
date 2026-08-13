#!/usr/bin/env python3
"""
Fix all broken method signatures in ts_batch1.py
"""

# Read the file
with open('cleaned_operators/polars_native/ts_batch1.py', 'r') as f:
    lines = f.readlines()

fixed_lines = []
for i, line in enumerate(lines):
    # Check if this line matches the pattern of a broken method signature
    # It should have some indentation, then start with (self,
    stripped = line.lstrip()
    if stripped.startswith('(self,') and not line.strip().startswith('#'):
        # This is a broken method definition
        # Get the indentation from the previous non-empty line
        indent = '    '  # Default 4 spaces for method definition
        fixed_line = indent + 'def _calculate_series' + line.lstrip()
        fixed_lines.append(fixed_line)
        print(f"Fixed line {i+1}: {line.rstrip()} -> {fixed_line.rstrip()}")
    else:
        fixed_lines.append(line)

# Write back
with open('cleaned_operators/polars_native/ts_batch1.py', 'w') as f:
    f.writelines(fixed_lines)

print("\nDone!")
