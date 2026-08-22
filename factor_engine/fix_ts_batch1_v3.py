#!/usr/bin/env python3
"""
Fix all broken method signatures in ts_batch1.py
"""

# Read the file
with open('cleaned_operators/polars_native/ts_batch1.py', 'r') as f:
    lines = f.readlines()

fixed_lines = []
fixed_count = 0

for i, line in enumerate(lines):
    # Remove any control characters and check if line contains (self,
    clean_line = ''.join(c for c in line if ord(c) >= 32 or c in '\n\t')

    # Check if after cleaning control chars, the line starts with (self,
    stripped = clean_line.lstrip()
    if stripped.startswith('(self,') and '=' not in line[:line.find('(self,') if '(self,' in line else 0]:
        # This is a broken method definition
        indent = '    '  # Method indentation
        fixed_line = indent + 'def _calculate_series' + clean_line.lstrip()
        fixed_lines.append(fixed_line)
        fixed_count += 1
        print(f"Fixed line {i+1}: {repr(line[:50])} -> def _calculate_series...")
    else:
        # Keep original line but clean control characters
        if line != clean_line:
            print(f"Cleaned control chars from line {i+1}")
        fixed_lines.append(clean_line)

# Write back
with open('cleaned_operators/polars_native/ts_batch1.py', 'w') as f:
    f.writelines(fixed_lines)

print(f"\nFixed {fixed_count} method signatures!")
