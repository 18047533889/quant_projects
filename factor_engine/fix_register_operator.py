import re
import os

def fix_register_operator_calls():
    root_dir = 'cleaned_operators'
    pattern = re.compile(r'register_operator\((.*?)\)', re.DOTALL)

    for dirpath, _, filenames in os.walk(root_dir):
        if 'polars_native' in dirpath:
            continue
        for filename in filenames:
            if filename.endswith('.py'):
                filepath = os.path.join(dirpath, filename)
                with open(filepath, 'r') as f:
                    content = f.read()

                matches = list(pattern.finditer(content))
                if not matches:
                    continue

                modified_content = content
                offset = 0
                for match in matches:
                    full_match = match.group(0)
                    # Check if it has source="factor_dsl_np" and no replace=True
                    if 'source="factor_dsl_np"' in full_match and 'replace=True' not in full_match:
                        # Add replace=True, expected_old_source="factor_dsl_np", replacement_reason="Consolidating polars native operators"
                        # We need to find the closing parenthesis and insert before it
                        # But simpler: insert after the opening parenthesis if it's the start of args
                        # Actually, let's just append to the args before the closing paren
                        new_match = full_match[:-1] + ', replace=True, expected_old_source="factor_dsl_np", replacement_reason="Consolidating polars native operators")'
                        modified_content = modified_content.replace(full_match, new_match)

                if modified_content != content:
                    with open(filepath, 'w') as f:
                        f.write(modified_content)
                    print(f"Fixed {filepath}")

if __name__ == "__main__":
    fix_register_operator_calls()
