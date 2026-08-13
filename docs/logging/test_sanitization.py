"""
Test script to verify sensitive data sanitization works correctly.
"""
import sys
sys.path.insert(0, '/home/shw/quant_projects')

from dataaccess.logging_config import sanitize_message, sanitize_dict

# Test message sanitization
test_messages = [
    'password="secret123"',
    'api_key: sk-1234567890',
    'token=abc123xyz',
    'AWS_SECRET_ACCESS_KEY="AKIAIOSFODNN7EXAMPLE"',
    'Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9',
]

print("Testing message sanitization:")
print("-" * 60)
for msg in test_messages:
    sanitized = sanitize_message(msg)
    print(f"Original:  {msg}")
    print(f"Sanitized: {sanitized}")
    print()

# Test dictionary sanitization
test_dict = {
    'username': 'admin',
    'password': 'super_secret',
    'api_key': 'sk-1234567890abcdef',
    'database': 'mydb',
    'token': 'bearer_token_here',
    'nested': {
        'secret': 'nested_secret',
        'normal_field': 'normal_value'
    },
    'list_field': [
        {'credential': 'cred123', 'name': 'item1'},
        'plain_string'
    ]
}

print("\nTesting dictionary sanitization:")
print("-" * 60)
print("Original dict:")
import json
print(json.dumps(test_dict, indent=2))

sanitized = sanitize_dict(test_dict)
print("\nSanitized dict:")
print(json.dumps(sanitized, indent=2))

# Verify sensitive fields were redacted
print("\n" + "=" * 60)
print("VERIFICATION:")
assert sanitized['password'] == '***REDACTED***', "Password not redacted!"
assert sanitized['api_key'] == '***REDACTED***', "API key not redacted!"
assert sanitized['token'] == '***REDACTED***', "Token not redacted!"
assert sanitized['nested']['secret'] == '***REDACTED***', "Nested secret not redacted!"
assert sanitized['username'] == 'admin', "Non-sensitive field was changed!"
print("✓ All sensitive data properly sanitized")
print("✓ Non-sensitive data preserved")
