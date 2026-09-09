"""Fixed safety bounds for finite-manifest text and dependency metadata."""
from __future__ import annotations


MAX_FACTOR_NAME_UTF8 = 1024
MAX_DEPENDENCY_NAME_UTF8 = 1024
MAX_DEPENDENCIES_PER_FACTOR = 4096
MAX_ERROR_CODE_CHARS = 128
MANIFEST_ROW_FIXED_BYTES = 64
MAX_MANIFEST_ROW_METADATA_BYTES = (
    MAX_FACTOR_NAME_UTF8 + MAX_ERROR_CODE_CHARS + MANIFEST_ROW_FIXED_BYTES
)


def controlled_wave_buffer_bytes(max_items: int, max_definition_bytes: int) -> int:
    """Admission for serialized definition bytes plus bounded UTF-8 metadata.

    This is not a bound on objects produced by pickle decoding. A compact
    pickle may expand into an arbitrarily large Python object graph, which
    requires separate execution-process memory containment.
    """
    if type(max_items) is not int or max_items < 1:
        raise ValueError("max_items must be a positive integer")
    if type(max_definition_bytes) is not int or max_definition_bytes < 1:
        raise ValueError("max_definition_bytes must be a positive integer")
    return max_definition_bytes + max_items * MAX_MANIFEST_ROW_METADATA_BYTES


def bounded_utf8_size(value: object, limit: int) -> int | None:
    """Return UTF-8 size, stopping once limit is exceeded; reject surrogates."""
    if type(limit) is not int or limit < 0 or not isinstance(value, str):
        return None
    size = 0
    for character in value:
        codepoint = ord(character)
        if codepoint <= 0x7f:
            size += 1
        elif codepoint <= 0x7ff:
            size += 2
        elif 0xd800 <= codepoint <= 0xdfff:
            return None
        elif codepoint <= 0xffff:
            size += 3
        else:
            size += 4
        if size > limit:
            return None
    return size
