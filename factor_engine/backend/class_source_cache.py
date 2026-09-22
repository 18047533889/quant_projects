"""Bounded reuse of inspect source text, not semantic identity payloads."""
from functools import lru_cache
import inspect
import os


@lru_cache(maxsize=512)
def _read_class_source(cls, module, qualname, version, reader):
    # Exceptions never enter lru_cache: a transient failure must be retried.
    return reader(cls)


def class_source(cls):
    """Return inspect.getsource unchanged, reusing unchanged file-backed classes.

    Custom metaclasses and dynamic/loader-only sources use the original path.
    No disk cache and no caching of mutable defaults, closure cells or hashes.
    """
    resolved = inspect.unwrap(cls)
    if type(resolved) is not type:
        return inspect.getsource(cls)
    filename = inspect.getsourcefile(resolved)
    if not filename:
        return inspect.getsource(cls)
    try:
        stat = os.stat(filename)
    except OSError:
        return inspect.getsource(cls)
    version = (filename, stat.st_dev, stat.st_ino, stat.st_size,
               stat.st_mtime_ns, stat.st_ctime_ns)
    return _read_class_source(resolved, resolved.__module__, resolved.__qualname__,
                              version, inspect.getsource)

