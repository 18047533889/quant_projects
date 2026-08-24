====================================================================
ARCHIVED — NON-CANONICAL COPY OF THE OPERATOR RUNTIME LIBRARY
====================================================================

This directory is the ARCHIVED duplicate of the canonical operator
runtime library.  It is NOT the production runtime and MUST NOT be
imported, packaged, or treated as source authority.

  Canonical location (production runtime + evidence hashing authority):
      <repo_root>/cleaned_operators/
      (i.e. /home/sunhaiwei/quant_projects/cleaned_operators/)

  This archive:
      <repo_root>/factor_engine/cleaned_operators_archived/

Reason for archiving (P0-7 source-authority split):
--------------------------------------------------------------------
The repo previously contained TWO copies of the operator library:

  * <repo_root>/cleaned_operators/        (canonical, 459 tracked files)
  * factor_engine/cleaned_operators/      (duplicate, 451 tracked files)

Both claimed to be the "唯一 production runtime 层" and both shipped an
``__init__.py`` (blob SHAs differed).  The release/runtime wiring
consistently points at the ROOT copy:

  - ``evidence/current.py:_module_source_digest()`` resolves operator
    modules as ``<repo>/cleaned_operators/<rest>.py``
    (``_REPO_ROOT / "cleaned_operators"``).
  - The factor-engine wheel is built from the repo ROOT and packages
    ``factor_engine.cleaned_operators*`` — a stale ``include`` in
    ``pyproject.toml`` that this archive makes inert (nothing named
    ``factor_engine.cleaned_operators`` exists to be found by
    ``setuptools.packages.find``).

All factor-engine code imports the TOP-LEVEL ``cleaned_operators``
package (resolved to the ROOT copy on sys.path); there is no import of
``factor_engine.cleaned_operators`` anywhere in the tree.  Therefore the
duplicate was dead weight and is preserved here for archaeology only.

This copy is NOT maintained.  No changes should be made here.  The
production runtime lives at ``<repo_root>/cleaned_operators/``.
====================================================================
