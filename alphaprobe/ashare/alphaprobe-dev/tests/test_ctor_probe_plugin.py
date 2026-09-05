"""Temp pytest plugin to introspect SchemaRegistry ctor under pytest."""
import os
import alphaprobe.research_space.registry as m

_orig = m.SchemaRegistry.__post_init__


def _dbg(self):
    dbp = self.db_path
    before = os.path.isdir(dbp) if dbp else None
    _orig(self)
    print("CTOR db_path=%r isdir_before=%s isdir_after=%s isfile_after=%s"
          % (dbp, before, os.path.isdir(self.db_path), os.path.isfile(self.db_path)))


m.SchemaRegistry.__post_init__ = _dbg

import alphaprobe.research_space.contracts as _c
_orig_plan_ctor = _c.SchemaPlan.__init__


def _plan_dbg(self, *a, **k):
    print("PLAN CTOR called with db_path env:", os.environ.get("ALPHAPROBE_SCHEMA_DB"))
    return _orig_plan_ctor(self, *a, **k)


_c.SchemaPlan.__init__ = _plan_dbg
