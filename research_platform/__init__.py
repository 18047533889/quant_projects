"""research_platform: pure-stdlib research governance / artifact layer.

Only top-level modules are exported as submodules; import them explicitly:

    from research_platform.artifacts import DataSnapshotArtifact
    from research_platform.firewall import AlphaGenerationFirewall
"""

__version__ = "0.1.0"

from . import artifacts as artifacts
from . import campaign as campaign
from . import firewall as firewall
from . import graph as graph
from . import health as health
from . import scorecard as scorecard

__all__ = [
    "artifacts",
    "campaign",
    "firewall",
    "graph",
    "health",
    "scorecard",
]
