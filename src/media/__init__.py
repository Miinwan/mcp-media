"""Media: on-device media intelligence for the Premiere Pro and After Effects MCP servers.

Runs standalone. Results and transcription caches live in a hidden `.media/` folder next to the footage; downloaded
models live in the Media data folder.
"""

import os
from pathlib import Path

__version__ = "0.1.0"
THU_MUC = ".media"


def thu_muc_ket_qua(source: str | Path, cache_goc: str | Path | None = None) -> Path:
    """`<cache_goc or source>/.media`."""
    return Path(cache_goc or source).resolve() / THU_MUC


def thu_muc_du_lieu() -> Path:
    """Downloaded models and environments (override with MEDIA_DU_LIEU)."""
    goc = os.environ.get("MEDIA_DU_LIEU")
    return Path(goc).expanduser().resolve() if goc else Path.home() / "Library" / "Application Support" / "MCP Adobe" / "Media"
