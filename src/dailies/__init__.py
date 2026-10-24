"""Dailies: on-device footage intelligence for the Premiere Pro and After Effects MCP servers.

Runs standalone. Results and transcription caches live in a hidden `.dailies/` folder next to the footage; downloaded
models live in the Dailies data folder.
"""

import os
from pathlib import Path

__version__ = "0.1.0"
THU_MUC = ".dailies"


THU_MUC_CU = ".media"  # name before the rename to Dailies (2026-10-24)


def thu_muc_ket_qua(source: str | Path, cache_goc: str | Path | None = None) -> Path:
    """`<cache_goc or source>/.dailies`. A `.media` folder left by older versions is renamed in place, so transcripts
    and scans already made are kept."""
    goc = Path(cache_goc or source).resolve()
    moi, cu = goc / THU_MUC, goc / THU_MUC_CU
    if not moi.exists() and cu.is_dir():
        try:
            cu.rename(moi)
        except OSError:
            return cu
    return moi


def bien_moi_truong(ten: str) -> str | None:
    """`DAILIES_<ten>`, or the older `MEDIA_<ten>`."""
    return os.environ.get(f"DAILIES_{ten}") or os.environ.get(f"MEDIA_{ten}")


def thu_muc_du_lieu() -> Path:
    """Downloaded models and environments (override with DAILIES_DU_LIEU)."""
    goc = bien_moi_truong("DU_LIEU")
    return Path(goc).expanduser().resolve() if goc else Path.home() / "Library" / "Application Support" / "MCP Adobe" / "Dailies"
