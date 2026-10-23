"""Media: năng lực media cho MCP Premiere và After Effects (brief MCP Adobe, giai đoạn media).

Dùng lại phần nghe, quét, chấm hình của W.cut làm thư viện (không viết hai lần); kết quả gom trong một thư mục ẩn
`.media/` cạnh footage (gồm cả cache nghe và proxy của W.cut ở `.media/.wcut/`).
"""

import os
from pathlib import Path

__version__ = "0.1.0"
THU_MUC = ".media"


def thu_muc_ket_qua(source: str | Path, cache_goc: str | Path | None = None) -> Path:
    """`<cache_goc hoặc source>/.media`."""
    return Path(cache_goc or source).resolve() / THU_MUC


def thu_muc_du_lieu() -> Path:
    """Model và môi trường tải về (biến MEDIA_DU_LIEU để đổi chỗ)."""
    goc = os.environ.get("MEDIA_DU_LIEU")
    return Path(goc).expanduser().resolve() if goc else Path.home() / "Library" / "Application Support" / "MCP Adobe" / "Media"
