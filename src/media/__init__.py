"""Media: năng lực media cho MCP Premiere và After Effects (brief MCP Adobe, giai đoạn media).

Tự chạy độc lập. Kết quả, cache nghe, proxy gom trong một thư mục ẩn `.media/` cạnh footage; model tải về nằm ở thư mục
dữ liệu Media.
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
