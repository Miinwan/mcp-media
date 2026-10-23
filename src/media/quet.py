"""Quét source bằng ffprobe; nhận diện file theo nội dung (mã băm) thay vì đường dẫn."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path

DUOI_VIDEO = {".mp4", ".mov", ".mxf"}
KICH_THUOC_KHOI = 1024 * 1024


def tim_clip(thu_muc: str | Path) -> list[Path]:
    """Tìm video đệ quy, bỏ đường dẫn ẩn (gồm `.media/`) và proxy."""
    goc = Path(thu_muc)
    if not goc.is_dir():
        raise ValueError(f"Không phải thư mục source: {goc}")
    ket_qua = []
    for hien_tai, thu_muc_con, tep in os.walk(goc):
        thu_muc_con[:] = [x for x in thu_muc_con if not x.startswith(".") and "proxy" not in x.lower()]
        for ten in tep:
            p = Path(hien_tai) / ten
            if ten.startswith(".") or "proxy" in p.stem.lower() or p.suffix.lower() not in DUOI_VIDEO:
                continue
            ket_qua.append(p)
    return sorted(ket_qua)


def ma_bam(tep: str | Path) -> str:
    """SHA-256 của kích thước và ba khối 1 MB ở đầu, giữa, cuối."""
    p = Path(tep)
    kich_thuoc = p.stat().st_size
    bam = hashlib.sha256()
    bam.update(kich_thuoc.to_bytes(8, "big"))
    with p.open("rb") as f:
        for vi_tri in (0, max(0, (kich_thuoc - KICH_THUOC_KHOI) // 2), max(0, kich_thuoc - KICH_THUOC_KHOI)):
            f.seek(vi_tri)
            bam.update(f.read(KICH_THUOC_KHOI))
    return "sha256:" + bam.hexdigest()


def media_id(tep: str | Path) -> str:
    return "m-" + ma_bam(tep).split(":", 1)[1][:12]


def ffprobe(tep: Path) -> dict:
    kq = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(tep)],
                        capture_output=True, text=True)
    if kq.returncode:
        raise ValueError(f"ffprobe không đọc được {tep}: {kq.stderr.strip()}")
    return json.loads(kq.stdout)


def _fps(chuoi: str | None) -> dict | None:
    try:
        f = Fraction(chuoi or "0")
    except (ValueError, ZeroDivisionError):
        return None
    return {"tu": f.numerator, "mau": f.denominator} if f > 0 else None


def xoay(video: dict) -> int:
    for muc in video.get("side_data_list", []):
        if "rotation" in muc:
            return round(float(muc["rotation"]) / 90) * 90 % 360
    try:
        return round(float(video.get("tags", {}).get("rotate", 0)) / 90) * 90 % 360
    except ValueError:
        return 0


def doc_media(tep: str | Path) -> tuple[str, dict]:
    """(media_id, thông số) với đúng một lần ffprobe."""
    p = Path(tep)
    du_lieu = ffprobe(p)
    video = next((x for x in du_lieu.get("streams", []) if x.get("codec_type") == "video"), None)
    if video is None:
        raise ValueError(f"Không có luồng hình: {p}")
    audio = next((x for x in du_lieu["streams"] if x.get("codec_type") == "audio"), None)
    bam = ma_bam(p)
    fps = _fps(video.get("r_frame_rate"))
    media = {
        "ma_bam": bam, "ten_file": p.name, "duong_dan_gan_nhat": str(p.resolve()),
        "rong": int(video["width"]), "cao": int(video["height"]), "xoay": xoay(video),
        "am_thanh": {"kieu": "trong_file" if audio else "khong_co", "so_kenh": int(audio.get("channels", 0)) if audio else 0},
    }
    if fps:
        media["fps"] = fps
    so_frame = video.get("nb_frames")
    if so_frame and str(so_frame).isdigit():
        media["so_frame"] = int(so_frame)
    elif fps:
        try:
            media["so_frame"] = max(0, round(float(video.get("duration") or du_lieu["format"]["duration"]) * fps["tu"] / fps["mau"]))
        except (KeyError, ValueError, TypeError):
            pass
    return "m-" + bam.split(":", 1)[1][:12], media


def quet_files(cac_tep: list[Path], *, workers: int | None = None) -> list[tuple[str, dict]]:
    """Quét song song; giữ thứ tự đầu vào."""
    with ThreadPoolExecutor(max_workers=workers or min(8, max(1, len(cac_tep)))) as pool:
        return list(pool.map(doc_media, cac_tep))


def thoi_luong(tep: str | Path) -> float | None:
    kq = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(tep)],
                        capture_output=True, text=True)
    try:
        return float(kq.stdout.strip())
    except ValueError:
        return None


def tim_proxy_premiere(tep: str | Path, source: str | Path | None = None) -> Path | None:
    """Proxy Premiere `Proxies/<tên>_Proxy.mov` (MCP Premiere `create_proxies`) cạnh file hoặc ở gốc thư mục source.

    Chỉ nhận khi thời lượng khớp file gốc (lệch tối đa 0,1 giây) và có cả hình lẫn tiếng."""
    p = Path(tep)
    cho = [p.parent / "Proxies"] + ([Path(source) / "Proxies"] if source else [])
    goc = None
    for thu_muc in dict.fromkeys(cho):
        for duoi in (".mov", ".MOV", ".mp4", ".MP4"):
            ung_vien = thu_muc / f"{p.stem}_Proxy{duoi}"
            if not ung_vien.is_file():
                continue
            goc = goc if goc is not None else thoi_luong(p)
            tl = thoi_luong(ung_vien)
            if goc is None or tl is None or abs(tl - goc) > 0.1:
                continue
            luong = {x.get("codec_type") for x in ffprobe(ung_vien).get("streams", [])}
            if {"video", "audio"} <= luong:
                return ung_vien
    return None
