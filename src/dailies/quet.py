"""Scan sources with ffprobe; identify files by content hash rather than path."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

DUOI_VIDEO = {".mp4", ".mov", ".mxf"}
KICH_THUOC_KHOI = 1024 * 1024


def tim_clip(thu_muc: str | Path) -> list[Path]:
    """Find videos recursively, skipping hidden paths (including `.dailies/`) and proxies."""
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
    """SHA-256 of the size plus three 1 MB blocks at start, middle and end."""
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


def ffprobe(tep: str | Path) -> dict:
    """ffprobe streams and format, remembered per file (path, size, modification time) for this process: a Sony
    MP4 takes about 0.2 s to probe and one run asks several times. Treat the result as read-only."""
    p = Path(tep)
    try:
        st = p.stat()
    except OSError:
        return _ffprobe(str(p), -1, -1)
    return _ffprobe(str(p.resolve()), st.st_size, st.st_mtime_ns)


@lru_cache(maxsize=4096)
def _ffprobe(duong: str, _co: int, _sua: int) -> dict:
    kq = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", duong],
                        capture_output=True, text=True)
    if kq.returncode:
        raise ValueError(f"ffprobe không đọc được {duong}: {kq.stderr.strip()}")
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


def kich_thuoc_hien(tep: str | Path) -> tuple[int, int]:
    """Displayed width and height of the first video stream: pixel aspect (anamorphic) and rotation applied."""
    video = next((s for s in ffprobe(tep).get("streams", []) if s.get("codec_type") == "video"), None)
    if video is None:
        raise ValueError(f"Không có luồng hình: {tep}")
    w, h = int(video["width"]), int(video["height"])
    try:
        par = Fraction((video.get("sample_aspect_ratio") or "1:1").replace(":", "/"))
    except (ValueError, ZeroDivisionError):
        par = Fraction(1)
    if par > 0 and par != 1:
        w = round(w * par)
    return (h, w) if xoay(video) in (90, 270) else (w, h)


def doc_media(tep: str | Path) -> tuple[str, dict]:
    """(media_id, properties) with exactly one ffprobe call."""
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
    """Scan in parallel; keeps input order."""
    with ThreadPoolExecutor(max_workers=workers or min(8, max(1, len(cac_tep)))) as pool:
        return list(pool.map(doc_media, cac_tep))


def thoi_luong(tep: str | Path) -> float | None:
    try:
        return float(ffprobe(tep)["format"]["duration"])
    except (ValueError, KeyError, TypeError):
        return None


def tim_proxy(tep: str | Path, source: str | Path | None = None, chi_dinh: str | Path | None = None) -> Path | None:
    """Proxy for one clip. Proxies are made in Premiere before Dailies is called; Dailies never creates them.

    `chi_dinh`: proxy path the MCP read from the Premiere project (error if the duration is wrong). Otherwise look in
    Premiere's default location, `Proxies/<name>_Proxy.mov` next to the file or at the source folder root. Only accept
    a proxy matching the original duration (within 0.1 s) with all of its streams (video, plus audio if the original has it)."""
    p = Path(tep)
    goc = thoi_luong(p)
    can = {x.get("codec_type") for x in ffprobe(p).get("streams", [])} & {"video", "audio"}

    def hop_le(ung_vien: Path) -> bool:
        tl = thoi_luong(ung_vien) if ung_vien.is_file() else None
        if goc is None or tl is None or abs(tl - goc) > 0.1:
            return False
        return can <= {x.get("codec_type") for x in ffprobe(ung_vien).get("streams", [])}

    if chi_dinh:
        cd = Path(chi_dinh).expanduser().resolve()
        if not hop_le(cd):
            raise ValueError(f"Proxy {cd} không đọc được, thiếu hình hoặc tiếng, hay lệch thời lượng với {p.name}")
        return cd
    for thu_muc in dict.fromkeys([p.parent / "Proxies"] + ([Path(source) / "Proxies"] if source else [])):
        for duoi in (".mov", ".MOV", ".mp4", ".MP4"):
            if hop_le(thu_muc / f"{p.stem}_Proxy{duoi}"):
                return thu_muc / f"{p.stem}_Proxy{duoi}"
    return None


class ThieuProxy(ValueError):
    """Clips without a proxy. Proxies are made in Premiere first; Dailies never creates them or decodes 4K."""

    def __init__(self, cac_file: list[str]):
        super().__init__("Chưa có proxy cho: " + ", ".join(Path(x).name for x in cac_file)
                         + ". Tạo proxy trong Premiere rồi chạy lại (hoặc --cho-phep-goc để đọc file gốc)")
        self.cac_file = cac_file


def nguon_vao(thu_muc: str | Path | None = None, danh_sach: str | Path | None = None,
              cho_phep_goc: bool = False) -> tuple[Path, list[Path], dict[str, Path | None]]:
    """(root folder, clips, proxy per clip) for one run.

    `danh_sach`: JSON `{"clip": [{"file": ..., "proxy": ...}]}` the MCP read from the Premiere project after culling;
    the root is the clips' common folder (where `.dailies/` goes). Without a list, take every clip in `thu_muc`.
    Clips with no proxy raise `ThieuProxy`, unless `cho_phep_goc` (proxy is None, read the original)."""
    chi_dinh: dict[str, str | None] = {}
    if danh_sach:
        cac_tep = []
        for c in json.loads(Path(danh_sach).read_text(encoding="utf-8")).get("clip", []):
            tep = Path(c["file"]).expanduser().resolve()
            if not tep.is_file():
                raise ValueError(f"Không có file: {tep}")
            if tep not in cac_tep:
                cac_tep.append(tep)
                chi_dinh[str(tep)] = c.get("proxy")
        if not cac_tep:
            raise ValueError("Danh sách không có clip nào")
        goc = Path(os.path.commonpath([str(p.parent) for p in cac_tep]))
    elif thu_muc:
        goc = Path(thu_muc).resolve()
        cac_tep = tim_clip(goc)
        if not cac_tep:
            raise ValueError("Không tìm thấy MP4/MOV/MXF")
    else:
        raise ValueError("Cần thư mục source hoặc --danh-sach")
    with ThreadPoolExecutor(max_workers=min(8, len(cac_tep))) as pool:
        tim = list(pool.map(lambda t: tim_proxy(t, goc, chi_dinh.get(str(t))), cac_tep))
    proxy = dict(zip(map(str, cac_tep), tim))
    thieu = [t for t, p in proxy.items() if p is None]
    if thieu and not cho_phep_goc:
        raise ThieuProxy(thieu)
    return goc, cac_tep, proxy

