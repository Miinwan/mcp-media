"""Synthetic clips (test pattern and Vietnamese `say` voice), cached under .cache/mau by parameters."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

THU_MUC_MAU = Path(__file__).resolve().parents[1] / ".cache" / "mau"
CAU_MAC_DINH = ("Xin chào, đây là cảnh quay thử.", "Máy quay đang ghi hình.", "Cảnh cuối đã hoàn thành.")


def _chay(lenh: list[str]) -> None:
    kq = subprocess.run(lenh, capture_output=True, text=True)
    if kq.returncode:
        raise RuntimeError(f"Lệnh lỗi: {' '.join(lenh)}\n{kq.stderr[-2000:]}")


def _khoa(x) -> str:
    return hashlib.sha256(json.dumps(x, ensure_ascii=False).encode()).hexdigest()[:16]


def _giong_viet() -> str | None:
    ds = subprocess.run(["say", "-v", "?"], capture_output=True, text=True).stdout
    dong = next((d for d in ds.splitlines() if re.search(r"\bvi_VN\b", d)), None)
    return dong.split("vi_VN", 1)[0].strip() if dong else None


def _am_thanh(cau: tuple[str, ...], giong: str | None = None) -> Path | None:
    """Sentences joined with 0.8 s of silence; the default voice is the system Vietnamese voice."""
    if not cau or not shutil.which("say"):
        return None
    tep = THU_MUC_MAU / f"tieng-{_khoa((giong, cau))}.m4a"
    if tep.is_file():
        return tep
    giong = giong or _giong_viet()
    lang = THU_MUC_MAU / "lang-08.aiff"
    if not lang.is_file():
        _chay(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "anullsrc=r=22050:cl=mono", "-t", "0.8", "-y", str(lang)])
    ds = []
    for i, c in enumerate(cau):
        phan = THU_MUC_MAU / f"noi-{_khoa((giong, c))}.aiff"
        if not phan.is_file():
            _chay(["say"] + (["-v", giong] if giong else []) + ["-o", str(phan), c])
        ds += [f"file '{phan}'"] + ([f"file '{lang}'"] if i < len(cau) - 1 else [])
    danh_sach = THU_MUC_MAU / f"ds-{_khoa(cau)}.txt"
    danh_sach.write_text("\n".join(ds) + "\n")
    _chay(["ffmpeg", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(danh_sach), "-ar", "44100",
           "-c:a", "aac", "-b:a", "96k", "-y", str(tep)])
    return tep


def sinh_clip(ten: str, rong: int = 1920, cao: int = 1080, giay: float = 12, fps: str = "30000/1001",
              cau: list[str] | tuple[str, ...] | None = None, giong: str | None = None) -> Path:
    """H.264 clip with a testsrc2 picture; `cau=[]` means no audio."""
    cau_chot = tuple(CAU_MAC_DINH if cau is None else cau)
    THU_MUC_MAU.mkdir(parents=True, exist_ok=True)
    dich = THU_MUC_MAU / f"{ten}-{_khoa((ten, rong, cao, giay, fps, cau_chot, giong))}.mp4"
    if dich.is_file():
        return dich
    tieng = _am_thanh(cau_chot, giong)
    lenh = ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", f"testsrc2=size={rong}x{cao}:rate={fps}:duration={giay}"]
    if tieng:
        lenh += ["-i", str(tieng), "-map", "0:v:0", "-map", "1:a:0", "-c:a", "aac", "-af", "apad", "-t", str(giay)]
    tam = dich.with_suffix(".tmp.mp4")
    _chay(lenh + ["-c:v", "h264_videotoolbox", "-b:v", "3M", "-pix_fmt", "nv12", "-y", str(tam)])
    tam.replace(dich)
    return dich


def tao_proxy_premiere(thu_muc: Path) -> list[Path]:
    """Premiere-style proxies (`Proxies/<name>_Proxy.mov`, ProRes, PCM audio) for every clip in the folder."""
    ra = []
    for tep in sorted(Path(thu_muc).glob("*.mp4")):
        dich = tep.parent / "Proxies" / f"{tep.stem}_Proxy.mov"
        dich.parent.mkdir(exist_ok=True)
        _chay(["ffmpeg", "-v", "error", "-i", str(tep), "-c:v", "prores_videotoolbox", "-profile:v", "0",
               "-c:a", "pcm_s16le", "-y", str(dich)])
        ra.append(dich)
    return ra
