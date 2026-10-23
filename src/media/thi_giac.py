"""Fast on-device per-sample frame scoring, no tokens spent.

Reads the proxy (if any) at 2 samples per second, small frames 320 px wide: sharpness (Laplacian variance), motion
between samples (mean difference, both shake and visual pace), mean brightness, clipped highlight and crushed black
ratios, saturation, face count (YuNet, MIT, a few hundred KB, downloaded on demand). Cached by file hash.
"""

import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

PHIEN_BAN = 1
MAU_MOI_GIAY = 2
RONG = 320
_MAT = None


NGUON_YUNET = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"


def tep_yunet() -> Path:
    from media import thu_muc_du_lieu
    return thu_muc_du_lieu() / "models" / "face_yunet.onnx"


def tai_yunet() -> Path:
    """Download YuNet (opencv_zoo, MIT, about 230 KB) to the Media data folder."""
    import urllib.request
    dich = tep_yunet()
    if not dich.is_file():
        dich.parent.mkdir(parents=True, exist_ok=True)
        tam = dich.with_suffix(".tam")
        urllib.request.urlretrieve(NGUON_YUNET, tam)
        tam.replace(dich)
    return dich


def _tep_yunet() -> Path | None:
    return tep_yunet() if tep_yunet().is_file() else None


def _bo_do_mat(rong: int, cao: int):
    """YuNet built for the image size; returns None without the model (face count skipped)."""
    global _MAT
    tep = _tep_yunet()
    if tep is None:
        return None
    if _MAT is None:
        _MAT = cv2.FaceDetectorYN.create(str(tep), "", (rong, cao), 0.7)
    _MAT.setInputSize((rong, cao))
    return _MAT


def doc_khung(duong_dan: str, mau_moi_giay: int = MAU_MOI_GIAY, rong: int = RONG) -> np.ndarray:
    """Small RGB frames evenly spaced in time, via ffmpeg (handles 10-bit log and camera HEVC)."""
    cao = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                          "-of", "csv=p=0", duong_dan], capture_output=True, text=True, check=True).stdout.split(",")
    w, h = int(cao[0]), int(cao[1])
    h2 = max(2, round(h * rong / w) // 2 * 2)
    ra = subprocess.run(["ffmpeg", "-v", "error", "-i", duong_dan, "-vf", f"fps={mau_moi_giay},scale={rong}:{h2},format=rgb24",
                         "-f", "rawvideo", "-"], capture_output=True, check=True)
    return np.frombuffer(ra.stdout, np.uint8).reshape(-1, h2, rong, 3)


def cham_khung(anh: np.ndarray, mau_moi_giay: int = MAU_MOI_GIAY) -> list[dict]:
    """Raw per-sample metrics (not normalised across clips)."""
    kq, truoc = [], None
    mat = _bo_do_mat(anh.shape[2], anh.shape[1]) if len(anh) else None
    for i, a in enumerate(anh):
        xam = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(a, cv2.COLOR_RGB2HSV)
        so_mat = None
        if mat is not None:
            _, mat_tim = mat.detect(cv2.cvtColor(a, cv2.COLOR_RGB2BGR))
            so_mat = 0 if mat_tim is None else len(mat_tim)
        kq.append({
            "t": round(i / mau_moi_giay, 3),
            "net": round(float(cv2.Laplacian(xam, cv2.CV_64F).var()), 2),
            "chuyen_dong": round(float(np.abs(xam.astype(np.int16) - truoc).mean()), 3) if truoc is not None else 0.0,
            "sang": round(float(xam.mean()), 2),
            "chay": round(float((xam >= 250).mean()), 4),
            "den": round(float((xam <= 5).mean()), 4),
            "bao_hoa": round(float(hsv[..., 1].mean()), 2),
            "mat": so_mat,
        })
        truoc = xam.astype(np.int16)
    return kq


def diem_dep(mau: list[dict]) -> list[float]:
    """0-1 score per sample for ranking good shots. Sharpness is normalised per clip (cameras differ)."""
    if not mau:
        return []
    net = np.array([m["net"] for m in mau])
    dong = np.array([m["chuyen_dong"] for m in mau])
    net_c = np.clip(net / (np.percentile(net, 90) or 1), 0, 1)
    rung = np.clip(dong / 25, 0, 1)  # mean difference above 25 grey levels between samples 0.5 s apart means shake or a fast pan
    phoi = np.array([1 - min(1, 4 * m["chay"] + 4 * m["den"] + (abs(m["sang"] - 120) > 90)) for m in mau])
    mau_sac = np.clip(np.array([m["bao_hoa"] for m in mau]) / 120, 0, 1)
    nguoi = np.array([min(1, m["mat"] or 0) for m in mau])
    d = 0.4 * net_c + 0.25 * (1 - rung) + 0.2 * phoi + 0.1 * mau_sac + 0.05 * nguoi
    return [round(float(x), 4) for x in d]


def quet(media_id: str, media: dict, thu_muc_cache: Path, doc=doc_khung) -> dict:
    """Scan one media file (proxy if available), cached by hash. Returns {mau: [...], diem: [...]}."""
    tep_cache = thu_muc_cache / f"{media_id}.json"
    if tep_cache.is_file():
        cu = json.loads(tep_cache.read_text())
        if cu.get("phien_ban") == PHIEN_BAN and cu.get("ma_bam") == media["ma_bam"]:
            return cu
    nguon = media.get("proxy") or media["duong_dan_gan_nhat"]
    mau = cham_khung(doc(nguon))
    kq = {"phien_ban": PHIEN_BAN, "ma_bam": media["ma_bam"], "mau_moi_giay": MAU_MOI_GIAY, "mau": mau, "diem": diem_dep(mau)}
    thu_muc_cache.mkdir(parents=True, exist_ok=True)
    tam = tep_cache.with_suffix(".tam")
    tam.write_text(json.dumps(kq, ensure_ascii=False))
    tam.replace(tep_cache)
    return kq
