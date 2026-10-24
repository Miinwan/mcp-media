"""Fast on-device per-sample frame scoring, no tokens spent.

Frames come from `lap_khung`, which decodes only the frames it returns (PyAV). Reads the proxy (if any) at 2 samples
per second, small frames 320 px wide: sharpness (Laplacian variance), motion
between samples (mean difference, both shake and visual pace), mean brightness, clipped highlight and crushed black
ratios, saturation, face count (YuNet, MIT, a few hundred KB, downloaded on demand). Cached by file hash.
"""

import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

PHIEN_BAN = 2
MAU_MOI_GIAY = 2
RONG = 320
_MAT = None


NGUON_YUNET = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"


def tep_yunet() -> Path:
    from dailies import thu_muc_du_lieu
    return thu_muc_du_lieu() / "models" / "face_yunet.onnx"


def tai_yunet() -> Path:
    """Download YuNet (opencv_zoo, MIT, about 230 KB) to the Dailies data folder."""
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


# All-intra codecs: every packet decodes on its own, so packets between samples are skipped without decoding
INTRA = {"prores", "dnxhd", "mjpeg", "cfhd", "v210", "v410", "r10k", "r210", "dvvideo", "png", "qtrle", "rawvideo",
         "hap", "jpeg2000", "tiff"}


def _ffmpeg_khung(duong_dan: str, mau_moi_giay: float, rong: int):
    """Fallback (codec PyAV cannot open): ffmpeg decodes everything and keeps `mau_moi_giay` frames per second."""
    from dailies.quet import kich_thuoc_hien
    w, h = kich_thuoc_hien(duong_dan)
    h2 = max(2, round(h * rong / w) // 2 * 2)
    ra = subprocess.run(["ffmpeg", "-v", "error", "-i", duong_dan, "-vf", f"fps={mau_moi_giay},scale={rong}:{h2},format=rgb24",
                         "-f", "rawvideo", "-"], capture_output=True, check=True)
    for i, a in enumerate(np.frombuffer(ra.stdout, np.uint8).reshape(-1, h2, rong, 3)):
        yield i / mau_moi_giay, a


def lap_khung(duong_dan: str, mau_moi_giay: float = MAU_MOI_GIAY, rong: int = RONG, *, keyframe_thua_nhat: float | None = None):
    """Yield (seconds, RGB frame `rong` px wide) about `mau_moi_giay` times per second, decoding only what is needed.

    All-intra proxies (ProRes, DNxHR): only the packets at the sample times are decoded (about 12x faster than decoding
    every 4K frame). Long-GOP (H.264/HEVC): keyframes only, when they are at most `keyframe_thua_nhat` seconds apart
    (default 1.5 sample steps; Sony XAVC has one every 0.5 s), otherwise a full decode. Rotation metadata is applied, like
    ffmpeg does."""
    import av
    from dailies.quet import ffprobe, xoay
    try:
        c = av.open(duong_dan)
    except av.error.FFmpegError:
        yield from _ffmpeg_khung(duong_dan, mau_moi_giay, rong)
        return
    try:
        v = c.streams.video[0]
        v.thread_type = "AUTO"
        w, h = v.codec_context.width, v.codec_context.height
        quay = xoay(next((s for s in ffprobe(duong_dan)["streams"] if s.get("codec_type") == "video"), {})) // 90
        if quay % 2:
            w, h = h, w  # `rong` is the displayed width
        cao = max(2, round(h * rong / w) // 2 * 2)
        giai_w, giai_h = (cao, rong) if quay % 2 else (rong, cao)
        tb = float(v.time_base)
        goc = float(v.start_time * v.time_base) if v.start_time is not None else 0.0
        buoc = 1.0 / mau_moi_giay
        nua_frame = 0.5 / float(v.average_rate or 25)

        def ra_anh(fr):
            a = fr.reformat(width=giai_w, height=giai_h, format="rgb24", interpolation="AREA").to_ndarray()
            return np.ascontiguousarray(np.rot90(a, quay)) if quay else a

        if v.codec_context.name in INTRA:
            # Jump straight to each sample (every packet is a keyframe), so only about 1 packet in fps/mau is read
            # from disk: this matters in quiet mode, where macOS throttles background disk reads. Slice threads, as
            # frame threads would hold the single packet back.
            v.thread_type = "SLICE"
            dai = float(v.duration * tb) if v.duration else (c.duration / 1e6 if c.duration else 0.0)
            k, cuoi = 0, None
            while k * buoc < dai + nua_frame:
                dich = k * buoc
                c.seek(int(round((dich + goc) / tb)), stream=v, backward=True, any_frame=True)
                for pk in c.demux(v):
                    if pk.pts is None or pk.size == 0:
                        continue
                    t = pk.pts * tb - goc
                    if t + nua_frame < dich:
                        continue
                    if cuoi is None or t > cuoi + nua_frame:
                        for fr in pk.decode():
                            yield round(fr.pts * tb - goc, 4), ra_anh(fr)
                        cuoi = t
                    break
                else:
                    break
                k += 1
            return
        # Keyframe spacing from the first keyframe packets (no decoding)
        with av.open(duong_dan) as dem:
            moc = []
            for pk in dem.demux(dem.streams.video[0]):
                if pk.is_keyframe and pk.pts is not None:
                    moc.append(pk.pts * tb)
                    if len(moc) == 4:
                        break
        moc.sort()
        theo_keyframe = len(moc) >= 2 and max(b - a for a, b in zip(moc, moc[1:])) <= (keyframe_thua_nhat or 1.5 * buoc)
        if theo_keyframe:
            # Keyframes leave the decoder slightly out of order (B-frame delay): a small heap puts them back in order.
            # Each frame belongs to the nearest sample time; per sample time the closest frame is kept. Frame
            # threading only adds latency when every decoded frame is a keyframe.
            import heapq
            v.thread_type = "SLICE"
            v.codec_context.skip_frame = "NONKEY"
            dong, giu = [], None  # giu: (sample index, distance, t, frame)

            def nhan(t, fr):
                nonlocal giu
                k = round(t / buoc)
                kc = abs(t - k * buoc)
                if giu is not None and giu[0] != k:
                    yield round(giu[2], 4), ra_anh(giu[3])
                    giu = None
                if giu is None or kc < giu[1]:
                    giu = (k, kc, t, fr)

            for so, fr in enumerate(c.decode(v)):
                if fr.pts is None:
                    continue
                heapq.heappush(dong, (fr.pts, so, fr))
                if len(dong) > 8:
                    pts, _, f = heapq.heappop(dong)
                    yield from nhan(pts * tb - goc, f)
            while dong:
                pts, _, f = heapq.heappop(dong)
                yield from nhan(pts * tb - goc, f)
            if giu is not None:
                yield round(giu[2], 4), ra_anh(giu[3])
            return
        tiep = 0.0
        for fr in c.decode(v):
            if fr.pts is None:
                continue
            t = fr.pts * tb - goc
            if t + nua_frame < tiep:
                continue
            yield round(t, 4), ra_anh(fr)
            while tiep <= t + nua_frame:
                tiep += buoc
    finally:
        c.close()


def doc_khung(duong_dan: str, mau_moi_giay: float = MAU_MOI_GIAY, rong: int = RONG) -> np.ndarray:
    """Small RGB frames evenly spaced in time (see `lap_khung`)."""
    anh = [a for _, a in lap_khung(duong_dan, mau_moi_giay, rong)]
    return np.stack(anh) if anh else np.empty((0, 2, rong, 3), np.uint8)


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
