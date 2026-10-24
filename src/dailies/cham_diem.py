"""Score footage and pick usable segments ("selects"), on device, no tokens spent.

Every clip is sampled about 4 times per second from its proxy (`thi_giac.lap_khung`, only the needed frames are
decoded). Per sample:
- sharpness: Laplacian on edge pixels only (the edge threshold rises in dark areas so sensor noise does not count),
  on a 6x6 grid, keeping the 3 sharpest cells, so a shallow-focus shot with a sharp subject scores as sharp; plus edge
  density (detail);
- exposure: mean and spread of luma, clipped highlights, crushed blacks;
- camera motion: corners tracked with pyramidal Lucas-Kanade (forward-backward checked), a similarity transform fitted
  with RANSAC gives pan, tilt, zoom and roll per second; what the transform does not explain is subject motion;
- face count (YuNet) and saturation, for information.
Raw metrics are cached per file (content hash). Scores are computed afterwards for the whole run: sharpness is relative
to the sharpest footage of the run (cameras and lenses differ), shake is the jitter of camera motion around its local
trend (a smooth gimbal move is not shake). Samples above the threshold become segments; short dips are bridged.
"""

from __future__ import annotations

import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

PHIEN_BAN = 2
TEN_TEP = "cham_diem.json"
MAU_MOI_GIAY = 4
RONG_NET = 960  # sharpness needs more pixels than motion: slight softness is invisible at 640
RONG_DONG = 320
LUOI = 6

# Group weights and how camera movement is judged, per kind of footage
KIEU = {
    "can_bang": {"mo_ta": "Mặc định: nét, đủ sáng, máy đứng yên hoặc di chuyển mượt",
                 "trong_so": {"net": 0.45, "phoi": 0.2, "chuyen_dong": 0.15, "on_dinh": 0.2}, "may_toi_da": 0.5},
    "b_roll": {"mo_ta": "B-roll gimbal, drone: máy di chuyển không bị trừ nếu mượt",
               "trong_so": {"net": 0.4, "phoi": 0.2, "chuyen_dong": 0.05, "on_dinh": 0.35}, "may_toi_da": 2.0},
    "tinh": {"mo_ta": "Chỉ lấy shot máy đứng yên (phỏng vấn, sản phẩm, kiến trúc)",
             "trong_so": {"net": 0.4, "phoi": 0.2, "chuyen_dong": 0.3, "on_dinh": 0.1}, "may_toi_da": 0.08},
}
# Flags that remove a sample outright; the others only lower their group score
CO_LOAI = ("den_kit", "mot_mau", "chay_trang", "cat_cung")
CHUYEN_DONG_MAY = {  # (component, sign) -> class; thresholds per second
    ("lia", 1): "lia_phai", ("lia", -1): "lia_trai", ("nghieng", 1): "nghieng_len", ("nghieng", -1): "nghieng_xuong",
    ("zoom", 1): "zoom_vao", ("zoom", -1): "zoom_ra", ("xoay", 1): "xoay_trai", ("xoay", -1): "xoay_phai",
}
NGUONG_MAY = {"lia": 0.04, "nghieng": 0.04, "zoom": 0.02, "xoay": 1.0}
_LOC_NHIEU = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)


# ---------------------------------------------------------------- per-sample metrics

def do_net(xam: np.ndarray) -> tuple[float, float]:
    """(sharpness, edge density) of a grey frame."""
    g = xam.astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    do_lon = cv2.magnitude(gx, gy)
    lap = np.abs(cv2.Laplacian(g, cv2.CV_32F, ksize=1))
    # Noise level of the frame (median of a noise-only filter, so edges barely count), and darker areas get a higher
    # bar: sensor and compression noise is not detail
    nhieu = 1.4826 * float(np.median(np.abs(cv2.filter2D(g, -1, _LOC_NHIEU))[::2, ::2])) / 6
    nguong = np.maximum(24.0, 10.0 * nhieu) + 0.3 * np.maximum(0.0, 64.0 - g)
    canh = (do_lon > nguong) & (lap > 6.0)
    gia_tri = np.where(canh, np.minimum(lap, 255.0), 0.0)
    h, w = g.shape
    o = []
    for i in range(LUOI):
        for j in range(LUOI):
            y0, y1 = i * h // LUOI, (i + 1) * h // LUOI
            x0, x1 = j * w // LUOI, (j + 1) * w // LUOI
            n = int(canh[y0:y1, x0:x1].sum())
            o.append(float(gia_tri[y0:y1, x0:x1].sum()) / n if n >= 0.002 * (y1 - y0) * (x1 - x0) else 0.0)
    o.sort(reverse=True)
    return float(np.mean(o[:3])), float(canh.mean())


def do_chuyen_dong(truoc: np.ndarray, sau: np.ndarray, dt: float) -> dict:
    """Camera motion per second between two grey frames: pan (+ right), tilt (+ up), zoom (+ in), roll (+ left,
    degrees), subject motion (frame widths per second), confidence (inlier ratio)."""
    trong = {"lia": None, "nghieng": None, "zoom": None, "xoay": None, "chu_the": None, "tin_cay": 0.0}
    h, w = truoc.shape
    p0 = cv2.goodFeaturesToTrack(truoc, maxCorners=300, qualityLevel=0.01, minDistance=7, blockSize=7)
    if p0 is None or len(p0) < 12:
        return trong
    tham_so = {"winSize": (21, 21), "maxLevel": 3, "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)}
    p1, st, _ = cv2.calcOpticalFlowPyrLK(truoc, sau, p0, None, **tham_so)
    p0r, st2, _ = cv2.calcOpticalFlowPyrLK(sau, truoc, p1, None, **tham_so)
    tot = (st.ravel() == 1) & (st2.ravel() == 1) & (np.linalg.norm((p0 - p0r).reshape(-1, 2), axis=1) < 1.0)
    if tot.sum() < 12:
        return trong
    a, b = p0[tot].reshape(-1, 2), p1[tot].reshape(-1, 2)
    m, trong_mau = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC, ransacReprojThreshold=1.5)
    if m is None:
        return trong
    tam = np.array([w / 2, h / 2, 1.0])
    dx, dy = m @ tam - tam[:2]
    ty_le = math.hypot(m[0, 0], m[1, 0])
    goc = math.degrees(math.atan2(m[1, 0], m[0, 0]))  # positive: the picture turns clockwise, the camera rolls left
    du = np.linalg.norm(b - (a @ m[:, :2].T + m[:, 2]), axis=1)
    return {"lia": round(-dx / w / dt, 5), "nghieng": round(dy / h / dt, 5), "zoom": round(math.log(max(ty_le, 1e-6)) / dt, 5),
            "xoay": round(goc / dt, 4), "chu_the": round(float(np.maximum(du - 1.0, 0).mean()) / w / dt, 5),
            "tin_cay": round(float(trong_mau.mean()), 3)}


def do_mau(rgb: np.ndarray, truoc: np.ndarray | None, dt: float, bo_mat, dem_mat: bool = True) -> tuple[dict, np.ndarray]:
    """Raw metrics of one sample (`rgb` is RONG_NET wide) and its small grey frame for the next motion step."""
    xam = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    nho = cv2.resize(xam, (RONG_DONG, max(2, round(xam.shape[0] * RONG_DONG / xam.shape[1]))), interpolation=cv2.INTER_AREA)
    net, chi_tiet = do_net(xam)
    kq = {"net": round(net, 3), "chi_tiet": round(chi_tiet, 5), "sang": round(float(xam.mean()), 2),
          "tuong_phan": round(float(xam.std()), 2), "chay": round(float((xam >= 250).mean()), 5),
          "den": round(float((xam <= 6).mean()), 5)}
    nho_rgb = cv2.resize(rgb, nho.shape[::-1], interpolation=cv2.INTER_AREA)
    kq["bao_hoa"] = round(float(cv2.cvtColor(nho_rgb, cv2.COLOR_RGB2HSV)[..., 1].mean()), 2)
    if bo_mat is not None and dem_mat:
        bo_mat.setInputSize(nho.shape[::-1])
        _, mat = bo_mat.detect(cv2.cvtColor(nho_rgb, cv2.COLOR_RGB2BGR))
        kq["mat"] = 0 if mat is None else len(mat)
    else:
        kq["mat"] = None
    if truoc is None:
        kq.update({"lia": None, "nghieng": None, "zoom": None, "xoay": None, "chu_the": None, "tin_cay": 0.0,
                   "lech": None, "tuong_quan": None})
    else:
        kq.update(do_chuyen_dong(truoc, nho, dt))
        kq["lech"] = round(float(cv2.absdiff(truoc, nho).mean()), 3)
        h1 = cv2.calcHist([truoc], [0], None, [32], [0, 256])
        h2 = cv2.calcHist([nho], [0], None, [32], [0, 256])
        kq["tuong_quan"] = round(float(cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)), 4)
    return kq, nho


def quet_clip(nguon: str, mau_moi_giay: float = MAU_MOI_GIAY, doc=None) -> list[dict]:
    """Raw metrics of every sample of one file."""
    from dailies import thi_giac
    doc = doc or (lambda p: thi_giac.lap_khung(p, mau_moi_giay, RONG_NET, keyframe_thua_nhat=0.6))
    bo_mat = _bo_mat_rieng()
    mau, truoc, t_truoc, t_mat, mat = [], None, None, None, None
    for t, rgb in doc(nguon):
        dt = (t - t_truoc) if t_truoc is not None else 1.0 / mau_moi_giay
        dem_mat = t_mat is None or t - t_mat >= 0.99  # faces once a second is enough, and they cost the most
        kq, truoc = do_mau(rgb, truoc if dt > 0 else None, max(dt, 1e-3), bo_mat, dem_mat)
        if dem_mat:
            t_mat, mat = t, kq["mat"]
        else:
            kq["mat"] = mat
        mau.append({"t": round(float(t), 4), **kq})
        t_truoc = t
    return mau


def _bo_mat_rieng():
    """A YuNet instance per thread (the detector keeps its input size)."""
    from dailies import thi_giac
    tep = thi_giac._tep_yunet()
    return cv2.FaceDetectorYN.create(str(tep), "", (RONG_DONG, 180), 0.7) if tep else None


# ---------------------------------------------------------------- scores and segments

def _cua_so(x: np.ndarray, r: int = 2) -> np.ndarray:
    """Running median over 2r+1 samples (NaN ignored)."""
    if len(x) == 0:
        return x
    p = np.pad(x, r, mode="edge")
    with np.errstate(all="ignore"):
        return np.array([np.nanmedian(p[i:i + 2 * r + 1]) if not np.all(np.isnan(p[i:i + 2 * r + 1])) else np.nan
                         for i in range(len(x))])


def _rms_bien_doi(x: np.ndarray, r: int = 2) -> np.ndarray:
    """RMS of sample-to-sample change over 2r+1 samples, divided by sqrt(2) (differencing doubles noise variance)."""
    if len(x) < 2:
        return np.zeros(len(x))
    d2 = np.diff(x, prepend=x[0]) ** 2
    d2[0] = d2[1]
    p = np.pad(d2, r, mode="edge")
    return np.sqrt(np.convolve(p, np.ones(2 * r + 1) / (2 * r + 1), mode="valid") / 2)


def _mang(mau: list[dict], khoa: str) -> np.ndarray:
    return np.array([np.nan if m.get(khoa) is None else m[khoa] for m in mau], dtype=float)


def tham_chieu(tat_ca: list[list[dict]]) -> dict:
    """Run-wide references: the 90th percentile sharpness and detail of usable samples."""
    net = [m["net"] for mau in tat_ca for m in mau if m["tuong_phan"] >= 5 and m["sang"] >= 16]
    ct = [m["chi_tiet"] for mau in tat_ca for m in mau if m["tuong_phan"] >= 5 and m["sang"] >= 16]
    return {"net": float(np.percentile(net, 90)) if net else 1.0, "chi_tiet": float(np.percentile(ct, 90)) if ct else 1.0}


def cham(mau: list[dict], tc: dict, kieu: str = "can_bang") -> list[dict]:
    """Per-sample group scores, flags and the 0-1 score."""
    if not mau:
        return []
    cd = KIEU[kieu]
    net = _mang(mau, "net")
    f_net = np.clip(net / (tc["net"] or 1), 0, 1)
    f_ct = np.clip(_mang(mau, "chi_tiet") / (tc["chi_tiet"] or 1), 0, 1)
    tv = _cua_so(net)
    on_net = 1 - np.clip(np.abs(net - tv) / np.maximum(tv, 1e-6), 0, 1)
    diem_net = (0.7 * f_net + 0.3 * f_ct) * (0.85 + 0.15 * on_net)

    sang, tp = _mang(mau, "sang"), _mang(mau, "tuong_phan")
    diem_phoi = 1 - np.clip(5 * _mang(mau, "chay") + 3 * _mang(mau, "den"), 0, 1)
    diem_phoi *= np.clip(sang / 40, 0, 1) * np.clip((255 - sang) / 35, 0, 1) * np.clip(tp / 15, 0.3, 1)

    # Missing motion (first sample, flat frames): borrow the neighbours' trend
    thanh_phan = {k: _mang(mau, k) for k in ("lia", "nghieng", "zoom", "xoay", "chu_the")}
    xu_huong = {k: _cua_so(v) for k, v in thanh_phan.items()}
    for k, v in thanh_phan.items():
        thanh_phan[k] = np.where(np.isnan(v), np.nan_to_num(xu_huong[k]), v)
    lia, nghieng, zoom, xoay = (thanh_phan[k] for k in ("lia", "nghieng", "zoom", "xoay"))
    toc_do = np.hypot(lia, nghieng) + 0.5 * np.abs(zoom) + np.abs(xoay) / 30
    # Shake: how much camera motion changes from one sample to the next (RMS over 5 samples); a steady move, even a
    # fast one, changes little
    lech = np.sqrt(sum(_rms_bien_doi(thanh_phan[k]) ** 2 * he
                       for k, he in (("lia", 1), ("nghieng", 1), ("zoom", 0.25), ("xoay", 1 / 900))))
    diem_on_dinh = 1 - np.clip(lech / 0.12, 0, 1)
    chu_the = thanh_phan["chu_the"]
    diem_cd = 1 - 0.7 * np.clip(toc_do / cd["may_toi_da"], 0, 1) - 0.3 * np.clip(chu_the / 0.25, 0, 1)

    ts = cd["trong_so"]
    diem = ts["net"] * diem_net + ts["phoi"] * diem_phoi + ts["chuyen_dong"] * diem_cd + ts["on_dinh"] * diem_on_dinh
    # A soft or badly exposed frame is not saved by being steady: those two also scale the whole score
    diem *= np.sqrt(np.clip(diem_net / 0.6, 0, 1) * np.clip(diem_phoi / 0.6, 0, 1))
    kq = []
    for i, m in enumerate(mau):
        co = []
        if m["sang"] < 16 and m["tuong_phan"] < 10:
            co.append("den_kit")
        elif m["tuong_phan"] < 4:
            co.append("mot_mau")
        if m["chay"] > 0.25:
            co.append("chay_trang")
        if m["sang"] < 45 and "den_kit" not in co:
            co.append("thieu_sang")
        if m.get("lech") is not None:
            if m["lech"] < 0.05:  # a frozen frame; a locked-off shot still differs by sensor noise
                co.append("trung")
            # A cut needs two sharp frames that do not match; a whip pan is blurred and is not a cut
            if ((m["tuong_quan"] or 1) < 0.5 and m["lech"] > 20 and m["tin_cay"] < 0.3
                    and i > 0 and min(f_net[i], f_net[i - 1]) >= 0.5):
                co.append("cat_cung")
            # Flicker: brightness goes up then down (or down then up) while neither the camera nor the subject moves;
            # one step is an exposure change or something bright entering
            if (0 < i < len(mau) - 1 and toc_do[i] < 0.05 and chu_the[i] < 0.02
                    and (sang[i] - sang[i - 1]) * (sang[i + 1] - sang[i]) < 0
                    and min(abs(sang[i] - sang[i - 1]), abs(sang[i + 1] - sang[i])) > 6):
                co.append("nhap_nhay")
        if lech[i] > 0.12:
            co.append("rung")
        if f_net[i] < 0.35 and toc_do[i] > 0.3:
            co.append("nhoe")
        d = 0.0 if any(c in CO_LOAI for c in co) else float(np.clip(diem[i], 0, 1))
        kq.append({"t": m["t"], "diem": round(d, 4), "net": round(float(diem_net[i]), 4), "phoi": round(float(diem_phoi[i]), 4),
                   "chuyen_dong": round(float(diem_cd[i]), 4), "on_dinh": round(float(diem_on_dinh[i]), 4), "co": co,
                   "may": {"lia": round(float(lia[i]), 4), "nghieng": round(float(nghieng[i]), 4),
                           "zoom": round(float(zoom[i]), 4), "xoay": round(float(xoay[i]), 3)}})
    return kq


def phan_loai_may(diem: list[dict]) -> str:
    """Camera movement class of a run of samples, from the median pan, tilt, zoom and roll."""
    tv = {k: float(np.median([d["may"][k] for d in diem])) for k in NGUONG_MAY}
    muc = sorted(((abs(v) / NGUONG_MAY[k], k, v) for k, v in tv.items()), reverse=True)
    if muc[0][0] < 1:
        return "dung_yen"
    if muc[1][0] >= 1 and muc[1][0] >= 0.6 * muc[0][0]:
        return "hon_hop"
    return CHUYEN_DONG_MAY[(muc[0][1], 1 if muc[0][2] > 0 else -1)]


def dung_doan(diem: list[dict], thoi_luong: float, *, nguong: float = 0.85, ngan_nhat: float = 2.0,
              noi_khoang: float = 2.0, sau_nhat: float = 0.15) -> list[dict]:
    """Segments of samples at or above `nguong`. A dip of at most `noi_khoang` seconds whose samples stay above
    `nguong - sau_nhat` and carry no removing flag is bridged; segments shorter than `ngan_nhat` are dropped."""
    if not diem:
        return []
    buoc = float(np.median(np.diff([d["t"] for d in diem]))) if len(diem) > 1 else 0.25
    chay, hien = [], None
    for i, d in enumerate(diem):
        if d["diem"] >= nguong:
            if hien is None:
                hien = [i, i]
            else:
                hien[1] = i
        elif hien is not None:
            chay.append(hien)
            hien = None
    if hien is not None:
        chay.append(hien)
    gop = []
    for c in chay:
        if gop:
            truoc = gop[-1]
            khe = diem[truoc[1] + 1:c[0]]
            if (diem[c[0]]["t"] - diem[truoc[1]]["t"] - buoc <= noi_khoang + 1e-6
                    and all(d["diem"] >= nguong - sau_nhat and not set(d["co"]) & set(CO_LOAI) for d in khe)):
                truoc[1] = c[1]
                continue
        gop.append(list(c))
    doan = []
    for a, b in gop:
        vao = max(0.0, diem[a]["t"] - buoc / 2)
        ra = min(thoi_luong, diem[b]["t"] + buoc / 2) if thoi_luong else diem[b]["t"] + buoc / 2
        if ra - vao + 1e-6 < ngan_nhat:
            continue
        nhom = diem[a:b + 1]
        gia_tri = [d["diem"] for d in nhom]
        dinh = max(range(len(nhom)), key=lambda k: gia_tri[k])
        tb = float(np.mean(gia_tri))
        doan.append({"vao_giay": round(vao, 3), "ra_giay": round(ra, 3), "dai_giay": round(ra - vao, 3),
                     "diem_tb": round(tb, 4), "diem_dinh": round(gia_tri[dinh], 4), "giay_dinh": nhom[dinh]["t"],
                     "muc": "tot" if tb >= nguong + 0.08 else "kha" if tb >= nguong + 0.04 else "dat",
                     "may": phan_loai_may(nhom), "co": sorted({c for d in nhom for c in d["co"]})})
    return doan


# ---------------------------------------------------------------- run over a folder or list

def _quet_mot(media_id: str, muc: dict, nguon: str, cache_dir: Path, mau_moi_giay: float, doc) -> dict:
    tep = cache_dir / f"{media_id}.json"
    try:
        cu = json.loads(tep.read_text(encoding="utf-8"))
        if cu.get("phien_ban") == PHIEN_BAN and cu.get("ma_bam") == muc["ma_bam"] and cu.get("mau_moi_giay") == mau_moi_giay:
            return {**cu, "tu_cache": True}
    except (OSError, ValueError):
        pass
    from dailies.may import nghi_neu_nong
    nghi_neu_nong()
    bat_dau = time.perf_counter()
    mau = quet_clip(nguon, mau_moi_giay, doc)
    kq = {"phien_ban": PHIEN_BAN, "ma_bam": muc["ma_bam"], "nguon": nguon, "mau_moi_giay": mau_moi_giay,
          "giay_xu_ly": round(time.perf_counter() - bat_dau, 3), "mau": mau}
    tam = tep.with_suffix(".tam")
    tam.write_text(json.dumps(kq, ensure_ascii=False), encoding="utf-8")
    tam.replace(tep)
    return {**kq, "tu_cache": False}


def cham_thu_muc(source: str | Path | None = None, *, danh_sach: str | Path | None = None, cho_phep_goc: bool = False,
                 kieu: str = "can_bang", nguong: float = 0.85, ngan_nhat: float = 2.0, noi_khoang: float = 2.0,
                 mau_moi_giay: float = MAU_MOI_GIAY, cache_goc: str | Path | None = None, doc=None) -> dict:
    """Score every clip of a folder or list and write `.dailies/cham_diem.json`."""
    from dailies import thu_muc_ket_qua
    from dailies.khoa import giu_khoa
    from dailies.may import che_do
    from dailies.quet import nguon_vao, quet_files
    if kieu not in KIEU:
        raise ValueError(f"Kiểu phải là một trong {', '.join(KIEU)}")
    if not 0 < nguong < 1:
        raise ValueError("Ngưỡng phải trong khoảng 0-1")
    source, cac_tep, proxy = nguon_vao(source, danh_sach, cho_phep_goc)
    goc = thu_muc_ket_qua(source, cache_goc)
    cache_dir = goc / "cham_diem"
    cache_dir.mkdir(parents=True, exist_ok=True)
    bat_dau = time.perf_counter()
    with giu_khoa(goc, "cham-diem"):
        cac_muc = quet_files(cac_tep)
        so_luong = 2 if che_do() == "em" else 4
        xong = [0]

        def mot(cap):
            media_id, muc = cap
            p = proxy.get(muc["duong_dan_gan_nhat"])
            kq = _quet_mot(media_id, muc, str(p or muc["duong_dan_gan_nhat"]), cache_dir, mau_moi_giay, doc)
            xong[0] += 1
            print(f"Chấm điểm: {xong[0]}/{len(cac_muc)} {Path(muc['duong_dan_gan_nhat']).name}", file=sys.stderr, flush=True)
            return media_id, muc, p, kq

        with ThreadPoolExecutor(max_workers=so_luong) as pool:
            ket_qua = list(pool.map(mot, cac_muc))
    tc = tham_chieu([kq["mau"] for *_, kq in ket_qua])
    clip = []
    for media_id, muc, p, kq in ket_qua:
        duong = Path(muc["duong_dan_gan_nhat"])
        fps = muc.get("fps") or {"tu": 25, "mau": 1}
        thoi_luong = float(Fraction(muc.get("so_frame", 0)) / Fraction(fps["tu"], fps["mau"]))
        diem = cham(kq["mau"], tc, kieu)
        doan = dung_doan(diem, thoi_luong, nguong=nguong, ngan_nhat=ngan_nhat, noi_khoang=noi_khoang)
        for d in doan:
            d["vao_frame"] = round(d["vao_giay"] * fps["tu"] / fps["mau"])
            d["ra_frame"] = round(d["ra_giay"] * fps["tu"] / fps["mau"])
        gia_tri = [d["diem"] for d in diem]
        dem = {}
        for d in diem:
            for c in d["co"]:
                dem[c] = dem.get(c, 0) + 1
        clip.append({"id": media_id, "media_id": media_id, "file": str(duong),
                     "ten_file": str(duong.relative_to(source)) if duong.is_relative_to(source) else duong.name,
                     "fps": fps, "thoi_luong_giay": round(thoi_luong, 3), "co_proxy": bool(p), "so_mau": len(diem),
                     "diem_tb": round(float(np.mean(gia_tri)), 4) if gia_tri else None,
                     "diem_tot_nhat": max((d["diem_tb"] for d in doan), default=None),
                     "may": phan_loai_may(diem) if diem else None, "co": dem, "doan": doan,
                     "tep_mau": str(cache_dir / f"{media_id}.json")})
    from dailies.mo_ta_shot import _khoa_ten
    clip.sort(key=lambda c: _khoa_ten(c["ten_file"]))
    for stt, c in enumerate(clip, 1):
        c["stt"] = stt
    kq = {"phien_ban": PHIEN_BAN, "thu_muc": str(source), "kieu": kieu, "mo_ta_kieu": KIEU[kieu]["mo_ta"],
          "cai_dat": {"nguong": nguong, "doan_ngan_nhat": ngan_nhat, "noi_khoang": noi_khoang, "mau_moi_giay": mau_moi_giay},
          "tham_chieu": {k: round(v, 4) for k, v in tc.items()}, "so_clip": len(clip),
          "so_clip_co_doan": sum(1 for c in clip if c["doan"]), "so_doan": sum(len(c["doan"]) for c in clip),
          "giay_xu_ly": round(time.perf_counter() - bat_dau, 2),
          "so_clip_quet_moi": sum(1 for *_, k in ket_qua if not k["tu_cache"]), "clip": clip}
    dich = goc / TEN_TEP
    tam = dich.with_suffix(".tmp")
    tam.write_text(json.dumps(kq, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tam.replace(dich)
    return {**kq, "file_ket_qua": str(dich)}
