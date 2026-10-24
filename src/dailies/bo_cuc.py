"""Framing for the rough cut: scale and position per take to place (usually 4K) footage in a lighter sequence (Full HD, 9:16).

Same aspect ratio: fit to frame (4K into Full HD is 50%), keeping the camera framing. Different aspect ratio: fill the
frame, then shift towards detected faces (YuNet on the proxy, a few frames per take): face centred horizontally, face
centre at 40% of frame height (headroom); no face means centre. Position is always clamped so the image covers the
frame with no black edges. Position follows Premiere's Motion convention: anchor at clip centre, in sequence pixels;
a normalised 0-1 copy is included.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

TAM_MAT_DOC = 0.4
SO_ANH_TOI_DA = 8


def doc_khung_chuoi(chuoi: str) -> tuple[int, int]:
    """'1920x1080' or '9:16' (read as 1080x1920), '16:9' (1920x1080)."""
    chuoi = chuoi.strip().lower()
    tat = {"16:9": (1920, 1080), "9:16": (1080, 1920), "1:1": (1080, 1080), "4:5": (1080, 1350)}
    if chuoi in tat:
        return tat[chuoi]
    rong, _, cao = chuoi.partition("x")
    if not (rong.isdigit() and cao.isdigit()) or not int(rong) or not int(cao):
        raise ValueError(f"Khung sequence không hợp lệ: {chuoi!r} (ví dụ 1920x1080, 9:16)")
    return int(rong), int(cao)


def tam_mat(proxy: str, vao: float, ra: float) -> tuple[tuple[float, float] | None, int]:
    """Mean face centre (normalised 0-1 to the source frame) within [vao, ra) of the proxy, plus the max face count.
    Decoding goes through the ffmpeg CLI; cv2 only runs the face detector (decision 017)."""
    import cv2
    import numpy as np
    from dailies import thi_giac
    dai = max(0.1, ra - vao)
    fps = min(1.0, SO_ANH_TOI_DA / dai)
    from dailies.quet import kich_thuoc_hien
    w, h = kich_thuoc_hien(proxy)
    rong = 320
    cao = max(2, round(h * rong / w) // 2 * 2)
    ra_anh = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{vao:.3f}", "-t", f"{dai:.3f}", "-i", proxy,
                             "-vf", f"fps={fps:.4f},scale={rong}:{cao},format=bgr24", "-f", "rawvideo", "-"],
                            capture_output=True, check=True).stdout
    anh = np.frombuffer(ra_anh, np.uint8).reshape(-1, cao, rong, 3)
    bo = thi_giac._bo_do_mat(rong, cao) if len(anh) else None
    if bo is None:
        return None, 0
    tam, nhieu_nhat = [], 0
    for a in anh:
        _, mat = bo.detect(np.ascontiguousarray(a))
        if mat is None:
            continue
        nhieu_nhat = max(nhieu_nhat, len(mat))
        dien_tich = mat[:, 2] * mat[:, 3]
        cx = float(((mat[:, 0] + mat[:, 2] / 2) * dien_tich).sum() / dien_tich.sum()) / rong
        cy = float(((mat[:, 1] + mat[:, 3] / 2) * dien_tich).sum() / dien_tich.sum()) / cao
        tam.append((cx, cy))
    if not tam:
        return None, 0
    return (sum(x for x, _ in tam) / len(tam), sum(y for _, y in tam) / len(tam)), nhieu_nhat


def _dat(s: float, nguon: tuple[int, int], khung: tuple[int, int], diem: tuple[float, float] | None,
         doc: float) -> tuple[float, float]:
    """Clip centre that puts `diem` (0-1 of source) horizontally centred and at `doc` of frame height; clamped to cover the frame."""
    w, h = nguon
    W, H = khung
    rong, cao = w * s, h * s
    x, y = W / 2, H / 2
    if diem:
        x = W / 2 - (diem[0] - 0.5) * rong
        y = H * doc - (diem[1] - 0.5) * cao
    if rong >= W:
        x = min(max(x, W - rong / 2), rong / 2)
    if cao >= H:
        y = min(max(y, H - cao / 2), cao / 2)
    return x, y


def _ket_qua(nguon, khung, cach, s, x, y, tam, so_mat) -> dict:
    W, H = khung
    return {"nguon": {"rong": nguon[0], "cao": nguon[1]}, "khung": {"rong": W, "cao": H}, "cach": cach,
            "scale_phan_tram": round(s * 100, 3), "vi_tri": {"x": round(x, 1), "y": round(y, 1)},
            "vi_tri_chuan": {"x": round(x / W, 5), "y": round(y / H, 5)},
            "tam_chu_the": {"x": round(tam[0], 4), "y": round(tam[1], 4)} if tam else None, "so_mat": so_mat}


def tinh(nguon: tuple[int, int], khung: tuple[int, int], tam: tuple[float, float] | None, so_mat: int = 0) -> dict:
    """Scale (%) and clip-centre position in the sequence."""
    w, h = nguon
    W, H = khung
    if abs(w / h - W / H) < 0.01 * (W / H):
        s, cach, diem = min(W / w, H / h), "vua_khung", None
    else:
        s = max(W / w, H / h)
        cach, diem = ("day_khung_theo_mat", tam) if tam else ("day_khung_giua", None)
    x, y = _dat(s, nguon, khung, diem, TAM_MAT_DOC)
    return _ket_qua(nguon, khung, cach, s, x, y, tam, so_mat)


def vung_hien(nguon: tuple[int, int], khung_goc: tuple[int, int], ngang: dict) -> tuple[float, float, float, float] | None:
    """Visible part of the clip in the landscape frame (left, top, right, bottom in landscape pixels): the clip box after
    scale, position, anchor point and the Crop effect, intersected with the frame. None when the clip is not visible."""
    w, h = nguon
    W0, H0 = khung_goc
    s0 = (ngang["scale_phan_tram"] / 100) if ngang.get("scale_phan_tram") else min(W0 / w, H0 / h)
    vt = ngang.get("vi_tri") or {"x": W0 / 2, "y": H0 / 2}
    neo = ngang.get("diem_neo") or {"x": w / 2, "y": h / 2}
    crop = ngang.get("crop") or {}
    trai = vt["x"] - neo["x"] * s0
    tren = vt["y"] - neo["y"] * s0
    l = trai + w * s0 * crop.get("trai", 0) / 100
    t = tren + h * s0 * crop.get("tren", 0) / 100
    r = trai + w * s0 * (1 - crop.get("phai", 0) / 100)
    d = tren + h * s0 * (1 - crop.get("duoi", 0) / 100)
    l, t, r, d = max(l, 0), max(t, 0), min(r, W0), min(d, H0)
    return (l, t, r, d) if r - l > 1 and d - t > 1 else None


def tu_ban_ngang(nguon: tuple[int, int], khung_goc: tuple[int, int], ngang: dict, khung: tuple[int, int],
                 tam: tuple[float, float] | None, so_mat: int = 0) -> dict | None:
    """Square / vertical framing for one clip, cut from the landscape frame exactly as the editor adjusted it.

    The landscape frame (after the editor's scale, position, anchor point and Crop) is the source image; the new version
    is a window of that image scaled to fill the new frame, so pixels match the landscape cut and never reveal cropped
    areas or black edges. The window stays inside the clip's visible area; if that area is narrower, it zooms further.
    If a face is inside the landscape frame, the window shifts to centre it horizontally with the face centre at 40%
    height (within limits); otherwise it centres on the visible area.
    New position = (landscape position - window corner) x zoom, valid for any anchor point; new scale = scale x zoom."""
    w, h = nguon
    W0, H0 = khung_goc
    W, H = khung
    vung = vung_hien(nguon, khung_goc, ngang)
    if vung is None:
        return None
    l, t, r, d = vung
    s0 = (ngang["scale_phan_tram"] / 100) if ngang.get("scale_phan_tram") else min(W0 / w, H0 / h)
    vt = ngang.get("vi_tri") or {"x": W0 / 2, "y": H0 / 2}
    neo = ngang.get("diem_neo") or {"x": w / 2, "y": h / 2}
    canh_bao = []
    if ngang.get("xoay"):
        canh_bao.append("clip có xoay ở bản ngang: chưa tính góc xoay, anh xem lại khung này")
    k = max(W / W0, H / H0)
    if (r - l) * k < W - 1e-6 or (d - t) * k < H - 1e-6:
        k = max(W / (r - l), H / (d - t))
        canh_bao.append("phần có hình ở bản ngang hẹp hơn khung mới nên phóng thêm")
    cw, ch = W / k, H / k  # window in landscape-frame pixels
    tam_ngang = None
    if tam:
        fx = vt["x"] + (tam[0] * w - neo["x"]) * s0
        fy = vt["y"] + (tam[1] * h - neo["y"]) * s0
        if l <= fx <= r and t <= fy <= d:
            tam_ngang = (fx, fy)
    if tam_ngang:
        cach = "cat_tu_ban_ngang_theo_mat"
        ox, oy = tam_ngang[0] - cw / 2, tam_ngang[1] - TAM_MAT_DOC * ch
    else:
        cach = "cat_tu_ban_ngang_giua"
        ox, oy = (l + r) / 2 - cw / 2, (t + d) / 2 - ch / 2
    ox = min(max(ox, l), r - cw)
    oy = min(max(oy, t), d - ch)
    x, y = (vt["x"] - ox) * k, (vt["y"] - oy) * k
    kq = _ket_qua(nguon, khung, cach, s0 * k, x, y, tam, so_mat)
    kq["cua_so_ban_ngang"] = {"x": round(ox, 2), "y": round(oy, 2), "rong": round(cw, 2), "cao": round(ch, 2)}
    kq["he_so_phong"] = round(k, 5)
    kq["canh_bao"] = canh_bao
    return kq


def kich_thuoc_nguon(tep: str) -> tuple[int, int]:
    """Width, height after applying rotation metadata."""
    from dailies.quet import ffprobe, xoay
    du_lieu = ffprobe(Path(tep))
    video = next((x for x in du_lieu.get("streams", []) if x.get("codec_type") == "video"), None)
    if video is None:
        raise ValueError(f"Không có luồng hình: {tep}")
    w, h = int(video["width"]), int(video["height"])
    return (h, w) if xoay(video) in (90, 270) else (w, h)


def tam_mat_file(tep: str, vao: float, ra: float, proxy: str | None = None) -> tuple[tuple[float, float] | None, int]:
    """Detect faces on the proxy (the `proxy` passed by the MCP, else Premiere's default location); without one, seek to
    4 points in the original file and decode one frame at each."""
    from dailies.quet import tim_proxy
    proxy = tim_proxy(tep, None, proxy)
    if proxy:
        return tam_mat(proxy, vao, ra)
    import tempfile
    with tempfile.TemporaryDirectory() as tam_dir:
        cac_anh = []
        for k in range(4):
            g = vao + (ra - vao) * (k + 0.5) / 4
            dich = Path(tam_dir) / f"{k}.png"
            kq = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{g:.3f}", "-i", tep, "-frames:v", "1",
                                 "-vf", "scale=320:-2", "-y", str(dich)], capture_output=True)
            if kq.returncode == 0 and dich.is_file():
                cac_anh.append(dich)
        return _tam_mat_anh(cac_anh)


def _tam_mat_anh(cac_anh: list[Path]) -> tuple[tuple[float, float] | None, int]:
    import cv2
    import numpy as np
    from dailies import thi_giac
    tam, nhieu_nhat = [], 0
    for p in cac_anh:
        a = cv2.imread(str(p))
        if a is None:
            continue
        cao, rong = a.shape[:2]
        bo = thi_giac._bo_do_mat(rong, cao)
        if bo is None:
            return None, 0
        _, mat = bo.detect(np.ascontiguousarray(a))
        if mat is None:
            continue
        nhieu_nhat = max(nhieu_nhat, len(mat))
        dt = mat[:, 2] * mat[:, 3]
        tam.append((float(((mat[:, 0] + mat[:, 2] / 2) * dt).sum() / dt.sum()) / rong,
                    float(((mat[:, 1] + mat[:, 3] / 2) * dt).sum() / dt.sum()) / cao))
    if not tam:
        return None, 0
    return (sum(x for x, _ in tam) / len(tam), sum(y for _, y in tam) / len(tam)), nhieu_nhat


def doi_ban(du_lieu: dict, cac_khung: list[tuple[int, int]]) -> dict:
    """Square / vertical versions from the landscape clip list (input of `dailies bo-cuc`, see README.md)."""
    khung_goc = (int(du_lieu["khung"]["rong"]), int(du_lieu["khung"]["cao"]))
    da_do: dict = {}
    ket_qua = [{"khung": {"rong": W, "cao": H}, "clip": []} for W, H in cac_khung]
    for c in du_lieu.get("clip", []):
        tep = c.get("file")
        if not tep or not Path(tep).is_file():
            for b in ket_qua:
                b["clip"].append({"id": c.get("id"), "file": tep, "bo_cuc": None, "ly_do": "khong_co_file"})
            continue
        try:
            nguon = kich_thuoc_nguon(tep)
        except (ValueError, subprocess.CalledProcessError, KeyError):
            for b in ket_qua:
                b["clip"].append({"id": c.get("id"), "file": tep, "bo_cuc": None, "ly_do": "khong_doc_duoc_hinh"})
            continue
        vao = float(c.get("vao_giay", 0))
        ra = float(c.get("ra_giay", vao + 1))
        khoa = (tep, round(vao, 2), round(ra, 2))
        if khoa not in da_do:
            da_do[khoa] = tam_mat_file(tep, vao, ra, c.get("proxy"))
        tam, so_mat = da_do[khoa]
        for (W, H), b in zip(cac_khung, ket_qua):
            bc = tu_ban_ngang(nguon, khung_goc, c, (W, H), tam, so_mat)
            b["clip"].append({"id": c.get("id"), "file": tep, "bo_cuc": bc, **({} if bc else {"ly_do": "khong_hien_o_ban_ngang"})})
    return {"phien_ban": "1", "khung_goc": {"rong": khung_goc[0], "cao": khung_goc[1]}, "ban": ket_qua}
