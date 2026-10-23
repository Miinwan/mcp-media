"""Bố cục cho rough cut: scale và vị trí từng take để đặt footage (thường 4K) vào sequence nhẹ hơn (Full HD, dọc 9:16).

Cùng tỉ lệ khung: vừa khung (4K vào Full HD là 50%), giữ nguyên bố cục máy quay. Khác tỉ lệ: phóng cho đầy khung rồi
dời theo mặt người (YuNet trên proxy, vài ảnh trong take): mặt nằm giữa theo chiều ngang, tâm mặt ở 40% chiều cao khung
(chừa khoảng trên đầu); không thấy mặt thì lấy giữa. Vị trí luôn bị giới hạn để hình phủ kín khung, không lộ viền đen.
Vị trí theo quy ước Motion của Premiere: điểm neo là tâm clip, đơn vị pixel của sequence; kèm bản chuẩn hoá 0-1.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

TAM_MAT_DOC = 0.4
SO_ANH_TOI_DA = 8


def doc_khung_chuoi(chuoi: str) -> tuple[int, int]:
    """'1920x1080' hoặc '9:16' (hiểu là 1080x1920), '16:9' (1920x1080)."""
    chuoi = chuoi.strip().lower()
    tat = {"16:9": (1920, 1080), "9:16": (1080, 1920), "1:1": (1080, 1080), "4:5": (1080, 1350)}
    if chuoi in tat:
        return tat[chuoi]
    rong, _, cao = chuoi.partition("x")
    if not (rong.isdigit() and cao.isdigit()) or not int(rong) or not int(cao):
        raise ValueError(f"Khung sequence không hợp lệ: {chuoi!r} (ví dụ 1920x1080, 9:16)")
    return int(rong), int(cao)


def tam_mat(proxy: str, vao: float, ra: float) -> tuple[tuple[float, float] | None, int]:
    """Tâm trung bình các mặt (chuẩn hoá 0-1 theo khung nguồn) trong đoạn [vao, ra) của proxy; kèm số mặt nhiều nhất.
    Giải mã qua ffmpeg dòng lệnh, cv2 chỉ chạy mạng dò mặt (quyết định 017)."""
    import cv2
    import numpy as np
    from media import thi_giac
    dai = max(0.1, ra - vao)
    fps = min(1.0, SO_ANH_TOI_DA / dai)
    kq = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                         "-of", "csv=p=0", proxy], capture_output=True, text=True, check=True).stdout.split(",")
    w, h = int(kq[0]), int(kq[1])
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
    """Tâm clip để điểm `diem` (0-1 theo nguồn) nằm giữa ngang và ở `doc` chiều cao khung; giới hạn để phủ kín khung."""
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
    """Scale (%) và vị trí tâm clip trong sequence."""
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
    """Phần khung ngang có hình của clip (trái, trên, phải, dưới theo pixel khung ngang): khung clip sau scale, vị trí,
    điểm neo và effect Crop, giao với khung ngang. None khi clip không hiện trong khung."""
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
    """Bố cục bản vuông, dọc của một clip, cắt ra từ chính khung ngang anh đã chỉnh.

    Coi khung ngang (sau scale, vị trí, điểm neo, Crop của anh) là ảnh gốc; bản mới là một cửa sổ của ảnh đó phóng cho
    đầy khung mới, nên pixel giống hệt bản ngang, không bao giờ hiện vùng anh đã cắt bỏ hay viền đen. Cửa sổ chỉ nằm
    trong phần có hình của clip; phần đó hẹp hơn cửa sổ thì phóng thêm. Có mặt người nằm trong khung ngang thì cửa sổ
    dời cho mặt vào giữa ngang, tâm mặt ở 40% chiều cao (trong giới hạn); không thì lấy giữa phần có hình.
    Vị trí mới = (vị trí ngang - góc cửa sổ) x hệ số phóng, đúng với mọi điểm neo; scale mới = scale ngang x hệ số."""
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
    cw, ch = W / k, H / k  # cửa sổ theo pixel khung ngang
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
    """Rộng, cao sau khi áp xoay metadata."""
    from media.quet import ffprobe, xoay
    du_lieu = ffprobe(Path(tep))
    video = next((x for x in du_lieu.get("streams", []) if x.get("codec_type") == "video"), None)
    if video is None:
        raise ValueError(f"Không có luồng hình: {tep}")
    w, h = int(video["width"]), int(video["height"])
    return (h, w) if xoay(video) in (90, 270) else (w, h)


def tam_mat_file(tep: str, vao: float, ra: float, proxy: str | None = None) -> tuple[tuple[float, float] | None, int]:
    """Dò mặt trên proxy (`proxy` MCP truyền, không thì theo mặc định Premiere); không có thì nhảy tới 4 mốc trong file
    gốc, mỗi mốc giải mã một ảnh."""
    from media.quet import tim_proxy
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
    from media import thi_giac
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
    """Bản vuông, dọc từ danh sách clip bản ngang (đầu vào lệnh `media bo-cuc`, README.md)."""
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
