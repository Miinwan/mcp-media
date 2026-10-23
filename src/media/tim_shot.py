"""Tìm shot bằng câu chữ ("cô gái cầm menu", "cận mặt", "toàn cảnh nhà hàng") với SigLIP 2 chạy tại máy.

Ảnh mẫu đọc từ proxy (anh tạo trong Premiere) mỗi `buoc` giây, mã hoá một lần rồi cache trong `.media/nhung/` theo
mã băm file; mỗi câu tìm chỉ mã hoá câu chữ (gộp vài cách diễn đạt, `cac_cach_noi`) rồi so cosine, nên tìm lại
câu khác gần như tức thì. Điểm từng mẫu được
làm mượt (trung bình 3 mẫu), mỗi đỉnh thành một đoạn: các mẫu liền kề có điểm không thấp hơn đỉnh quá `NGUONG_DOAN`.
Đoạn xếp theo điểm đỉnh trên mọi clip. Điểm là cosine SigLIP, chỉ so được trong cùng một câu tìm.
Hợp đồng JSON: README.md mục 4.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

PHIEN_BAN = "1"
TEN_TEP = "tim_shot.json"
BUOC = 1.0           # giây giữa hai ảnh mẫu
NGUONG_DOAN = 0.015  # cosine; mẫu kề đỉnh còn trong đoạn nếu thấp hơn đỉnh không quá mức này
DOAN_TOI_DA = 20.0   # giây; đoạn dài hơn thì cắt quanh đỉnh
# Thuật ngữ cỡ cảnh của editor: SigLIP hiểu bản tiếng Anh tốt hơn hẳn (thử FSI: "cận mặt" ra cảnh rộng, "close-up of a
# face" ra đúng cảnh cận). Không phải bộ dịch, chỉ thêm cách diễn đạt khi câu có các cụm này.
CO_CANH = {
    "can mat": "a close-up shot of a person's face", "can canh": "a close-up shot",
    "can chi tiet": "an extreme close-up detail shot", "trung canh": "a medium shot of a person from the waist up",
    "trung can": "a medium close-up shot of a person", "toan canh": "a wide shot of the whole scene",
    "toan than": "a full body shot of a person", "goc rong": "a wide angle shot", "tu tren xuong": "a top-down shot",
}


def _nhung_clip(media_id: str, muc: dict, nguon: str, buoc: float, thu_muc: Path) -> np.ndarray:
    """Embedding SigLIP từng ảnh mẫu của một clip (cache theo mã băm, model, bước)."""
    from media import hinh_chu, thi_giac
    khoa = hashlib.sha256(json.dumps([PHIEN_BAN, muc["ma_bam"], hinh_chu.TEN_MODEL, buoc]).encode()).hexdigest()[:16]
    tep = thu_muc / f"{media_id}-{khoa}.npy"
    if tep.is_file():
        return np.load(tep)
    anh = thi_giac.doc_khung(nguon, mau_moi_giay=1 / buoc, rong=384)
    vec = hinh_chu.ma_hoa_anh(anh).astype(np.float32)
    thu_muc.mkdir(parents=True, exist_ok=True)
    tam = tep.with_suffix(".tam.npy")
    np.save(tam, vec)
    tam.replace(tep)
    return vec


def cac_doan(diem: np.ndarray, buoc: float, thoi_luong: float) -> list[dict]:
    """Một đoạn cho mỗi đỉnh của điểm đã làm mượt (trung bình 3 mẫu).

    Mỗi đỉnh chiếm cả sườn đi xuống hai phía (để mẫu kề đỉnh không thành đoạn riêng); đoạn trả về là phần sườn còn
    cao không kém đỉnh quá `NGUONG_DOAN`, tối đa `DOAN_TOI_DA` giây. Mốc đỉnh là mẫu có điểm gốc cao nhất trong đoạn.
    Đoạn liền nhau có điểm sát nhau được gộp. Trả theo điểm giảm dần."""
    if not len(diem):
        return []
    muot = np.convolve(np.pad(diem, 1, mode="edge"), np.ones(3) / 3, mode="valid") if len(diem) > 2 else diem
    da_dung = np.zeros(len(muot), bool)
    doan = []
    for dinh in np.argsort(-muot, kind="stable"):
        if da_dung[dinh]:
            continue
        a = b = dinh
        while a > 0 and not da_dung[a - 1] and muot[a - 1] <= muot[a] + 1e-9:
            a -= 1
        while b + 1 < len(muot) and not da_dung[b + 1] and muot[b + 1] <= muot[b] + 1e-9:
            b += 1
        da_dung[a:b + 1] = True
        san, toi_da = muot[dinh] - NGUONG_DOAN, int(DOAN_TOI_DA / buoc) // 2
        a2, b2 = dinh, dinh
        while a2 > max(a, dinh - toi_da) and muot[a2 - 1] >= san:
            a2 -= 1
        while b2 < min(b, dinh + toi_da) and muot[b2 + 1] >= san:
            b2 += 1
        cao_nhat = a2 + int(np.argmax(diem[a2:b2 + 1]))
        doan.append({"vao": max(0.0, (a2 - 0.5) * buoc), "ra": min(thoi_luong, (b2 + 0.5) * buoc),
                     "diem": float(muot[dinh]), "dinh": cao_nhat * buoc})
    # Hai đoạn liền nhau, điểm sát nhau là cùng một cảnh bị nhiễu tách đôi: gộp (giữ đỉnh cao hơn)
    doan.sort(key=lambda d: d["vao"])
    gop = []
    for d in doan:
        t = gop[-1] if gop else None
        if (t and d["vao"] - t["ra"] <= buoc / 2 and abs(d["diem"] - t["diem"]) <= NGUONG_DOAN
                and d["ra"] - t["vao"] <= DOAN_TOI_DA):
            cao = t if t["diem"] >= d["diem"] else d
            gop[-1] = {"vao": t["vao"], "ra": d["ra"], "diem": cao["diem"], "dinh": cao["dinh"]}
        else:
            gop.append(d)
    return sorted(gop, key=lambda d: -d["diem"])


def cac_cach_noi(cau: str) -> list[str]:
    """Các cách diễn đạt cùng ý để gộp embedding: câu gốc, câu kèm "ảnh chụp" (dạng chú thích ảnh lúc model học),
    và bản tiếng Anh của thuật ngữ cỡ cảnh có trong câu."""
    from media.khau_lenh import bo_dau
    cau = cau.strip()
    ra = [cau, f"ảnh chụp {cau}"]
    khong_dau = " ".join(re.findall(r"[a-z0-9]+", bo_dau(cau)))
    ra += [en for vi, en in CO_CANH.items() if vi in khong_dau]
    return ra


def _khoa_ten(duong: str) -> list:
    from media.khau_lenh import bo_dau
    return [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", bo_dau(duong))]


def tim(cau: str, source: str | Path | None = None, *, danh_sach: str | Path | None = None, cho_phep_goc: bool = False,
        top: int = 10, buoc: float = BUOC, cache_goc: str | Path | None = None) -> dict:
    """Các đoạn khớp `cau` nhất trên mọi clip, xếp theo điểm; ghi `.media/tim_shot.json`."""
    from media import hinh_chu, thu_muc_ket_qua
    from media.khoa import giu_khoa
    from media.quet import nguon_vao, quet_files
    if not cau.strip():
        raise ValueError("Câu tìm đang trống")
    if not hinh_chu.da_co_model():
        raise FileNotFoundError("Chưa có SigLIP 2: chạy media tai-model")
    source, cac_tep, proxy = nguon_vao(source, danh_sach, cho_phep_goc)
    goc = thu_muc_ket_qua(source, cache_goc)
    ket_qua = []
    with giu_khoa(goc, "tim-shot"):
        vec_chu = hinh_chu.ma_hoa_chu(cac_cach_noi(cau)).mean(axis=0)
        vec_chu /= np.linalg.norm(vec_chu)
        for media_id, muc in dict(quet_files(cac_tep)).items():
            duong = Path(muc["duong_dan_gan_nhat"])
            pr = proxy.get(str(duong))
            fps = muc.get("fps") or {"tu": 25, "mau": 1}
            f = Fraction(fps["tu"], fps["mau"])
            thoi_luong = float(Fraction(muc.get("so_frame", 0)) / f)
            vec = _nhung_clip(media_id, muc, str(pr or duong), buoc, goc / "nhung")
            print(f"Tìm shot: {duong.name}", file=sys.stderr, flush=True)
            tuong_doi = str(duong.relative_to(source)) if duong.is_relative_to(source) else duong.name
            for d in cac_doan(vec @ vec_chu, buoc, thoi_luong):
                vao_f, ra_f = round(Fraction(str(d["vao"])) * f), round(Fraction(str(d["ra"])) * f)
                ket_qua.append({"file": str(duong), "ten_file": tuong_doi, "media_id": media_id, "fps": fps,
                                "proxy": str(pr) if pr else None,
                                "vao_giay": round(d["vao"], 3), "ra_giay": round(d["ra"], 3),
                                "do_dai_giay": round(d["ra"] - d["vao"], 3),
                                "vao_frame": vao_f, "ra_frame": max(vao_f + 1, ra_f),
                                "dinh_giay": round(d["dinh"], 3), "diem": round(d["diem"], 4)})
    ket_qua.sort(key=lambda r: (-r["diem"], _khoa_ten(r["ten_file"]), r["vao_giay"]))
    ket_qua = ket_qua[:top]
    for hang, r in enumerate(ket_qua, 1):
        r["hang"] = hang
    kq = {"phien_ban": PHIEN_BAN, "cau": cau.strip(), "thu_muc": str(source), "model": hinh_chu.TEN_MODEL,
          "buoc_giay": buoc, "so_ket_qua": len(ket_qua), "ket_qua": ket_qua}
    dich = goc / TEN_TEP
    tam = dich.with_suffix(".tmp")
    tam.write_text(json.dumps(kq, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tam.replace(dich)
    return {**kq, "file_ket_qua": str(dich)}
