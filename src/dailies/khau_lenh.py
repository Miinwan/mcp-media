"""Rough take detection from on-set voice cues in the original audio: Vietnamese, English and Spanish ("3, 2, 1",
"three two one", "action", "rolling", "diễn", "đi", "tres, dos, uno", "acción", "corriendo" ... "cắt", "cut", "ok",
"okay", "that's good", "corte", "cortamos").

Transcribed with word-level Whisper timestamps (`phan_tich`, cached in `.dailies/phan_tich`); reruns never re-transcribe.
One source can yield several takes; each runs from just after a start cue to just before an end cue, or else to the
next start cue or end of file (trailing silence trimmed). Words that also occur in normal dialogue ("đi", "ok", "cắt",
"cut", "được rồi", "that's good") only count when standing alone between pauses; a 3-2-1 countdown counts anywhere.
"Rolling" followed by another start cue (with no end cue in between) yields and does not become its own take. Output
is sorted a→z by file name, then time, so the editor can reorder in the NLE. JSON contract: Dailies README.md.
"""

from __future__ import annotations

import json
import re
import unicodedata
from fractions import Fraction
from pathlib import Path

PHIEN_BAN = "1"
TEN_TEP = "take_khau_lenh.json"

SO = {"3": "3", "ba": "3", "three": "3", "tres": "3", "2": "2", "hai": "2", "two": "2", "dos": "2",
      "1": "1", "mot": "1", "one": "1", "uno": "1"}
# (diacritic-free phrase, kind, confidence weight, must stand alone)
LENH_VAO = [
    (("action",), "action", 0.9, False), (("acsion",), "action", 0.8, False), (("acshun",), "action", 0.8, False),
    (("ac", "san"), "action", 0.75, True), (("ac", "sion"), "action", 0.75, True), (("ac", "son"), "action", 0.7, True),
    (("dien",), "dien", 0.7, True), (("di",), "di", 0.6, True),
    (("rolling",), "rolling", 0.6, False), (("we", "re", "rolling"), "rolling", 0.65, False),
    (("accion",), "action", 0.85, True), (("y", "accion"), "action", 0.9, False),
    (("corriendo",), "rolling", 0.6, False), (("corre", "camara"), "rolling", 0.6, False),
    (("corriendo", "camara"), "rolling", 0.6, False),
]
LENH_RA = [
    (("cat",), "cat", 0.85, True), (("cut",), "cut", 0.9, True),
    (("ok",), "ok", 0.7, True), (("okay",), "ok", 0.7, True), (("oke",), "ok", 0.7, True), (("o", "ke"), "ok", 0.65, True),
    (("duoc", "roi"), "duoc_roi", 0.7, True),
    (("that", "s", "good"), "thats_good", 0.75, True), (("thats", "good"), "thats_good", 0.75, True),
    (("cortamos",), "corte", 0.85, False), (("y", "cortamos"), "corte", 0.85, False), (("y", "corte"), "corte", 0.85, False), (("corte",), "corte", 0.8, True),
    (("corta",), "corte", 0.75, True),
]
NGUNG = 0.3            # seconds of silence before and after for a word to count as a standalone cue
NOI_DEM_NGUOC = 1.5    # a cue ("diễn", "action") within this window after a countdown merges into one cue
DEM_VAO = 0.1          # seconds of padding after the start cue
DEM_RA = 0.05          # seconds of padding before the end cue
GIU_LANG = 0.3         # seconds of silence kept at the end of a take when trimming
TAKE_NGAN_NHAT = 0.5
RA_YEU = {"ok", "duoc_roi", "thats_good"}  # weak end cues yield to a strong one that follows
NHUONG_RA_YEU = 10


def bo_dau(chu: str) -> str:
    chu = chu.lower().replace("đ", "d")
    return "".join(c for c in unicodedata.normalize("NFD", chu) if unicodedata.category(c) != "Mn")


def _manh(chu: str) -> list[str]:
    """One Whisper word can be "3,2,1", "321" or "Action!": split into word and number pieces."""
    ket_qua = []
    for m in re.findall(r"[a-z]+|\d+", bo_dau(chu)):
        ket_qua.extend(list(m) if m.isdigit() and len(m) == 3 and set(m) <= set("123") else [m])
    return ket_qua


def _cau_bia(cac_chu: list[dict], fps: Fraction, lang: list[tuple[float, float]]) -> bool:
    """Hallucinated Whisper segment ("Cảm ơn các bạn đã theo dõi"): words collapse to almost one timestamp, or mostly sit
    inside a long silence. One- or two-word segments skip the silence check: short per-turn cues can drift into padding ("Cut")."""
    if len(cac_chu) >= 4:
        dai = sorted((t["ra"] - t["vao"]) / fps for t in cac_chu)
        if dai[len(dai) // 2] < 0.05:
            return True
    trong = sum(any(e - s > 1.0 and s <= (t["vao"] + t["ra"]) / 2 / fps <= e for s, e in lang) for t in cac_chu)
    return len(cac_chu) >= 3 and trong >= 0.8 * len(cac_chu)


def _tu_phang(transcript: dict, fps: Fraction, lang: list[tuple[float, float]] = ()) -> list[dict]:
    tu = []
    so_chu = 0
    for cau in transcript.get("cau", []):
        if _cau_bia(cau.get("tu", []), fps, lang):
            continue
        for t in cau.get("tu", []):
            so_chu += 1
            for m in _manh(t["chu"]):
                tu.append({"m": m, "chu": t["chu"], "w": so_chu, "vao": t["vao"] / fps, "ra": t["ra"] / fps,
                           "p": float(t.get("xac_suat", 1.0))})
    tu.sort(key=lambda x: (x["vao"], x["ra"]))
    return tu


def _rieng(tu: list[dict], i: int, j: int) -> bool:
    truoc = tu[i]["vao"] - tu[i - 1]["ra"] if i > 0 else NGUNG
    sau = tu[j + 1]["vao"] - tu[j]["ra"] if j + 1 < len(tu) else NGUNG
    return truoc >= NGUNG - 1e-6 and sau >= NGUNG - 1e-6


def _khop(tu: list[dict], i: int, mau: list[tuple]) -> tuple[int, str, float] | None:
    """Longest cue phrase starting at i: (last word index, kind, weight)."""
    for cum, loai, trong_so, can_rieng in sorted(mau, key=lambda x: -len(x[0])):
        j = i + len(cum) - 1
        if j < len(tu) and tuple(t["m"] for t in tu[i:j + 1]) == cum and (not can_rieng or _rieng(tu, i, j)):
            return j, loai, trong_so
    return None


def _dem_nguoc(tu: list[dict], i: int) -> int | None:
    """Consecutive 3-2-1 (digits or words, under 1.5 s apart): returns the index of "1". Counting "3, 2" and then calling
    the cue ("vamos en 3, 2, acción") also counts: returns the index of "2"; the cue is merged in the next step."""
    if i + 1 >= len(tu) or [SO.get(t["m"]) for t in tu[i:i + 2]] != ["3", "2"] or tu[i + 1]["vao"] - tu[i]["ra"] > 1.5:
        return None
    if i + 2 < len(tu) and SO.get(tu[i + 2]["m"]) == "1" and tu[i + 2]["vao"] - tu[i + 1]["ra"] <= 1.5:
        return i + 2
    kem = _khop(tu, i + 2, [(c, l, w, False) for c, l, w, _ in LENH_VAO]) if i + 2 < len(tu) else None
    if kem and tu[i + 2]["vao"] - tu[i + 1]["ra"] <= NOI_DEM_NGUOC:
        return i + 1
    return None


def _noi(tu: list[dict]) -> str:
    """Pieces of the same Whisper word are printed once."""
    return " ".join(t["chu"] for k, t in enumerate(tu) if k == 0 or tu[k - 1]["w"] != t["w"])


def _moc(tu: list[dict], i: int, j: int, loai: str, trong_so: float) -> dict:
    p = sum(t["p"] for t in tu[i:j + 1]) / (j - i + 1)
    return {"loai": loai, "chu": _noi(tu[i:j + 1]), "vao": tu[i]["vao"], "ra": tu[j]["ra"],
            "i": i, "j": j, "tin_cay": round(trong_so * (0.5 + 0.5 * p), 3)}


def do_khau_lenh(tu: list[dict]) -> tuple[list[dict], list[dict]]:
    """Detect start and end cues in a flat word list (already in seconds)."""
    vao, ra = [], []
    i = 0
    while i < len(tu):
        j = _dem_nguoc(tu, i)
        if j is not None:
            moc = _moc(tu, i, j, "dem_nguoc", 0.95)
            # A cue right after the countdown ("3, 2, 1, diễn") is merged in; it need not stand alone here
            if j + 1 < len(tu) and tu[j + 1]["vao"] - tu[j]["ra"] <= NOI_DEM_NGUOC:
                kem = _khop(tu, j + 1, [(c, l, w, False) for c, l, w, _ in LENH_VAO])
                if kem:
                    moc = _moc(tu, i, kem[0], "dem_nguoc+" + kem[1], 1.0)
            vao.append(moc)
            i = moc["j"] + 1
            continue
        k = _khop(tu, i, LENH_VAO)
        if k:
            moc = _moc(tu, i, k[0], k[1], k[2])
            if vao and vao[-1]["loai"].endswith(k[1]) and moc["vao"] - vao[-1]["ra"] <= NOI_DEM_NGUOC:
                vao[-1]["ra"], vao[-1]["j"] = moc["ra"], moc["j"]  # "action! action!" shouted by several people: one cue
            else:
                vao.append(moc)
            i = k[0] + 1
            continue
        k = _khop(tu, i, LENH_RA)
        if k:
            ra.append(_moc(tu, i, k[0], k[1], k[2]))
            i = k[0] + 1
            continue
        i += 1
    # "Rolling" (camera running) is often followed by the slate and set tweaks before the real call: yield to the next start
    # cue if no end cue sits between them
    vao = [v for n, v in enumerate(vao) if not (v["loai"] == "rolling" and n + 1 < len(vao)
                                               and not any(v["ra"] <= r["vao"] < vao[n + 1]["vao"] for r in ra))]
    return vao, ra


def _lang_cuoi(vao: float, ra: float, khoang_lang: list[tuple[float, float]], tu: list[dict]) -> float:
    """Trim silence touching the end of a take, keeping GIU_LANG seconds; no trim if speech follows the silence."""
    for s, e in khoang_lang:
        if s < ra <= e + 0.05 and s > vao + TAKE_NGAN_NHAT and not any(s < t["vao"] < ra for t in tu):
            return min(ra, s + GIU_LANG)
    return ra


def cat_take(transcript: dict, fps: Fraction, thoi_luong: float, khoang_lang: list[dict] | None = None) -> list[dict]:
    """Takes (in seconds) for one file. A file with no start cue returns an empty list."""
    lang = [(x["vao"] / fps, x["ra"] / fps) for x in khoang_lang or []]
    tu = _tu_phang(transcript, fps, lang)
    lenh_vao, lenh_ra = do_khau_lenh(tu)
    takes = []
    for n, bd in enumerate(lenh_vao):
        sau = lenh_vao[n + 1] if n + 1 < len(lenh_vao) else None
        gioi_han = sau["vao"] if sau else Fraction(thoi_luong)
        vao = bd["ra"] + Fraction(DEM_VAO)
        tiep = next((t for t in tu[bd["j"] + 1:] if t["vao"] >= bd["ra"]), None)
        if tiep is not None and tiep["vao"] < vao:
            vao = tiep["vao"]
        kt = next((r for r in lenh_ra if r["vao"] >= vao + Fraction(TAKE_NGAN_NHAT) and r["vao"] < gioi_han), None)
        if kt and kt["loai"] in RA_YEU:
            # a director's "ok" before "cắt", "cut", "corte": take the strong cue if it arrives within 10 seconds
            manh = next((r for r in lenh_ra if r["loai"] not in RA_YEU and kt["vao"] < r["vao"] < gioi_han
                         and r["vao"] - kt["ra"] <= NHUONG_RA_YEU), None)
            kt = manh or kt
        if kt:
            ra, ket_thuc = kt["vao"] - Fraction(DEM_RA), "khau_lenh_ra"
        else:
            ra = gioi_han - (Fraction(DEM_RA) if sau else 0)
            ket_thuc = "khau_lenh_vao_ke_tiep" if sau else "het_file"
            moi = _lang_cuoi(float(vao), float(ra), lang, tu)
            if moi < ra - Fraction(1, 20):
                ra, ket_thuc = Fraction(str(round(moi, 4))), "lang_cuoi"
        ra = min(ra, Fraction(thoi_luong))
        if ra - vao < Fraction(1, 10):
            continue
        loi = _noi([t for t in tu if vao <= t["vao"] < ra])
        tin_cay = bd["tin_cay"] * (kt["tin_cay"] if kt else 0.9)
        takes.append({"vao": vao, "ra": ra, "khau_lenh_vao": bd, "khau_lenh_ra": kt,
                      "ket_thuc_bang": ket_thuc, "do_tin_cay": round(tin_cay, 3), "loi_thoai": loi})
    return takes


def _khoa_ten(duong: str) -> list:
    """Natural sort: A2 before A10, case-insensitive."""
    return [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", bo_dau(duong))]


def _giay(x: Fraction) -> float:
    return round(float(x), 3)


def _lenh_json(moc: dict | None) -> dict | None:
    if not moc:
        return None
    return {"loai": moc["loai"], "chu": moc["chu"], "vao_giay": _giay(moc["vao"]), "ra_giay": _giay(moc["ra"]),
            "do_tin_cay": moc["tin_cay"]}


def lap_danh_sach(source: Path, media: dict[str, dict], cache: dict[str, str], *, giu_file_khong_lenh: bool = True,
                  khung: tuple[int, int] = (1920, 1080)) -> dict:
    """Assemble takes from all files into the stable JSON result (Dailies README.md), with framing for sequence `khung`."""
    from dailies import bo_cuc
    hang, khong_lenh = [], []
    for media_id, muc in media.items():
        phan_tich = json.loads(Path(cache[media_id]).read_text(encoding="utf-8"))
        fps_d = phan_tich["fps"]
        fps = Fraction(fps_d["tu"], fps_d["mau"])
        fps_nguon = muc.get("fps") or fps_d
        thoi_luong = Fraction(muc["so_frame"], 1) / Fraction(fps_nguon["tu"], fps_nguon["mau"]) if muc.get("so_frame") else None
        if thoi_luong is None:
            cuoi = [t["ra"] for c in phan_tich["transcript"]["cau"] for t in c.get("tu", [])]
            thoi_luong = Fraction(max(cuoi, default=0), 1) / fps
        duong = Path(muc["duong_dan_gan_nhat"])
        tuong_doi = str(duong.relative_to(source)) if duong.is_relative_to(source) else duong.name
        chung = {"file": str(duong), "ten_file": tuong_doi, "media_id": media_id, "fps": fps_d,
                 "thoi_luong_file_giay": _giay(thoi_luong), "ngon_ngu": phan_tich["transcript"].get("ngon_ngu")}
        takes = cat_take(phan_tich["transcript"], fps, thoi_luong, phan_tich.get("khoang_lang"))
        if not takes:
            ly_do = "khong_co_tieng" if muc["am_thanh"]["kieu"] != "trong_file" else "khong_nghe_thay_khau_lenh"
            khong_lenh.append({**chung, "ly_do": ly_do})
            if giu_file_khong_lenh:
                takes = [{"vao": Fraction(0), "ra": thoi_luong, "khau_lenh_vao": None, "khau_lenh_ra": None,
                          "ket_thuc_bang": "het_file", "do_tin_cay": 0.0, "loi_thoai": "", "ca_file": True}]
        from dailies.quet import kich_thuoc_hien
        nguon = kich_thuoc_hien(muc["duong_dan_gan_nhat"])  # rotation and pixel aspect applied
        cung_ti_le = abs(nguon[0] / nguon[1] - khung[0] / khung[1]) < 0.01 * khung[0] / khung[1]
        proxy = (phan_tich.get("proxy") or {}).get("duong_dan")
        for so, t in enumerate(takes, 1):
            tam, so_mat = (None, 0)
            if not cung_ti_le and proxy and Path(proxy).is_file():  # only detect faces when the frame must be filled
                tam, so_mat = bo_cuc.tam_mat(proxy, float(t["vao"]), float(t["ra"]))
            vao_f = round(t["vao"] * fps)
            ra_f = max(vao_f + 1, round(t["ra"] * fps))
            hang.append({**chung, "take_trong_file": so,
                         "vao_giay": _giay(t["vao"]), "ra_giay": _giay(t["ra"]), "do_dai_giay": _giay(t["ra"] - t["vao"]),
                         "vao_frame": vao_f, "ra_frame": ra_f,
                         "khau_lenh_vao": _lenh_json(t["khau_lenh_vao"]), "khau_lenh_ra": _lenh_json(t["khau_lenh_ra"]),
                         "ket_thuc_bang": t["ket_thuc_bang"], "do_tin_cay": t["do_tin_cay"],
                         "ca_file": bool(t.get("ca_file")), "loi_thoai": t["loi_thoai"],
                         "bo_cuc": bo_cuc.tinh(nguon, khung, tam, so_mat)})
    hang.sort(key=lambda h: (_khoa_ten(h["ten_file"]), h["vao_giay"]))
    for stt, h in enumerate(hang, 1):
        h["stt"] = stt
        h["id"] = f"{h['media_id']}-t{h['take_trong_file']:02d}"
    khong_lenh.sort(key=lambda h: _khoa_ten(h["ten_file"]))
    return {"phien_ban": PHIEN_BAN, "thu_muc": str(source), "khung": {"rong": khung[0], "cao": khung[1]},
            "so_file": len(media), "so_take": len(hang),
            "take": hang, "file_khong_co_khau_lenh": khong_lenh}


def take_thu_muc(source: str | Path | None = None, *, danh_sach: str | Path | None = None, cho_phep_goc: bool = False,
                 model_ten: str = "large-v3", cache_goc: str | Path | None = None, fps: dict | None = None,
                 model_factory=None, giu_file_khong_lenh: bool = True, khung: tuple[int, int] = (1920, 1080),
                 ngon_ngu: str = "auto") -> dict:
    """Detect takes for every clip in a folder or list (reusing the transcript cache), writes `.dailies/take_khau_lenh.json`."""
    from dailies import thu_muc_ket_qua
    from dailies.khoa import giu_khoa
    from dailies.phan_tich import phan_tich_thu_muc
    from dailies.quet import nguon_vao
    source, cac_tep, proxy = nguon_vao(source, danh_sach, cho_phep_goc)
    goc = thu_muc_ket_qua(source, cache_goc)
    with giu_khoa(goc, "take-khau-lenh"):
        media, tom_tat = phan_tich_thu_muc(goc, cac_tep, proxy, model_ten=model_ten, fps_dat=fps,
                                          model_factory=model_factory, ngon_ngu=ngon_ngu)
        kq = {**lap_danh_sach(source, media, tom_tat["cache"], giu_file_khong_lenh=giu_file_khong_lenh, khung=khung),
              "model": model_ten, "ngon_ngu_dat": ngon_ngu}
    dich = goc / TEN_TEP
    dich.parent.mkdir(parents=True, exist_ok=True)
    tam = dich.with_suffix(".tmp")
    tam.write_text(json.dumps(kq, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tam.replace(dich)
    return {**kq, "file_ket_qua": str(dich)}
