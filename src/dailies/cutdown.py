"""Read a Cutdown session (`.cdwn`) and compare it with `dailies cham-diem`, to calibrate Dailies' scoring.

Cutdown (Softworks) is a separate, commercial macOS app; Dailies only reads the session files it saves (interoperability),
never its code. Layout of format 9, worked out from the files:
`b"CDWN"`, u32 version, u32 length + zstd(JSON: timeline, media pool, preset, segments per clip), then u32 clip count and
per clip 16-byte clip id, 8 bytes, u32 length + zstd(MessagePack: per-frame scores and flags). Little-endian.
"""

from __future__ import annotations

import json
import os
import struct
import uuid
from pathlib import Path

import numpy as np

PHIEN_BAN_HO_TRO = {9}
LOP_MAY = {"static": "dung_yen", "pan_left": "lia_trai", "pan_right": "lia_phai", "tilt_up": "nghieng_len",
           "tilt_down": "nghieng_xuong", "zoom_in": "zoom_vao", "zoom_out": "zoom_ra", "twist_ccw": "xoay_trai",
           "twist_cw": "xoay_phai", "mixed": "hon_hop"}


def _giai(khoi: bytes) -> bytes:
    import zstandard
    return zstandard.ZstdDecompressor().decompressobj().decompress(khoi)


def _ty(fps: dict) -> float:
    return fps["numerator"] / fps["denominator"]


def doc_cdwn(duong_dan: str | Path) -> dict:
    """Clips with Cutdown's segments (seconds from the clip start) and per-sample scores."""
    import msgpack
    b = Path(duong_dan).read_bytes()
    if b[:4] != b"CDWN":
        raise ValueError(f"Không phải file Cutdown: {duong_dan}")
    phien_ban = struct.unpack_from("<I", b, 4)[0]
    if phien_ban not in PHIEN_BAN_HO_TRO:
        raise ValueError(f"File Cutdown phiên bản {phien_ban}, Dailies mới đọc được {sorted(PHIEN_BAN_HO_TRO)}")
    n = struct.unpack_from("<I", b, 8)[0]
    phien = json.loads(_giai(b[12:12 + n]))
    p = 12 + n
    khung = {}
    if p + 4 <= len(b):
        so = struct.unpack_from("<I", b, p)[0]
        p += 4
        for _ in range(so):
            ma = str(uuid.UUID(bytes=b[p:p + 16]))
            dai = struct.unpack_from("<I", b, p + 24)[0]
            p += 28
            khung[ma] = msgpack.unpackb(_giai(b[p:p + dai]), raw=False, strict_map_key=False)
            p += dai
    tl = phien["timeline"]
    media = {m["id"]: m for m in tl["media_pool"]}
    clip_tl = {c["id"]: c for tr in tl["tracks"] for c in tr["clips"]}
    clip = []
    for r in phien.get("results", []):
        c = clip_tl.get(r["clip_instance_id"])
        m = media.get(c["source_media_id"]) if c else None
        if m is None:
            continue
        tc = c["tc_mapping"]
        fps_tl, fps_src = _ty(tc["timeline_fps"]), _ty(tc["source_fps"])
        lech = tc.get("source_start_frame", 0) / fps_src - c["timeline_in"] / fps_tl
        doan = [{"vao_giay": round(s["timeline_start_frame"] / fps_tl + lech, 3),
                 "ra_giay": round((s["timeline_end_frame"] + 1) / fps_tl + lech, 3),
                 "diem_tb": round(s["mean_score"], 4), "diem_dinh": round(s["peak_score"], 4),
                 "may": LOP_MAY.get((s.get("camera_motion_summary") or {}).get("class"), "hon_hop")} for s in r["segments"]]
        fps_media = _ty(m["video"]["fps"]) if m.get("video") else fps_src
        mau = [{"t": round(f["source_frame_index"] / fps_media, 4), "diem": round(f["composite_score"], 4),
                "dat": bool(f.get("qualifies")),
                "co": sorted(k.removeprefix("is_").removeprefix("has_") for k, v in (f.get("flags") or {}).items() if v),
                "chi_so": {k: round(v, 4) for k, v in (f.get("scores") or {}).items()}}
               for f in khung.get(r["clip_instance_id"], [])]
        clip.append({"file": m.get("original_path") or m["path"], "proxy": m.get("proxy_path"),
                     "ten_file": m.get("name") or Path(m["path"]).name, "doan": doan, "mau": mau})
    pr = phien.get("preset", {})
    return {"phien_ban": phien_ban, "kieu": pr.get("name"), "nguong": pr.get("score_threshold"),
            "mau_moi_giay": (pr.get("sampling_rate") or {}).get("PerSecond"), "so_clip": len(clip), "clip": clip}


def _hang(x: np.ndarray) -> np.ndarray:
    """Ranks with ties averaged."""
    thu_tu = np.argsort(x, kind="mergesort")
    h = np.empty(len(x))
    h[thu_tu] = np.arange(len(x))
    for v in np.unique(x):
        cung = x == v
        if cung.sum() > 1:
            h[cung] = h[cung].mean()
    return h


def spearman(a, b) -> float | None:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3:
        return None
    ha, hb = _hang(a), _hang(b)
    if ha.std() == 0 or hb.std() == 0:
        return None
    return round(float(np.corrcoef(ha, hb)[0, 1]), 3)


def _do_dai(ds: list[dict]) -> float:
    return sum(d["ra_giay"] - d["vao_giay"] for d in ds)


def _giao(a: list[dict], b: list[dict]) -> float:
    return sum(max(0.0, min(x["ra_giay"], y["ra_giay"]) - max(x["vao_giay"], y["vao_giay"])) for x in a for y in b)


def doi_chieu(cdwn: str | Path, ket_qua: str | Path | None = None) -> dict:
    """Agreement between Cutdown and `dailies cham-diem` on the same clips (matched by file name)."""
    cd = doc_cdwn(cdwn)
    if ket_qua is None:
        if not cd["clip"]:
            raise ValueError("File Cutdown không có clip nào")
        goc = Path(os.path.commonpath([str(Path(c["file"]).parent) for c in cd["clip"]]))
        ket_qua = goc / ".dailies" / "cham_diem.json"
    ket_qua = Path(ket_qua)
    if not ket_qua.is_file():
        raise ValueError(f"Chưa có {ket_qua}; chạy dailies cham-diem trên thư mục đó trước")
    ta = json.loads(ket_qua.read_text(encoding="utf-8"))
    cua_ta = {Path(c["file"]).name.lower(): c for c in ta["clip"]}
    dong, mau_ta, mau_cd, tb_ta, tb_cd = [], [], [], [], []
    dong_y = khop_may = so_cap_may = 0
    for c in cd["clip"]:
        t = cua_ta.get(Path(c["file"]).name.lower())
        if t is None:
            continue
        doan_ta, doan_cd = t["doan"], c["doan"]
        giao = _giao(doan_ta, doan_cd)
        hop = _do_dai(doan_ta) + _do_dai(doan_cd) - giao
        chon_ta, chon_cd = bool(doan_ta), bool(doan_cd)
        dong_y += chon_ta == chon_cd
        for x in doan_cd:
            y = max(doan_ta, key=lambda d: _giao([d], [x]), default=None)
            if y is not None and _giao([y], [x]) > 0:
                so_cap_may += 1
                khop_may += y["may"] == x["may"]
        dong.append({"ten_file": t["ten_file"], "doan_dailies": len(doan_ta), "doan_cutdown": len(doan_cd),
                     "giay_dailies": round(_do_dai(doan_ta), 2), "giay_cutdown": round(_do_dai(doan_cd), 2),
                     "trung_khop": round(giao / hop, 3) if hop > 0 else 1.0,
                     "diem_tb_dailies": t["diem_tb"], "diem_tb_cutdown": round(float(np.mean([m["diem"] for m in c["mau"]])), 4) if c["mau"] else None})
        if t["diem_tb"] is not None and dong[-1]["diem_tb_cutdown"] is not None:
            tb_ta.append(t["diem_tb"])
            tb_cd.append(dong[-1]["diem_tb_cutdown"])
    for c in cd["clip"]:
        t = cua_ta.get(Path(c["file"]).name.lower())
        if t is None or not c["mau"] or not Path(t["tep_mau"]).is_file():
            continue
        mau = json.loads(Path(t["tep_mau"]).read_text(encoding="utf-8"))["mau"]
        diem = _diem_media(mau, ta)
        if not diem:
            continue
        tg = np.array([d["t"] for d in diem])
        for m in c["mau"]:
            i = int(np.abs(tg - m["t"]).argmin())
            if abs(tg[i] - m["t"]) <= 0.15:
                mau_ta.append(diem[i]["diem"])
                mau_cd.append(m["diem"])
    gio_ta = sum(d["giay_dailies"] for d in dong)
    gio_cd = sum(d["giay_cutdown"] for d in dong)
    dong.sort(key=lambda d: d["trung_khop"])
    return {"cutdown": {"file": str(cdwn), "phien_ban": cd["phien_ban"], "kieu": cd["kieu"], "nguong": cd["nguong"],
                        "so_clip": cd["so_clip"]},
            "dailies": {"file": str(ket_qua), "kieu": ta["kieu"], "nguong": ta["cai_dat"]["nguong"]},
            "so_clip_khop": len(dong),
            "dong_y_chon_bo": round(dong_y / len(dong), 3) if dong else None,
            "trung_khop_tb": round(float(np.mean([d["trung_khop"] for d in dong])), 3) if dong else None,
            "giay_chon": {"dailies": round(gio_ta, 1), "cutdown": round(gio_cd, 1)},
            "tuong_quan_mau": spearman(mau_ta, mau_cd), "so_mau_so_sanh": len(mau_ta),
            "tuong_quan_clip": spearman(tb_ta, tb_cd),
            "khop_chuyen_dong_may": round(khop_may / so_cap_may, 3) if so_cap_may else None,
            "lech_nhieu_nhat": dong[:15], "clip": dong}


def _diem_media(mau: list[dict], ta: dict) -> list[dict]:
    from dailies.cham_diem import cham
    return cham(mau, ta["tham_chieu"], ta["kieu"])
