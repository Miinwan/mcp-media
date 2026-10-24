"""Read every clip's properties before editing, and recommend the sequence and how each clip goes into it.

Read without decoding a frame: ffprobe (codec, profile, bit depth, chroma, colour tags, size, rotation, pixel aspect,
frame rate, variable frame rate, field order, timecode, audio, make and model tags) plus the XML Sony writes inside
XAVC files (`NonRealTimeMeta`: capture and file frame rate, so slow motion and time-lapse recorded in camera show up;
capture gamma such as S-Log3, which ffprobe reports as unknown; camera model). Other log formats are recognised from
colour tags (HLG, PQ) or from words in the file's metadata (Log3G10, C-Log, V-Log, F-Log, D-Log, Apple Log, LogC), which
is less certain and marked so.

Recommendation: the sequence takes the delivery frame and rate when given, otherwise the most common displayed size
and the most common frame rate among normal-speed clips. Every clip that is not exactly the sequence size is fit to the
frame (4K in Full HD is 50%; another aspect ratio fills the frame, centred), using the displayed size after rotation
and pixel aspect. Frame rate relations, in-camera slow motion and time-lapse, log or HDR, variable frame rate and
interlacing are flagged with what to do.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path

PHIEN_BAN = 1
TEN_TEP = "thuoc_tinh.json"
KHOI_XML = 2_000_000
# Words in metadata that name a log or HDR curve, checked in this order (most specific first)
TU_KHOA_LOG = [
    (r"s-?log3", "S-Log3"), (r"s-?log2", "S-Log2"), (r"log3g10", "RED Log3G10"), (r"logc4", "ARRI LogC4"),
    (r"logc3?\b|log ?c\b", "ARRI LogC3"), (r"c-?log3|canon ?log ?3", "Canon Log 3"), (r"c-?log2|canon ?log ?2", "Canon Log 2"),
    (r"v-?log", "Panasonic V-Log"), (r"f-?log2", "Fujifilm F-Log2"), (r"f-?log", "Fujifilm F-Log"),
    (r"d-?log ?m", "DJI D-Log M"), (r"d-?log", "DJI D-Log"), (r"apple ?log", "Apple Log"), (r"n-?log", "Nikon N-Log"),
    (r"film ?gen ?5|bmd ?film", "Blackmagic Film Gen 5"),
]
GAMMA_SONY = {"s-log3-cine": "S-Log3", "s-log3": "S-Log3", "s-log2": "S-Log2", "hlg": "HLG", "hlg1": "HLG", "hlg2": "HLG",
              "hlg3": "HLG", "rec709": None, "rec709-xvycc": None, "s-cinetone": None, "still": None}


def _ty_le(chuoi: str | None) -> Fraction | None:
    try:
        f = Fraction(chuoi or "0")
    except (ValueError, ZeroDivisionError):
        return None
    return f if f > 0 else None


def _fps_sony(chuoi: str | None) -> Fraction | None:
    """'59.94p', '23.98p', '3.74p', '120p' to a frame rate (NTSC values snapped to x/1001)."""
    if not chuoi:
        return None
    m = re.match(r"([\d.]+)", chuoi)
    if not m:
        return None
    v = float(m.group(1))
    for chuan in (23.976, 29.97, 47.952, 59.94, 119.88, 239.76):
        if abs(v - chuan) < 0.011:
            return Fraction(round(chuan * 1.001) * 1000, 1001)
    return Fraction(v).limit_denominator(1000)


def xml_sony(duong: str) -> dict | None:
    """NonRealTimeMeta written by Sony cameras inside the file (start or end)."""
    try:
        co = os.path.getsize(duong)
        with open(duong, "rb") as f:
            for vi_tri in (max(0, co - KHOI_XML), 0):
                f.seek(vi_tri)
                d = f.read(KHOI_XML)
                i = d.find(b"<NonRealTimeMeta")
                if i >= 0:
                    j = d.find(b"</NonRealTimeMeta>", i)
                    x = d[i:j if j > 0 else len(d)].decode("utf-8", "replace")
                    lay = lambda mau: (re.search(mau, x) or [None, None])[1]
                    return {"fps_quay": lay(r'captureFps="([^"]+)"'), "fps_file": lay(r'formatFps="([^"]+)"'),
                            "gamma": lay(r'CaptureGammaEquation" value="([^"]+)"'),
                            "gam_mau": lay(r'CaptureColorPrimaries" value="([^"]+)"'),
                            "hang": lay(r'<Device manufacturer="([^"]+)"'), "may": lay(r'modelName="([^"]+)"'),
                            "che_do_ghi": lay(r'<RecordingMode type="([^"]+)"')}
    except OSError:
        return None
    return None


def _log_tu_khoa(the: dict) -> str | None:
    chu = " ".join(f"{k} {v}" for k, v in the.items()).lower()
    for mau, ten in TU_KHOA_LOG:
        if re.search(mau, chu):
            return ten
    return None


def doc_thuoc_tinh(tep: Path, muc: dict, proxy: str | None) -> dict:
    """All properties of one clip, from ffprobe and embedded camera XML."""
    from dailies.quet import ffprobe, xoay
    from dailies.xuat import frame_sang_tc, tc_sang_frame
    du = ffprobe(tep)
    v = next((s for s in du["streams"] if s.get("codec_type") == "video"), {})
    a = [s for s in du["streams"] if s.get("codec_type") == "audio"]
    fmt = du.get("format", {})
    the = {**(fmt.get("tags") or {}), **(v.get("tags") or {})}
    fps = _ty_le(v.get("r_frame_rate"))
    fps_tb = _ty_le(v.get("avg_frame_rate"))
    goc = xoay(v)
    w, h = int(v.get("width") or 0), int(v.get("height") or 0)
    par = _ty_le((v.get("sample_aspect_ratio") or "1:1").replace(":", "/")) or Fraction(1)
    rong_hien, cao_hien = (round(w * par), h) if par != 1 else (w, h)
    if goc in (90, 270):
        rong_hien, cao_hien = cao_hien, rong_hien
    pix = v.get("pix_fmt") or ""
    bit = int(v.get("bits_per_raw_sample") or 0) or (12 if "12" in pix else 10 if "10" in pix else 8 if pix else 0)
    sony = xml_sony(str(tep))
    fps_quay = _fps_sony(sony["fps_quay"]) if sony else None
    # Log / HDR: Sony XML first (certain), then colour tags (certain), then metadata words (less certain)
    log, nguon_log = None, None
    if sony and sony.get("gamma"):
        g = sony["gamma"].lower()
        log = GAMMA_SONY.get(g, sony["gamma"])
        nguon_log = "xml_sony"
    if not log:
        tr = (v.get("color_transfer") or "").lower()
        if tr == "arib-std-b67":
            log, nguon_log = "HLG", "the_mau"
        elif tr == "smpte2084":
            log, nguon_log = "PQ (HDR10)", "the_mau"
    if not log and not sony:
        tk = _log_tu_khoa(the)
        if tk:
            log, nguon_log = tk, "tu_khoa_metadata"
    tc = the.get("timecode") or next(((s.get("tags") or {}).get("timecode") for s in du["streams"] if (s.get("tags") or {}).get("timecode")), None)
    vfr = bool(fps and fps_tb and abs(float(fps_tb) - float(fps)) / float(fps) > 0.01)
    may = (sony or {}).get("may") or the.get("com.apple.quicktime.model") or the.get("model")
    hang = (sony or {}).get("hang") or the.get("com.apple.quicktime.make") or the.get("make")
    so_frame = muc.get("so_frame") or 0
    return {
        "file": str(tep), "ten_file": tep.name, "media_id": None, "proxy": proxy,
        "vo": fmt.get("format_name"), "codec": v.get("codec_name"), "profile": v.get("profile"), "bit": bit,
        "pix_fmt": pix, "chroma": "4:2:2" if "422" in pix else "4:4:4" if "444" in pix else "4:2:0" if pix else None,
        "dai_mau": "full" if v.get("color_range") == "pc" or pix.startswith("yuvj") else "limited",
        "the_mau": {"primaries": v.get("color_primaries"), "transfer": v.get("color_transfer"), "matrix": v.get("color_space")},
        "rong": w, "cao": h, "xoay": goc, "par": f"{par.numerator}:{par.denominator}",
        "rong_hien": rong_hien, "cao_hien": cao_hien,
        "fps": {"tu": fps.numerator, "mau": fps.denominator} if fps else None, "fps_so": round(float(fps), 3) if fps else None,
        "fps_quay": round(float(fps_quay), 3) if fps_quay else None, "vfr": vfr,
        "quet_xen": v.get("field_order") not in (None, "progressive", "unknown"),
        "so_frame": so_frame, "thoi_luong_giay": round(so_frame / float(fps), 3) if fps and so_frame else round(float(fmt.get("duration") or 0), 3),
        "timecode": frame_sang_tc(tc_sang_frame(tc, fps), fps) if tc and fps else tc,
        "log": log, "nguon_log": nguon_log, "hang": hang, "may": may, "ngay_quay": the.get("creation_time"),
        "am_thanh": {"kenh": sum(int(x.get("channels") or 0) for x in a), "tan_so": int(a[0].get("sample_rate") or 0) if a else 0,
                     "codec": a[0].get("codec_name") if a else None},
    }


def _ten_fps(f: float) -> str:
    for chuan, ten in ((23.976, "23.976"), (29.97, "29.97"), (59.94, "59.94"), (119.88, "119.88")):
        if abs(f - chuan) < 0.01:
            return ten
    return f"{f:g}"


def de_xuat(clip: list[dict], khung: tuple[int, int] | None = None, fps: Fraction | None = None) -> dict:
    """Sequence settings and what to do with each clip. Pure, for tests."""
    from dailies.bo_cuc import tinh
    binh_thuong = [c for c in clip if c["fps"] and not (c["fps_quay"] and abs(c["fps_quay"] - c["fps_so"]) > 0.05)]
    if fps is None:
        dem = Counter((c["fps"]["tu"], c["fps"]["mau"]) for c in (binh_thuong or clip) if c["fps"])
        fps = Fraction(*dem.most_common(1)[0][0]) if dem else Fraction(25)
        ly_do_fps = "fps có nhiều clip quay tốc độ thường nhất"
    else:
        ly_do_fps = "fps bản giao anh chọn"
    if khung is None:
        dem = Counter((c["rong_hien"], c["cao_hien"]) for c in clip if c["rong_hien"])
        khung = dem.most_common(1)[0][0] if dem else (1920, 1080)
        ly_do_khung = "kích thước có nhiều clip nhất"
    else:
        ly_do_khung = "khung bản giao anh chọn"
    W, H = khung
    f_seq = float(fps)
    ket_qua, canh_bao = [], Counter()
    for c in clip:
        xu_ly, chu_y = {}, []
        # Size: every clip not exactly the sequence frame is fit to it
        if (c["rong_hien"], c["cao_hien"]) == (W, H) and c["par"] == "1:1":
            xu_ly["khung"] = {"cach": "giu_nguyen", "scale_phan_tram": 100.0}
        elif c["rong_hien"]:
            b = tinh((c["rong_hien"], c["cao_hien"]), (W, H), None)
            xu_ly["khung"] = {"cach": b["cach"], "scale_phan_tram": b["scale_phan_tram"], "vi_tri": b["vi_tri"]}
            if b["cach"] != "vua_khung":
                chu_y.append("khác tỉ lệ khung: lấp đầy khung, căn giữa (rough cut sẽ dời theo mặt người)")
                canh_bao["khac_ti_le"] += 1
            elif b["scale_phan_tram"] > 100.01:
                chu_y.append(f"nhỏ hơn khung sequence: phóng {b['scale_phan_tram']:.0f}%, sẽ mềm hơn")
                canh_bao["phong_to"] += 1
            else:
                canh_bao["fit_to_frame"] += 1
        # Frame rate
        if c["fps_so"]:
            r = c["fps_so"] / f_seq
            if c["fps_quay"] and c["fps_quay"] < c["fps_so"] - 0.05:
                he = c["fps_so"] / c["fps_quay"]
                xu_ly["toc_do"] = {"loai": "tua_nhanh", "nhanh_gap": round(he, 2)}
                chu_y.append(f"quay {_ten_fps(c['fps_quay'])} fps ghi thành {_ten_fps(c['fps_so'])}: time-lapse, chạy nhanh {he:.1f} lần")
                canh_bao["tua_nhanh"] += 1
            elif c["fps_quay"] and c["fps_quay"] > c["fps_so"] + 0.05:
                pt = c["fps_so"] / c["fps_quay"] * 100
                xu_ly["toc_do"] = {"loai": "slow_motion_trong_may", "toc_do_phan_tram": round(pt, 1)}
                chu_y.append(f"slow motion quay trong máy ({_ten_fps(c['fps_quay'])} fps), đã chậm {pt:.0f}% sẵn")
                canh_bao["slow_motion_trong_may"] += 1
            if abs(r - 1) < 0.002:
                xu_ly["fps"] = {"quan_he": "khop"}
            elif r > 1.002:
                boi = round(r)
                if abs(r - boi) < 0.01 and boi >= 2:
                    xu_ly["fps"] = {"quan_he": "boi_so", "co_the_slow_motion_phan_tram": round(100 / r, 1)}
                    chu_y.append(f"{_ten_fps(c['fps_so'])} fps trong sequence {_ten_fps(f_seq)}: phát thường bỏ bớt frame; "
                                 f"interpret footage về {_ten_fps(f_seq)} để slow motion {100 / r:.0f}%")
                elif r >= 1.9:
                    xu_ly["fps"] = {"quan_he": "cao_hon", "co_the_slow_motion_phan_tram": round(100 / r, 1)}
                    chu_y.append(f"{_ten_fps(c['fps_so'])} fps cao hơn sequence: có thể slow motion {100 / r:.0f}% (frame blend hoặc interpret)")
                else:
                    xu_ly["fps"] = {"quan_he": "le_cao_hon"}
                    chu_y.append(f"{_ten_fps(c['fps_so'])} fps vào sequence {_ten_fps(f_seq)}: bỏ frame không đều, chuyển động có thể giật")
                    canh_bao["fps_le"] += 1
                canh_bao["fps_cao_hon"] += 1
            else:
                xu_ly["fps"] = {"quan_he": "thap_hon"}
                chu_y.append(f"{_ten_fps(c['fps_so'])} fps thấp hơn sequence {_ten_fps(f_seq)}: frame được lặp lại, chuyển động kém mượt hơn")
                canh_bao["fps_thap_hon"] += 1
        # Colour
        if c["log"] in ("HLG", "PQ (HDR10)"):
            xu_ly["mau"] = {"loai": "hdr", "duong_cong": c["log"]}
            chu_y.append(f"HDR {c['log']}: bật quản lý màu HDR hoặc tone map về Rec.709")
            canh_bao["hdr"] += 1
        elif c["log"]:
            xu_ly["mau"] = {"loai": "log", "duong_cong": c["log"], "chac_chan": c["nguon_log"] != "tu_khoa_metadata"}
            canh_bao["log"] += 1
        if c["vfr"]:
            chu_y.append("fps biến thiên (thường là điện thoại): nên chuyển sang fps cố định trước khi dựng để tiếng khớp hình")
            canh_bao["vfr"] += 1
        if c["quet_xen"]:
            chu_y.append("quét xen kẽ (interlaced): cần khử xen kẽ")
            canh_bao["quet_xen"] += 1
        if c["par"] != "1:1":
            chu_y.append(f"điểm ảnh không vuông ({c['par']}, anamorphic): đã tính vào khung hiển thị")
            canh_bao["anamorphic"] += 1
        if c["xoay"]:
            chu_y.append(f"có xoay {c['xoay']}°: đã tính vào khung hiển thị")
            canh_bao["xoay"] += 1
        ket_qua.append({**c, "xu_ly": xu_ly, "chu_y": chu_y})
    logs = Counter(c["log"] for c in clip if c["log"] and c["log"] not in ("HLG", "PQ (HDR10)"))
    viec = []
    if logs:
        ten = ", ".join(f"{k} ({v} clip)" for k, v in logs.items())
        viec.append(f"Footage log {ten}: bật Auto Detect Log trong Color Management của sequence (Premiere 2025 trở lên) "
                    "hoặc áp LUT log sang Rec.709 trước khi chấm màu; xem trên màn hình chưa chuyển màu sẽ nhạt.")
    if canh_bao["tua_nhanh"]:
        viec.append(f"{canh_bao['tua_nhanh']} clip time-lapse quay trong máy: dùng như cảnh tua nhanh, không phải cảnh thường.")
    if canh_bao["fps_le"]:
        viec.append(f"{canh_bao['fps_le']} clip fps không chia hết cho sequence: kiểm chuyển động ở các clip này.")
    if canh_bao["vfr"]:
        viec.append(f"{canh_bao['vfr']} clip fps biến thiên: chuyển sang fps cố định trước khi dựng.")
    return {
        "sequence": {"rong": W, "cao": H, "fps": {"tu": fps.numerator, "mau": fps.denominator}, "fps_ten": _ten_fps(f_seq),
                     "ly_do": f"khung: {ly_do_khung}; fps: {ly_do_fps}"},
        "tong_hop": {
            "so_clip": len(clip),
            "kich_thuoc": {f"{w}x{h}": n for (w, h), n in Counter((c["rong_hien"], c["cao_hien"]) for c in clip).most_common()},
            "fps": {_ten_fps(f): n for f, n in Counter(c["fps_so"] for c in clip if c["fps_so"]).most_common()},
            "log": dict(logs), "may": dict(Counter(c["may"] for c in clip if c["may"])),
            "codec": dict(Counter(f"{c['codec']} {c['bit']}-bit {c['chroma']}" for c in clip)),
            "canh_bao": dict(canh_bao),
        },
        "viec_can_lam": viec, "clip": ket_qua,
    }


def quet_thu_muc(source: str | Path | None = None, *, danh_sach: str | Path | None = None,
                 khung: tuple[int, int] | None = None, fps: Fraction | None = None,
                 cache_goc: str | Path | None = None) -> dict:
    """Read properties of every clip (no proxy needed: only metadata is read) and write `.dailies/thuoc_tinh.json`."""
    from dailies import thu_muc_ket_qua
    from dailies.quet import nguon_vao, quet_files
    bat_dau = time.perf_counter()
    source, cac_tep, proxy = nguon_vao(source, danh_sach, cho_phep_goc=True)
    cac_muc = quet_files(cac_tep)
    with ThreadPoolExecutor(max_workers=min(8, max(1, len(cac_muc)))) as pool:
        clip = list(pool.map(lambda x: {**doc_thuoc_tinh(Path(x[1]["duong_dan_gan_nhat"]), x[1],
                                                         str(proxy.get(x[1]["duong_dan_gan_nhat"]) or "") or None), "media_id": x[0]},
                             cac_muc))
    from dailies.mo_ta_shot import _khoa_ten
    clip.sort(key=lambda c: _khoa_ten(c["ten_file"]))
    kq = {"phien_ban": PHIEN_BAN, "thu_muc": str(source), **de_xuat(clip, khung, fps),
          "giay_xu_ly": round(time.perf_counter() - bat_dau, 2)}
    goc = thu_muc_ket_qua(source, cache_goc)
    goc.mkdir(parents=True, exist_ok=True)
    dich = goc / TEN_TEP
    tam = dich.with_suffix(".tmp")
    tam.write_text(json.dumps(kq, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tam.replace(dich)
    print(f"Thuộc tính: {len(clip)} clip trong {kq['giay_xu_ly']} giây", file=sys.stderr, flush=True)
    return {**kq, "file_ket_qua": str(dich)}
