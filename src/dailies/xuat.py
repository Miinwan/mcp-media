"""Export the selects of `dailies cham-diem` for another editor: FCP7 XML (Premiere Pro, DaVinci Resolve), FCPXML (Final
Cut Pro, Resolve), CMX3600 EDL (any NLE) and CSV.

Every format points at the original camera files (never the proxies) and carries the source timecode read from the file,
so the receiving editor can relink by path, name or timecode. Segments are laid end to end on one sequence, ordered
by file name then time (or by score); each carries a marker at its peak with the score and camera move, and a clip
colour by grade (good green, fair yellow, pass orange).
"""

from __future__ import annotations

import csv
import json
import re
from fractions import Fraction
from pathlib import Path
from urllib.parse import quote
from xml.sax.saxutils import escape

DINH_DANG = {"xml": ".xml", "fcpxml": ".fcpxml", "edl": ".edl", "csv": ".csv"}
THU_MUC_CHUNG = {"source", "sources", "footage", "raw", "video", "videos", "media", "clips", "private", "m4root", "dcim", "nguồn"}
NHAN_MUC = {"tot": "tốt", "kha": "khá", "dat": "đạt"}
MAU_FCP7 = {"tot": "Forest", "kha": "Yellow", "dat": "Mango"}
MAU_FCPX = {"tot": "Green", "kha": "Yellow", "dat": "Orange"}
TEN_MAY = {"dung_yen": "đứng yên", "lia_trai": "lia trái", "lia_phai": "lia phải", "nghieng_len": "nghiêng lên",
           "nghieng_xuong": "nghiêng xuống", "zoom_vao": "zoom vào", "zoom_ra": "zoom ra", "xoay_trai": "xoay trái",
           "xoay_phai": "xoay phải", "hon_hop": "hỗn hợp"}


# ---------------------------------------------------------------- timecode

def la_ntsc(fps: Fraction) -> bool:
    return fps.denominator == 1001


def tc_sang_frame(tc: str, fps: Fraction) -> int:
    """Timecode string to a frame count at `fps` (nominal base round(fps)). 29.97 and 59.94 are counted drop-frame,
    as Premiere does for camera files (Sony writes the start timecode of 59.94p as drop-frame on a 60 base)."""
    h, m, s, f = (int(x) for x in re.split(r"[:;.]", tc.strip()))
    co_so = round(fps)
    tong = ((h * 60 + m) * 60 + s) * co_so + f
    if la_ntsc(fps) and co_so in (30, 60):
        bo = co_so // 15  # frames dropped each minute: 2 at 30, 4 at 60
        phut = h * 60 + m
        tong -= bo * (phut - phut // 10)
    return tong


def frame_sang_tc(n: int, fps: Fraction) -> str:
    co_so = round(fps)
    df = la_ntsc(fps) and co_so in (30, 60)
    if df:
        bo = co_so // 15
        moi_10_phut = co_so * 600 - bo * 9
        moi_phut = co_so * 60 - bo
        d, r = divmod(n, moi_10_phut)
        n += bo * 9 * d + (bo * ((r - bo) // moi_phut) if r > bo else 0)
    f = n % co_so
    s = n // co_so
    return f"{s // 3600:02d}:{s // 60 % 60:02d}:{s % 60:02d}{';' if df else ':'}{f:02d}"


def thong_tin_file(duong: str) -> dict:
    """fps, size, audio channels and start timecode (frames) of the original file."""
    from dailies.quet import ffprobe
    du_lieu = ffprobe(duong)
    video = next((s for s in du_lieu["streams"] if s.get("codec_type") == "video"), {})
    audio = [s for s in du_lieu["streams"] if s.get("codec_type") == "audio"]
    fps = Fraction(video.get("r_frame_rate") or "25")
    tc = None
    for s in [video] + [s for s in du_lieu["streams"] if s is not video] + [du_lieu.get("format", {})]:
        tc = (s.get("tags") or {}).get("timecode")
        if tc:
            break
    from dailies.quet import kich_thuoc_hien
    rong_hien, cao_hien = kich_thuoc_hien(duong)
    return {"fps": fps, "rong": int(video.get("width") or 1920), "cao": int(video.get("height") or 1080),
            "rong_hien": rong_hien, "cao_hien": cao_hien,
            "kenh": sum(int(a.get("channels") or 0) for a in audio), "tan_so": int(audio[0].get("sample_rate") or 48000) if audio else 48000,
            "tc": tc, "tc_frame": tc_sang_frame(tc, fps) if tc else 0,
            "so_frame": round(float(du_lieu["format"].get("duration") or 0) * fps)}


# ---------------------------------------------------------------- layout

def bo_cuc_doan(kq: dict, thu_tu: str = "file", khoang: float = 0.0, dem: float = 0.0,
                khung_seq: dict | None = None) -> tuple[list[dict], Fraction]:
    """Segments laid end to end on a sequence. Positions are whole sequence frames; `dem` adds handles each side
    (within the file). `khung_seq` ({rong, cao, fps, scale: {file: %}}, from `dailies thuoc-tinh`) sets the sequence;
    without it, the most common frame rate and size. Every clip not at the sequence size is fit to frame."""
    from dailies.bo_cuc import tinh
    hang = []
    for c in kq["clip"]:
        for d in c["doan"]:
            hang.append({"clip": c, "doan": d})
    if thu_tu == "diem":
        hang.sort(key=lambda x: -x["doan"]["diem_tb"])
    tt = {}
    for h in hang:
        tt.setdefault(h["clip"]["file"], thong_tin_file(h["clip"]["file"]))
    dem_fps = {}
    for f, t in tt.items():
        dem_fps[t["fps"]] = dem_fps.get(t["fps"], 0) + 1
    fps_seq = max(dem_fps, key=dem_fps.get) if dem_fps else Fraction(25)
    dem_khung = {}
    for t in tt.values():
        dem_khung[(t["rong_hien"], t["cao_hien"])] = dem_khung.get((t["rong_hien"], t["cao_hien"]), 0) + 1
    W, H = max(dem_khung, key=dem_khung.get) if dem_khung else (1920, 1080)
    ti_le = {}
    if khung_seq:
        fps_seq = khung_seq.get("fps") or fps_seq
        W, H = khung_seq.get("rong") or W, khung_seq.get("cao") or H
        ti_le = khung_seq.get("scale") or {}
    con_tro = 0
    ra = []
    for i, h in enumerate(hang, 1):
        t = tt[h["clip"]["file"]]
        d = h["doan"]
        vao_s = max(0.0, d["vao_giay"] - dem)
        ra_s = min(h["clip"]["thoi_luong_giay"] or d["ra_giay"] + dem, d["ra_giay"] + dem)
        vao = round(vao_s * t["fps"])
        ra_f = max(vao + 1, round(ra_s * t["fps"]))
        dai_seq = max(1, round((ra_f - vao) / t["fps"] * fps_seq))
        # FCP7 XML counts a clip's in/out on the sequence's time base, whatever the file's own rate
        vao_q = round(vao / t["fps"] * fps_seq)
        sc = ti_le.get(h["clip"]["file"])
        if sc is None:
            sc = 100.0 if (t["rong_hien"], t["cao_hien"]) == (W, H) else tinh((t["rong_hien"], t["cao_hien"]), (W, H), None)["scale_phan_tram"]
        ra.append({"stt": i, "file": h["clip"]["file"], "ten": Path(h["clip"]["file"]).name, "tt": t, "doan": d,
                   "scale": round(float(sc), 3), "seq_rong": W, "seq_cao": H,
                   "vao": vao, "ra": ra_f, "dinh": round(d["giay_dinh"] * t["fps"]),
                   "vao_q": vao_q, "ra_q": vao_q + dai_seq, "dinh_q": round(d["giay_dinh"] * fps_seq),
                   "dai_q": round(t["so_frame"] / t["fps"] * fps_seq),
                   "seq_vao": con_tro, "seq_ra": con_tro + dai_seq})
        con_tro += dai_seq + round(khoang * fps_seq)
    return ra, fps_seq


def ghi_chu(d: dict) -> str:
    phan = [f"điểm {d['diem_tb']:.2f} (đỉnh {d['diem_dinh']:.2f})", NHAN_MUC[d["muc"]], TEN_MAY.get(d["may"], d["may"])]
    if d["co"]:
        phan.append("cờ: " + ", ".join(d["co"]))
    return " · ".join(phan)


def _url(duong: str) -> str:
    return "file://localhost" + quote(str(Path(duong).resolve()))


def _rate(fps: Fraction) -> str:
    return f"<rate><timebase>{round(fps)}</timebase><ntsc>{'TRUE' if la_ntsc(fps) else 'FALSE'}</ntsc></rate>"


# ---------------------------------------------------------------- writers

def ghi_xml(hang: list[dict], fps: Fraction, ten: str) -> str:
    """FCP7 XML (xmeml 4): one sequence, V1 plus one audio track per source channel (up to 2), linked."""
    the_file, ds_v, so_kenh = set(), [], max([min(2, h["tt"]["kenh"]) for h in hang] + [0])
    ds_a = [[] for _ in range(so_kenh)]
    for h in hang:
        tt = h["tt"]
        fid = "file-" + re.sub(r"[^A-Za-z0-9]", "_", h["file"])[-60:]
        if fid in the_file:
            file_xml = f'<file id="{fid}"/>'
        else:
            the_file.add(fid)
            tc = (f"<timecode>{_rate(tt['fps'])}<string>{frame_sang_tc(tt['tc_frame'], tt['fps'])}</string>"
                  f"<frame>{tt['tc_frame']}</frame><displayformat>{'DF' if la_ntsc(tt['fps']) and round(tt['fps']) in (30, 60) else 'NDF'}</displayformat></timecode>")
            am = (f"<audio><samplecharacteristics><depth>16</depth><samplerate>{tt['tan_so']}</samplerate></samplecharacteristics>"
                  f"<channelcount>{tt['kenh']}</channelcount></audio>") if tt["kenh"] else ""
            file_xml = (f'<file id="{fid}"><name>{escape(h["ten"])}</name><pathurl>{escape(_url(h["file"]))}</pathurl>'
                        f"{_rate(tt['fps'])}<duration>{tt['so_frame']}</duration>{tc}<media><video><samplecharacteristics>"
                        f"<width>{tt['rong']}</width><height>{tt['cao']}</height></samplecharacteristics></video>{am}</media></file>")
        d = h["doan"]
        v_id = f"clipitem-v{h['stt']}"
        a_ids = [f"clipitem-a{h['stt']}-{k + 1}" for k in range(min(so_kenh, tt["kenh"]))]
        lien = "".join(f"<link><linkclipref>{x}</linkclipref><mediatype>{'video' if x == v_id else 'audio'}</mediatype>"
                       f"<trackindex>{1 if x == v_id else int(x.rsplit('-', 1)[1])}</trackindex><clipindex>{h['stt']}</clipindex></link>"
                       for x in [v_id] + a_ids)
        chung = (f"<name>{escape(h['ten'])}</name><enabled>TRUE</enabled><duration>{h['dai_q']}</duration>{_rate(fps)}"
                 f"<start>{h['seq_vao']}</start><end>{h['seq_ra']}</end><in>{h['vao_q']}</in><out>{h['ra_q']}</out>")
        moc = (f"<marker><name>{escape(f'{d['diem_tb']:.2f} {NHAN_MUC[d['muc']]}')}</name><comment>{escape(ghi_chu(d))}</comment>"
               f"<in>{h['dinh_q']}</in><out>-1</out></marker>")
        # Fit to frame: Basic Motion scale, read by Premiere and Resolve
        loc = ("" if abs(h["scale"] - 100) < 0.001 else
               "<filter><effect><name>Basic Motion</name><effectid>basic</effectid><effectcategory>motion</effectcategory>"
               "<effecttype>motion</effecttype><mediatype>video</mediatype><parameter><parameterid>scale</parameterid>"
               f"<name>Scale</name><valuemin>0</valuemin><valuemax>10000</valuemax><value>{h['scale']:g}</value></parameter></effect></filter>")
        ds_v.append(f'<clipitem id="{v_id}">{chung}{file_xml}{loc}{moc}<labels><label2>{MAU_FCP7[d["muc"]]}</label2></labels>{lien}</clipitem>')
        for k, a_id in enumerate(a_ids):
            ds_a[k].append(f'<clipitem id="{a_id}">{chung}<file id="{fid}"/><sourcetrack><mediatype>audio</mediatype>'
                           f"<trackindex>{k + 1}</trackindex></sourcetrack><labels><label2>{MAU_FCP7[d['muc']]}</label2></labels>{lien}</clipitem>")
    dai = hang[-1]["seq_ra"] if hang else 0
    rong, cao = (hang[0]["seq_rong"], hang[0]["seq_cao"]) if hang else (1920, 1080)
    am = "".join(f"<track>{''.join(x)}</track>" for x in ds_a)
    # Timeline markers at each peak too, so the scores show without opening the clips
    moc_seq = "".join(f"<marker><name>{escape(f'{h['doan']['diem_tb']:.2f} {NHAN_MUC[h['doan']['muc']]}')}</name>"
                      f"<comment>{escape(h['ten'] + ' · ' + ghi_chu(h['doan']))}</comment>"
                      f"<in>{h['seq_vao'] + h['dinh_q'] - h['vao_q']}</in><out>-1</out></marker>" for h in hang)
    return ('<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE xmeml>\n<xmeml version="4"><sequence id="sequence-1">'
            f"<name>{escape(ten)}</name><duration>{dai}</duration>{_rate(fps)}<timecode>{_rate(fps)}<string>{frame_sang_tc(0, fps)}</string>"
            f"<frame>0</frame></timecode><media><video><format><samplecharacteristics>{_rate(fps)}<width>{rong}</width><height>{cao}</height>"
            f"<pixelaspectratio>square</pixelaspectratio></samplecharacteristics></format><track>{''.join(ds_v)}</track></video>"
            f"<audio>{am}</audio></media>{moc_seq}</sequence></xmeml>\n")


def _t(n: int, fps: Fraction) -> str:
    """FCPXML rational time of `n` frames."""
    s = Fraction(n) / fps
    return f"{s.numerator}/{s.denominator}s" if s.denominator != 1 else f"{s.numerator}s"


def ghi_fcpxml(hang: list[dict], fps: Fraction, ten: str) -> str:
    """FCPXML 1.10: one project, segments as asset-clips on the spine, a marker at each peak, a colour keyword."""
    dinh_dang, tai_san = {}, {}
    for h in hang:
        tt = h["tt"]
        k = (tt["fps"], tt["rong"], tt["cao"])
        if k not in dinh_dang:
            dinh_dang[k] = f"r{len(dinh_dang) + 1}"
    seq_k = (fps, hang[0]["seq_rong"], hang[0]["seq_cao"]) if hang else (fps, 1920, 1080)
    if seq_k not in dinh_dang:
        dinh_dang[seq_k] = f"r{len(dinh_dang) + 1}"
    for h in hang:
        if h["file"] not in tai_san:
            tai_san[h["file"]] = f"r{len(dinh_dang) + len(tai_san) + 1}"
    tn = []
    for (f, w, c), rid in dinh_dang.items():
        tn.append(f'<format id="{rid}" frameDuration="{_t(1, f)}" width="{w}" height="{c}"/>')
    da = set()
    for h in hang:
        if h["file"] in da:
            continue
        da.add(h["file"])
        tt = h["tt"]
        rid = dinh_dang[(tt["fps"], tt["rong"], tt["cao"])]
        am = f' hasAudio="1" audioSources="1" audioChannels="{tt["kenh"]}" audioRate="{tt["tan_so"]}"' if tt["kenh"] else ""
        tn.append(f'<asset id="{tai_san[h["file"]]}" name="{escape(h["ten"], {chr(34): "&quot;"})}" start="{_t(tt["tc_frame"], tt["fps"])}" '
                  f'duration="{_t(tt["so_frame"], tt["fps"])}" hasVideo="1" format="{rid}"{am}>'
                  f'<media-rep kind="original-media" src="{escape(_url(h["file"]).replace("file://localhost", "file://"))}"/></asset>')
    xuong = []
    for h in hang:
        tt, d = h["tt"], h["doan"]
        bat_dau = tt["tc_frame"] + h["vao"]
        xuong.append(f'<asset-clip ref="{tai_san[h["file"]]}" name="{escape(h["ten"], {chr(34): "&quot;"})}" '
                     f'offset="{_t(h["seq_vao"], fps)}" start="{_t(bat_dau, tt["fps"])}" duration="{_t(h["seq_ra"] - h["seq_vao"], fps)}" '
                     f'tcFormat="{"DF" if la_ntsc(tt["fps"]) else "NDF"}"><adjust-conform type="fit"/>'
                     f'<marker start="{_t(tt["tc_frame"] + h["dinh"], tt["fps"])}" duration="{_t(1, tt["fps"])}" '
                     f'value="{escape(f"{d["diem_tb"]:.2f} {NHAN_MUC[d["muc"]]}", {chr(34): "&quot;"})}" note="{escape(ghi_chu(d), {chr(34): "&quot;"})}"/>'
                     f'<keyword start="{_t(bat_dau, tt["fps"])}" duration="{_t(h["ra"] - h["vao"], tt["fps"])}" value="{MAU_FCPX[d["muc"]]}"/></asset-clip>')
    dai = hang[-1]["seq_ra"] if hang else 0
    return ('<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE fcpxml>\n<fcpxml version="1.10"><resources>' + "".join(tn) +
            f'</resources><library><event name="{escape(ten)}"><project name="{escape(ten)}"><sequence format="{dinh_dang[seq_k]}" '
            f'duration="{_t(dai, fps)}" tcStart="0s" tcFormat="{"DF" if la_ntsc(fps) else "NDF"}"><spine>' + "".join(xuong) +
            "</spine></sequence></project></event></library></fcpxml>\n")


def ghi_edl(hang: list[dict], fps: Fraction, ten: str) -> str:
    """CMX3600 at the sequence rate. Reel names are the last 8 characters of the file name; the full name and path are
    in the comment lines Premiere and Resolve read."""
    df = la_ntsc(fps) and round(fps) in (30, 60)
    dong = [f"TITLE: {ten}", f"FCM: {'DROP FRAME' if df else 'NON-DROP FRAME'}", ""]
    for h in hang:
        tt, d = h["tt"], h["doan"]
        cuon = re.sub(r"[^A-Za-z0-9]", "", Path(h["file"]).stem)[-8:] or "AX"
        kenh = "AA/V" if tt["kenh"] >= 2 else ("A/V" if tt["kenh"] else "V")
        # Source in/out at the sequence rate (CMX3600 has one rate per list)
        vao_seq = round((tt["tc_frame"] + h["vao"]) / tt["fps"] * fps)
        ra_seq = vao_seq + (h["seq_ra"] - h["seq_vao"])
        dong.append(f"{h['stt']:03d}  {cuon:<8} {kenh:<5} C        {frame_sang_tc(vao_seq, fps)} {frame_sang_tc(ra_seq, fps)} "
                    f"{frame_sang_tc(h['seq_vao'], fps)} {frame_sang_tc(h['seq_ra'], fps)}")
        dong.append(f"* FROM CLIP NAME: {h['ten']}")
        dong.append(f"* SOURCE FILE: {h['file']}")
        dong.append(f"* COMMENT: {ghi_chu(d)}")
        dong.append("")
    return "\r\n".join(dong) + "\r\n"


def ghi_csv(hang: list[dict], fps: Fraction, tep: Path) -> None:
    with tep.open("w", newline="", encoding="utf-8-sig") as f:  # BOM so Excel reads Vietnamese correctly
        w = csv.writer(f)
        w.writerow(["stt", "file", "ten_file", "tc_vao", "tc_ra", "vao_giay", "ra_giay", "dai_giay", "diem_tb", "diem_dinh",
                    "giay_dinh", "muc", "may", "co", "tc_timeline_vao"])
        for h in hang:
            tt, d = h["tt"], h["doan"]
            w.writerow([h["stt"], h["file"], h["ten"], frame_sang_tc(tt["tc_frame"] + h["vao"], tt["fps"]),
                        frame_sang_tc(tt["tc_frame"] + h["ra"], tt["fps"]), round(float(h["vao"] / tt["fps"]), 3), round(float(h["ra"] / tt["fps"]), 3),
                        round(float((h["ra"] - h["vao"]) / tt["fps"]), 3), d["diem_tb"], d["diem_dinh"], d["giay_dinh"], NHAN_MUC[d["muc"]],
                        TEN_MAY.get(d["may"], d["may"]), ", ".join(d["co"]), frame_sang_tc(h["seq_vao"], fps)])


def xuat(ket_qua: str | Path, dinh_dang: list[str], ra: str | Path | None = None, *, ten: str | None = None,
         thu_tu: str = "file", khoang: float = 0.0, dem: float = 0.0, khung: tuple[int, int] | None = None,
         fps: Fraction | None = None) -> dict:
    """Write the requested formats; `ket_qua` is a cham_diem.json or a source folder. The sequence follows `khung` /
    `fps` when given, else the property scan saved next to it (`dailies thuoc-tinh`), else the most common clip."""
    p = Path(ket_qua)
    if p.is_dir():
        from dailies import thu_muc_ket_qua
        p = thu_muc_ket_qua(p) / "cham_diem.json"
    if not p.is_file():
        raise ValueError(f"Chưa có kết quả chấm điểm {p}; chạy dailies cham-diem trước")
    sai = [x for x in dinh_dang if x not in DINH_DANG]
    if sai:
        raise ValueError(f"Định dạng không hỗ trợ: {', '.join(sai)} (có: {', '.join(DINH_DANG)})")
    if thu_tu not in ("file", "diem"):
        raise ValueError("Thứ tự phải là file hoặc diem")
    kq = json.loads(p.read_text(encoding="utf-8"))
    goc = Path(kq["thu_muc"])
    # "Source", "Footage", a card folder... say nothing; the job folder above does
    ten_job = goc.parent.name if goc.name.lower() in THU_MUC_CHUNG or re.fullmatch(r"(a|b|c)?\d{3}|card ?\d*|clip", goc.name.lower()) else goc.name
    ten = ten or f"Selects {ten_job}"
    thu_muc = Path(ra) if ra else Path(kq["thu_muc"]) / "Dailies selects"
    thu_muc.mkdir(parents=True, exist_ok=True)
    khung_seq = None
    tep_tt = p.parent / "thuoc_tinh.json"
    if tep_tt.is_file() and not (khung or fps):
        tt = json.loads(tep_tt.read_text(encoding="utf-8"))
        sq = tt["sequence"]
        khung_seq = {"rong": sq["rong"], "cao": sq["cao"], "fps": Fraction(sq["fps"]["tu"], sq["fps"]["mau"]),
                     "scale": {c["file"]: c["xu_ly"]["khung"]["scale_phan_tram"] for c in tt["clip"] if c.get("xu_ly", {}).get("khung")}}
    elif khung or fps:
        khung_seq = {"rong": khung[0] if khung else None, "cao": khung[1] if khung else None, "fps": fps}
    hang, fps = bo_cuc_doan(kq, thu_tu, khoang, dem, khung_seq)
    if not hang:
        raise ValueError("Không có đoạn nào đạt ngưỡng để xuất; hạ --nguong khi chấm điểm")
    goc_ten = re.sub(r'[/:*?"<>|]', "-", ten)
    tep = {}
    for dd in dinh_dang:
        dich = thu_muc / f"{goc_ten}{DINH_DANG[dd]}"
        if dd == "csv":
            ghi_csv(hang, fps, dich)
        else:
            noi_dung = {"xml": ghi_xml, "fcpxml": ghi_fcpxml, "edl": ghi_edl}[dd](hang, fps, ten)
            dich.write_text(noi_dung, encoding="utf-8")
        tep[dd] = str(dich)
    return {"ten": ten, "so_doan": len(hang), "so_file": len({h["file"] for h in hang}),
            "khung": {"rong": hang[0]["seq_rong"], "cao": hang[0]["seq_cao"]},
            "fit_to_frame": sum(1 for h in hang if abs(h["scale"] - 100) > 0.001),
            "fps": {"tu": fps.numerator, "mau": fps.denominator}, "thoi_luong_giay": round(float(hang[-1]["seq_ra"] / fps), 3),
            "thu_tu": thu_tu, "tep": tep, "nguon": str(p)}
