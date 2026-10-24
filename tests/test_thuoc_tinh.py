"""Clip properties before editing: camera XML, recommendations (fit to frame, frame rates, speed, log), the CLI."""

import json
import shutil
import subprocess
from fractions import Fraction

from dailies import thuoc_tinh
from dailies.cli import main


def _clip(ten="A.MP4", rong=3840, cao=2160, fps=(60000, 1001), fps_quay=None, log=None, xoay=0, par="1:1",
          vfr=False, nguon_log="xml_sony"):
    f = Fraction(*fps)
    hien = (rong, cao) if xoay not in (90, 270) else (cao, rong)
    return {"file": f"/x/{ten}", "ten_file": ten, "rong": rong, "cao": cao, "xoay": xoay, "par": par,
            "rong_hien": hien[0], "cao_hien": hien[1], "fps": {"tu": f.numerator, "mau": f.denominator},
            "fps_so": round(float(f), 3), "fps_quay": fps_quay, "vfr": vfr, "quet_xen": False, "log": log,
            "nguon_log": nguon_log if log else None, "may": "ILCE-7M4", "codec": "h264", "bit": 10, "chroma": "4:2:2"}


def test_fps_sony():
    assert thuoc_tinh._fps_sony("59.94p") == Fraction(60000, 1001)
    assert thuoc_tinh._fps_sony("23.98p") == Fraction(24000, 1001)
    assert abs(float(thuoc_tinh._fps_sony("3.74p")) - 3.74) < 1e-3
    assert thuoc_tinh._fps_sony("120p") == Fraction(120)
    assert thuoc_tinh._fps_sony(None) is None


def test_4k_fit_to_frame_trong_sequence_full_hd():
    kq = thuoc_tinh.de_xuat([_clip("A.MP4"), _clip("B.MP4", 1920, 1080)], (1920, 1080), Fraction(24000, 1001))
    a, b = kq["clip"]
    assert a["xu_ly"]["khung"] == {"cach": "vua_khung", "scale_phan_tram": 50.0, "vi_tri": {"x": 960.0, "y": 540.0}}
    assert b["xu_ly"]["khung"]["cach"] == "giu_nguyen"
    assert kq["tong_hop"]["canh_bao"]["fit_to_frame"] == 1


def test_mac_dinh_theo_so_dong_va_bo_qua_clip_tua_nhanh():
    clip = [_clip(f"A{i}.MP4") for i in range(3)] + [_clip(f"T{i}.MP4", 1920, 1080, (24000, 1001), fps_quay=3.74) for i in range(5)]
    kq = thuoc_tinh.de_xuat(clip)
    # Five time-lapse clips at 23.976 do not decide the rate; the three normal clips at 59.94 do
    assert kq["sequence"]["fps_ten"] == "59.94" and (kq["sequence"]["rong"], kq["sequence"]["cao"]) == (1920, 1080)
    t = kq["clip"][3]
    assert t["xu_ly"]["toc_do"] == {"loai": "tua_nhanh", "nhanh_gap": 6.41}
    assert any("time-lapse" in v for v in kq["viec_can_lam"])


def test_quan_he_fps():
    seq = Fraction(24000, 1001)
    kq = thuoc_tinh.de_xuat([_clip("A", fps=(60000, 1001)), _clip("B", fps=(48000, 1001)), _clip("C", fps=(30000, 1001)),
                             _clip("D", fps=(24000, 1001)), _clip("E", fps=(120000, 1001), fps_quay=119.88)], (3840, 2160), seq)
    r = {c["ten_file"]: c["xu_ly"]["fps"] for c in kq["clip"]}
    assert r["A"] == {"quan_he": "cao_hon", "co_the_slow_motion_phan_tram": 40.0}
    assert r["B"] == {"quan_he": "boi_so", "co_the_slow_motion_phan_tram": 50.0}
    assert r["C"] == {"quan_he": "le_cao_hon"} and r["D"] == {"quan_he": "khop"}
    kq2 = thuoc_tinh.de_xuat([_clip("S", fps=(24000, 1001), fps_quay=119.88)], (3840, 2160), seq)
    assert kq2["clip"][0]["xu_ly"]["toc_do"] == {"loai": "slow_motion_trong_may", "toc_do_phan_tram": 20.0}


def test_xoay_anamorphic_log_vfr():
    doc = _clip("P.MOV", 1920, 1080, (30000, 1001), xoay=90, vfr=True)
    ana = _clip("N.MOV", 1440, 1080, (24000, 1001), par="4:3")
    ana["rong_hien"] = 1920
    hdr = _clip("H.MOV", log="HLG", nguon_log="the_mau")
    kq = thuoc_tinh.de_xuat([doc, ana, hdr, _clip("L.MP4", log="S-Log3")], (1920, 1080), Fraction(30000, 1001))
    d, n, h, l = kq["clip"]
    assert d["xu_ly"]["khung"]["cach"] == "day_khung_giua" and any("biến thiên" in x for x in d["chu_y"])
    assert any("anamorphic" in x for x in n["chu_y"])
    assert h["xu_ly"]["mau"]["loai"] == "hdr" and l["xu_ly"]["mau"] == {"loai": "log", "duong_cong": "S-Log3", "chac_chan": True}
    assert kq["tong_hop"]["log"] == {"S-Log3": 1}


def test_doc_xml_sony(clip_mau, tmp_path):
    tep = tmp_path / "C0001.MP4"
    shutil.copyfile(clip_mau("tt-a", rong=320, cao=180, giay=2, cau=[]), tep)
    with tep.open("ab") as f:  # Sony puts its XML inside the file; appended here the same way it is read
        f.write(b'<?xml version="1.0"?><NonRealTimeMeta><VideoFrame videoCodec="AVC" captureFps="119.88p" formatFps="23.98p"/>'
                b'<Item name="CaptureGammaEquation" value="s-log3-cine"/><Device manufacturer="Sony" modelName="ILCE-7M4"/>'
                b'<RecordingMode type="normal"/></NonRealTimeMeta>')
    x = thuoc_tinh.xml_sony(str(tep))
    assert x["fps_quay"] == "119.88p" and x["gamma"] == "s-log3-cine" and x["may"] == "ILCE-7M4"


def test_cli_thuoc_tinh(clip_mau, tmp_path, capsys):
    source = tmp_path / "source"
    source.mkdir()
    shutil.copyfile(clip_mau("tt-b", rong=640, cao=360, giay=2, cau=[]), source / "A001.mp4")
    subprocess.run(["ffmpeg", "-v", "error", "-display_rotation", "90", "-i", str(clip_mau("tt-c", rong=640, cao=360, giay=2, cau=[])),
                    "-c", "copy", str(source / "A002.mp4")], check=True)
    assert main(["thuoc-tinh", str(source), "--khung", "1920x1080", "--fps", "25", "--json"]) == 0
    kq = json.loads(capsys.readouterr().out)
    assert kq["ok"] and kq["sequence"]["fps_ten"] == "25" and kq["tong_hop"]["so_clip"] == 2
    a, b = kq["clip"]
    assert a["xu_ly"]["khung"]["scale_phan_tram"] == 300.0 and (b["rong_hien"], b["cao_hien"]) == (360, 640)
    assert b["xu_ly"]["khung"]["cach"] == "day_khung_giua"
    assert (source / ".dailies" / "thuoc_tinh.json").is_file()
