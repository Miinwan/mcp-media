"""Footage scoring: fast frame reading, sharpness, camera motion, segments, the CLI, and reading Cutdown sessions."""

import json
import shutil
import struct
import subprocess
import uuid

import cv2
import numpy as np
import pytest

from dailies import cham_diem, cutdown, thi_giac
from dailies.cli import main


def _anh_hoa_van(rong=640, cao=360, hat=0):
    """Textured grey test picture (random blocks, smoothed a little so tracking has corners)."""
    rng = np.random.default_rng(hat)
    anh = cv2.resize(rng.integers(0, 255, (cao // 8, rong // 8), dtype=np.uint8), (rong, cao), interpolation=cv2.INTER_NEAREST)
    return cv2.GaussianBlur(anh, (3, 3), 0)


def _ffmpeg_khung(tep, rong=320):
    """First frame as ffmpeg decodes it (rotation applied)."""
    ra = subprocess.run(["ffmpeg", "-v", "error", "-i", str(tep), "-frames:v", "1", "-vf", f"scale={rong}:-2:flags=area,format=rgb24",
                         "-f", "rawvideo", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(ra, np.uint8)


# ---------------------------------------------------------------- reading

@pytest.mark.parametrize("proxy", [True, False])
def test_doc_nhanh_dung_so_khung_va_giong_ffmpeg(clip_mau, tmp_path, proxy):
    goc = clip_mau("cd-doc", rong=640, cao=360, giay=4, cau=[])
    tep = goc
    if proxy:
        tep = tmp_path / "p.mov"
        subprocess.run(["ffmpeg", "-v", "error", "-i", str(goc), "-c:v", "prores_videotoolbox", "-profile:v", "0", str(tep)], check=True)
    khung = list(thi_giac.lap_khung(str(tep), 2, 320))
    thoi_gian = [t for t, _ in khung]
    assert 7 <= len(khung) <= 9 and thoi_gian == sorted(thoi_gian)
    khe = [b - a for a, b in zip(thoi_gian, thoi_gian[1:])]
    # ProRes: exact sample times; H.264: the nearest keyframes (0.4 s apart here), so the steps are uneven
    assert all(abs(k - 0.5) < 0.1 for k in khe) if proxy else all(0.3 < k < 0.9 for k in khe)
    assert khung[0][1].shape == (180, 320, 3)
    dau = _ffmpeg_khung(tep).reshape(180, 320, 3)
    assert np.abs(khung[0][1].astype(int) - dau).mean() < 2  # same first frame, same colour conversion


def test_doc_nhanh_xoay_theo_metadata(clip_mau, tmp_path):
    goc = clip_mau("cd-xoay", rong=640, cao=360, giay=2, cau=[])
    tep = tmp_path / "xoay.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-display_rotation", "90", "-i", str(goc), "-c", "copy", str(tep)], check=True)
    t, anh = next(thi_giac.lap_khung(str(tep), 2, 180))
    ff = _ffmpeg_khung(tep, rong=180).reshape(320, 180, 3)
    assert anh.shape == (320, 180, 3)
    assert np.abs(anh.astype(int) - ff).mean() < 3


def test_doc_khung_giu_dang_mang(clip_mau):
    anh = thi_giac.doc_khung(str(clip_mau("cd-mang", rong=320, cao=180, giay=2, cau=[])), 2, 160)
    assert anh.ndim == 4 and anh.shape[1:] == (90, 160, 3) and len(anh) >= 3


# ---------------------------------------------------------------- metrics

def test_net_theo_luoi_va_mo():
    net = _anh_hoa_van()
    mo = cv2.GaussianBlur(net, (0, 0), 4)
    xoa_phong = mo.copy()
    xoa_phong[120:240, 213:426] = net[120:240, 213:426]  # sharp subject on a blurred background
    n_net, ct_net = cham_diem.do_net(net)
    n_mo, ct_mo = cham_diem.do_net(mo)
    n_xp, _ = cham_diem.do_net(xoa_phong)
    assert n_net > 2 * n_mo and ct_net > ct_mo
    assert n_xp > 0.8 * n_net  # the sharpest cells decide, the bokeh does not pull the score down


def test_nhieu_vung_toi_khong_tinh_la_net():
    rng = np.random.default_rng(1)
    toi = np.clip(20 + rng.normal(0, 6, (360, 640)), 0, 255).astype(np.uint8)
    assert cham_diem.do_net(toi)[1] < 0.01


@pytest.mark.parametrize("phep, khoa, dau", [
    ("dich_phai", "lia", -1),     # picture moves right: the camera pans left
    ("dich_xuong", "nghieng", 1),  # picture moves down: the camera tilts up
    ("phong_to", "zoom", 1),
    ("quay", "xoay", 1),
])
def test_chuyen_dong_may(phep, khoa, dau):
    a = _anh_hoa_van(320, 180, 2)
    h, w = a.shape
    m = {"dich_phai": np.float32([[1, 0, 6], [0, 1, 0]]), "dich_xuong": np.float32([[1, 0, 0], [0, 1, 5]]),
         "phong_to": cv2.getRotationMatrix2D((w / 2, h / 2), 0, 1.05),
         "quay": cv2.getRotationMatrix2D((w / 2, h / 2), -3, 1.0)}[phep]  # OpenCV positive angle is counter-clockwise
    b = cv2.warpAffine(a, m, (w, h), borderMode=cv2.BORDER_REFLECT)
    kq = cham_diem.do_chuyen_dong(a, b, 0.25)
    assert kq["tin_cay"] > 0.6
    assert np.sign(kq[khoa]) == dau
    khac = {k: abs(kq[k]) / cham_diem.NGUONG_MAY[k] for k in cham_diem.NGUONG_MAY if k != khoa}
    assert abs(kq[khoa]) / cham_diem.NGUONG_MAY[khoa] > 2 * max(khac.values())


def test_chuyen_dong_may_anh_phang():
    phang = np.full((180, 320), 128, np.uint8)
    assert cham_diem.do_chuyen_dong(phang, phang, 0.25)["tin_cay"] == 0.0


# ---------------------------------------------------------------- scores and segments

def _mau(n, **dat):
    co_ban = {"net": 50.0, "chi_tiet": 0.05, "sang": 120.0, "tuong_phan": 50.0, "chay": 0.0, "den": 0.0, "bao_hoa": 80.0,
              "mat": None, "lia": 0.0, "nghieng": 0.0, "zoom": 0.0, "xoay": 0.0, "chu_the": 0.0, "tin_cay": 1.0,
              "lech": 3.0, "tuong_quan": 0.99}
    return [{"t": i * 0.25, **co_ban, **{k: (v(i) if callable(v) else v) for k, v in dat.items()}} for i in range(n)]


def test_cham_va_dung_doan_noi_khoang_ngan():
    # 12 s: blurred from 4 to 4.5 s (short dip, bridged) and black from 8 s on (removed)
    mau = _mau(48, net=lambda i: 10.0 if 16 <= i < 18 else 50.0, sang=lambda i: 5.0 if i >= 32 else 120.0,
               tuong_phan=lambda i: 2.0 if i >= 32 else 50.0)
    diem = cham_diem.cham(mau, cham_diem.tham_chieu([mau]))
    assert all(d["diem"] == 0 and "den_kit" in d["co"] for d in diem[32:])
    assert diem[0]["diem"] > 0.85
    doan = cham_diem.dung_doan(diem, 12.0, nguong=0.7, noi_khoang=2.0, sau_nhat=0.5)
    assert len(doan) == 1 and doan[0]["vao_giay"] == 0 and abs(doan[0]["ra_giay"] - 7.875) < 1e-6
    assert doan[0]["may"] == "dung_yen" and doan[0]["muc"] == "tot"
    # Without bridging the dip splits the segment in two
    assert len(cham_diem.dung_doan(diem, 12.0, nguong=0.7, noi_khoang=0.0)) == 2


def test_rung_khac_lia_muot():
    muot = _mau(40, lia=0.3)
    rung = _mau(40, lia=lambda i: 0.3 if i % 2 else -0.3)
    tc = cham_diem.tham_chieu([muot])
    d_muot = cham_diem.cham(muot, tc, "b_roll")
    d_rung = cham_diem.cham(rung, tc, "b_roll")
    assert min(d["on_dinh"] for d in d_muot) > 0.95 and not any("rung" in d["co"] for d in d_muot)
    assert np.mean([d["on_dinh"] for d in d_rung]) < 0.2 and sum("rung" in d["co"] for d in d_rung) > 30
    assert cham_diem.phan_loai_may(d_muot) == "lia_phai"
    # The static kind punishes a steady move, b_roll does not
    assert cham_diem.cham(muot, tc, "tinh")[10]["diem"] < d_muot[10]["diem"] - 0.2


def test_doan_ngan_bi_bo():
    mau = _mau(20, net=lambda i: 50.0 if i < 6 else 5.0)
    diem = cham_diem.cham(mau, {"net": 50.0, "chi_tiet": 0.05})
    assert cham_diem.dung_doan(diem, 5.0, ngan_nhat=2.0) == []


# ---------------------------------------------------------------- CLI

def _source(clip_mau, tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    shutil.copyfile(clip_mau("cd-a", rong=640, cao=360, giay=4, cau=[]), source / "A001.mp4")
    shutil.copyfile(clip_mau("cd-b", rong=640, cao=360, giay=3, cau=[]), source / "A002.mp4")
    from sinh_mau import tao_proxy_premiere
    tao_proxy_premiere(source)
    return source


def test_cli_cham_diem_va_cache(clip_mau, tmp_path, capsys):
    source = _source(clip_mau, tmp_path)
    assert main(["cham-diem", str(source), "--json"]) == 0
    kq = json.loads(capsys.readouterr().out)
    assert kq["ok"] and kq["so_clip"] == 2 and kq["so_clip_quet_moi"] == 2 and kq["kieu"] == "can_bang"
    c = kq["clip"][0]
    assert c["ten_file"] == "A001.mp4" and c["co_proxy"] and 14 <= c["so_mau"] <= 17
    assert set(c) >= {"id", "media_id", "file", "fps", "thoi_luong_giay", "diem_tb", "diem_tot_nhat", "may", "co", "doan", "tep_mau"}
    for d in c["doan"]:
        assert set(d) == {"vao_giay", "ra_giay", "dai_giay", "diem_tb", "diem_dinh", "giay_dinh", "muc", "may", "co",
                          "vao_frame", "ra_frame"}
        assert 0 <= d["vao_giay"] < d["ra_giay"] <= c["thoi_luong_giay"] + 1e-6
    assert (source / ".dailies" / "cham_diem.json").is_file()
    # A different threshold reuses the per-file scan
    assert main(["cham-diem", str(source), "--json", "--nguong", "0.9", "--kieu", "tinh"]) == 0
    lan_hai = json.loads(capsys.readouterr().out)
    assert lan_hai["so_clip_quet_moi"] == 0 and lan_hai["cai_dat"]["nguong"] == 0.9


def test_cli_sai_kieu(capsys):
    assert main(["cham-diem", "/khong/co", "--kieu", "la", "--json"]) == 2


# ---------------------------------------------------------------- Cutdown sessions

def _tao_cdwn(tep, file_clip):
    import msgpack
    import zstandard
    nen = zstandard.ZstdCompressor().compress
    id_clip, id_media = str(uuid.uuid4()), str(uuid.uuid4())
    phien = {
        "timeline": {"media_pool": [{"id": id_media, "path": str(file_clip), "original_path": str(file_clip), "name": file_clip.name,
                                     "proxy_path": None, "video": {"fps": {"numerator": 30000, "denominator": 1001}}}],
                     "tracks": [{"clips": [{"id": id_clip, "source_media_id": id_media, "timeline_in": 0,
                                            "tc_mapping": {"timeline_fps": {"numerator": 24, "denominator": 1},
                                                           "source_fps": {"numerator": 30000, "denominator": 1001},
                                                           "source_start_frame": 0}}]}]},
        "preset": {"name": "All-Rounder", "score_threshold": 0.8, "sampling_rate": {"PerSecond": 4.0}},
        "results": [{"clip_instance_id": id_clip, "segments": [
            {"timeline_start_frame": 0, "timeline_end_frame": 47, "mean_score": 0.85, "peak_score": 0.9,
             "camera_motion_summary": {"class": "static"}}]}],
    }
    khung = [{"source_frame_index": i * 7, "composite_score": 0.8, "qualifies": True,
              "flags": {"is_shaky": i == 3, "is_black_frame": False}, "scores": {"laplacian_variance": 0.5}} for i in range(12)]
    j = nen(json.dumps(phien).encode())
    f = nen(msgpack.packb(khung))
    du_lieu = b"CDWN" + struct.pack("<II", 9, len(j)) + j + struct.pack("<I", 1) + uuid.UUID(id_clip).bytes + b"\0" * 8 + struct.pack("<I", len(f)) + f
    tep.write_bytes(du_lieu)


def test_doc_cdwn(tmp_path):
    _tao_cdwn(tmp_path / "a.cdwn", tmp_path / "A001.mp4")
    cd = cutdown.doc_cdwn(tmp_path / "a.cdwn")
    assert cd["kieu"] == "All-Rounder" and cd["nguong"] == 0.8 and cd["so_clip"] == 1
    c = cd["clip"][0]
    assert c["doan"] == [{"vao_giay": 0.0, "ra_giay": 2.0, "diem_tb": 0.85, "diem_dinh": 0.9, "may": "dung_yen"}]
    assert len(c["mau"]) == 12 and c["mau"][3]["co"] == ["shaky"] and abs(c["mau"][1]["t"] - 7 * 1001 / 30000) < 1e-3


def test_doc_cdwn_sai_dinh_dang(tmp_path):
    (tmp_path / "x.cdwn").write_bytes(b"CDWN" + struct.pack("<II", 3, 0))
    with pytest.raises(ValueError, match="phiên bản 3"):
        cutdown.doc_cdwn(tmp_path / "x.cdwn")


def test_doi_chieu_cutdown(clip_mau, tmp_path, capsys):
    source = _source(clip_mau, tmp_path)
    assert main(["cham-diem", str(source), "--json"]) == 0
    capsys.readouterr()
    _tao_cdwn(tmp_path / "a.cdwn", source / "A001.mp4")
    assert main(["doi-chieu-cutdown", str(tmp_path / "a.cdwn"), "--json"]) == 0
    kq = json.loads(capsys.readouterr().out)
    assert kq["ok"] and kq["so_clip_khop"] == 1 and kq["so_mau_so_sanh"] > 5
    assert kq["clip"][0]["giay_cutdown"] == 2.0 and 0 <= kq["clip"][0]["trung_khop"] <= 1


def test_spearman():
    assert cutdown.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == 1.0
    assert cutdown.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == -1.0
    assert cutdown.spearman([1, 1, 1], [1, 2, 3]) is None


def test_lia_nhanh_mo_khong_phai_cat_cung():
    doi = lambda i: 0.2 if i == 10 else 0.99
    lech = lambda i: 40.0 if i == 10 else 3.0
    tin = lambda i: 0.0 if i == 10 else 1.0
    net_mo = _mau(20, tuong_quan=doi, lech=lech, tin_cay=tin, net=lambda i: 5.0 if i == 10 else 50.0)
    net_ro = _mau(20, tuong_quan=doi, lech=lech, tin_cay=tin)
    tc = {"net": 50.0, "chi_tiet": 0.05}
    assert "cat_cung" not in cham_diem.cham(net_mo, tc)[10]["co"]
    assert "cat_cung" in cham_diem.cham(net_ro, tc)[10]["co"]


def test_co_nhap_nhay_va_trung():
    tc = {"net": 50.0, "chi_tiet": 0.05}
    nhay = _mau(12, sang=lambda i: 120.0 + (15 if i % 2 else 0))
    mot_lan = _mau(12, sang=lambda i: 120.0 if i < 6 else 140.0)
    assert sum("nhap_nhay" in d["co"] for d in cham_diem.cham(nhay, tc)) >= 8
    assert not any("nhap_nhay" in d["co"] for d in cham_diem.cham(mot_lan, tc))
    tinh = _mau(8, lech=0.4)
    dung_hinh = _mau(8, lech=0.0)
    assert not any("trung" in d["co"] for d in cham_diem.cham(tinh, tc))
    assert all("trung" in d["co"] for d in cham_diem.cham(dung_hinh, tc)[1:])
