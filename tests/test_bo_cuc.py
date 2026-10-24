"""Rough-cut framing: scale and position for placing footage in a sequence."""

import pytest

from dailies.bo_cuc import doc_khung_chuoi, tinh


def test_4k_vao_full_hd_vua_khung():
    b = tinh((3840, 2160), (1920, 1080), (0.8, 0.3), 1)
    assert b["cach"] == "vua_khung" and b["scale_phan_tram"] == 50.0
    assert b["vi_tri"] == {"x": 960.0, "y": 540.0} and b["vi_tri_chuan"] == {"x": 0.5, "y": 0.5}


def test_ngang_vao_doc_theo_mat():
    b = tinh((3840, 2160), (1080, 1920), (0.7, 0.45), 1)
    assert b["cach"] == "day_khung_theo_mat" and b["scale_phan_tram"] == pytest.approx(88.889, abs=1e-3)
    rong = 3840 * 1920 / 2160
    # A face at 70% of source width moves to the centre of the vertical frame
    assert b["vi_tri"]["x"] == pytest.approx(540 - 0.2 * rong, abs=0.1)
    assert b["vi_tri"]["y"] == 960.0  # fills the height, no room to move vertically


def test_khong_lo_vien_khi_mat_sat_mep():
    b = tinh((3840, 2160), (1080, 1920), (0.99, 0.5), 1)
    rong = 3840 * 1920 / 2160
    assert b["vi_tri"]["x"] == pytest.approx(1080 - rong / 2, abs=0.1)  # image right edge meets frame right edge


def test_khong_mat_thi_lay_giua():
    b = tinh((1920, 1080), (1080, 1920), None)
    assert b["cach"] == "day_khung_giua" and b["vi_tri"] == {"x": 540.0, "y": 960.0} and b["tam_chu_the"] is None


@pytest.mark.parametrize("chuoi, ket_qua", [("1920x1080", (1920, 1080)), ("9:16", (1080, 1920)), ("3840X2160", (3840, 2160))])
def test_doc_khung(chuoi, ket_qua):
    assert doc_khung_chuoi(chuoi) == ket_qua


def test_khung_sai():
    with pytest.raises(ValueError):
        doc_khung_chuoi("abc")


def _dung(anh, khung, scale, vi_tri, neo=None, crop=None):
    """Render a frame like Premiere: Crop in % of clip, scale around the anchor point, anchor at position; black outside the clip."""
    import cv2
    import numpy as np
    h, w = anh.shape[:2]
    a = anh.copy()
    if crop:
        a[:, :round(w * crop.get("trai", 0) / 100)] = 0
        a[:round(h * crop.get("tren", 0) / 100)] = 0
        a[:, w - round(w * crop.get("phai", 0) / 100):] = 0
        a[h - round(h * crop.get("duoi", 0) / 100):] = 0
    s = scale / 100
    neo = neo or {"x": w / 2, "y": h / 2}
    m = np.float32([[s, 0, vi_tri["x"] - neo["x"] * s], [0, s, vi_tri["y"] - neo["y"] * s]])
    return cv2.warpAffine(a, m, khung, flags=cv2.INTER_AREA, borderValue=0)


def _anh_nguon(w=1920, h=1080):
    import numpy as np
    y, x = np.mgrid[0:h, 0:w]
    anh = np.stack([40 + (x * 200 // w), 40 + (y * 200 // h), 40 + ((x + y) * 150 // (w + h))], -1)
    return anh.astype(np.uint8)  # every pixel brighter than 40: a black pixel in the rendered frame is an exposed edge


NGANG = [
    {"scale_phan_tram": 100 / 1, "vi_tri": {"x": 960, "y": 540}},                                       # fit to frame
    {"scale_phan_tram": 160, "vi_tri": {"x": 1200, "y": 600}},                                         # editor zoomed and moved
    {"scale_phan_tram": 130, "vi_tri": {"x": 700, "y": 400}, "diem_neo": {"x": 300, "y": 200}},        # anchor point changed
    {"scale_phan_tram": 120, "vi_tri": {"x": 960, "y": 540}, "crop": {"tren": 8, "duoi": 12, "trai": 5}},  # effect Crop
]


@pytest.mark.parametrize("ngang", NGANG)
@pytest.mark.parametrize("khung", [(1080, 1080), (1080, 1920)])
@pytest.mark.parametrize("tam", [None, (0.3, 0.4), (0.9, 0.1)])
def test_ban_moi_trung_pixel_ban_ngang_khong_lo_vien(ngang, khung, tam):
    import cv2
    import numpy as np
    from dailies.bo_cuc import tu_ban_ngang
    anh = _anh_nguon()
    ngang_anh = _dung(anh, (1920, 1080), ngang["scale_phan_tram"], ngang["vi_tri"], ngang.get("diem_neo"), ngang.get("crop"))
    b = tu_ban_ngang((1920, 1080), (1920, 1080), ngang, khung, tam, 1 if tam else 0)
    moi = _dung(anh, khung, b["scale_phan_tram"], b["vi_tri"], ngang.get("diem_neo"), ngang.get("crop"))
    cs = b["cua_so_ban_ngang"]
    mong_doi = cv2.resize(ngang_anh[round(cs["y"]):round(cs["y"] + cs["cao"]), round(cs["x"]):round(cs["x"] + cs["rong"])],
                          khung, interpolation=cv2.INTER_LINEAR)
    vien = moi[2:-2, 2:-2].min(axis=2) < 30
    assert not vien.any(), f"lộ viền đen {vien.mean():.2%}"
    lech = np.abs(moi.astype(int) - mong_doi.astype(int))[4:-4, 4:-4].mean()
    assert lech < 3, f"khác bản ngang trung bình {lech:.1f} mức"


def test_mat_ngoai_khung_ngang_khong_duoc_dung():
    from dailies.bo_cuc import tu_ban_ngang
    # The editor zoomed into the left half; a face in the (cropped-out) right half must not pull the window
    b = tu_ban_ngang((1920, 1080), (1920, 1080), {"scale_phan_tram": 200, "vi_tri": {"x": 1920, "y": 540}},
                     (1080, 1920), (0.8, 0.5), 1)
    assert b["cach"] == "cat_tu_ban_ngang_giua"


def test_vung_hep_thi_phong_them_va_canh_bao():
    from dailies.bo_cuc import tu_ban_ngang
    # Landscape clip scaled to 50% (picture in picture): visible area only 960x540
    b = tu_ban_ngang((1920, 1080), (1920, 1080), {"scale_phan_tram": 50, "vi_tri": {"x": 960, "y": 540}}, (1080, 1920), None)
    assert b["he_so_phong"] == pytest.approx(1920 / 540, abs=1e-4) and b["canh_bao"]


def test_xoay_thi_canh_bao():
    from dailies.bo_cuc import tu_ban_ngang
    b = tu_ban_ngang((1920, 1080), (1920, 1080), {"xoay": 5}, (1080, 1080), None)
    assert any("xoay" in c for c in b["canh_bao"])


def test_lenh_bo_cuc_bo_qua_clip_khong_file(clip_mau, tmp_path, capsys):
    import json
    from dailies.cli import main
    clip = clip_mau("bc-ngang", rong=640, cao=360, giay=3, cau=[])
    vao = tmp_path / "clip.json"
    vao.write_text(json.dumps({"khung": {"rong": 1920, "cao": 1080}, "clip": [
        {"id": "a", "file": str(clip), "vao_giay": 0.5, "ra_giay": 2.5},
        {"id": "adj", "file": None}]}))
    assert main(["bo-cuc", str(vao), "--json"]) == 0
    kq = json.loads(capsys.readouterr().out)
    assert [b["khung"] for b in kq["ban"]] == [{"rong": 1080, "cao": 1080}, {"rong": 1080, "cao": 1920}]
    vuong = kq["ban"][0]["clip"]
    assert vuong[0]["bo_cuc"]["scale_phan_tram"] == 300.0 and vuong[0]["bo_cuc"]["vi_tri"] == {"x": 540.0, "y": 540.0}
    assert vuong[0]["bo_cuc"]["cua_so_ban_ngang"] == {"x": 420.0, "y": 0.0, "rong": 1080.0, "cao": 1080.0}
    assert vuong[1] == {"id": "adj", "file": None, "bo_cuc": None, "ly_do": "khong_co_file"}
