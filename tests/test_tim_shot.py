"""Tìm shot bằng câu chữ: đoạn quanh đỉnh điểm, gộp cách diễn đạt, lệnh CLI với model giả và SigLIP thật."""

import json
import shutil

import numpy as np
import pytest

from media import hinh_chu, tim_shot
from media.cli import main


def test_doan_quanh_dinh():
    diem = np.array([0.01, 0.02, 0.10, 0.11, 0.10, 0.02, 0.01, 0.05, 0.06, 0.05, 0.01])
    d = tim_shot.cac_doan(diem, 1.0, 11.0)
    assert [round(x["dinh"]) for x in d[:2]] == [3, 8]
    assert d[0]["vao"] == pytest.approx(2.5) and d[0]["ra"] == pytest.approx(3.5)  # chỉ phần sát đỉnh
    assert d[0]["diem"] > d[1]["diem"]
    assert all(not (a["vao"] < b["dinh"] < a["ra"]) for a in d for b in d if a is not b)  # đỉnh không nằm trong đoạn khác


def test_cach_noi_them_tieng_anh_cho_co_canh():
    ra = tim_shot.cac_cach_noi("Cận mặt cô gái")
    assert ra[:2] == ["Cận mặt cô gái", "ảnh chụp Cận mặt cô gái"] and "a close-up shot of a person's face" in ra
    assert tim_shot.cac_cach_noi("người phục vụ áo đỏ") == ["người phục vụ áo đỏ", "ảnh chụp người phục vụ áo đỏ"]


def _source(clip_mau, tmp_path):
    from sinh_mau import tao_proxy_premiere
    source = tmp_path / "source"
    source.mkdir()
    shutil.copyfile(clip_mau("ts-a", rong=320, cao=180, giay=6, cau=[]), source / "A001.mp4")
    shutil.copyfile(clip_mau("ts-b", rong=320, cao=180, giay=4, cau=[]), source / "A002.mp4")
    tao_proxy_premiere(source)
    return source


def test_cli_voi_model_gia(clip_mau, tmp_path, monkeypatch, capsys):
    """Ảnh giây thứ 4 của A001 khớp câu nhất; embedding cache lại, câu sau không mã hoá ảnh lần nữa."""
    source = _source(clip_mau, tmp_path)
    so_lan = {"anh": 0}

    def ma_hoa_anh(anh):
        so_lan["anh"] += 1
        v = np.zeros((len(anh), 4), np.float32)
        v[:, 0] = 1
        if len(anh) >= 6:
            v[4] = [0.6, 0.8, 0, 0]
        return v

    monkeypatch.setattr(hinh_chu, "da_co_model", lambda: True)
    monkeypatch.setattr(hinh_chu, "ma_hoa_anh", ma_hoa_anh)
    monkeypatch.setattr(hinh_chu, "ma_hoa_chu", lambda cac: np.tile([[0, 1, 0, 0]], (len(cac), 1)).astype(np.float32))
    assert main(["tim-shot", "cô gái cầm menu", str(source), "--top", "3", "--json"]) == 0
    kq = json.loads(capsys.readouterr().out)
    assert kq["phien_ban"] == "1" and kq["cau"] == "cô gái cầm menu" and kq["so_ket_qua"] == 2  # mỗi clip một đồi điểm
    r = kq["ket_qua"][0]
    assert set(r) == {"hang", "file", "ten_file", "media_id", "fps", "proxy", "vao_giay", "ra_giay", "do_dai_giay",
                      "vao_frame", "ra_frame", "dinh_giay", "diem"}
    assert r["hang"] == 1 and r["ten_file"] == "A001.mp4" and r["dinh_giay"] == 4.0 and r["proxy"].endswith("A001_Proxy.mov")
    assert r["vao_giay"] < 4 < r["ra_giay"]
    assert (source / ".media" / "tim_shot.json").is_file() and so_lan["anh"] == 2
    assert main(["tim-shot", "cận mặt", str(source), "--json"]) == 0
    capsys.readouterr()
    assert so_lan["anh"] == 2


@pytest.mark.skipif(not hinh_chu.da_co_model(), reason="chưa tải SigLIP 2")
def test_siglip_that(clip_mau, tmp_path):
    source = _source(clip_mau, tmp_path)
    kq = tim_shot.tim("hình sọc màu thử nghiệm của máy quay", source, top=2)
    assert kq["so_ket_qua"] == 2 and all(r["diem"] > 0 for r in kq["ket_qua"])


def test_thieu_model_tra_ma_4(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(hinh_chu, "da_co_model", lambda: False)
    assert main(["tim-shot", "cận mặt", str(tmp_path), "--json"]) == 4
    assert json.loads(capsys.readouterr().out)["loi"] == "thieu_model"
