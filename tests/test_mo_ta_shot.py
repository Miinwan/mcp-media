"""Shot description: fast mode on sample clips, thorough mode with a fake model, the mlx-vlm subprocess protocol."""

import json
import shutil
import sys
import textwrap

from media.cli import main
from media import hinh_chu
from media import mo_ta_shot


def _source(clip_mau, tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    shutil.copyfile(clip_mau("mt-a", rong=320, cao=180, giay=4, cau=[]), source / "A010.mp4")
    shutil.copyfile(clip_mau("mt-b", rong=180, cao=320, giay=3, cau=[]), source / "A002.mp4")
    from sinh_mau import tao_proxy_premiere
    tao_proxy_premiere(source)
    return source


def test_nhanh_xuat_json_va_cache(clip_mau, tmp_path, capsys):
    source = _source(clip_mau, tmp_path)
    assert main(["mo-ta-shot", str(source), "--json"]) == 0
    kq = json.loads(capsys.readouterr().out)
    assert kq["ok"] and kq["che_do"] == "nhanh" and kq["model"] is None and kq["so_shot"] == 2
    assert [s["ten_file"] for s in kq["shot"]] == ["A002.mp4", "A010.mp4"]
    s = kq["shot"][1]
    assert set(s) == {"stt", "id", "file", "ten_file", "media_id", "fps", "thoi_luong_giay", "che_do", "model", "mo_ta",
                      "mo_ta_nhanh", "nhan", "chat_luong", "co_proxy", "anh"}
    assert len(s["anh"]) == 3 and all(a["mo_ta"] is None for a in s["anh"])
    assert s["mo_ta"].endswith(".") and "nét" in s["mo_ta"]
    assert s["chat_luong"]["rung"] is not None and s["co_proxy"]  # frames and shake come from the Premiere proxy
    if hinh_chu.da_co_model():
        assert set(s["nhan"]) == {"co_canh", "boi_canh", "noi_dung"}
        assert s["nhan"]["noi_dung"]["nhan"] == "hình thử nghiệm"  # testsrc2 is a colour-bar test pattern
    assert (source / ".media" / mo_ta_shot.TEN_TEP).is_file()
    # Second run reads the cache, no recompute
    assert main(["mo-ta-shot", str(source), "--json"]) == 0
    lan_hai = json.loads(capsys.readouterr().out)
    assert lan_hai["shot"] == kq["shot"] and lan_hai["giay_moi_anh"] is None


def test_nhanh_khi_chua_co_siglip(clip_mau, tmp_path, monkeypatch):
    source = _source(clip_mau, tmp_path)
    monkeypatch.setattr(hinh_chu, "da_co_model", lambda: False)
    kq = mo_ta_shot.mo_ta_thu_muc(source)
    assert all(s["nhan"] == {} for s in kq["shot"]) and kq["canh_bao"]


class MayGia:
    def __init__(self):
        self.so_lan = 0

    def mo_ta(self, anh):
        self.so_lan += 1
        assert anh.is_file()
        return {"mo_ta": "Một người đứng giữa sảnh trung tâm thương mại, trung cảnh, máy ngang tầm mắt.", "giay": 0.9}


def test_ky_dung_mo_ta_cua_model(clip_mau, tmp_path):
    source = _source(clip_mau, tmp_path)
    may = MayGia()
    kq = mo_ta_shot.mo_ta_thu_muc(source, che_do="ky", bo_mo_ta=lambda: may)
    assert may.so_lan == 2 and kq["model"] == "Qwen3.5-2B-MLX-4bit"
    s = kq["shot"][0]
    assert s["mo_ta"].startswith("Một người") and s["anh"][0]["mo_ta"] == s["mo_ta"]
    assert s["mo_ta_nhanh"] != s["mo_ta"]


def test_tien_trinh_con_theo_giao_thuc(tmp_path):
    gia = tmp_path / "gia.py"
    gia.write_text(textwrap.dedent('''
        import json, sys
        print(json.dumps({"san_sang": True, "giay_nap": 0.1}), flush=True)
        for dong in sys.stdin:
            yc = json.loads(dong)
            print(json.dumps({"mo_ta": "ảnh " + yc["anh"], "giay": 0.01}, ensure_ascii=False), flush=True)
    '''))
    may = mo_ta_shot.MoTaMlx(lenh=[sys.executable, str(gia)])
    assert may.mo_ta(tmp_path / "x.jpg")["mo_ta"].endswith("x.jpg")
    may.dong()


def test_ky_chua_cai_thi_hoi_hoac_bao_thieu(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(mo_ta_shot, "da_cai_ky", lambda ten="2b": False)
    assert main(["mo-ta-shot", str(tmp_path), "--che-do", "ky", "--json"]) == 4
    assert json.loads(capsys.readouterr().out)["loi"] == "thieu_model"
    assert main(["cai-mo-ta", "--json"]) == 3
    assert "--dong-y" in json.loads(capsys.readouterr().out)["cau_hoi"]
