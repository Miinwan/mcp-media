"""Cắt take thô theo khẩu lệnh: dò trên transcript giả và chạy cả lệnh CLI trên clip có giọng `say`."""

import json
import shutil
from fractions import Fraction
from types import SimpleNamespace

import pytest

from media.cli import main
from media.khau_lenh import cat_take, lap_danh_sach

FPS = Fraction(25)


def _ts(*cau):
    """cau: danh sách câu, mỗi câu là list (chữ, giây vào, giây ra). Mốc đổi ra frame 25 fps như cache phân tích."""
    return {"cau": [{"vao": round(c[0][1] * 25), "ra": round(c[-1][2] * 25), "noi_dung": " ".join(x[0] for x in c),
                     "tu": [{"chu": w, "vao": round(a * 25), "ra": round(b * 25), "xac_suat": 0.95} for w, a, b in c]}
                    for c in cau]}


@pytest.mark.parametrize("dem", [["3,", "2,", "1,"], ["Ba,", "hai,", "một."], ["321"], ["3,2,1"], ["ba", "2", "một"]])
def test_dem_nguoc_moi_cach_viet(dem):
    buoc = 0.3
    cau = [(w, i * buoc, i * buoc + 0.25) for i, w in enumerate(dem)]
    t = cat_take(_ts(cau, [("Xin", 2.0, 2.3), ("chào", 2.3, 2.6)]), FPS, 6.0)
    assert len(t) == 1 and t[0]["khau_lenh_vao"]["loai"] == "dem_nguoc"
    assert float(t[0]["vao"]) == pytest.approx(cau[-1][2] + 0.1, abs=0.05)
    assert t[0]["ket_thuc_bang"] == "het_file" and float(t[0]["ra"]) == 6.0


@pytest.mark.parametrize("lenh, loai", [(["Action!"], "action"), (["ắc", "sần"], "action"), (["Diễn."], "dien"), (["Đi!"], "di")])
def test_lenh_bat_dau_dung_rieng(lenh, loai):
    cau = [(w, 1.0 + i * 0.2, 1.15 + i * 0.2) for i, w in enumerate(lenh)]
    t = cat_take(_ts(cau, [("Hôm", 2.5, 2.8), ("nay", 2.8, 3.0)]), FPS, 5.0)
    assert [x["khau_lenh_vao"]["loai"] for x in t] == [loai]


def test_tu_thuong_trong_cau_khong_tinh():
    """"đi", "ok", "cắt" nói liền trong câu thoại không phải khẩu lệnh."""
    cau = [("Chúng", 0.0, 0.2), ("ta", 0.2, 0.35), ("đi", 0.35, 0.5), ("chợ", 0.5, 0.7), ("ok", 0.7, 0.85),
           ("rồi", 0.85, 1.0), ("cắt", 1.0, 1.2), ("rau", 1.2, 1.4)]
    assert cat_take(_ts(cau), FPS, 3.0) == []


def test_nhieu_take_mot_file_va_khau_lenh_ket_thuc():
    ts = _ts([("3,", 0.0, 0.3), ("2,", 0.4, 0.7), ("1,", 0.8, 1.1), ("diễn.", 1.3, 1.6)],
             [("Câu", 2.0, 2.3), ("một.", 2.3, 2.6)],
             [("Cắt.", 3.5, 3.8)],
             [("Nói", 4.2, 4.4), ("lại", 4.4, 4.6), ("nhé.", 4.6, 4.8)],  # lời giữa hai take: bị bỏ
             [("Action.", 6.0, 6.4)],
             [("Câu", 7.0, 7.3), ("hai.", 7.3, 7.6)],
             [("Được", 8.5, 8.7), ("rồi.", 8.7, 9.0)],
             [("Action!", 10.0, 10.4)],
             [("Câu", 11.0, 11.3), ("ba.", 11.3, 11.6)])
    lang = [{"vao": round(11.7 * 25), "ra": round(14.0 * 25)}]
    t = cat_take(ts, FPS, 14.0, lang)
    assert [x["khau_lenh_vao"]["loai"] for x in t] == ["dem_nguoc+dien", "action", "action"]
    assert [x["ket_thuc_bang"] for x in t] == ["khau_lenh_ra", "khau_lenh_ra", "lang_cuoi"]
    assert [x["loi_thoai"] for x in t] == ["Câu một.", "Câu hai.", "Câu ba."]
    assert float(t[0]["ra"]) == pytest.approx(3.45, abs=0.04) and float(t[1]["ra"]) == pytest.approx(8.45, abs=0.04)
    assert float(t[2]["ra"]) == pytest.approx(12.0, abs=0.04)  # cắt bớt lặng cuối, giữ 0,3 giây
    assert t[0]["do_tin_cay"] > t[2]["do_tin_cay"] > 0


def test_khong_co_khau_lenh_ket_thuc_thi_toi_lenh_ke_tiep():
    ts = _ts([("Action.", 1.0, 1.4)], [("Một", 2.0, 2.3)], [("Action.", 4.0, 4.4)], [("Hai", 5.0, 5.3)])
    t = cat_take(ts, FPS, 6.0)
    assert t[0]["ket_thuc_bang"] == "khau_lenh_vao_ke_tiep" and float(t[0]["ra"]) == pytest.approx(3.95)


def test_bo_cau_whisper_bia():
    bia = [(w, 3.0, 3.01) for w in "Cảm ơn các bạn đã theo dõi".split()]
    ts = _ts([("Action.", 1.0, 1.4)], bia, [("Đi.", 4.0, 4.3)], [("Hai", 5.0, 5.3)])
    t = cat_take(ts, FPS, 6.0)
    assert [x["khau_lenh_vao"]["loai"] for x in t] == ["action", "di"]


@pytest.mark.parametrize("cau, loai_vao, loai_ra", [
    ([[("Three,", 0.0, 0.3), ("two,", 0.4, 0.7), ("one,", 0.8, 1.1), ("action!", 1.3, 1.7)]], "dem_nguoc+action", None),
    ([[("Rolling.", 0.0, 0.5)], [("And", 3.0, 3.2), ("action.", 3.2, 3.6)]], "action", None),
    ([[("We're", 0.0, 0.2), ("rolling.", 0.2, 0.6)]], "rolling", None),
])
def test_khau_lenh_tieng_anh(cau, loai_vao, loai_ra):
    ts = _ts(*cau, [("Good", 5.0, 5.3), ("evening.", 5.3, 5.8)], [("That's", 7.0, 7.3), ("good.", 7.3, 7.6)])
    t = cat_take(ts, FPS, 10.0)
    assert [x["khau_lenh_vao"]["loai"] for x in t] == [loai_vao]  # "rolling" trước "action" nhường, không thành take riêng
    assert t[0]["khau_lenh_ra"]["loai"] == "thats_good" and t[0]["loi_thoai"] == "Good evening."


def test_cut_okay_trong_cau_tieng_anh_khong_tinh():
    cau = [("Please", 2.0, 2.2), ("cut", 2.2, 2.4), ("the", 2.4, 2.5), ("cake,", 2.5, 2.8), ("okay", 2.8, 3.0),
           ("that's", 3.0, 3.2), ("good", 3.2, 3.4), ("food.", 3.4, 3.7)]
    t = cat_take(_ts([("Action.", 0.0, 0.4)], cau), FPS, 6.0)
    assert t[0]["khau_lenh_ra"] is None and t[0]["loi_thoai"].startswith("Please cut")


class ModelGia:
    """Trả lời thoại theo tên clip (mỗi clip một độ dài âm thanh khác nhau)."""

    def transcribe(self, pcm, **_):
        w = SimpleNamespace
        if len(pcm) > 16000 * 5:
            return [w(start=0.0, end=1.6, text="Ba hai một diễn", words=[
                        w(word="Ba,", start=0.0, end=0.3, probability=0.9), w(word="hai,", start=0.4, end=0.7, probability=0.9),
                        w(word="một,", start=0.8, end=1.1, probability=0.9), w(word="diễn.", start=1.3, end=1.6, probability=0.9)]),
                    w(start=2.0, end=2.6, text="Xin chào", words=[w(word="Xin", start=2.0, end=2.3, probability=1.0),
                                                                  w(word="chào", start=2.3, end=2.6, probability=1.0)]),
                    w(start=4.0, end=4.3, text="Cắt.", words=[w(word="Cắt.", start=4.0, end=4.3, probability=0.9)]),
                    w(start=6.0, end=6.4, text="Action.", words=[w(word="Action.", start=6.0, end=6.4, probability=0.9)])], None
        return [], None


def test_cli_xuat_json_on_dinh(clip_mau, tmp_path, monkeypatch):
    dai = clip_mau("kl-dai", rong=320, cao=180, giay=10)
    dai2 = clip_mau("kl-dai-2", rong=320, cao=180, giay=10)  # khác mã băm để không bị gộp làm một media
    ngan = clip_mau("kl-ngan", rong=320, cao=180, giay=3, cau=["Không có khẩu lệnh."])
    source = tmp_path / "source"
    (source / "Ngày 2").mkdir(parents=True)
    shutil.copyfile(dai, source / "A010.mp4")
    shutil.copyfile(dai2, source / "Ngày 2" / "C001.mp4")
    shutil.copyfile(ngan, source / "A002.mp4")
    from media import khau_lenh
    from media import phan_tich
    goc = phan_tich.phan_tich_thu_muc
    monkeypatch.setattr(phan_tich, "khoang_lang", lambda *a, **k: [])  # lời giả không khớp khoảng lặng của giọng `say`
    monkeypatch.setattr(phan_tich, "phan_tich_thu_muc", lambda *a, **k: goc(*a, **{**k, "model_factory": lambda _: ModelGia()}))
    ma = main(["take-khau-lenh", str(source), "--model", "tiny", "--json"])
    assert ma == 0
    kq = json.loads((source / ".media" / khau_lenh.TEN_TEP).read_text())
    assert kq["phien_ban"] == "1" and kq["so_file"] == 3
    assert [(t["ten_file"], t["take_trong_file"]) for t in kq["take"]] == [
        ("A002.mp4", 1), ("A010.mp4", 1), ("A010.mp4", 2), ("Ngày 2/C001.mp4", 1), ("Ngày 2/C001.mp4", 2)]
    assert [t["stt"] for t in kq["take"]] == [1, 2, 3, 4, 5]
    t = kq["take"][1]
    assert set(t) == {"stt", "id", "file", "ten_file", "media_id", "fps", "thoi_luong_file_giay", "take_trong_file",
                      "vao_giay", "ra_giay", "do_dai_giay", "vao_frame", "ra_frame", "khau_lenh_vao", "khau_lenh_ra",
                      "ket_thuc_bang", "do_tin_cay", "ca_file", "loi_thoai", "bo_cuc", "ngon_ngu"}
    assert kq["khung"] == {"rong": 1920, "cao": 1080}
    assert t["bo_cuc"]["cach"] == "vua_khung" and t["bo_cuc"]["scale_phan_tram"] == 600.0  # clip mẫu 320x180
    assert t["khau_lenh_vao"]["loai"] == "dem_nguoc+dien" and t["khau_lenh_ra"]["loai"] == "cat"
    assert t["loi_thoai"] == "Xin chào" and t["vao_frame"] < t["ra_frame"]
    fps = t["fps"]["tu"] / t["fps"]["mau"]
    assert t["vao_frame"] == round(t["vao_giay"] * fps)
    assert kq["take"][0]["ca_file"] and kq["take"][0]["do_tin_cay"] == 0
    assert [f["ten_file"] for f in kq["file_khong_co_khau_lenh"]] == ["A002.mp4"]
    # Chạy lại dùng cache, kết quả giống hệt (MCP Premiere gọi nhiều lần)
    assert main(["take-khau-lenh", str(source), "--model", "tiny", "--json", "--bo-file-khong-lenh"]) == 0
    lan_hai = json.loads((source / ".media" / khau_lenh.TEN_TEP).read_text())
    assert lan_hai["take"] == [dict(x, stt=x["stt"] - 1) for x in kq["take"][1:]]


def test_nghe_that_giong_tieng_anh(clip_mau, tmp_path):
    """Whisper tiny thật (chỉ dùng bản đã có trên máy, không tải trong pytest), giọng `say` tiếng Anh, tự nhận ngôn ngữ."""
    import subprocess
    from pathlib import Path
    hub = Path.home() / ".cache" / "huggingface" / "hub" / "models--Systran--faster-whisper-tiny" / "snapshots"
    if not hub.is_dir() or not any(hub.glob("*/model.bin")):
        pytest.skip("Chưa có Whisper tiny trên máy")
    giong = next((g for g in ("Samantha", "Alex", "Fred", "Albert")
                  if g in subprocess.run(["say", "-v", "?"], capture_output=True, text=True).stdout), None)
    if giong is None:
        pytest.skip("Không có giọng tiếng Anh cho say")
    clip = clip_mau("kl-anh", rong=320, cao=180, giay=14, giong=giong,
                    cau=["Three, two, one, action.", "Good evening, welcome to our restaurant.", "Cut.",
                         "Rolling.", "And action.", "Would you like some wine?"])
    source = tmp_path / "source"
    source.mkdir()
    shutil.copyfile(clip, source / "A001.mp4")
    from media.khau_lenh import take_thu_muc
    kq = take_thu_muc(source, model_ten="tiny")
    assert [t["ngon_ngu"] for t in kq["take"]] == ["en", "en"] and kq["ngon_ngu_dat"] == "auto"
    assert [t["khau_lenh_vao"]["loai"] for t in kq["take"]] == ["dem_nguoc+action", "action"]
    assert kq["take"][0]["khau_lenh_ra"]["loai"] == "cut"
    assert "welcome" in kq["take"][0]["loi_thoai"].lower() and "wine" in kq["take"][1]["loi_thoai"].lower()

