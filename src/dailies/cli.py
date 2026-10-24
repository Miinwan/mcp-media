"""The `dailies` command line for the Premiere Pro and After Effects MCP servers.

Convention: `--json` prints exactly one JSON object to stdout; logs and progress go to stderr. Exit codes: 0 done;
1 data error; 2 usage error; 3 needs user confirmation (with `cau_hoi`); 4 model missing; 5 another run is busy on
the same source folder (with `pid`).
"""

import argparse
import json
import sys
from pathlib import Path

from dailies import __version__


class LoiCuPhap(Exception):
    """argparse error that exits with code 2."""


class Parser(argparse.ArgumentParser):
    def add_subparsers(self, **kwargs):
        kwargs.setdefault("parser_class", Parser)
        return super().add_subparsers(**kwargs)

    def error(self, message: str) -> None:
        raise LoiCuPhap(message)


def _doc_fps(chuoi: str | None) -> dict | None:
    """'25', '29.97', '30000/1001' to {tu, mau}."""
    if not chuoi:
        return None
    from fractions import Fraction
    ntsc = {"23.976": "24000/1001", "23.98": "24000/1001", "29.97": "30000/1001", "47.952": "48000/1001",
            "59.94": "60000/1001", "119.88": "120000/1001"}
    f = Fraction(ntsc.get(chuoi.strip(), chuoi.strip()))
    if f <= 0:
        raise ValueError(f"FPS không hợp lệ: {chuoi}")
    return {"tu": f.numerator, "mau": f.denominator}


def _take_khau_lenh(args) -> tuple[int, dict]:
    from dailies.bo_cuc import doc_khung_chuoi
    from dailies.khau_lenh import take_thu_muc
    kq = take_thu_muc(args.thu_muc, danh_sach=args.danh_sach, cho_phep_goc=args.cho_phep_goc, model_ten=args.model, cache_goc=args.cache, fps=_doc_fps(args.fps),
                      giu_file_khong_lenh=not args.bo_file_khong_lenh, khung=doc_khung_chuoi(args.khung),
                      ngon_ngu=args.ngon_ngu)
    return 0, {"ok": True, **kq}


def _mo_ta_shot(args) -> tuple[int, dict]:
    from dailies import mo_ta_shot
    if args.che_do == "ky" and not mo_ta_shot.da_cai_ky(args.model):
        ten, _, co = mo_ta_shot.MODEL_KY[args.model]
        return 4, {"ok": False, "loi": "thieu_model", "thong_bao": f"Chế độ kỹ cần {ten} ({co}) và mlx-vlm; "
                   f"chạy: dailies cai-mo-ta --model {args.model} --dong-y"}
    kq = mo_ta_shot.mo_ta_thu_muc(args.thu_muc, danh_sach=args.danh_sach, cho_phep_goc=args.cho_phep_goc, che_do=args.che_do, so_anh=args.so_anh, model=args.model, cache_goc=args.cache)
    return 0, {"ok": True, **kq}


def _cai_mo_ta(args) -> tuple[int, dict]:
    from dailies import mo_ta_shot
    ten, nguon, co = mo_ta_shot.MODEL_KY[args.model]
    if mo_ta_shot.da_cai_ky(args.model):
        return 0, {"ok": True, "thong_bao": f"Đã có {ten}", "model": str(mo_ta_shot.thu_muc_model_ky(args.model))}
    if not args.dong_y:
        return 3, {"ok": False, "loi": "can_xac_nhan", "cau_hoi": f"Tải {ten} ({co}, {nguon}) và cài {mo_ta_shot.MLX_VLM} "
                   f"vào môi trường riêng {mo_ta_shot.thu_muc_moi_truong()}? Gọi lại với --dong-y"}
    return 0, {"ok": True, **mo_ta_shot.cai_ky(args.model)}


def _tim_shot(args) -> tuple[int, dict]:
    from dailies import hinh_chu
    from dailies.tim_shot import tim
    if not hinh_chu.da_co_model():
        return 4, {"ok": False, "loi": "thieu_model", "thong_bao": "Tìm shot cần SigLIP 2; chạy: dailies tai-model"}
    kq = tim(args.cau, args.thu_muc, danh_sach=args.danh_sach, cho_phep_goc=args.cho_phep_goc, top=args.top,
             buoc=args.buoc, cache_goc=args.cache)
    return 0, {"ok": True, **kq}


def _tai_model(args) -> tuple[int, dict]:
    from dailies import hinh_chu, thi_giac
    return 0, {"ok": True, "siglip": str(hinh_chu.tai_model()), "yunet": str(thi_giac.tai_yunet())}


def _bo_cuc(args) -> tuple[int, dict]:
    from dailies.bo_cuc import doc_khung_chuoi, doi_ban
    du_lieu = json.loads(Path(args.clip_json).read_text(encoding="utf-8"))
    cac_khung = [doc_khung_chuoi(k) for k in args.khung.split(",") if k.strip()]
    return 0, {"ok": True, **doi_ban(du_lieu, cac_khung)}


def _cham_diem(args) -> tuple[int, dict]:
    from dailies.cham_diem import cham_thu_muc
    kq = cham_thu_muc(args.thu_muc, danh_sach=args.danh_sach, cho_phep_goc=args.cho_phep_goc, kieu=args.kieu,
                      nguong=args.nguong, ngan_nhat=args.ngan_nhat, noi_khoang=args.noi_khoang,
                      mau_moi_giay=args.mau_moi_giay, cache_goc=args.cache)
    return 0, {"ok": True, **kq}


def _doi_chieu_cutdown(args) -> tuple[int, dict]:
    from dailies.cutdown import doi_chieu
    return 0, {"ok": True, **doi_chieu(args.cdwn, args.ket_qua)}


def _thuoc_tinh(args) -> tuple[int, dict]:
    from dailies.bo_cuc import doc_khung_chuoi
    from dailies.thuoc_tinh import quet_thu_muc
    from fractions import Fraction
    f = _doc_fps(args.fps)
    kq = quet_thu_muc(args.thu_muc, danh_sach=args.danh_sach, khung=doc_khung_chuoi(args.khung) if args.khung else None,
                      fps=Fraction(f["tu"], f["mau"]) if f else None, cache_goc=args.cache)
    return 0, {"ok": True, **kq}


def _xuat(args) -> tuple[int, dict]:
    from dailies.xuat import xuat
    dd = ["xml", "fcpxml", "edl", "csv"] if args.dinh_dang == "tat-ca" else [x.strip() for x in args.dinh_dang.split(",") if x.strip()]
    from dailies.bo_cuc import doc_khung_chuoi
    from fractions import Fraction
    f = _doc_fps(args.fps)
    return 0, {"ok": True, **xuat(args.ket_qua, dd, args.ra, ten=args.ten, thu_tu=args.thu_tu, khoang=args.khoang, dem=args.dem,
                                  khung=doc_khung_chuoi(args.khung) if args.khung else None,
                                  fps=Fraction(f["tu"], f["mau"]) if f else None)}


def _them_json(p: argparse.ArgumentParser) -> None:
    p.add_argument("--json", action="store_true", default=argparse.SUPPRESS)


def _them_nguon(p: argparse.ArgumentParser) -> None:
    p.add_argument("thu_muc", nargs="?", help="Thư mục source (bỏ trống khi dùng --danh-sach)")
    p.add_argument("--danh-sach", help='JSON {"clip": [{"file", "proxy"}]}: các clip anh giữ sau khi lọc rác, kèm proxy Premiere')
    p.add_argument("--cho-phep-goc", action="store_true", help="Clip thiếu proxy thì đọc file gốc thay vì báo lỗi")


def main(argv: list[str] | None = None) -> int:
    parser = Parser(prog="dailies", description="Xem footage thay editor cho MCP Premiere và After Effects.")
    parser.add_argument("--version", action="version", version=f"dailies {__version__}")
    parser.add_argument("--json", action="store_true")
    sub = parser.add_subparsers(dest="lenh", required=True)

    p = sub.add_parser("take-khau-lenh", help="Cắt take thô theo khẩu lệnh trong audio gốc (3-2-1, action, diễn ... cắt), kèm bố cục cho sequence")
    _them_json(p)
    _them_nguon(p)
    p.add_argument("--khung", default="1920x1080", help="Khung sequence để tính scale, vị trí (1920x1080, 9:16...)")
    p.add_argument("--model", default="large-v3", help="Model faster-whisper (mặc định large-v3)")
    p.add_argument("--ngon-ngu", choices=["auto", "vi", "en"], default="auto", help="Ngôn ngữ nghe (mặc định tự nhận theo từng file)")
    p.add_argument("--fps", help="Tính frame theo fps này (25, 29.97, 30000/1001); mặc định fps từng file")
    p.add_argument("--cache", help="Thư mục đặt .dailies (mặc định thư mục source)")
    p.add_argument("--bo-file-khong-lenh", action="store_true", help="Bỏ file không nghe thấy khẩu lệnh (mặc định giữ cả file)")
    p.set_defaults(xu_ly=_take_khau_lenh)

    p = sub.add_parser("mo-ta-shot", help="Mô tả nội dung từng shot bằng tiếng Việt (nhanh: thuật toán, kỹ: model vision)")
    _them_json(p)
    _them_nguon(p)
    p.add_argument("--che-do", choices=["nhanh", "ky"], default="nhanh")
    p.add_argument("--so-anh", type=int, help="Số ảnh mỗi shot (mặc định nhanh 3, kỹ 1)")
    p.add_argument("--model", choices=["2b", "0.8b"], default="2b", help="Model chế độ kỹ")
    p.add_argument("--cache", help="Thư mục đặt .dailies (mặc định thư mục source)")
    p.set_defaults(xu_ly=_mo_ta_shot)

    p = sub.add_parser("tim-shot", help="Tìm shot bằng câu chữ (SigLIP 2): \"cô gái cầm menu\", \"cận mặt\"...")
    _them_json(p)
    p.add_argument("cau", help="Câu mô tả cần tìm (tiếng Việt hoặc tiếng Anh)")
    _them_nguon(p)
    p.add_argument("--top", type=int, default=10, help="Số đoạn trả về (mặc định 10)")
    p.add_argument("--buoc", type=float, default=1.0, help="Giây giữa hai ảnh mẫu (mặc định 1)")
    p.add_argument("--cache", help="Thư mục đặt .dailies (mặc định thư mục source)")
    p.set_defaults(xu_ly=_tim_shot)

    p = sub.add_parser("cham-diem", help="Chấm điểm footage (nét, sáng, rung, chuyển động máy) và chọn đoạn dùng được")
    _them_json(p)
    _them_nguon(p)
    p.add_argument("--kieu", choices=["can_bang", "b_roll", "tinh"], default="can_bang",
                   help="can_bang (mặc định), b_roll (máy di chuyển mượt không bị trừ), tinh (chỉ shot đứng yên)")
    p.add_argument("--nguong", type=float, default=0.85, help="Điểm tối thiểu để chọn (0-1, mặc định 0.85, khớp mặc định của Cutdown trên footage thật)")
    p.add_argument("--ngan-nhat", type=float, default=2.0, help="Đoạn ngắn nhất, giây (mặc định 2)")
    p.add_argument("--noi-khoang", type=float, default=2.0, help="Nối hai đoạn khi điểm chỉ tụt ngắn hơn số giây này (mặc định 2)")
    p.add_argument("--mau-moi-giay", type=float, default=4, help="Số mẫu mỗi giây (mặc định 4)")
    p.add_argument("--cache", help="Thư mục đặt .dailies (mặc định thư mục source)")
    p.set_defaults(xu_ly=_cham_diem)

    p = sub.add_parser("thuoc-tinh", help="Đọc thuộc tính mọi clip (fps quay, log, khung, xoay, anamorphic...) và đề xuất sequence, fit to frame từng clip")
    _them_json(p)
    p.add_argument("thu_muc", nargs="?", help="Thư mục source (bỏ trống khi dùng --danh-sach)")
    p.add_argument("--danh-sach", help='JSON {"clip": [{"file", "proxy"}]} các clip anh giữ')
    p.add_argument("--khung", help="Khung bản giao (1920x1080, 3840x2160, 9:16...); mặc định kích thước có nhiều clip nhất")
    p.add_argument("--fps", help="fps bản giao (25, 23.976, 29.97, 59.94...); mặc định fps có nhiều clip quay thường nhất")
    p.add_argument("--cache", help="Thư mục đặt .dailies (mặc định thư mục source)")
    p.set_defaults(xu_ly=_thuoc_tinh)

    p = sub.add_parser("xuat", help="Xuất selects của cham-diem cho editor khác: XML (Premiere, Resolve), FCPXML, EDL, CSV")
    _them_json(p)
    p.add_argument("ket_qua", help="cham_diem.json hoặc thư mục source đã chấm điểm")
    p.add_argument("--dinh-dang", default="tat-ca", help="xml, fcpxml, edl, csv, cách nhau dấu phẩy, hoặc tat-ca (mặc định)")
    p.add_argument("--ra", help="Thư mục ghi file (mặc định <thư mục source>/Dailies selects)")
    p.add_argument("--ten", help="Tên sequence và file (mặc định Selects <tên thư mục>)")
    p.add_argument("--thu-tu", choices=["file", "diem"], default="file", help="Xếp theo tên file rồi thời gian (mặc định) hoặc điểm cao trước")
    p.add_argument("--khoang", type=float, default=0.0, help="Giây trống giữa các đoạn (mặc định 0)")
    p.add_argument("--dem", type=float, default=0.0, help="Giây dư thêm mỗi đầu đoạn (handle, mặc định 0)")
    p.add_argument("--khung", help="Khung sequence (1920x1080...); mặc định theo thuoc-tinh đã quét, clip khác khung được fit to frame")
    p.add_argument("--fps", help="fps sequence; mặc định theo thuoc-tinh đã quét")
    p.set_defaults(xu_ly=_xuat)

    p = sub.add_parser("doi-chieu-cutdown", help="So kết quả cham-diem với một phiên Cutdown (.cdwn) trên cùng footage")
    _them_json(p)
    p.add_argument("cdwn", help="File .cdwn của Cutdown (phiên tự lưu: ~/Library/Application Support/com.softworks.cutdown/recovery.cdwn)")
    p.add_argument("--ket-qua", help="cham_diem.json của Dailies (mặc định <thư mục source>/.dailies/cham_diem.json)")
    p.set_defaults(xu_ly=_doi_chieu_cutdown)

    p = sub.add_parser("cai-mo-ta", help="Tải model vision Qwen3.5 MLX và mlx-vlm cho chế độ mô tả kỹ")
    _them_json(p)
    p.add_argument("--model", choices=["2b", "0.8b"], default="2b")
    p.add_argument("--dong-y", action="store_true", help="Đồng ý tải (không có thì chỉ hỏi, mã thoát 3)")
    p.set_defaults(xu_ly=_cai_mo_ta)

    p = sub.add_parser("tai-model", help="Tải SigLIP 2 (khoảng 380 MB) và YuNet cho mô tả nhanh, dò mặt")
    _them_json(p)
    p.set_defaults(xu_ly=_tai_model)

    p = sub.add_parser("bo-cuc", help="Scale, vị trí từng clip khi đổi bản ngang sang vuông, dọc (cắt từ khung ngang đã chỉnh)")
    _them_json(p)
    p.add_argument("clip_json", help="JSON clip bản ngang: {khung: {rong, cao}, clip: [{id, file, vao_giay, ra_giay, scale_phan_tram, vi_tri, diem_neo, crop, xoay}]}")
    p.add_argument("--khung", default="1:1,9:16", help="Các khung cần đổi, cách nhau dấu phẩy (mặc định 1:1,9:16)")
    p.set_defaults(xu_ly=_bo_cuc)

    try:
        args = parser.parse_args(argv)
    except LoiCuPhap as exc:
        co_json = "--json" in (argv if argv is not None else sys.argv[1:])
        ket_qua = {"ok": False, "loi": "cu_phap", "thong_bao": str(exc)}
        print(json.dumps(ket_qua, ensure_ascii=False) if co_json else f"Sai cú pháp: {exc}",
              file=sys.stdout if co_json else sys.stderr)
        return 2
    from dailies.may import ap_dung
    ap_dung()  # quiet (default) or fast mode via DAILIES_CHE_DO, inherited by subprocesses
    from dailies.khoa import DangChay
    from dailies.quet import ThieuProxy
    try:
        ma, ket_qua = args.xu_ly(args)
    except ThieuProxy as exc:
        ma, ket_qua = 1, {"ok": False, "loi": "thieu_proxy", "file": exc.cac_file, "thong_bao": str(exc)}
    except DangChay as exc:
        ma, ket_qua = 5, {"ok": False, "loi": "dang_chay", "pid": exc.pid, "lenh": exc.lenh, "thong_bao": str(exc)}
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        ma, ket_qua = 1, {"ok": False, "loi": "du_lieu", "thong_bao": str(exc)}
    if getattr(args, "json", False):
        print(json.dumps(ket_qua, ensure_ascii=False))
    else:
        print(json.dumps(ket_qua, ensure_ascii=False, indent=2), file=sys.stderr if ma else sys.stdout)
    return ma


if __name__ == "__main__":
    sys.exit(main())
