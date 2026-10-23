"""Cửa dòng lệnh `media` cho MCP Premiere và After Effects.

Quy ước: `--json` in đúng một object JSON ra stdout, log và tiến độ ra stderr. Mã thoát: 0 xong; 1 lỗi dữ liệu;
2 sai cú pháp; 3 cần anh xác nhận (có `cau_hoi`); 4 thiếu model.
"""

import argparse
import json
import sys
from pathlib import Path

from media import __version__


class LoiCuPhap(Exception):
    """Lỗi argparse để trả mã 2."""


class Parser(argparse.ArgumentParser):
    def add_subparsers(self, **kwargs):
        kwargs.setdefault("parser_class", Parser)
        return super().add_subparsers(**kwargs)

    def error(self, message: str) -> None:
        raise LoiCuPhap(message)


def _doc_fps(chuoi: str | None) -> dict | None:
    """'25', '29.97', '30000/1001' thành {tu, mau}."""
    if not chuoi:
        return None
    from fractions import Fraction
    ntsc = {"23.976": "24000/1001", "29.97": "30000/1001", "59.94": "60000/1001"}
    f = Fraction(ntsc.get(chuoi.strip(), chuoi.strip()))
    if f <= 0:
        raise ValueError(f"FPS không hợp lệ: {chuoi}")
    return {"tu": f.numerator, "mau": f.denominator}


def _take_khau_lenh(args) -> tuple[int, dict]:
    from media.bo_cuc import doc_khung_chuoi
    from media.khau_lenh import take_thu_muc
    kq = take_thu_muc(args.thu_muc, model_ten=args.model, cache_goc=args.cache, fps=_doc_fps(args.fps),
                      giu_file_khong_lenh=not args.bo_file_khong_lenh, khung=doc_khung_chuoi(args.khung))
    return 0, {"ok": True, **kq}


def _mo_ta_shot(args) -> tuple[int, dict]:
    from media import mo_ta_shot
    if args.che_do == "ky" and not mo_ta_shot.da_cai_ky(args.model):
        ten, _, co = mo_ta_shot.MODEL_KY[args.model]
        return 4, {"ok": False, "loi": "thieu_model", "thong_bao": f"Chế độ kỹ cần {ten} ({co}) và mlx-vlm; "
                   f"chạy: media cai-mo-ta --model {args.model} --dong-y"}
    kq = mo_ta_shot.mo_ta_thu_muc(args.thu_muc, che_do=args.che_do, so_anh=args.so_anh, model=args.model, cache_goc=args.cache)
    return 0, {"ok": True, **kq}


def _cai_mo_ta(args) -> tuple[int, dict]:
    from media import mo_ta_shot
    ten, nguon, co = mo_ta_shot.MODEL_KY[args.model]
    if mo_ta_shot.da_cai_ky(args.model):
        return 0, {"ok": True, "thong_bao": f"Đã có {ten}", "model": str(mo_ta_shot.thu_muc_model_ky(args.model))}
    if not args.dong_y:
        return 3, {"ok": False, "loi": "can_xac_nhan", "cau_hoi": f"Tải {ten} ({co}, {nguon}) và cài {mo_ta_shot.MLX_VLM} "
                   f"vào môi trường riêng {mo_ta_shot.thu_muc_moi_truong()}? Gọi lại với --dong-y"}
    return 0, {"ok": True, **mo_ta_shot.cai_ky(args.model)}


def _tai_model(args) -> tuple[int, dict]:
    from media import hinh_chu, thi_giac
    return 0, {"ok": True, "siglip": str(hinh_chu.tai_model()), "yunet": str(thi_giac.tai_yunet())}


def _bo_cuc(args) -> tuple[int, dict]:
    from media.bo_cuc import doc_khung_chuoi, doi_ban
    du_lieu = json.loads(Path(args.clip_json).read_text(encoding="utf-8"))
    cac_khung = [doc_khung_chuoi(k) for k in args.khung.split(",") if k.strip()]
    return 0, {"ok": True, **doi_ban(du_lieu, cac_khung)}


def _them_json(p: argparse.ArgumentParser) -> None:
    p.add_argument("--json", action="store_true", default=argparse.SUPPRESS)


def main(argv: list[str] | None = None) -> int:
    parser = Parser(prog="media", description="Năng lực media cho MCP Premiere và After Effects.")
    parser.add_argument("--version", action="version", version=f"media {__version__}")
    parser.add_argument("--json", action="store_true")
    sub = parser.add_subparsers(dest="lenh", required=True)

    p = sub.add_parser("take-khau-lenh", help="Cắt take thô theo khẩu lệnh trong audio gốc (3-2-1, action, diễn ... cắt), kèm bố cục cho sequence")
    _them_json(p)
    p.add_argument("thu_muc", help="Thư mục source")
    p.add_argument("--khung", default="1920x1080", help="Khung sequence để tính scale, vị trí (1920x1080, 9:16...)")
    p.add_argument("--model", default="large-v3", help="Model faster-whisper (mặc định large-v3)")
    p.add_argument("--fps", help="Tính frame theo fps này (25, 29.97, 30000/1001); mặc định fps từng file")
    p.add_argument("--cache", help="Thư mục đặt .media (mặc định thư mục source)")
    p.add_argument("--bo-file-khong-lenh", action="store_true", help="Bỏ file không nghe thấy khẩu lệnh (mặc định giữ cả file)")
    p.set_defaults(xu_ly=_take_khau_lenh)

    p = sub.add_parser("mo-ta-shot", help="Mô tả nội dung từng shot bằng tiếng Việt (nhanh: thuật toán, kỹ: model vision)")
    _them_json(p)
    p.add_argument("thu_muc", help="Thư mục source")
    p.add_argument("--che-do", choices=["nhanh", "ky"], default="nhanh")
    p.add_argument("--so-anh", type=int, help="Số ảnh mỗi shot (mặc định nhanh 3, kỹ 1)")
    p.add_argument("--model", choices=["2b", "0.8b"], default="2b", help="Model chế độ kỹ")
    p.add_argument("--cache", help="Thư mục đặt .media (mặc định thư mục source)")
    p.set_defaults(xu_ly=_mo_ta_shot)

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
    from media.may import ap_dung
    ap_dung()  # chế độ êm (mặc định) hoặc nhanh qua MEDIA_CHE_DO, áp cho cả tiến trình con
    try:
        ma, ket_qua = args.xu_ly(args)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        ma, ket_qua = 1, {"ok": False, "loi": "du_lieu", "thong_bao": str(exc)}
    if getattr(args, "json", False):
        print(json.dumps(ket_qua, ensure_ascii=False))
    else:
        print(json.dumps(ket_qua, ensure_ascii=False, indent=2), file=sys.stderr if ma else sys.stdout)
    return ma


if __name__ == "__main__":
    sys.exit(main())
