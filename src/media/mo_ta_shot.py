"""Mô tả nội dung từng shot của source bằng tiếng Việt, chạy tại máy.

Chế độ nhanh: không tải gì thêm. Lấy vài ảnh mỗi file (proxy nếu đã có trong cache phân tích, không thì nhảy thẳng tới
mốc trong file gốc), chấm nét, sáng, mặt người bằng bộ quét tầng 1 (`thi_giac`), nếu có proxy thì thêm rung; gán cỡ cảnh,
bối cảnh, nội dung bằng SigLIP 2 đã có (`hinh_chu`), rồi ghép câu mô tả theo luật.
Chế độ kỹ: thêm model vision (Qwen3.5 MLX 4-bit qua mlx-vlm) mô tả ảnh giữa shot. mlx-vlm nằm trong môi trường Python
riêng ở thư mục dữ liệu Media và chạy ở tiến trình con (`_mo_ta_mlx.py`), vì nó kéo opencv-python và libav riêng
(quyết định 017); model tải khi người dùng đồng ý (`media cai-mo-ta`). Một shot ở đây là một file source.
Hợp đồng JSON: README.md của Media.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

PHIEN_BAN = "1"
TEN_TEP = "mo_ta_shot.json"
MLX_VLM = "mlx-vlm==0.7.10"
MODEL_KY = {
    "2b": ("Qwen3.5-2B-MLX-4bit", "mlx-community/Qwen3.5-2B-MLX-4bit", "1,75 GB"),
    "0.8b": ("Qwen3.5-0.8B-MLX-4bit", "mlx-community/Qwen3.5-0.8B-MLX-4bit", "0,65 GB"),
}
CAU_HOI = ("Mô tả khung hình này bằng tiếng Việt trong một hoặc hai câu ngắn: chủ thể là ai hoặc vật gì, đang làm gì, "
           "ở đâu, cỡ cảnh (cận, trung hay toàn) và góc máy. Chỉ trả lời câu mô tả, không mở đầu.")
CANH_ANH = 448
# Nhóm nhãn cho SigLIP 2: (nhãn in ra, câu đưa vào model)
NHAN = {
    "co_canh": [
        ("cận mặt", "ảnh chụp cận cảnh khuôn mặt một người"),
        ("trung cảnh", "ảnh chụp nửa người từ thắt lưng trở lên"),
        ("toàn cảnh", "ảnh chụp toàn thân người trong khung cảnh rộng"),
        ("cận chi tiết", "ảnh chụp cận cảnh chi tiết một đồ vật"),
        ("cảnh rộng", "ảnh chụp phong cảnh rộng không có người"),
    ],
    "boi_canh": [
        ("trong nhà", "ảnh chụp trong một căn phòng"),
        ("ngoài trời", "ảnh chụp ngoài trời"),
        ("quán cà phê, nhà hàng", "ảnh chụp trong quán cà phê hoặc nhà hàng"),
        ("trung tâm thương mại", "ảnh chụp trong trung tâm thương mại"),
        ("đường phố", "ảnh chụp đường phố có xe cộ"),
        ("thiên nhiên", "ảnh chụp cây cối, núi, biển"),
        ("sân khấu, sự kiện", "ảnh chụp sân khấu sự kiện"),
        ("bếp", "ảnh chụp trong nhà bếp"),
    ],
    "noi_dung": [
        ("người nói trước máy quay", "một người đang nói chuyện trước máy quay"),
        ("món ăn, đồ uống", "ảnh chụp món ăn hoặc đồ uống"),
        ("sản phẩm", "ảnh chụp sản phẩm được trưng bày"),
        ("đám đông", "ảnh chụp nhiều người đông đúc"),
        ("chữ, biển hiệu", "ảnh chụp biển hiệu có chữ"),
        ("hình thử nghiệm", "hình sọc màu thử nghiệm của máy quay"),
    ],
}


def thu_muc_moi_truong() -> Path:
    from media import thu_muc_du_lieu
    return thu_muc_du_lieu() / "moi_truong" / "mlx-vlm"


def python_mlx() -> Path:
    return thu_muc_moi_truong() / "bin" / "python"


def thu_muc_model_ky(ten: str = "2b") -> Path:
    from media import thu_muc_du_lieu
    return thu_muc_du_lieu() / "models" / MODEL_KY[ten][0]


def da_cai_ky(ten: str = "2b") -> bool:
    return python_mlx().is_file() and (thu_muc_model_ky(ten) / "config.json").is_file()


def cai_ky(ten: str = "2b") -> dict:
    """Tạo môi trường mlx-vlm riêng và tải model (chỉ gọi khi người dùng đã đồng ý)."""
    env = thu_muc_moi_truong()
    if not python_mlx().is_file():
        env.parent.mkdir(parents=True, exist_ok=True)
        uv = shutil.which("uv")
        if uv:
            subprocess.run([uv, "venv", "-q", "--python", "3.12", str(env)], check=True)
            subprocess.run([uv, "pip", "install", "-q", "--python", str(python_mlx()), MLX_VLM], check=True)
        else:
            subprocess.run([sys.executable, "-m", "venv", str(env)], check=True)
            subprocess.run([str(python_mlx()), "-m", "pip", "install", "-q", MLX_VLM], check=True)
    dich = thu_muc_model_ky(ten)
    if not (dich / "config.json").is_file():
        subprocess.run([str(python_mlx()), "-I", "-c",
                        "import sys; from huggingface_hub import snapshot_download; "
                        "snapshot_download(sys.argv[1], local_dir=sys.argv[2])", MODEL_KY[ten][1], str(dich)], check=True)
    return {"moi_truong": str(env), "model": str(dich), "ten_model": MODEL_KY[ten][0]}


class MoTaMlx:
    """Giữ một tiến trình con mlx-vlm suốt lượt quét (nạp model một lần)."""

    def __init__(self, ten: str = "2b", lenh: list[str] | None = None):
        lenh = lenh or [str(python_mlx()), "-I", str(Path(__file__).with_name("_mo_ta_mlx.py")), str(thu_muc_model_ky(ten))]
        self.p = subprocess.Popen(lenh, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                  text=True, encoding="utf-8")
        dau = self._doc()
        if not dau.get("san_sang"):
            raise ValueError(f"Model vision không chạy được: {dau.get('loi', dau)}")
        self.giay_nap = dau.get("giay_nap")

    def _doc(self) -> dict:
        dong = self.p.stdout.readline()
        if not dong:
            raise ValueError("Tiến trình model vision dừng bất ngờ")
        return json.loads(dong)

    def mo_ta(self, anh: Path, cau_hoi: str = CAU_HOI) -> dict:
        self.p.stdin.write(json.dumps({"anh": str(anh), "cau_hoi": cau_hoi}, ensure_ascii=False) + "\n")
        self.p.stdin.flush()
        return self._doc()

    def dong(self) -> None:
        if self.p.poll() is None:
            self.p.stdin.close()
            try:
                self.p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.p.kill()


def _lay_anh(nguon: str, cac_giay: list[float], thu_muc: Path, ten: str) -> list[Path]:
    """Một ảnh JPG mỗi mốc, cạnh dài CANH_ANH px; -ss trước -i để nhảy thẳng, không giải mã cả file."""
    thu_muc.mkdir(parents=True, exist_ok=True)
    ra = []
    for i, giay in enumerate(cac_giay):
        dich = thu_muc / f"{ten}-{i}.jpg"
        if not dich.is_file():
            subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{giay:.3f}", "-i", nguon, "-frames:v", "1",
                            "-vf", f"scale='if(gt(iw,ih),{CANH_ANH},-2)':'if(gt(iw,ih),-2,{CANH_ANH})',format=yuvj420p",
                            "-q:v", "3", "-y", str(dich)], capture_output=True, check=True)
        ra.append(dich)
    return ra


def _doc_rgb(cac_anh: list[Path]):
    import cv2
    import numpy as np
    anh = [cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for p in cac_anh]
    cao, rong = anh[0].shape[:2]
    return np.stack([cv2.resize(a, (rong, cao)) for a in anh])


def _nhan_siglip(rgb) -> dict:
    from media import hinh_chu
    if not hinh_chu.da_co_model():
        return {}
    vec_anh = hinh_chu.ma_hoa_anh(rgb).mean(axis=0, keepdims=True)
    kq = {}
    for nhom, ds in NHAN.items():
        diem = hinh_chu.do_khop(vec_anh, hinh_chu.ma_hoa_chu([c for _, c in ds]))[0]
        tot = int(diem.argmax())
        kq[nhom] = {"nhan": ds[tot][0], "diem": round(float(diem[tot]), 4)}
    return kq


def _chat_luong(rgb, quet_proxy: dict | None) -> dict:
    from media import thi_giac
    mau = thi_giac.cham_khung(rgb)
    so_mat = [m["mat"] for m in mau if m["mat"] is not None]
    kq = {
        "net": round(sum(m["net"] for m in mau) / len(mau), 2),
        "sang": round(sum(m["sang"] for m in mau) / len(mau), 2),
        "chay": round(max(m["chay"] for m in mau), 4),
        "den": round(max(m["den"] for m in mau), 4),
        "so_mat": max(so_mat) if so_mat else None,
        "rung": None, "diem_dep": None,
    }
    if quet_proxy and quet_proxy.get("mau"):
        dong = [m["chuyen_dong"] for m in quet_proxy["mau"][1:]] or [0.0]
        kq["rung"] = round(sum(dong) / len(dong), 3)
        kq["diem_dep"] = round(sum(quet_proxy["diem"]) / len(quet_proxy["diem"]), 4)
    return kq


def _cau_nhanh(nhan: dict, cl: dict) -> str:
    """Ghép mô tả tiếng Việt từ nhãn và chất lượng hình."""
    phan = [nhan[k]["nhan"] for k in ("co_canh", "boi_canh", "noi_dung") if k in nhan]
    if cl["so_mat"]:
        phan.append(f"{cl['so_mat']} người trong khung" if cl["so_mat"] > 1 else "1 người trong khung")
    chat = []
    chat.append("có thể mờ" if cl["net"] < 20 else "nét")
    if cl["rung"] is not None:
        chat.append("rung hoặc lia nhanh" if cl["rung"] > 12 else "ổn định")
    if cl["chay"] > 0.05:
        chat.append("cháy sáng")
    elif cl["den"] > 0.2 or cl["sang"] < 50:
        chat.append("tối")
    else:
        chat.append("đủ sáng")
    dau = ", ".join(phan)
    return (dau[:1].upper() + dau[1:] + "; " if dau else "") + ", ".join(chat) + "."


def _khoa(media: dict, che_do: str, so_anh: int, model: str | None) -> str:
    noi_dung = json.dumps([PHIEN_BAN, media["ma_bam"], che_do, so_anh, model, CAU_HOI if model else None, NHAN],
                          ensure_ascii=False, sort_keys=True)
    return "sha256:" + hashlib.sha256(noi_dung.encode()).hexdigest()


def _proxy(goc: Path, media_id: str, duong: Path, source: Path) -> str | None:
    """Proxy Premiere `Proxies/<tên>_Proxy.mov` nếu có, không thì proxy take-khau-lenh đã tạo."""
    from media.quet import tim_proxy_premiere
    pr = tim_proxy_premiere(duong, source)
    if pr:
        return str(pr)
    tep = goc / "phan_tich" / f"{media_id}.json"
    try:
        p = json.loads(tep.read_text(encoding="utf-8"))["proxy"]["duong_dan"]
        return p if Path(p).is_file() else None
    except (OSError, KeyError, json.JSONDecodeError):
        return None


def _khoa_ten(duong: str) -> list:
    import re
    from media.khau_lenh import bo_dau
    return [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", bo_dau(duong))]


def mo_ta_thu_muc(source: str | Path, *, che_do: str = "nhanh", so_anh: int | None = None, model: str = "2b",
                  cache_goc: str | Path | None = None, bo_mo_ta=None) -> dict:
    """Mô tả mọi shot trong thư mục, có cache theo mã băm; ghi `.media/mo_ta_shot.json`."""
    from media import thu_muc_ket_qua
    from media.khoa import giu_khoa
    from media.quet import tim_clip
    if che_do not in ("nhanh", "ky"):
        raise ValueError("Chế độ phải là nhanh hoặc ky")
    source = Path(source).resolve()
    goc = thu_muc_ket_qua(source, cache_goc)
    cac_tep = tim_clip(source)
    if not cac_tep:
        raise ValueError("Không tìm thấy MP4/MOV/MXF")
    so_anh = so_anh or (3 if che_do == "nhanh" else 1)
    ten_model = MODEL_KY[model][0] if che_do == "ky" else None
    cache_dir = goc / "mo_ta"
    cache_dir.mkdir(parents=True, exist_ok=True)
    with giu_khoa(goc, "mo-ta-shot"):
        return _mo_ta(source, goc, cac_tep, che_do, so_anh, model, ten_model, cache_dir, bo_mo_ta)


def _mo_ta(source: Path, goc: Path, cac_tep: list[Path], che_do: str, so_anh: int, model: str, ten_model: str | None,
           cache_dir: Path, bo_mo_ta) -> dict:
    from fractions import Fraction
    from media import thi_giac
    from media.quet import quet_files
    shot, canh_bao, thoi_gian_anh = [], [], []
    may = None
    try:
        for media_id, muc in dict(quet_files(cac_tep)).items():
            duong = Path(muc["duong_dan_gan_nhat"])
            tuong_doi = str(duong.relative_to(source)) if duong.is_relative_to(source) else duong.name
            fps = muc.get("fps") or {"tu": 25, "mau": 1}
            thoi_luong = float(Fraction(muc.get("so_frame", 0)) / Fraction(fps["tu"], fps["mau"]))
            khoa = _khoa(muc, che_do, so_anh, ten_model)
            tep_cache = cache_dir / f"{media_id}-{che_do}.json"
            try:
                cu = json.loads(tep_cache.read_text(encoding="utf-8"))
                if cu.get("khoa") == khoa and all(Path(a["duong_dan"]).is_file() for a in cu["anh"]):
                    shot.append({**cu["shot"], "file": str(duong), "ten_file": tuong_doi})
                    continue
            except (OSError, KeyError, json.JSONDecodeError):
                pass
            proxy = _proxy(goc, media_id, duong, source)
            cac_giay = [round(thoi_luong * (k + 0.5) / so_anh, 3) for k in range(so_anh)] if so_anh > 1 else [round(thoi_luong / 2, 3)]
            cac_anh = _lay_anh(proxy or str(duong), cac_giay, cache_dir / "anh", f"{media_id}-{che_do}")
            bat_dau = time.perf_counter()
            rgb = _doc_rgb(cac_anh)
            quet_proxy = None
            if proxy:  # rung cần quét liên tục; chỉ làm trên proxy để không giải mã lại footage gốc (quyết định 027)
                quet_proxy = thi_giac.quet(media_id, {**muc, "proxy": proxy}, goc / "thi_giac")
            nhan = _nhan_siglip(rgb)
            cl = _chat_luong(rgb, quet_proxy)
            mo_ta = _cau_nhanh(nhan, cl)
            anh = [{"giay": g, "frame": round(g * fps["tu"] / fps["mau"]), "duong_dan": str(p), "mo_ta": None}
                   for g, p in zip(cac_giay, cac_anh)]
            if che_do == "ky":
                if may is None:
                    may = bo_mo_ta() if bo_mo_ta else MoTaMlx(model)
                giua = anh[len(anh) // 2]
                for a in anh:
                    tl = may.mo_ta(Path(a["duong_dan"]))
                    if "loi" in tl:
                        canh_bao.append(f"{tuong_doi}: model vision lỗi ở {a['giay']} giây: {tl['loi']}")
                        continue
                    a["mo_ta"] = tl["mo_ta"].strip()
                if giua["mo_ta"]:
                    mo_ta = giua["mo_ta"]
            thoi_gian_anh.append((time.perf_counter() - bat_dau) / len(anh))
            muc_shot = {"file": str(duong), "ten_file": tuong_doi, "media_id": media_id, "fps": fps,
                        "thoi_luong_giay": round(thoi_luong, 3), "che_do": che_do, "model": ten_model,
                        "mo_ta": mo_ta, "mo_ta_nhanh": _cau_nhanh(nhan, cl), "nhan": nhan, "chat_luong": cl,
                        "co_proxy": bool(proxy), "anh": anh}
            tam = tep_cache.with_suffix(".tam")
            tam.write_text(json.dumps({"khoa": khoa, "anh": anh, "shot": muc_shot}, ensure_ascii=False), encoding="utf-8")
            tam.replace(tep_cache)
            shot.append(muc_shot)
            print(f"Mô tả shot: {len(shot)}/{len(cac_tep)} {tuong_doi}", file=sys.stderr, flush=True)
    finally:
        if may is not None and hasattr(may, "dong"):
            may.dong()
    shot.sort(key=lambda s: _khoa_ten(s["ten_file"]))
    for stt, s in enumerate(shot, 1):
        s["stt"] = stt
        s["id"] = s["media_id"]
    from media import hinh_chu
    if not hinh_chu.da_co_model():
        canh_bao.append("Chưa có SigLIP 2 nên không gán cỡ cảnh, bối cảnh (media tai-model)")
    kq = {"phien_ban": PHIEN_BAN, "thu_muc": str(source), "che_do": che_do, "model": ten_model, "so_shot": len(shot),
          "giay_moi_anh": round(sum(thoi_gian_anh) / len(thoi_gian_anh), 3) if thoi_gian_anh else None,
          "shot": shot, "canh_bao": canh_bao}
    dich = goc / TEN_TEP
    tam = dich.with_suffix(".tmp")
    tam.write_text(json.dumps(kq, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tam.replace(dich)
    return {**kq, "file_ket_qua": str(dich)}
