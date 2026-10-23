"""Proxy, lời thoại có mốc từng chữ (faster-whisper) và khoảng lặng của source; cache trong `.media/`.

Mốc tính theo frame của fps file (hoặc fps truyền vào). Proxy H.264 cạnh ngắn 720 px (VideoToolbox), dùng cho mọi
bước đọc hình sau đó để không giải mã lại footage gốc. Ngôn ngữ mặc định tự nhận theo từng file (crew có thể nói tiếng
Anh); bật VAD để Whisper không bịa câu ở đoạn lặng ("Cảm ơn các bạn đã theo dõi"), mốc chữ vẫn theo thời gian gốc.
Cache theo mã băm file, model, ngôn ngữ, cấu hình nghe và gợi ý.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from fractions import Fraction
from pathlib import Path

from media.quet import quet_files, tim_clip

PHIEN_BAN = "2"  # 2: tự nhận ngôn ngữ, bật VAD (bản 1 ép tiếng Việt, không VAD, bịa câu ở đoạn lặng)
NGON_NGU = ("auto", "vi", "en")
# VAD nhạy hơn mặc định và đệm rộng để giữ khẩu lệnh ngắn, nói xa micro ("cut", "action")
VAD = {"threshold": 0.35, "min_silence_duration_ms": 500, "speech_pad_ms": 400}
KHOA_CAN = ("media_id", "ma_bam", "ma_phan_tich", "fps", "transcript", "khoang_lang", "proxy")


def _chay(lenh: list[str]) -> None:
    kq = subprocess.run(lenh, capture_output=True, text=True)
    if kq.returncode:
        raise ValueError(f"Lệnh media lỗi: {' '.join(lenh)}\n{kq.stderr[-2000:]}")


def _fr(fps: dict) -> Fraction:
    return Fraction(fps["tu"], fps["mau"])


def giay_sang_frame(giay: float, fps: Fraction) -> int:
    so = Fraction(str(giay)) * fps
    return (2 * so.numerator + so.denominator) // (2 * so.denominator) if so >= 0 else -giay_sang_frame(-giay, fps)


def _ma_phan_tich(ma_bam: str, model: str, fps: dict, prompt: str, ngon_ngu: str = "auto") -> str:
    noi_dung = json.dumps([PHIEN_BAN, ma_bam, model, fps, prompt, ngon_ngu, VAD], ensure_ascii=False, sort_keys=True)
    return "sha256:" + hashlib.sha256(noi_dung.encode()).hexdigest()


def _kich_thuoc(rong: int, cao: int, canh: int) -> tuple[int, int]:
    he_so = canh / min(rong, cao)
    return max(2, round(rong * he_so / 2) * 2), max(2, round(cao * he_so / 2) * 2)


def tao_proxy(tep: Path, media_id: str, media: dict, fps: dict, proxy_dir: Path) -> dict:
    """FFmpeg tự áp xoay metadata trước scale; đầu ra luôn CFR."""
    rong, cao = media["rong"], media["cao"]
    if media.get("xoay", 0) in (90, 270):
        rong, cao = cao, rong
    w, h = _kich_thuoc(rong, cao, 720)
    dich = proxy_dir / f"{media_id}-{fps['tu']}_{fps['mau']}-v{PHIEN_BAN}.mp4"
    if not dich.is_file() or not dich.stat().st_size:
        tam = dich.with_suffix(".tmp.mp4")
        try:
            _chay(["ffmpeg", "-v", "error", "-i", str(tep), "-map", "0:v:0", "-map", "0:a?",
                   "-vf", f"scale={w}:{h},format=nv12", "-r", f"{fps['tu']}/{fps['mau']}",
                   "-fps_mode:v", "cfr", "-c:v", "h264_videotoolbox", "-b:v", "2M",
                   "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart", "-y", str(tam)])
            tam.replace(dich)
        finally:
            tam.unlink(missing_ok=True)
    return {"duong_dan": str(dich), "rong": w, "cao": h}


def _giai_ma_pcm(tep: Path):
    """PCM float32 16 kHz mono qua ffmpeg cho Whisper."""
    import numpy as np
    kq = subprocess.run(["ffmpeg", "-v", "error", "-i", str(tep), "-vn", "-ac", "1", "-ar", "16000", "-f", "f32le", "-"],
                        capture_output=True)
    if kq.returncode:
        raise ValueError(f"Không giải mã được tiếng {tep}: {kq.stderr.decode(errors='replace')[-1000:]}")
    return np.frombuffer(kq.stdout, dtype=np.float32).copy()


def _frame_khoang(vao: float, ra: float, fps: Fraction) -> tuple[int, int]:
    start = max(0, giay_sang_frame(vao, fps))
    return start, max(start + 1, giay_sang_frame(ra, fps))


def nghe(tep: Path, model, model_ten: str, fps: dict, prompt: str = "", ngon_ngu: str = "auto") -> tuple[dict, float, float]:
    """Nghe một clip bằng model đã nạp; trả transcript (kèm ngôn ngữ thật), giây nghe, giây âm thanh."""
    mau = _giai_ma_pcm(tep)
    do_dai = len(mau) / 16000
    if not len(mau):
        return {"model": model_ten, "ngon_ngu": None, "cau": []}, 0.0, 0.0
    bat_dau = time.perf_counter()
    doan, info = model.transcribe(mau, language=None if ngon_ngu == "auto" else ngon_ngu, word_timestamps=True,
                                  vad_filter=True, vad_parameters=dict(VAD),
                                  condition_on_previous_text=False, initial_prompt=prompt or None)
    f = _fr(fps)
    cau = []
    for phan in doan:
        vao, ra = _frame_khoang(phan.start, phan.end, f)
        cac_tu = []
        for tu in phan.words or []:
            if not tu.word.strip():
                continue
            t_vao, t_ra = _frame_khoang(tu.start, tu.end, f)
            cac_tu.append({"chu": tu.word.strip(), "vao": t_vao, "ra": t_ra,
                           "xac_suat": min(1.0, max(0.0, float(tu.probability)))})
        cau.append({"vao": vao, "ra": ra, "noi_dung": phan.text.strip(), "tu": cac_tu})
    tim = getattr(info, "language", None) or (None if ngon_ngu == "auto" else ngon_ngu)
    xs = getattr(info, "language_probability", None)
    return ({"model": model_ten, "ngon_ngu": tim, "xac_suat_ngon_ngu": round(float(xs), 3) if xs is not None else None,
             "cau": cau}, time.perf_counter() - bat_dau, do_dai)


def khoang_lang(tep: Path, fps: dict, thoi_luong: float | None = None) -> list[dict]:
    """Khoảng lặng (dưới -35 dB, từ 0,3 giây) bằng FFmpeg silencedetect."""
    import re
    kq = subprocess.run(["ffmpeg", "-hide_banner", "-i", str(tep), "-vn", "-af", "silencedetect=noise=-35dB:d=0.3",
                         "-f", "null", "-"], capture_output=True, text=True)
    if kq.returncode:
        raise ValueError(f"silencedetect lỗi: {kq.stderr[-1000:]}")
    f = _fr(fps)
    dang_im, ket_qua = None, []
    for dong in kq.stderr.splitlines():
        bat_dau = re.search(r"silence_start:\s*([\d.]+)", dong)
        ket_thuc = re.search(r"silence_end:\s*([\d.]+)", dong)
        if bat_dau:
            dang_im = float(bat_dau.group(1))
        if ket_thuc and dang_im is not None:
            vao, ra = _frame_khoang(dang_im, float(ket_thuc.group(1)), f)
            ket_qua.append({"vao": vao, "ra": ra})
            dang_im = None
    if dang_im is not None and thoi_luong and thoi_luong > dang_im:
        vao, ra = _frame_khoang(dang_im, thoi_luong, f)
        ket_qua.append({"vao": vao, "ra": ra})
    return ket_qua


def _model_da_co(ten: str) -> str:
    """Ưu tiên snapshot đã có trong cache Hugging Face để khỏi kiểm mạng."""
    if Path(ten).is_dir():
        return ten
    hub = Path.home() / ".cache" / "huggingface" / "hub" / f"models--Systran--faster-whisper-{ten}" / "snapshots"
    for snapshot in sorted(hub.glob("*")) if hub.is_dir() else []:
        if (snapshot / "model.bin").is_file():
            return str(snapshot)
    return ten


def _cache_hop_le(cu: dict, khoa: str) -> bool:
    return (all(k in cu for k in KHOA_CAN) and cu["ma_phan_tich"] == khoa
            and Path(cu["proxy"]["duong_dan"]).is_file())


def phan_tich_thu_muc(source: str | Path, goc: str | Path, *, model_ten: str = "large-v3", fps_dat: dict | None = None,
                      model_factory=None, prompt: str = "", ngon_ngu: str = "auto") -> tuple[dict, dict]:
    """Proxy và nghe mọi clip trong `source`; cache ở `goc` (thư mục `.media`). Trả (media, tóm tắt có `cache`)."""
    from media.may import che_do, nghi_neu_nong, so_luong
    if ngon_ngu not in NGON_NGU:
        raise ValueError(f"Ngôn ngữ phải là {', '.join(NGON_NGU)}")
    source = Path(source).resolve()
    cac_tep = tim_clip(source)
    if not cac_tep:
        raise ValueError("Không tìm thấy MP4/MOV/MXF")
    da_quet = quet_files(cac_tep)
    media = dict(da_quet)
    duong_nguon = {mid: tep for tep, (mid, _) in zip(cac_tep, da_quet)}
    goc = Path(goc)
    phan_tich_dir, proxy_dir = goc / "phan_tich", goc / "proxy"
    for p in (phan_tich_dir, proxy_dir):
        p.mkdir(parents=True, exist_ok=True)
    jobs, ket_qua = [], {}
    for mid, muc in media.items():
        fps = fps_dat or muc.get("fps") or {"tu": 25, "mau": 1}
        khoa = _ma_phan_tich(muc["ma_bam"], model_ten, fps, prompt, ngon_ngu)
        dich = phan_tich_dir / f"{mid}.json"
        try:
            if _cache_hop_le(json.loads(dich.read_text(encoding="utf-8")), khoa):
                ket_qua[mid] = str(dich)
                continue
        except (OSError, json.JSONDecodeError, KeyError, TypeError):
            pass
        jobs.append((mid, muc, duong_nguon[mid], fps, khoa, dich))
    tong = len(media) + len(jobs)
    da_xong = len(ket_qua)
    if da_xong:
        print(f"Nạp source: {da_xong}/{tong} từ cache", file=sys.stderr, flush=True)
    proxy_kq = [None] * len(jobs)
    if jobs:
        with ThreadPoolExecutor(max_workers=1 if che_do() == "em" else min(2, len(jobs))) as pool:
            futures = {pool.submit(tao_proxy, tep, mid, muc, fps, proxy_dir): n
                       for n, (mid, muc, tep, fps, _, _) in enumerate(jobs)}
            for future in as_completed(futures):
                proxy_kq[futures[future]] = future.result()
                da_xong += 1
                print(f"Tạo proxy: {da_xong}/{tong} {jobs[futures[future]][2].name}", file=sys.stderr, flush=True)
    model, tong_nghe, tong_am = None, 0.0, 0.0
    for (mid, muc, tep, fps, khoa, dich), proxy in zip(jobs, proxy_kq):
        if muc["am_thanh"]["kieu"] == "trong_file":
            if model is None:
                if model_factory is None:
                    from faster_whisper import WhisperModel
                    model_factory = lambda ten: WhisperModel(_model_da_co(ten), device="auto", compute_type="int8",
                                                             cpu_threads=so_luong())
                model = model_factory(model_ten)
            nghi_neu_nong()
            transcript, giay_nghe, giay_am = nghe(tep, model, model_ten, fps, prompt, ngon_ngu)
            tong_nghe += giay_nghe
            tong_am += giay_am
            im_lang = khoang_lang(tep, fps, giay_am)
        else:
            transcript, im_lang = {"model": model_ten, "ngon_ngu": None, "cau": []}, []
        du_lieu = {"media_id": mid, "ma_bam": muc["ma_bam"], "ma_phan_tich": khoa, "phien_ban_phan_tich": PHIEN_BAN,
                   "fps": fps, "transcript": transcript, "khoang_lang": im_lang, "proxy": proxy}
        tam = dich.with_suffix(".tmp")
        tam.write_text(json.dumps(du_lieu, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tam.replace(dich)
        ket_qua[mid] = str(dich)
        da_xong += 1
        print(f"Nạp source: {da_xong}/{tong} {tep.name}", file=sys.stderr, flush=True)
    toc_do = 60 * tong_nghe / tong_am if tong_am else None
    if toc_do is not None:
        print(f"Whisper {model_ten}: {toc_do:.2f} giây nghe / phút âm thanh", file=sys.stderr, flush=True)
    return media, {"cache": ket_qua, "so_clip": len(media), "so_clip_tu_cache": len(media) - len(jobs),
                   "model": model_ten, "giay_nghe_moi_phut": round(toc_do, 3) if toc_do is not None else None}
