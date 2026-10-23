"""Word-timestamped transcripts (faster-whisper) and silences for each clip; cached in `.media/phan_tich`.

The editor imports, culls and builds proxies in Premiere first; Media listens to the proxy audio (`quet.nguon_vao`) and
never creates proxies. Timestamps are frames at the file's fps (or the fps passed in).

Language (`auto`, the default) is detected per speech turn: VAD splits the audio into turns, Whisper detects and
transcribes each turn separately, then offsets back to file time. When the crew calls cues in one language and the
actors speak another, each keeps its own language (Whisper picks one language per 30 s window, so whole-file
transcription would translate or drop the other). Only speech is transcribed, so no hallucinated sentences in
silence ("Cảm ơn các bạn đã theo dõi"). Forcing `vi` or `en` transcribes the whole file in one language, faster.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from fractions import Fraction
from pathlib import Path

from media.quet import quet_files

PHIEN_BAN = "7"  # 2: auto language, VAD on; 4: per-turn transcription; 5: drop stray languages; 7: short turns with context
NGON_NGU = ("auto", "vi", "en")
# VAD more sensitive than default with wide padding, to keep short off-mic cues ("cut", "action")
VAD = {"threshold": 0.35, "min_silence_duration_ms": 500, "speech_pad_ms": 400}
LUOT_NOI = {"noi_khi_lang_duoi": 0.6, "dai_toi_da": 28.0}  # seconds; merge VAD segments into speech turns for language detection
# Short turns (under 2 s) lack context on their own ("y acción" heard as "y ya se va"): re-transcribe with 3 s of audio
# before and 1 s after, keeping only words centred inside the turn. No initial_prompt: tested on real footage,
# Whisper copies the prompt into ambiguous turns and invents fake cues.
LUOT_NGAN = 2.0
NGU_CANH = (3.0, 1.0)
# A file's valid languages: at least 10% of speech time or at least 3 seconds
TIENG_TOI_THIEU = (0.10, 3.0)
KHOA_CAN = ("media_id", "ma_bam", "ma_phan_tich", "fps", "transcript", "khoang_lang", "proxy")


def _fr(fps: dict) -> Fraction:
    return Fraction(fps["tu"], fps["mau"])


def giay_sang_frame(giay: float, fps: Fraction) -> int:
    so = Fraction(str(giay)) * fps
    return (2 * so.numerator + so.denominator) // (2 * so.denominator) if so >= 0 else -giay_sang_frame(-giay, fps)


def _ma_phan_tich(ma_bam: str, model: str, fps: dict, prompt: str, ngon_ngu: str = "auto") -> str:
    noi_dung = json.dumps([PHIEN_BAN, ma_bam, model, fps, prompt, ngon_ngu, VAD, LUOT_NOI, LUOT_NGAN, NGU_CANH,
                           TIENG_TOI_THIEU],
                          ensure_ascii=False, sort_keys=True)
    return "sha256:" + hashlib.sha256(noi_dung.encode()).hexdigest()


def _thong_so_proxy(p: Path) -> dict:
    from media.quet import ffprobe
    v = next(x for x in ffprobe(p)["streams"] if x.get("codec_type") == "video")
    return {"duong_dan": str(p), "rong": int(v["width"]), "cao": int(v["height"])}


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


def luot_noi(mau, sr: int = 16000) -> list[tuple[int, int]]:
    """Speech turns (first sample, last sample) from VAD: segments under 0.6 s apart merge, each turn at most 28 seconds."""
    from faster_whisper.vad import VadOptions, get_speech_timestamps
    doan = get_speech_timestamps(mau, VadOptions(**VAD), sampling_rate=sr)
    luot = []
    for d in doan:
        if luot and d["start"] - luot[-1][1] < LUOT_NOI["noi_khi_lang_duoi"] * sr \
                and d["end"] - luot[-1][0] <= LUOT_NOI["dai_toi_da"] * sr:
            luot[-1][1] = d["end"]
        else:
            luot.append([d["start"], d["end"]])
    return [(a, b) for a, b in luot]


def _chuyen_doan(doan, lech: float, f: Fraction) -> list[dict]:
    cau = []
    for phan in doan:
        vao, ra = _frame_khoang(phan.start + lech, phan.end + lech, f)
        cac_tu = []
        for tu in phan.words or []:
            if not tu.word.strip():
                continue
            t_vao, t_ra = _frame_khoang(tu.start + lech, tu.end + lech, f)
            cac_tu.append({"chu": tu.word.strip(), "vao": t_vao, "ra": t_ra,
                           "xac_suat": min(1.0, max(0.0, float(tu.probability)))})
        cau.append({"vao": vao, "ra": ra, "noi_dung": phan.text.strip(), "tu": cac_tu})
    return cau


def _giu_trong_luot(cau: list[dict], vao: Fraction, ra: Fraction) -> list[dict]:
    """Join words centred within [vao, ra] (frames) into one segment; drop the context audio."""
    tu = [t for c in cau for t in c["tu"] if vao <= (t["vao"] + t["ra"]) / 2 <= ra]
    if not tu:
        return []
    return [{"vao": tu[0]["vao"], "ra": tu[-1]["ra"], "noi_dung": " ".join(t["chu"] for t in tu), "tu": tu}]


def nghe(tep: Path, model, model_ten: str, fps: dict, prompt: str = "", ngon_ngu: str = "auto") -> tuple[dict, float, float]:
    """Transcribe one clip with the loaded model; returns transcript (with main language), seconds spent, seconds of audio.

    `auto`: detect language and transcribe each speech turn separately. Forced `vi`, `en`: whole file in one pass with VAD."""
    from collections import Counter
    mau = _giai_ma_pcm(tep)
    do_dai = len(mau) / 16000
    if not len(mau):
        return {"model": model_ten, "ngon_ngu": None, "cau": []}, 0.0, 0.0
    bat_dau = time.perf_counter()
    f = _fr(fps)
    chung = dict(word_timestamps=True, condition_on_previous_text=False, initial_prompt=prompt or None)
    if ngon_ngu != "auto":
        doan, _ = model.transcribe(mau, language=ngon_ngu, vad_filter=True, vad_parameters=dict(VAD), **chung)
        return ({"model": model_ten, "ngon_ngu": ngon_ngu, "cau": _chuyen_doan(doan, 0.0, f)},
                time.perf_counter() - bat_dau, do_dai)
    ket, theo_tieng = [], Counter()
    for a, b in luot_noi(mau):
        doan, info = model.transcribe(mau[a:b], language=None, vad_filter=False, **chung)
        moi = _chuyen_doan(doan, a / 16000, f)
        tieng = getattr(info, "language", None)
        ket.append([a, b, tieng, dict(getattr(info, "all_language_probs", None) or []), moi])
        if tieng and moi:
            theo_tieng[tieng] += (b - a) / 16000
    # Short turns are often misdetected ("y acción" as Japanese): keep only languages common enough in the file; a turn
    # detected as another language is re-transcribed in that turn's most likely valid language
    tong = sum(theo_tieng.values())
    hop_le = {t for t, giay in theo_tieng.items()
              if tong and (giay >= TIENG_TOI_THIEU[0] * tong or giay >= TIENG_TOI_THIEU[1])}
    cau, theo_tieng = [], Counter()
    for a, b, tieng, xac_suat, moi in ket:
        doi_tieng = bool(hop_le) and tieng not in hop_le
        if doi_tieng:
            tieng = max(hop_le, key=lambda t: xac_suat.get(t, 0.0))
        if (b - a) / 16000 < LUOT_NGAN:
            a2, b2 = max(0, a - int(NGU_CANH[0] * 16000)), min(len(mau), b + int(NGU_CANH[1] * 16000))
            doan, _ = model.transcribe(mau[a2:b2], language=tieng, vad_filter=False, **chung)
            moi = _giu_trong_luot(_chuyen_doan(doan, a2 / 16000, f), (a / 16000 - 0.3) * f, (b / 16000 + 0.1) * f)
        elif doi_tieng:
            doan, _ = model.transcribe(mau[a:b], language=tieng, vad_filter=False, **chung)
            moi = _chuyen_doan(doan, a / 16000, f)
        for c in moi:
            c["ngon_ngu"] = tieng
        cau += moi
        if tieng and moi:
            theo_tieng[tieng] += (b - a) / 16000
    chinh = theo_tieng.most_common(1)[0][0] if theo_tieng else None
    return ({"model": model_ten, "ngon_ngu": chinh, "ngon_ngu_cac_luot": dict(theo_tieng.most_common()), "cau": cau},
            time.perf_counter() - bat_dau, do_dai)


def khoang_lang(tep: Path, fps: dict, thoi_luong: float | None = None) -> list[dict]:
    """Silences (below -35 dB, at least 0.3 s) via FFmpeg silencedetect."""
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
    """Prefer a snapshot already in the Hugging Face cache to avoid network checks."""
    if Path(ten).is_dir():
        return ten
    hub = Path.home() / ".cache" / "huggingface" / "hub" / f"models--Systran--faster-whisper-{ten}" / "snapshots"
    for snapshot in sorted(hub.glob("*")) if hub.is_dir() else []:
        if (snapshot / "model.bin").is_file():
            return str(snapshot)
    return ten


def _cache_hop_le(cu: dict, khoa: str) -> bool:
    return (all(k in cu for k in KHOA_CAN) and cu["ma_phan_tich"] == khoa
            and (cu["proxy"] is None or Path(cu["proxy"]["duong_dan"]).is_file()))


def phan_tich_thu_muc(goc: str | Path, cac_tep: list[Path], proxy: dict[str, Path | None], *, model_ten: str = "large-v3",
                      fps_dat: dict | None = None, model_factory=None, prompt: str = "",
                      ngon_ngu: str = "auto") -> tuple[dict, dict]:
    """Transcribe clips (input from `quet.nguon_vao`); cache in `goc` (the `.media` folder). Audio comes from the proxy, or
    from the original (audio only) for clips without one. Returns (media, summary with `cache`)."""
    from media.may import nghi_neu_nong, so_luong
    if ngon_ngu not in NGON_NGU:
        raise ValueError(f"Ngôn ngữ phải là {', '.join(NGON_NGU)}")
    da_quet = quet_files(cac_tep)
    media = dict(da_quet)
    duong_nguon = {mid: tep for tep, (mid, _) in zip(cac_tep, da_quet)}
    pr = {mid: proxy.get(str(tep)) for mid, tep in duong_nguon.items()}
    goc = Path(goc)
    phan_tich_dir = goc / "phan_tich"
    phan_tich_dir.mkdir(parents=True, exist_ok=True)
    jobs, ket_qua = [], {}
    for mid, muc in media.items():
        fps = fps_dat or muc.get("fps") or {"tu": 25, "mau": 1}
        khoa = _ma_phan_tich(muc["ma_bam"], model_ten, fps, prompt, ngon_ngu)
        dich = phan_tich_dir / f"{mid}.json"
        proxy = _thong_so_proxy(pr[mid]) if pr[mid] else None
        try:
            cu = json.loads(dich.read_text(encoding="utf-8"))
            if cu.get("ma_phan_tich") == khoa and cu.get("proxy") != proxy:
                cu["proxy"] = proxy  # proxy changed (regenerated or relinked): update without re-transcribing
                dich.write_text(json.dumps(cu, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            if _cache_hop_le(cu, khoa):
                ket_qua[mid] = str(dich)
                continue
        except (OSError, json.JSONDecodeError, KeyError, TypeError):
            pass
        jobs.append((mid, muc, duong_nguon[mid], fps, khoa, dich, proxy))
    tong = len(media)
    da_xong = len(ket_qua)
    if da_xong:
        print(f"Nạp source: {da_xong}/{tong} từ cache", file=sys.stderr, flush=True)
    model, tong_nghe, tong_am = None, 0.0, 0.0
    for mid, muc, tep, fps, khoa, dich, proxy in jobs:
        if muc["am_thanh"]["kieu"] == "trong_file":
            if model is None:
                if model_factory is None:
                    from faster_whisper import WhisperModel
                    model_factory = lambda ten: WhisperModel(_model_da_co(ten), device="auto", compute_type="int8",
                                                             cpu_threads=so_luong())
                model = model_factory(model_ten)
            nghi_neu_nong()
            nguon_tieng = Path(proxy["duong_dan"]) if proxy else tep  # proxy audio equals the original, and the file is smaller
            transcript, giay_nghe, giay_am = nghe(nguon_tieng, model, model_ten, fps, prompt, ngon_ngu)
            tong_nghe += giay_nghe
            tong_am += giay_am
            im_lang = khoang_lang(nguon_tieng, fps, giay_am)
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
